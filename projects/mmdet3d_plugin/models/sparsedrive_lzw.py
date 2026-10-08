from inspect import signature # 从 Python 标准库 inspect 中导入 signature：signature 可以用来检查某个函数的参数列表，这里后面会用它判断 img_backbone.forward 是否支持 metas 参数
import torch                  # 导入 PyTorch：SparseDrive 的图像张量、特征图、reshape 等操作都依赖 torch


# 从 MMCV 的 runner 模块导入两个精度控制装饰器
# force_fp32：强制某些输入转成 fp32，一般用于 loss、forward 等需要稳定数值的地方
# auto_fp16：自动把某些输入转成 fp16，用于混合精度训练，加快计算并节省显存
from mmcv.runner import force_fp32, auto_fp16

# 从 MMCV 中导入 build_from_cfg：build_from_cfg 可以根据配置字典 cfg 和 registry 注册表构建模块，这里用于构建 depth_branch
from mmcv.utils import build_from_cfg

# 从 MMCV 的 CNN 插件层注册表中导入 PLUGIN_LAYERS
# 自定义层、插件层通常会注册到这个 registry 里
# depth_branch 若是自定义插件模块，就可以通过这个注册表构建
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

# 从 MMDetection 中导入模型注册器和模型构建函数
from mmdet.models import (
    DETECTORS,      # DETECTORS 是检测器注册表：SparseDrive 会通过 @DETECTORS.register_module() 注册进去
    BaseDetector,   # BaseDetector 是 MMDetection 中检测器的基类：SparseDrive 继承它，从而接入 MMDetection 的训练、测试、推理框架
    build_backbone, # build_backbone 根据配置字典构建 backbone：例如 ResNet、VoVNet、ConvNeXt 等图像主干网络
    build_head,     # build_head 根据配置字典构建检测/规划头：SparseDrive 的核心检测、跟踪、地图、运动规划等输出通常在 head 里完成
    build_neck,     # build_neck 根据配置字典构建 neck：neck 常用于多尺度特征融合，例如 FPN
)


# 从当前目录下导入 GridMask 数据增强模块：GridMask 是一种图像遮挡增强方法，会随机遮挡图像中的网格区域
from .grid_mask import GridMask

# 尝试导入自定义 CUDA/算子中的 feature_maps_format：该函数一般用于把多层特征图整理成 deformable aggregation 需要的格式
try:
    from ..ops import feature_maps_format # 从上一级目录的 ops 包中导入 feature_maps_format：这里的 ..ops 通常对应 projects/mmdet3d_plugin/ops
    DAF_VALID = True                      # 若导入成功，说明 deformable aggregation function 可用
except:                                   # 若导入失败，说明对应的自定义算子没有编译或环境不可用
    DAF_VALID = False                     # 标记 deformable aggregation function 不可用


# __all__ 用于控制 from xxx import * 时暴露哪些对象
__all__ = ["SparseDrive"] # 这里表示该模块主要暴露 SparseDrive 类


# 将 SparseDrive 注册到 MMDetection 的 DETECTORS 注册表中，这样配置文件里写 type='SparseDrive' 时，框架就能自动找到这个类
@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    '''
        定义 SparseDrive 检测器类，继承自 BaseDetector
        SparseDrive 检测器类是整个模型的外层封装，负责：
            1. 构建 backbone、neck、head
            2. 提取图像特征
            3. 区分训练和测试流程
            4. 调用 head 完成损失计算或后处理
    '''

    # 一、__init__ 函数：构建模型组件，处理配置参数
    def __init__(
        self,                      # self 表示当前 SparseDrive 对象
        img_backbone,              # img_backbone 是图像主干网络配置，例如 dict(type='ResNet', depth=50, ...)，是构建 backbone 的配置字典，必须提供
        head,                      # head 是 SparseDrive 的核心任务头配置，检测、跟踪、地图、运动规划等主要逻辑通常在 head 中，例如 dict(type='SparseDriveHead', ...)，是构建 head 的配置字典，必须提供
        img_neck=None,             # img_neck 是图像 neck 配置，默认 None，若提供了 neck 配置（例如 dict(type='FPN', ...）），则会构建 neck 模块；若不提供，则不使用 neck，直接把 backbone 输出的多尺度特征送到 head 里
        init_cfg=None,             # init_cfg 是 MMCV/MMDetection 的初始化配置，用于指定权重初始化方式或预训练权重，通常是一个字典或字典列表，例如 dict(type='Pretrained', checkpoint='path/to/checkpoint.pth')。SparseDrive 会把这个 init_cfg 传给 BaseDetector，BaseDetector 会根据 init_cfg 来加载预训练权重或进行权重初始化。
        train_cfg=None,            # train_cfg 是训练配置，通常包含训练超参数、数据增强策略、损失权重等信息，当前这个类里没有直接使用，但保留接口以兼容 MMDetection 风格
        test_cfg=None,             # test_cfg 是测试配置，通常包含测试时的后处理策略、NMS 参数等信息，当前这个类里没有直接使用，但保留接口以兼容 MMDetection 风格
        pretrained=None,           # pretrained 是旧版 MMDetection 常见的预训练参数入口，新版通常更推荐使用 init_cfg
        use_grid_mask=True,        # 是否使用 GridMask 图像增强，若为 True，则会在 extract_feat() 中对输入图像进行 GridMask 增强；若为 False，则不使用 GridMask，直接把原始图像送入 backbone 提取特征
        use_deformable_func=False, # 是否使用自定义 deformable aggregation function，若为 True，则会在 extract_feat() 中把 backbone/neck 输出的 feature maps 转成 deformable aggregation function 需要的格式；若为 False，则保持原始格式，直接送到 head 里。Stage1 和 Stage2 配置里都设置 use_deformable_func=True，因此会进入该分支。
        depth_branch=None,         # depth_branch 是可选的深度预测分支配置，若提供了 depth_branch 配置（例如 dict(type='DepthBranch', ...）），则会构建该分支模块，并在 extract_feat() 中计算深度预测结果；若不提供，则不使用深度分支，extract_feat() 中的 depths 会被设置为 None，这样训练时就不会计算 dense depth loss
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg) # 调用父类 BaseDetector 的初始化函数：init_cfg 会传给 BaseDetector，用于模型初始化

        # 1. 若用户传入了 pretrained，说明想给 backbone 加载预训练权重：
        if pretrained is not None:
            # 注意：这里源码中写的是 backbone.pretrained
            # 但当前作用域里并没有 backbone 这个变量
            # 按逻辑看，这里很可能应该是 img_backbone.pretrained = pretrained
            # 否则 pretrained 不为 None 时会触发 NameError
            backbone.pretrained = pretrained


        # 2. 配置模型三组件：backbone、neck、head
        # (2.1) 根据 img_backbone 配置构建图像主干网络（例如 build_backbone(dict(type='ResNet', ...))）：
        self.img_backbone = build_backbone(img_backbone)

        # (2.2) 若配置中提供了 img_neck，则根据 img_neck 配置构建 neck 模块（例如 build_neck(dict(type='FPN', ...))）：
        if img_neck is not None:
            self.img_neck = build_neck(img_neck) # 根据 img_neck 配置构建 neck（neck 通常用于多尺度特征融合，例如 FPN）

        # (2.3) 根据 head 配置构建 SparseDrive 的任务头（任务头是模型输出预测结果、计算 loss、做 post_process 的核心部分）：
        self.head = build_head(head) # 见 /home/lzw/SparseDrive/projects/mmdet3d_plugin/models/sparsedrive_head_lzw.py 中的 SparseDriveHead 类


        # 3.1 保存是否使用 GridMask 的标志：
        self.use_grid_mask = use_grid_mask


        # 4. deformable function 相关设置：
        # (4.1) 若启用 deformable function，则检查自定义 deformable aggregation 算子是否成功导入：
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up." # 检查自定义 deformable aggregation 算子是否成功导入，若没有导入成功则直接报错assert
        # (4.2) 保存是否使用 deformable function 的标志：
        self.use_deformable_func = use_deformable_func


        # 5. depth_branch 相关设置：
        if depth_branch is not None:                                        # (1) 若配置了 depth_branch，则构建该 depth_branch 分支模块：
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS) # 使用 build_from_cfg 从 PLUGIN_LAYERS 注册表中构建 depth_branch【depth_branch 通常是自定义插件模块，不一定属于标准 backbone/head/neck】
        else:                                                               # (2) 若没有配置 depth_branch，说明不需要深度监督，那么直接把 self.depth_branch 设置为 None，这样后续训练时就不会计算 dense depth loss
            self.depth_branch = None


        # 3.2 若启用了 GridMask，则构建 GridMask 数据增强模块：
        if use_grid_mask:
            self.grid_mask = GridMask( # 构建 GridMask 数据增强模块【见 grid_mask.py 的 GridMask 类】
                True,                  # 第一个 True：通常表示是否沿 h 方向使用 mask
                True,                  # 第二个 True：通常表示是否沿 w 方向使用 mask 
                rotate=1,              # rotate=1 表示允许小范围旋转 grid mask
                offset=False,          # offset=False 表示不使用随机 offset 填充值
                ratio=0.5,             # ratio=0.5 表示遮挡比例相关参数
                mode=1,                # mode=1 表示 GridMask 的遮挡模式
                prob=0.7               # prob=0.7 表示使用 GridMask 的概率为 0.7
            )


    # 二、extract_feat 函数：从输入图像中提取多尺度图像特征
    # auto_fp16 装饰器用于自动混合精度；pply_to=("img",) 表示只对 img 参数做 fp16 转换；out_fp32=True 表示函数输出会转回 fp32
    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        '''
            extract_feat()函数 用于从输入图像中提取多尺度图像特征
            输入 img 可能是单目图像或多相机图像：
                1. 单目图像：[B, C=3, H=256, W=704]
                2. 多相机图像：[B, N=6, C=3, H=256, W=704]
                其中 B=batch size，N=numbers_of_cameras=6，C=numsber_of_channels=3，H=Image_height=256，W=Image_width=704
                    (1) 这里已在 sparsedrive_small_stage1.py和sparsedrive_small_stage2.py 设置好了 num_cams = 6 和 input_shape = (704, 256)，即H=Image_height=256，W=Image_width=704 
                    (2) 在Stage1和Stage2中，官方设置的batch_size不同：
                            见 sparsedrive_small_stage1.py 中的 total_batch_size = 64、num_gpus = 8、batch_size = 8
                            见 sparsedrive_small_stage2.py 中的 total_batch_size = 48、num_gpus = 8、batch_size = 6
        '''

        # 获取 batch _size：
        bs = img.shape[0]
        '''
            若 img 是 [B, N, C, H, W]，则bs = B
            若 img 是 [B, C, H, W]，则bs = B        
        '''

        # 1. 将 img 形状调整为 backbone 可接受的格式：多相机图像 img 的形状调整为 [B*N, C, H, W]、单相机图像保持 [B, C, H, W]：
        if img.dim() == 5:               # (1) 若 img 是 5 维（即[B, N, C, H, W]），说明是多视角输入，则将 batch 维和 camera 维合并，即 [B, N, C, H, W] -> [B*N, C, H, W]
            num_cams = img.shape[1]      # 获取相机数量 num_cams（多视角输入形状一般是 [B, num_cams, C, H, W]）
            img = img.flatten(end_dim=1) # 将 batch 维和 camera 维合并，即 [B, N, C, H, W] -> [B*N, C, H, W] 。这样 backbone 就可以像处理普通 2D 图像一样处理所有相机图像
        else:                            # (2) 若 img 不是 5 维（即 [B, C, H, W]），说明不是多相机格式，则直接使用原始 img 形状 [B, C, H, W] 进行后续处理
            num_cams = 1                 # 单相机情况下，相机数量记为 1

        # 2. 若启用了 GridMask，则对图像进行 GridMask 增强，并返回增强后的图像（形状不变）：
        if self.use_grid_mask:
            img = self.grid_mask(img) # 对图像进行 GridMask 增强【输入输出形状不变，仍然是 [B*N, C, H, W] 或 [B, C, H, W]】

        # 3. 将图像 img 送入 backbone (采用ResNet-50) 和 neck (采用FPN) 提取多尺度特征，得到融合多尺度信息且通道数统一为256的多尺度特征列表 feature_map
        # (3.1) 将图像 img 输入 backbone (采用ResNet-50) 提取特征，得到 ResNet 的 4 个 stage 输出，作为特征图列表 feature_maps【形状是 list，每个元素是一层特征图，例如 [B*N, C_l, H_l, W_l]】。
        '''
            SparseDrive 的 Stage1 和 Stage2 配置（sparsedrive_small_stage1.py和sparsedrive_small_stage2.py）里 img_backbone 都是 ResNet-50，并且 out_indices=(0, 1, 2, 3)。
            因此 backbone 会输出 C2/C3/C4/C5 四层特征。
            若输入 img 是多相机图像，前面已经把 [B, N, 3, 256, 704] 展平成 [B*N, 3, 256, 704]。
            对 ResNet-50 来说，backbone 输出是：
                C2 即 feature_maps[0]: [B*N,  256, 64, 176]，stride=4
                C3 即 feature_maps[1]: [B*N,  512, 32,  88]，stride=8
                C4 即 feature_maps[2]: [B*N, 1024, 16,  44]，stride=16
                C5 即 feature_maps[3]: [B*N, 2048,  8,  22]，stride=32
            这里的 256 和 704 来自配置里的 final_dim=(256, 704)，即输入给 backbone 的图像高 H=256、宽 W=704。        
        '''
        # a. 检查 img_backbone.forward 的参数列表中是否包含 metas（有些自定义 backbone 可能需要 metas，例如相机参数、图像元信息等）：
        if "metas" in signature(self.img_backbone.forward).parameters:
            # 若 backbone 支持 metas 参数，则把 num_cams 和 metas 一起传进去【当前 Stage1 和 Stage2 里配置的 ResNet-50 只接收 img，因此通常不会走这个分支】
            feature_maps = self.img_backbone(img, num_cams, metas=metas) # feature_maps 是一个 list/tuple，每个元素是一层特征图，例如 [B*N, C_l, H_l, W_l]
        # b. 若 backbone.forward 不支持 metas：
        else:
            # 只传入图像 img【标准 CNN backbone 通常就是这种形式，当前 Stage1 和 Stage2 里配置的 ResNet-50 只接收 img，走的就是该分支】
            # 输入:  [B*N, 3, 256, 704]
            # 输出:  tuple/list，共 4 层：
            #          feature_maps[0] 的 shape 为 [B*N,  256, 64, 176],
            #          feature_maps[1] 的 shape 为 [B*N,  512, 32,  88],
            #          feature_maps[2] 的 shape 为 [B*N, 1024, 16,  44],
            #          feature_maps[3] 的 shape 为 [B*N, 2048,  8,  22]
            feature_maps = self.img_backbone(img)

        # (3.2) 若模型配置了 img_neck，则将 backbone (采用ResNet-50) 输出的 feature_maps 送入 neck (采用FPN) 做通道统一和多尺度融合：
        '''
            SparseDrive 的 Stage1 和 Stage2 配置（sparsedrive_small_stage1.py和sparsedrive_small_stage2.py）里 img_neck 都是 FPN：
                in_channels=[256, 512, 1024, 2048]
                out_channels=256
                num_outs=4
            因此 FPN 不改变这 4 个尺度的大致 H/W，只把每层通道数都变成 256。
            FPN 输入的 feature_maps 列表 (即 backbone (采用ResNet-50) 输出的 C2/C3/C4/C5 四层特征图):
                feature_maps[0]: [B*N,  256, 64, 176],
                feature_maps[1]: [B*N,  512, 32,  88],
                feature_maps[2]: [B*N, 1024, 16,  44],
                feature_maps[3]: [B*N, 2048,  8,  22]
            FPN 输出的 feature_maps 列表 (输出依旧是多尺度特征图，但通道数统一为 256):
                P2 即 feature_maps[0]: [B*N, 256, 64, 176]
                P3 即 feature_maps[1]: [B*N, 256, 32,  88]
                P4 即 feature_maps[2]: [B*N, 256, 16,  44]
                P5 即 feature_maps[3]: [B*N, 256,  8,  22]        
        '''
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps)) # 将 backbone (采用ResNet-50) 输出送入 neck (采用FPN)，neck 输出也是多尺度特征图 # list(...) 是为了确保后续可以通过索引修改 feature_maps[i]

        # (3.3) 遍历每一层 FPN 特征图，把 [B*N, C, H, W] 还原为 [B, N, C, H, W]：
        '''
            将每层特征图从 [B*N, C, H, W] 还原成 [B, N, C, H, W]，还原后，head 可以显式区分 batch 维和 camera 维（即 head 可以显式知道每个 batch 里有 N 个 camera）。
            以 nuScenes 的 6 相机为例：
                (1) 官方训练 Stage1 每卡 batch_size=8，则 B=8、N=6、B*N=48【见 sparsedrive_small_stage1.py 中的 total_batch_size = 64、num_gpus = 8、batch_size = 8】
                        还原前 P2/P3/P4/P5:
                            [48, 256, 64, 176], [48, 256, 32, 88], [48, 256, 16, 44], [48, 256, 8, 22]
                        还原后 P2/P3/P4/P5:
                            [8, 6, 256, 64, 176], [8, 6, 256, 32, 88], [8, 6, 256, 16, 44], [8, 6, 256, 8, 22]
                (2) 官方训练 Stage2 每卡 batch_size=6，则 B=6、N=6、B*N=36【见 sparsedrive_small_stage2.py 中的 total_batch_size = 48、num_gpus = 8、batch_size = 6】
                        还原前 P2/P3/P4/P5:
                            [48, 256, 64, 176], [48, 256, 32, 88], [48, 256, 16, 44], [48, 256, 8, 22]
                        还原后 P2/P3/P4/P5:
                            [6, 6, 256, 64, 176], [6, 6, 256, 32, 88], [6, 6, 256, 16, 44], [6, 6, 256, 8, 22]        
        '''
        for i, feat in enumerate(feature_maps): # 遍历每一层特征图，i 是特征层索引，feat 是对应特征图
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])
            '''
                feat: 特征图列表中的每一层特征图，例如 [B*N, 256, 64, 176]
                (bs, num_cams) + feat.shape[1:] 计算目标形状：
                    计算公式是: (bs, num_cams) + feat.shape[1:] = (B, N) + (C, H, W) = (B, N, C, H, W)
                    若 feat.shape 是 [B*N, C, H, W]，那么 reshape 后就是 [B, N, C, H, W]
            '''

        # 4. 若需要返回深度且配置了 depth_branch，则调用 depth_branch 预测深度
        if return_depth and self.depth_branch is not None: # 若需要返回深度且配置了 depth_branch
            # 使用 depth_branch 预测深度：输入是多尺度 feature_maps 和 metas.get("focal")【metas.get("focal") 通常用于获取相机焦距信息, 以提供深度预测可能需要的相机焦距信息，因为深度估计经常需要焦距参与尺度恢复或归一化】
            depths = self.depth_branch(feature_maps, metas.get("focal"))
            '''
                预测 dense depth，用作辅助监督。
                这通常发生在训练阶段，因为 forward_train 会调用 extract_feat(img, True, data)。
                DenseDepthNet 详见 《blocks.py》 和 《4.2.4_blocks.py里的DenseDepthNet类的理解.md》。
                1. DenseDepthNet 的输入：
                        DenseDepthNet 使用前 num_depth_layers=3 个 FPN level，通常是前三层【在配置文件 sparsedrive_small_stage1.py和sparsedrive_small_stage2.py 中设置了 num_depth_layers=3】，
                        因此 DenseDepthNet 的输入特征形状是：[B, N=6, 256, 64, 176]、[B, N=6, 256, 32, 88]、[B, N=6, 256, 16, 44]。
                2. DenseDepthNet 的结构：
                        源码里 DenseDepthNet 初始化时，会创建 self.depth_layers = nn.ModuleList()，然后每个深度层都是一个：
                            nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
                        也就是说，每个 FPN level 都用一个 1×1 卷积，把 256 通道变成 1 通道深度图。
                3. DenseDepthNet 的输出：
                        DenseDepthNet 输出为预测深度 depths，它是一个 list，包含 num_depth_layers=3 个元素，每个元素是对应 FPN level 的深度预测图，形状分别是：[B, N=6, 1, 64, 176]、[B, N=6, 1, 32, 88]、[B, N=6, 1, 16, 44]。
                4. DenseDepthNet 的监督：gt_depth 用于 DenseDepthNet 辅助监督，将 gt_depth 与 DenseDepthNet 输出的预测深度 depths 进行 loss 计算得到 dense_depth_loss，帮助 backbone 和 neck 学习更有深度感知的特征。
                5. DenseDepthNet 在训练时怎么接入总 loss：
                        在 SparseDrive.forward_train() 里，先调用：
                                feature_maps, depths = self.extract_feat(img, True, data)
                        这里 True 表示 return_depth=True，所以 extract_feat() 会返回图像特征和 depth 分支预测结果。之后主 head 算完自己的 loss 后，若 depths 不为空且 data 里有 gt_depth，就额外加入：
                                output["loss_dense_depth"] = self.depth_branch.loss(depths, data["gt_depth"])
                        源码里就是这个流程。
                        而在配置文件 sparsedrive_small_stage1.py和sparsedrive_small_stage2.py 中设置了 loss_weight=0.2，这是 dense_depth_loss 在总 loss 中的权重为0.2，
                        所以 Stage2 的总 loss 可以理解成：
                                总 loss =
                                    detection loss
                                    + map loss
                                    + motion loss
                                    + planning loss
                                    + 0.2 * dense depth loss
                        而 Stage1 没有 motion/planning，所以 Stage1 的总 loss 更像：
                                总 loss =
                                    detection loss
                                    + map loss
                                    + 0.2 * dense depth loss
            '''
        else: # 若不需要返回深度，或者没有配置 depth_branch，则 depths 设置为 None
            depths = None

        # 5. 若启用了 deformable function，则调用 feature_maps_format() 把 feature_maps 转成自定义算子 deformable aggregation function 需要的格式：
        if self.use_deformable_func: # Stage1 和 Stage2 配置里都设置 use_deformable_func=True，因此会进入该if分支
            feature_maps = feature_maps_format(feature_maps)
            '''
                转换前：特征图列表 feature_maps 是4个特征图张量组成的 list，4个特征图张量的形状依次是 [B,N=6,256,64,176]、[B,N=6,256,32,88]、[B,N=6,256,16,44]、[B,N=6,256,8,22]
                转换后：feature_maps 变成 [col_feats, spatial_shape, scale_start_index]：
                            col_feats 是把所有特征图展平并拼接成一个大特征矩阵，形状是 [B, N*(64*176+32*88+16*44+8*22), 256]，即 [B, 89760, 256]。
                            spatial_shape 是每个相机每层特征图的空间形状，形状是 [N=6, 4, 2]，其中 N=6 是相机数量，4 是特征层数，2 是 H/W。
                            scale_start_index 是每个相机每层特征图在 col_feats 中的起始索引，形状是 [N=6, 4]。
            '''

        # 6. 返回 feature maps 和 dense depth，或只返回 feature maps
        # (a) 当 return_depth=True 时，同时返回 feature maps 和 dense depth。
        if return_depth:                # 训练时 forward_train() 会设置 return_depth=True，因此训练时会同时返回图像特征和深度预测结果，因为 forward_train() 需要计算 loss_dense_depth
            return feature_maps, depths # 返回图像特征和深度预测结果

        # (b) 默认只返回图像特征，测试阶段 simple_test() 则会使用该分支，只返回图像特征即可，因为 simple_test() 不需要计算 dense depth loss
        return feature_maps # 默认只返回图像特征


    # 三.1、forward 函数：模型的统一前向入口，区分训练和测试流程
    # force_fp32 装饰器用于把 img 参数强制转成 fp32，这里用于 forward 入口，保证输入进入训练/测试流程前是 fp32
    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        '''
            函数作用: forward() 是模型统一入口，训练和测试都会先进入这里
            函数传参: 
                img 是图像输入
                data 是其他数据，例如 metas、gt_bboxes、gt_labels、gt_depth 等        
        '''

        # (1) 若当前模型处于训练模式，即 model.train() 状态：
        if self.training:
            # 走训练前向流程
            # 返回值通常是一个 loss 字典
            return self.forward_train(img, **data)

        # (2) 若当前模型处于测试/推理模式，即 model.eval() 状态：
        else:
            # 走测试前向流程
            # 返回值通常是后处理后的预测结果
            return self.forward_test(img, **data)


    # 三.2、forward_train() 函数：训练阶段的前向传播函数，负责计算损失
    def forward_train(self, img, **data):
        '''
            forward_train 是训练阶段的前向传播函数
            主要流程：
                1. 提取图像特征（图像会依次输入backbone(采用ResNet-50)、GridMask图像增强(若启用)、neck(采用FPN)得到多尺度特征图列表 feature_maps，同时若启用 depth_branch() 则还会得到深度预测结果 depths）
                2. head 前向预测
                3. 计算 head loss
                4. 可选计算 depth loss        
        '''

        # 1. 提取图像特征，并计算预测深度 depth（图像会依次输入backbone(采用ResNet-50)、GridMask图像增强(若启用)、neck(采用FPN)得到多尺度特征图列表 feature_maps，同时若启用 depth_branch() 则还会得到深度预测结果 depths）：
        feature_maps, depths = self.extract_feat(img, True, data)
        '''
            输入：
                data 作为 metas 传入 extract_feat()，提供给 backbone/neck 可能需要的元信息，例如相机参数等。
                第二个参数 True 对应 return_depth=True，表示 extract_feat() 需要同时返回图像特征和深度预测结果。
            输出：
                feature_maps 是图像多尺度特征图列表 list，包含 4 个元素，每个元素是对应 FPN level 的特征图，形状分别是 [B, N=6(个相机), 256, 64, 176]、[B, N=6, 256, 32, 88]、[B, N=6, 256, 16, 44]、[B, N=6, 256, 8, 22]
                depths 是 depth_branch 的预测深度结果，可能为 None       
        '''

        # 2. 将图像特征和数据字典送入 head，得到 head 的原始输出 model_outs【model_outs 是 head 的原始输出，包含检测、跟踪、地图、规划等多个任务的中间预测结果】：
        model_outs = self.head(feature_maps, data) # 调用 /home/lzw/SparseDrive/projects/mmdet3d_plugin/models/sparsedrive_head_lzw.py 中的 SparseDriveHead 类的 forward()
        '''
            model_outs 是一个列表：
                [
                    det_output      # 检测输出
                    map_output      # 地图输出
                    motion_output   # agent运动预测输出
                    planning_output # ego自车规划输出 
                ]
        '''
        
        # 3. 调用 head.loss 计算训练损失：
        output = self.head.loss(model_outs, data) # 调用 /home/lzw/SparseDrive/projects/mmdet3d_plugin/models/sparsedrive_head_lzw.py 中的 SparseDriveHead 类的 loss()
        '''
            output 通常是一个 dict，例如：
                {
                "loss_cls": ...,
                "loss_reg": ...,
                ...
                }
        '''

        # 4. 若 depth_branch 有输出，且 data 里有 gt_depth 真值，则计算 dense depth loss，并把它加入 output 字典：
        if depths is not None and "gt_depth" in data:
            # 计算 dense depth loss，这个 loss 作为额外监督项加入总 loss 字典
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths,          # depth_branch 预测的深度
                data["gt_depth"] # 数据集中提供的深度真值
            )

        # 5. 返回所有 loss，MMDetection 训练器会根据这个 loss 字典进行反向传播
        return output


    # 三.3、forward_test() 函数：测试阶段的前向传播函数，负责区分普通测试和测试时增强 TTA
    def forward_test(self, img, **data): # forward_test() 是测试/推理阶段的入口，负责区分普通测试和测试时增强 TTA，并调用对应的处理函数，它会判断是否是 test-time augmentation 输入
        if isinstance(img, list):                # 若 img 是 list，说明使用了测试时增强 TTA，例如多尺度、翻转等增强版本
            return self.aug_test(img, **data)    # 调用 aug_test 处理增强测试
        else:                                    # 若 img 不是 list，说明是普通测试
            return self.simple_test(img, **data) # 调用 simple_test() 处理单次前向推理


    # 三.4、simple_test() 函数：普通测试流程，负责提取特征、调用 head 预测、后处理输出结果
    def simple_test(self, img, **data): # simple_test() 是普通测试流程，负责提取特征、调用 head 预测、后处理输出结果，它并不包含真正的多尺度测试时增强
        # 1. 提取图像特征：
        feature_maps = self.extract_feat(img) # 这里默认 return_depth=False，所以只返回 feature_maps

        # 2. 将图像特征送入 head，得到原始模型输出：
        model_outs = self.head(feature_maps, data)

        # 3. 对 head 的原始输出进行后处理
        # post_process 通常会做：
        # (1) 解码预测框
        # (2) 过滤低置信度结果
        # (3) 坐标变换
        # (4) 组织检测/跟踪/地图/规划结果
        results = self.head.post_process(model_outs, data)

        # 4. 将每个样本的结果包装成 MMDetection 常见格式（每个 result 被放到 {"img_bbox": result} 中），并返回最终推理结果
        output = [{"img_bbox": result} for result in results]
        return output # 返回最终推理结果


    # 三.5、aug_test() 函数：测试时增强 TTA 流程，负责处理多增强版本的输入，只取第一个增强版本调用 simple_test() 进行推理
    def aug_test(self, img, **data):
        '''
            aug_test 名义上是测试时增强 TTA
            但这里注释写的是 fake test time augmentation
            说明它并没有真正融合多个增强结果，只是取第一个增强版本进行 simple_test() 推理，因此这不是真正的 TTA 融合        
        '''

        # 遍历 data 字典中的所有 key，找到那些包含增强版本数据的 key，并只保留第一个增强版本的数据：
        for key in data.keys():
            if isinstance(data[key], list): # 若某个 data[key] 是 list，说明它也包含多个增强版本的数据
                data[key] = data[key][0]    # 只取第一个增强版本，丢弃其他增强版本的数据，因此这不是真正的 TTA 融合，因为 simple_test() 只能处理单个版本的输入，所以这里直接取第一个版本进行推理

        # 只取 img 的第一个增强版本图像 img[0] 进行 simple_test()
        return self.simple_test(img[0], **data)
