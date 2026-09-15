from typing import List, Optional, Tuple # 从 typing 中导入类型标注
import numpy as np                       # 导入 numpy。这个文件中 np 实际没有直接使用，属于冗余导入
import torch                             # 导入 PyTorch。用于张量计算、矩阵乘法、拼接、reshape 等
import torch.nn as nn                    # 导入 PyTorch 神经网络模块。用于 nn.Module、nn.Conv2d、nn.Dropout、nn.LayerNorm 等

# 导入 autocast
# 用于控制自动混合精度
# DenseDepthNet.loss 里会临时关闭 autocast，保证深度 loss 用 fp32 计算
from torch.cuda.amp.autocast_mode import autocast


# 从 MMCV 中导入 Linear、激活函数构建器、归一化层构建器
# Linear 是 MMCV 封装过的线性层
# build_activation_layer 根据配置构建激活函数，例如 ReLU
# build_norm_layer 根据配置构建归一化层，例如 LN、BN
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer

# 从 MMCV runner 中导入 Sequential 和 BaseModule
# Sequential 类似 torch.nn.Sequential
# BaseModule 是 MMCV 对 nn.Module 的封装，支持 init_cfg 初始化
from mmcv.runner.base_module import Sequential, BaseModule

# 从 MMCV transformer 模块中导入 FFN
# 当前文件中 FFN 实际没有直接使用，属于冗余导入
from mmcv.cnn.bricks.transformer import FFN

# 从 MMCV 导入 build_from_cfg
# 用于根据配置字典和注册表构建模块
from mmcv.utils import build_from_cfg

# 导入 dropout 构建函数
# AsymmetricFFN 中用于构建可配置 dropout_layer
from mmcv.cnn.bricks.drop import build_dropout

# 导入初始化函数
# xavier_init 用于线性层 Xavier 初始化
# constant_init 用于常数初始化
from mmcv.cnn import xavier_init, constant_init

# 导入 MMCV 的注册表
from mmcv.cnn.bricks.registry import (
    # ATTENTION 注册表
    # DeformableFeatureAggregation 会注册到这里
    ATTENTION,

    # PLUGIN_LAYERS 注册表
    # DenseDepthNet 会注册到这里
    # kps_generator、temporal_fusion_module 也会从这里构建
    PLUGIN_LAYERS,

    # FEEDFORWARD_NETWORK 注册表
    # AsymmetricFFN 会注册到这里
    FEEDFORWARD_NETWORK,
)


# 尝试导入自定义 CUDA 加速算子 deformable_aggregation_function
try:
    # DAF 是 deformable aggregation function 的缩写
    # 若编译了 projects/mmdet3d_plugin/ops，就可以使用这个高效实现
    from ..ops import deformable_aggregation_function as DAF
except:        # 若导入失败
    DAF = None # 则 DAF 设为 None，后面若 use_deformable_func=True，则会 assert 报错


# __all__ 用于控制 from blocks import * 时暴露的对象
__all__ = [
    "DeformableFeatureAggregation", # 多相机多尺度 deformable 特征聚合模块
    "DenseDepthNet",                # 深度辅助监督网络
    "AsymmetricFFN",                # 非对称 FFN 网络
]

# 工具函数，用于快速堆叠 [Linear+ReLU]*N + LayerNorm 的重复块，如用在 camera_encoder 中把相机投影矩阵编码成 camera embedding 等
def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:      # 若没有指定输入维度
        input_dims = embed_dims # 则默认输入维度等于 embed_dims
    layers = []                 # 创建空列表，用来存放网络层

    # (1) 外层循环 out_loops 次，每一轮内层堆叠若干Linear+ReLU，末尾拼接一层LayerNorm
    for _ in range(out_loops):
        # (2) 内层循环 in_loops 次，每轮添加一组 Linear + ReLU
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims)) # 添加线性层：输入维度 input_dims，输出维度 embed_dims
            layers.append(nn.ReLU(inplace=True))          # 添加 ReLU 激活：inplace=True 表示尽量原地操作，节省显存
            input_dims = embed_dims                       # 下一层输入维度更新为当前输出维度 embed_dims

        # (3) 每完成一组内层 Linear+ReLU 堆叠后，便追加层归一化 LayerNorm
        layers.append(nn.LayerNorm(embed_dims))

    # 返回层列表，外部调用时需用 nn.Sequential(*layers) 解包并封装成连续网络【注意这里返回的不是 nn.Sequential，而是 list，调用方会用 Sequential(*layers) 包起来】
    return layers
    '''
        例如调用 linear_relu_ln(embed_dims=256, in_loops=1, out_loops=2, input_dims=12)，则会返回一个列表：
            [
                nn.Linear(12, 256), nn.ReLU(inplace=True), # 因为 in_loops=1
                nn.LayerNorm(256),

                nn.Linear(256, 256), nn.ReLU(inplace=True), # 因为 in_loops=1
                nn.LayerNorm(256) # 因为 out_loops=2
            ]

        例如调用 linear_relu_ln(embed_dims=256, in_loops=2, out_loops=2)，则会返回一个列表：
            [
                nn.Linear(256, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 256), nn.ReLU(inplace=True), # 因为 in_loops=2
                nn.LayerNorm(256),

                nn.Linear(256, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 256), nn.ReLU(inplace=True), # 因为 in_loops=2
                nn.LayerNorm(256) # 因为 out_loops=2
            ]
    '''


# 一、DeformableFeatureAggregation 类是 SparseDrive 里“让每个 3D anchor 去多相机、多尺度图像特征中精准取证，并融合成新的 instance feature”的核心模块。它输出的是更可靠的 instance_feature，随后交给 FFN 和 SparseBox3DRefinementModule 再更新 box、类别和质量分数
@ATTENTION.register_module() # 将 DeformableFeatureAggregation 注册到 ATTENTION 注册表，在配置文件中 deformable_model=dict(type="DeformableFeatureAggregation", ...) 就会构建这个类
class DeformableFeatureAggregation(BaseModule):
    '''    
        DeformableFeatureAggregation 是 SparseDrive 里非常核心的模块，作用：
            1. 根据 3D anchor 生成 3D key points
            2. 将 3D key points 投影到多个相机图像平面
            3. 在多尺度 feature map 上采样特征
            4. 根据 attention weights 融合多相机、多尺度、多点特征
            5. 输出更新后的 instance feature
    '''

    # 1. 初始化函数：构建 key points 生成器、temporal fusion 模块、输出投影层、camera encoder 和权重预测层
    def __init__(
        self,                           # self 表示当前模块对象
        embed_dims: int = 256,          # embedding 维度，默认 256
        num_groups: int = 8,            # 分组数，默认 8，后面会把 embed_dims 分成 num_groups 组做加权融合
        num_levels: int = 4,            # FPN 特征层数量，默认 4
        num_cams: int = 6,              # 相机数量，nuScenes 一般是 6 个相机
        proj_drop: float = 0.0,         # 输出投影后的 dropout 概率
        attn_drop: float = 0.0,         # attention weight dropout 概率
        kps_generator: dict = None,     # key points 生成器配置。检测任务可能是 SparseBox3DKeyPointsGenerator，地图任务可能是 SparsePoint3DKeyPointsGenerator
        temporal_fusion_module=None,    # temporal fusion 模块配置。当前代码里虽然构建了 self.temp_module，但 forward 中没有使用
        use_temporal_anchor_embed=True, # 是否使用 temporal anchor embedding。当前文件中保存了该变量，但 forward 中没有实际使用
        use_deformable_func=False,      # 是否使用自定义 CUDA deformable aggregation function
        use_camera_embed=False,         # 是否使用相机 embedding，True 时会根据 projection_mat 生成 camera_embed，让权重和相机相关

        # 残差模式
        # "add"：输出和输入 instance_feature 相加
        # "cat"：输出和输入 instance_feature 拼接
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__() # 调用父类 BaseModule 初始化

        # embed_dims 必须能被 num_groups 整除，因为后续会把 embedding 分成 num_groups 组
        if embed_dims % num_groups != 0:
            raise ValueError(f"embed_dims must be divisible by num_groups, but got {embed_dims} and {num_groups}") # 若不能整除，直接报错
 
        # (1) 保存参数：
        self.group_dims = int(embed_dims / num_groups)                           # 每组的通道维度。如 embed_dims=256, num_groups=8，则 group_dims=32
        self.embed_dims = embed_dims                                             # 保存 embedding 维度
        self.num_levels = num_levels                                             # 保存 FPN level 数
        self.num_groups = num_groups                                             # 保存 group 数
        self.num_cams = num_cams                                                 # 保存相机数量
        self.use_temporal_anchor_embed = use_temporal_anchor_embed               # 保存是否使用 temporal anchor embedding，当前实现里没有进一步使用
        if use_deformable_func:                                                  # 若启用自定义 deformable function
            assert DAF is not None, "deformable_aggregation needs to be set up." # 则检查 DAF 是否成功导入，若没有编译 ops，这里会报错
        self.use_deformable_func = use_deformable_func                           # 保存是否使用自定义 DAF
        self.attn_drop = attn_drop                                               # 保存 attention dropout 概率
        self.residual_mode = residual_mode                                       # 保存残差模式
        self.proj_drop = nn.Dropout(proj_drop)                                   # 输出投影后的 dropout

        # (2) 根据配置构建 key points 生成器
        kps_generator["embed_dims"] = embed_dims                          # 将 embed_dims 写入 kps_generator 配置，保证 keypoint 生成器知道 embedding 维度
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS) # 从 PLUGIN_LAYERS 注册表中找对应 type
        self.num_pts = self.kps_generator.num_pts                         # 从 key points 生成器中读取采样点数量，如 box 可能有固定点 + 可学习点

        # (3) 构建 temporal fusion module
        if temporal_fusion_module is not None: # 若配置了 temporal fusion module，则从 PLUGIN_LAYERS 中构建 temporal fusion module
            if "embed_dims" not in temporal_fusion_module:        # 若配置里没有 embed_dims
                temporal_fusion_module["embed_dims"] = embed_dims # 自动补上 embed_dims
            self.temp_module = build_from_cfg(temporal_fusion_module, PLUGIN_LAYERS) # 从 PLUGIN_LAYERS 中构建 temporal fusion module
        else:                        # 若没有配置 temporal fusion module
            self.temp_module = None  # temp_module 设为 None

        # (4) 构建输出投影层：将融合后的 features 再映射到 embed_dims
        self.output_proj = Linear(embed_dims, embed_dims)

        # (5) 构建 camera encoder 和权重预测层：
        # (5.1) 若启用 camera embedding
        if use_camera_embed: 
            # (a) 构建 camera_encoder：输入维度是 12，因为 projection_mat[:, :, :3] reshape 后每个相机是 3x4=12 维，输出维度是 embed_dims
            self.camera_encoder = Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
            # (b) 构建权重预测层：启用 camera embedding 时，feature 已经带有相机维度，所以这里只需要预测 num_groups * num_levels * num_pts
            self.weights_fc = Linear(embed_dims, num_groups * num_levels * self.num_pts)
        # (5.2) 若不启用 camera embedding
        else:
            self.camera_encoder = None                                                              # (a) 则不使用 camera encoder
            self.weights_fc = Linear(embed_dims, num_groups * num_cams * num_levels * self.num_pts) # (b) 则权重预测层需要一次性预测所有相机、所有 level、所有点、所有 group 的权重

    # 2. 初始化模块权重【Sparse4DHead.init_weights() 中会遍历子模块并调用 init_weight()】
    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)               # 将 weights_fc 初始化为 0，这样一开始 softmax 后各采样位置权重比较均匀
        xavier_init(self.output_proj, distribution="uniform", bias=0.0) # 对 output_proj 使用 Xavier uniform 初始化

    # 3. DeformableFeatureAggregation 前向传播：
    def forward(
        self,                             # self 表示当前模块对象
        instance_feature: torch.Tensor,   # 当前 query/instance 的特征，shape 通常是 [B, num_anchor, C]
        anchor: torch.Tensor,             # 当前 query 对应的几何 anchor：检测任务中是 3D box anchor，map 任务中是 point/line anchor
        anchor_embed: torch.Tensor,       # anchor 的位置编码，shape 通常是 [B, num_anchor, C]
        feature_maps: List[torch.Tensor], # 多尺度多相机图像特征。list 中每个元素 shape 通常是 [B, num_cams, C, H_l, W_l]
        metas: dict,                      # 数据字典，包含 projection_mat、image_wh 等相机投影信息
        **kwargs: dict,                   # 额外参数，当前函数没有显式使用
    ):
        '''
            【函数总览】
                给定每个 sparse instance 的当前语义特征 instance_feature 与几何 anchor，完成：
                    (1) 根据 anchor 生成多个 3D key points；
                    (2) 预测“每个 query 应从哪台相机 / 哪个 FPN 尺度 / 哪个 key point / 哪组通道取多少信息”的权重；
                    (3) 将 3D key points 投影到多相机图像，并采样多尺度视觉特征；
                    (4) 用可学习权重融合视觉证据；
                    (5) 将融合结果与原 instance_feature 做残差 add / cat，输出更新后的 instance feature。
            
            【Detection 标准 shape】
                instance_feature : [B, N=900, C=256]
                anchor           : [B, N=900, D=11]
                anchor_embed     : [B, N=900, C=256]
                feature_maps[l]  : [B, num_cams=6, C=256, H_l, W_l]
                返回（residual_mode="cat"）: [B, N=900, 2C=512]       
        '''
        # 获取 batch size 和 anchor 数量
        bs, num_anchor = instance_feature.shape[:2]

        # (1) 根据 anchor 和 instance_feature 生成 3D key points，shape [B, num_anchor, num_pts, 3]
        key_points = self.kps_generator(anchor, instance_feature)
        '''
            detection 的 key_points 为：box 中心、固定相对点、由 instance_feature 预测的可学习点
            map 的 key_points 为：polyline 各采样点周围的 3D 取样点
        '''

        # (2) 根据 “语义特征instance_feature、锚框几何编码anchor_embed、相机信息metas” 预测融合权重 weights，shape [B, num_anchor, num_cams, num_levels, num_pts, num_groups]
        weights = self._get_weights(instance_feature, anchor_embed, metas)


        # (3) 从多相机、多尺度 feature_maps 中取得每个 key point 的视觉证据。       
        # (3.1) 若启用自定义 CUDA deformable aggregation：调用 SparseDrive 自定义 CUDA DAF，速度更快
        if self.use_deformable_func:
            # (a) 将 3D key points 投影到 2D 图像坐标【将 [B,N,P,3] 的 3D key points 投影到每个相机的归一化 2D 坐标：先 project_points() 初始返回 [B,num_cams,N,P,2]，再调整成 DAF 所需布局】
            points_2d = (
                self.project_points(
                    key_points,              # 3D key points
                    metas["projection_mat"], # 相机投影矩阵
                    metas.get("image_wh"),   # 图像宽高，用于归一化
                )                                                        # project_points 后 shape [B, num_cams, num_anchor, num_pts, 2]
                .permute(0, 2, 3, 1, 4)                                  # permute 后变成 [B, num_anchor, num_pts, num_cams, 2]
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2) # reshape 成 DAF 需要的格式
            )

            # (b) 将权重 weights 调整为 DAF 所需的维度顺序 [B,N,P,num_cams,num_levels,num_groups]
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5) # [B, num_anchor, num_cams, num_levels, num_pts, num_groups] -> [B, num_anchor, num_pts, num_cams, num_levels, num_groups]
                .contiguous()                     # 保证内存连续
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, self.num_levels, self.num_groups) # reshape 成 DAF 需要的格式
            )

            # (c) 调用 CUDA 加速版 DAF：在各相机、各尺度的 2D 坐标处采样，再用分组权重 weights 完成分组加权融合
            # 输入：多个 feature map、2D points、weights
            # 输出 features reshape 成 [B, num_anchor, embed_dims]
            features = DAF(*feature_maps, points_2d, weights).reshape(bs, num_anchor, self.embed_dims)

        # (3.2) 若不使用自定义 DAF：则调用下方 feature_sampling() + multi_view_level_fusion()，作为纯 PyTorch / grid_sample 的等价回退实现
        else:
            # (a) 使用 PyTorch grid_sample 在各 FPN level、各相机图像特征上采样，输出上采样后的图像特征 features shape [B, num_anchor, num_cams, num_levels, num_pts, embed_dims]
            features = self.feature_sampling(
                feature_maps,            # 多尺度特征
                key_points,              # 3D key points
                metas["projection_mat"], # 相机投影矩阵
                metas.get("image_wh"),   # 图像宽高
            )

            # (b) 对 features 分组加权后融合多相机、多尺度特征，输出“每个 3D key point 的分组加权多相机多尺度融合视觉特征”，shape [B, num_anchor, num_pts, embed_dims]
            features = self.multi_view_level_fusion(features, weights)

            # (c) 对同一个 anchor 的多个 key points 的特征求和合并：[B, num_anchor, num_pts, C] -> [B, num_anchor, C]
            features = features.sum(dim=2)  # fuse multi-point features


        # (4) 对融合后的视觉特征进行 “输出投影 + dropout”：[B, num_anchor, C] -> [B, num_anchor, C]
        output = self.proj_drop(self.output_proj(features))

        # (5) 将新视觉特征与输入 instance feature 进行残差融合
        if self.residual_mode == "add":                            # 若 residual_mode 是 "add"：要求两边都是 C 维，输出仍为 [B,N,C]
            output = output + instance_feature                     # 残差相加，输出维度仍然是 C
        elif self.residual_mode == "cat":                          # 若 residual_mode 是 "cat"：视觉聚合 feature 与旧 instance feature 都保留，拼接成 [B,N,2C]，随后由 AsymmetricFFN 压回 C 维
            output = torch.cat([output, instance_feature], dim=-1) # 残差拼接，输出维度变成 2C，这也是为什么 detection3d_head.py 里 FFN 的输入 in_channels 常常是 embed_dims * 2 =512

        # (6) 返回聚合后的 instance feature
        return output

    # 4. 根据 instance_feature 和 anchor_embed 预测 “多相机、多尺度、多 key point、分组通道” 的融合权重
    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        '''
            该函数作用是：
                对每个 instance 预测一张“视觉取证配额表”：一个 query 在每台相机、每个 FPN level、每个 3D key point、每个通道组上各取多少权重。
                最终返回：weights [B,N,num_cams,num_levels,P,num_groups]。
        '''

        # 获取 batch size 和 anchor 数量
        bs, num_anchor = instance_feature.shape[:2]

        # (1) 融合 query 的两类先验信息：内容特征 + 位置编码，得到 feature shape: [B, num_anchor, C]
        feature = instance_feature + anchor_embed # instance_feature 是语义特征，anchor_embed是几何位置、尺寸、朝向、速度编码。

        # (2) 可选，把相机外参内参相关的投影矩阵编码为 camera embedding
        # 若启用 camera_encoder，启用后，同一个 anchor 面对不同相机可以得到不同的权重预测条件
        if self.camera_encoder is not None:
            # (a) 取每台相机投影矩阵 projection_mat 的前 3×4 部分并展平为 12 维，编码成 C 维 camera_embed【projection_mat 通常是 4x4 或 3x4 投影矩阵，这里取 [:3] 并 reshape 成 12 维】
            camera_embed = self.camera_encoder(metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1)) # 输入 [B, num_cams, 12]，输出 [B, num_cams, C]

            # (b) 给每个 anchor 加上每个相机的 camera embedding【即将每个 anchor 的语义+几何条件广播到每台相机，再加上相机自身编码】
            feature = feature[:, :, None] + camera_embed[:, None] # [B, num_anchor, 1, C] + [B, 1, num_cams, C] -> [B, num_anchor, num_cams, C]

        # (3) 通过 weights_fc 预测并归一化所有视觉采样点的分组权重
        weights = (
            self.weights_fc(feature) # (a) 先线性层 weights_fc 给每个通道组预测所有候选视觉位置的未归一化分数 logits
            .reshape(bs, num_anchor, -1, self.num_groups) # reshape 成 [B, num_anchor, *, num_groups]【不使用 camera_embed 时 * = num_cams * num_levels * num_pts；使用 camera_embed 时，feature 本身已有 camera 维，因此 * = num_levels * num_pts】
            .softmax(dim=-2) # (b) 再在采样维度上做 softmax，dim=-2 表示对所有 camera/level/point 位置归一化【归一化的意思是：对固定 [B,N,group]，所有 camera×level×point 权重之和为 1】
            .reshape(bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups) # (c) 最终 reshape 成统一格式
        )

        # (4) 训练时可对注意力权重做 dropout，避免长期依赖同一台相机或同一个 key point
        if self.training and self.attn_drop > 0: # 若是训练模式，并且设置了 attention dropout
            # (a) 生成随机保留 mask [B, num_anchor, num_cams, 1, num_pts, 1]【注意 level 维是 1，所以同一个 camera/point 对各 level 共用 mask】
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype) # 将 mask 移动到 weights 相同设备和类型

            # (b) 对 weights 做 inverted dropout：使 mask > attn_drop 的位置保留，并且权重除以 (1-attn_drop)以保持期望不变；而丢弃位置的权重置 0
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)

        # (5) 返回融合权重供后续 multi_view_level_fusion() 使用
        return weights

    # 5. 静态方法：把当前 ego/lidar 坐标系中的 3D key points 投影到各相机 2D 图像平面，得到平面图像坐标
    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        '''
            作用：输入当前 ego/lidar 坐标系下的 3D 点，使用每台相机的 projection_mat 计算其图像坐标

            输入：
                key_points: [B, num_anchor, num_pts, 3]
                projection_mat: [B, num_cams, 4, 4] 或兼容形状
                image_wh: None或[B, num_cams, 2]，图像宽高
            输出：
                points_2d: [B, num_cams, num_anchor, num_pts, 2]
            当传入给定 image_wh 时，则输出是由像素坐标归一化到大致 [0,1] 的图像坐标，供 grid_sample / DAF 使用。
        '''
        # 读取 batch size、anchor 数量、采样点 key_points 数量
        bs, num_anchor, num_pts = key_points.shape[:3] # 这里只用于 key_points 的维度含义说明，实际并未使用

        # (1) 给 3D 点补齐齐次坐标：[x, y, z] -> [x, y, z, 1]，得到 pts_extend [B, N, P, 4]
        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)

        # (2) 对每台相机进行批量齐次投影：使用 projection_mat 做矩阵乘法
        points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        '''
            projection_mat[:, :, None, None] shape: [B, num_cams, 1, 1, 4, 4]
            pts_extend[:, None, ..., None] shape: [B, 1, num_anchor, num_pts, 4, 1]
            广播矩阵乘法后的结果 shape: [B, num_cams, num_anchor, num_pts, 4, 1]        
        '''

        # (3) 透视除法：用投影后的齐次坐标前两维 x,y 再除以深度 z，得到相机像素平面坐标
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5) # clamp(min=1e-5) 防止 z 太小导致除零或数值爆炸

        # (4) 可选：当传入给定 image_wh 时，则用每台相机自己的 [width,height] 将像素坐标归一化
        if image_wh is not None: # 若传入 image_wh
            # 用图像宽高归一化坐标：image_wh[:, :, None, None] shape: [B, num_cams, 1, 1, 2]，归一化后坐标大致在 [0,1]
            points_2d = points_2d / image_wh[:, :, None, None] # [B,num_cams,N,P,2] / [B,num_cams,1,1,2] -> [B,num_cams,N,P,2]

        # (5) 返回每台相机下每个 3D key point 的 2D 坐标
        return points_2d

    # 6. 静态方法：纯 PyTorch 回退实现：在投影后的 2D 坐标处，使用 PyTorch grid_sample 从多相机多尺度特征图中采样特征，这是不使用 CUDA DAF 时的 fallback 实现
    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],        # feature_maps：多尺度图像特征列表
        key_points: torch.Tensor,                # key_points：3D key points
        projection_mat: torch.Tensor,            # projection_mat：相机投影矩阵
        image_wh: Optional[torch.Tensor] = None, # image_wh：图像宽高，可选
    ) -> torch.Tensor:
        '''
            该函数不计算融合权重，只负责“按给定 3D key point 的投影位置，从每张相机、每个 FPN level 取视觉特征”。
                输入 feature_maps[l]: [B,num_cams,C,H_l,W_l]；
                最终输出 features: [B,N,num_cams,num_levels,P,C]。        
        '''

        # 读取多尺度特征层数、相机数及 3D key point 的 B/N/P 尺寸
        num_levels = len(feature_maps)                 # 特征层数量
        num_cams = feature_maps[0].shape[1]            # 相机数量，feature_maps[0] shape: [B, num_cams, C, H, W]
        bs, num_anchor, num_pts = key_points.shape[:3] # batch size、anchor 数量、采样点数量

        # (1) 调用 project_points()：将 [B,N,P,3] 的 3D key points 投影到每个相机，输出 points_2d [B,num_cams,N,P,2]，若给 image_wh 则坐标已在 [0,1] 空间
        points_2d = DeformableFeatureAggregation.project_points(key_points, projection_mat, image_wh) # 将 3D key points 投影到 2D 图像坐标，若给 image_wh 则输出归一化坐标 [0,1]        
        
        # (2) 为 grid_sample 准备坐标与 batch 维布局。
        points_2d = points_2d * 2 - 1            # grid_sample 要求坐标范围是 [-1, 1]，所以把 [0,1] 线性转换成 [-1,1]
        points_2d = points_2d.flatten(end_dim=1) # 将 batch 维和 camera 维合并展平，后续把每个相机视为一个独立样本：[B, num_cams, num_anchor, num_pts, 2] -> [B*num_cams, num_anchor, num_pts, 2]

        # (3) 遍历所有 FPN level，分别对每张相机执行 grid_sample
        features = []           # 用于保存每个 FPN level 采样到的特征
        for fm in feature_maps: # 遍历每一层 feature map
            # 对该层特征执行 grid_sample，每个 grid_sample 输出 shape [B*num_cams, C, num_anchor, num_pts]
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), # 展平：[B, num_cams, C, H, W] -> [B*num_cams, C, H, W]
                    points_2d              # 采样点坐标，shape [B*num_cams, num_anchor, num_pts, 2]
                )
            )
            '''
                fm: [B,num_cams,C,H_l,W_l] -> [B*num_cams,C,H_l,W_l]；
                points_2d: [B*num_cams,N,P,2]；
                采样输出: [B*num_cams,C,N,P]。            
            '''

        # (4) 汇总并整理所有 level 的采样结果
        features = torch.stack(features, dim=1) # 将不同 level 的采样结果堆叠，stack 后 shape [B*num_cams, num_levels, C, num_anchor, num_pts]
        features = features.reshape(bs, num_cams, num_levels, -1, num_anchor, num_pts).permute(0, 4, 1, 2, 5, 3) # reshape + permute 成统一格式：[bs, num_anchor, num_cams, num_levels, num_pts, embed_dims]

        # (5) 返回“尚未加权融合”的原始多相机、多尺度采样特征
        return features

    # 7. 分组加权的多相机多尺度特征融合：利用可学习分组注意力权重对多相机、多尺度层级特征加权后聚合，输出每个采样点融合后的统一特征
    def multi_view_level_fusion(
        self,                   # self 表示当前模块对象
        features: torch.Tensor, # [B, num_anchor, num_cams, num_levels, num_pts, embed_dims]，即[批次B，锚框数A，相机数C，特征层级数L，采样点数P，总嵌入维度D]，这里 D=G×d，d是单组维度group_dims，总特征拆成G个并行特征组
        weights: torch.Tensor,  # [B, num_anchor, num_cams, num_levels, num_pts, num_groups]，即[批次B，锚框数A，相机数C，特征层级数L，采样点数P，分组数G]，权重 weights 是每个 (相机 + 层级 + 采样点) 在各组的注意力权重
    ):
        '''
            输入：
                features：[批次B，锚框数A，相机数C，特征层级数L，采样点数P，总嵌入维度D]：每个 3D 锚点的各相机、各尺度投影关键点原始视觉特征
                weights：[批次B，锚框数A，相机数C，特征层级数L，采样点数P，分组数G]：网络学习得到的分组注意力权重，用来给不同相机、尺度、关键点的各组特征分配重要程度。
            输出：
                融合后 features：[批次B, 锚框数A, 采样点数P，总嵌入维度D]：每个锚点下每个关键点的 经分组注意力加权后的 融合 “全部相机 + 全部尺度” 的最终特征

            直观类比理解：
                假设：
                    G=2：两组，组 1 管形状，组 2 管颜色
                    C=3：左相机、中相机、右相机
                    L=2：层级，浅层和深层
                multi_view_level_fusion()：
                    1. 每个相机、每层特征拆成形状、颜色两组；
                    2. 网络给每组打分：比如右相机浅层形状权重高、左相机深层颜色权重高；
                    3. 每个组内把左中右相机特征全部加起来；
                    4. 再把浅层、深层特征加起来；
                    5. 最后把形状组、颜色组特征拼在一起，得到融合完多视角多尺度的单点特征。
        '''

        # 获取 batch size 和 anchor 数量
        bs, num_anchor = weights.shape[:2]

        # (1) 特征分组 + 加权融合：将 features 的最后一维 embed_dims 拆成 [num_groups, group_dims]，然后乘以 weights
        features = weights[..., None] * features.reshape(features.shape[:-1] + (self.num_groups, self.group_dims))
        '''
            特征分组 + 加权融合：
                features.reshape(..., (G, dg))：shape变为 [B,A,C,L,P,G,dg]，意思是：总特征维度 D=G×d，d是单组维度group_dims，这里将总特征拆成G个并行特征组
                weights[..., None]：在 weights [B,A,C,L,P,G] 末尾新增一维 → [B,A,C,L,P,G,1]
                features = weights[..., None] * 分组后features

            (1) 的逻辑理解：
                a. 特征分组：分组多头注意力机制，即把单条特征向量切分成G个独立子组，每组单独分配一套权重weight：
                        不同组可以关注不同视觉信息：一组关注轮廓、一组关注纹理、一组关注深度；
                        权重w由网络学习得到，是软注意力分数，范围通常 0~1；
                        权重只缩放对应分组特征，各组之间互不干扰，表达能力更强。
                b. 求和融合 = 加权叠加多源信息：这里用sum而非均值，属于累加式融合：
                        若某相机 / 层级特征置信度高，网络会学习更大w，累加后贡献更大；
                        若某相机遮挡、某层级噪声大，网络会学习接近 0 的w，自动抑制无效特征；
                        数学本质：线性加权聚合，整体是线性变换操作。
        '''

        # (2) “跨相机求和 + 跨层级求和” 的多视图多尺度融合，即对 camera 维和 level 维求和：
        features = features.sum(dim=2).sum(dim=2) # 原 shape [B, num_anchor, num_cams, num_levels, num_pts, num_groups, group_dims]，sum(dim=2) 后去掉 num_cams 相机维，再 sum(dim=2) 去掉 num_levels 层级维
        '''
            (2) 的逻辑理解：
                    两层求和分别解决两个问题：
                        sum(dim=C)：融合多相机环视图像，消除单相机遮挡、视角盲区；
                        sum(dim=L)：融合多层级特征（浅层细粒度纹理 + 深层全局语义），兼顾大小物体检测；
                    最终每个 3D 采样点num_pts，拥有融合了全部相机、全部尺度、分组注意力筛选的统一特征。
        '''

        # (3) 特征重组，重组回原始嵌入维度
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims) # reshape 回 [B, num_anchor, num_pts, embed_dims]
        return features # 返回融合后的每个 key point 特征【最终每个 3D 采样点num_pts，拥有融合了全部相机、全部尺度、分组注意力筛选的统一特征】


'''
    二、DenseDepthNet类
        DenseDepthNet类 是深度辅助监督分支：它从多尺度图像特征中预测 dense depth map
        DenseDepthNet类 用于训练时辅助监督，提供深度信息引导特征学习，它的输出不一定直接用于最终检测和规划，它主要是帮助图像 backbone 学到更好的几何深度信息。
            训练时和 gt_depth 做 L1 误差
            注意它主要用于辅助监督，不一定作为最终推理输出使用
        调用的地方：在《sparsedrive.py》里的 extract_feat() 函数中会调用 depth_branch()（即 DenseDepthNet）来预测深度，并在训练时计算深度 loss。
'''
# 将 DenseDepthNet 注册到 PLUGIN_LAYERS，在配置文件 depth_branch=dict(type="DenseDepthNet", ...) 中会用到
@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    # 1. 初始化函数，定义模块参数和层：
    def __init__(
        self,               # self 表示当前模块对象
        embed_dims=256,     # 输入特征通道数，默认 256
        num_depth_layers=1, # 使用多少个 FPN level 做深度预测
        equal_focal=100,    # 等效焦距：用于对不同相机焦距下的深度预测做尺度修正
        max_depth=60,       # 最大深度，loss 里会 clip 到该范围
        loss_weight=1.0,    # 深度 loss 权重
    ):
        super().__init__() # 调用父类初始化

        # (1) 保存输入参数：
        self.embed_dims = embed_dims             # 保存输入通道数
        self.equal_focal = equal_focal           # 保存等效焦距
        self.num_depth_layers = num_depth_layers # 保存深度预测层数量
        self.max_depth = max_depth               # 保存最大深度
        self.loss_weight = loss_weight           # 保存 loss 权重

        # (2) 创建深度预测层列表:
        self.depth_layers = nn.ModuleList() # 存储深度预测层列表
        for i in range(num_depth_layers):   # 遍历每个深度预测层
            self.depth_layers.append(nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)) # 深度预测层列表的每个 level 用一个 1x1 卷积从 embed_dims 通道预测 1 通道深度

    # 2. DenseDepthNet 前向传播：
    def forward(self, feature_maps, focal=None, gt_depths=None):
        '''
            输入：
                feature_maps：SparseDrive 提取的多尺度特征
                focal：每张图像的焦距，用于尺度校正
                gt_depths：可选，若传入且处于训练模式，则直接返回 loss        
        '''

        # (1) 获取 focal 参数: 
        if focal is None:             # (a) 若没有传入 focal，则使用 equal_focal 作为默认焦距 
            focal = self.equal_focal  # 使用 equal_focal 作为默认焦距
        else:                         # (b) 若传入了 focal，则将focal拉平为一维，以匹配 feature_maps 的 batch size 和相机数量
            focal = focal.reshape(-1) # 拉平成一维：训练中 feature_maps flatten 后是 [B*num_cams, ...]，所以 focal 也要对应到 B*num_cams

        # (2) 从前 num_depth_layers 个 feature map 中预测深度：
        depths = []                                                      # 保存每个 level 的深度预测
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]): # 遍历前 num_depth_layers 个 feature map：feat 是当前特征图列表的特征图，i是其索引
            # feat 原 shape 通常是 [B, num_cams, C, H, W]
            # flatten(end_dim=1) 后为 [B*num_cams, C, H, W]
            # float() 保证深度分支用 fp32 输入
            # 1x1 conv 输出 [B*num_cams, 1, H, W]
            # exp() 保证深度预测为正数
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()

            # 将第 0 维转到最后，方便与 focal 做广播
            # depth shape: [B*num_cams, 1, H, W]
            # transpose(0, -1) 后，B*num_cams 维到最后
            depth = depth.transpose(0, -1) * focal / self.equal_focal

            # 再转回原来的维度顺序
            depth = depth.transpose(0, -1)

            # 保存当前 level 深度预测
            depths.append(depth)

        # (3.a) 如果传入了 gt_depths，并且处于训练模式，则计算并返回深度 loss：
        if gt_depths is not None and self.training: # 若传入 gt_depths 并且当前是训练模式
            loss = self.loss(depths, gt_depths)     # 直接计算深度 loss
            return loss                             # 返回深度 loss

        # (3.b) 否则返回深度预测列表
        return depths

    # 3. 计算深度 loss：
    def loss(self, depth_preds, gt_depths):
        '''
            计算深度预测 loss
                depth_preds：模型预测的多尺度深度
                gt_depths：GT 多尺度深度        
        '''

        # 初始化总 loss
        loss = 0.0

        # 遍历每个尺度的预测和 GT，计算 L1 误差并累加到总 loss：
        for pred, gt in zip(depth_preds, gt_depths):
            # (1) 将 pred 和 gt 拉平成一维，方便后续计算：
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1) # pred 原 shape 是 [B*num_cams, 1, H, W]，permute 成 [B*num_cams, H, W, 1]，再拉平成一维
            gt = gt.reshape(-1)                                      # GT 也拉平成一维

            # (2) 设置前景有效 mask：
            # 同时满足以下2个条件时，该像素被认为是有效的前景像素：
            # 1. gt > 0，说明该像素有有效深度
            # 2. pred 不是 NaN
            fg_mask = torch.logical_and(gt > 0.0, torch.logical_not(torch.isnan(pred)))

            # (3) 只保留有效像素的 GT 和 pred：
            gt = gt[fg_mask]     # 只保留有效 GT
            pred = pred[fg_mask] # 只保留有效 pred

            # (4) 将预测深度限制在 [0, max_depth]
            pred = torch.clip(pred, 0.0, self.max_depth)

            # (5) 关闭 autocast，确保计算 loss 时使用 fp32 精度：
            # 即使外部使用 fp16 混合精度，这里也强制用 fp32 算 loss
            with autocast(enabled=False):
                '''
                    1. autocast 全称 Automatic Mixed Precision（自动混合精度）
                    2. with autocast() 的 核心行为：
                            默认 enabled=True 时，with autocast(): 包裹的代码块内，PyTorch 会自动为不同算子选择匹配的数值精度：
                                卷积、矩阵乘、线性层等计算密集型算子，自动用 float16（半精度）执行，显著提速、节省显存；
                                求和、对数、指数、损失计算等数值敏感型算子，自动保留 float32（单精度），避免精度丢失和数值溢出。
                            它只负责前向传播的精度自动切换，梯度防溢出由配套的 GradScaler 完成，二者搭配构成完整的混合精度训练方案。
                    3. with autocast(enabled=False) 的作用是：在当前代码块内强制禁用自动混合精度，所有计算严格使用张量原本的精度（绝大多数场景为 float32），不做任何自动降精度处理。
                '''
                # a. 计算 L1 误差总和
                error = torch.abs(pred - gt).sum()

                # b. 计算当前尺度的平均 _loss，公式为 _loss = error / (有效像素数 * 尺度数) * loss_weight
                # len(gt) 是有效像素数
                # len(depth_preds) 是尺度数
                # max(1.0, ...) 防止除以 0
                # 再乘 loss_weight
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )

            # (6) 累加到总 loss
            loss = loss + _loss

        # 返回深度 loss
        return loss


# 三、AsymmetricFFN类，它是 SparseDrive decoder 中使用的前馈网络，用于处理多相机、多尺度、多 key point 的融合特征
# 将 AsymmetricFFN 注册到 FEEDFORWARD_NETWORK，在 detection3d_head.py 的 ffn=dict(type="AsymmetricFFN", ...) 会构建它
@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    # AsymmetricFFN 是 SparseDrive decoder 中使用的前馈网络
    # 它与标准 Transformer FFN 类似：Linear -> ReLU -> Dropout -> Linear -> Dropout -> Residual
    # “Asymmetric” 主要体现在：
    # 1. 输入通道 in_channels 可以不等于输出 embed_dims
    # 2. 可用 identity_fc 将 residual 分支投影到 embed_dims
    # 这对 residual_mode="cat" 特别重要，因为 cat 后通道会变成 2C

    # 1. 
    def __init__(
        self,                                    # self 表示当前模块对象
        in_channels=None,                        # 输入通道数。若 None，则默认等于 embed_dims
        pre_norm=None,                           # 前置归一化配置，如 pre_norm=dict(type="LN")
        embed_dims=256,                          # 输出 embedding 维度
        feedforward_channels=1024,               # FFN 中间层通道数
        num_fcs=2,                               # FC 层数量，至少为 2
        act_cfg=dict(type="ReLU", inplace=True), # 激活函数配置
        ffn_drop=0.0,                            # FFN 内部 dropout 概率
        dropout_layer=None,                      # 额外 dropout layer 配置
        add_identity=True,                       # 是否添加 residual identity
        init_cfg=None,                           # MMCV 初始化配置
        **kwargs,                                # 接收额外参数
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)                                 # 调用 BaseModule 初始化
        assert num_fcs >= 2, ("num_fcs should be no less " f"than 2. got {num_fcs}.") # 要求 FC 层数至少为 2

        # 1. 保存参数：
        self.in_channels = in_channels                   # 保存输入通道数
        self.pre_norm = pre_norm                         # 保存 pre_norm 配置或模块
        self.embed_dims = embed_dims                     # 保存输出 embedding 维度
        self.feedforward_channels = feedforward_channels # 保存中间层通道数
        self.num_fcs = num_fcs                           # 保存 FC 层数量
        self.act_cfg = act_cfg                           # 保存激活函数配置
        self.activate = build_activation_layer(act_cfg)  # 根据 act_cfg 构建激活函数


        layers = [] # 创建 FFN 层列表
        # 2. 构建 FFN 主分支：
        # (1) 若没有指定输入通道，默认输入通道等于 embed_dims
        if in_channels is None:
            in_channels = embed_dims # 默认输入通道等于 embed_dims

        # (2) 若配置了前置归一化，则构建归一化层
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1] # 构建归一化层：build_norm_layer 返回 (name, layer)，这里取 [1] 得到 layer 本身

        # (3) 构建前 num_fcs - 1 个 Linear + activation + dropout 块
        for _ in range(num_fcs - 1):
            # 添加一个 Sequential 块
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), # 线性层：in_channels -> feedforward_channels
                    self.activate,                             # 激活函数，例如 ReLU
                    nn.Dropout(ffn_drop),                      # dropout
                )
            )

            # 后续层输入通道变成 feedforward_channels
            in_channels = feedforward_channels

        # (4) 添加最后一个线性层：从 feedforward_channels 映射回 embed_dims
        layers.append(Linear(feedforward_channels, embed_dims))
        # (5) 添加输出 dropout
        layers.append(nn.Dropout(ffn_drop))
        # (6) 将所有层封装成 Sequential
        self.layers = Sequential(*layers)


        # 3. 构建额外 dropout layer：若 dropout_layer 配置存在，就用 build_dropout，否则使用 Identity
        self.dropout_layer = (build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity())


        # 4. 构建 identity 分支映射：
        self.add_identity = add_identity # 保存是否使用 residual
        if self.add_identity:            # 若使用 residual，则构建 identity 分支映射：
            # 若当前 in_channels 等于 embed_dims，就不需要额外投影、直接使用 nn.Identity()，
            # 否则需要用 Linear() 把 identity 从 self.in_channels 映射到 embed_dims【注意这里使用 self.in_channels，而不是循环后的 in_channels】
            self.identity_fc = (torch.nn.Identity() if in_channels == embed_dims else Linear(self.in_channels, embed_dims)) 


    # 2. AsymmetricFFN 前向传播：
    def forward(self, x, identity=None): 
        '''
            x shape 通常是 [B, num_anchor, C]，
            identity 是可选 residual 输入
        '''
        # (1) 若有 pre_norm，则先对 x 做归一化
        if self.pre_norm is not None:
            x = self.pre_norm(x) # 先对 x 做归一化

        # (2) 经过 FFN 主分支
        out = self.layers(x)

        # (3) 若不使用 residual，则直接返回 dropout 后的主分支输出
        if not self.add_identity:
            return self.dropout_layer(out) # 只返回 dropout 后的主分支输出

        # (4) 若没有显式传入 identity，则默认使用 x 作为 residual
        if identity is None:
            identity = x # 默认使用 x 作为 residual

        # (5) 将 identity 映射到 embed_dims【当输入通道和输出通道不一致时，这一步很关键】
        identity = self.identity_fc(identity)

        # (6) residual 相加
        return identity + self.dropout_layer(out)
    
