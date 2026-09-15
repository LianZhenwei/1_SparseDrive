import torch                                       # 导入 PyTorch，主要用于 Tensor 操作，例如 topk、cat、where、tile 等
from torch import nn                               # 从 torch 中导入 nn 模块，InstanceBank 继承自 nn.Module，同时 anchor 和 instance_feature 会被定义成 nn.Parameter
import torch.nn.functional as F                    # 导入 torch.nn.functional，后面使用 F.pad 对 instance_id 进行补齐
import numpy as np                                 # 导入 numpy，主要用于读取 .npy anchor 文件，以及处理坐标变换矩阵
from mmcv.utils import build_from_cfg              # 从 MMCV 中导入 build_from_cfg，用于根据配置字典和 registry 构建 anchor_handler
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS # 导入 MMCV 的插件层注册表，InstanceBank 自己会注册到 PLUGIN_LAYERS 中，anchor_handler 也会从 PLUGIN_LAYERS 中构建
'''
    这个文件的 InstanceBank 不是“存放所有检测结果的数据库”，而是一个跨帧的 sparse instance 状态管理器。它管理三类东西：
        A. 初始 instance templates：learnable anchor + instance_feature；
        B. temporal memory：上一帧输出中 top-k 的 feature / anchor / confidence；
        C. tracking memory：与 top-k temporal instances 对齐的 instance_id。

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 论文与代码的对应关系                                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ SparseDrive 图3：instance memory queue 用于 temporal modeling。          │
    │ SparseDrive §3.2 / 图4：检测实例由 instance feature + anchor box 表示；  │
    │   temporal decoder 同时接收 current instances 与 historical instances。 │
    │ Sparse4D v2 §3.1：单帧层先处理 newly emerging objects；随后把历史输出   │
    │   和当前高分实例共同送入 multi-frame layers；每层 anchor 总数保持不变。  │
    │ Sparse4D v2 §3.2：instance = anchor + instance_feature + anchor_embed；│
    │   跨帧传播时，只投影 anchor，并重新编码 anchor_embed；feature 可保持。  │
    │ Sparse4D v3 §3.4 / Algorithm 1：超过阈值的实例分配 ID；top-k 同时完成  │
    │   temporal instance 的生命周期管理，无需额外 data association。          │
    └─────────────────────────────────────────────────────────────────────────┘

    本项目 detection 的标准配置（见 sparsedrive_small_stage1.py / stage2.py）：
        B      : batch size（Stage1 每卡通常 8；Stage2 每卡通常 6）
        N=900  : 检测总 instance / anchor 数 self.num_anchor
        T=600  : temporal memory 数 self.num_temp_instances
        M=300  : 当前帧补充实例数，M = N - T
        C=256  : instance feature 通道数 self.embed_dims
        D=11   : detection anchor 的未解码 box-state 维度
                    [x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz]
                    注：D=11 由官方 core/box3d.py 的 X...VZ 索引定义决定。

    因此 detection 的核心 shape 在大多数时刻为：
        learnable instance_feature : [N=900, C=256]
        learnable anchor           : [N=900, D=11]
        current instance_feature   : [B, 900, 256]
        current anchor             : [B, 900, 11]
        cached_feature             : [B, T=600, 256]
        cached_anchor              : [B, T=600, 11]
        cached confidence          : [B, T=600]
        cached instance_id         : [B, N=900]，前 T 位有效，其余 N-T 位为 -1

    地图分支复用同一类，但含义不同：
        N_map=100；anchor 是 20 个二维采样点组成的 polyline，flatten 后 D_map=40；
        Stage1: T_map=0（不真正缓存 map temporal instances）；
        Stage2: T_map=33（会缓存 33 个历史 map instances）。

    =============================================================================
    【InstanceBank 的双任务阅读地图：Detection 与 Map 共用同一类，但各有独立对象】
    =============================================================================
    SparseDriveHead 会分别为 det_head 与 map_head 构建一套独立的 InstanceBank。
    它们不共享 self.anchor、cached_feature、cached_anchor、confidence 或 instance_id。

    ┌───────────────┬─────────────────────────────┬──────────────────────────────┐
    │               │ Detection InstanceBank      │ Map InstanceBank             │
    ├───────────────┼─────────────────────────────┼──────────────────────────────┤
    │ 稀疏 instance │ 一个动态 3D object box       │ 一条局部 vectorized map line  │
    │ num_anchor    │ N_det = 900                 │ N_map = 100                  │
    │ anchor 语义   │ 3D box state，D_det = 11     │ 20 个二维点，D_map = 20×2=40 │
    │ anchor shape  │ [B,900,11]                  │ [B,100,40]                   │
    │ feature shape │ [B,900,256]                 │ [B,100,256]                  │
    │ 分类 logits   │ [B,900,10]                  │ [B,100,3]                    │
    │ feat_grad     │ False：初始 feature 不学习   │ True：初始 map query 可学习   │
    │ Stage1 memory │ T_det = 600                  │ T_map = 0，实际不缓存         │
    │ Stage2 memory │ T_det = 600                  │ T_map = 33                    │
    │ tracking ID   │ 使用，with_instance_id=True  │ 不使用，with_instance_id=False│
    └───────────────┴─────────────────────────────┴──────────────────────────────┘

    map anchor 的存储形式：K-Means 文件通常是 [100,20,2]，即 100 条折线、每条 20 个 (x,y) 点；
    本类会将其 flatten 成 [100,40]。这只是存储格式改变，SparsePoint3DEncoder /
    SparsePoint3DRefinementModule / SparsePoint3DKeyPointsGenerator 会把最后 40 维重新按 20 个二维点理解。

    Map 时序的几何含义：历史地图线本身是静态环境元素，但自车坐标系在运动。
    因此 Stage2 仍要把历史 polyline 从“历史局部坐标系”旋转、平移到“当前局部坐标系”；
    与 detection 不同，map 不做速度外推，map 的 anchor_projection 也不会使用 time_intervals。
    =============================================================================
'''

# =============================================================================
# 【Map 任务补充总览：本类在 map head 中到底管理什么？】
#
# 1. map 也会单独构建一个 InstanceBank；它与 detection InstanceBank 是两个对象。
#    因此 det / map 的 self.anchor、instance_feature、cached_feature、cached_anchor、
#    confidence、instance_id 都完全独立，不能把 det 的 900 个 object slot 理解成 map 的输入。
#
# 2. map 的一个 instance = 一条 vectorized local map polyline，而不是一个 3D object box：
#       map k-means 原始 anchor : [N_map=100, num_sample=20, 2]
#       InstanceBank 内的 anchor : [100, 40]，40 = 20 × (x,y)
#       batch 后的 anchor        : [B, 100, 40]
#       batch 后的 feature       : [B, 100, 256]
#       分类 logits              : [B, 100, 3]，对应 ped_crossing / divider / boundary。
#
# 3. Stage1 与 Stage2 的核心差别：
#       Stage1：T_map=0；cache() 立即 return，因此每帧都只用当前 100 条 map query。
#       Stage2：T_map=33；cache() 保存 top-33 map line，下一帧 update() 组合为
#               “历史 33 条线 + 当前 top-67 条线 = 100 条线”。
#
# 4. map 的 anchor 是局部 ego/lidar 坐标系下的 polyline。
#    地图在 global 坐标系中近似静态，但 ego 在移动；所以 Stage2 必须把每个历史点用
#    T_temp2cur = T_global_inv(cur) @ T_global(temp) 变换到当前局部坐标系。
#    SparsePoint3DKeyPointsGenerator.anchor_projection() 只做 2D 的 R_xy @ p + t_xy；
#    它不读取 time_intervals，因此 map 不做 detection box 的 velocity 外推。
#
# 5. map_head 配置 with_instance_id=False。
#    所以本类的 get_instance_id() / update_instance_id() 仅服务 detection tracking；
#    map 有 temporal memory，不代表 map line 会获得/继承 tracking ID。
# =============================================================================

# 控制 from instance_bank import * 时暴露哪些对象
__all__ = ["InstanceBank"] # 当前文件主要暴露 InstanceBank


# 工具函数 topk() 的作用是：根据 confidence 选出每个 batch 中置信度最高的 k 个 instance，然后同步从 inputs 中取出对应的 feature、anchor、instance_id 等
'''
    confidence : [B, N]
    inputs[i]  : 通常是 [B, N, C]、[B, N, D] 或 [B, N]
    输出：
        selected confidence : [B, k]
        selected inputs[i]  : 统一为 [B, k, tail_dim]；即使输入是 [B, N] 的 instance_id，也会暂时成为 [B, k, 1]。
'''
# 【Map 任务中的 top-k】
# - map 的 confidence 在调用本函数前，会从 [B,100,3] 沿类别维取 max，得到 [B,100]；该分数代表“一条 20 点 polyline 属于任一 map 类别的最高 logit”。
# - topk 同步选取的 anchor 是 [B,100,40]；因此它选中/淘汰的是“整条地图线 query”，不是在一条线的 20 个点中再做 top-k。
# - Stage2 cache(): k=33；Stage2 update(): 从当前 100 条线选 k=67 用于补齐历史 33 条线。
def topk(confidence, k, *inputs): # 输入与输出示例：[B, N] → [B, k]
    # (0) confidence shape 通常是 [B, N]，因此 B 是 batch_size，N 是 instance 数量
    bs, N = confidence.shape[:2]

    # (1) 对每个 batch 内的 N 个 instance 按 confidence 取 top-k
    # confidence 输入输出前后 shape 变化：[B, N] -> [B, k]
    # indices 输出 shape 是 [B, k]，表示每个 batch 内 top-k 的索引 index（即每个 index 属于各自 batch 内的 [0, N)）
    confidence, indices = torch.topk(confidence, k, dim=1)

    # (2) 将 batch 内局部索引转换成 flatten 后的全局索引
    # 原本 indices 是每个 batch 内部的索引范围 [0, N)，当把 [B, N, ...] flatten 成 [B*N, ...] 后，则第 b 个 batch 的第 i 个元素全局索引是 b*N + i
    # 用 b*N 偏移后，才能从 input.flatten(0,1) 的 [B*N, ...] 中正确抽取每个 batch 的 top-k
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(-1) # 【shape】[B, k] 的 batch-local indices -> [B*k] 的 flattened global indices

    # (3) 用于保存同步筛选后的 outputs
    outputs = []

    # (4) 遍历所有输入张量，输入张量 inputs 可以是 instance_feature、anchor、instance_id 等
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
        '''
            先把 input 的 batch 维和 instance 维展平，则 input shape: [B, N, C] -> [B*N, C]
            然后用全局 indices 取 top-k 对应的数据
            最后 reshape 回 [B, k, -1]        
        '''

    # (5) 返回 top-k confidence，以及同步筛选后的 outputs【outputs 是一个 list】
    return confidence, outputs


'''
    InstanceBank类：它是 SparseDrive 中管理稀疏 instance 的模块
    
    【重要】它不是 detection 专属“检测框库”。对 detection，它管理 3D box instances；对 map，它管理 vectorized polyline instances。两类任务复用完全相同的 get → update → cache 接口。

    【一句话定义】InstanceBank = “当前帧的初始化 query 工厂 + 上一帧的稀疏 temporal memory + tracking ID 延续器”。

    它不是 decoder 本身：不做 attention、不从图像采样特征、不直接输出 box；
    它为 detection3d_head.py 的 decoder 准备/更新/缓存 instance 状态。
    
    【在外层 forward 的真实调用顺序】
        1) head.forward() 开始：InstanceBank.get()
                -> 返回当前 N=900 个 template，以及上一帧 T=600 个 temporal instances。
        2) 第一个单帧 decoder refine 后：InstanceBank.update()
                -> [历史600] + [当前 top300]，凑回 N=900，供 temporal decoder 使用。
        3) 最后一层 decoder 后：InstanceBank.cache()
                -> 当前 N=900 中选 top600，留给下一帧。
        4) 推理/评估 tracking 时：InstanceBank.get_instance_id()
                -> 对高分实例发新 ID，并保存 top600 对应 ID。
    
    【论文对照】
        - SparseDrive 图3明确画出 instance memory queue。
        - SparseDrive §3.2 / 图4 说明 temporal decoder 的输入来自当前帧和历史帧。
        - Sparse4D v2 §3.2 提供了 InstanceBank 背后的关键原则：跨帧只变换 anchor，instance feature 不需要因 ego motion 而重算。
'''
# 【Map 任务的同一套调用链】
# map_head 也调用 get() -> update() -> cache()，但每一步的含义如下：
#   (1) get()：
#       Stage1 -> 返回当前 [B,100,256] / [B,100,40]，temp_* 为 None；
#       Stage2 -> 还会返回已投影到当前坐标系的历史 [B,33,256] / [B,33,40]。
#   (2) update()：
#       Stage1 -> cached_feature=None，直接保留当前 100 条线；
#       Stage2 -> 前 33 slots 是历史 polyline，后 67 slots 是当前帧最高分 polyline。
#   (3) cache()：
#       Stage1 -> T=0，直接 return；
#       Stage2 -> 对 [B,100,3] map logits 排序，缓存 top-33 feature/anchor/confidence。
#   (4) map 不调用 get_instance_id()，因为 with_instance_id=False。

# 将 InstanceBank 注册到 MMCV 的 PLUGIN_LAYERS 注册表中，这样配置文件里写 type="InstanceBank" 时，就能自动构建该模块
@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    '''
        InstanceBank 是 SparseDrive 中管理稀疏 instance 的模块
        它的核心作用：
            1. 保存 learnable anchor
            2. 保存 learnable instance feature
            3. 缓存上一帧高置信度 instance
            4. 将历史 anchor 投影到当前帧坐标系
            5. 为 temporal GNN 提供 temp_instance_feature 和 temp_anchor
            6. 管理 instance_id，用于跟踪任务    
    '''
    # -------------------------------------------------------------------------
    # 【Map InstanceBank 的长生命周期状态】
    # self.anchor           : [100,40]，一行 = 20 个 (x,y) 采样点 flatten 后的一条 polyline。
    # self.instance_feature : [100,256]，map 配置 feat_grad=True，因此它是可学习的 query embedding。
    # self.cached_anchor    : Stage2 为 [B,33,40]；保存的仍是局部坐标系折线，而不是 global HD map。
    # self.cached_feature   : Stage2 为 [B,33,256]；将与历史 anchor 对齐地进入 temporal map decoder。
    # self.confidence       : Stage2 为 [B,33]；只用于 top-k temporal memory 生命周期，不表示 map tracking ID。
    # -------------------------------------------------------------------------
    # 1. 初始化：“创建源自 .npy 的 self.anchor” “创建全0的 self.instance_feature” “创建值为 None 的 self.cached_anchor 和 self.cached_feature”
    def __init__(
        self,                      # self 表示当前 InstanceBank 对象
        num_anchor,                # anchor 数量。检测任务中例如 num_anchor=900，地图任务中例如 num_anchor=100
        embed_dims,                # instance feature 的维度，通常是 256
        anchor,                    # anchor 初始化来源，可以是 .npy 文件路径、也可以是 list/tuple。检测任务一般是 data/kmeans/kmeans_det_900.npy，地图任务一般是 data/kmeans/kmeans_map_100.npy
        anchor_handler=None,       # anchor_handler 配置，用于做历史 anchor 到当前帧坐标系的投影。detection 用 3D box projection，map 用 polyline point-wise projection。例如 SparseBox3DKeyPointsGenerator 或 SparsePoint3DKeyPointsGenerator
        num_temp_instances=0,      # 需要缓存的历史 temporal instance 数量。检测任务中可能是 600，地图任务中可能是 33，若 <=0，表示不使用 temporal cache
        default_time_interval=0.5, # 默认时间间隔，nuScenes 关键帧间隔通常约 0.5s
        confidence_decay=0.6,      # 历史置信度衰减系数，用于让上一帧缓存的 confidence 随时间衰减
        anchor_grad=True,          # anchor 是否参与梯度更新，True 表示 anchor 是可学习参数
        feat_grad=True,            # instance feature 是否参与梯度更新，True 表示 instance_feature 是可学习 query embedding
        max_time_interval=2,       # 最大允许时间间隔，若当前帧和缓存帧时间差超过该值，则认为缓存无效
    ):
        '''lzw: 
            【本函数创建的 self.anchor 和 self.instance_feature 不是“某一帧的预测结果” 这种瞬时状态，而是长生命周期状态】
                self.anchor          : 可学习的默认几何 proposal 模版，训练时是模型参数，源自 .npy 文件。detection 的初始 anchor [num_anchor=900, 11]，map 的初始 anchor [num_anchor=100, 40]
                self.instance_feature: 默认语义 query 模版，初始化为全0。detection 任务的初始 feature [900, 256]，且不可学习；map 任务的初始 feature [100, 256]，可学习。

            【论文对照】
                Sparse4D v3 §3.1：一组 anchor 用 k-means 初始化并作为 learnable parameters；
                SparseDrive §3.2：周围 agents 由一组 instance features 与 anchor boxes 表示。
        '''
        super(InstanceBank, self).__init__() # 调用 nn.Module 初始化

        # 1. 保存参数：
        # (1) 保存embed_dims、num_temp_instances、default_time_interval、confidence_decay、max_time_interval
        self.embed_dims = embed_dims                       # 保存 instance feature 维度，通常是 256，记 C=self.embed_dims=256
        self.num_temp_instances = num_temp_instances       # 保存需要缓存的历史 temporal instance 数量。它不是“历史帧数”，而是“从上一帧留下多少个稀疏实例”，即 top-k 的 k # 【shape】T=self.num_temp_instances；detection 配置 T=600
        self.default_time_interval = default_time_interval # 保存默认时间间隔，形状为 [B]，默认值为 0.5s 以对应 nuScenes 2Hz keyframe
        self.confidence_decay = confidence_decay           # 保存置信度衰减系数
        self.max_time_interval = max_time_interval         # 保存最大时间间隔

        # (2) 若配置了 anchor_handler，则根据配置从 PLUGIN_LAYERS 注册表中构建 anchor_handler
        if anchor_handler is not None:
            # 【论文对照】Sparse4D v2 §3.2：不同任务可定义不同 anchor 和 projection function：detection 用 3D box projection，map 用 polyline point-wise projection
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS) # 根据配置从 PLUGIN_LAYERS 注册表中构建 anchor_handler
            assert hasattr(anchor_handler, "anchor_projection")            # 要求 anchor_handler 必须有 anchor_projection() 方法，因为后面 get() 中要把历史 anchor 投影到当前帧坐标系
        self.anchor_handler = anchor_handler # 保存 anchor_handler                            


        # 2. 创建初始 anchor【detection 的初始 anchor [num_anchor=900, 11]，map 的初始 anchor [num_anchor=100, 40]】：
        # (1) 依据 anchor 初始化来源（可以是 .npy 文件路径、也可以是 list/tuple）来构建 anchor：
        if isinstance(anchor, str):             # 若 anchor 是字符串
            anchor = np.load(anchor)            # 认为它是 .npy 文件路径，直接读取【检测任务是 data/kmeans/kmeans_det_900.npy，其anchor形状为[num_anchor=900, 11]；地图任务是 data/kmeans/kmeans_map_100.npy，其anchor形状为[num_anchor=100, 20, 2]】
        elif isinstance(anchor, (list, tuple)): # 若 anchor 是 list 或 tuple
            anchor = np.array(anchor)           # 转成 numpy array

        # (2) 统一 map 任务的 anchor 的形状：若 anchor 是 3 维（这表示这是 map anchor，如 [num_anchor=100, num_sample=20, point_dim=2]），则将其拉平成 [num_anchor=100, num_sample * point_dim=40]
        if len(anchor.shape) == 3:                       # for map
            anchor = anchor.reshape(anchor.shape[0], -1) # 将 map anchor 拉平成 [num_anchor=100, num_sample * point_dim=40]
            '''
                【shape】map anchor: [N_map=100, 20, 2] -> [N_map=100, 40]；“flatten 只是存储格式变化”，SparsePoint3DEncoder / refine layer 会再按 20 个点理解它。
                【Map 补充】这 40 维不表示 40 个独立的 map query：
                        SparsePoint3DEncoder(anchor) 将 [B,100,40] 编码为 anchor_embed；
                        SparsePoint3DRefinementModule 输出同为 40 维的残差并加回 anchor；
                        SparsePoint3DKeyPointsGenerator 会把它 view 回 [B,100,20,2]，再围绕每个二维点生成带多个固定高度的 3D image-sampling keypoints。
            '''

        # (3) 记录实际使用的 anchor 数量：若 .npy 文件中 anchor 数 len(anchor) 超过传参 num_anchor，就截断
        self.num_anchor = min(len(anchor), num_anchor) # detection 任务的 len(anchor) 为 900，map 任务的 len(anchor) 为 100

        # (4) 只取前 num_anchor 个 anchor
        anchor = anchor[:num_anchor]

        # (5) 【核心】将 anchor 注册成可学习参数
        # 最终初始化检测任务的 self.anchor：[num_anchor=900, box_dim=11]
        # 最终初始化地图任务的 self.anchor：[num_anchor=100, 2*num_sample = 2*20 = 40]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32), # 将 numpy anchor 转成 float32 tensor
            requires_grad=anchor_grad,                 # 是否允许 anchor 参与训练更新
        )

        # (6) 保存一份 anchor 初始化值，init_weight() 里会用它重置 anchor
        self.anchor_init = anchor


        # 3. 创建初始 learnable instance feature【shape [N=num_anchor, C=embed_dims=256]，即每个 anchor 对应一个可学习 query feature，初始化为全 0【detection 任务 [900, 256] 且不可学习，map 任务 [100, 256] 且可学习】
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]), # 初始化为全 0
            requires_grad=feat_grad,                              # 是否允许 instance feature 参与训练更新
        )
        '''lzw：
                配置文件 sparsedrive_small_stage1_lzw.py 和 sparsedrive_small_stage2_lzw.py 里均设置了：
                    设置了 detection 任务的 feat_grad=False，因此它保持 0（即instance_feature 不参与梯度，常用于历史缓存稳定处理）
                    设置了 map 任务的 feat_grad=True，因此 map query 可以被直接学习（即map instance feature 参与梯度更新）
                
                第一个 decoder 的 deformable aggregation 会基于 anchor 从图像提取特征，随后才形成有语义的 instance_feature。
                
                【map 的 feat_grad=True 与 detection 的 feat_grad=False 是刻意的任务差异】：
                        map 的初始 [100,256] query 会在训练中学习；
                        而 detection 的初始 feature 保持为不可学习的全 0 模板，后续主要依靠图像聚合 deformable aggregation 和时序缓存 temporal cache 形成语义。
        '''

        # 4. 初始化缓存状态：初始化 self.cached_anchor、self.cached_feature 等全为None
        self.reset()

    # 2. 初始化 InstanceBank 的权重【恢复 KMeans anchor 初值；可训练 feature 做 Xavier 初始化】，这个函数 init_weight() 会被 Sparse4DHead.init_weights() 间接调用
    def init_weight(self):
        '''
            【作用】每次新建模型时恢复 K-means anchor 初值；若 feature 可训练，再初始化 query embedding。
            【注意】这只在模型初始化时调用；它不是每帧调用，因此不会覆盖 runtime cached_*。        
        '''
        # (1) 将 anchor 重置为最初加载的 anchor_init
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)

        # (2) 若 instance_feature 是可学习的，则使用 Xavier uniform 初始化 instance_feature
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)  # 使用 Xavier uniform 初始化 instance_feature

    # 3. 清空所有 temporal cache【即清空上一序列的 temporal memory 与 ID】，在 batch size 不一致、序列中断、时间间隔过大等情况下会调用 reset()：
    # 【作用】开始一条新 scene 或 首帧 或 无历史帧时，把“上一个序列”的记忆彻底丢弃。这防止 scene A 的车辆实例被错误当作 scene B 的历史实例。
    # 【shape】所有 cached_* 置 None；prev_id 重置为 0，因此 instance_id 仅保证单个连续序列内唯一。
    def reset(self):                # 关键输入输出：所有 cache → None
        self.cached_feature = None  # 缓存的历史 instance feature
        self.cached_anchor = None   # 缓存的历史 anchor
        self.metas = None           # 缓存帧对应的元信息
        self.mask = None            # 当前帧与历史帧是否有效匹配的 mask
        self.confidence = None      # 缓存的历史 instance confidence
        self.temp_confidence = None # 当前帧临时 confidence
        self.instance_id = None     # 缓存的 instance id
        self.prev_id = 0            # 下一个新 instance id 的起始编号

    # 4. 为当前帧 decoder 准备 “新生实例 + 历史实例” ：为当前帧准备一套新的当前帧模板（instance_feature 和 anchor），同时读取上一帧缓存的 cached_feature 和 cached_anchor、再将历史 anchor 从历史 ego 坐标系投影到当前 ego 坐标系
    def get(self, batch_size, metas=None, dn_metas=None):
        '''
            get() 在 Sparse4DHead.forward() 一开始被调用
            实现逻辑：
                1. 复制 learnable instance_feature 和 anchor 到 batch 维
                2. 若有历史缓存，则返回 temp_instance_feature 和 temp_anchor
                3. 若有 anchor_handler，则把历史 anchor 投影到当前帧坐标系
                4. 计算当前帧和历史帧的 time_interval   

            【论文对照】
                - SparseDrive §3.2：non-temporal decoder 的输入是新初始化实例；temporal decoder 的输入同时来自当前和历史帧。
                - Sparse4D v2 §3.2：将上一帧 anchor 投影到当前帧；instance feature 可以保持不变；
                投影后的 anchor 会重新经 anchor encoder 变成新的 positional embedding。 
        '''

        # 1. 复制 learnable instance_feature 和 anchor 到 batch 维
        # (1) 将 learnable instance_feature 复制到 batch 维：
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1)) # 注意，在 __init__() 函数中 self.instance_feature 初始化为全0
        '''
            self.instance_feature shape: [N, C]
            instance_feature shape: [B, N, C]
            【shape】[N, C] -> [1, N, C] -> [B, N, C]，这 N 个是“本帧从零开始可用于发现新物体”的 slots，不等同于 temporal cache。
            【shape】detection 任务: [900, 256] -> [B, 900, 256]
            【shape】map 任务: [100, 256] -> [B, 100, 256]
        '''
        # (2) 将 learnable anchor 复制到 batch 维：
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        '''
            self.anchor shape: [N, D]
            anchor shape: [B, N, D]
            【shape】[N, D] -> [B, N, D]，每个 batch 样本起始时拿同一套 K-means anchor 模板，而真正的场景差异由 decoder refine 产生。
            【shape】detection 任务: [900, 11] -> [B, 900, 11]
            【shape】map 任务: [100, 40] -> [B, 100, 40]
        '''

        # 2. 【核心】将 cached_anchor 和 DN anchor 从上一帧坐标系投影到当前帧坐标系：
        if (self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]): # 【Step2：第2帧及之后走该分支】若存在历史 cached_anchor【有 temporal cache 才能走 temporal 路径】且 batch size 和当前 batch size 一致【强制 B 一致，避免分布式/数据尾批错位】，则将 cached_anchor 和 DN anchor 从历史帧坐标系投影到当前帧坐标系
            '''
                【Shape】这里 cached_anchor 的形状为: [B, T, D]
                【Detection】detection 任务中 cached_anchor 为 [B, T=600, 11]，所以会进入此分支
                【Map】Stage1 的 T_map=0，cache() 从不写入 cached_anchor，因此每帧都走下方 reset/default-interval 分支。
                【Map】Stage2 的 cached_anchor 为 [B, T=33, 40]，所以会进入此分支；
            '''
            # (1) 计算时间间隔 time_interval = 当前帧时间戳 - 上一帧时间戳：
            history_time = self.metas["timestamp"]                         # 取出上一帧的 timestamp
            time_interval = metas["timestamp"] - history_time              # 当前帧时间戳 - 上一帧时间戳 #【shape】timestamp / history_time: [B]，因此 time_interval: [B]，单位通常是秒。           
            time_interval = time_interval.to(dtype=instance_feature.dtype) # 转成与 instance_feature 相同的数据类型

            # (2) 构建上一帧缓存是否有效的 self.mask【若时间间隔 time_interval 超出最大允许范围 max_time_interval，则对应位置的 self.mask 为 False，表示该上一帧缓存无效】：
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            '''
                self.mask shape: [B]；每个 batch 样本独立决定能否复用历史记忆：
                    True 表示该 batch 样本可以使用历史缓存
                    False 表示历史缓存无效
                当前 config max_time_interval=2s：掉帧过多/序列断开时，宁可不用历史，也不注入错误历史。            
            '''

            # (3) 若存在 anchor_handler，则将 cached_anchor 从历史帧坐标系投影到当前帧坐标系：
            if self.anchor_handler is not None:
                # (3.a) 构造从历史帧坐标系到当前帧坐标系的变换矩阵 T_temp2cur：
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                '''
                    self.metas["img_metas"][i]["T_global"]：历史帧 ego/lidar 到 global 的变换
                    x["T_global_inv"]：当前帧 global 到当前 ego/lidar 的变换
                    所以：T_temp2cur = T_cur_global_inv @ T_temp_global 表示把历史帧坐标下的点变换到当前帧坐标下
                    
                    【论文对照】Sparse4D v2 §3.2, Eq.(2)：利用 ego motion 将 t-1 时刻 3D box anchor 投影至 t 时刻坐标系。此处实现同一件事：T_global_inv(cur) @ T_global(temp)。
                    【shape】T_temp2cur: [B,4,4]；每个 batch 各有一个 SE(3) 齐次变换。
                '''

                # (3.b) 将 cached_anchor 从历史帧坐标系投影到当前帧坐标系：
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,              # 历史缓存 anchor
                    [T_temp2cur],                    # 坐标变换矩阵列表
                    time_intervals=[-time_interval], # 时间间隔取负号。因为这里是把历史 anchor 投影到当前时刻，具体正负取决于 anchor_projection 内部定义
                )[0]
                '''
                    【论文对照】Sparse4D v2 §3.2：feature 不动，anchor 依据 ego pose + 时间间隔变换；之后 detection3d_head.py 会重新调用 anchor_encoder(temp_anchor)，产生当前时刻的 geometry embedding。
                    【shape】输入/输出都是 [B,T,D]；detection 为 [B,600,11]。                
                '''
            '''
                【Map 的坐标语义】
                    历史 map anchor 里的 20 个 (x,y) 是“历史 ego/lidar 局部坐标”下的点。
                    虽然 divider / boundary 等世界元素是静态的，但车辆行驶后当前局部原点和朝向改变；
                    因而必须用 T_temp2cur 将每个历史点表达成当前 ego/lidar 坐标。
                    否则 temporal map decoder 读到的历史线会发生与自车运动相反的系统性几何错位。

                【Map anchor_projection 的精确 shape 与运算】
                    对 SparsePoint3DKeyPointsGenerator：
                        输入 anchor：[B, 33, 40]
                        reshape：[B, 33, 20, 2] -> [B, 660, 2]
                        逐点变换：p_cur = T_temp2cur[:2, :2] @ p_temp + T_temp2cur[:2, 3]
                        reshape 回：[B, 660, 2] -> [B, 33, 20, 2] -> [B, 33, 40]
                    统一接口虽传入 time_intervals=[-time_interval]，但 map 的 anchor_projection() 源码不会读取它。
                    因而 map 做的是“静态 polyline 的刚体坐标变换”，不是“带 velocity 的动态 box 外推”。
            '''

            # (4) 若同时满足以下3条件，则对 DN anchor 同样做历史到当前帧的坐标投影：
            if (
                self.anchor_handler is not None                  # 有 anchor_handler
                and dn_metas is not None                         # 有 dn_metas
                and batch_size == dn_metas["dn_anchor"].shape[0] # dn_metas 的 batch size 和当前 batch size 一致
            ):
                # (4.a) 获取 DN anchor 的 group 数量和每组 DN 数量。dn_anchor shape 可能是 [B, num_dn_group, num_dn, D]：
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]

                # (4.b) 对 DN anchor 同样做历史到当前帧的坐标投影
                # 【论文对照】Sparse4D v3 §3.1：temporal denoising 的 noisy anchors 也遵从同样的 temporal propagation。
                # 【注意】当前提供的官方 Stage1/Stage2 config 都是 num_dn_groups=0，因此正常训练不会进入该块。
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2), # 先把 group 维和 dn 维展平：[B, G, N_dn, D] -> [B, G*N_dn, D]
                    [T_temp2cur],                        # 坐标变换
                    time_intervals=[-time_interval],     # 时间间隔
                )[0]

                # (4.c) 投影后再 reshape 回 [B, G, N_dn, D]：
                dn_metas["dn_anchor"] = dn_anchor.reshape(batch_size, num_dn_group, num_dn, -1)

            # (5) 对 time_interval 做修正：若 time_interval != 0 且 mask=True，则使用真实时间间隔，否则使用 default_time_interval
            # 【作用】首帧/重复 timestamp/无效历史样本不用 0 或异常 Δt，而是回退到默认 0.5s。【shape】仍为 [B]。
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )

        # 【Step1：第1帧时走的是该else分支】若 没有历史缓存 或 batch size 不一致，则重置缓存、time_interval 使用默认值：
        else:
            self.reset()                                                                           # 重置缓存：self.cached_feature = None、self.cached_anchor = None 等
            time_interval = instance_feature.new_tensor([self.default_time_interval] * batch_size) # time_interval 使用默认值 # shape: [B]

        # 3. 返回当前帧 instance、anchor，以及历史缓存 feature、anchor、时间间隔 time_interval：
        return (
            instance_feature,    # 当前 decoder 的 instance feature 输入
            anchor,              # 当前 decoder 的 anchor 输入
            self.cached_feature, # 上一帧缓存的 top-k feature，可能为 None
            self.cached_anchor,  # 上一帧缓存、且已投影到当前坐标系的 anchor，可能为 None
            time_interval,       # 当前帧和缓存帧时间间隔
        )
        '''
        return (
            instance_feature,   # 当前 decoder 要处理的 instance feature。
                                # 初始时来自 InstanceBank 的基础 learnable/fixed feature；
                                # 在第一个单帧 decoder 后，可能变成“历史 instance + 当前高分 instance”的混合。
                                # shape: [B, N, C]

            anchor,             # 当前 decoder 要处理的 anchor，统一表达在“当前帧 ego/lidar 坐标系”下。
                                # 初始时来自 K-Means anchor；
                                # 在 temporal update 后，前一部分可能来自历史 anchor 投影到当前坐标系。
                                # detection: [B, N, D_box]，通常 D_box=11
                                # map:       [B, N, D_map]，通常 D_map=20*2=40

            temp_instance_feature, # 上一次缓存帧中 top-k 高置信度 instance 的 feature。
                                   # 实际来自 self.cached_feature。
                                   # 用作 temporal GNN / temporal cross-attention 的 Key 和 Value。
                                   # shape: [B, T, C]
                                   # detection 常见 T=600；Stage2 map 常见 T=33；
                                   # 第一帧、缓存失效、或未启用 temporal 时为 None。

            temp_anchor,        # 上一次缓存帧中 top-k instance 的 anchor，
                                # 但在 get() 返回前，已经通过 ego pose / 时间间隔投影到“当前帧坐标系”。
                                # 实际来自 self.cached_anchor。
                                # 用于生成 temporal Key 的位置编码 temp_anchor_embed。
                                # shape: [B, T, D_box] 或 [B, T, D_map]

            time_interval,      # 当前帧 timestamp - 历史缓存帧 timestamp。
                                # shape: [B]
                                # 通常约为 0.5 秒；若无历史缓存、缓存失效或时间差为 0，
                                # 代码会回退为 default_time_interval。
                                # 它既用于 anchor 的跨帧传播，也会传给 refine layer。
        )

            【Detection 任务的 get() 返回 shape】
                instance_feature      : [B, 900, 256]，当前帧的默认/新生 instance templates
                anchor                : [B, 900, 11]，当前帧默认/新生 anchors
                temp_instance_feature : None 或 [B, 600, 256]，上一帧已缓存的实例语义特征
                temp_anchor           : None 或 [B, 600, 11]，已投影到“当前坐标系”的历史 anchors
                time_interval         : [B]，用于 box 的 velocity / temporal compensation    

            【Map 任务的 get() 返回 shape】
                Stage1：
                    instance_feature      : [B, 100, 256]
                    anchor                : [B, 100, 40]
                    temp_instance_feature : None
                    temp_anchor           : None
                Stage2：
                    instance_feature      : [B, 100, 256]
                    anchor                : [B, 100, 40]
                    temp_instance_feature : None 或 [B, 33, 256]
                    temp_anchor           : None 或 [B, 33, 40]
                注意：time_interval 仍会统一返回 [B]，但 map projection 不会使用它做速度外推。
        '''

    # 5. 单帧 decoder 后，update() 混合历史实例与新实例【即单帧 decoder 后，update() 将历史 600 和当前 top-300 拼为 900，即把 decoder 的输入从 “纯当前模板” 切换成 “历史 + 当前新目标”】：
    def update(self, instance_feature, anchor, confidence):
        '''
            三个输入变量：
                instance_feature,  # 当前第一个单帧 decoder 输出后的 instance feature
                anchor,            # 当前第一个单帧 decoder refine 后的 anchor
                confidence,        # 当前第一个单帧 decoder 的分类 logits，代码里传入的是 cls

            返回：return instance_feature, anchor
                返回的不是“原来的当前帧结果”，而是：
                    历史 temporal instances
                    +
                    当前帧 top-k instances
                    +
                    可能附加的 DN instances
                组成的新 instance 集合。        
        '''
        '''
            update() 在 Sparse4DHead.forward() 中单帧 decoder 结束后调用
            
            【关键作用：把 decoder 的输入从“纯当前模板”切换成“历史 + 当前新目标”】
                SparseDrive config 中 num_single_frame_decoder=1：
                    第 1 个 decoder 先用当前 900 个 slots 从图像中发现新目标；
                    第 1 次 refine 后调用本函数；
                    之后的 temporal decoder 再处理 [历史600 + 当前top300]。
            
            【shape】输入/输出 normal instances 统一维持 [B,N=900,*]，只是前 T 个 slot 的来源变了。
            【作用】将历史 cached_feature/cached_anchor 和当前高置信度 instance 合并，构造后续 temporal decoder 使用的 instance_feature 和 anchor。        
        '''

        # 【Map 任务的 update() 位置布局】
        # Stage1: T_map=0 -> cached_feature 恒为 None -> 直接 return 当前 [B,100,256] / [B,100,40]。
        # Stage2: T_map=33 -> 后续会重组为：
        #         slots [0 : 33)   = 已投影、已缓存的历史 map line；
        #         slots [33 : 100) = 当前首个单帧 decoder 产生的 top-67 map line。
        # 这和 detection 的 [历史600 + 当前300] 代码路径完全相同，只是 instance 的几何含义和 T/N 数值不同。

        # (1) 【核心，对于第1帧，其 self.cached_feature=None，因此直接返回当前帧结果】若没有历史缓存或为首帧，则不能做“历史 + 当前”重组，则直接保留当前帧结果，即直接返回当前 instance_feature 和 anchor：
        if self.cached_feature is None:
            '''
                首帧没有历史信息：保持 [B,900,256] / [B,900,11]，后续 decoder 仍可运行。
                此时 detection3d_head.py 传入的 temporal key/value 为 None，注意力层不会读取历史实例；
                因而这里没有 temporal cross-attention 的跨帧信息，只保留当前帧自身的计算路径。            
            '''
            return instance_feature, anchor # 直接返回当前 instance_feature 和 anchor # 【shape】直接返回 [B, N, C] 和 [B, N, D]。

        # (2) 暂时拆分训练专用的 DN instances【因为 DN instances 不能参与 top-k，也不允许写入 temporal memory，所以需暂时拆开 DN instances；当普通 instances 重组完成后，DN 会被重新拼回最后】
        # 若存在 DN，则从 instance_feature 中拆出 dn_instance_feature、从 anchor 中拆出 dn_anchor，此时 instance_feature 为普通 learnable instance_feature、此时 anchor 为普通 anchor
        num_dn = 0 # DN query 数量初始化为 0
        '''
            a. 若存在 DN，则意味着输入 instance feature 为 [B,N+N_dn,C]、输入 anchor 为 [B,N+N_dn,D]，其中：
                    self.num_anchor 是 anchor 数量，即 self.num_anchor = N，对于 detection 任务的 N=900
                    instance_feature.shape[1] 是当前 instance 数量，此时 instance_feature.shape[1] = N+N_dn
                    num_dn 是 DN query 数量，即N_dn，由 num_dn = instance_feature.shape[1] - self.num_anchor 计算得到
            b. 然后从 instance_feature 中拆出 dn_instance_feature、从 anchor 中拆出 dn_anchor，
               此时 instance_feature 为普通 learnable instance_feature、此时 anchor 为普通 anchor
        '''
        if instance_feature.shape[1] > self.num_anchor:              # 若当前 instance 数量大于 self.num_anchor，说明后面拼接了 denoising query，则拆分训练专用的 DN instances
            num_dn = instance_feature.shape[1] - self.num_anchor     # DN query 数量 = 当前总 instance 数 - 普通 anchor 数
            dn_instance_feature = instance_feature[:, -num_dn:]      # 取出 DN instance feature
            dn_anchor = anchor[:, -num_dn:]                          # 取出 DN anchor
            instance_feature = instance_feature[:, :self.num_anchor] # 只保留普通 learnable instance
            anchor = anchor[:, : self.num_anchor]                    # 只保留普通 anchor
            confidence = confidence[:, :self.num_anchor]             # confidence 也只保留普通 anchor 部分

        # (3) 从当前帧普通 instances 中选择 top-k 新实例，补足历史 memory 以外的 slot。
        # (3.a) 保存 “当前帧需要新选入的 instance 数量” 为N
        N = self.num_anchor - self.num_temp_instances # detection 任务: N = 900 - 600 = 300；Stage2 map 任务: N = 100 - 33 = 67。
        # 【Map】N=67 不是“新建 67 条世界地图元素”，而是当前帧可重新从图像发现的 67 个 polyline slots。即便道路边界/车道线物理上静态，也可能刚进入 ROI、被遮挡后重新可见，或上一帧没有进入 top-33 cache。
        ''' 
            self.num_temp_instances 是历史缓存数量，对于 detection 任务就是600
            self.num_anchor 是总 anchor 数，对于 detection 任务就是900
            所以此时 N = 当前帧新 instance 数，即此时 N 表示当前帧需要新选入的 instance 数量
            【shape】detection 任务: N = N_total - T = 900 - 600 = 300。表示专门留下 300 个 slots 给 “当前帧新出现 / 上一帧未缓存” 的目标。 
            【shape】map 任务：N = N_total - T = 100 - 33 = 67
        '''

        # (3.b) 取每个 instance 在所有类别中的最大 logit【此处只需要排序，而 sigmoid() 与否不影响同一帧内的 arg-topk 顺序，因此只需对 logit 进行 max() 即可】
        confidence = confidence.max(dim=-1).values # 【shape】confidence 原始 shape 可能是 [B, N, num_cls]，这里 cls logits [B,N,num_cls=10] -> max logit [B,N]
        # 【Map】map logits 的实际 shape 为 [B,100,3] -> [B,100]；这是对一条 polyline 的“最高 map 类别 logit”排序。

        # (3.c) 【核心】在当前帧普通 instance 中选置信度最高的 N 个 instance 的 feature 和 anchor，即选出 (selected_feature, selected_anchor)
        _, (selected_feature, selected_anchor) = topk(
            confidence,       # 当前帧 confidence
            N,                # 需要选出的数量
            instance_feature, # 同步筛选 instance_feature 和 anchor
            anchor            # 同步筛选 instance_feature 和 anchor
        )

        # (4) 【核心】按固定 slot 布局重组：前 T 个 slot = 历史实例，后 N 个 slot = 当前 top-N 实例【这样后续 temporal GNN、confidence cache 与 tracking ID 都按同一顺序对齐的、都约定“temporal slots 位于前缀即前 T 个 slot”】
        # (4.a1) 将历史缓存 feature 放在前面、当前帧 top-N feature 放在后面，得到新的 selected_feature
        # 【shape】[B, T=num_temp_instances, C] + [B, M, C] -> [B, N=num_anchor=num_temp_instances+M, C]，即前 T 个是历史实例的特征、后 M 个是当前 topk 实例的特征
        # 【shape】detection 任务: [B, 600, 256] + [B, 300, 256] -> [B, 900, 256]，前 600 个是历史实例的特征、后300个是当前 topk 实例的特征
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1) 
        # 【Map Stage2 shape】[B,33,256] + [B,67,256] -> [B,100,256]。

        # (4.a2) anchor 同理，将历史缓存 anchor 放在前面、当前帧 top-N anchor 放在后面，得到新的 selected_anchor
        # 【shape】[B, T, D] + [B, M, D] -> [B, N, D]，即前 T 个是历史实例的anchor、后 M 个是当前 topk 实例的anchor
        # 【shape】detection 任务: [B, 600, 11] + [B, 300, 11] -> [B, 900, 11]，前 600 个是历史实例的anchor、后 300 个是当前 topk 实例的anchor
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)
        # 【Map Stage2 shape】[B,33,40] + [B,67,40] -> [B,100,40]；每个 slot 仍代表完整 20 点 polyline。

        # (4.b1) 根据 self.mask 判断是否使用历史融合结果
        # 若 self.mask=True，则使用 selected_feature，即前 T 个是历史实例的特征、后 M 个是当前 topk 实例的特征
        # 若 self.mask=False，则保留原始 instance_feature
        # 【shape】mask [B] -> [B, 1, 1] 广播到 [B, N, C]；
        # 对 mask=False 的样本，逐样本回退成纯当前 query，而不是让某个坏样本拖累整个 batch。
        instance_feature = torch.where(self.mask[:, None, None], selected_feature, instance_feature)

        # (4.b2) anchor 同理，根据 self.mask 判断是否使用历史融合结果
        # 【shape】mask [B] -> [B, 1, 1] 广播到 [B, N, D]。
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)

        # (5) 历史有效性保护：若某个 batch 样本的 history 已失效，则同步清空它的历史 confidence 和 tracking ID，避免跨 scene / 断序列误继承。
        # (5.a) 更新 confidence：若 mask=True，保留历史 confidence；若 mask=False，把 confidence 置 0
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            self.confidence.new_tensor(0)
        )

        # (5.b) 更新 instance_id：若已经维护了 instance_id，则将无效历史样本的 instance_id 置为 -1
        if self.instance_id is not None:
            # 对无效历史样本，将 instance_id 置为 -1
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )

        # (6) 把暂存的 DN instances 追加回末尾【最终前 self.num_anchor 个仍是普通 temporal instances，末尾才是 DN instances】
        if num_dn > 0: # 若存在 DN query
            # (6.a) 把 DN feature 拼回 instance_feature 的最后
            # 【shape】[B,N,C] + [B,N_dn,C] -> [B,N+N_dn,C]；normal temporal slots 仍在前 N 个，DN 永远追加末尾，便于后续 split。
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)

            # (6.b) 把 DN anchor 拼回 anchor 的最后
            # 【shape】[B,N,D] + [B,N_dn,D] -> [B,N+N_dn,D]；normal temporal slots 仍在前 N 个，DN 永远追加末尾，便于后续 split。
            anchor = torch.cat([anchor, dn_anchor], dim=1)

        # (7) 返回更新后的 instance_feature 和 anchor
        return instance_feature, anchor

    # 6. 最终 decoder 后，从当前帧预测中选出 top-T = 600 个（即 top-num_temp_instances 个）高置信度 instance，缓存为下一帧的历史 temporal instances
    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ): # 关键输入输出：[B,900,*] → [B,600,*]
        '''
            cache() 在 Sparse4DHead.forward() 结尾调用
            
            【论文对照】
                - SparseDrive 图3：这里就是 instance memory queue 的写入端。
                - Sparse4D v2 §3.1：从 refined current instances 中选高分子集，传递到下一帧。
                - Sparse4D v3 §3.4：top-k 同时承担 temporal-instance / tracklet 的生命周期管理。
            
            【输入 shape（detection 最后一层 refine 后）】
                instance_feature: [B,N=900,C=256]
                anchor          : [B,N=900,D=11]
                confidence      : [B,N=900,num_cls=10] 的 classification logits
            【写入后 shape】
                cached_feature  : [B,T=600,C=256]
                cached_anchor   : [B,T=600,D=11]
                confidence      : [B,T=600]
            
            作用：
                从当前帧预测中选出 top-T 个（即 top-num_temp_instances 个）高置信度 instance，
                缓存为下一帧的历史 temporal instances。          
        '''

        # 【Map 任务的 cache()】
        # Stage1 map：
        #   输入可视作 [B,100,256] / [B,100,40] / [B,100,3]，
        #   但 T_map=0，所以直接 return；temporal_map=True 并不等于真的保留 temporal map memory。
        # Stage2 map：
        #   将 [B,100,3] -> [B,100] 的 max-class probability 后，选择 top-33，
        #   写入 cached_feature=[B,33,256]、cached_anchor=[B,33,40]、confidence=[B,33]。
        #   下一帧 get() 会用 ego pose 将这 33 条局部 polyline 投影到当前坐标系。

        # (1) 判断该任务是否启用 temporal memory【Stage1 map 的 num_temp_instances=0 会直接在这里返回，不写入历史 map memory】
        # 若不需要缓存 temporal instances，则直接返回、并不写入历史 map memory
        if self.num_temp_instances <= 0:
            return # 直接返回 # map Stage1 的 T_map=0 会在此返回：即使 temporal_map=True，实际上并不写入历史 map memory。

        # (2) 将当前帧状态从计算图 detach，并保存当前帧 metas【缓存会给下一帧读取，但梯度不会跨帧反向传播】
        # (2.a) 避免梯度跨时间传播到上一帧缓存
        # 【训练语义】detach 截断跨帧反向传播：当前 step 只对当前帧计算图求梯度；InstanceBank 实现的是 recurrent state propagation，而不是把整段视频展开做 BPTT。
        instance_feature = instance_feature.detach() # detach 当前帧 instance feature
        anchor = anchor.detach()                     # detach anchor
        confidence = confidence.detach()             # detach confidence

        # (2.b) 保存当前帧 metas
        # 【shape】metas 内至少保存 timestamp:[B]、img_metas 长度 B、每个样本含 T_global:[4,4]。
        # 下一帧 get() 时会用它计算时间间隔和坐标变换，下一帧 get() 依据它构造 T_temp2cur。
        self.metas = metas

        # (3) 将分类 logits 转成每个 instance 的排序分数【得到的分数决定“哪些实例可以进入下一帧 temporal memory”】，并对历史实例应用 confidence decay。
        # (3.a) 将分类 logits 转成每个 instance 的排序分数
        confidence = confidence.max(dim=-1).values.sigmoid()
        # 【Map】对 map 而言：[B,100,3] -> [B,100]；该 confidence 只决定 polyline 是否留在 temporal memory，
        #       不会被用于 map tracking ID 分配。
        '''
            confidence 原始 shape 可能是 [B, N, num_cls]
            先取最大类别 logit，再 sigmoid 成概率
            [B, N, num_cls] -> max()和sigmoid() -> 分类结果的置信度概率probability [B, N]；max class score 后 sigmoid 得到用于 “跨帧生存” 比较的概率。            
        '''

        # (3.b) 对历史实例应用 confidence decay：若已有历史 confidence，则依据历史 confidence 和 confidence_decay 对新 confidence 做 max 更新
        if self.confidence is not None:
            '''
                【位置约定】由于 update() 将历史 T 个 slots 放到前缀，因此 confidence[:, :T] 正好是“被传播进当前帧的历史实例”。
                
                【作用】
                    对前 T=num_temp_instances 个历史 instance 的 confidence 做更新
                    新 confidence = max(历史 confidence 衰减后, 当前 confidence)
                    这样历史 instance 不会因为一帧置信度略低就立刻消失
                
                【论文对照】Sparse4D v3 Algorithm 1 定义了 confidence decay scale；
                公式语义：max(旧置信度×衰减系数, 当前帧重新预测分数)。            
            '''
            confidence[:, :self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, :self.num_temp_instances],
            )

        # (3.c) 保存当前帧临时 confidence【在 update_instance_id() 中可能会用到】
        self.temp_confidence = confidence

        # (4) 【核心】从当前 N 个 instances 同步选择 top-T，并写入下一帧的 memory【因为 feature、anchor、confidence 必须用同一套 top-k 索引，才能表示同一个 instance】
        # 从当前所有 instance 中选 top-T 个（即 top-num_temp_instances 个）作为历史缓存
        (
            self.confidence,                           # 缓存的 top-k confidence
            (self.cached_feature, self.cached_anchor), # 缓存的 feature 和 anchor
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor) # 【shape】从 [B, N=900, *] 选出 [B, T=600, *]；同一个 top-k index 同步选择 confidence / feature / anchor，三者顺序严格对齐。
        # 【Map Stage2】同一行会变成 [B,100,*] -> [B,33,*]，而 feature / anchor / confidence 的第 i 个位置严格指向同一条 polyline。

    # 7. 为当前帧 instance 分配或延续 instance id【仅 Detection Tracking 使用，而 map_head 的 with_instance_id=False，因此 map forward/test 不会调用下面两个 ID 方法即 get_instance_id() 和 update_instance_id()】
    def get_instance_id(self, confidence, anchor=None, threshold=None): # 关键输入输出：[B, 900, 10] → [B, 900]
        '''
            get_instance_id() 用于为当前帧 instance 分配或延续 instance id
            
            【论文对照】
                SparseDrive §3.2 “Sparse Tracking”：超过阈值就锁定目标并赋 ID，temporal propagation 中 ID 不变。
                Sparse4D v3 §3.4 / Algorithm 1：不做传统 tracking-by-detection 的显式 data association；
                temporal instance 自身的传播 + top-k 生命周期管理已经带来 ID consistency。
            
            【shape】
                输入 confidence: [B,N=900,num_cls] logits；输出 instance_id: [B,N=900] int64。
                -1 表示当前 slot 没有合法 track ID；合法整数表示本序列内的目标 ID。
                主要服务于 tracking 任务
            
            逻辑：
                1. 计算当前 instance confidence
                2. 若已有历史 instance_id，则延续前面一部分 id
                3. 对没有 id 且置信度超过阈值的 instance 分配新 id
                4. 更新缓存中的 instance_id        
        '''

        # (1) 初始化置信度 confidence：将当前分类 logits 转成每个 instance 的总体置信度】。初始化 instance_id：初始化一个全为 -1 的 ID 容器，-1 表示无有效 id 即 “当前没有有效 tracking ID”。
        # (1.a) 初始化置信度 confidence：将当前分类 logits 转成每个 instance 的总体置信度
        confidence = confidence.max(dim=-1).values.sigmoid()
        '''
            confidence 原始 shape 可能是 [B, N, num_cls]
            先取最大类别 logit，再 sigmoid 成概率
            【shape】[B, N, num_cls] -> max(dim=-1)和sigmoid() -> 分类结果的置信度概率probability [B, N]；这里的 probability 与 cache() 选 top-k 的依据一致。          
        '''

        # (1.b) 初始化 instance_id 容器为全 -1【-1 表示无有效 id】【先全 -1，随后分为“历史 ID 继承”与“新 ID 分配”两步】
        instance_id = confidence.new_full(confidence.shape, -1).long() # 【shape】instance_id 的 shape 与 confidence 的 shape 一样: [B, N]


        # (2) 继承前缀 temporal slots 的历史 ID【update() 已把历史 instances 放在最前面 T 个，因此前 T 个 slot 可以直接沿用上一帧 ID】
        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):  # 若之前已经有缓存 instance_id 且 batch size 和当前一致
            # 则将历史 instance_id 复制到当前 instance_id 前面部分【因为 update() 中已把历史 temporal instance 放在前面】
            # 【shape】self.instance_id 已 pad 成 [B,N]；其前 T 位是上帧 top-k 的 ID，后 M 位为 -1。这里借助 update() 的“历史实例置前”约定直接延续 ID。
            instance_id[:, :self.instance_id.shape[1]] = self.instance_id


        # (3) 【核心】为未继承 ID、且置信度达到阈值的当前实例分配新 ID【这些通常是新出现目标，或未进入上一帧 temporal memory 的目标】
        # (3.a) 只对 “没有 id” 且 “confidence >= threshold” 的 instance 分配新 id
        mask = instance_id < 0                      # (i) 找出还没有 id 的 instance
        if threshold is not None:                   # 若设置了置信度阈值 threshold【置信度阈值 threshold 来自 detection3d_head.py 的 self.decoder.score_threshold】
            mask = mask & (confidence >= threshold) # (ii) 只对 “没有 id” 且 “confidence >= threshold” 的 instance 分配新 id

        # (3.b) 统计需要新分配 id 的 instance 数量 num_new_instance
        num_new_instance = mask.sum()

        # (3.c) 生成新的 id（从 id 起点 self.prev_id 处开始递增）
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id

        # (3.d) 把新 id 填入对应位置
        instance_id[torch.where(mask)] = new_ids

        # (3.e) 更新下一个可用 id 的起点【prev_id 是 Python 标量式的全局递增计数器：同一连续序列内不会复用旧 ID】
        self.prev_id += num_new_instance


        # (4) 按与 feature / anchor cache 相同的 top-k 排序，缓存下一帧要继承的 ID
        self.update_instance_id(instance_id, confidence) # 更新缓存中的 instance_id
        return instance_id                               # 返回当前帧每个 instance 的 id

    # 8. 从当前 instance_id 中选出 top-T = 600（即 top-num_temp_instances）对应的 id，并缓存到 self.instance_id 中，供下一帧延续 tracking id，保证与 feature/anchor 顺序对齐
    def update_instance_id(self, instance_id=None, confidence=None): # 关键输入输出：[B, 900] → [B, 600] → [B, 900]
        '''
            update_instance_id() 用于从当前 instance_id 中选出 top temporal instances 的 id，并缓存到 self.instance_id 中，供下一帧延续 tracking id 
            【作用】让 ID cache 与 cache() 的 feature/anchor cache 使用同一套 top-k 排序依据，从而保证下一帧的 cached_feature[i]、cached_anchor[i]、instance_id[i] 指向同一个实例。
            【论文对照】Sparse4D v3 §3.4 / Algorithm 1 的 “Select highest-confidence instances as temporal instances”。       
        '''

        # (1) 确定 ID 缓存用的排序分数【优先使用 cache() 保存的 temp_confidence，确保 ID 与 feature / anchor cache 的 top-k 一致】
        if self.temp_confidence is None:                  # a. 若没有 temp_confidence：
            if confidence.dim() == 3:                          # 若 confidence 是三维[bs, num_anchor, num_cls]，
                temp_conf = confidence.max(dim=-1).values      # 则取最大类别 confidence【经 max(dim=-1) 后 confidence 变为 [bs, num_amchor]】
            else:                                              # 若 confidence 已经是二维[bs, num_anchor]，
                temp_conf = confidence                         # 则直接使用 confidence。
        else:                                             # b. 若已有 temp_confidence：
            temp_conf = self.temp_confidence              # 优先使用 cache() 中保存的 temp_confidence

        # (2) 【核心】用该排序分数 temp_conf 同步选择 top-T = 600（即 top-num_temp_instances）的 instance ID【选出的第 i 个 ID 必须对应 cached_feature[:, i] 与 cached_anchor[:, i]】
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][0] # topk 返回的 instance_id shape 是 [B, K, 1]
        instance_id = instance_id.squeeze(dim=-1)                                 # 去掉最后一维，得到 [B, K]
        '''
            【shape】
                temp_conf: [B, N]；
                instance_id: [B, N] -> top-k 返回 [B, T, 1]，随后 squeeze 成 [B, T]。
        '''

        # (3) 【核心】将 top-T ID 放到前缀并用 -1 补齐为 [B, N]，后 N-T 个位置预留给下一帧的新目标 slot
        # 将 instance_id padding 到 num_anchor 长度【这样下一帧 get_instance_id() 时可以按 num_anchor 对齐】：前 num_temp_instances 是真实缓存 id，后面补 -1
        # 【shape】[B,T] -> pad -> [B,N]；保持与 update() 输出位置布局一致：前 T 位是传播实例，后 M=N-T 位预留给下一帧的“新目标 slots”。
        self.instance_id = F.pad(
            instance_id,                                    # top temporal instance id
            (0, self.num_anchor - self.num_temp_instances), # 在最后一维右侧补齐
            value=-1,                                       # 补充值为 -1
        )
