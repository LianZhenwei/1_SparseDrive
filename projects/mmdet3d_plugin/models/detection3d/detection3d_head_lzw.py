from typing import List, Optional, Tuple, Union # 从 typing 模块导入类型标注工具
import warnings                                 # 导入 warnings 模块，用于输出警告信息。但在当前文件中 warnings 实际没有被使用，属于冗余导入
import numpy as np                              # 导入 numpy。但当前文件中 np 实际没有被使用，属于冗余导入
import torch                                    # 导入 PyTorch。本文件中大量使用 torch.Tensor、torch.cat、torch.where 等操作
import torch.nn as nn                           # 导入 PyTorch 神经网络模块。主要用到了 nn.ModuleList、nn.Linear、nn.Identity、nn.init 等

# 从 MMCV 的 registry 中导入各种模块注册表【Sparse4DHead 会根据配置字典，从这些注册表中构建 attention、FFN、norm、refine 等模块】
from mmcv.cnn.bricks.registry import (
    ATTENTION,           # attention 注册表，例如 MultiheadAttention、MultiheadFlashAttention、DeformableFeatureAggregation 等
    PLUGIN_LAYERS,       # 插件层注册表，例如 InstanceBank、SparseBox3DRefinementModule 等自定义模块
    POSITIONAL_ENCODING, # 位置编码注册表，例如 SparseBox3DEncoder、SparsePoint3DEncoder
    FEEDFORWARD_NETWORK, # 前馈网络注册表，例如 AsymmetricFFN
    NORM_LAYERS,         # 归一化层注册表，例如 LN、BN 等
)

from mmcv.runner import BaseModule, force_fp32    # BaseModule 是 MMCV 对 nn.Module 的封装，支持 init_cfg 初始化机制；force_fp32 是装饰器，用于在混合精度训练中强制某些输入转成 fp32
from mmcv.utils import build_from_cfg             # build_from_cfg 根据配置字典和 registry 构建模块，例如 build_from_cfg(cfg, ATTENTION)
from mmdet.core.bbox.builder import BBOX_SAMPLERS # BBOX_SAMPLERS 是 bbox sampler 注册表，这里用于构建 SparseBox3DTarget 或 SparsePoint3DTarget 等目标分配模块
from mmdet.core.bbox.builder import BBOX_CODERS   # BBOX_CODERS 是 bbox coder 注册表，这里用于构建 SparseBox3DDecoder 或 SparsePoint3DDecoder
from mmdet.models import HEADS, LOSSES            # HEADS 是检测头注册表，LOSSES 是损失函数注册表
from mmdet.core import reduce_mean                # reduce_mean 用于分布式训练中对不同 GPU 上的数值做平均，这里主要用于统计正样本数量 num_pos

# 从上一级 blocks 中导入 DeformableFeatureAggregation，并重命名为 DFG。但当前文件中 DFG 没有被直接使用，属于冗余导入
from ..blocks import DeformableFeatureAggregation as DFG

# 控制 from xxx import * 时暴露的对象，当前模块主要暴露 Sparse4DHead
__all__ = ["Sparse4DHead"]



'''
    =============================================================================
    detection3d_head.py源码对照着论文的总览：
        1. SparseDrive 论文第 3.2 节、图 4：稀疏检测由 1 个非时序 decoder 和多个时序 decoder 构成。
                非时序 decoder：deformable aggregation -> FFN -> refine。
                时序 decoder：在上述基础上额外增加 temporal cross-attention 与 instance self-attention。
        2. SparseDrive 论文第 3.2 节：anchor box 经 anchor_encoder 得到 anchor embedding，并作为 temporal cross-attention / self-attention 的 positional encoding。
        3. Sparse4D v3 论文第 3.3 节、图 5：Decoupled Attention。
                其关键改动是把“instance feature 与 anchor embedding 相加”改为在通道维拼接，
                以减轻不同模态信息在 attention 权重计算阶段的相互干扰。
        4. Sparse4D v3 论文第 3.1 节、图 4：Temporal Instance Denoising（DN）。

    shape 记号：
        B：batch size。
        N_free：普通检测 query 数；发布的 SparseDrive-S 检测配置通常为 900。
        N_temp：历史帧传播来的 temporal instance 数；发布配置通常为 600。
        N_dn：训练时附加的 denoising query 数；未开启 DN 时为 0。
        N = N_free + N_dn：当前 decoder 内参与计算的 query 总数。
        C = embed_dims = 256。
        D_box = 11：检测 anchor 的状态维度，通常为
        [x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz]。
        K = 10：nuScenes 的检测类别数。
        开启 decouple_attn 后，attention 内部宽度为 2C=512；attention 结束后再投影回 C=256。
    =============================================================================
'''

# Sparse4DHead 类：SparseDrive 和 Sparse4D 的核心稀疏检测头
@HEADS.register_module() # 将 Sparse4DHead 注册到 MMDetection 的 HEADS 注册表，配置文件中写 type="Sparse4DHead" 时，就会构建这个类
class Sparse4DHead(BaseModule):
    '''
        Sparse4DHead 是 SparseDrive/Sparse4D 的核心稀疏检测头，它的主要功能：
            1. 从 InstanceBank 中取出 learnable anchor/query
            2. 可选融合历史帧 instance，实现 temporal modeling
            3. 可选加入 denoising anchors，辅助训练
            4. 通过 gnn、temp_gnn、deformable、ffn、refine 多层迭代更新 anchor
            5. 输出分类、回归、质量估计、instance_id 等结果
    '''

    # =============================================================================
    # 【Sparse4DHead 的统一阅读地图：检测头与地图头共用本类】
    # =============================================================================
    # 这个文件名虽然叫 detection3d_head.py，但 SparseDrive 的 map_head 也会用同一个 Sparse4DHead；两者的控制流、decoder 调度、loss 总框架完全相同，只替换以下
    # “任务插件”：InstanceBank / anchor_encoder / refine_layer / sampler / decoder / loss。
    #
    # ┌──────────────────┬───────────────────────────────┬────────────────────────────────┐
    # │                  │ Detection 3D                  │ Vectorized Map                 │
    # ├──────────────────┼───────────────────────────────┼────────────────────────────────┤
    # │ 一个 instance     │ 一个动态 3D object             │ 一条局部地图 polyline           │
    # │ anchor           │ [x,y,z,logw,logl,logh,sin,cos,│ 20 个二维点 flatten 后的 40 维  │
    # │                  │  vx,vy,vz]，D=11              │ D=20×2=40                      │
    # │ normal query 数  │ N_det=900                     │ N_map=100                      │
    # │ 分类输出          │ [B,900,10]                    │ [B,100,3]                      │
    # │ temporal memory  │ T_det=600                     │ Stage1: T_map=0；Stage2: 33    │
    # │ instance_id      │ 开启：用于 sparse tracking      │ 关闭：with_instance_id=False   │
    # └──────────────────┴───────────────────────────────┴────────────────────────────────┘
    #
    # 【整个 forward() 的状态流】
    # (1) InstanceBank.get()：拿到当前帧默认 query/anchor，以及可能存在的历史 memory。
    # (2) 训练时可选 DN：把 noisy DN queries 追加到普通 queries 的尾部。
    # (3) anchor_encoder：把几何 anchor 变成 attention 可用的 anchor_embed。
    # (4) decoder loop：按照 operation_order 依次执行 temp_gnn / gnn / deformable / ffn / refine；第一次单帧 refine 后通过 InstanceBank.update() 接入 temporal slots。
    # (5) 输出收尾：普通 query 与 DN query 分离；普通 query 写入 InstanceBank.cache()；detection 额外生成 instance_id，map 则跳过该步骤。
    # =============================================================================

    # 1. 初始化函数：
    def __init__(
        self,                                        # self 表示当前 Sparse4DHead 对象
        instance_bank: dict,                         # instance_bank 配置：用于构建和维护稀疏 instance/query/anchor
        anchor_encoder: dict,                        # anchor_encoder 配置：用于把 anchor 的几何参数编码成 embedding
        graph_model: dict,                           # 当前帧 query 之间交互的 graph attention 配置
        norm_layer: dict,                            # 归一化层配置
        ffn: dict,                                   # FFN 前馈网络配置
        deformable_model: dict,                      # deformable feature aggregation 配置：用于根据 anchor 关键点从多相机多尺度图像特征中采样聚合特征
        refine_layer: dict,                          # refine layer 配置：用于根据 instance_feature 更新 anchor，并输出分类和质量估计
        num_decoder: int = 6,                        # decoder 层数，默认 6 层【Stage1和Stage2的配置文件均传入 num_decoder=6】
        num_single_frame_decoder: int = -1,          # 单帧 decoder 层数。在单帧层之后才会更新 instance_bank 并引入 temporal instance。【Stage1和Stage2的配置文件均传入 num_single_frame_decoder=1】
        temp_graph_model: dict = None,               # temporal graph attention 配置：用于当前帧 instance 和历史帧 instance 之间的信息交互      
        loss_cls: dict = None,                       # 分类损失配置
        loss_reg: dict = None,                       # 回归损失配置
        decoder: dict = None,                        # decoder 配置：用于把网络输出解码成最终 bbox 或 map line 结果
        sampler: dict = None,                        # sampler 配置：用于训练时将预测和 GT 匹配，并生成 cls_target、reg_target、reg_weights
        gt_cls_key: str = "gt_labels_3d",            # GT 分类标签在 data 字典中的 key：对检测任务一般是 "gt_labels_3d"，对 map 任务可能是 "gt_map_labels"
        gt_reg_key: str = "gt_bboxes_3d",            # GT 回归目标在 data 字典中的 key：对检测任务一般是 "gt_bboxes_3d"，对 map 任务可能是 "gt_map_pts"
        gt_id_key: str = "instance_id",              # GT instance id 在 img_metas 中的 key：用于 temporal denoising 或 tracking 关联
        with_instance_id: bool = True,               # 是否输出 instance_id：检测任务一般需要，用于 tracking，map 任务可能不需要
        task_prefix: str = 'det',                    # 任务前缀：det head 一般是 'det'，map head 一般是 'map'，loss 名称会带这个前缀
        reg_weights: List = None,                    # 回归维度权重：检测任务可能是 10 维 bbox 状态权重，map 任务可能是 40 维点序列权重
        operation_order: Optional[List[str]] = None, # 操作顺序：控制 decoder 每一层执行 temp_gnn、gnn、norm、deformable、ffn、refine 的顺序
        cls_threshold_to_reg: float = -1,            # 分类分数阈值：若 >0，则只有分类置信度超过该阈值的预测才参与回归 loss
        dn_loss_weight: float = 5.0,                 # denoising loss 权重。注意：当前代码中保存了这个变量，但 loss 里没有显式乘 dn_loss_weight
        decouple_attn: bool = True,                  # 是否使用解耦 attention：True 时将 instance_feature 和 anchor_embed 拼接后送入 attention【Stage1和Stage2的配置文件均传入 detection任务和motion任务的 decouple_attn=True、而map任务的 decouple_attn=False】
        init_cfg: dict = None,                       # MMCV 初始化配置
        **kwargs,                                    # 接收额外关键字参数，当前代码中没有显式使用
    ):
        super(Sparse4DHead, self).__init__(init_cfg) # 调用父类 BaseModule 初始化，init_cfg 用于控制权重初始化

        # (1.1) 保存参数：
        self.num_decoder = num_decoder                           # 保存 decoder 层数
        self.num_single_frame_decoder = num_single_frame_decoder # 保存单帧 decoder 层数
        self.gt_cls_key = gt_cls_key                             # 保存 GT 分类 key
        self.gt_reg_key = gt_reg_key                             # 保存 GT 回归 key
        self.gt_id_key = gt_id_key                               # 保存 GT instance id key
        self.with_instance_id = with_instance_id                 # 保存是否生成 instance_id 的开关
        self.task_prefix = task_prefix                           # 保存任务前缀
        self.cls_threshold_to_reg = cls_threshold_to_reg         # 保存分类阈值
        self.dn_loss_weight = dn_loss_weight                     # 保存 denoising loss 权重，当前文件中后续没有实际使用这个权重
        self.decouple_attn = decouple_attn                       # 保存是否使用解耦 attention

        # (1.2) 设置回归权重：
        if reg_weights is None:            # 若没有传入 reg_weights
            self.reg_weights = [1.0] * 10  # 默认使用 10 维回归权重，对 3D box 检测来说，常见状态可能包括 x, y, z, w, l, h, yaw, vx, vy 等
        else:                              # 若传入了 reg_weights
            self.reg_weights = reg_weights # 使用配置文件中的回归权重

        # (1.3) 【这是一张“执行操作清单”】设置 decoder 操作序列：
        if operation_order is None: # 若没有显式指定 operation_order，则构造默认 decoder 操作序列
            # (a） 构造默认 decoder 操作序列
            operation_order = [
                "temp_gnn",   # temporal graph attention，用于融合历史帧 instance 信息
                "gnn",        # 当前帧 query 之间的 graph attention
                "norm",       # 归一化
                "deformable", # 从多相机多尺度图像特征中进行 deformable feature aggregation
                "norm",       # 归一化
                "ffn",        # FFN 前馈网络
                "norm",       # 归一化
                "refine",     # refine，更新 anchor 并输出预测
            ] * num_decoder   # 上面这一组操作重复 num_decoder=6 次

            # (b) 删除第一组 transformer block 前面的 temp_gnn、gnn、norm：operation_order[3:] 会跳过前三个操作："temp_gnn", "gnn", "norm"，也就是第一层直接从 deformable 开始
            operation_order = operation_order[3:]
        self.operation_order = operation_order # 保存最终的操作顺序
        '''
            但 /home/lzw/SparseDrive/projects/configs/sparsedrive_small_stage1.py 和 /home/lzw/SparseDrive/projects/configs/sparsedrive_small_stage2.py 中，
            这俩配置文件实际传入的 operation_order 非 None，因此并不走这里的 if 分支的 operation_order：
                【detection任务】对于 detection 任务，这俩配置文件实际传入的 operation_order 为：
                                    第 1 个 decoder：deformable → ffn → norm → refine
                                    第 2~6 个 decoder：temp_gnn → gnn → norm → deformable → ffn → norm → refine
                                    总共 4+7*5=39 个模块【称 deformable 为1个模块、称 ffn 为1个模块、称 norm 为1个模块...】
                【map任务】对于 map 任务，这俩配置文件实际传入的 operation_order 为：
                                    第 1 个 decoder：gnn → norm → deformable → ffn → norm → refine
                                    第 2~6 个 decoder：temp_gnn → gnn → norm → deformable → ffn → norm → refine
                                    总共 6+7*5=41 个模块
        '''

        # (2) 下面根据配置字典构建各个任务插件子模块
        # (2.1) 定义内部 build() 函数
        def build(cfg, registry): # cfg 是配置字典，registry 是对应注册表
            if cfg is None:                      # 若配置为空，
                return None                      # 则返回 None，
            return build_from_cfg(cfg, registry) # 否则根据 cfg 和 registry 构建模块

        # (2.2) 根据配置字典构建各个子模块
        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)         # 构建 instance bank，例如 InstanceBank，它负责维护 learnable anchors、历史 anchors、instance_feature 缓存等
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING) # 构建 anchor encoder，例如 SparseBox3DEncoder 或 SparsePoint3DEncoder
        self.sampler = build(sampler, BBOX_SAMPLERS)                     # 构建 sampler，例如 SparseBox3DTarget  或 SparsePoint3DTarget
        self.decoder = build(decoder, BBOX_CODERS)                       # 构建 decoder，例如 SparseBox3DDecoder 或 SparsePoint3DDecoder
        self.loss_cls = build(loss_cls, LOSSES)                          # 构建分类损失，例如 FocalLoss
        self.loss_reg = build(loss_reg, LOSSES)                          # 构建回归损失，例如 SparseBox3DLoss 或 SparseLineLoss

        # (2.3) 建立“操作名称”到“配置注册表”的映射，后面会根据 operation_order 逐个 build 对应 layer
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],   # temporal GNN 使用 temp_graph_model，并从 ATTENTION 注册表构建
            "gnn": [graph_model, ATTENTION],             # 当前帧 GNN 使用 graph_model，并从 ATTENTION 注册表构建
            "norm": [norm_layer, NORM_LAYERS],           # norm 使用 norm_layer，并从 NORM_LAYERS 注册表构建
            "ffn": [ffn, FEEDFORWARD_NETWORK],           # FFN 使用 ffn，并从 FEEDFORWARD_NETWORK 注册表构建
            "deformable": [deformable_model, ATTENTION], # deformable 使用 deformable_model，并从 ATTENTION 注册表构建
            "refine": [refine_layer, PLUGIN_LAYERS],     # refine 使用 refine_layer，并从 PLUGIN_LAYERS 注册表构建
        }

        # (2.4) 【这是按“执行清单”逐项创建出的 39或41 个真实 PyTorch 模块】根据 operation_order 构建每一个实际 layer：
        self.layers = nn.ModuleList([build(*self.op_config_map.get(op, [None, None])) for op in self.operation_order])
        '''
            self.layers 不按“第几层 decoder”手工嵌套存储，而是按 operation_order 展平成一个 ModuleList；forward() 会再次用同一份 operation_order 调度它
                self.op_config_map.get(op, [None, None])：根据 op 名称取出对应的 [cfg, registry]，若没找到，就返回 [None, None]
                build(*...)：用配置和注册表构建对应模块        
                for op in self.operation_order：遍历 operation_order 中的每个操作

            那么此时 self.operation_order[i] 长啥样就决定了 self.layers[i] 长啥样：
                【detection任务】对于 detection 任务，2个配置文件实际传入的 operation_order 均为：
                                    第 1 个 decoder：deformable → ffn → norm → refine
                                    第 2~6 个 decoder：temp_gnn → gnn → norm → deformable → ffn → norm → refine
                                    总共 4+7*5=39 个模块【称 deformable 为1个模块、称 ffn 为1个模块、称 norm 为1个模块...】
                【map任务】对于 map 任务，2个配置文件实际传入的 operation_order 均为：
                                    第 1 个 decoder：gnn → norm → deformable → ffn → norm → refine
                                    第 2~6 个 decoder：temp_gnn → gnn → norm → deformable → ffn → norm → refine
                                    总共 6+7*5=41 个模块

            下面是 detection 任务的完整的 self.layers[i] 的展开表：
                   self.layers[i] 的 i  | 所属 decoder  | operation_order[i]  |    self.layers[i] 的真实模块
                ---------------------------------------------------------------------------------------------------
                         0              |  第 1 层      |    deformable       |  DeformableFeatureAggregation
                         1              |  第 1 层      |    ffn              |  AsymmetricFFN
                         2              |  第 1 层      |    norm             |  LayerNorm(256)
                         3              |  第 1 层      |    refine           |  SparseBox3DRefinementModule
                         4              |  第 2 层      |    temp_gnn         |  MultiheadFlashAttention
                         5              |  第 2 层      |    gnn              |  MultiheadFlashAttention
                         6              |  第 2 层      |    norm             |  LayerNorm(256)
                         7              |  第 2 层      |    deformable       |  DeformableFeatureAggregation
                         8              |  第 2 层      |    ffn              |  AsymmetricFFN
                         9              |  第 2 层      |    norm             |  LayerNorm(256)
                         10             |  第 2 层      |    refine           |  SparseBox3DRefinementModule
                         11~17          |  第 3 层      |    同第 2 层         |      同第 2 层的七类模块
                         18~24          |  第 4 层      |    同第 2 层         |      同第 2 层的七类模块
                         25~31          |  第 5 层      |    同第 2 层         |      同第 2 层的七类模块
                         32~38          |  第 6 层      |    同第 2 层         |      同第 2 层的七类模块
            注意：虽然第 2、3、4、5、6 层 decoder 里的模块配置相同，但它们不是同一个对象，因为它们都是重新 build() 出来的独立模块，参数不共享。
            
            不过有两个模块不在 self.layers 中，而是所有 attention 共用：
                    self.fc_before = Linear(256, 512, bias=False)
                    self.fc_after  = Linear(512, 256, bias=False)
            它们负责 decoupled attention 的 256 ↔ 512 维转换；10 个 temp_gnn/gnn attention 都会共用它们。
        '''

        # (3) 构建 Decoupled Attention 的 256↔512 维桥接线性层：
        self.embed_dims = self.instance_bank.embed_dims # 从 instance_bank 中读取 embedding 维度，一般是 256
        if self.decouple_attn: # (3.1) 若启用解耦 attention，则需设置解耦 attention 前的线性层和解耦 attention 后的线性层
            # (a) 设置 attention 前的线性层：将 value 从 256 维映射到 512 维【因为 decouple 模式下 query 会拼接 instance_feature 和 anchor_embed】
            self.fc_before = nn.Linear(
                self.embed_dims,     # 输入维度 256
                self.embed_dims * 2, # 输出维度 512
                bias=False           # 不使用 bias
            )

            # (b) 设置 attention 后的线性层：将 512 维再映射回 256 维
            self.fc_after = nn.Linear(
                self.embed_dims * 2, # 输入维度 512
                self.embed_dims,     # 输出维度 256
                bias=False           # 不使用 bias
            )
        else:                              # (3.2) 若不启用解耦 attention
            self.fc_before = nn.Identity() # (a) 前处理层直接恒等映射
            self.fc_after = nn.Identity()  # (b) 后处理层直接恒等映射
        '''
            1. 若 decouple_attn=True 时，则 attention 内部宽度是 512、但 attention 输出会重新映射回 256：
                    (1) Q/K = [instance_feature ; anchor_embed]，Q/K 宽度由 C=256 变为 2C=512；此时 V 也必须先投影到 512。
                    (2) attention 输出再从 512 映射回 256，保证后续 FFN / deformable / refine 仍使用统一的 embed_dims=256。
            2. Stage1和Stage2的配置文件均传入 detection 任务和 motion 任务的 decouple_attn=True，因此 detection 和 motion 走 Linear 分支；
               Stage1和Stage2的配置文件均传入 map 任务的 decouple_attn=False，因此 map 走 Identity 分支        
        '''

    # 2. 初始化 Sparse4DHead 中各个子模块的权重
    def init_weights(self):
        # (1) 遍历 operation_order 中每一层，对 operation_order 中的非 refine layer 做通用 Xavier 初始化
        for i, op in enumerate(self.operation_order):
            # (a) 若当前 layer 是 None，则跳过
            if self.layers[i] is None:
                continue # 跳过

            # (b) 对 operation_order 中的非 refine layer 做通用 Xavier 初始化【而 refine layer 不在这里强行初始化，因为它可能有任务专属的分类 bias、Scale 等初始化规则，留给自身 init_weight() 处理】
            elif op != "refine":                      # 若当前操作不是 'refine'
                for p in self.layers[i].parameters(): # 遍历当前 layer 的所有参数
                    if p.dim() > 1:                   # 若参数维度大于 1（通常表示这是 Linear/Conv 的权重，而不是 bias 或 norm 参数）
                        nn.init.xavier_uniform_(p)    # 则使用 Xavier uniform 初始化

        # (2) 再遍历当前模块及其所有子模块，调用每个模块自己的 init_weight()
        for m in self.modules():
            if hasattr(m, "init_weight"): # 若子模块有 init_weight 方法（注意这里方法名是 init_weight，不是常见的 init_weights）
                m.init_weight()           # 则调用子模块自己的初始化函数

    # 3. attention：
    def graph_model(
        self,           # self 表示当前 Sparse4DHead 对象
        index,          # index 表示当前要调用 self.layers[index]
        query,          # query 是 attention 的 query 输入
        key=None,       # key 是 attention 的 key 输入，默认 None
        value=None,     # value 是 attention 的 value 输入，默认 None
        query_pos=None, # query_pos 是 query 的位置编码
        key_pos=None,   # key_pos 是 key 的位置编码
        **kwargs,       # 其他传给 attention 的参数，例如 attn_mask
    ):
        '''
            【graph_model() 的三步】
                (1) 组织 Q / K：决定几何 anchor_embed 如何进入 attention。
                    (a) decouple_attn=True：沿通道 concat，语义与几何在注意力输入中保留独立子空间
                    (b) decouple_attn=False：不在此处拼接，直接把 query_pos / key_pos 交给底层 attention
                        备注：
                            Stage1和Stage2的配置文件均传入 detection 任务和 motion 任务的 decouple_attn=True，因此 detection 和 motion 走 Linear 分支；
                            Stage1和Stage2的配置文件均传入 map 任务的 decouple_attn=False，因此 map 走 Identity 分支
                (2) 组织 V：若 decouple=True，V 同步从 C 投影到 2C，保证 Q/K/V 维度相容。
                (3) 调用 self.layers[index] 的 attention，并将结果投影回 C。
            
            该函数既服务于：
                - temp_gnn：Q=当前 instances，K/V=历史 instances；
                - gnn：Q/K/V=当前 instances（底层 attention 的 key=None 时按其实现处理）。
        '''

        # (1) 若使用 decouple attention，则将 query位置编码 拼到 query 后、将 key位置编码 拼到 key 后
        if self.decouple_attn:
            # (a) 将 query 内容特征和 query 位置编码在最后一维拼接
            # [B, N_q, C=256] + [B, N_q, C=256] -> [B, N_q, 2*C=512]：前 256 维是当前 instance 的语义特征、后 256 维是当前 instance 的 anchor 几何编码
            query = torch.cat([query, query_pos], dim=-1)

            # (b) 将 key 内容特征和 key 位置编码在最后一维拼接
            # [B, N_temp, C=256] + [B, N_temp, C=256] -> [B, N_temp, 2*C=512]：前 256 维是历史帧缓存的 instance 的语义特征、后 256 维是历史帧缓存的 instance 的 anchor 几何编码
            if key is not None:                         # 若 key 不为空
                key = torch.cat([key, key_pos], dim=-1) # 将 key 内容特征和 key 位置编码拼接

            # (c) 拼接后已经把位置信息放进 query/key 里，所以不再额外传 query_pos/key_pos【“位置编码已经手工拼进 Q/K 里了，底层 attention 不许再额外加位置编码”】
            query_pos, key_pos = None, None

        # (2) 由于 query 和 key 已经因拼接从 256 变成了 512，即 attention 的内部通道宽度也变成 512，因此 value 必须匹配，即 value 也要从 256 投影到 512：
        if value is not None: # 若 value 不为空
            # 将 value 映射到 attention 需要的维度
            # decouple_attn=True 时 256 -> 512【Stage1和Stage2的配置文件均传入 detection 任务和 motion 任务的 decouple_attn=True，因此 detection 和 motion 走 Linear 分支】
            # decouple_attn=False 时 Identity【Stage1和Stage2的配置文件均传入 map 任务的 decouple_attn=False，因此 map 走 Identity 分支】
            value = self.fc_before(value)

        # (3) 最后，attention 输出仍然是 512 维，因此调用第 index 层 attention，然后再从 512 映射回原来的 embed_dims=256：
        return self.fc_after(
            self.layers[index](      # 调用具体 attention layer
                query,               # query 输入
                key,                 # key 输入
                value,               # value 输入
                query_pos=query_pos, # query 位置编码
                key_pos=key_pos,     # key 位置编码              
                **kwargs,            # 其他参数，例如 attn_mask
            )
        )

    # 4. 前向传播：
    def forward(
        self,                                    # self 表示当前 Sparse4DHead 对象
        feature_maps: Union[torch.Tensor, List], # feature_maps 是图像特征
        metas: dict,                             # metas 是数据字典，训练时包含 img_metas、gt_labels_3d、gt_bboxes_3d 等，测试时包含相机参数、时间戳、ego 信息等
    ):
        '''
            【forward() 的八段逻辑】
                (0) 统一输入并检查 batch 对齐。
                (1) 从 InstanceBank 读取当前 templates 和历史 temporal memory。
                (2) 训练时构造 DN queries、DN targets 与 attention mask。
                (3) 对当前 anchor 和 历史 anchor 做几何编码。
                (4) 按 operation_order 执行 decoder；单帧阶段结束后接入 temporal instances。
                (5) 若存在 DN，将 normal outputs 与 DN outputs 拆开。
                (6) 组装普通预测输出字典。
                (7) 将普通高分实例缓存到下一帧。
                (8) 仅 detection：生成 / 延续 sparse tracking 的 instance_id。
        '''
        # (0) 准备工作：检查 batch_size 对齐
        # a. 若 feature_maps 是单个 Tensor，则将其包装成 list，统一后续处理
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps] # 包装成 list，统一后续处理

        # b. 获取 batch size，feature_maps[0] 的第 0 维就是 B 即 batch_size
        batch_size = feature_maps[0].shape[0]

        # c. 确保不会发生 “缓存的 dn_anchor batch size 和当前 batch size不一致” 的错误
        if (
            self.sampler.dn_metas is not None                             # 若 sampler 中缓存了 dn_metas
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size # 但缓存的 dn_anchor batch size 和当前 batch size 不一致
        ):
            self.sampler.dn_metas = None # 清空缓存，避免 batch size 不一致导致错误


        # (1) 读取 InstanceBank：当前帧 default templates + 上一帧 temporal memory
        #     (a) instance_feature / anchor 是“本帧用于发现新目标或新地图线”的 normal slots。
        #     (b) temp_instance_feature / temp_anchor 是上一帧 top-k memory；get() 已把历史 anchor 投影到当前 ego/lidar 坐标系。
        #     (c) detection 常见 shape：normal [B,900,256] / [B,900,11]，temp [B,600,*]。
        #     (d) map 常见 shape：normal [B,100,256] / [B,100,40]。Stage1 的 T_map=0，temp_* 为 None；Stage2 的 T_map=33，历史 33 条 polyline 可进入 temp_gnn。
        #     (e) time_interval 对 detection 的 velocity-aware box refinement 有实际意义；map 的 point refinement 虽也接收该参数，但不利用速度外推。

        # (1) 读取 InstanceBank：从 instance_bank 中获取当前帧 learnable instances 和上一帧 temporal instances 信息
        (
            instance_feature,      # 当前帧 instance feature。detection 任务 shape [B, num_anchor=900, C=256]，map 任务 shape [B, num_anchor=100, C=256]
            anchor,                # 当前帧 anchor。detection 任务 shape [B, num_anchor=900, box_dim=11]，map 任务 shape [B, num_anchor=100, point_dim=40]
            temp_instance_feature, # 上一帧 top-k instance feature，用于 temporal GNN。首帧时无上一帧因此为 None，首帧之后：detection 任务 shape [B, 600, 256]，map 任务 shape [B, 600, 256]
            temp_anchor,           # 上一帧 anchor，get()已将其投影到当前帧坐标系。首帧时无上一帧因此为 None，首帧之后：detection 任务 shape [B, 33, 256]，map 任务 shape [B, 33, 40]
            time_interval,         # 当前帧和上一帧之间的时间间隔，即当前帧 timestamp - 上一帧 timestamp
        ) = self.instance_bank.get(
            batch_size,                    # batch size
            metas,                         # 元信息
            dn_metas=self.sampler.dn_metas # 传入 denoising metas，用于 temporal denoising
        )


        # (2) 可选的去噪训练分支 Denoising (DN) ：只在训练阶段(即 self.training=True)且 sampler 支持时启用去噪训练分支
        '''
            (2.1) 生成带噪声的 anchors 和对应 GT
            (2.2) 把普通 learnable anchors 和 denoising anchors 拼接
            (2.3) 构造 attention mask，限制普通 query 和 DN query 的交互：normal queries 彼此可见；DN queries 只按 dn_attn_mask 在其内部通信，防止不同 DN group 或 normal/DN 之间产生不希望的泄漏。
        '''
        # (2.0) 初始化
        attn_mask = None          # attention mask 初始化为 None
        dn_metas = None           # denoising metas 初始化为 None
        temp_dn_reg_target = None # temporal denoising 回归目标初始化为 None

        # (2.1) 仅训练阶段且 sampler 支持 get_dn_anchors 时生成 DN【测试或推理时 dn_metas 保持 None，不增加任何 query】
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            # (a) 先取得 GT instance_id（若数据有提供），供 temporal DN 做跨帧匹配
            if self.gt_id_key in metas["img_metas"][0]: # 若 img_metas 中存在 gt_id_key（默认 gt_id_key 是 "instance_id"），即若有 instance id
                gt_instance_id = [                      # 则从每个样本的 img_metas 中取出 instance_id，并转成 CUDA Tensor
                    torch.from_numpy(x[self.gt_id_key]).cuda() # x[self.gt_id_key] 是 numpy 数组，先 torch.from_numpy() 转成 Tensor，再 .cuda() 放到 GPU 上                   
                    for x in metas["img_metas"]                # 遍历 batch 中每个样本的 img_metas
                ]
            else:                     # 若没有 instance id
                gt_instance_id = None # 则 gt_instance_id 设为 None

            # (b) 再调用 sampler.get_dn_anchors() 生成 denoising anchors：
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key], # GT 类别标签
                metas[self.gt_reg_key], # GT 回归目标
                gt_instance_id,         # GT instance id
            )

        # (2.2) 成功生成 DN 成功后，再把 DN 状态并入当前 decoder 输入
        if dn_metas is not None: # 若成功生成 dn_metas
            # (2.2.a) 解包 denoising 信息
            (
                dn_anchor,     # 带噪声的 anchor
                dn_reg_target, # denoising 回归目标
                dn_cls_target, # denoising 分类目标
                dn_attn_mask,  # denoising attention mask
                valid_mask,    # 有效 mask
                dn_id_target,  # denoising instance id target
            ) = dn_metas                       # 解包 denoising 信息
            num_dn_anchor = dn_anchor.shape[1] # 获取 denoising anchor 数量
            '''
                dn_anchor    : [B, N_dn, D]，N_dn 是 DN anchor 的数量。它是 GT 加噪声后得到的 DN anchor，它后续会拼接到 normal 900 个 anchor 的尾部
                dn_reg_target: [B, N_dn, D]。它是原始 GT box，作为回归目标（即 DN 回归监督目标）
                dn_cls_target: [B, N_dn]，其中 >=0 为正类、-3 为有效负 DN、-1 为无效或 padding。它是原始 GT 类别，作为分类目标
                dn_attn_mask : [N_dn, N_dn]，True=禁止 attention。它控制 DN query 之间的 attention 隔离
                valid_mask   : [B, N_dn]。它决定哪些 DN query 参与 DN classification loss，即它决定哪些 DN 位置确实有效
                dn_id_target : [B, N_dn] 或 None。它是 temporal DN 时的 GT instance ID 对应关系，即 Temporal DN 用它跨帧识别同一个 GT instance。
            '''

            # (2.2.b) 对齐 DN anchor 的状态维度【常见于 detection：DN target 可能不含速度等尾部状态，但 normal anchor 是 11 维；这里在最后一维补 0，只为了保证后续统一通过 anchor_encoder / refine】
            # 若 dn_anchor 的最后一维状态维度和普通 anchor 不一致，则给 dn_anchor 后面补零，使 dn_anchor 状态维度与普通 anchor 一致
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                # 计算还差多少维
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]

                # 给 dn_anchor 后面补零，使其 dn_anchor 状态维度与普通 anchor 一致
                dn_anchor = torch.cat(
                    [
                        dn_anchor,           # 原始 dn_anchor
                        dn_anchor.new_zeros( # 末尾补零
                            batch_size,       # batch size
                            num_dn_anchor,    # denoising anchor 数量
                            remain_state_dims # 需要补齐的状态维度
                        ),
                    ],
                    dim=-1, # 在最后一维拼接
                )

            # (2.2.c) 追加 DN queries：normal 在前，DN 永远在后【该前后位置约定会贯穿 attention mask、InstanceBank.update()、输出 split 和 DN cache】
            # (c.1) 将 denoising anchor 拼接到普通 anchor 后
            anchor = torch.cat([anchor, dn_anchor], dim=1) # anchor shape 从 [B, N, D] 变成 [B, N + N_dn, D]
            # (c.2) 将 denoising instance_feature 拼接到普通 instance_feature 后
            instance_feature = torch.cat(
                [
                    instance_feature,           # 拼接位置在前：普通 instance feature
                    instance_feature.new_zeros( # 拼接位置在后：DN instance feature，初始化为 0【这里 DN 的 feature 初始化为 0，也就是将 0 初始化的 DN feature 追加到 normal query 的尾部】
                        batch_size,                # batch size
                        num_dn_anchor,             # DN anchor 数量
                        instance_feature.shape[-1] # feature 维度
                    ),
                ],
                dim=1, # 在 instance 数量维拼接
            )
            # (c.3) 记录拼接后的 instance 总数，记录普通 instance 总数
            num_instance = instance_feature.shape[1]         # 记录拼接后的 instance 总数
            num_free_instance = num_instance - num_dn_anchor # 记录普通 learnable instance 数量
            '''
                因此最终拼接得到的 query 为 “普通 900 query 在前” 拼接 “DN query 在后”：
                    普通 900 query 部分是：
                        anchor：[B, 900, 11]
                        instance_feature：[B, 900, 256]
                    DN query 部分是：
                        dn_anchor：[B, num_dn_anchor, 11]，dn_anchor 由 get_dn_anchors() 生成且末尾补 0 以使状态维度与普通 anchor 保持统一【dn_anchor 是“接近 GT”的加噪 noisy 3D anchor】
                        dn_instance_feature：[B, num_dn_anchor, 256]，初始为全0
            '''

            # (2.3) 构造 DN attention mask，限制普通 query 和 DN query 的交互：True=禁止 attention，False=允许 attention；normal-normal 子块显式设为 False
            # (2.3.a) 构造 attention mask：其 shape 为 [num_instance, num_instance]，True 表示禁止 attention、False 表示允许 attention
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), # mask 形状
                dtype=torch.bool              # bool 类型
            )
            # (2.3.b) 普通 instance 之间允许相互 attention
            attn_mask[:num_free_instance, :num_free_instance] = False
            # (2.3.c) DN instance 内部使用 sampler 给出的 dn_attn_mask，以决定 DN instance 内部是否交互
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask


        # (3) 对当前 anchor 和上一帧 temp_anchor 分别进行编码，得到嵌入 anchor_embed 和 temp_anchor_embed【这俩嵌入是后续所有 attention 的位置 / 几何条件】
        # (3.a) 对当前 anchor 做位置编码：
        anchor_embed = self.anchor_encoder(anchor) # anchor_embed shape 通常是 [B, N, C]
        # (3.b) 对上一帧 temp_anchor 做位置编码：
        if temp_anchor is not None:                              # 若存在上一帧 anchor，
            temp_anchor_embed = self.anchor_encoder(temp_anchor) # 则对上一帧 anchor 做位置编码；
        else:                                                    # 若不存在上一帧 anchor，
            temp_anchor_embed = None                             # 则上一 anchor embedding 设为 None。
        '''
            当前 anchor：
                【detection】：SparseBox3DEncoder 将 3D box state 编成 [B, N, C]。
                【map】：SparsePoint3DEncoder 将 20×2=40 维 polyline 点坐标编成 [B, N, C]。
            
            上一帧 temp_anchor_embed 必须来自“已投影到当前坐标系”的历史 anchor，不能直接使用历史坐标系中的旧 anchor。
        '''

        # (4) 执行 decoder 操作流：
        # 先保存每个 refine 层输出的回归预测、分类预测、质量估计
        prediction = []     # 保存每个 refine 层输出的回归预测
        classification = [] # 保存每个 refine 层输出的分类预测
        quality = []        # 保存每个 refine 层输出的质量估计

        # 正式进入 decoder 层循环，按 operation_order 顺序执行每个 layer
        for i, op in enumerate(self.operation_order): # 按 operation_order 顺序执行每个 layer
            '''
                self.layers[i] 与 self.operation_order[i] 一一对齐。每遇到一个 refine，prediction / classification / quality 就新增“一个 decoder stage 的输出”。
                    (4.a) temp_gnn：当前 query 从历史 temporal instances 读取跨帧上下文。
                    (4.b) gnn：当前帧 query 之间做 self-attention，建模对象-对象或地图线-地图线关系。
                    (4.c) norm / ffn：做特征稳定化和通道内非线性变换。
                    (4.d) deformable：根据 anchor 几何生成关键点，从 6 相机、多尺度图像特征取样聚合。
                    (4.e) refine：更新 anchor，并输出 cls / quality；每个 refine 表示一层 decoder 结束。
                    (4.f) 单帧 decoder 结束后调用 InstanceBank.update()，把 temporal prefix 与当前 top-k normal instances 重组，供后续 temporal decoder 继续处理。
            '''
            if self.layers[i] is None: # 若当前 layer 是 None，则跳过
                continue               # 跳过

            # (4.a) 若当前操作是 temporal GNN：当前 query 从上一帧 temporal instances 读取跨帧上下文【即当前帧 instance 与上一帧 temporal instances 交互】
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,                                                              # 当前 layer index
                    instance_feature,                                               # query：当前帧 instance feature
                    temp_instance_feature,                                          # key  ：上一帧 instance feature
                    temp_instance_feature,                                          # value：上一帧 instance feature
                    query_pos=anchor_embed,                                         # 当前 anchor 位置编码
                    key_pos=temp_anchor_embed,                                      # 历史 anchor 位置编码
                    attn_mask=attn_mask if temp_instance_feature is None else None, # 若没有历史 instance，就使用 attn_mask；若有历史 instance，则不传这个 attn_mask
                )
                '''
                    (4.a) temporal GNN：
                            - 有历史时：Q 来自当前帧 instance [B,N,C]，K/V 来自历史缓存 temporal memory instance [B,N_temp,C]，实现当前 instance 和历史 instance 的跨帧 cross-attention 交互，输出更新后的当前 instance [B,N,C]
                            - 无历史时（即 temp_instance_feature 和 temp_anchor_embed 为 None 时）：key/value 为 None，底层 attention 模块 会走其 self-attention 回退逻辑，因此传入 DN mask 以保持 DN group 相互隔离
                            - Stage1 map 正常属于“无历史”情况；Stage2 map 可读取 T_map=33 条线。
                '''

            # (4.b) 若当前操作是当前帧 GNN：当前帧 instance 之间做 self-attention
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,                      # 当前 layer index
                    instance_feature,       # query：当前帧 instance feature # key 没传，attention 模块内部通常会默认 key=query
                    value=instance_feature, # value：当前帧 instance feature
                    query_pos=anchor_embed, # 当前 anchor 位置编码
                    attn_mask=attn_mask,    # attention mask，用于限制 DN query 和普通 query 的交互
                )
                '''
                    (4.b) 当前帧 GNN：当前帧 instance 之间做 self-attention:
                            - Query/Value：当前 instance [B,N,C]
                            - 输出：更新后的当前 instance [B,N,C]
                            注：当前帧 instance 包括 normal query 和 DN query，因此实际上是 normal query 之间做交互建立关系、可能的 DN query 之间做交互建立关系，因此传入 attn_mask：attn_mask 只在 DN 存在时非 None，用来阻断不应发生的 DN 信息交换
                            论文对应 — SparseDrive 第 3.2 节/图 4：当前 instance 之间的 self-attention。
                '''

            # (4.c) 若当前操作是 norm 或 ffn：直接调用对应层，不改变 query 数量与 anchor 几何，只更新 feature 表示
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
                '''
                    - norm：[B,N,C] -> [B,N,C]
                    - detection 任务的 FFN：它接收 DeformableFeatureAggregation(residual_mode="cat") 后的 [B,N,2C]，并通过 AsymmetricFFN 返回 [B,N,C]
                    论文对应 — SparseDrive 第 3.2 节：FFN 是非时序和时序 decoder 的子模块                
                '''

            # (4.d) 若当前操作是 Deformable Feature Aggregation：Deformable aggregation 在每个 3D anchor 锚框周围生成固定或可学习的关键点，并将关键点投影到多相机多尺度 feature_maps 中采样图像特征，并将采样图像特征融合进对应 query，这是“anchor 几何 → 图像证据”的桥梁
            # 【"deformable" 是真正让 instance_feature 从 0 变成有意义特征的关键】deformable 根据当前 anchor 指定的 3D 位置，从六个相机、多尺度图像 feature map 中取样、聚合视觉证据：
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature, # 当前 instance feature
                    anchor,           # 当前 anchor
                    anchor_embed,     # anchor embedding
                    feature_maps,     # 图像特征
                    metas,            # 元信息，例如 projection_mat、image_wh 等
                )
                '''
                    (4.d) 若当前操作是 Deformable Feature Aggregation：Deformable aggregation 在每个 3D anchor 锚框周围生成固定或可学习的关键点，并将关键点投影到多相机多尺度 feature_maps 中采样图像特征，并将采样图像特征融合进对应 query，这是“anchor 几何 → 图像证据”的桥梁
                            输入：instance_feature [B,N,C]、anchor [B,N,D_box]、anchor_embed [B,N,C] 与 multi-view feature_maps
                            detection 任务的输出为 [B,N,2C]，因为 residual_mode="cat" 会拼接采样 feature 与输入 feature
                            论文对应 — SparseDrive 第 3.2 节：在 anchor 周围生成固定/可学习 keypoint，将其投影到 feature map 后采样，再把采样 feature 融合进 instance feature。
                            
                            回顾2个任务的 anchor：
                                - detection 任务：从 3D box 产生中心、边缘等 3D keypoints，再投影到图像 
                                - map 任务：从 polyline 的 20 个地面点及其多个高度采样点生成 keypoints
                                - 两个任务最终都把视觉证据写回 instance_feature
                '''

            # (4.e) 若当前操作是 Refine：Refinement/classification 输出层，refine layer 根据 instance_feature + anchor_embed 更新 anchor，并输出预测的 anchor 的 分类 cls 和质量估计 qt
            elif op == "refine":
                anchor, cls, qt = self.layers[i]( # 见 detection3d_blocks.py 的 SparseBox3DRefinementModule 类
                    instance_feature,             # 当前 instance feature
                    anchor,                       # 当前 anchor
                    anchor_embed,                 # anchor 位置编码
                    time_interval=time_interval,  # 时间间隔，对带速度的 3D box 更新很重要
                    return_cls=True,              # 要求返回分类结果
                )
                prediction.append(anchor)  # 保存当前 decoder 层的 anchor 预测
                classification.append(cls) # 保存当前 decoder 层的分类预测
                quality.append(qt)         # 保存当前 decoder 层的质量估计
                '''
                    (4.e) refine layer 根据 instance_feature + anchor_embed 更新 anchor，并输出预测的 anchor 的 分类 cls 和质量估计 qt：
                            输入 instance_feature：[B, N=900, C]
                            输出：
                                anchor：[B, N=900, D_box=11]
                                cls   ：[B, N=900, K=10]
                                qt    ：开启时 qt 为 [B, N=900, 2]，其中 2 对应 centerness 和 yawness，否则 qt 为 None
                            论文对应 — SparseDrive 第 3.2 节：输出层 refined box，并预测分类分数和 anchor offset。可选 quality branch 继承自 Sparse4D v3 第 3.2 节。
                            
                            detection 任务与 map 任务的 refine 层输出：
                                - detection refine 输出新的 3D box state、10 类 logits、可选 quality
                                - map refine 输出新的 40 维 polyline 点序列、3 类 logits，quality 通常为 None
                                - 这里返回的 anchor 会覆盖旧 anchor，因此 refine 后必须重新编码 anchor_embed
                '''

                # (4.f) “第1个decoder(单帧发现阶段)” → “第2个decoder(时序精修阶段)” 的关键切换点：只在第 self.num_single_frame_decoder=1 次 refine 后才执行一次 update()：
                #  detection：历史 T=600 + 当前 top-(900-600)=300，仍保持 N=900。
                #  map Stage1：cache 为空，update() 直接返回当前 100 条线，不发生重组。
                #  map Stage2：历史 T=33 + 当前 top-(100-33)=67，仍保持 N=100。
                #  DN query 始终放在 normal N 个 query 后面；InstanceBank.update() 内部会暂时拆开 DN，重组 normal，再把 DN 拼回尾部。
                if len(prediction) == self.num_single_frame_decoder: # 若当前已经完成了第1个 decoder 部分
                    '''
                        (f.0) 《if len(prediction) == self.num_single_frame_decoder》：第一个非时序 decoder 与后续时序 decoder 的边界，即若当前已经完成了单帧 decoder 部分：
                                    若历史帧有效，InstanceBank 保留 N_temp=600 个投影后的历史 instance，再以置信度最高的 300 个当前 instance 填满剩余位置，
                                    总 shape 仍为 [B,900,*]，此时有 len(prediction) == self.num_single_frame_decoder。
                    '''

                    # (f.1) 更新 instance_bank：选择高置信度 instance，作为后续 temporal modeling 的基础
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, # 当前 instance feature
                        anchor,           # 当前 anchor
                        cls               # 当前分类结果
                    )

                    # (f.2) 若当前存在 DN metas，并且启用了 temporal DN groups，并且有 DN id target，则更新 temporal denoising 信息
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        # 更新 temporal denoising 信息
                        (
                            instance_feature,   # 更新后的 instance feature
                            anchor,             # 更新后的 anchor
                            temp_dn_reg_target, # temporal DN 回归目标
                            temp_dn_cls_target, # temporal DN 分类目标
                            temp_valid_mask,    # temporal DN 有效 mask
                            dn_id_target,       # 更新后的 DN id target
                        ) = self.sampler.update_dn(
                            instance_feature,              # 当前 instance feature
                            anchor,                        # 当前 anchor
                            dn_reg_target,                 # 原始 DN 回归目标
                            dn_cls_target,                 # 原始 DN 分类目标
                            valid_mask,                    # 原始 valid mask
                            dn_id_target,                  # DN id target
                            self.instance_bank.num_anchor, # instance_bank 中普通 anchor 数量
                            self.instance_bank.mask,       # instance_bank 中的 mask
                        )

                # (4.g) 重新编码更新后的 anchor：这是必要步骤：下一次 gnn / temp_gnn 使用的 geometry embedding 必须与最新 refined anchor 对齐，而不是与进入本层前的旧 anchor 对齐
                # (g.1) refine 后 anchor 已经更新，需要重新编码 anchor 位置
                anchor_embed = self.anchor_encoder(anchor)

                # (g.2) temporal prefix 的几何编码同步更新：update() 后 normal slots 的前 T 位就是历史传播实例；因此后续 temp_gnn 的 key_pos 必须取 anchor_embed 的前 T 位，不能继续使用 get() 时的旧 temp_anchor_embed：
                if (len(prediction) > self.num_single_frame_decoder and temp_anchor_embed is not None): # 若已经超过单帧 decoder 阶段，并且存在历史 anchor embedding
                    # InstanceBank.update() 后，前 num_temp_instances 个槽位是与历史对齐的 instance，因此这里从当前 anchor_embed 中取前 num_temp_instances 个作为 temporal anchor embedding，用于后续 temporal GNN
                    temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]

            # 若 operation_order 中出现未知操作，则直接报错
            else:
                raise NotImplementedError(f"{op} is not supported.")


        # (5) 若 DN 存在，则将 “normal query 输出” 与 “DN query 输出” 严格拆开：
        '''
            为要什么将 “normal query 输出” 与 “DN query 输出” 严格拆开的原因：
                1. 约定：normal queries 永远位于 [:num_free_instance]，DN queries 永远位于 [num_free_instance:]。
                2. 普通 loss / post_process / temporal cache 只能使用 normal 部分。
                3. DN 部分只服务训练期辅助 loss 和 temporal-DN cache，绝不能当作真实预测。

            仅在 (4) 中全部 decoder 层执行完成后，才拆分 normal 与 DN prediction，对每一个 decoder 输出：
                normal classification、normal prediction 为 [B,N_free,...]
                DN classification、DN prediction 为 [B,N_dn,...]
        '''
        # 初始化输出字典
        output = {}

        # 若使用了 denoising training，则需要把普通预测和 DN 预测拆开
        if dn_metas is not None:
            # (5.a) 把普通预测和 DN 预测拆开：
            dn_classification = [x[:, num_free_instance:] for x in classification]           # 从 classification 中切出 DN 部分【普通 instance 在前 num_free_instance 个，DN instance 在后 num_dn_anchor 个】
            classification = [x[:, :num_free_instance] for x in classification]              # 保留普通 instance 的分类输出
            dn_prediction = [x[:, num_free_instance:] for x in prediction]                   # 从 prediction 中切出 DN 部分
            prediction = [x[:, :num_free_instance] for x in prediction]                      # 保留普通 instance 的预测输出
            quality = [x[:, :num_free_instance] if x is not None else None for x in quality] # quality 只保留普通 instance 部分

            # (5.b) 把 DN 相关输出写入 output：
            output.update(
                {
                    "dn_prediction": dn_prediction,         # DN 回归预测
                    "dn_classification": dn_classification, # DN 分类预测
                    "dn_reg_target": dn_reg_target,         # DN 回归目标
                    "dn_cls_target": dn_cls_target,         # DN 分类目标
                    "dn_valid_mask": valid_mask,            # DN 有效 mask
                }
            )

            # (5.c) 若存在 temporal DN target，则同理把 temporal DN 相关输出写入 output
            if temp_dn_reg_target is not None:
                # 把 temporal DN 相关输出写入 output：
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target, # temporal DN 回归目标
                        "temp_dn_cls_target": temp_dn_cls_target, # temporal DN 分类目标
                        "temp_dn_valid_mask": temp_valid_mask,    # temporal DN 有效 mask
                        "dn_id_target": dn_id_target,             # DN id target
                    }
                )

                # 后续缓存：
                dn_cls_target = temp_dn_cls_target # 后续缓存 DN 时使用 temporal DN 分类目标
                valid_mask = temp_valid_mask       # 后续缓存 DN 时使用 temporal valid mask
            dn_instance_feature = instance_feature[:, num_free_instance:] # 取出最终 instance_feature 中的 DN 部分
            dn_anchor = anchor[:, num_free_instance:]                     # 取出最终 anchor 中的 DN 部分
            instance_feature = instance_feature[:, :num_free_instance]    # 普通 instance feature 只保留前 num_free_instance 个
            anchor_embed = anchor_embed[:, :num_free_instance]            # 普通 anchor_embed 只保留前 num_free_instance 个
            anchor = anchor[:, :num_free_instance]                        # 普通 anchor 只保留前 num_free_instance 个
            cls = cls[:, :num_free_instance]                              # 普通 cls 只保留前 num_free_instance 个

            # (5.d) 缓存 DN 信息，供下一帧 temporal denoising 使用
            self.sampler.cache_dn(
                dn_instance_feature, # DN instance feature
                dn_anchor,           # DN anchor
                dn_cls_target,       # DN 分类目标
                valid_mask,          # DN 有效 mask
                dn_id_target,        # DN instance id
            )


        # (6) 写入普通预测输出 normal outputs：这里的列表长度 = refine 次数 = decoder stage 数【classification[j] / prediction[j] / quality[j] 三者严格对应第 j 次 refine】
        output.update(
            {
                "classification": classification,     # 每个 refine 层的分类输出 list
                "prediction": prediction,             # 每个 refine 层的回归输出 list
                "quality": quality,                   # 每个 refine 层的质量估计 list
                "instance_feature": instance_feature, # 最终 instance feature
                "anchor_embed": anchor_embed,         # 最终 anchor embedding
            }
        )


        # (7) 将当前帧 top-N_temp 个最终 normal instance 写入缓存 temporal memory【只缓存 normal instances】，供后续帧 temporal modeling 使用
        self.instance_bank.cache(
            instance_feature, # 当前帧 instance feature，[B,N_free,C]，缓存后 [B,N_temp,C]
            anchor,           # 当前帧 anchor，[B,N_free,D_box]，缓存后 [B,N_temp,D_box]
            cls,              # 当前帧分类结果，[B,N_free,K]
            metas,            # 元信息
            feature_maps      # 图像特征
        )
        '''
            detection：在最后一层 cls 中按分数选 top-600，保存 feature / box anchor / score。
            map Stage2：按分数选 top-33，保存 feature / polyline anchor / score。
            map Stage1：num_temp_instances=0，InstanceBank.cache() 立即返回，因此没有跨帧 map memory。

            缓存后 shape：[B,N_temp,C] 和 [B,N_temp,D_box]。
            cache 内部 detach，故不会把梯度反传穿过上一帧。
        '''


        # (8) 可选稀疏跟踪 sparse tracking ID：
        '''
            仅 detection 任务会使用 sparse tracking ID
            map_head 的 with_instance_id=False，因此不会进入该分支：map 线的跨帧传播只靠 InstanceBank 的 temporal feature/anchor，不给每条 polyline 分配 tracking ID。        
        '''
        if self.with_instance_id: # 若需要 instance id
            # (8.a) 根据分类结果和 anchor 生成或更新 instance id，其中 decoder.score_threshold 是得分阈值
            instance_id = self.instance_bank.get_instance_id(
                cls,                         # 分类 logits
                anchor,                      # anchor
                self.decoder.score_threshold # 分数阈值
            )

            # (8.b) 将 instance_id 加入输出字典【instance_id 的 shape 为 [B,N_free]，其中高置信度 query 被分配稳定 ID，无效或未锁定 query 的 ID 为 -1】
            output["instance_id"] = instance_id

        # (9) 返回输出字典
        '''
            核心字段及其形状：
                Detection 任务：
                    output = {                                   # 检测任务头的一个 Query 表示一个候选动态目标
                        "classification": 6 × [B, 900, 10],          # 6个 refine 层的每个检测框 Query 的 10 类类别 logits
                        "prediction":     6 × [B, 900, 11],          # 6个 refine 层的每个检测框 Query 的 11 维 3D 框状态
                        "quality":        6 × ([B, 900, 2] or None), # 6个 refine 层的每个检测框 Query 的 2 个质量预测值 centerness 和 yawness
                        "instance_feature":   [B, 900, 256],         # 最终 900 个检测框的语义特征
                        "anchor_embed":       [B, 900, 256],         # 最终 900 个检测框的几何位置编码
                        "instance_id":        [B, 900],              # 每个 Query 的跨帧跟踪 ID

                        # ---------- DN：仅训练阶段且开启 DN 时存在 ----------
                        # DN(Denoising)：由 GT 3D box 加噪得到额外 DN Query，
                        # 与普通 900 个 Query 共用 Decoder，训练其恢复原始 GT；推理和后处理阶段不使用 DN。
                        # N_dn 为当前 batch 的 DN Query 数，开启正/负 DN 时通常 N_dn = 2 × num_dn_groups × max_dn_gt。
                        "dn_classification": 6 × [B, N_dn, 10],      # 6个 refine 层的 DN Query 类别 logits
                        "dn_prediction":     6 × [B, N_dn, 11],      # 6个 refine 层的 DN Query 3D 框预测
                        "dn_cls_target":         [B, N_dn],          # 普通 DN 分类目标
                        "dn_reg_target":         [B, N_dn, 11],      # 普通 DN 回归目标，即需要恢复的 GT 3D box
                        "dn_valid_mask":         [B, N_dn],          # DN slot 是否有效，过滤 padding / 无效 DN

                        # ---------- Temporal DN：仅开启时序 DN 时存在 ----------
                        "temp_dn_cls_target":    [B, N_dn],          # Temporal Decoder 使用的 DN 分类目标
                        "temp_dn_reg_target":    [B, N_dn, 11],      # Temporal Decoder 使用的当前帧同一 GT 的回归目标
                        "temp_dn_valid_mask":    [B, N_dn],          # Temporal DN slot 是否有效
                        "dn_id_target":          [B, N_dn],          # GT instance ID，用于跨帧匹配同一个 GT
                    }
                
                Mapping 任务：
                    output = {                           # 地图任务头的一个 Query 表示一条候选地图 polyline
                        "classification": 6 × [B, 100, 3],   # 6个 refine 层的每个地图 polyline Query 的 3 类类别 logits
                        "prediction":     6 × [B, 100, 40],  # 6个 refine 层的每个地图 polyline Query 的 20 个点的 (x,y) 二维坐标预测
                        "quality":        6 × None,          # 地图头通常不启用质量分支
                        "instance_feature":   [B, 100, 256], # 最终 100 条地图 polyline 的语义特征
                        "anchor_embed":       [B, 100, 256], # 最终 100 条地图 polyline 的几何位置编码
                        # 没有 instance_id
                    }
        '''
        return output

    # 5. loss 函数用于训练阶段计算 Sparse4DHead 的损失，包括普通预测 loss 和可选 denoising loss
    # force_fp32() 用于确保 loss 计算在 fp32 下进行，提升数值稳定性；此外，注意这里 apply_to=("model_outs") 实际是字符串、不是 tuple，更标准写法是 apply_to=("model_outs",)
    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        '''
            【loss() 的四段逻辑】
                (1) 读取每个 refine stage 的 normal cls / reg / quality 输出。
                (2) 对每个 stage：sampler 匹配 GT → 构造正样本 → 分类 loss + 回归 loss。
                (3) 若没有 DN output，normal loss 至此结束。
                (4) 若有 DN output：按 DN valid/positive mask 额外计算每一层的 DN loss。
            
            本函数对 det / map 的共性：循环、flatten、mask、loss 字典命名完全相同。
            差异由 self.sampler / self.loss_reg / self.reg_weights / gt key 决定：detection：GT box + 3D box loss；map：GT polyline + line loss。        
        '''
        # 初始化 loss 输出字典
        output = {}
        
        # ===================== 一、计算 prediction losses ======================
        # 1. 取出所有 decoder stage 的预测序列 cls_scores、reg_preds、quality【如 6 个 refine，就分别有长度为 6 的 cls_scores / reg_preds / quality】
        cls_scores = model_outs["classification"] # 取出每个 decoder refine 层的分类输出 cls_scores：长度为 L 的 [B,N_free,K] tensor 列表
        reg_preds = model_outs["prediction"]      # 取出每个 decoder refine 层的回归输出 reg_preds ：长度为 L 的 [B,N_free,D_box] tensor 列表
        quality = model_outs["quality"]           # 取出每个 decoder refine 层的质量估计 quality   ：长度为 L 的 [B,N_free,2] tensor 列表或 None

        # 2. 遍历每个 decoder 层的输出，计算 normal prediction loss
        '''
            (2.a) 截取需要监督的回归维度并通过 sampler 与 GT 匹配。
            (2.b) 根据 reg_target 构造正样本 mask，并在多卡上统计 num_pos。
            (2.c) 可选用 cls_threshold_to_reg 进一步过滤“分类信心不足”的正样本回归。
            (2.d) flatten + mask，得到真正交给 loss 的一维样本集合。
            (2.e) 分别计算 cls_loss 和 reg_loss，按 task_prefix / decoder_idx 写入 output。
        '''
        for decoder_idx, (cls, reg, qt) in enumerate(zip(cls_scores, reg_preds, quality)): # zip(cls_scores, reg_preds, quality)：cls_scores、reg_preds、quality 都是 list，每个元素对应一个 refine 层的输出
            # (1) 对齐当前任务的回归状态维度，截取需要监督的回归维度并通过 sampler 与 GT 匹配：
            # (1.a) 只有 D_reg 个通道进入这个主回归目标（即只取需要计算 loss 的回归维度，len(self.reg_weights) 决定监督多少维）：
            reg = reg[..., : len(self.reg_weights)] # detection 常用前 10 维监督；map 配置的 self.reg_weights 长度为 40，对应 20 个 (x,y) 采样点

            # (1.b)【核心：匈牙利匹配构建 target】调用 sampler 进行预测和 GT 的匈牙利匹配，返回分类目标、回归目标和每个维度的回归权重
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,                   # 分类预测
                reg,                   # 回归预测
                data[self.gt_cls_key], # GT 分类标签
                data[self.gt_reg_key], # GT 回归目标
            )

            # (1.c) GT 回归目标也只保留需要监督的维度
            reg_target = reg_target[..., : len(self.reg_weights)]
            reg_target_full = reg_target.clone() # 复制一份完整 reg_target，但当前代码后面没有使用 reg_target_full，属于冗余变量

            # (2) 构造 normal 正样本 mask【此实现约定：回归 target 的全部维度均为 0 时，不将它作为有效回归正样本】
            # 生成正样本 query 的 mask，shape 为 [B,N_free]，若一个 reg_target 所有维度都是 0，就认为不是有效正样本
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))         # 生成正样本 query 的 mask
            mask_valid = mask.clone()                                            # 复制一份 mask，但当前代码后面没有使用 mask_valid，属于冗余变量
            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0) # 计算正样本数量，其中reduce_mean() 用于多 GPU 平均，max(..., 1.0) 防止除以 0

            # (3) 可选的“分类置信度门控回归”【它不影响 cls_loss，仅决定哪些匹配正样本继续贡献 reg_loss】
            if self.cls_threshold_to_reg > 0: # 若启用了分类阈值筛选回归样本
                threshold = self.cls_threshold_to_reg # 取阈值

                # 只有 原本是正样本 且 最大分类分数 sigmoid 超过阈值后，才参与回归 loss
                mask = torch.logical_and(
                    mask,                                        # 原本的正样本 mask
                    cls.max(dim=-1).values.sigmoid() > threshold # 当前 query 的最大类别概率是否大于阈值
                )

            # (4)【核心：计算分类 loss】将 batch/query 两维拉平，再用同一个 mask 同步筛选正样本的 reg、target、weight、quality，并计算分类 loss
            # (4.a.1) 拉平 cls_target，计算分类 loss：
            cls = cls.flatten(end_dim=1)                                  # 展平 batch/query 轴，即将分类预测从 [B, N, C] 拉平成 [B*N, C]
            cls_target = cls_target.flatten(end_dim=1)                    # 将分类目标从 [B, N] 拉平成 [B*N]
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos) # 计算分类 loss【self.loss_cls() 采用的是 FocalLoss】

            # (4.b) 将 mask 从 [B, N] 拉平成 [B*N]
            mask = mask.reshape(-1)

            # (4.c) 拉平 reg_target、reg、reg_weights，并只保留 mask=True 的正样本 [N_pos,D_reg]
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights) # 将 sampler 给出的 reg_weights 和配置中的 self.reg_weights 相乘，其中 self.reg_weights 控制不同回归维度的重要性
            reg_target = reg_target.flatten(end_dim=1)[mask]             # 拉平 reg_target，并只保留 mask=True 的正样本 [N_pos,D_reg]
            reg = reg.flatten(end_dim=1)[mask]                           # 拉平 reg，并只保留 mask=True 的正样本 [N_pos,D_reg]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]           # 拉平 reg_weights，并只保留 mask=True 的正样本 [N_pos,D_reg]
            reg_target = torch.where(                                    # 将 reg_target 中的 NaN 替换为 0，避免 loss 计算出现 NaN
                reg_target.isnan(),  # 找出 NaN 位置
                reg.new_tensor(0.0), # NaN 替换为 0
                reg_target           # 非 NaN 保持原值
            )

            # (4.a.2) cls_target 也只保留参与回归 loss 的正样本，后续 loss_reg 可能需要类别信息，如 barrier 方向反转等
            cls_target = cls_target[mask]

            # (4.d) 拉平 qt，并只保留参与回归 loss 的正样本 
            if qt is not None:                   # 若质量估计 qt 不为空
                qt = qt.flatten(end_dim=1)[mask] # 则拉平 qt，并只保留参与回归 loss 的正样本 [N_pos,2]


            # (5)【核心】计算回归 loss 和质量估计 loss：
            # detection 的 loss_reg 可利用 cls_target 与 quality 处理 yawness、centerness；map 的 SparseLineLoss 则比较整条 polyline 的点序列几何误差。
            # SparseBox3DLoss 包括 L1 box 项，以及在配置开启时的 Sparse4D v3 quality-estimation 项。
            reg_loss = self.loss_reg(          # self.loss_reg()：Detection 任务见 detection3d/losses.py 的 SparseBox3DLoss 类，Map 任务见 map/losses.py 的 SparseLineLoss 类
                reg,                           # 回归预测
                reg_target,                    # 回归 GT
                weight=reg_weights,            # 回归维度权重
                avg_factor=num_pos,            # 平均因子
                prefix=f"{self.task_prefix}_", # loss 名称前缀
                suffix=f"_{decoder_idx}",      # loss 名称后缀，标记 decoder 层
                quality=qt,                    # 质量估计
                cls_target=cls_target,         # 类别目标
            )
            '''
                最终生成：
                    det_loss_box_0
                    det_loss_box_1
                    ...
                    det_loss_box_5
                这里的 weight 综合了三类信息：
                    GT 对应维度是否有效；
                    Sampler 给出的样本或维度权重；
                    配置文件中的全局 reg_weights。            
            '''


            # (6) 保存 loss：
            # (6.a) 保存当前 decoder 层分类 loss
            output[f"{self.task_prefix}_loss_cls_{decoder_idx}"] = cls_loss
            '''
                每一层都会计算分类损失 cls_loss，对应损失名称：
                    det_loss_cls_0
                    det_loss_cls_1
                    ...
                    det_loss_cls_5
                它监督每个检测 Query 属于：
                    car
                    truck
                    bus
                    trailer
                    construction_vehicle
                    pedestrian
                    motorcycle
                    bicycle
                    traffic_cone
                    barrier
                中的哪个类别，或者属于背景。
                实际使用哪种分类损失由配置文件中的 loss_cls 决定，SparseDrive 配置文件中实际使用的是基于 Sigmoid 的 Focal Loss。
            '''

            # (6.b) 合并回归 loss 字典
            output.update(reg_loss)


        # 3. 普通分支提前返回：推理阶段和未启用 DN 的训练阶段都会走这里
        if "dn_prediction" not in model_outs: # 若模型输出中没有 denoising prediction，
            return output                     # 则直接返回普通 prediction losses



        # ===================== 二、计算 denoising losses【若存在 denoising 输出，则继续计算 DN loss】 ======================
        # 4. DN loss 分支：DN queries 不重新走 Hungarian matching，而是直接使用构造 DN 时，已知的 dn_cls_target / dn_reg_target，因此提供更稳定、更直接的辅助监督。
        # (4.a) 取出 DN 分类预测、 DN 回归预测：
        dn_cls_scores = model_outs["dn_classification"] # 取出 DN 分类预测       
        dn_reg_preds = model_outs["dn_prediction"]      # 取出 DN 回归预测

        # (4.b) 准备 DN loss 所需的 mask、target、权重、正样本数量：
        (
            dn_valid_mask, # DN 有效 mask
            dn_cls_target, # DN 分类目标
            dn_reg_target, # DN 回归目标
            dn_pos_mask,   # DN 正样本 mask
            reg_weights,   # 回归权重
            num_dn_pos,    # DN 正样本数量
        ) = self.prepare_for_dn_loss(model_outs)

        # (4.c) 遍历每个 decoder 层的 DN 输出计算 DN 分类 loss 和回归 loss：当到达单帧 decoder 结束位置且存在 temporal DN target 时，切换为 temporal DN target，使后续时序 decoder 使用与重组后 query 对齐的新 target
        for decoder_idx, (cls, reg) in enumerate(zip(dn_cls_scores, dn_reg_preds)): # 遍历每个 decoder 层的 DN 输出 # zip(dn_cls_scores, dn_reg_preds) 表示 DN 分类和回归输出逐层对应
            # (c.1) 准备 temporal DN loss 所需的 mask、target、权重、正样本数量：
            if ("temp_dn_valid_mask" in model_outs and decoder_idx == self.num_single_frame_decoder): # 若模型输出中有 temporal DN mask 且当前 decoder_idx 等于单帧 decoder 层数
                # 切换到 temporal DN loss 所需的 target
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            # (c.2) 计算 DN 分类 loss
            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask], # 先拉平成 [B*N_dn, C]，再只取有效 DN query
                dn_cls_target,                         # DN 分类 target
                avg_factor=num_dn_pos,                 # 平均因子
            )

            # (c.3) 计算 DN 回归 loss
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][..., : len(self.reg_weights)], # 先拉平并取有效 DN query，再根据 dn_pos_mask 只保留正样本，最后只取需要监督的回归维度
                dn_reg_target,                                                                    # DN 回归目标
                avg_factor=num_dn_pos,                                                            # 平均因子
                weight=reg_weights,                                                               # 回归权重
                prefix=f"{self.task_prefix}_",                                                    # loss 名称前缀
                suffix=f"_dn_{decoder_idx}",                                                      # loss 名称后缀，标记 DN 和 decoder 层
            )

            # (c.4) 保存当前 decoder 层 DN 分类 loss
            output[f"{self.task_prefix}_loss_cls_dn_{decoder_idx}"] = cls_loss

            # (c.5) 合并 DN 回归 loss
            output.update(reg_loss)

        # 5. 返回全部 loss
        return output

    # 6. 整理 denoising loss 需要的 mask 和 target：
    def prepare_for_dn_loss(self, model_outs, prefix=""):
        '''
            这个函数专门整理 denoising loss 需要的 mask 和 target
                prefix="" 时处理普通 DN
                prefix="temp_" 时处理 temporal DN   

            【prepare_for_dn_loss() 的五步】
                (1) 用 prefix 选择 normal DN 或 temporal DN 的字段来源。
                (2) flatten valid mask，并删去 padding / 无效 DN slots。
                (3) 从有效 slots 中取 cls / reg targets。
                (4) 从 cls target 得到 DN 正样本 mask；仅正样本参与回归。
                (5) 为每个正样本复制回归维度权重，并统计跨卡平均因子。     
        '''

        # 展平前：dn_valid_mask 为 [B,N_dn]，dn_cls_target 为 [B,N_dn]，dn_reg_target 为 [B,N_dn,D_box]
        # 经过 valid 过滤后，dn_cls_target 为 [N_valid]，dn_reg_target 为 [N_valid,D_reg]。
        # 正 DN 回归目标：[N_dn_pos,D_reg]

        # (a) 取出 DN 有效 mask，并把 [B, N_dn] 拉平成 [B*N_dn]
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)

        # (b) 取出 DN 分类目标：先从 [B, N_dn] 拉平成 [B*N_dn]，再根据 dn_valid_mask 只保留有效 DN query
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[dn_valid_mask]

        # (c.1) 取出 DN 回归目标：先拉平，再取有效 DN query，最后只保留需要监督的回归维度
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[dn_valid_mask][..., : len(self.reg_weights)]

        # (d) DN 正样本 mask：分类目标 >= 0 认为是正样本，分类目标为负数通常代表 ignore 或 background
        dn_pos_mask = dn_cls_target >= 0 # 经过 valid 过滤后，dn_cls_target 为 [N_valid]，dn_reg_target 为 [N_valid,D_reg]

        # (c.2) 回归目标只保留正样本
        dn_reg_target = dn_reg_target[dn_pos_mask] # 正 DN 回归目标：[N_dn_pos,D_reg]

        # (e) 构造回归权重：self.reg_weights shape 是 [D]，这里扩展成 [num_pos, D]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], # 每个正样本一份权重
            1                       # 第二维保持 1 份 self.reg_weights
        )

        # (f) 计算 DN 正样本数量
        num_dn_pos = max(reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)), 1.0) # 使用 dn_valid_mask 的数量作为平均因子，reduce_mean() 用于分布式多卡平均，max(..., 1.0) 防止除以 0

        # 返回 DN loss 需要的所有变量
        return (
            dn_valid_mask, # DN 有效 mask
            dn_cls_target, # DN 分类目标
            dn_reg_target, # DN 回归目标
            dn_pos_mask,   # DN 正样本 mask
            reg_weights,   # 回归权重
            num_dn_pos,    # DN 正样本数量
        )

    # 7. 后处理函数：即调用 decoder 将最终 decoder 层的输出分类、回归、instance_id、quality 解码成最终结果（用于评测或推理）
    # post_process() 也强制使用 fp32【force_fp32() 用于确保 loss 计算在 fp32 下进行，提升数值稳定性】；此外，注意这里 apply_to=("model_outs") 同样不是标准 tuple 写法，更标准写法是 apply_to=("model_outs",)
    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        '''
            decoder.decode() 负责把网络内部表示转换成评估/提交使用的最终格式：
            【detection】：解码为 3D boxes、score、label、tracking id 等。
            【map】：解码为 vectorized polylines、score、map class；没有 instance_id。
            最终层输入 shape：classification [B,N_free,K]、prediction
            decoder 会执行 sigmoid/top-k，并解码 log-size/sin-cos-yaw box。
        '''
        return self.decoder.decode(
            model_outs["classification"],  # 所有 decoder 层的分类输出，6 × [B, 900, 10]
            model_outs["prediction"],      # 所有 decoder 层的回归输出，6 × [B, 900, 11]
            model_outs.get("instance_id"), # instance id，可能不存在，[B, 900]
            model_outs.get("quality"),     # quality 质量估计，可能不存在，6 × ([B, 900, 2] 或 None)
            output_idx=output_idx,         # 使用哪个 decoder 层的输出，默认 output_idx=-1 表示只使用最后一次 refine 的结果
        )
