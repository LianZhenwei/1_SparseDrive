# =============================================================================
# losses_lzw.py —— SparseDrive 3D box + quality 联合损失（层次化注释版）
# =============================================================================
#
# 【本文件在训练链中的位置】
#   Sparse4DHead.loss()
#       (1) 对每个 decoder refine stage：Hungarian matching 得到正样本 query；
#       (2) 计算分类 loss（由 head 中的 self.loss_cls 负责）；
#       (3) 对正样本调用 self.loss_reg(...)；
#             └─ SparseBox3DLoss.forward() 【本文件核心】
#                  (a) box regression loss；
#                  (b) centerness quality loss（可选）；
#                  (c) yawness quality loss（可选）。
#
# 【关键理解】
#   - 分类 loss 不在本类中：它由 Sparse4DHead.loss() 单独算。
#   - 本类只接收已经 Hungarian 匹配好的正样本 box；
#     因此输入第一维通常是 num_pos，而不是全部 N=900 个 query。
#   - quality 不是没有监督的“参考分数”：
#       centerness 与 yawness 都会在这里构造 target 并计算 loss。
#   - 推理阶段 decoder_lzw.py 会使用 centerness 做 final score 重标定；
#     yawness 在当前 decoder 中没有直接乘进最终 score，但它仍作为辅助监督存在。
#
# 【典型输入 shape】
#   P = 当前 decoder stage Hungarian 匹配到的正样本总数（跨 batch flatten 后）
#   D = 被监督的 box state 维度（Detection 常见 D=10）
#   box / box_target / weight : [P, D]
#   quality                   : [P, 2]，通常 [centerness_logit, yawness_logit]
#   cls_target                : [P]
# =============================================================================

import torch          # 导入 PyTorch，用于 Tensor 计算，例如 cosine_similarity、where、norm、exp 等
import torch.nn as nn # 导入 PyTorch 神经网络模块，SparseBox3DLoss 继承自 nn.Module
from mmcv.utils import build_from_cfg   # 从 MMCV 中导入 build_from_cfg，用于根据配置字典和注册表构建 loss 模块
from mmdet.models.builder import LOSSES # 从 mmdet 中导入 LOSSES 注册表：SparseBox3DLoss 会注册到 LOSSES 里，配置文件中 loss_reg=dict(type="SparseBox3DLoss", ...) 时就会构建这个类


# 导入 box3d.py 中定义的 box 状态索引常量
# 例如：
# X, Y, Z
# SIN_YAW, COS_YAW
# CNS, YNS
# 这些常量用于从 box 或 quality 的最后一维中取出对应字段
from projects.mmdet3d_plugin.core.box3d import *


# SparseBox3DLoss：3D box 回归损失 + 可选质量估计损失
@LOSSES.register_module() # 将 SparseBox3DLoss 注册到 mmdet 的 LOSSES 注册表
class SparseBox3DLoss(nn.Module):
    '''
        【类作用】
            给已经与 GT 匹配的正样本计算：
                (1) box loss：预测 3D box 是否贴近 GT box；
                (2) centerness loss：预测框中心是否贴近 GT 中心；
                (3) yawness loss：预测框朝向是否大致与 GT 同向；
                (4) 可选处理“正反方向等价”的类别，例如某些 barrier。

        【输出字典示例】
            {
                "det_loss_box_0": ...,
                "det_loss_cns_0": ...,
                "det_loss_yns_0": ...,
            }
        其中 prefix 由任务决定（det_ / map_ 等），suffix 由 decoder stage 决定（_0 ... _5）。    
    '''
    # SparseBox3DLoss 是 SparseDrive 3D 检测分支的 box loss 封装器
    # 它内部可以包含：
    # 1. box regression loss，例如 L1Loss
    # 2. centerness loss，例如 CrossEntropyLoss
    # 3. yawness loss，例如 GaussianFocalLoss
    # 4. 对某些类别允许 yaw 方向反转的特殊处理

    # 1. 根据配置构建各个子 loss，并保存方向等价类别
    def __init__(
        self,                   # self 表示当前 SparseBox3DLoss 对象
        loss_box,               # box 回归损失配置，常为加权 L1Loss，如：loss_box=dict(type="L1Loss", loss_weight=0.25)
        loss_centerness=None,   # centerness 监督损失配置，如：loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True)；None 表示关闭
        loss_yawness=None,      # yawness 监督损失配置，如：loss_yawness=dict(type="GaussianFocalLoss")；None 表示关闭
        cls_allow_reverse=None, # 允许方向反转的类别 id 列表，如 barrier 类别，正反方向可能等价【换言之，对于“正反方向等价”的类别，允许 GT yaw + pi 与预测匹配】
    ):
        super().__init__() # 调用 nn.Module 的初始化函数

        # (1) 定义内部工具函数 build()，它是局部构建器，用于根据配置 cfg 和 registry 构建模块：cfg=None 时明确关闭该子 loss，否则从 LOSSES 注册表构建
        def build(cfg, registry):
            if cfg is None:                      # 若未给配置，说明该损失分支不启用
                return None                      # 返回 None，表示不使用该 loss
            return build_from_cfg(cfg, registry) # 否则，按 MMCV/MMDetection 配置 cfg 从 registry 中构建具体 loss 对象模块

        # (2) 构建损失模块：
        self.loss_box = build(loss_box, LOSSES)        # 构建主 box 回归损失，通常是 L1Loss
        self.loss_cns = build(loss_centerness, LOSSES) # 构建 centerness 损失，若 loss_centerness=None 时 self.loss_cns=None
        self.loss_yns = build(loss_yawness, LOSSES)    # 构建 yawness 损失，若loss_yawness=None 时 self.loss_yns=None

        # (3) 保存允许 yaw 反向等价的类别编号列表
        self.cls_allow_reverse = cls_allow_reverse # 保存允许方向反转的类别列表，例如 nuScenes 中 barrier 的方向正反通常没有明显区别

    # 2. forward()：为一组正样本生成 box / centerness / yawness 联合损失
    def forward(
        self,            # self 表示当前 SparseBox3DLoss 对象
        box,             # 模型预测的 box，shape [num_pos, box_dim]，box 内部包含 x,y,z,w,l,h,sin_yaw,cos_yaw,vx...
        box_target,      # 匹配到的 GT box，shape 与 box 对应
        weight=None,     # 每个回归维度的权重，shape [num_pos, box_dim]
        avg_factor=None, # loss 平均因子，通常是正样本数量 num_pos
        prefix="",       # loss 名称前缀，如 detection 任务中 prefix="det_"
        suffix="",       # loss 名称后缀，如不同 decoder 层 suffix="_0", "_1"
        quality=None,    # 质量估计输出，shape [num_pos, 2]，两维可能分别对应 centerness 和 yawness
        cls_target=None, # 正样本对应的类别标签，用于判断哪些类别允许 yaw 方向反转
        **kwargs,        # 接收额外参数，当前函数中没有直接使用
    ):
        '''
            【输入】
                box        : [P,D]，当前 decoder stage 的正样本预测 box state。
                box_target : [P,D]，与 box 一一对应的 GT box state。
                weight     : [P,D]，逐样本、逐状态维度的回归权重。
                avg_factor : 归一化因子，通常来自跨 GPU 平均后的 num_pos。
                quality    : None 或 [P,2]，通常 [centerness_logit, yawness_logit]。
                cls_target : None 或 [P]，仅用于方向等价类别处理。
            
            【主流程】
                (1) 可选：将“允许 yaw 反转”的 GT yaw 调整到更接近 prediction 的等价方向。
                (2) 计算基础 box regression loss。
                (3) quality 存在时，构造 centerness target 并计算 centerness loss。
                (4) quality 存在时，构造 yawness target 并计算 yawness loss。
                (5) 返回按 prefix / suffix 命名的 loss dict。
        '''

        # 1. 可选：处理“方向正反等价”的类别【某些类别不区分正反方向，例如 nuScenes 里的 barrier，对这些类别，如果预测方向和 GT 方向相反，也可以认为方向是等价的】
        if self.cls_allow_reverse is not None and cls_target is not None: # 如果配置了允许反转的类别，并且传入了类别标签
            # (1) 先判断 GT yaw 与 prediction yaw 是否大致相反：box state 中 yaw 由 [sin(yaw), cos(yaw)] 表示，两个单位方向向量余弦相似度 < 0 表示夹角大于 90°，则视为“方向相反”
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]], # GT yaw 向量
                    box[..., [SIN_YAW, COS_YAW]],        # 预测 yaw 向量
                    dim=-1,                              # 在最后一维计算 cosine similarity
                )   # a. 计算 “GT 朝向向量” 和 “预测朝向向量” 的余弦相似度
                < 0 # b. 如果 cosine_similarity < 0，说明两个方向夹角大于 90 度，则可以认为方向大致相反
            )

            # (2) 再进一步限制：只有 “类别允许反向(即类别属于 cls_allow_reverse)” 且 “预测与实际 GT 反向” 时，才标记为可反转
            if_reverse = (
                torch.isin(
                    cls_target,                                   # 当前正样本类别
                    cls_target.new_tensor(self.cls_allow_reverse) # 允许反转类别列表转成 Tensor
                )            # (a) 判断每个正样本类别 cls_target 是否属于允许反转的类别集合 cls_allow_reverse
                & if_reverse # (b) 与上一步 (1.1) 得到的“方向相反”条件取逻辑与，即同时要求方向确实相反
            )

            # (3) 对可反转样本，将 GT 的 yaw vector 取反，即 [sin(yaw), cos(yaw)] → [-sin(yaw), -cos(yaw)] 
            # 因为方向向量 [sin, cos] 取负后，相当于 yaw + pi，对 barrier 等正反等价目标，这样可以让 GT 方向被改为与预测更接近的方向
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],                # if_reverse 的 shape [P=num_pos]，if_reverse[..., None] 扩展成 [P=num_pos, 1]，方便广播到两个 yaw state
                -box_target[..., [SIN_YAW, COS_YAW]], # (a) 若允许反向：则 GT yaw 向量取负
                box_target[..., [SIN_YAW, COS_YAW]],  # (b) 若不允许反向：则 GT yaw 向量保持不变
            )


        output = {} # 初始化当前 decoder stage 的 loss 输出字典
        # 2. 【核心】计算基础 3D box 回归 loss 并写入 output【self.loss_box() 采用的是 L1Loss】
        box_loss = self.loss_box(
            box,                  # 预测 box state，[P,D]
            box_target,           # 匹配到的 GT box state，[P,D]
            weight=weight,        # 每个状态维度的回归权重，[P,D] 或可广播形状
            avg_factor=avg_factor # loss 的归一化因子，通常是正样本数
        )
        output[f"{prefix}loss_box{suffix}"] = box_loss # 将 box 主回归 loss 写入输出字典：假设 prefix="det_", suffix="_0"，则 key 就是 "det_loss_box_0"
        '''
            weight 同时编码：
                (a) GT 某个 state 是否有效；
                (b) 该类别 / 该 state 的手工权重；
                (c) Sparse4DHead.loss() 中的全局 reg_weights。        
        '''


        # 3.【核心】可选：quality 分支联合监督：计算 centerness loss 和 yawness loss，并写入 output
        if quality is not None: # 如果传入了 quality
            # (1) 拆出 quality 的两个预测分量
            cns = quality[..., CNS]           # 取 centerness 预测 cns，shape [num_pos]，cns 保持 logit【所用 CrossEntropyLoss(use_sigmoid=True) 会在内部处理 sigmoid】。CNS 是 quality 中 centerness 的索引
            yns = quality[..., YNS].sigmoid() # 取 yawness 预测 yns，这里明确将 yns 进行 sigmoid 成 [0,1] 质量分数，这与本项目配置的 yawness loss 使用方式匹配。YNS 是 quality 中 yawness 的索引

            # (2) 构造 centerness target【这里的 centerness 不是传统 FCOS 中的中心度，而是根据预测中心和 GT 中心的距离生成一个质量标签】：
            # (2.a) 先计算预测中心与 GT 中心的 3D 欧氏距离：distance = || center_gt - center_pred ||_2
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], # GT 中心和预测中心之差
                p=2,                                              # L2 距离
                dim=-1                                            # 在 xyz 维度上求 norm
            )
            # (2.b) 再将距离转换成 (0,1] 的连续质量标签：centerness_target = exp(-distance)。距离为 0 时 target=1，距离越远 target 越接近 0。
            cns_target = torch.exp(-cns_target) # 将预测中心与 GT 中心的 3D 欧氏距离转成质量分数：距离越小，exp(-distance) 越接近 1；距离越大，exp(-distance) 越接近 0

            # (3) 计算 centerness loss 并写入 output【self.loss_cns() 采用的是 CrossEntropyLoss(use_sigmoid=True)】
            cns_loss = self.loss_cns(
                cns,                   # centerness prediction logit：[P]
                cns_target,            # 连续 centerness target：[P]
                avg_factor=avg_factor, # 与主 box loss 使用同一正样本平均因子
            )
            output[f"{prefix}loss_cns{suffix}"] = cns_loss # 将 centerness loss 写入输出字典：假设 prefix="det_", suffix="_0"，则 key 就是 "det_loss_cns_0"

            # (4) 构造 yawness target：yawness 表示预测方向是否和 GT 方向一致：
            # (4.a) 用预测 yaw 方向向量与 GT yaw 方向向量的余弦相似度判断“朝向是否大体正确”：cosine > 0  <=> 夹角小于 90°  <=> target=1；cosine <=0  <=> 夹角不小于90° <=> target=0
            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]], # GT yaw 向量
                    box[..., [SIN_YAW, COS_YAW]],        # 预测 yaw 向量
                    dim=-1,                              # 在最后一维算余弦相似度【即在 [sin,cos] 方向向量维度上计算】
                )
                > 0 # 如果 cosine similarity > 0，说明夹角小于 90 度，则认为方向大致正确
            )
            yns_target = yns_target.float() # (4.b) bool target 转成 float target：True -> 1.0，False -> 0.0

            # (5) 计算 yawness loss 并写入 output【self.loss_yns() 采用的是 GaussianFocalLoss】
            yns_loss = self.loss_yns(
                yns,                   # [0,1] 范围的 yawness prediction
                yns_target,            # 二值 yawness target
                avg_factor=avg_factor, # 与其他分支保持相同的正样本平均因子
            )
            output[f"{prefix}loss_yns{suffix}"] = yns_loss # 将 yawness loss 写入输出字典：假设 prefix="det_", suffix="_0"，则 key 就是 "det_loss_yns_0"


        # 4. 返回 loss 字典【返回当前 decoder stage 的 box + 可选 quality loss 字典】
        return output

