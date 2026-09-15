import torch
import torch.nn as nn
from mmcv.utils import build_from_cfg                   # 导入mmcv的配置构建工具
from mmdet.models.builder import LOSSES                 # 从mmdet导入损失注册器
from mmdet.models.losses import l1_loss, smooth_l1_loss # 导入mmdet实现的L1损失【l1_loss 计算普通绝对值误差】、Smooth L1损失【smooth_l1_loss 在误差较小时更平滑，常用于检测框/点坐标回归】

# 一、LinesL1Loss 类：计算 map 折线点坐标回归的底层损失：接收 pred 和 target，计算二者在线坐标维度上的 L1 或 Smooth L1 误差
@LOSSES.register_module() # 将 LinesL1Loss 注册为 MMDetection 可配置 loss。注册后，配置文件可以通过 dict(type='LinesL1Loss', ...) 构建该类。
class LinesL1Loss(nn.Module):
    # 1. 初始化
    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        super().__init__()
        self.reduction = reduction     # 保存损失降维方式，可为 'none'、'mean'、'sum'
        self.loss_weight = loss_weight # 保存损失权重，最终返回 loss * loss_weight
        self.beta = beta               # 保存Smooth L1的beta参数：beta>0时启用Smooth L1，否则使用普通 L1

    # 2. 前向传播：计算损失值
    def forward(
            self,
            pred,                    # 预测线坐标，shape [B, num_query, 2*num_sample]
            target,                  # GT 线坐标，shape [B, num_query, 2*num_sample]
            weight=None,             # 每个预测元素的 loss 权重掩码，正样本为1负样本为0，用于只监督匹配到 GT 的正样本。
            avg_factor=None,         # 平均因子，即损失平均的分母，通常是正样本数量
            reduction_override=None, # 临时覆盖 self.reduction 的规约方式。
        ):
        '''
            函数实现步骤：
                Step1. 计算预测地图线 pred 与目标地图线 target 之间的坐标回归 loss。
                Step2. 如果 beta > 0，则使用 Smooth L1；否则使用 L1。
                Step3. 最后除以 num_points，使 loss 变成“每个点的平均损失”，避免采样点数量改变导致 loss 尺度改变。
        '''
        # 校验reduction参数合法性
        assert reduction_override in (None, 'none', 'mean', 'sum')

        # 优先使用传入的reduction，否则用默认值
        reduction = (reduction_override if reduction_override else self.reduction)

        # (1.1) beta>0时使用Smooth L1损失【Smooth L1 在误差较小时接近 L2，在误差较大时接近 L1，这样可以兼顾平滑梯度和抗异常值能力】
        if self.beta > 0:
            loss = smooth_l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
            '''
                传参：
                    pred 和 target 逐元素比较。
                    weight 用于控制哪些元素参与 loss。
                    avg_factor 控制平均分母。
                    beta 是 Smooth L1 的阈值。
            '''
        # (1.2) beta=0时使用标准L1损失【普通 L1 就是 |pred - target|】
        else:
            loss = l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        # (2) 按采样点数量归一化：消除点数量对损失量级的影响
        num_points = pred.shape[-1] // 2 # 计算一条线包含多少个二维点：最后一维是 [x1,y1,x2,y2,...,xN,yN]，长度为 2*num_points，因此除以 2 得到 num_points
        loss = loss / num_points         # 将 loss 除以点数【这样 loss 更像“每个采样点平均误差”，如果 num_sample 从 20 改成 40，loss 尺度不会直接翻倍】
        
        # (3) 乘以总损失权重后返回
        return loss * self.loss_weight # 返回最终加权后的损失标量【loss_weight 用于调节该 loss 在总训练目标中的占比，例如总 loss = cls_loss + line_loss + planning_loss 等】


# 二、SparseLineLoss 类：稀疏线检测的高层损失封装：内置坐标归一化，统一输出损失字典
@LOSSES.register_module() # 将 SparseLineLoss 注册为 MMDetection 可配置 loss。注册后，配置文件可以通过 dict(type='SparseLineLoss', ...) 构建该类。
class SparseLineLoss(nn.Module):
    # 1. 构建回归损失实例
    def __init__(
        self,
        loss_line,         # 内部线回归 loss 的配置字典，例如 LinesL1Loss。
        num_sample=20,     # 单条线的采样点数，默认 20。
        roi_size=(30, 60), # BEV ROI 尺寸，用于坐标归一化。(30, 60) 表示 x 方向 30m、y 方向 60m 的 ROI区域。
    ):
        super().__init__()

        # 内部构建函数：根据配置和注册器构建实例，空配置返回None
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        # (1) 构建底层的线回归损失实例
        self.loss_line = build(loss_line, LOSSES)

        # (2) 保存参数
        self.num_sample = num_sample # 保存采样点数量
        self.roi_size = roi_size     # 保存ROI尺寸

    # 2. 前向传播：计算线损失并返回字典
    def forward(
        self,
        line,            # 预测线坐标（物理空间，米），[B, num_query, 2*num_sample]
        line_target,     # GT 线坐标（物理空间，米），[B, num_query, 2*num_sample]
        weight=None,     # 回归权重掩码，正样本为 1、背景 query 为 0。
        avg_factor=None, # 平均因子，通常与正样本数量有关。
        prefix="",       # loss 名称前缀，例如某个 decoder 层编号。
        suffix="",       # loss 名称后缀，例如不同阶段或辅助分支标记。
        **kwargs,        # 预留参数，兼容上层统一 loss 调用接口。
    ):
        # 初始化输出字典
        output = {}
        
        # (1) 归一化
        line = self.normalize_line(line)               # 对预测线做归一化（物理空间→(0,1)区间）
        line_target = self.normalize_line(line_target) # 对真值线做归一化，与预测保持相同尺度

        # (2) 调用底层损失函数计算损失
        line_loss = self.loss_line(line, line_target, weight=weight, avg_factor=avg_factor)
        
        # (3) 将损失存入字典，key支持前后缀自定义
        output[f"{prefix}loss_line{suffix}"] = line_loss

        # (4) 返回损失字典
        return output # dict: 损失字典，key为损失名称，value为损失值

    # 3. 线坐标归一化：将物理空间（米）的坐标映射到(0,1)区间【与target.py中的归一化逻辑完全一致，保证匹配和训练尺度统一】
    def normalize_line(self, line):
        """
            传参:
                line (Tensor): 原始线坐标，形状 [..., 2*num_sample]
            返回值:
                Tensor: 归一化后的线坐标，形状不变
        """
        # 线数量为0时直接返回
        if line.shape[0] == 0:
            return line
        
        # 将展平的坐标重塑为 [..., num_sample, 2]，便于逐点处理
        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        # 计算ROI的原点偏移：ROI以坐标原点为中心，原点坐标是负的半宽高
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])

        # 坐标平移：将ROI左上角移到坐标原点
        line = line - origin
        
        # 极小值防止除零
        eps = 1e-5

        # 归一化分母：ROI尺寸 + eps
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps

        # 坐标除以ROI尺寸，映射到(0, 1)区间
        line = line / norm
        
        # 重新展平最后两维，恢复原始形状
        line = line.flatten(-2, -1)
        
        # 返回归一化后的坐标
        return line
