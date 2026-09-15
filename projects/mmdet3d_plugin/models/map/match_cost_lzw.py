# =============================================================================
# match_cost.py（逐行详细注释版）
# =============================================================================
# 1. 文件作用
#    (1) 本文件定义 map 分支 Hungarian matching 使用的 cost。
#    (2) LinesL1Cost：计算预测 polyline 与 GT polyline 之间的 L1 / SmoothL1 几何距离。
#    (3) MapQueriesCost：把分类 cost、线回归 cost、可选 IoU cost 加权合成为总匹配代价。
#
# 2. 为什么 map line 需要 permute
#    (1) 一条 vectorized map line 由一串采样点表示。
#    (2) 对某些线来说，从 A->B 采样和从 B->A 采样几何上是同一条线。
#    (3) 因此 GT 可能保存多个等价排列，例如正序和反序。
#    (4) permute=True 时，本文件会在多个 GT 排列中选代价最小的那个，减少点序方向带来的监督歧义。
# =============================================================================

import torch                                               # 导入 PyTorch，用于 torch.cdist 计算 L1 距离矩阵
from mmdet.core.bbox.match_costs.builder import MATCH_COST # 导入 MMDetection 的匹配代价注册表
from mmdet.core.bbox.match_costs import build_match_cost   # 导入 build_match_cost，用配置字典构建 cls/reg/iou cost
from torch.nn.functional import smooth_l1_loss             # 导入 Smooth L1 loss，用于 beta > 0 时的鲁棒回归距离


# 一、LinesL1Cost 类：定义 polyline 几何匹配代价，核心是点序列之间的 L1 或 SmoothL1 距离
@MATCH_COST.register_module() # 将 LinesL1Cost 注册到 MATCH_COST，配置中 type='LinesL1Cost' 时可自动构建
class LinesL1Cost(object):
    # 1. 初始化参数
    def __init__(self, weight=1.0, beta=0.0, permute=False):  # 初始化 line regression cost
        self.weight = weight   # 保存 cost 权重【当前 regression cost 的权重，最终返回 dist_mat * weight】
        self.permute = permute # 保存是否启用匹配置换 permutation-invariant matching【是否处理 GT line 的多种等价点序排列】【是否启用置换不变匹配，用于解决线的起点/方向歧义】
        self.beta = beta       # 保存 SmoothL1 的 beta 参数: beta=0 表示不用 SmoothL1，而用 torch.cdist 的 L1 距离；beta>0 时启用 Smooth L1 距离

    # 2. 计算预测线与 GT 线之间的匹配代价矩阵
    def __call__(self, lines_pred, gt_lines, **kwargs):
        """
            可调用方法：计算预测线与真值线的回归成本矩阵
                Args:
                    lines_pred (Tensor): 归一化后的预测线坐标，[num_query, 2*num_points] （每个采样点xy坐标展平）
                    gt_lines (Tensor)  : 归一化后的真值线坐标，形状: 
                                                permute=False时为[num_gt, 2*num_points]
                                                permute=True时为[num_gt, num_permute, 2*num_points]（包含多种排列的真值）
                Returns:
                    普通模式：回归成本矩阵，形状 [num_pred, num_gt]
                    置换模式：元组(成本矩阵, 最优置换索引)
                        成本矩阵: [num_pred, num_gt]
                        gt_permute_index: [num_pred, num_gt]，每个预测-真值对对应的最优置换下标
        """ 
        # 维度校验：置换模式下真值必须是3维，普通模式必须是2维
        if self.permute:                    # 如果启用 permutation-invariant 模式
            assert len(gt_lines.shape) == 3 # 要求 GT shape 为 [num_gt, num_permute, 2*num_points]
        else:                               # 如果不启用 permutation-invariant 模式
            assert len(gt_lines.shape) == 2 # 要求 GT shape 为 [num_gt, 2*num_points]

        # 获取预测线数量、真值线 GT 数量
        num_pred, num_gt = len(lines_pred), len(gt_lines)

        # (1) 置换模式：将真值的 "真值数×排列数" 两个维度展平，统一计算距离
        if self.permute:                      # 如果 GT 含多个点序排列
            gt_lines = gt_lines.flatten(0, 1) # 把 [num_gt, num_permute, 2*num_pts] 拉成 [num_gt*num_permute, 2*num_pts]

        # (2) 计算单条线的采样点数量 = 总坐标数/2
        num_pts = lines_pred.shape[-1] // 2  # 根据最后一维坐标长度计算点数；例如 40//2=20

        # (3.1) beta>0 时使用 Smooth L1 损失计算距离
        if self.beta > 0:
            # 扩展维度：预测扩展真值维度，真值扩展预测维度，形成两两配对的张量
            lines_pred = lines_pred.unsqueeze(1).repeat(1, len(gt_lines), 1) # [num_pred, D] -> [num_pred, num_gt_or_perm,D]
            gt_lines = gt_lines.unsqueeze(0).repeat(num_pred, 1, 1)          # [num_gt_or_perm, D] -> [num_pred, num_gt_or_perm,D]

            # 计算逐元素Smooth L1距离，在坐标维度求和，得到距离矩阵
            dist_mat = smooth_l1_loss(lines_pred, gt_lines, reduction='none', beta=self.beta).sum(-1) # 对坐标维求和，得到 pairwise SmoothL1 距离

        # (3.2) beta=0时使用标准L1距离，调用torch.cdist高效计算
        else:  
            dist_mat = torch.cdist(lines_pred, gt_lines, p=1) # 计算 pairwise L1 距离矩阵，p=1表示曼哈顿距离（L1距离），shape [num_pred, num_gt_or_perm]

        # (4) 按采样点数量归一化，消除点数量对成本尺度的影响
        dist_mat = dist_mat / num_pts # 用点数归一化，使 cost 不随采样点数量线性变大

        # (5.1) 置换模式：从所有排列中选出成本最小的排列
        if self.permute:                                        # 如果启用了多排列 GT
            dist_mat = dist_mat.view(num_pred, num_gt, -1)      # dist_mat当前形状 (num_pred, num_gt*num_permute)，重塑为 (num_pred, num_gt, num_permute)
            dist_mat, gt_permute_index = torch.min(dist_mat, 2) # 对每个 pred-gt pair 选择代价最小的 GT 排列，并记录排列索引：在排列维度取最小值，得到最小成本和对应的排列索引
            return dist_mat * self.weight, gt_permute_index     # 返回加权后的最小代价，以及最优排列 index【加权后的成本矩阵 + 最优置换索引】

        # (5.2) 普通模式：直接返回加权后的成本矩阵
        return dist_mat * self.weight # 不启用 permute 时，直接返回加权距离矩阵 [num_pred,num_gt]


# 二、计算 map query 的总匹配代价：分类 cost + 回归 cost + 可选 IoU cost
@MATCH_COST.register_module() # 将 MapQueriesCost 注册到 MATCH_COST，配置中 type='MapQueriesCost' 时可自动构建
class MapQueriesCost(object):
    # 1. 初始化总 cost 组合器
    def __init__(self, cls_cost, reg_cost, iou_cost=None):
        """
            地图查询的总匹配成本：聚合分类成本、回归成本、可选IoU成本
            Args:
                cls_cost: 分类成本的配置字典，例如 FocalLossCost
                reg_cost: 回归成本的配置字典，例如 LinesL1Cost
                iou_cost: 可选，IoU成本的配置字典
        """
        self.cls_cost = build_match_cost(cls_cost) # 根据配置构建分类 cost 实例
        self.reg_cost = build_match_cost(reg_cost) # 根据配置构建回归 cost 实例，通常是 LinesL1Cost
        self.iou_cost = None                           # 默认不使用 IoU cost
        if iou_cost is not None:                       # 如果配置中提供了 iou_cost
            self.iou_cost = build_match_cost(iou_cost) # 根据配置构建 IoU cost 实例

    # 2. 计算 map query 与 GT 的总匹配代价
    def __call__(self, preds: dict, gts: dict, ignore_cls_cost: bool):  
        """
        可调用方法：计算总匹配成本矩阵
        Args:
            preds (dict): 预测结果字典，包含'scores'分类logits、'lines'坐标，可选'masks'掩码：
                scores: [num_pred, num_cls] 分类 logits / score。
                lines:  [num_pred, 2*num_sample] 归一化后的预测 polyline。
                masks:  可选 mask，用于某些动态 line cost。
            gts (dict): GT 字典，包含'labels'类别标签、'lines'坐标，可选'masks'掩码：
                labels: [num_gt] GT 类别。
                lines:  [num_gt, 2*num_sample] 或 [num_gt, num_permute, 2*num_sample]。
                masks:  可选 GT mask。
            ignore_cls_cost (bool): 是否忽略分类代价【纯回归匹配时使用：设为 True 时只用几何回归代价匹配】

        Returns:
            普通模式：返回总成本矩阵，形状 [num_pred, num_gt]
            置换模式：返回元组(总成本矩阵, 最优置换索引)，二者形状均为 [num_pred, num_gt]
        """

        # 1. 计算分类成本
        cls_cost = self.cls_cost(preds['scores'], gts['labels']) # 根据分类预测和 GT label 计算分类匹配代价 [num_pred,num_gt]


        # 2. 计算回归成本
        # (1) 创建回归成本的额外参数字典，默认 LinesL1Cost 不需要额外参数
        regkwargs = {}

        # (2) 如果预测和真值都包含掩码 'masks'，且回归成本是 DynamicLinesCost 类型，则传入掩码参数，尝试走动态线代价逻辑
        if 'masks' in preds and 'masks' in gts:
            assert isinstance(self.reg_cost, DynamicLinesCost), ' Issues!!' # DynamicLinesCost 当前文件未导入；若走到该分支会依赖外部定义/导入
            regkwargs = {                                                   # 构造传给动态 regression cost 的 mask 参数
                'masks_pred': preds['masks'], # 预测 mask
                'masks_gt': gts['masks'],     # GT mask
            }

        # (3) 计算回归成本
        reg_cost = self.reg_cost(preds['lines'], gts['lines'], **regkwargs) # 计算 polyline 回归匹配代价，可能返回 cost 或 (cost, permute_idx)

        # (4) 如果回归成本启用了置换模式，解包出成本和置换索引
        if self.reg_cost.permute:               # 如果回归 cost 启用了 GT permutation 处理
            reg_cost, gt_permute_idx = reg_cost # 拆出真实回归代价和每个 pred-gt pair 的最优 GT 排列索引

        # 3. 加权融合总成本
        # weighted sum of above three costs  # 原源码注释：将各项 cost 加权求和；权重已在各 cost 内部处理
        if ignore_cls_cost:            # (a) 如果调用者要求忽略分类代价
            cost = reg_cost            # 则总代价只使用回归代价
        else:                          # (b) 如果不忽略分类代价
            cost = cls_cost + reg_cost # 则总代价 = 分类代价 + 回归代价

        # 4. 可选：加上 IoU 成本
        if self.iou_cost is not None:                              # 如果配置了 IoU cost
            iou_cost = self.iou_cost(preds['lines'], gts['lines']) # 计算 line/polyline 的 IoU 相关代价
            cost += iou_cost                                       # 将 IoU 代价加入总匹配代价

        # 5. 置换模式下，返回总成本+置换索引；普通模式下，只返回总成本
        if self.reg_cost.permute:       # 如果 regression cost 返回了最优排列索引
            return cost, gt_permute_idx # 返回总 cost 和 GT 最优排列索引
        return cost                     # 否则只返回总 cost
