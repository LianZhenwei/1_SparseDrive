# =============================================================================
# target.py（逐行详细注释版）
# =============================================================================
# 1. 文件作用
#    (1) 本文件定义 map 分支训练时的 target 分配逻辑。
#    (2) SparsePoint3DTarget：负责把 GT map polyline 分配给预测 query，并生成 loss 需要的 cls/reg target。
#    (3) HungarianLinesAssigner：负责基于 cost matrix 执行 Hungarian matching，得到一对一匹配结果。
#
# 2. 训练链条中的位置
#    (1) map head 输出 cls_preds 和 pts_preds。
#    (2) SparsePoint3DTarget.sample() 对每张图分别归一化预测线和 GT 线。
#    (3) HungarianLinesAssigner.assign() 计算 cost matrix 并调用 scipy 的 linear_sum_assignment。
#    (4) sample() 根据匹配结果生成 output_cls_target、output_box_target、output_reg_weights。
#    (5) 后续 loss 函数用这些 target 监督 map polyline 分类和坐标回归。
# =============================================================================

import torch                                     # 导入 PyTorch；当前文件主要使用 tensor 创建、dtype、shape 等能力
import numpy as np                               # 导入 NumPy；当前源码未直接使用，属于冗余导入但保留原文件一致性
import torch.nn.functional as F                  # 导入 PyTorch 函数式接口；当前源码未直接使用，属于冗余导入
from scipy.optimize import linear_sum_assignment # 导入 Hungarian algorithm 实现，用于最小总代价二分图匹配

from mmdet.core.bbox.builder import (BBOX_SAMPLERS, BBOX_ASSIGNERS) # 导入 MMDetection 的 sampler 和 assigner 注册表
from mmdet.core.bbox.match_costs import build_match_cost            # 导入 cost 构建函数，用于根据配置构建 MapQueriesCost
from mmdet.core import (build_assigner, build_sampler)              # build_assigner 用于构建 assigner；build_sampler 当前未使用，保留原源码一致性
from mmdet.core.bbox.assigners import (AssignResult, BaseAssigner)  # BaseAssigner 是 assigner 基类；AssignResult 当前未使用，因为本实现直接返回索引

from ..base_target import BaseTargetWithDenoising # 导入带 denoising 机制的 target 基类，map target 继承它以复用 DN 相关逻辑


# 一、定义 map polyline 的训练 target 生成器：完成预测与真值的匹配，生成训练用的分类目标、回归目标、回归权重
@BBOX_SAMPLERS.register_module() # 将 SparsePoint3DTarget 注册到 BBOX_SAMPLERS，配置中 type='SparsePoint3DTarget' 时可自动构建
class SparsePoint3DTarget(BaseTargetWithDenoising): 
    """
        SparsePoint3DTarget 的作用：
            接收 map head 每层输出的分类预测 cls_preds 和点坐标预测 pts_preds。
            接收 GT 类别 cls_targets 和 GT polyline 点坐标 pts_targets。
            调用 assigner 做 Hungarian matching，生成3个训练目标：分类 target、回归 target 和回归权重。
    """

    # 1. 初始化 map target 生成器
    def __init__( 
        self,                 # 当前对象
        assigner=None,        # assigner 分配器配置字典，用于构建匹配分配器，例如 HungarianLinesAssigner
        num_dn_groups=0,      # DN group 数量【去噪分组数量】，默认 0 表示不使用 DN 或由配置决定
        dn_noise_scale=0.5,   # DN 噪声尺度【去噪噪声的缩放系数】，默认 0.5
        max_dn_gt=32,         # 每张图最多使用的 DN GT 数量【单帧最多参与去噪的真值数量】
        add_neg_dn=True,      # 是否添加负样本 DN
        num_temp_dn_groups=0, # temporal DN group 数量
        num_cls=3,            # map 类别数，默认 3
        num_sample=20,        # 每条 map line 的采样点数量，默认 20
        roi_size=(30, 60),    # ROI 大小（x方向、y方向长度，单位米），默认 x 方向 30、y 方向 60
    ):
        super(SparsePoint3DTarget, self).__init__(num_dn_groups, num_temp_dn_groups) # 调用 BaseTargetWithDenoising 的初始化逻辑，传入普通 DN group 和 temporal DN group 数量
        
        # (1) 根据配置构建 assigner 分配器实例，通常是 HungarianLinesAssigner
        self.assigner = build_assigner(assigner) 
        
        # (2) 保存参数
        self.dn_noise_scale = dn_noise_scale # 保存 DN 噪声尺度
        self.max_dn_gt = max_dn_gt           # 保存每张图最大 DN GT 数量
        self.add_neg_dn = add_neg_dn         # 保存是否添加 DN 负样本
        self.num_cls = num_cls               # 保存 map 类别数；背景 label 会设为 num_cls
        self.num_sample = num_sample         # 保存每条 polyline 的采样点数
        self.roi_size = roi_size             # 保存 ROI 大小，用于 normalize_line 中把坐标缩放到 [0,1]

    # 2. 做匈牙利匹配生成训练 target：分类 target、回归 target、回归权重
    def sample(  
        self,        # 当前对象
        cls_preds,   # 当前 decoder stage 的 map polyline 分类预测，[B, N=100, num_cls=3]
        pts_preds,   # 当前 decoder stage 的 map polyline 点坐标预测，[B, N=100, 2*num_sample=40]
        cls_targets, # GT 类别 list，长度 B，每个元素 shape [num_gt]
        pts_targets, # GT polyline list，长度 B，每个元素可为 [num_gt, num_permute, num_sample, 2] 或已 flatten
    ):
        # 对真值做维度规整：如果是4维（含置换维度）就展平成3维
        pts_targets = [x.flatten(2, 3) if len(x.shape) == 4 else x for x in pts_targets] # 若 GT 为 [G,P,S,2]，则展平成 [G,P,2S]

        # 初始化每帧的匹配索引列表
        indices = [] # 保存 batch 内每张图的 Hungarian 匹配结果

        # (1) 逐帧遍历预测与真值
        for (cls_pred, pts_pred, cls_target, pts_target) in zip(cls_preds, pts_preds, cls_targets, pts_targets): # 遍历 batch 内每张图的预测和 GT，zip 为四类输入一一对应：分类预测、点预测、类别 GT、点 GT
            # normalize to (0, 1)                           # 原源码注释：把坐标归一化到 0~1 附近，便于不同尺度下 cost 稳定
            pts_pred = self.normalize_line(pts_pred)        # 将预测 polyline 从局部坐标归一化到 [0,1]，统一匹配尺度
            pts_target = self.normalize_line(pts_target)    # 将 GT polyline 从局部坐标归一化到 [0,1]
            preds = dict(lines=pts_pred, scores=cls_pred)   # 构造 assigner 需要的预测字典
            gts = dict(lines=pts_target, labels=cls_target) # 构造 assigner 需要的 GT 字典
            indice = self.assigner.assign(preds, gts)       # 执行 Hungarian matching，返回预测索引 pred_idx、真值索引 target_idx、置换索引 gt_permute_index
            indices.append(indice)                          # 保存当前图的匹配结果

        # (2) 初始化
        bs, num_pred, num_cls = cls_preds.shape                                                 # 读取 batch size、query 数量和类别数【num_cls 通常等于 self.num_cls】
        output_cls_target = cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls # 初始化分类目标：全部填充为 num_cls（即全部填充为背景类别）
        output_box_target = pts_preds.new_zeros(pts_preds.shape)                                # 初始化回归目标：全 0，shape 与 pts_preds 相同
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)                               # 初始化回归权重：全 0，未匹配 query 不参与回归 loss

        # (3) 遍历每张图的匹配结果，逐帧填充目标
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0: # 如果当前图没有 GT，则直接跳过
                continue                 # 直接跳过，所有 query 保持背景且回归权重为 0

            # 根据置换索引，获取与预测点序对齐的真值点序
            permute_idx = gt_permute_index[pred_idx, target_idx] # 对每个匹配 pred-gt pair 取最优 GT 点序排列 index

            # 填充：
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]              # 填充分类目标：将匹配到 GT 的预测位置 query 设为 GT 类别
            output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx] # 填充回归目标：按置换后的顺序填入真值坐标【将回归 target 设为最优点序排列下的 GT polyline】
            output_reg_weights[i, pred_idx] = 1                                      # 回归权重设为1：仅正样本参与回归损失计算【将匹配到 GT 的 query 的所有坐标维回归权重设为 1】

        # (4) 返回三类训练目标：分类 target、坐标 target、回归权重
        '''
            output_cls_target : 分类目标，[B, num_pred]，未匹配 query 的类别为背景类比 num_cls
            output_box_target : 回归目标，[B, num_pred, 40]，匹配 query 的回归目标为对应 GT line
            output_reg_weights: 回归权重，[B, num_pred, 40]，正样本为1、负样本为0【即匹配 query 为 1、未匹配为 0】
        '''
        return output_cls_target, output_box_target, output_reg_weights

    # 3. 将 polyline 坐标从物理空间 BEV 局部坐标（米）归一化到 (0,1) 区间【目的：消除物理尺度影响，让匹配成本更稳定】
    def normalize_line(self, line):
        """
            坐标变换逻辑：
                原始坐标通常以 ego 为中心，例如 x in [-roi_w/2, roi_w/2]，y in [-roi_h/2, roi_h/2]。
                origin = [-roi_w/2, -roi_h/2]。
                line - origin 等价于整体平移到左下角为 0。
                再除以 [roi_w, roi_h]，得到近似 [0,1] 的归一化坐标。
        """
        # 如果线的数量为0，直接返回
        if line.shape[0] == 0: # 如果没有任何 GT line / pred line
            return line        # 直接返回空 tensor，避免 view 或除法出错

        # (1) 将展平的坐标重塑为 [..., num_sample, 2]
        line = line.view(line.shape[:-1] + (self.num_sample, -1)) # 将最后一维 [2S] 还原成 [S,2]，兼容前面可能存在的 batch/permute 维

        # (2) 计算ROI的原点偏移（ROI以原点为中心，所以原点是负的半宽高）
        origin = -line.new_tensor([self.roi_size[0] / 2, self.roi_size[1] / 2]) # 构造 ROI 左下角坐标 [-w/2,-h/2]
        
        # (3) 坐标平移：将ROI中心移到原点
        line = line - origin # 坐标平移：以 ROI 左下角为新原点，使范围从 [-w/2,w/2] 变成 [0,w]

        # (4) 坐标除以ROI尺寸，归一化到(0,1)
        # transform from range [0, 1] to (0, 1)  # 原源码注释表述略不严谨；实际是从 ROI 尺度坐标除以 roi_size，得到接近 [0,1]
        eps = 1e-5                                                         # 添加极小值，避免理论上的除零，并让边界值略小于 1
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps # 构造归一化分母 [roi_w, roi_h] + eps
        line = line / norm                                                 # 坐标除以 ROI 尺寸，归一化到 (0,1) 【x 除以 roi_w，y 除以 roi_h】
        line = line.flatten(-2, -1)                                        # 将 [S,2] 再 flatten 回 [2S]，方便后续 LinesL1Cost 计算

        # (5) 返回归一化后的 polyline 坐标
        return line  


# 二、定义 map line 的 Hungarian 一对一匹配器：基于成本矩阵实现预测与真值的一对一最优匹配【成本由分类成本 + 回归L1成本加权组成，支持点序置换匹配】
@BBOX_ASSIGNERS.register_module()  # 将 HungarianLinesAssigner 注册到 BBOX_ASSIGNERS，配置中 type='HungarianLinesAssigner' 时可自动构建
class HungarianLinesAssigner(BaseAssigner):
    # 作者英文注释：原始英文 docstring；注意本实现没有返回 AssignResult，而是直接返回匹配索引

    # 1. 根据配置构建匹配成本函数
    def __init__(self, cost=dict, **kwargs): # 传参 cost (dict): 匹配代价配置，通常为 MapQueriesCost，内部包含 cls_cost 和 reg_cost
        self.cost = build_match_cost(cost) # 根据配置构建总 cost 对象，通常是 MapQueriesCost

    # 2. 执行一张图内的 pred line 与 GT line 的匈牙利匹配 Hungarian matching
    def assign(
        self,                  # 当前 assigner 对象
        preds: dict,           # 预测字典，包含 lines（归一化坐标）、scores（分类logits）
        gts: dict,             # GT 字典，包含 lines 和 labels
        ignore_cls_cost=False, # 是否忽略分类代价，只用回归几何代价匹配
        gt_bboxes_ignore=None, # 忽略真值，本实现不支持，必须为 None
        eps=1e-7               # 数值稳定极小值项，当前实现没有直接使用，保留接口兼容
    ):
        # 作者英文注释：原始 docstring 来自 DETR/Hungarian assigner 风格；当前实际返回 row/col index，不返回 AssignResult
        
        # 断言：暂不支持忽略真值的情况
        assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'

        # 读取 GT 数量和预测 query 数量
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)
        
        # 如果 GT 或预测数量为0，返回空匹配【上层 sample 会在无 GT 时跳过该图】
        if num_gts == 0 or num_lines == 0:
            return None, None, None
        
        # (1) 初始化置换索引为None
        gt_permute_idx = None # (num_preds, num_gts)

        # (2) 求最优置换索引
        if self.cost.reg_cost.permute:                                    # 如果回归成本支持点序置换模式 permutation-invariant
            cost, gt_permute_idx = self.cost(preds, gts, ignore_cls_cost) # 则计算成本矩阵，同时返回最优置换索引
        else:                                                             # 如果不需要处理 GT 多排列
            cost = self.cost(preds, gts, ignore_cls_cost)                 # 则仅计算成本矩阵，不做置换

        # (3) 成本矩阵转CPU、转numpy，用于scipy的匈牙利算法
        cost = cost.detach().cpu().numpy()
        
        # (4) 执行匈牙利算法，得到最优匹配的行、列索引
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        
        # (5) 返回预测索引、GT 索引，以及可选的最优 GT 点序排列索引
        """
            matched_row_inds: 匹配的预测索引（行索引）
            matched_col_inds: 匹配的真值索引（列索引）
            gt_permute_idx  : 真值点的置换索引（用于对齐点序）
        """
        return matched_row_inds, matched_col_inds, gt_permute_idx

