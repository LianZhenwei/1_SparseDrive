
import torch                    # 导入 PyTorch。用于 Tensor 计算、逻辑判断、张量拼接、reshape 等
import numpy as np              # 导入 numpy。用于把 cost tensor 转成 numpy 后交给 scipy 的匈牙利匹配函数
import torch.nn.functional as F # 导入 torch.nn.functional。这里主要使用 F.pad 对不同数量的 GT 做 padding
from scipy.optimize import linear_sum_assignment  # 从 scipy 中导入匈牙利匹配算法：linear_sum_assignment 用于根据 cost matrix 找到预测和 GT 的最优一对一匹配
from mmdet.core.bbox.builder import BBOX_SAMPLERS # 导入 mmdet 的 BBOX_SAMPLERS 注册表：SparseBox3DTarget 会注册到这里，配置文件中 sampler=dict(type="SparseBox3DTarget", ...) 时，就会构建这个类
from projects.mmdet3d_plugin.core.box3d import *  # 导入 box3d.py 中定义的 3D box 字段索引，例如 X, Y, Z, W, L, H, YAW, SIN_YAW, COS_YAW 等
from ..base_target import BaseTargetWithDenoising # 导入带 denoising 接口的 target 基类，SparseBox3DTarget 继承它，从而支持 sample、get_dn_anchors、update_dn、cache_dn 等接口

# 控制 from target import * 时暴露的对象
__all__ = ["SparseBox3DTarget"] # 当前文件主要暴露 SparseBox3DTarget


'''
    SparseBox3DTarget 是 3D 检测任务的目标分配器，它主要负责：
        1. 把 GT box 编码成网络内部使用的格式
        2. 根据分类 cost 和 box cost 做 Hungarian matching
        3. 输出每个 query 对应的 cls_target、box_target、reg_weights
        4. 训练时可选生成 denoising anchors
        5. 时序训练时可选缓存和更新 temporal denoising anchors
'''
# 将 SparseBox3DTarget 注册到 BBOX_SAMPLERS 注册表，这样 SparseDrive 配置文件中的 sampler=dict(type="SparseBox3DTarget", ...)，就可以自动实例化这个类
@BBOX_SAMPLERS.register_module()
class SparseBox3DTarget(BaseTargetWithDenoising):
    # 1. 初始化：保存参数
    def __init__(
        self,                      # self 表示当前 SparseBox3DTarget 对象
        cls_weight=2.0,            # 分类匹配 cost 权重
        alpha=0.25,                # Focal Loss 里的 alpha 参数，用于分类 cost 计算
        gamma=2,                   # Focal Loss 里的 gamma 参数，用于分类 cost 计算
        eps=1e-12,                 # 防止 log(0) 的极小值
        box_weight=0.25,           # box 匹配 cost 权重
        reg_weights=None,          # 各个 box 回归维度的权重。若为 None，后面会默认设置
        cls_wise_reg_weights=None, # 按类别设置不同的回归维度权重，例如 traffic_cone 可以不强监督速度
        num_dn_groups=0,           # denoising group 数量。若为 0，表示不启用 DN training
        dn_noise_scale=0.5,        # DN 噪声尺度。可以是一个 float，也可以是每个维度不同的 list
        max_dn_gt=32,              # 每张图最多使用多少个 GT 生成 DN anchors
        add_neg_dn=True,           # 是否额外生成 negative DN anchors
        num_temp_dn_groups=0,      # temporal DN group 数量，用于跨帧缓存一部分 DN anchors
    ):
        # 调用父类 BaseTargetWithDenoising 初始化：父类会保存 num_dn_groups、num_temp_dn_groups，并初始化 self.dn_metas=None
        super(SparseBox3DTarget, self).__init__(num_dn_groups, num_temp_dn_groups)

        # (1) 保存参数：
        self.cls_weight = cls_weight # 保存分类 cost 权重
        self.box_weight = box_weight # 保存 box cost 权重
        self.alpha = alpha           # 保存 focal alpha
        self.gamma = gamma           # 保存 focal gamma
        self.eps = eps               # 保存 eps，防止 log 数值问题

        # (2) 保存回归权重参数：
        self.reg_weights = reg_weights                   # a. 保存回归维度权重
        if self.reg_weights is None:                     # 若没有传入 reg_weights
            self.reg_weights = [1.0] * 8 + [0.0] * 2     # 则默认前 8 维参与匹配，后 2 维不参与匹配，这10维的具体维度含义取决于 box3d.py 中的状态定义
        self.cls_wise_reg_weights = cls_wise_reg_weights # b. 保存按类别设置的回归维度权重  

        # (3) 保存dn相关参数：      
        self.dn_noise_scale = dn_noise_scale # 保存 DN 噪声尺度
        self.max_dn_gt = max_dn_gt           # 保存每张图最多用于 DN 的 GT 数量
        self.add_neg_dn = add_neg_dn         # 保存是否添加 negative DN anchors

    # 2. 将 GT box 编码成网络内部回归格式
    def encode_reg_target(self, box_target, device=None):
        '''
            encode_reg_target() 将原始 GT box 编码格式转化成网络内部回归格式：
                原始 GT box 通常格式类似：[x, y, z, w, l, h, yaw, vx, vy, ...]
                网络内部的格式会变成：[x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, ...]  
                因此后续的 DN 加噪不是直接对原始 GT box 格式操作，而是对已经编码后的内部 box state 操作          
        '''
        # outputs 用于保存 batch 中每张图的编码结果
        outputs = []

        # (1) 遍历 batch 中每张图的 GT boxes，拼接编码后的 box
        for box in box_target:
            # (a) 拼接编码后的 box
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],                    # 位置 x,y,z 保持原样
                    box[..., [W, L, H]].log(),              # 尺寸 w,l,h 取 log，这样网络预测时可以在 log-space 中回归尺寸
                    torch.sin(box[..., YAW]).unsqueeze(-1), # yaw 编码成 sin(yaw)
                    torch.cos(box[..., YAW]).unsqueeze(-1), # yaw 编码成 cos(yaw)
                    box[..., YAW + 1 :],                    # yaw 后面的其他状态保持原样，通常包括速度 vx, vy 等
                ],
                dim=-1, # 在最后一维拼接
            )

            # (b) 若指定了 device，则将编码后的 target 移动到指定设备
            if device is not None:
                output = output.to(device=device) # 将编码后的 target 移动到指定设备

            # (c) 保存当前样本编码结果
            outputs.append(output)

        # (2) 返回 list，每个元素对应一张图
        return outputs

    # 3. 普通检测目标分配流程
    def sample(
        self,       # self 表示当前 target 对象
        cls_pred,   # 分类预测，shape 通常为 [B, num_query, num_cls]
        box_pred,   # box 预测，shape 通常为 [B, num_query, box_dim]
        cls_target, # GT 类别标签 list，cls_target[i] shape 为 [num_gt_i]
        box_target, # GT box list，box_target[i] shape 为 [num_gt_i, box_dim]
    ):
        # 读取 batch size、预测 query 数量、类别数
        bs, num_pred, num_cls = cls_pred.shape

        # (1) 计算分类匹配 cost：返回 list，长度为 bs，每个 cost[i] shape 为 [num_pred, num_gt_i]
        cls_cost = self._cls_cost(cls_pred, cls_target)

        # (2) 将 GT box 编码成网络内部格式，并移动到 box_pred 所在设备
        box_target = self.encode_reg_target(box_target, box_pred.device)

        # (3) 设置每个 GT 的回归维度权重
        instance_reg_weights = []        # 保存每个 GT 的回归维度权重
        for i in range(len(box_target)): # 遍历 batch 中每张图
            # a. 对 box_target 中非 NaN 的位置赋予 1，NaN 的位置赋予 0，这样某些无效维度可以不参与回归
            weights = torch.logical_not(box_target[i].isnan()).to(dtype=box_target[i].dtype)

            # b. 若配置了按类别设置回归权重
            if self.cls_wise_reg_weights is not None:
                # 遍历每个特殊类别及其权重
                for cls, weight in self.cls_wise_reg_weights.items():
                    # 若当前 GT 属于该类别，就使用指定的 weight，否则保持原 weights
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None], # 当前 GT 类别是否等于 cls
                        weights.new_tensor(weight),      # 指定类别的回归权重
                        weights,                         # 默认权重
                    )

            # c. 保存当前样本的回归权重
            instance_reg_weights.append(weights)

        # (4) 计算 box 匹配 cost：返回 list，长度为 bs，每个 cost[i] shape 为 [num_pred, num_gt_i]
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        # (5) 进行匈牙利匹配：
        indices = []        # 保存每张图的匹配结果
        for i in range(bs): # 遍历 batch 中每张图
            # 若当前样本有 GT，分类 cost 和 box cost 都存在
            if cls_cost[i] is not None and box_cost[i] is not None:
                # 总 cost = 分类 cost + box cost
                # detach 后转到 CPU，再转 numpy，供 scipy 匈牙利算法使用
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()

                # 将 -inf 或 NaN 的 cost 替换成一个很大的值
                # 避免 linear_sum_assignment 出错
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)

                # 匈牙利匹配
                # assign 是二元组：
                # assign[0]：被选中的 pred index
                # assign[1]：对应的 target index
                assign = linear_sum_assignment(cost)

                # 将 numpy index 转成 torch tensor，并保存
                indices.append([cls_pred.new_tensor(x, dtype=torch.int64) for x in assign])

            # 若当前样本没有 GT，则匹配结果设为 [None, None]
            else:
                indices.append([None, None]) # 匹配结果设为 [None, None]

        # (6) 整理分类target、box target、回归权重，并返回
        # (a1) 初始化输出分类 target
        # shape: [B, num_pred]
        # 默认值为 num_cls，表示 background 类
        output_cls_target = cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls

        # (a2) 初始化输出 box target
        # shape 和 box_pred 相同
        # 未匹配 query 的 box target 默认为 0
        output_box_target = box_pred.new_zeros(box_pred.shape)

        # (a3) 初始化输出回归权重
        # 未匹配 query 的权重为 0，因此不会参与 box loss
        output_reg_weights = box_pred.new_zeros(box_pred.shape)

        # (b) 遍历每个样本的匹配结果，将匹配到 GT 的 query 填入对应类别、对应box target、对应回归权重
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0: # 若当前样本没有 GT，则跳过
                continue # 跳过
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]            # 对匹配到 GT 的 query 填入对应类别
            output_box_target[i, pred_idx] = box_target[i][target_idx]            # 对匹配到 GT 的 query 填入对应 box target
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][target_idx] # 对匹配到 GT 的 query 填入对应回归权重

        # (c) 缓存匹配 indices
        # motion planning loss 里可能会复用 det_head.sampler.indices
        self.indices = indices

        # (d) 返回分类target、box target、回归权重
        return output_cls_target, output_box_target, output_reg_weights
        '''
            output_cls_target:
                shape [B, num_query]
                匹配到 GT 的 query 填真实类别
                没匹配到的 query 填 num_cls，表示 background

            output_box_target:
                shape [B, num_query, box_dim]
                匹配到 GT 的 query 填对应 GT box
                没匹配到的 query 为 0

            output_reg_weights:
                shape [B, num_query, box_dim]
                匹配到 GT 的 query 有回归权重
                没匹配到的 query 权重为 0
        '''

    # 4.1 计算分类匹配 cost
    def _cls_cost(self, cls_pred, cls_target):
        '''
            计算分类匹配 cost，使用的是 Focal Loss 风格的分类代价
            
            输入：
                cls_pred: [B, num_pred, num_cls]
                cls_target: list，每个元素 shape [num_gt_i]
            输出：
                cost: list，每个元素 shape [num_pred, num_gt_i]
        '''

        # =============================================================================
        # 【_cls_cost() 的三段逻辑：为 Hungarian matching 构造“类别是否匹配”的代价矩阵】
        # (1) 将网络输出的分类 logits 转成每个类别的 sigmoid 概率，并准备 batch 级结果容器。
        # (2) 对 batch 中每张图：
        #     (2.1) 若该图没有 GT，则返回 None；sample() 会跳过该图的 Hungarian matching。
        #     (2.2) 依据 Focal Loss 的正 / 负两项，分别计算每个 prediction 对每个类别的代价。
        #     (2.3) 仅抽取每个 GT 自己所属类别的代价，形成 [num_pred, num_gt_i] cost matrix。
        # (3) 返回每张图各自的分类 cost，后续会与 _box_cost() 相加后一起做最小代价匹配。
        #
        # 【cost 的直觉】
        # - 对某一个 GT 类 c：若某 query 对类别 c 的概率越高，则它的 pos_cost 越小、neg_cost
        #   越大，因此 pos_cost - neg_cost 越小，更容易在 Hungarian matching 中被分给该 GT。
        # - 此处不直接计算分类 loss；它只回答“哪一个 query 最像哪一个 GT 类别”，供匹配使用。
        # =============================================================================

        # (1) 读取 batch size，并对分类 logits 做 sigmoid 变为独立类别概率。
        bs = cls_pred.shape[0]        # cls_pred 原始 shape：[B, num_pred, num_cls]，元素是未归一化 logits。
        cls_pred = cls_pred.sigmoid() # sigmoid 后 shape 不变，元素变为每个类别独立的前景概率 p。
        
        # (2) 逐图生成分类 cost matrix
        cost = []           # 保存每个样本的 cost，cost 是长度为 B 的 list；第 i 个元素为 None 或 [num_pred, num_gt_i]
        for i in range(bs): # 遍历 batch
            # 若当前样本没有 GT，则当前图无需 query-to-GT assignment。
            if len(cls_target[i]) > 0:
                # (a)为当前图中的每个 query、每个类别计算 Focal 风格的两类代价
                # (a1) 负样本 cost
                # 这是 focal loss 中负样本项：-log(1-p) * (1-alpha) * p^gamma
                # 当某类别真实标签为负类时，预测概率 p 越大，惩罚越大。
                # shape：[num_pred, num_cls]
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )

                # (a2) 正样本 cost
                # 这是 focal loss 中正样本项：-log(p) * alpha * (1-p)^gamma
                # 当某类别真实标签为正类时，预测概率 p 越小，惩罚越大。
                # shape：[num_pred, num_cls]。
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )

                # (b) 对每个 GT 类别取对应类别的 cost（即按每个 GT 的真实类别抽取对应列，并形成 prediction × GT 的代价矩阵）
                # cls_target[i] shape：[num_gt_i]，例如 [2, 0, 7]
                # pos_cost[:, cls_target[i]] shape: [num_pred, num_gt_i]，第 (q, g) 项表示“query q 分给 GT g 的类别代价”
                # neg_cost[:, cls_target[i]] shape: [num_pred, num_gt_i]，第 (q, g) 项表示“query q 分给 GT g 的类别代价”
                #
                # 这里使用 pos_cost - neg_cost 作为匹配分类代价，pos_cost - neg_cost 是 DETR 系列 Focal matching cost 的常见写法：对 GT 类别预测得越像前景，总代价越小。
                cost.append(
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                    * self.cls_weight
                )

            # 若当前样本没有 GT，则 cost 设为 None（用 None 表示“无需匹配”，而不是构造空矩阵）
            else:
                cost.append(None) # cost 设为 None

        # (3) 返回 batch 级分类 cost list；sample() 会将它与 box cost 相加。
        return cost
    
    # 4.2 计算 box 匹配 cost
    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        '''
            计算 box 匹配 cost，使用加权 L1 距离作为匹配代价
            
            box_pred: [B, num_pred, box_dim]
            box_target: list，每个元素 [num_gt_i, box_dim]，或者 tensor [B, num_gt, box_dim]
            instance_reg_weights: list 或 tensor，和 box_target 对应        
        '''

        # =============================================================================
        # 【_box_cost() 的2段逻辑：为 Hungarian matching 构造“几何是否匹配”的代价矩阵】
        # (1) 对 batch 中每张图：
        #     (a) 若该图没有 GT，则返回 None。
        #     (b) 用 broadcast 同时计算每个 prediction 与每个 GT 在每个 box state 维度上的绝对误差。
        #     (c) 依次乘“该 GT 的有效维度 / 类别专属权重”和“全局回归维度权重”，再沿 state 维求和。
        # (2) 返回每张图的 [num_pred, num_gt_i] 加权 L1 cost；sample() 会与分类 cost 相加。
        #
        # 【注意】
        # - box_target 在 sample() 中已经经过 encode_reg_target()，因此尺寸处于 log-space，
        #   yaw 处于 [sin(yaw), cos(yaw)] 表示；这里比较的是网络内部 box state，而不是原始 box 格式。
        # - instance_reg_weights[i] 由 GT 的 NaN 状态和 cls_wise_reg_weights 决定；
        #   self.reg_weights 是全局的维度权重。两者相乘后，某些维度可以被忽略或弱化。
        # =============================================================================

        # 获取 batch size
        bs = box_pred.shape[0] # box_pred shape：[B, num_pred, box_dim]。

        # (1) 逐图计算 prediction × GT 的几何代价。
        cost = []           # 保存每个样本的 box cost，即保存每张图 cost matrix 的 list
        for i in range(bs): # 遍历 batch
            # 若当前样本有 GT，有 GT 才能构造 pairwise box cost
            if len(box_target[i]) > 0:
                # (a) 利用 broadcast 得到所有 prediction 与所有 GT 的逐维绝对误差（即 L1 距离）：
                #     box_pred[i, :, None] shape: [num_pred, 1, box_dim]
                #     box_target[i][None] shape: [1, num_gt_i, box_dim]
                #     广播相减后                 : [num_pred, num_gt_i, box_dim]。
                # (b) 施加两级回归维度权重，并沿 box_dim 求和：
                #     (b.1) 乘 instance_reg_weights[i][None]：[1, num_gt_i, box_dim]，过滤无效维度或特殊类别维度（即屏蔽 NaN / 未标注维度），并支持类别专属权重。
                #     (b.2) 乘 self.reg_weights：[box_dim]，全局回归维度权重，控制所有类别共有的全局状态维度重要性。
                #     (b.3) sum(dim=-1)：从 [num_pred, num_gt_i, box_dim] → 得到 [num_pred, num_gt_i]。
                # (c) 乘 self.box_weight：调节几何 cost 与分类 cost 在 Hungarian matching 中的相对比例。
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None])
                        * instance_reg_weights[i][None]
                        * box_pred.new_tensor(self.reg_weights),
                        dim=-1,
                    )
                    * self.box_weight
                )

            # 若当前样本没有 GT，没有 GT 时无需做 prediction-to-GT 几何匹配，则 cost 设为 None
            else:
                cost.append(None) # cost 设为 None

        # (2) 返回 batch 级 box cost list；每个非 None 元素 shape 为 [num_pred, num_gt_i]
        return cost

    # 5.1 由 GT anchors 生成训练专用的 denoising anchors
    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        '''
            DN：
                DN training 的思想：把 GT box 加噪声，得到 noisy anchors，然后让模型学习把 noisy anchors 还原到 GT。
                DN（Denoising）的目标不是让模型“从零发现目标”，而是让模型学习：noisy GT anchor → 恢复正确 box，并给出正确类别。
        
            get_dn_anchors() 作用：由 GT 生成训练专用的 DN queries / targets / mask
            get_dn_anchors() 的输入输出：
                输入(即接收)：
                    cls_target    : 当前帧 GT 类别
                    box_target    : 当前帧 GT 3D box
                    gt_instance_id: 当前帧 GT 的真实 instance ID，可选【注意这里的 gt_instance_id 是数据集提供的真实跨帧物体标识，例如同一辆车在相邻帧中的 GT ID，它是训练标签，用于 Temporal DN 跨帧对齐】
                输出(即返回)：
                    dn_anchor    : [B, N_dn, D]，N_dn 是 DN anchor 的数量。它是 GT 加噪声后得到的 DN anchor，它后续会拼接到 normal 900 个 anchor 的尾部
                    dn_box_target: [B, N_dn, D]。它是原始 GT box，作为回归目标（即 DN 回归监督目标）
                    dn_cls_target: [B, N_dn]，其中 >=0 为正类、-3 为有效负 DN、-1 为无效或 padding。它是原始 GT 类别，作为分类目标
                    attn_mask    : [N_dn, N_dn]，True=禁止 attention。它控制 DN query 之间的 attention 隔离
                    valid_mask   : [B, N_dn]。它决定哪些 DN query 参与 DN classification loss，即它决定哪些 DN 位置确实有效
                    dn_id_target : [B, N_dn] 或 None。它是 temporal DN 时的 GT instance ID 对应关系，即 Temporal DN 用它跨帧识别同一个 GT instance。
        '''
        # (1) DN 开关、Temporal DN 开关和 GT 数量上限
        # (1.a) 未配置 DN group 时不创建任何辅助 query，即不启用 DN，直接返回 None
        if self.num_dn_groups <= 0: # 若 DN group 数量小于等于 0，这意味着未配置 DN group
            return None             # 不启用 DN，直接返回 None

        # (1.b) 只有 Temporal DN 才需要 GT instance_id 作为跨帧对齐键；普通 DN 只需单帧 box / class supervision，因此主动丢弃传入的 instance ID
        if self.num_temp_dn_groups <= 0: # 若 temporal DN group 数量小于等于 0，这意味着这是普通 DN
            gt_instance_id = None        # 不需要 instance id 做 temporal DN 对齐

        # (1.c) 若限制每张图最多保留 max_dn_gt 个用于 DN 的 GT，避免某些 GT 特别多的图导致 N_dn 失控：
        if self.max_dn_gt > 0:                                     # 若限制每张图最多用于 DN 的 GT 数量
            cls_target = [x[: self.max_dn_gt] for x in cls_target] # 每张图最多取前 max_dn_gt 个类别 target
            box_target = [x[: self.max_dn_gt] for x in box_target] # 每张图最多取前 max_dn_gt 个 box target
            if gt_instance_id is not None:                         # 若有 instance id，也同步截断
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]


        # (2) 将每张图 GT 数不同的 list 对齐为 batch tensor
        '''
            (2) 先统一 batch 内的 GT 数量：
                    一张图可能有 20 个 GT，另一张图可能有 45 个 GT。
                    代码会先找到当前 batch 中最大的 GT 数：P = max_dn_gt
                    然后把较少 GT 的样本 pad 到 P：
                        类别 padding：-1
                        box padding：0
                        instance_id padding: -1
                    于是：
                        原本：
                            图 A：20 个 GT
                            图 B：45 个 GT
                        padding 后：
                            图 A：[45 个位置，其中 25 个是 padding]
                            图 B：[45 个真实 GT]
                    后续 valid_mask 会告诉 loss：哪些 DN slot 是真实样本，哪些只是 padding。
        '''
        # (2.a) max_dn_gt 是截断后 batch 中最多的 GT 数，它决定每个 DN group 的基础长度
        max_dn_gt = max([len(x) for x in cls_target]) # 找到 batch 中最多 GT 数量
        if max_dn_gt == 0:                            # 若整个 batch 没有 GT
            return None                               # 无法生成 DN anchors

        # (2.b) 对类别目标 cls_target、box目标 box_target、gt_instance_id进行 padding
        # (2.b.1) 对类别目标 cls_target 做 padding：类别 target 右侧补 -1 padding，补齐到 max_dn_gt 长度，其 shape 为 [B, max_dn_gt]，-1 表示该位置不是实际 GT
        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1) # padding 的类别为 -1，表示 pad 无效项
                for x in cls_target
            ]
        )

        # (2.b.2) 对box目标 box_target 做 padding：box target 右侧补 0 padding，补齐到 max_dn_gt 长度，其 shape 为 [B, max_dn_gt, state_dims]，-1 表示该位置不是实际 GT
        box_target = self.encode_reg_target(box_target, cls_target.device)                          # 先将 box_target 编码成网络内部格式，encode_reg_target() 后的状态通常为 [x,y,z,log(w),log(l),log(h),sin(yaw),cos(yaw),vx,vy,...]
        box_target = torch.stack([F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]) # 再对 box_target 做 padding，每张图补齐到 max_dn_gt 个 box
        box_target = torch.where(cls_target[..., None] == -1, box_target.new_tensor(0), box_target) # 再对 pad 的位置，把 box_target 置 0【即对 cls=-1 的 padding slot 强制置零 box】，防止 padding 残值干扰 matching cost
        
        # (2.b.3) 同理，对 GT instance_id 使用与 cls_target 和 box_target 完全相同的 padding 规则
        # 若启用 Temporal DN 时，则对 gt_instance_id 做 padding：gt_instance_id 右侧补 -1 padding，补齐到 max_dn_gt 长度，其 shape 为 [B, max_dn_gt]，-1 表示该位置不是实际 GT
        if gt_instance_id is not None:                                                                              # 若有 GT instance id
            gt_instance_id = torch.stack([F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1) for x in gt_instance_id]) # 对 instance id 做 padding，pad 值为 -1


        # (3) 复制 groups，并产生正 noisy anchors 和负 noisy anchors
        '''
            (3) 复制多个 DN group：
                    (3.1) DN anchor 数量：
                        设：
                            G = num_dn_groups
                            P = 当前 batch 中 padding 后的 max GT 数
                        代码会把 GT 复制 G 份：
                            每个 GT 会被构造成多个独立的 noisy DN 样本。
                        原因是同一个 GT 只加一次噪声，训练信号太少；复制多个 group 后，每份都加不同随机噪声，模型会看到同一 GT 周围不同程度的扰动。
                        不加 negative DN 时：
                            N_dn = G × P
                        若启用 add_neg_dn=True，每个 group 还会增加一份更远的负 DN anchor：
                            N_dn = G × 2P
                    
                    (3.2) 正 DN anchor 与 负 DN anchor 是什么：
                        正 DN anchor：小噪声 GT
                            也就是：
                                GT box + 小随机扰动 → positive DN anchor
                            例如：
                                真实车框中心：x=20.0m，y=3.0m
                                DN anchor： x=20.4m，y=2.7m
                            训练目标仍然是原始真实框：
                                DN anchor：带噪起点
                                DN box target：真实 GT box
                            因此模型被训练成“把轻度偏移的框拉回正确位置”。

                        Negative DN anchor：大噪声版本
                            若：
                                add_neg_dn=True
                            代码还会构造更远的 noisy box：
                                noise_neg = torch.rand_like(box_target) + 1
                            噪声绝对值范围更大，大致对应：
                                比正 DN anchor 更偏离 GT 的 anchor
                            它们的作用是训练分类器：
                                这个 query 虽然在某个 GT 附近构造出来，但偏得太远，不应该被当成可靠正目标。
                        
                        代码中：
                            正 DN：
                                有真实类别和真实 box 回归 target。
                            负 DN：
                                有分类监督，但不会参与 box regression【意思是负DN只会被分类为“背景”，-3就表示背景，因此有分类监督；但“背景”并不需要box框出来，因此负样本只有分类监督、而不会参与 box regression】。
                            这里 dn_cls_target == -3 是该实现对 negative DN 的特殊标记；
                            后续 valid_mask 允许负 DN参与分类 loss，而 dn_pos_mask = dn_cls_target >= 0 会把负 DN排除出回归 loss。
        '''
        # (3.a) 依次读取 batch size、GT 数量【此处的 num_gt 是“每个 group 当前拥有的 DN slot 数”，若 add_neg_dn=True，后面会翻倍】、box 状态维度
        bs, num_gt, state_dims = box_target.shape

        # (3.b) 将 batch 中每张图的 GT 复制 num_dn_groups 份：复制后的第 0 维布局为 [group0 的 B 个样本, group1 的 B 个样本, ...]
        if self.num_dn_groups > 1: # 若 DN group 大于 1
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None: # 若有 instance id，也同步复制
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)
            '''
                将 cls_target、box_target、gt_instance_id 按 DN group 复制：
                    cls_target 原 shape: [B, num_gt]
                    cls_target 进行 tile 后 shape: [num_dn_groups * B, num_gt]
                    box_target 原 shape: [B, num_gt, state_dims]
                    box_target 进行 tile 后 shape: [num_dn_groups * B, num_gt, state_dims]
            '''

        # (3.c) positive DN：在每个编码后 GT box 周围加 [-1,1]×scale 的随机噪声
        # noise shape 和 dn_anchor shape：[num_dn_groups*B, num_gt, state_dims]
        noise = torch.rand_like(box_target) * 2 - 1         # 生成 [-1, 1] 范围内的随机噪声
        noise *= box_target.new_tensor(self.dn_noise_scale) # 按 dn_noise_scale 缩放噪声
        dn_anchor = box_target + noise                      # 正向 DN anchor：GT + 随机噪声

        # (3.d) 可选 negative DN：使用幅值 [1,2)×scale 的更大噪声，并随机赋予正负方向
        # negative DN 的分类 target 会保持 -3，其作用是教模型识别“远离 GT 的 noisy query 不是正目标”
        if self.add_neg_dn: # 若添加 negative DN anchors
            noise_neg = torch.rand_like(box_target) + 1                                                              # 生成 [1, 2) 范围的噪声，这个噪声比正向噪声更大，用于构造负样本
            flag = torch.where(torch.rand_like(box_target) > 0.5, noise_neg.new_tensor(1), noise_neg.new_tensor(-1)) # 随机生成正负号
            noise_neg *= flag                                                                                        # 加上正负方向
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)                                                  # 按 dn_noise_scale 缩放 negative 噪声
            
            # 每个 group 的 DN slot 由 num_gt 扩为 2*num_gt：前后两段分别来自正 / 负噪声
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1) # 拼接 positive DN anchor 和 negative DN anchor
            num_gt *= 2                                                       # 每组 DN 中 GT 数量翻倍


        # (4) 用匈牙利匹配将 noisy DN anchors 与 GT 监督目标对齐
        '''
            (4) 为什么 DN 内部还要 Hungarian matching
                    你可能会疑惑：
                        DN 不是已经知道 GT 了吗？
                        为什么 get_dn_anchors() 里还要 Hungarian matching？
                    此处的 Hungarian matching 不是普通检测分支那种：
                        900 个自由 query ↔ 当前帧所有 GT
                    而是 DN 生成器内部做的：
                        多个 noisy DN anchor ↔ 当前 DN group 中的 GT
                    它根据 box L1 cost 找到一对一对应关系，然后填入：
                        dn_box_target
                        dn_cls_target
                        dn_id_target
                    作用是保证：
                        每个被当成 positive 的 DN anchor 明确对应某一个 GT。
                    若加入了 negative DN，则通常只有一部分 anchor 被匹配成正样本，剩余的有效 anchor 保留为 negative DN。
        '''
        # (4.a.1) 计算 DN anchor 与 GT box 的 box cost【这里 instance_reg_weights 使用全 1，表示所有维度都参与，因此 box_cost 是使用所有维度权重为 1 的加权 L1 box cost】；DN assignment 不依赖分类预测
        # box_cost[i] shape：[num_gt, 原始max_dn_gt]；若有 negative DN，行数大于列数
        box_cost = self._box_cost(dn_anchor, box_target, torch.ones_like(box_target))

        # (4.a.2) 初始化“尚未获得监督”的 DN target 为：dn_box_target=0；dn_cls_target=-3；若有 temporal DN，dn_id_target=-1【后续 Hungarian 选中的 anchors 才会被写入真实 GT target】
        dn_box_target = torch.zeros_like(dn_anchor)         # 初始化 DN box target，初始化为全 0
        dn_cls_target = -torch.ones_like(cls_target) * 3    # 初始化 DN cls target，默认值为 -3，后面 valid_mask 中会把部分 -3 作为有效 negative DN 项
        if gt_instance_id is not None:                      # 若有 instance id
            dn_id_target = -torch.ones_like(gt_instance_id) # 初始化 DN id target 为 -1

        # (4.a.3) 若存在 negative DN，则将 target 容器也扩成 2*num_gt_base，与 dn_anchor 一一对齐
        if self.add_neg_dn: # 若添加了 negative DN
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1) # dn_cls_target 也要复制一份，和正负 DN anchor 对齐
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1) # id target 同步复制

        # (4.b) 对每个 group×batch 样本独立执行 Hungarian：Hungarian 的行是 DN anchor，列是 GT，每个真实 GT 最终对应一个最小几何代价 DN anchor
        for i in range(dn_anchor.shape[0]):                                  # 遍历每一个复制后的 batch/group 样本
            cost = box_cost[i].cpu().numpy()                                 # 取当前样本的 cost matrix，并转到 CPU numpy
            anchor_idx, gt_idx = linear_sum_assignment(cost)                 # 对 DN anchors 和 GT 做匈牙利匹配
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64) # 转回 tensor index
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)         # 转回 tensor index

            # (b.1) 正匹配 DN：写入要恢复的 encoded GT box
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx] # 根据匹配关系填入 DN box target

            # (b.2) 正匹配 DN：写入真实类别；padding GT 的类别为 -1，后面会通过 valid_mask 排除
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx] # 根据匹配关系填入 DN cls target

            # (b.3) Temporal DN：同时记录 GT instance_id，作为下一帧通过 ID 对齐 DN target 的键
            if gt_instance_id is not None: # 若有 instance id，则根据匹配关系填入 DN id target
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]


        # (5) 从 group-major 布局恢复为 batch-major 展平布局【即将 [group, batch, per_group_dn] 重新整理为 head.forward() 所需的 [batch, total_dn]】：
        # (5.a) 统一结果：将 dn_anchor、dn_box_target、dn_cls_target 和 dn_id_target 均从 [num_dn_groups * B, num_gt, state_dims] 整理成 [B, num_dn_groups * num_gt, state_dims]
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        ) # 整理 dn_anchor： → reshape → [num_dn_groups, B, num_gt, ...] → permute → [B, num_dn_groups, num_gt, ...] → flatten → [B, total_dn, ...]
        dn_box_target = ( 
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        ) # 同样整理 dn_box_target
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        ) # 同样整理 dn_cls_target
        if gt_instance_id is not None: # 若有 instance id，则同样整理 dn_id_target【DN id 需与 DN anchor 使用同一套 reshape / permute / flatten，以保证 slot 对齐】
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            ) # 同理整理 dn_id_target
        else: # 若没有 instance id，则 dn_id_target 设为 None
            dn_id_target = None

        # (5.b) 正 DN：类别 >=0，表示该 slot 被匹配到一个真实、非 padding 的 GT
        valid_mask = dn_cls_target >= 0 # valid_mask 表示有效正样本 DN 项，类别 >= 0 说明匹配到了真实类别

        # (5.c) negative DN：类别保持 -3，但只要其来源 GT 不是 padding，它仍是有效 DN 分类样本
        if self.add_neg_dn: # 若添加了 negative DN
            # 将原始 cls_target 也扩展到和 DN flattened 格式一致
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )

            # valid_mask 额外包含：
            # 原位置不是 pad，即 cls_target >= 0；
            # 但 dn_cls_target == -3，表示 negative DN
            valid_mask = torch.logical_or(valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))) # valid denotes the items is not from pad.


        # (6) 构造 DN attention mask，以隔离不同 DN group，避免各组 noisy copies 相互泄露信息
        '''
            (6) DN attention mask 是为了防止“答案泄漏”：
                    get_dn_anchors() 最后返回：
                        attn_mask
                    其逻辑是：
                        同一个 DN group 内：允许 attention
                        不同 DN group 间：禁止 attention
                    然后 detection3d_head.py 会构造一个更大的 attention mask：
                        normal ↔ normal：允许
                        normal ↔ DN：禁止
                        DN group A ↔ DN group B：禁止
                        同一个 DN group 内：允许
                    原因是：
                        DN query 的初始位置很接近 GT；
                        若普通 query 可以任意读取 DN query，
                        普通 query 就可能偷看 DN 的“答案线索”；
                        训练结果会虚高，但真实推理时没有 DN query 可用。
                    因此必须隔离 DN 和 normal query。
        '''
        # (6.a) 构造 DN attention mask：先初始全部置 1，最终 True 表示禁止 attention，shape: [num_gt*num_dn_groups, num_gt*num_dn_groups]，
        attn_mask = dn_box_target.new_ones(num_gt * self.num_dn_groups, num_gt * self.num_dn_groups)

        # (6.b) 同一 group 内的 attn_mask 置 0：允许同组 DN queries 互相 attention
        for i in range(self.num_dn_groups):     # 遍历每个 DN group
            start = num_gt * i                  # 当前 group 起点
            end = start + num_gt                # 当前 group 终点
            attn_mask[start:end, start:end] = 0 # 同一 DN group 内部允许互相 attention，所以 mask 置 0

        # (6.c) 将 attn_mask 转为 bool，将 dn_cls_target 转为 long【后续在 detection3d_head.py 中会被嵌入完整的 normal+DN attention mask】
        attn_mask = attn_mask == 1           # 转成 bool mask，True 表示禁止 attention，False 表示允许 attention
        dn_cls_target = dn_cls_target.long() # 分类 target 转成 long 类型


        # (7) 返回 DN 所需的全部信息【这是供 head.forward() 拼接、构造 loss 和 Temporal DN 使用的全部状态】
        return (
            dn_anchor,     # DN anchors，shape [B, total_dn, state_dims]
            dn_box_target, # DN box targets，shape [B, total_dn, state_dims]
            dn_cls_target, # DN cls targets，shape [B, total_dn]
            attn_mask,     # DN attention mask，shape [total_dn, total_dn]
            valid_mask,    # DN 有效 mask，shape [B, total_dn]
            dn_id_target,  # DN instance id target，可能为 None
        )
        '''
            【衔接 detection3d_head.py 里调用 get_dn_anchor() 的返回值含义】
            dn_anchor    : [B, N_dn, D]，N_dn 是 DN anchor 的数量。它是 GT 加噪声后得到的 DN anchor，它后续会拼接到 normal 900 个 anchor 的尾部
            dn_box_target: [B, N_dn, D]。它是原始 GT box，作为回归目标（即 DN 回归监督目标）
            dn_cls_target: [B, N_dn]，其中 >=0 为正类、-3 为有效负 DN、-1 为无效或 padding。它是原始 GT 类别，作为分类目标
            attn_mask    : [N_dn, N_dn]，True=禁止 attention。它控制 DN query 之间的 attention 隔离
            valid_mask   : [B, N_dn]。它决定哪些 DN query 参与 DN classification loss，即它决定哪些 DN 位置确实有效
            dn_id_target : [B, N_dn] 或 None。它是 temporal DN 时的 GT instance ID 对应关系，即 Temporal DN 用它跨帧识别同一个 GT instance。
        '''

    # 5.2 更新 temporal denoising
    def update_dn(
        self,                # self 表示当前 target 对象
        instance_feature,    # 当前 decoder 中所有 instance feature，包括 normal anchors 和 DN anchors
        anchor,              # 当前 decoder 中所有 anchors
        dn_reg_target,       # DN 回归 target
        dn_cls_target,       # DN 分类 target
        valid_mask,          # DN 有效 mask
        dn_id_target,        # DN instance id target
        num_noraml_anchor,   # normal anchor 数量。注意源码变量名写成 num_noraml_anchor，normal 拼错了
        temporal_valid_mask, # temporal_valid_mask 表示当前 batch 的历史缓存是否有效
    ):
        # =============================================================================
        # 【update_dn() 作用：在单帧 decoder 结束后，把历史 DN groups 接入 temporal decoder】
        #
        # 触发位置：detection3d_head.py 中第一次单帧 refine 后。
        # 此时 InstanceBank.update() 已经先重组 normal instances；本函数只处理尾部的 DN instances。
        #
        # update_dn() 的5段逻辑：
        #     (1) 检查 temporal DN cache 是否可用；无有效历史时直接退化为当前帧 DN。
        #     (2) 将 normal slots 与 DN slots 分离，并把展平 DN 还原为 [B, group, per_group_dn, ...]。
        #     (3) 用 GT instance_id 把上一帧缓存 DN 与当前帧 GT 对齐，得到当前时刻应使用的 temporal DN targets。
        #     (4) 处理上一帧 / 当前帧每组 DN 数不一致，并按每个 batch 的 temporal_valid_mask 决定使用历史还是当前 DN group。
        #     (5) 把重组后的 DN 重新展平、接回 normal slots，并返回与后续 temporal decoder 对齐的 targets。
        #
        # 【关键语义】
        # - 历史 DN feature / anchor 来自上一帧最终 decoder 的输出，并在下一帧 get() 中已随 ego motion 投影。
        # - 历史 DN 的“回归监督”不能继续使用上一帧坐标；它要通过同一个 GT instance_id 找到当前帧 GT，
        #   因此 temp_reg_target 是当前帧坐标系下的 target。
        # - 返回 DN groups 的前 num_temp_dn_groups 组为 temporal DN；剩余组保持当前帧 newly generated DN。
        # =============================================================================


        # 读取 batch size，读取当前总 anchor 数量 num_anchor【当前总 anchor 数量 = normal数 + DN数】
        bs, num_anchor = instance_feature.shape[:2]

        # (1) 检查历史 Temporal DN 是否存在、当前是否真的包含 DN slots
        # (1.a) 若 temporal_valid_mask 为 None，表示没有可对齐的历史 temporal memory，则主动清空 temporal DN cache，避免下一个分支误用旧序列 / 旧 batch 的 DN 状态
        if temporal_valid_mask is None:
            self.dn_metas = None # 清空 temporal DN cache

        # (1.b) 若没有缓存 DN metas 或当前没有 DN anchors，则不做 temporal replacement，直接返回原输入
        # self.dn_metas is None：上一帧没有缓存可复用 DN；
        # num_noraml_anchor >= num_anchor：当前输入根本没有附加 DN slots。
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            # 直接返回原输入
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # (2) 拆分 normal 与 DN，并将 DN 恢复回 DN group 维度：
        # (2.a) 将 normal instance feature 与 DN instance feature 拆开，将 normal anchors 与 DN anchors 拆开
        num_dn = num_anchor - num_noraml_anchor                    # 当前 DN anchor 数量 = 当前总 anchor 数 - normal anchor 数
        dn_instance_feature = instance_feature[:, -num_dn:]        # 取出 DN instance feature【因为 head.forward() 约定 normal 在前、DN 永远追加在尾部，所以 [:, -num_dn:] 取出的是 DN】
        dn_anchor = anchor[:, -num_dn:]                            # 取出 DN anchor
        instance_feature = instance_feature[:, :num_noraml_anchor] # 只保留 normal instance feature
        anchor = anchor[:, :num_noraml_anchor]                     # 只保留 normal anchor

        # (2.b) 将 DN 相关信息从展平格式 [B, total_num_all_dn, ...] 恢复成 group 格式 [B, num_dn_groups, num_dn_per_group, ...]
        # get_dn_anchors() 输出给 head 的 DN 是 [B, total_num_all_dn, ...]，这里按 group 数重新 view 回 [B, num_dn_groups, num_dn_per_group, ...]
        num_dn_groups = self.num_dn_groups                                   # 当前帧 DN group 数量
        num_dn = num_dn // num_dn_groups                                     # 每个 group 中的 DN 数量【num_dn 原来是 total_num_all_dn，现在它除以 num_dn_groups 后，变成 num_dn_per_group】
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1) # DN feature reshape 成 [B, num_dn_groups, num_dn_per_group, C]
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)         # DN anchor reshape 成 [B, num_dn_groups, num_dn_per_group, state_dim]
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1) # DN reg target reshape
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)     # DN cls target reshape
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)           # valid mask reshape
        if dn_id_target is not None:                                         # 【Temporal DN 必须使用 instance ID 对齐】若有 DN id target，则将 DN id target reshape 成 group 格式
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)


        # (3) 提取历史 DN cache 和当前 DN，并用 GT instance_id 对齐当前帧监督
        # (3.a) 取出上一帧缓存的 DN instance feature
        temp_dn_feat = self.dn_metas["dn_instance_feature"]         # 取出上一帧缓存的 DN instance feature【cache_dn() 保存的历史 DN feature 的 shape 为：temp_dn_feat：[B, num_temp_dn_groups, num_temp_dn, C]】
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3] # 读取 temporal DN group 数量和每组 temporal DN 数量
        
        # (3.b) 取出上一帧缓存的 DN instance id，并构造 temporal DN id 和当前 DN id 的匹配关系 match
        temp_dn_id = self.dn_metas["dn_id_target"]                           # 取出上一帧缓存的 DN id target
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None] # 构造 temporal DN id 和当前 DN id 的匹配关系 match
        '''
            temp_dn_id 是上一帧缓存 DN 的 GT instance ID
            dn_id 的前 num_temp_dn_groups 组是当前帧新建 DN 中对应的 GT instance ID
            match 是历史 DN 与当前 DN 的 ID 相等关系（即匹配关系）：
                temp_dn_id[..., None]              : [B, num_temp_dn_groups, num_temp_dn, 1]
                dn_id[:, :num_temp_dn_groups, None]: [B, num_temp_dn_groups, 1, num_dn]
                match                              : [B, num_temp_dn_groups, num_temp_dn, num_dn]
                match[b,g,i,j]=True 表示历史第 i 个 DN 与当前第 j 个 DN 对应同一 GT instance
        '''

        # (3.c) 根据 instance id 匹配关系，对齐上一帧缓存的 temporal DN 的 temp_reg_target 和 temp_cls_target
        # (3.c.1) 更新 temporal DN reg target：若某个历史 DN id 能在当前帧 DN id 中找到，就把当前对应 reg target 对齐过去
        '''
            temporal DN 的 feature 和 anchor 来自历史；
            但 temporal DN 的 regression target 必须改用当前帧同 ID GT：
                根据 id 匹配关系 match，从当前 DN reg target 中取出 temporal DN 对应的 reg target【即若某个 temporal DN id 能在当前 DN id 中找到，就把当前对应 reg target 对齐过去】，
                再沿 N_cur 维求和，输出 temp_reg_target：[B, G_temp, N_temp, D]。        
        '''
        temp_reg_target = (match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]).sum(dim=3)

        # (3.c.2) 更新 temporal DN cls target：
        # 若某个历史 DN 的 ID 在当前帧不存在，则 cls_target 置为 -1；
        # 否则，即若仍能找到同一 ID，则保留缓存中的该 DN 的原 dn_cls_target（同一 GT 的类别不应改变）。
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )

        # (3.d) 取出上一帧缓存的 valid mask【temp_valid_mask 决定历史 DN loss 的可用性】
        temp_valid_mask = self.dn_metas["valid_mask"]

        # (3.e) 取出上一帧缓存的 DN anchor【temp_dn_anchor 是已由 InstanceBank.get() 投影到当前坐标系的历史 noisy anchor】
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # (3.f) 对六种严格 slot 对齐的 meta 同时做“长度对齐 + 有效性选择 + group 拼接”：[feature, anchor, reg_target, cls_target, valid_mask, instance_id]。
        # (3.f.1) temporal DN metas 列表
        temp_dn_metas = [
            temp_dn_feat,    # temporal DN feature
            temp_dn_anchor,  # temporal DN anchor
            temp_reg_target, # temporal DN reg target
            temp_cls_target, # temporal DN cls target
            temp_valid_mask, # temporal DN valid mask
            temp_dn_id,      # temporal DN id
        ]

        # (3.f.2) 当前帧 DN metas 列表
        dn_metas = [
            dn_feat,       # 当前 DN feature
            dn_anchor,     # 当前 DN anchor
            dn_reg_target, # 当前 DN reg target
            dn_cls_target, # 当前 DN cls target
            valid_mask,    # 当前 DN valid mask
            dn_id,         # 当前 DN id
        ]


        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        # 处理 temporal DN 和当前 DN 数量不一致的问题，然后拼接 temporal DN 和当前 DN

        # (4) 将历史 temporal DN groups 放入当前 DN 序列前缀，并处理不同帧 GT 数导致的长度变化
        output = [] # 保存合并后的输出
        # 遍历 temporal DN metas 和当前 DN metas，进行逐项对齐：每个 meta 的 group 和 slot 顺序必须完全一致，不能只替换 feature 而遗漏 target 或 ID
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)): # 遍历 temporal DN metas 和当前 DN metas
            # (a.1) 若上一帧每组 DN 数更少【即若 temporal 每组 DN 数量小于当前每组 DN 数量】：则对 temp_meta 右侧补 0 到当前 num_dn 长度
            if num_temp_dn < num_dn:
                # 需要 padding 到当前 num_dn 长度
                pad = (0, num_dn - num_temp_dn) 

                # 若 temp_meta 是 4 维，即若 temp_meta 是 feature、anchor、reg target，即[B, G, N, C或D]，则 padding 不能触碰最后的 C 或 D，所以在 pad 参数前补 (0,0)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad # 最后一维是 feature 或 state，则不 pad，只在 num_dn 维度上 pad

                # 若不是 4 维，则要求必须是 3 维，即要求必须是 cls target、valid mask、id target，即 [B, G, N]，则只需要沿 N 维补齐
                else:
                    assert temp_meta.dim() == 3 # 必须是 3 维，例如 cls target、valid mask、id target

                # 对 temporal meta 做 padding
                temp_meta = F.pad(temp_meta, pad, value=0)

            # (a.2) 若上一帧每组 DN 数更多【即若 temporal DN 数量大于等于当前 DN 数量】：则将 temp_meta 截断到当前 num_dn 长度
            else:
                temp_meta = temp_meta[:, :, :num_dn] # 直接截断到当前 num_dn 长度

            # (b) temporal_valid_mask 是每个 batch 样本自己的历史有效标志，shape 通常为 [B]，这里将其 [B] → [B,1,1] 或 [B,1,1,1]，以便广播选择
            mask = temporal_valid_mask[:, None, None] # 扩展成 [B,1,1]，用于选择是否使用 temporal DN
            if meta.dim() == 4:                       # 若 meta 是 4 维，则再扩展一维成 [B,1,1,1]，以适配 [B, G, N, C]
                mask = mask.unsqueeze(dim=-1)         # 再扩展一维，适配 [B,G,N,C]

            # (c) 考虑历史无效的情况下的 temp_meta：
            # 若历史有效，则前 num_temp_dn_groups 组就正常使用历史 temp_meta
            # 若历史无效，则前 num_temp_dn_groups 组退回当前帧 meta 的前 num_temp_dn_groups 组的 freshly generated D
            temp_meta = torch.where(mask, temp_meta, meta[:, :num_temp_dn_groups])

            # (d) 固定输出布局：将 temporal DN group 放在前面，将当前 DN 剩余 group 接在后面【即前 num_temp_dn_groups = temporal DN，后续 groups = 当前帧普通 DN】
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)

            # (e) 再 flatten 恢复回 head.forward() 需要的展平 DN 格式 [B, total_dn, ...]：
            meta = meta.flatten(1, 2) # [B, G, N, ...] → [B, G*N, ...]

            # (f) 保存该 meta
            output.append(meta)
 
        # (5) 将重组后的 DN feature 和 DN anchor 重新拼到 normal instance feature 和 normal anchor 的尾部，得到完整 instance_feature 和完整 anchor，并返回
        output[0] = torch.cat([instance_feature, output[0]], dim=1) # output[0] 是 DN feature，将其拼回 normal instance feature，得到完整 instance_feature【output[0] 的前 normal 部分仍保持 InstanceBank.update() 的结果】
        output[1] = torch.cat([anchor, output[1]], dim=1)           # output[1] 是 DN anchor，将其拼回 normal anchor，得到完整 anchor【output[1] 的前 normal 部分仍保持 InstanceBank.update() 的结果】

        # 返回顺序必须与 detection3d_head.py 的解包顺序一致：instance_feature, anchor, dn_reg_target, dn_cls_target, valid_mask, dn_id_target
        return output

    # 5.3 缓存 temporal denoising：从当前帧最终 DN 状态中随机保留若干 group，以供下一帧 Temporal DN 使用
    def cache_dn(
        self,                # self 表示当前 target 对象
        dn_instance_feature, # DN instance feature
        dn_anchor,           # DN anchor
        dn_cls_target,       # DN cls target
        valid_mask,          # DN valid mask
        dn_id_target,        # DN instance id target
    ):
        # =============================================================================
        # 【cache_dn() 的四段逻辑：从当前帧最终 DN 状态中随机保留若干 group，供下一帧 Temporal DN 使用】
        #
        # (0) 检查是否允许缓存 Temporal DN，并读取当前 DN 的 group 结构。
        # (1) 在 num_dn_groups 个 DN group 中，随机选择 num_temp_dn_groups 个 group。
        # (2) 将选中的 feature / anchor / cls target / valid mask / instance ID 按相同索引 detach 并缓存。
        # (3) 写入 self.dn_metas；下一帧会先由 InstanceBank.get() 对缓存 anchor 做坐标投影，
        #     再由 update_dn() 根据 instance ID 与当前 GT 对齐 supervision。
        #
        # 【为什么随机选 group】
        # 每个 DN group 是同一 GT 集合的独立噪声副本。随机抽取一部分 group 作为 temporal memory，
        # 可以在控制 temporal DN 数量的同时，避免固定只沿用某一组噪声模式。
        # =============================================================================

        # (0) Temporal DN cache 的开关：若 temporal DN group 数量小于 0，则直接return而不缓存【正常调用链中，detection3d_head.py 只有在 num_temp_dn_groups>0 时才会调用本函数；这里仍保留源码的 <0 防御分支，不改动原有行为】
        if self.num_temp_dn_groups < 0:
            return # 不缓存

        # (1) 记录batch_szie，记录“DN总数 = num_dn”、“当前DN group数 = num_dn_groups”、“每个DN group内的DN数 = num_temp_dn”【公式为：num_temp_dn = num_dn // num_dn_groups】
        num_dn_groups = self.num_dn_groups         # 读取当前 DN group 数，由所有 groups 拼接而成
        bs, num_dn = dn_instance_feature.shape[:2] # 读取 batch_size 和 DN 总数，dn_instance_feature：[B, total_dn, C]
        num_temp_dn = num_dn // num_dn_groups      # 计算每个 DN group 内的 DN 数：get_dn_anchors() 保证 total_dn 可被 num_dn_groups 整除

        # (2) 【核心】在 group 维随机选择要跨帧保留的 num_temp_dn_groups 个 groups 进行保留缓存，得到这些 groups 的 mask 即 temp_group_mask
        temp_group_mask = (torch.randperm(num_dn_groups) < self.num_temp_dn_groups)
        '''
            temp_group_mask shape：[num_dn_groups]；
            True 的数量为 num_temp_dn_groups。
        '''
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device) # 将 mask 移动到 DN anchor 所在设备

        # (3) 只保留 temp_group_mask 的 group 的 DN meta 进行缓存【通过对所有彼此对齐的 DN meta 使用同一 temp_group_mask 实现】
        # (a) detach DN feature 和 detach DN anchor，并 reshape 成 group 格式，然后选择 temporal DN groups
        dn_instance_feature = dn_instance_feature.detach().reshape(bs, num_dn_groups, num_temp_dn, -1)[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(bs, num_dn_groups, num_temp_dn, -1)[:, temp_group_mask]
        '''
            DN feature 和 DN anchor 必须 detach：
                cache 是给下一帧读取的 recurrent state，不允许梯度跨帧反传回当前帧计算图。
                [B,total_dn,C_or_D] → [B,G,N,C_or_D] → 选组 → [B,G_temp,N,C_or_D]。
        '''

        # (b) reshape DN cls target 和 valid mask，并选择 temporal DN groups
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[:, temp_group_mask]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[:, temp_group_mask]
        '''
            cls target 与 valid mask 不需要梯度，因此不 detach；
            同样 reshape 并按同一组索引选择，保证缓存第 (g,n) 个 feature / anchor / target 描述的是同一个 DN slot。
        '''

        # (c) reshape DN id target，并选择 temporal DN groups【Temporal DN 的跨帧身份键：同样只保留被选 group 的 GT instance ID】
        if dn_id_target is not None: # 若有 DN id target，则 reshape DN id target，并选择 temporal DN groups
            dn_id_target = dn_id_target.reshape(bs, num_dn_groups, num_temp_dn)[:, temp_group_mask]

        # (d) 只将 temp_group_mask 的 group 的 temporal DN metas 写入缓存 self.dn_metas，以供下一帧 update_dn() 使用这些 temporal DN metas
        '''
            self.dn_metas 中各字段的 group / slot 维严格对齐：
                dn_instance_feature: [B,G_temp,N,C]
                dn_anchor          : [B,G_temp,N,D]
                dn_cls_target      : [B,G_temp,N]
                valid_mask         : [B,G_temp,N]
                dn_id_target       : [B,G_temp,N] 或 None
        '''
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature, # 缓存 DN feature
            dn_anchor=dn_anchor,                     # 缓存 DN anchor
            dn_cls_target=dn_cls_target,             # 缓存 DN cls target
            valid_mask=valid_mask,                   # 缓存 valid mask
            dn_id_target=dn_id_target,               # 缓存 DN id target
        )

