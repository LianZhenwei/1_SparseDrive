import torch                                      # 导入PyTorch核心库
from mmdet.core.bbox.builder import BBOX_SAMPLERS # 导入边界框采样器注册器

# 模块对外暴露的类
__all__ = ["MotionTarget", "PlanningTarget"]


# 工具函数1：为每个实例从多个轨迹模态中选出与真值最接近的模态索引，作为分类监督目标【选优准则：所有有效时间步的 6 个轨迹模态与 GT 轨迹的平均轨迹点距离最小】
def get_cls_target(
    reg_preds,  # 预测的多模态轨迹增量，形状 [bs, num_pred, mode, ts, d]
    reg_target, # 轨迹真值，形状 [bs, num_pred, ts, d]
    reg_weight, # 轨迹有效掩码，形状 [bs, num_pred, ts]，1表示有效，0表示无效
):
    # 解包输入的维度：批次、预测数、模态数、时间步、坐标维度
    bs, num_pred, mode, ts, d = reg_preds.shape

    # (1) 将预测轨迹和真值轨迹从增量累加为绝对坐标
    reg_preds_cum = reg_preds.cumsum(dim=-2)   # 预测轨迹从增量累加为绝对坐标，沿时间步维度累加
    reg_target_cum = reg_target.cumsum(dim=-2) # 真值轨迹同样累加为绝对坐标

    # (2) 计算每个模态与真值的轨迹点距离
    # 真值增加模态维度，与所有预测模态做广播计算距离，形状变化：[bs, num_pred, ts, d] -> [bs, num_pred, 1, ts, d]
    # 与预测 [bs, num_pred, mode, ts, d] 广播相减
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)

    # (3) 乘以有效掩码，无效时间步距离置0，不参与平均
    dist = dist * reg_weight.unsqueeze(2)
    
    # (4) 沿时间步维度求平均，得到每个模态的平均轨迹距离，形状 [bs, num_pred, mode]
    dist = dist.mean(dim=-1)

    # (5) 取距离最小的模态索引，作为分类目标，并返回
    mode_idx = torch.argmin(dist, dim=-1)
    return mode_idx # 返回最优模态索引，形状 [bs, num_pred]，值为0~mode-1的整数


# 工具函数2：选出每个实例对应的最优模态轨迹，作为回归监督的预测值【选优逻辑与get_cls_target逻辑完全一致，只是额外根据索引取出对应的轨迹张量】
def get_best_reg(
    reg_preds,  # 预测的多模态轨迹增量，形状 [bs, num_pred, mode, ts, d]
    reg_target, # 轨迹真值，形状 [bs, num_pred, ts, d]
    reg_weight, # 轨迹有效掩码，形状 [bs, num_pred, ts]
):
    # 解包输入的维度：批次、预测数、模态数、时间步、坐标维度
    bs, num_pred, mode, ts, d = reg_preds.shape

    # (1) 将预测轨迹和真值轨迹从增量累加为绝对坐标
    reg_preds_cum = reg_preds.cumsum(dim=-2)   # 预测轨迹从增量累加为绝对坐标，沿时间步维度累加
    reg_target_cum = reg_target.cumsum(dim=-2) # 真值轨迹同样累加为绝对坐标

    # (2) 计算每个模态与真值的轨迹点距离
    # 真值增加模态维度，与所有预测模态做广播计算距离，形状变化：[bs, num_pred, ts, d] -> [bs, num_pred, 1, ts, d]
    # 与预测 [bs, num_pred, mode, ts, d] 广播相减
    dist = torch.linalg.norm(reg_target_cum.unsqueeze(2) - reg_preds_cum, dim=-1)

    # (3) 乘以有效掩码，无效时间步距离置0，不参与平均
    dist = dist * reg_weight.unsqueeze(2)
    
    # (4) 沿时间步维度求平均，得到每个模态的平均轨迹距离，形状 [bs, num_pred, mode]
    dist = dist.mean(dim=-1)

    # (5) 取距离最小的模态索引，它就是最优模态
    mode_idx = torch.argmin(dist, dim=-1)

    # (6) 取出最优模态对应的轨迹，作为回归目标，并返回
    mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d) # 索引扩展维度，适配gather操作：形状 [bs, num_pred, 1, 1, 1]，广播到所有时间步和坐标维度
    best_reg = torch.gather(reg_preds, 2, mode_idx).squeeze(2)        # 沿模态维度gather，取出最优模态对应的轨迹，去掉模态维度
    return best_reg # 返回最优模态的轨迹预测，形状 [bs, num_pred, ts, d]


# 一、运动预测目标分配器：生成运动预测的分类、回归监督目标
@BBOX_SAMPLERS.register_module() # 注册到边界框采样器
class MotionTarget():
    """
        运动预测目标分配器
        核心逻辑：复用检测分支的匈牙利匹配结果，为每个预测实例分配对应真值轨迹，再选最优模态
        优势：避免重复匹配，保证检测与运动预测的监督对齐，减少计算量
    """

    # 1. 初始化
    def __init__(self):
        super(MotionTarget, self).__init__() # 初始化运动目标分配器，无额外参数

    # 2. 核心采样函数：生成 WTA 候选轨迹作为监督目标，即生成运动预测的分类监督目标、回归监督目标
    def sample(
        self,
        reg_pred,          # 预测的多模态轨迹，形状 [bs, num_anchor, mode, ts, d]
        gt_reg_target,     # 真值轨迹列表，列表的每个元素形状 [num_gt, ts, d]
        gt_reg_mask,       # 真值有效掩码列表，列表的每个元素形状 [num_gt, ts]
        motion_loss_cache, # 损失缓存字典，内含检测匹配的indices，键为'indices'
    ):
        # 解包预测维度
        bs, num_anchor, mode, ts, d = reg_pred.shape

        # (1) 初始化，并获取检测分支的匈牙利匹配结果
        # (1.1) 初始化回归真值张量和回归权重张量为全0
        reg_target = reg_pred.new_zeros((bs, num_anchor, ts, d)) # 初始化回归真值张量，全0填充，形状 [bs, num_anchor, ts, d]
        reg_weight = reg_pred.new_zeros((bs, num_anchor, ts))    # 初始化回归权重张量，全0填充，形状 [bs, num_anchor, ts]

        # (1.2) 从缓存中取出检测分支的匈牙利匹配结果，避免重复匹配
        indices = motion_loss_cache['indices']
        
        # (1.3) 初始化正样本计数器，初始为0
        num_pos = reg_pred.new_tensor([0])

        # (2) 逐帧填充真值
        for i, (pred_idx, target_idx) in enumerate(indices):
            # (a) 该帧无真值则跳过
            if len(gt_reg_target[i]) == 0:
                continue

            # (b) 按匹配索引，将对应真值轨迹填入回归目标对应位置
            reg_target[i, pred_idx] = gt_reg_target[i][target_idx]

            # (c) 同步填入有效掩码
            reg_weight[i, pred_idx] = gt_reg_mask[i][target_idx]

            # (d) 正样本数量累加
            num_pos += len(pred_idx)
        
        # (3) 选出每个实例的最优模态索引，作为分类目标；选出每个实例对应的最优轨迹预测，作为回归目标
        # (a) 选出每个实例的最优模态索引，作为分类目标
        cls_target = get_cls_target(reg_pred, reg_target, reg_weight)
        # (b) 分类权重：只要有任意一个有效时间步则为有效，形状 [bs, num_anchor]
        cls_weight = reg_weight.any(dim=-1)
        # (c) 选出每个实例对应的最优轨迹预测，作为回归目标，用于回归损失计算
        best_reg = get_best_reg(reg_pred, reg_target, reg_weight)

        # (4) 返回6元组结果
        '''
            cls_target (Tensor): 分类目标（最优模态索引），形状 [bs, num_anchor]
            cls_weight (Tensor): 分类损失权重掩码，形状 [bs, num_anchor]
            best_reg   (Tensor): 最优模态的轨迹预测，形状 [bs, num_anchor, ts, d]
            reg_target (Tensor): 轨迹回归真值，形状 [bs, num_anchor, ts, d]
            reg_weight (Tensor): 回归损失权重掩码，形状 [bs, num_anchor, ts]
            num_pos    (Tensor): 正样本总数量，标量
        '''
        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos


# 二、规划目标分配器：生成自车规划的分类、回归监督目标
@BBOX_SAMPLERS.register_module() # 注册到边界框采样器
class PlanningTarget():
    """
        规划目标分配器
        核心逻辑：先根据驾驶命令筛选对应模态组，再从该组中选出与真值最接近的最优轨迹
        设计：自车规划分左转、直行、右转三组命令，每组对应多个轨迹模态
    """

    # 1. 初始化规划目标分配器
    def __init__(
        self,
        ego_fut_ts,   # 自车规划的未来时间步数
        ego_fut_mode, # 单驾驶命令下的轨迹模态数
    ):
        super(PlanningTarget, self).__init__()
        self.ego_fut_ts = ego_fut_ts     # 保存自车规划时间步数
        self.ego_fut_mode = ego_fut_mode # 保存单命令模态数

    # 2. 核心采样函数：生成自车规划的分类、回归监督目标
    def sample(
        self,
        cls_pred,      # 规划分类置信度，形状 [bs, 1, 3*ego_fut_mode]
        reg_pred,      # 规划轨迹预测，形状 [bs, 1, 3*ego_fut_mode, ts, 2]
        gt_reg_target, # 自车轨迹真值，形状 [bs, ts, 2]
        gt_reg_mask,   # 自车轨迹有效掩码，形状 [bs, ts]
        data,          # 数据字典，含'gt_ego_fut_cmd'驾驶命令标签
    ):
        # (1) 对真值和掩码增加模态维度，适配多模态距离计算
        gt_reg_target = gt_reg_target.unsqueeze(1) # 真值增加模态维度，适配多模态距离计算
        gt_reg_mask = gt_reg_mask.unsqueeze(1)     # 掩码同步增加模态维度
        bs = reg_pred.shape[0] # 获取批次大小

        # (2) 准备批次索引和命令索引，用于按命令筛选模态
        # (2.1) 生成批次索引，用于按命令筛选模态
        bs_indices = torch.arange(bs, device=reg_pred.device)
        
        # (2.2) 取出真值驾驶命令，取概率最大的命令索引（0左转/1直行/2右转）
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        # (3) 选出高层驾驶指令下的 WTA 模态轨迹
        # (3.1) 重塑分类预测的形状和轨迹预测的形状
        cls_pred = cls_pred.reshape(bs, 3, 1, self.ego_fut_mode)                     # 分类预测重塑为 [bs, 3, 1, ego_fut_mode]，3对应三种驾驶命令
        reg_pred = reg_pred.reshape(bs, 3, 1, self.ego_fut_mode, self.ego_fut_ts, 2) # 轨迹预测重塑为 [bs, 3, 1, ego_fut_mode, ts, 2]

        # (3.2) 分层目标选择第一级：按高层驾驶指令选择分组【按真值驾驶命令，筛选出对应命令组的分类置信度和轨迹预测】
        cls_pred = cls_pred[bs_indices, cmd] # 按真值驾驶命令，筛选出对应命令组的分类置信度
        reg_pred = reg_pred[bs_indices, cmd] # 按真值驾驶命令，筛选出对应命令组的轨迹预测

        # (3.3) 分层目标选择第二级：在当前指令组内选择 WTA 模态【在该命令组中选出最优模态的分类目标、轨迹预测，及分类权重】
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask) # 在该命令组内选出最优模态索引，作为分类目标
        cls_weight = gt_reg_mask.any(dim=-1)                              # 分类权重：只要有任意有效时间步则为有效
        best_reg = get_best_reg(reg_pred, gt_reg_target, gt_reg_mask)     # 选出最优模态的轨迹预测

        # (4) 返回6元组结果
        """
            cls_pred (Tensor): 筛选后的分类置信度，形状 [bs, 1, ego_fut_mode]
            cls_target (Tensor): 分类目标（最优模态索引），形状 [bs, 1]
            cls_weight (Tensor): 分类损失权重，形状 [bs, 1]
            best_reg (Tensor): 最优模态轨迹预测，形状 [bs, 1, ts, 2]
            gt_reg_target (Tensor): 轨迹真值，形状 [bs, 1, ts, 2]
            gt_reg_mask (Tensor): 轨迹有效掩码，形状 [bs, 1, ts]
        """
        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask
