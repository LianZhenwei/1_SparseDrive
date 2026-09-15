from typing import Optional
import numpy as np
import torch
from mmdet.core.bbox.builder import BBOX_CODERS # 边界框编码器/解码器注册器

from projects.mmdet3d_plugin.core.box3d import *                    # 导入3D框各维度索引常量（X/Y/Z/W/L/H/YAW/SIN_YAW/COS_YAW等）
from projects.mmdet3d_plugin.models.detection3d.decoder import *    # 导入检测模块的基础解码器，运动解码器继承复用检测解码逻辑
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners # 3D框转8个角点坐标的工具函数


# 一、3D检测+运动预测联合解码器
@BBOX_CODERS.register_module() # 注册到边界框解码器注册器
class SparseBox3DMotionDecoder(SparseBox3DDecoder):
    """
    3D检测+运动预测联合解码器
    继承检测解码器的基础逻辑（TopK筛选、质量加权、阈值过滤、框解码），额外补充运动轨迹解码与时序队列输出
    输出结构化的轨迹结果，用于评估与可视化
    """

    # 1. 初始化
    def __init__(self):
        super(SparseBox3DMotionDecoder, self).__init__() # 初始化：直接调用父类检测解码器的初始化，复用所有基础配置

    # 2. 核心解码函数：将原始张量预测转为结构化的检测+轨迹结果
    def decode(
        self,
        cls_scores,         # 【来自 Detection 任务】各层分类预测，每层形状 [bs, num_pred, num_cls]
        box_preds,          # 【来自 Detection 任务】各层3D框预测，每层形状 [bs, num_pred, box_dims]
        instance_id=None,   # 【来自 Tracking  任务】实例跟踪ID，形状 [bs, num_pred]
        quality=None,       # 【来自 Detection 任务】各层质量预测（centerness等）
        motion_output=None, # 【来自 Motion    任务】运动预测输出字典，含轨迹预测、分类、时序队列等
        output_idx=-1,      # 使用第几层的输出做解码，默认-1取最后一层
    ):
        # ===================== 一、跟随 Detection 推理阶段后处理的筛选结果 =====================
        # 1. 获取 topk 预测的分类分数 cls_scores 及其对应的分类 ID cls_ids
        # (1) 标记是否为单类别压缩模式：存在实例ID时说明已做过类别判定，分类维度已压缩
        squeeze_cls = instance_id is not None

        # (2) 取最后一层的分类分数，经sigmoid转为0~1概率
        cls_scores = cls_scores[output_idx].sigmoid()

        # (3) 压缩模式：取每个实例最大概率的类别，同时得到类别ID
        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            # 分数扩展一维，保持和非压缩模式的维度对齐
            cls_scores = cls_scores.unsqueeze(dim=-1)

        # (4) 取最后一层的3D框预测
        box_preds = box_preds[output_idx]

        # (5) 解包维度：批次、预测数、类别数
        bs, num_pred, num_cls = cls_scores.shape

        # (6) 分类分数展平为二维，取TopK个最高置信度的预测
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(self.num_output, dim=1, sorted=self.sorted)

        # (7) 非压缩模式：从展平索引反推对应的类别ID（索引对类别数取余）
        if not squeeze_cls:
            cls_ids = indices % num_cls


        # 2. 若设置了置信度阈值，生成掩码标记高于阈值的有效预测
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold


        # 3. 做质量预测加权打分
        # (1) 若该层没有质量预测，质量分支置空
        if quality[output_idx] is None:
            quality = None

        # (2) 质量分支有效时，做质量加权打分，提升排序准确性
        if quality is not None:
            # (a) 对 topk 预测的分类分数 cls_scores 做中心度质量分数 centerness 加权打分，得到新的 cls_scores
            # 取出中心度质量分数
            centerness = quality[output_idx][..., CNS]
            # 按TopK索引筛选对应实例的中心度
            centerness = torch.gather(centerness, 1, indices // num_cls)
            # 保存原始分类分数，用于输出
            cls_scores_origin = cls_scores.clone()
            # 分类分数乘以中心度的sigmoid值，做质量加权
            cls_scores *= centerness.sigmoid()
            # 按加权后的分数重新降序排序
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)

            # (b) 同步排序类别ID
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            # (c) 同步排序阈值掩码
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            # (d) 同步排序原始索引
            indices = torch.gather(indices, 1, idx)


        # ===================== 二、Motion 解码：核心是《轨迹从增量累加为绝对坐标，再叠加当前3D框的xy位置，得到全局坐标系下的完整轨迹》=====================
        # 4. 逐帧处理解码结果，返回所有帧的解码结果列表
        # 初始化输出结果列表
        output = []
        # 取出他车未来轨迹的时序锚点队列列表，沿时间维度堆叠，形状 [bs, num_pred, queue_len, box_dims]
        anchor_queue = motion_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        # 取出每个实例的存活周期
        period = motion_output["period"]

        # 逐帧处理解码结果
        for i in range(bs):
            # (1) 取出该帧的类别 ID 及其对应的置信度分数、3D框
            # (a) 取出该帧的类别ID
            category_ids = cls_ids[i]
            # 压缩模式下，按TopK索引取出对应实例的类别
            if squeeze_cls:
                category_ids = category_ids[indices[i]]

            # (b) 取出该帧的置信度分数
            scores = cls_scores[i]
            # (c) 按TopK索引取出对应实例的3D框
            box = box_preds[i, indices[i] // num_cls]

            # (2) 置信度阈值过滤：剔除低置信度结果
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]

            # （3）质量分支有效时，保存原始分类分数
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            # (4) 解码3D框：将log尺寸、sin/cos角度转换为真实物理坐标与尺寸
            box = decode_box(box)

            # (5) 取出最后一层的轨迹预测增量和轨迹分类置信度
            trajs = motion_output["prediction"][-1]                  # 取出最后一层的轨迹预测增量
            traj_cls = motion_output["classification"][-1].sigmoid() # 取出轨迹的分类置信度，转sigmoid概率

            # (6) 按TopK索引取出对应实例的轨迹及其轨迹置信度
            traj = trajs[i, indices[i] // num_cls]        # 按TopK索引取出对应实例的轨迹
            traj_cls = traj_cls[i, indices[i] // num_cls] # 取出对应实例的轨迹置信度

            # (7) 置信度阈值过滤轨迹结果
            if self.score_threshold is not None:
                traj = traj[mask[i]]
                traj_cls = traj_cls[mask[i]]

            # (8) 【核心】轨迹从增量累加为绝对坐标，再叠加当前3D框的xy位置，得到全局坐标系下的完整轨迹
            traj = traj.cumsum(dim=-2) + box[:, None, None, :2]

            # (9) 保存轨迹与轨迹分数到结果字典
            output.append(
                {
                    "trajs_3d": traj.cpu(),       # 3D轨迹坐标，移到CPU节省显存
                    "trajs_score": traj_cls.cpu() # 轨迹置信度分数
                }
            )

            # (10) 保存时序锚点队列和存活周期到结果字典
            # 取出对应实例的时序锚点队列
            temp_anchor = anchor_queue[i, indices[i] // num_cls]
            # 取出对应实例的存活周期
            temp_period = period[i, indices[i] // num_cls]

            # 置信度阈值过滤时序结果
            if self.score_threshold is not None:
                temp_anchor = temp_anchor[mask[i]]
                temp_period = temp_period[mask[i]]

            # 解包时序锚点的维度：实例数、队列长度
            num_pred, queue_len = temp_anchor.shape[:2]
            # 展平实例和队列维度，批量解码锚点
            temp_anchor = temp_anchor.flatten(0, 1)
            # 解码3D锚点为真实物理值
            temp_anchor = decode_box(temp_anchor)
            # 恢复为 [实例数, 队列长度, 框维度] 的形状
            temp_anchor = temp_anchor.reshape([num_pred, queue_len, box.shape[-1]])

            # 时序锚点队列存入结果字典
            output[-1]['anchor_queue'] = temp_anchor.cpu()
            # 存活周期存入结果字典
            output[-1]['period'] = temp_period.cpu()
        
        # 返回所有帧的解码结果列表
        return output # 每帧一个结果字典，包含轨迹、分数、时序锚点队列、存活周期等


# 二、分层规划解码器，对应论文中的「分层规划选择策略」
@BBOX_CODERS.register_module() # 注册到边界框解码器注册器
class HierarchicalPlanningDecoder(object):
    """
    分层规划解码器，对应论文中的「分层规划选择策略」
    核心流程：1. 按驾驶命令筛选模态组 → 2. 碰撞感知重打分 → 3. 选取得分最高的最终轨迹
    是SparseDrive规划安全设计的核心组件，通过碰撞检测剔除危险轨迹
    """

    # 1. 初始化分层规划解码器
    def __init__(
        self,
        ego_fut_ts,        # 自车规划的未来时间步数量
        ego_fut_mode,      # 单驾驶命令下的轨迹模态数量
        use_rescore=False, # 是否启用碰撞感知重打分，True则开启碰撞检测
    ):
        super(HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts     # 保存规划时间步数
        self.ego_fut_mode = ego_fut_mode # 保存单命令模态数
        self.use_rescore = use_rescore   # 保存是否启用重打分
    
    # 2. 规划解码主入口：分层选择+碰撞重打分，输出最终规划结果
    def decode(
        self, 
        det_output,
        motion_output,
        planning_output, 
        data,
    ):
        """
            规划解码主入口：分层选择+碰撞重打分，输出最终规划结果
            Args:
                det_output      (dict): 检测输出字典
                motion_output   (dict): 运动预测输出字典
                planning_output (dict): 规划原始输出字典
                data            (dict): 输入数据字典，含驾驶命令等
            Returns:
                list[dict]: 每帧一个规划结果字典
        """
        # (1) 取出最后一层的规划分类置信度和规划轨迹增量
        classification = planning_output['classification'][-1] # 取出最后一层的规划分类置信度
        prediction = planning_output['prediction'][-1]         # 取出最后一层的规划轨迹增量
        bs = classification.shape[0] # 获取批次大小

        # (2) 将分类分数和轨迹增量的形状均重塑为3种驾驶命令格式，并对轨迹增量沿时间步累加为绝对坐标
        # (a) 分类分数重塑为 [bs, 3, ego_fut_mode]，3对应左转/直行/右转三种驾驶命令
        classification = classification.reshape(bs, 3, self.ego_fut_mode)
        # (b) 轨迹增量重塑为命令分组格式，再沿时间步累加为绝对坐标
        prediction = prediction.reshape(bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)

        # (3) 调用分层选择方法，得到重打分后的分数与最终规划轨迹
        classification, final_planning = self.select(det_output, motion_output, classification, prediction, data)

        # (4.1) 取出自车时序锚点队列，沿时间维度堆叠
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        # (4.2) 取自车存活周期
        period = planning_output["period"]

        # (5) 逐帧组装输出结果
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            output.append(
                {
                    "planning_score": cls.sigmoid().cpu(),                 # 所有模态的规划置信度
                    "planning": pred.cpu(),                                # 所有模态的规划轨迹
                    "final_planning": final_planning[i].cpu(),             # 最终选中的最优轨迹
                    "ego_period": period[i].cpu(),                         # 自车存活周期
                    "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(), # 历史锚点队列
                }
            )
        # 返回所有帧的规划结果
        return output

    # 3. 分层选择核心逻辑：命令级筛选 + 可选碰撞重打分 + 最优模态选取
    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        """
        分层选择核心逻辑：命令级筛选 + 可选碰撞重打分 + 最优模态选取
        Args:
            det_output    (dict): 检测输出字典
            motion_output (dict): 运动预测输出字典
            plan_cls    (Tensor): 规划分类分数，形状 [bs, 3, ego_fut_mode]
            plan_reg    (Tensor): 规划轨迹，形状 [bs, 3, ego_fut_mode, ts, 2]
            data          (dict): 数据字典，含驾驶命令真值
        Returns:
            plan_cls_full  (Tensor): 重打分后的全部分类分数，形状 [bs, 3, ego_fut_mode]
            final_planning (Tensor): 最终最优轨迹，形状 [bs, ts, 2]
        """
        # 获取目标检测与他车运动预测的结果：获取检测分类分数、检测3D框、每个检测实例的最大置信度，获取运动预测分类分数、运动预测轨迹增量
        det_classification = det_output["classification"][-1].sigmoid() # 提取检测分类分数，转sigmoid概率
        det_anchors = det_output["prediction"][-1]                      # 提取检测3D框
        det_confidence = det_classification.max(dim=-1).values          # 计算每个检测实例的最大置信度
        motion_cls = motion_output["classification"][-1].sigmoid() # 提取他车运动预测的分类分数
        motion_reg = motion_output["prediction"][-1]               # 提取他车运动预测的轨迹增量
        bs = motion_cls.shape[0] # 获取 batch_size
        
        # ========== (1) 第一步：按驾驶命令筛选对应模态组 ==========
        # (a.1) 生成批次索引，用于按命令索引取值
        bs_indices = torch.arange(bs, device=motion_cls.device)
        # (a.2) 取出真值驾驶命令，取概率最大的命令索引
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        # (b) 保存原始全部分类分数的副本，用于最终输出
        plan_cls_full = plan_cls.detach().clone()

        # (c) 按命令取出对应组的分类分数和轨迹
        plan_cls = plan_cls[bs_indices, cmd] # 按命令取出对应组的分类分数
        plan_reg = plan_reg[bs_indices, cmd] # 按命令取出对应组的轨迹

        # ========== (2) 第二步：碰撞感知重打分（可选） ==========
        if self.use_rescore:
            plan_cls = self.rescore(
                plan_cls,
                plan_reg, 
                motion_cls,
                motion_reg, 
                det_anchors,
                det_confidence,
            )

        # 更新完整分数中对应命令组的分值
        plan_cls_full[bs_indices, cmd] = plan_cls

        # ========== (3) 第三步：选取得分最高的模态作为最终规划 ==========
        mode_idx = plan_cls.argmax(dim=-1)
        # 按索引取出最优轨迹
        final_planning = plan_reg[bs_indices, mode_idx]

        # ========== (4) 最后：返回完整分数与最终轨迹 ==========
        return plan_cls_full, final_planning

    # 4. 碰撞感知重打分：检测自车规划轨迹与周围障碍物未来轨迹的碰撞，碰撞轨迹大幅扣分
    def rescore(
        self, 
        plan_cls,
        plan_reg, 
        motion_cls,
        motion_reg, 
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):
        """
        碰撞感知重打分：检测自车规划轨迹与周围障碍物未来轨迹的碰撞，碰撞轨迹大幅扣分（扣 999 分）
        Args:
            plan_cls         (Tensor): 规划分类分数，形状 [bs, ego_fut_mode]
            plan_reg         (Tensor): 规划轨迹，形状 [bs, ego_fut_mode, ts, 2]
            motion_cls       (Tensor): 运动预测分类分数，形状 [bs, num_anchor, fut_mode]
            motion_reg       (Tensor): 运动预测轨迹增量，形状 [bs, num_anchor, fut_mode, ts, 2]
            det_anchors      (Tensor): 检测3D框，形状 [bs, num_anchor, box_dims]
            det_confidence   (Tensor): 检测置信度，形状 [bs, num_anchor]
            score_thresh      (float): 检测置信度阈值，低于阈值的障碍物忽略
            static_dis_thresh (float): 静态物体距离阈值，小于该值认为静止，航向保持初始值
            dim_scale         (float): 障碍物尺寸缩放系数，增加安全裕度
            num_motion_mode     (int): 取TopK个运动模态参与碰撞检测
            offset            (float): 自车碰撞盒的前向偏移量，模拟车头碰撞点
        Returns:
            plan_cls (Tensor): 重打分后的规划分类分数，碰撞模态分数极低
        """
        
        # (1) 内部工具函数：在轨迹开头拼接一个(0,0)点，对应当前时刻的初始位置
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2)) # 生成零点张量，形状和轨迹除时间维外一致，时间维为1
            traj_cat = torch.cat([zeros, traj], dim=-2)      # 拼接到轨迹开头
            return traj_cat
        
        # (2) 内部工具函数：根据轨迹点序列计算每个时刻的航向角
        def get_yaw(traj, start_yaw=np.pi/2):
            """
            内部工具：根据轨迹点序列计算每个时刻的航向角
            原理：用相邻点的差分方向近似航向，静态物体保持初始航向
            Args:
                traj: 轨迹坐标，形状 [..., ts, 2]
                start_yaw: 初始航向角，默认沿y轴正方向（90度）
            Returns:
                yaw: 各时刻航向角，形状 [..., ts, 1]
            """
            # 初始化航向角张量为全 0
            yaw = traj.new_zeros(traj.shape[:-1])

            # (a) 计算轨迹在各时刻的航向角
            # (a.1) 中间时刻的航向：用前后点差分计算反正切
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            # (a.2) 最后一个时刻的航向：用最后两点差分计算反正切
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            # (a.3) 第一个时刻用初始航向
            yaw[..., 0] = start_yaw

            # (b) 静态物体保持初始航向
            # (b.1) 计算轨迹总位移
            start = traj[..., 0, :] # 轨迹第一点的坐标
            end = traj[..., -1, :]  # 轨迹最后一点的坐标
            dist = torch.linalg.norm(end - start, dim=-1) # 计算轨迹总位移
            # (b.2) 静态物体掩码
            mask = dist < static_dis_thresh # 静态物体修正：位移小于阈值 static_dis_thresh 的物体（视为静态物体），航向保持初始值，避免抖动
            # (b.3) 初始航向扩展维度
            start_yaw = yaw[..., 0].unsqueeze(-1)
            # (b.4) 若为静态物体，则用初始航向覆盖
            yaw = torch.where(mask.unsqueeze(-1), start_yaw, yaw)
            # 增加最后一维返回
            return yaw.unsqueeze(-1)
        
        # 获取 batch_size
        bs = plan_reg.shape[0]

        # ========== 1. 构造自车每个时刻的3D包围盒 ==========
        # 自车轨迹拼接初始零点
        plan_reg_cat = cat_with_zero(plan_reg)

        # (0) 初始化自车框张量为全 0：[bs, 模态数, 时间步, 7维框参数(x,y,z,w,l,h,yaw)]
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        # (1) 填充xy坐标
        ego_box[..., [X, Y]] = plan_reg_cat
        # (2) 填充长宽高，乘以缩放系数增加安全裕度
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        # (3) 计算各时刻航向角并填充
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        # ========== 2. 构造障碍物每个时刻的3D包围盒 ==========
        # 截取和规划相同长度的时间步，轨迹累加为绝对坐标
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        # 拼接初始零点，再叠加障碍物当前框的xy位置，得到全局轨迹
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]

        # 取出TopK模态对应的轨迹
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1) # 取TopK置信度的运动模态
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2) # 索引扩展维度，适配gather操作
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx) # 取出TopK模态对应的轨迹

        # (0) 初始化障碍物框张量为全 0
        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        # (1) 填充xy坐标
        motion_box[..., [X, Y]] = motion_reg
        # (2) 填充障碍物尺寸，从检测框的log值还原
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        # (3) 计算各时刻航向角并填充
        box_yaw = torch.atan2(det_anchors[..., SIN_YAW], det_anchors[..., COS_YAW]) # 计算障碍物当前航向角
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1)) # 计算障碍物轨迹各时刻的航向角，初始值为当前航向
        # (4) 低置信度障碍物的框设为极大值，相当于忽略不参与碰撞检测
        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        # ========== 3. 批量碰撞检测 ==========
        # (1) 去掉开头的零点时刻，只计算未来时刻的碰撞
        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        # (2) 解包维度
        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape

        # (3) 对自车框和障碍物框扩展维度
        # 自车框扩展维度，和所有障碍物、所有运动模态配对，展平后批量计算
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        # 障碍物框扩展维度，和所有自车模态配对，展平后批量计算
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        # (4) 自车碰撞盒向前偏移offset距离，模拟车头碰撞点（更保守的安全判定）
        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])

        # (5) 批量计算碰撞结果
        col = check_collision(ego_box, motion_box)

        # ========== 4. 碰撞结果重塑与分数惩罚 ==========
        # 碰撞结果重塑回原始维度结构
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)

        # (1) 判断每个自车模态是否和任意障碍物、任意运动模态、任意时刻发生碰撞
        col = col.flatten(2, -1).any(dim=-1)

        # (2) 特殊情况：如果所有模态都碰撞，则不扣分（避免无有效轨迹可选）
        all_col = col.all(dim=-1)
        col[all_col] = False 

        # (3) 碰撞的模态分数减去999，相当于直接排除该候选轨迹
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset

        # (4) 返回重打分后的分数
        return plan_cls


# 粗略碰撞检测【使用的是双向角点包含检查】：双向检查两个框的角点是否在对方内部，有任一重合则判定碰撞
def check_collision(
        boxes1, # 形状 [N, 7]，格式为[x, y, z, w, l, h, yaw]
        boxes2  # 形状 [N, 7]，格式为[x, y, z, w, l, h, yaw]
    ):
    # 检查boxes2的角点是否在boxes1内
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())

    # 检查boxes1的角点是否在boxes2内
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())

    # 双向取或，只要有一个方向判定相交就算碰撞
    collision = torch.logical_or(col_1, col_2)
    return collision # 返回布尔张量，形状 [N]，True 表示碰撞

# 判断boxes2的四个底角点是否在boxes1的包围盒内【原理：将两个框都变换到boxes1的局部坐标系，直接判断xy坐标是否在长宽范围内】【lzw 回顾: 李老师实验室的碰撞检测使用的是分离轴定理（SAT, Separating Axis Theorem）】
def corners_in_box(
        boxes1, # 参考框，形状 [N, 7]，格式为[x, y, z, w, l, h, yaw]
        boxes2  # 待检测框，形状 [N, 7]，格式为[x, y, z, w, l, h, yaw]
    ):
    # 空输入直接返回False
    if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    # (1) 取出参考框的航向角和中心位置
    boxes1_yaw = boxes1[:, 6].clone()  # 取出参考框的航向角
    boxes1_loc = boxes1[:, :3].clone() # 取出参考框的中心位置

    # (2) 计算反向旋转的余弦和正弦
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)

    # (3) 构建旋转矩阵的转置
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )

    # (4) 处理参考框
    # 将参考框平移到原点
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    # 将参考框旋转到自身局部坐标系（航向角归零）
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    # 参考框航向角归零
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    # (5) 处理待检测框
    # 将待检测框做相同的平移变换
    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    # 将待检测框做相同的旋转变换
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    # 待检测框航向角同步变换
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    # (6) 生成待检测框的4个底角点坐标，取底面四个角
    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device) # numpy结果转回tensor

    # (7) 取出参考框的长和宽
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    # (8) 判断所有角点的x、y坐标是否都在参考框的长宽范围内
    collision = torch.logical_and(
        # x坐标在 [-长/2, 长/2] 范围内
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        # y坐标在 [-宽/2, 宽/2] 范围内
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )

    # (9) 只要有一个角点在框内，就判定为相交
    collision = collision.any(dim=-1)
    return collision # 布尔张量，形状 [N]，True表示boxes2有角点在boxes1内

