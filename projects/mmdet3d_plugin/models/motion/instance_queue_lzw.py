
import copy # 深拷贝工具，用于配置、对象的完整复制
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mmcv.utils import build_from_cfg                       # MMCV配置构建工具，用于从注册器实例化模块
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS          # 插件层注册器
from projects.mmdet3d_plugin.ops import feature_maps_format # 特征图格式转换工具，用于适配不同算子的特征排布
from projects.mmdet3d_plugin.core.box3d import *            # 3D框各维度的索引常量（X/Y/Z/W/L/H/SIN_YAW/COS_YAW/VX/VY等）



@PLUGIN_LAYERS.register_module() # 将该类注册到插件层注册器，可通过配置文件按类名实例化
class InstanceQueue(nn.Module):
    """
        实例时序队列：运动规划模块的长时序状态管理器
        核心作用：缓存多帧历史的周围目标与自车状态，为运动预测和规划提供时序上下文
        与检测模块InstanceBank的区别：InstanceBank只缓存上一帧，本队列可缓存多帧，专门服务于时序交互建模
    """
    # 1. 初始化
    def __init__(
        self,
        embed_dims,             # 特征嵌入维度，与主干网络特征维度一致
        queue_length=0,         # 时序队列最大长度，即缓存多少帧历史+1帧当前
        tracking_threshold=0,   # 跟踪置信度阈值，低于该阈值的历史实例不参与时序匹配
        feature_map_scale=None, # 特征图尺寸，用于计算自车特征编码器的池化核大小
    ):
        super(InstanceQueue, self).__init__() # 调用父类初始化

        # (1) 保存参数
        self.embed_dims = embed_dims                 # 保存特征嵌入维度
        self.queue_length = queue_length             # 保存队列最大长度，即最多保存 “queue_length-1 帧历史 + 1 帧当前”【在 sparsedrive_small_stage1.py 和 stagedrive_small_stage2.py 中均配置为 queue_length=4 帧，即 3 帧历史 + 当前帧】
        self.tracking_threshold = tracking_threshold # 保存跟踪置信度阈值

        # (2) 计算自车特征编码器的最终平均池化核大小：特征图尺寸的一半
        kernel_size = tuple([int(x / 2) for x in feature_map_scale])

        # (3) 自车特征编码器：从FPN特征图中编码出自车的全局特征向量
        # 结构：两次卷积+BN+ReLU+平均池化，最终输出维度为embed_dims的向量
        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, stride=1, padding=1, bias=False), # 第一个3x3卷积，保持尺寸不变
            nn.BatchNorm2d(embed_dims),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False), # 第二个3x3卷积，步长为2，尺寸减半
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size), # 平均池化到1x1，得到全局特征向量 [B, 1, 256]
        )

        # (4) 自车的固定初始3D锚点，编码格式与检测框完全一致
        # 维度含义：x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz
        # 数值为典型乘用车尺寸：宽1.73m、长4.08m、高1.56m，初始航向沿y轴正方向，速度初始全0
        self.ego_anchor = nn.Parameter(
            torch.tensor([[0, 0.5, -1.84 + 1.56/2, np.log(4.08), np.log(1.73), np.log(1.56), 1, 0, 0, 0, 0],], dtype=torch.float32),
            requires_grad=False, # 锚点固定，不参与梯度更新
        )

        # (5) 调用重置方法，初始化所有缓存状态
        self.reset()

    # 2. 重置所有时序缓存状态【调用时机：序列第一帧、历史帧失效、batch大小变化时】
    def reset(self):
        # (1) 重置上一帧数据为 None：
        self.metas = None # 上一帧的元数据（时间戳、标定矩阵等）

        # (2) 重置周围目标历史数据为空
        self.prev_instance_id = None     # 上一帧的实例ID，用于跨帧匹配同一目标
        self.prev_confidence = None      # 上一帧的实例置信度，用于阈值过滤低质量目标
        self.period = None               # 每个实例的存活周期（已连续存在多少帧），形状 [B, 900]
        self.instance_feature_queue = [] # 周围agent的特征时序队列，是一个长度的 queue_len 的列表，队列列表的每个元素对应一帧，最后一个元素是当前帧，元素的形状为 [B, 900, embed_dims=256]
        self.anchor_queue = []           # 周围agent的锚点时序队列，是一个长度的 queue_len 的列表，队列列表的每个元素对应一帧，最后一个元素是当前帧，元素的形状为 [B, 900, box_dims=11]

        # (3) 重置自车历史数据为 None 或空
        self.prev_ego_status = None # 上一帧的自车状态（速度、航向等）
        self.ego_period = None      # 自车的存活周期
        self.ego_feature_queue = [] # 自车特征的时序队列，是一个长度的 queue_len 的列表，队列列表的每个元素对应一帧，最后一个元素是当前帧，元素的形状为 [B, 1, embed_dims=256]
        self.ego_anchor_queue = []  # 自车锚点的时序队列，是一个长度的 queue_len 的列表，队列列表的每个元素对应一帧，最后一个元素是当前帧，元素的形状为 [B, 1, box_dims=11]

    # 3. 主入口函数：获取当前帧自车状态与多帧历史时序状态
    def get(
        self,
        det_output,     # 检测模块输出字典，含实例特征、预测框、实例ID等
        feature_maps,   # 多尺度特征图，用于提取自车特征
        metas,          # 当前帧元数据，含标定矩阵、时间戳等
        batch_size,     # 当前批次大小
        mask,           # 时序有效掩码，标记哪些样本的历史帧可用
        anchor_handler, # 锚点处理器，提供坐标投影能力
    ):
        # ========== 1. 【核心】历史缓存有效时，做坐标对齐【即把历史目标锚框队列 self.anchor_queue 和历史自车锚框队列 self.ego_anchor_queue 里的所有 anchor 都从历史坐标系投影到投影到当前帧坐标系】 ==========
        # 条件：周期不为空（有历史）且batch大小一致
        if (self.period is not None and batch_size == self.period.shape[0]):
            # 存在锚点处理器时，将历史锚点从历史坐标系投影到当前坐标系
            if anchor_handler is not None:
                # (1) 计算历史帧→当前帧的全局变换矩阵
                # 公式：T_temp2cur = 当前帧全局逆矩阵 @ 历史帧全局矩阵
                # 作用：把历史坐标系下的点转换到当前坐标系下
                T_temp2cur = feature_maps[0].new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )

                # (2) 遍历每一帧周围目标历史锚点 self.anchor_queue ，逐一投影到当前坐标系
                for i in range(len(self.anchor_queue)):
                    # 取出第i帧的历史锚点
                    temp_anchor = self.anchor_queue[i]
                    # 调用锚点投影方法，完成坐标变换
                    temp_anchor = anchor_handler.anchor_projection(
                        temp_anchor,
                        [T_temp2cur],
                    )[0]
                    # 更新队列中的锚点为投影后的当前坐标系值
                    self.anchor_queue[i] = temp_anchor

                # (3) 同理，对自车历史锚点 self.ego_anchor_queue 也做坐标投影到当前坐标系
                for i in range(len(self.ego_anchor_queue)):
                    temp_anchor = self.ego_anchor_queue[i]
                    temp_anchor = anchor_handler.anchor_projection(
                        temp_anchor,
                        [T_temp2cur],
                    )[0]
                    self.ego_anchor_queue[i] = temp_anchor
                '''
                    因此队列中的 Anchor 坐标系会持续变化：
                        第t帧保存时：
                            表达在第t帧坐标系
                        进入第t+1帧：
                            全部转到第t+1帧坐标系
                        进入第t+2帧：
                            再从第t+1帧坐标系转到第t+2帧坐标系
                '''

        # ========== 2. 历史缓存失效时，重置所有状态为 None 或空 ==========
        else:
            self.reset()

        # ========== 3. 【核心】通过 Instance ID 时序对齐周围目标队列 self.instance_feature_queue 和 self.anchor_queue ==========
        # 根据实例ID匹配，将历史特征重排到与当前实例一一对应的位置
        self.prepare_motion(det_output, mask)

        # ========== 4. 【核心】编码前视相机的最后一级特征图得到当前帧自车特征 ego_feature、继承自固定模版并设置其VY速度为上一帧预测VY速度得到当前帧自车锚点 ego_anchor，并返回二者，同时将二者尾插进自车历史队列 self.ego_feature_queue 和 self.ego_anchor_queue ==========
        # 从特征图编码自车特征，并初始化自车锚点，同时维护自车时序队列
        ego_feature, ego_anchor = self.prepare_planning(feature_maps, mask, batch_size)
        '''
            注意 ego_feature 每帧都重新由图像生成，不是从 self.prev_ego_status 创建的，
            而 ego_anchor 每帧都会先从同一个固定几何模板 self.ego_anchor 重新创建、然后其VY速度继承自 self.prev_ego_status 的VY。
        '''

        # ========== 5. 将各帧队列堆叠成时序 Tensor ==========
        # (1) 将队列列表沿时间维度堆叠
        temp_instance_feature = torch.stack(self.instance_feature_queue, dim=2) # 形状为 [B, 900, queue_len=4, embed_dims]
        temp_anchor = torch.stack(self.anchor_queue, dim=2)                     # 形状为 [B, 900, queue_len=4, 11]

        # (2) 同理堆叠自车的时序特征与锚点
        temp_ego_feature = torch.stack(self.ego_feature_queue, dim=2) # 形状为 [B, 1(即1个ego), queue_len=4, embed_dims] 
        temp_ego_anchor = torch.stack(self.ego_anchor_queue, dim=2)   # 形状为 [B, 1(即1个ego), queue_len=4, 11]

        # ========== 6. 将 Agent 时序队列后拼接 Ego 时序队列【因此总agent数 = agent数 + 1个自车】 ==========
        # (1) 拼接周围目标与自车的周期，形状 [B, 900+1]
        period = torch.cat([self.period, self.ego_period], dim=1)

        # (2) 拼接周围目标与自车的时序特征，自车作为第N+1个智能体
        temp_instance_feature = torch.cat([temp_instance_feature, temp_ego_feature], dim=1) # 形状变为 [B, 900+1, queue_len=4, embed_dims]
        temp_anchor = torch.cat([temp_anchor, temp_ego_anchor], dim=1)                      # 形状变为 [B, 900+1, queue_len=4, 11]

        # (3) 获取总智能体数量 = 周围目标数 + 1个自车
        num_agent = temp_anchor.shape[1]

        # ========== 7. 根据 period 生成时序注意力 Temporal Attention 的屏蔽掩码 ==========
        # 生成倒序索引：[queue_len, queue_len-1, ..., 1] = [4, 3, 2, 1]，形状 [4]
        temp_mask = torch.arange(len(self.anchor_queue), 0, -1, device=temp_anchor.device)
        # 扩展到 [B, num_agent=900+1, queue_len=4]
        temp_mask = temp_mask[None, None].repeat((batch_size, num_agent, 1))
        # 与每个实例的存活周期比较：大于周期的位置为True（无效，需屏蔽）
        # 作用：新出现的实例没有更早的历史，对应历史位置要mask掉，不能参与注意力
        temp_mask = torch.gt(temp_mask, period[..., None]) # [B, num_agent=900+1, queue_len=4]，True表示该历史位置无效，需屏蔽
        '''
            temp_mask 的作用是新目标的历史位置会被 Mask：
                例如队列长度是4，但某个目标刚出现1帧：
                    有效：当前帧
                    无效：t-1、t-2、t-3
                这些不存在的历史位置会通过：
                    key_padding_mask=temp_mask
                在 Temporal Attention 中被屏蔽。

            源码解读：
                temp_mask = arange(queue_length, 0, -1)，如 queue_length = 4，则：
                        temp_mask基础年龄序列 = [4,3,2,1]
                再比较：
                    temp_mask = temp_mask > period[...,None]
                Attention 中：
                    True：该历史位置无效，必须屏蔽
                    False：该历史位置有效，可以参与Attention
                例如某Agent period = 2，则：
                    [4,3,2,1] > 2
                    =
                    [True, True, False, False]
                即只允许最近2个队列位置参与。
        '''

        # ========== 8. 返回：由编码前视相机的最后一级特征图得到的当前帧自车特征、继承自固定模版并设置其VY速度为上一帧预测VY速度得到当前帧自车锚点 ego_anchor、时序agent特征+时序ego特征【agent已ID时序对齐】、已转换到当前坐标系的“时序agent anchor”+“时序ego anchor”【agent已ID时序对齐】、时序有效掩码 ==========
        return ego_feature, ego_anchor, temp_instance_feature, temp_anchor, temp_mask

    # 4.1 根据 Tracking ID 对齐并更新 Agent 历史队列【核心逻辑：通过实例ID匹配矩阵，将历史队列中每个目标的特征，放到当前帧同ID目标的对应位置】
    def prepare_motion(
        self,
        det_output, # 检测输出字典【注意：因此 Agent 队列存的是 Detection Head 输出的 Feature 和 Anchor，不是 Motion Head 精炼后的 Feature 和 Anchor】
        mask,       # 时序有效掩码
    ):
        # 读取当前帧 Detection 任务预测的 Detection Feature 和 Detection Anchor
        instance_feature = det_output["instance_feature"] # 提取当前帧检测的实例特征，[B, N, C]
        det_anchors = det_output["prediction"][-1]        # 提取当前帧最后一层的检测框预测，[B, N, 11]

        # ========== 1. 第一帧无历史时，初始化周期 self.period 为全 0 ==========
        if self.period == None:
            # 初始所有实例存活周期为0，形状 [B, 900] 即 [B, N]、长整型
            self.period = instance_feature.new_zeros(instance_feature.shape[:2]).long()

        # ========== 2. 【核心】有历史时，按ID做时序重排、且 self.period 加1【依据当前 instance_id 与上一帧 prev_instance_id 做匹配，从而将历史 Feature、Anchor 和 period 重新排列到当前 Agent 的顺序】 ==========
        else:
            # (1) 取出当前帧的实例ID tensor和上一帧的实例ID tensor
            instance_id = det_output['instance_id']  # 取出当前帧的实例ID，形状 [B, N_current]
            prev_instance_id = self.prev_instance_id # 取出上一帧的实例ID，形状 [B, N_previous]

            # (2) 构建ID匹配矩阵 match [B, N_current, N_previous]【因为 instance_id [B, N_current]、prev_instance_id [B, N_previous]，则二者的匹配矩阵 match [B, N_current, N_previous]】
            match = instance_id[..., None] == prev_instance_id[:, None]
            '''
                match[i,j] = True  表示当前第i个实例与历史第j个实例是同一个目标，即当前第i个实例ID == 上一帧第j个实例ID
                match[i,j] = False 表示当前第i个实例与历史第j个实例不是同一个目标，即当前第i个实例ID != 上一帧第j个实例ID
            '''

            # (3) 若设置了跟踪阈值，过滤掉历史置信度低的实例【所以历史低置信度 Instance 即使 ID 相等，也不会被当前 Agent 继承】
            if self.tracking_threshold > 0:
                # 生成置信度掩码：高于阈值的历史实例为True
                temp_mask = self.prev_confidence > self.tracking_threshold # self.tracking_threshold=0.2，因此历史置信度低于0.2时，匹配会失效
                # 与匹配矩阵相乘，低置信度的match矩阵元素值为0意味着匹配失效
                match = match * temp_mask.unsqueeze(1)

            # (4) 2个历史队列均重新按当前 ID 顺序排列【遍历每一帧历史，将该帧的特征按ID重排到当前对应位置】
            for i in range(len(self.instance_feature_queue)):
                # (a) 更新 self.instance_feature_queue：根据 instance_id，把历史 Agent Feature 从“上一帧 Instance 顺序”重新排列为“当前帧 Instance 顺序”
                # (a.1) 取出第i帧的历史特征 temp_feature [B, N_previous, embed_dims]
                temp_feature = self.instance_feature_queue[i]
                # (a.2) 矩阵乘法式重排：match @ 历史特征 = 重排后的历史特征【每个当前实例位置，只保留同ID的历史特征，其余为0，即未匹配到历史 ID 的当前 Instance，对应历史 Feature 和 Anchor 会变成全零】
                temp_feature = (match[..., None] * temp_feature[:, None]).sum(dim=2) # 矩阵乘法使重排后 temp_feature 变为 [B, N_current, embed_dims]
                # (a.3) 更新队列中的该帧特征为重排后的结果
                self.instance_feature_queue[i] = temp_feature

                # (b) 更新 self.anchor_queue：同理，对历史锚点做相同的ID重排
                temp_anchor = self.anchor_queue[i]
                temp_anchor = (match[..., None] * temp_anchor[:, None]).sum(dim=2)
                self.anchor_queue[i] = temp_anchor
                '''
                    因此对当前第 i 个 Agent 来说，其历史队列存储的是：
                         当前第 i 个 Agent 在 t-3 时的特征和 anchor
                         当前第 i 个 Agent 在 t-2 时的特征和 anchor
                         当前第 i 个 Agent 在 t-1 时的特征和 anchor
                         当前第 i 个 Agent 在 t   时的特征和 anchor
                    而不是其他 Agent 的历史特征。
                '''

            # (5) 更新每个实例的存活周期：匹配上的继承历史周期，未匹配的为0
            self.period = (match * self.period[:, None]).sum(dim=2)
            '''
                原来，period [B, N_previous]
                更新后，period [B, N_current]
                例如：
                    上一帧 ID 12的period = 3
                    当前帧某个Instance仍为ID 12：当前period继承3
                    当前帧新生Instance：无匹配历史，当前period变成0
                    随后统一 self.period += 1
                    所以：
                        连续存在的ID：3 → 4 → 5 → ...
                        新生ID：0 → 1
                因此 period 表示：当前 Agent Instance 可以使用多少个连续历史队列位置，最大被裁剪为 queue_length。
            '''

        # ========== 3. 当前帧 Detection Feature 和 Detection Anchor 加入队列 ==========
        # (1) 当前帧特征入队【detach截断梯度，缓存不参与反向传播，避免循环计算图】
        self.instance_feature_queue.append(instance_feature.detach())
        # (2) 当前帧锚点入队【detach截断梯度，缓存不参与反向传播，避免循环计算图】
        self.anchor_queue.append(det_anchors.detach())
        # (3) 所有实例的存活周期+1
        self.period += 1 # 第一帧时 self.period 全 0、形状 [B, 900]；第一帧结束后，此时 period 全部变成1，表示每个当前 Agent Instance 具有一帧历史记录；后续同理

        # ========== 4. 队列长度控制【删除超过 queue_length 的最旧帧】 ==========
        # 超过最大队列长度时，弹出最早的一帧
        if len(self.instance_feature_queue) > self.queue_length:
            self.instance_feature_queue.pop(0)
            self.anchor_queue.pop(0)

        # 周期值裁剪到0~队列长度之间，避免异常值【更新每个 Agent 的有效历史长度 period】
        self.period = torch.clip(self.period, 0, self.queue_length)

    # 4.2 构建当前帧 ego_feature 和 当前帧ego_anchor 并返回，同时更新 Ego 历史队列【核心逻辑：编码前视相机的最后一级特征图得到当前帧自车特征 ego_feature、继承自固定模版并设置其VY速度为上一帧预测VY速度得到当前帧自车锚点 ego_anchor，最后将这 2 者尾插入队】
    def prepare_planning(
        self,
        feature_maps, # 多尺度特征图
        mask,         # 时序有效掩码
        batch_size,   # 批次大小
    ):
        # ========== 1. 从 FPN 特征图里编码自车特征 ego_feature ==========
        # 特征图格式反转：从算子专用格式转回常规的 [B, C, H, W] 格式
        feature_maps_inv = feature_maps_format(feature_maps, inverse=True)

        # (1) 提取前视相机 CAM_FRONT 的最后一层 FPN 特征图 feature_map（也就是分辨率最低、语义最强的那层特征），形状 [B, embed_dims, H, W]
        feature_map = feature_maps_inv[0][-1][:, 0]

        # (2) 编码 feature_map 得到自车全局特征 ego_feature，[B, embed_dims, 1, 1]
        ego_feature = self.ego_feature_encoder(feature_map) # 自车特征编码器编码
        # 去掉 ego_feature 的空间维度、增加实例维度，形状变为 [B, 1, embed_dims]
        ego_feature = ego_feature.unsqueeze(1).squeeze(-1).squeeze(-1)

        # ========== 2. 设置自车锚点 ego_anchor 的VY速度为上一帧预测VY速度【在 __init__() 中自车 self.ego_anchor 的 x,y,z,l,w,h,vx,vz 均固定，但自车VY速度设置为上一帧预测VY速度】 ==========
        # (1) 将固定模版的单例自车锚点复制到batch维度，形状 [B, 1, box_dims=11]
        ego_anchor = torch.tile(self.ego_anchor[None], (batch_size, 1, 1))

        # (2) 有历史自车状态时，用上一帧的速度初始化当前锚点的纵向速度VY
        if self.prev_ego_status is not None:
            # 按时序掩码选择：有效样本用历史速度，无效样本置0
            prev_ego_status = torch.where(
                mask[:, None, None],
                self.prev_ego_status,
                self.prev_ego_status.new_tensor(0),
            )
            # 将历史VY速度赋值给当前锚点的VY维度
            ego_anchor[..., VY] = prev_ego_status[..., 6]

        # ========== 3. 维护自车存活周期【根据序列连续性 mask 更新 ego_period】 ==========
        if self.ego_period == None:
            # 第一帧初始化为0
            self.ego_period = ego_feature.new_zeros((batch_size, 1)).long()
        else:
            # 有时序掩码：有效样本继承周期，无效样本重置为0
            self.ego_period = torch.where(
                mask[:, None],
                self.ego_period,
                self.ego_period.new_tensor(0),
            )

        # ========== 4. 当前帧自车状态入队 ==========
        # 当前帧自车特征入队，detach截断梯度
        self.ego_feature_queue.append(ego_feature.detach())
        # 当前帧自车锚点入队，detach截断梯度【此时队列末尾的 ego_feature 还是由相机Feature直接编码得到的初始Ego Feature】
        self.ego_anchor_queue.append(ego_anchor.detach())
        # 自车周期+1
        self.ego_period += 1

        # ========== 5. 队列长度控制【删除超过 queue_length 的最旧帧】 ==========
        # 超过最大长度时弹出最早的一帧
        if len(self.ego_feature_queue) > self.queue_length:
            self.ego_feature_queue.pop(0)
            self.ego_anchor_queue.pop(0)

        # 周期值裁剪
        self.ego_period = torch.clip(self.ego_period, 0, self.queue_length)

        # 返回当前帧自车特征、当前帧自车锚点
        return ego_feature, ego_anchor

    # 4.3 缓存当前帧周围目标的关键状态，供下一帧时序匹配使用【调用时机：motion head前向结束后】
    def cache_motion(self, instance_feature, det_output, metas):
        # 取最后一层分类结果，转sigmoid概率
        det_classification = det_output["classification"][-1].sigmoid()
        # 计算每个实例的最大置信度
        det_confidence = det_classification.max(dim=-1).values
        # 取出实例ID
        instance_id = det_output['instance_id']

        # 保存当前帧元数据，供下一帧计算坐标变换
        self.metas = metas
        # 保存置信度，detach截断梯度
        self.prev_confidence = det_confidence.detach()
        # 保存实例ID，供下一帧ID匹配
        self.prev_instance_id = instance_id

    # 4.4 缓存当前帧自车的最终状态，供下一帧初始化使用【调用时机：motion head前向结束后】
    def cache_planning(self, ego_feature, ego_status):
        # (1) 保存自车状态（速度、航向等），detach截断梯度
        self.prev_ego_status = ego_status.detach()
        
        # (2) 更新队列中最后一帧的自车特征为精修后的最终特征
        self.ego_feature_queue[-1] = ego_feature.detach()
