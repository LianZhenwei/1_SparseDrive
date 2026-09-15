from typing import List, Optional, Tuple, Union  # 导入类型提示工具，用于参数、返回值的类型标注，提升代码可读性与IDE提示
import warnings                                  # 导入警告模块，用于发出非致命性提醒
import copy                                      # 导入深拷贝工具，用于对象的完整复制
import numpy as np                               # 导入numpy数值计算库，用于数组运算与数据加载
import cv2                                       # 导入OpenCV库，预留用于图像相关后处理
import torch                                     # 导入PyTorch核心库
import torch.nn as nn                            # 导入PyTorch神经网络模块
from mmcv.utils import build_from_cfg            # 从mmcv导入配置构建工具，用于根据配置字典从注册器实例化模块
from mmcv.cnn import Linear, bias_init_with_prob # 从mmcv导入线性层与偏置初始化工具
from mmcv.runner import BaseModule, force_fp32   # 从mmcv导入基础模块基类与强制FP32计算装饰器（避免混合精度下数值不稳定）
from mmcv.cnn.bricks.registry import (           # 从mmcv导入各类神经网络组件的注册器
    ATTENTION,           # 注意力模块注册器
    PLUGIN_LAYERS,       # 插件层注册器
    POSITIONAL_ENCODING, # 位置编码注册器
    FEEDFORWARD_NETWORK, # 前馈网络注册器
    NORM_LAYERS,         # 归一化层注册器
)

from mmdet.core import reduce_mean                             # 从mmdet导入分布式均值归约工具，多卡训练时同步全局正样本数量
from mmdet.models import HEADS                                 # 从mmdet导入检测头注册器
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS # 从mmdet导入边界框采样器、编码器/解码器注册器
from mmdet.models import build_loss                            # 从mmdet导入损失函数构建工具

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners # 导入项目内3D框转角点的工具函数
from projects.mmdet3d_plugin.core.box3d import *                    # 导入3D框各维度的索引常量（X/Y/Z/W/L/H/SIN_YAW/COS_YAW/VX等）
from ..attention import gen_sineembed_for_position                  # 导入正弦位置编码生成函数
from ..blocks import linear_relu_ln                                 # 导入线性+ReLU+LayerNorm的基础网络构建函数
from ..instance_bank import topk                                    # 导入实例银行中的TopK工具函数，用于按置信度筛选实例


# MotionPlanningHead 是运动预测与规划统一解码头，并行处理多模态轨迹输出
@HEADS.register_module() # 将该类注册到MMDetection的检测头注册器，可通过配置文件按类名实例化
class MotionPlanningHead(BaseModule):
    """
    运动预测与规划统一解码头
    核心设计：运动预测和规划共享同一套解码器架构，并行处理多模态轨迹输出
    支持时序信息融合、智能体间交互、地图约束交叉注意力、分层规划选择
    """

    # 1. 初始化
    def __init__(
        self,
        fut_ts=12,              # 运动预测的未来时间步数量（周围目标预测时长）
        fut_mode=6,             # 运动预测单目标的轨迹模态数（多候选轨迹）
        ego_fut_ts=6,           # 自车规划的未来时间步数量（自车规划时长）
        ego_fut_mode=3,         # 自车单驾驶命令下的轨迹模态数
        motion_anchor=None,     # 运动预测锚点的npy文件路径，按类别初始化轨迹
        plan_anchor=None,       # 规划锚点的npy文件路径，多模态初始轨迹
        embed_dims=256,         # 特征向量的嵌入维度，全网络统一特征维度
        decouple_attn=False,    # 是否启用解耦注意力（特征与位置拼接后输入注意力）
        instance_queue=None,    # 时序实例队列配置，管理历史帧智能体与自车状态
        operation_order=None,   # 解码器每层的操作执行顺序列表
        temp_graph_model=None,  # 时序交叉注意力配置，当前帧与历史帧交互
        graph_model=None,       # 自注意力图模型配置，当前帧实例间交互
        cross_graph_model=None, # 地图交叉注意力配置，实例与地图要素交互
        norm_layer=None,        # 归一化层配置
        ffn=None,               # 前馈网络配置
        refine_layer=None,      # 轨迹精修层配置，输出分类与回归结果
        motion_sampler=None,    # 运动预测目标采样器配置，训练时分配监督目标
        motion_loss_cls=None,   # 运动预测分类损失配置
        motion_loss_reg=None,   # 运动预测回归损失配置
        planning_sampler=None,  # 规划目标采样器配置
        plan_loss_cls=None,     # 规划分类损失配置
        plan_loss_reg=None,     # 规划回归损失配置
        plan_loss_status=None,  # 自车状态回归损失配置
        motion_decoder=None,    # 运动预测后处理解码器配置
        planning_decoder=None,  # 规划后处理解码器配置
        num_det=50,             # 参与交互建模的检测实例TopK数量
        num_map=10,             # 参与交互建模的地图实例TopK数量
    ):
        super(MotionPlanningHead, self).__init__() # 调用父类BaseModule的初始化方法，传入初始化配置

        # ============== 1. 基础超参数保存 ==============
        self.fut_ts = fut_ts                   # 保存运动预测时间步数
        self.fut_mode = fut_mode               # 保存运动预测模态数
        self.ego_fut_ts = ego_fut_ts           # 保存自车规划时间步数
        self.ego_fut_mode = ego_fut_mode       # 保存自车规划模态数
        self.decouple_attn = decouple_attn     # 保存是否启用解耦注意力
        self.operation_order = operation_order # 保存解码器操作顺序

        # ============== 2. 子模块构建工具函数 ==============
        def build(cfg, registry):
            """内部工具：根据配置和注册器构建模块，空配置返回None
            Args:
                cfg: 模块配置字典，None表示不构建
                registry: 对应的注册器
            Returns:
                实例化后的模块，或None
            """
            # 根据配置从注册器构建模块实例，若配置为空则返回None
            if cfg is None:
                return None # 若配置为空则返回None
            return build_from_cfg(cfg, registry) # 根据配置从注册器构建模块实例
        
        # ============== 3. 核心业务组件构建 ==============
        # 什么是 WTA 模态：MotionTarget 和 PlanningTarget 中都会计算各模态与 GT 的距离，并通过 argmin 得到 WTA 模态
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)     # 3.1 构建时序实例队列：缓存历史帧的智能体与自车状态，实现时序信息复用
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)     # 3.2 【从多个Agent轨迹模态中选WTA模态】           构建运动预测目标采样器：训练时分配轨迹监督目标
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS) # 3.3 【先按驾驶指令选分组，再选规划WTA模态】        构建规划目标采样器：训练时分配自车规划监督目标
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)       # 3.4 【将Agent轨迹增量累加并转换到全局/当前坐标表达】构建运动预测后处理解码器：将原始预测转为结构化轨迹结果
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)   # 3.5 【从规划模态中选择最终轨迹，并可进行碰撞重打分】 构建规划后处理解码器：实现分层规划选择与碰撞重打分

        # ============== 4. 解码器层构建 ==============
        # (1.1) 建立操作名到「配置字典+对应注册器」的映射表，用于批量构建解码器层
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],   # 时序交叉注意力
            "gnn": [graph_model, ATTENTION],             # 帧内自注意力
            "cross_gnn": [cross_graph_model, ATTENTION], # 地图交叉注意力
            "norm": [norm_layer, NORM_LAYERS],           # 归一化层
            "ffn": [ffn, FEEDFORWARD_NETWORK],           # 前馈网络
            "refine": [refine_layer, PLUGIN_LAYERS],     # 轨迹精修层
        }
        # (1.2) 根据操作顺序批量实例化所有解码器层，存入ModuleList统一管理参数：sparsedrive_small_stage2.py 中定义的解码器层顺序是 [temp_gnn -> gnn -> norm -> cross_gnn -> norm -> ffn -> norm]*重复3次 → refine
        self.layers = nn.ModuleList(
            [
                # 从映射表取出对应配置和注册器，调用build函数构建
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        # (2) 保存特征嵌入维度
        self.embed_dims = embed_dims
        
        # ============== 5. 解耦注意力投影层 ==============
        if self.decouple_attn:
            # 注意力计算前，将value特征投影为2倍维度，增强特征表达能力
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            # 注意力计算后，将结果投影回原始维度，保持输出维度一致
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            # 不启用解耦注意力时，使用恒等映射，不做任何变换
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        # ============== 6. 损失函数构建 ==============
        self.motion_loss_cls = build_loss(motion_loss_cls)   # 构建运动预测分类损失
        self.motion_loss_reg = build_loss(motion_loss_reg)   # 构建运动预测回归损失
        self.plan_loss_cls = build_loss(plan_loss_cls)       # 构建规划分类损失
        self.plan_loss_reg = build_loss(plan_loss_reg)       # 构建规划回归损失
        self.plan_loss_status = build_loss(plan_loss_status) # 构建自车状态回归损失

        # ============== 7.1 运动预测锚点与编码器 ==============
        # (1) 从.npy文件加载按类别预设的轨迹锚点（不同类别初始轨迹不同）
        motion_anchor = np.load(motion_anchor)
        # (2) 将agent锚点注册为不可学习的参数，随模型一起保存与加载
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False, # 锚点固定，不参与梯度更新
        )
        # (3) 运动锚点位置编码器：将轨迹末端坐标编码为高维位置嵌入
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1), # 线性+ReLU+LN的基础块
            Linear(embed_dims, embed_dims),    # 最终输出线性层
        )
        '''
            self.motion_anchor：固定的几何轨迹模板，不训练
            self.motion_anchor_encoder：可学习编码器，需要训练
        '''

        # ============== 7.2 规划锚点与编码器 ==============
        # (1) 从.npy文件加载自车规划的多模态初始轨迹锚点
        plan_anchor = np.load(plan_anchor)
        # (2) 将ego锚点注册为不可学习参数
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        # (3) 规划锚点位置编码器：将规划轨迹末端坐标编码为高维位置嵌入
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )
        '''
            self.plan_anchor：固定的几何轨迹模板，不训练
            self.plan_anchor_encoder：可学习编码器，需要训练
        '''

        # ============== 8. 感知实例筛选数量 ==============
        self.num_det = num_det # 参与交互建模的检测目标TopK数量，保证稀疏高效【值为50，即帧内交互时选取置信度最高的50个 Agent 作为上下文】
        self.num_map = num_map # 参与交互建模的地图要素TopK数量【值为10，即选取置信度最高的10个地图实例作为上下文】

    # 2. 权重初始化函数：对非精修层的参数做xavier均匀初始化，精修层调用自身init_weight方法
    def init_weights(self):
        # (1) 遍历所有解码器层
        for i, op in enumerate(self.operation_order):
            # (a) 层为空则跳过
            if self.layers[i] is None:
                continue
            # (b) 非精修层的参数执行xavier均匀初始化
            elif op != "refine":
                # 遍历该层所有参数
                for p in self.layers[i].parameters():
                    # 仅对维度大于1的参数（权重矩阵）做初始化
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        # (2) 遍历所有子模块，若模块自身定义了init_weight方法则调用
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    # 3. 获取运动预测锚点：根据检测类别选取对应锚点，并转到激光雷达全局坐标系
    def get_motion_anchor(
        self, 
        classification, # 检测分类预测，[B, num_anchor, num_cls]
        prediction,     # 检测3D框预测，[B, num_anchor, box_dims]
    ):
        # 对每个检测实例取概率最大的类别索引，形状 [B, num_anchor]
        cls_ids = classification.argmax(dim=-1)

        # 按类别索引从锚点库中选取对应轨迹锚点，形状 [B, num_anchor, fut_mode=6, fut_ts=12, 2]
        motion_anchor = self.motion_anchor[cls_ids]

        # 检测框detach截断梯度，锚点变换不反向传播到检测分支
        prediction = prediction.detach()

        # 将智能体局部坐标系下的锚点旋转平移到激光雷达全局坐标系
        return self._agent2lidar(motion_anchor, prediction) # 返回激光雷达坐标系下的轨迹锚点，形状 [B, num_anchor, fut_mode=6, fut_ts=12, 2]

    # 4. 坐标变换：将智能体局部坐标系的轨迹，旋转到激光雷达全局坐标系
    def _agent2lidar(
        self, 
        trajs, # 智能体坐标系下的轨迹，形状 [B, num_anchor, fut_mode, fut_ts, 2]
        boxes  # 检测3D框，包含航向角sin/cos信息
    ):
        # (1) 计算 sin_yaw 和 cos_yaw
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW]) # 从sin/cos反算航向角yaw，形状 [B, num_anchor]
        cos_yaw = torch.cos(yaw)                                    # 计算航向角的余弦值
        sin_yaw = torch.sin(yaw)                                    # 计算航向角的正弦值

        # (2) 构建旋转矩阵的转置（用于将局部坐标旋转到全局）
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),  # 旋转矩阵第一行
                torch.stack([-sin_yaw, cos_yaw]), # 旋转矩阵第二行
            ]
        )

        # (3) 爱因斯坦求和约定：批量对所有轨迹点做旋转变换
        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T) # 维度含义：a=2x2旋转矩阵, b=2, c=B, i=num_anchor, j=fut_mode, k=fut_ts
        return trajs_lidar # 激光雷达坐标系下的轨迹，形状与输入 trajs 一致，为 [B, num_anchor, fut_mode, fut_ts, 2]

    # 5. 图注意力统一封装：处理解耦注意力逻辑，与检测头保持一致的注意力范式
    def graph_model(
        self,
        index,          # 对应层在layers列表中的索引
        query,          # 查询特征张量
        key=None,       # 键特征张量，为None时表示自注意力（key=query）
        value=None,     # 值特征张量
        query_pos=None, # 查询的位置嵌入
        key_pos=None,   # 键的位置嵌入
        **kwargs,       # 其他注意力参数（如注意力掩码、padding掩码等）
    ):
        # (1) 解耦注意力模式：将位置嵌入直接拼接到特征上，替代传统相加式位置编码
        if self.decouple_attn:
            # 查询特征拼接查询位置嵌入，维度翻倍
            query = torch.cat([query, query_pos], dim=-1)
            # 键特征非空时，拼接键位置嵌入
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            # 拼接后不再需要单独传入位置参数，置空
            query_pos, key_pos = None, None

        # (2) 值特征非空时，先通过fc_before做维度投影
        if value is not None:
            value = self.fc_before(value)

        # (3) 调用对应索引的注意力层计算，结果经fc_after投影回原维度后返回
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        ) # 返回注意力输出特征张量

    # 6. 核心前向传播函数：串联特征筛选、时序融合、交互建模、轨迹精修全流程
    def forward(
        self, 
        det_output,     # 检测分支输出字典，含实例特征、锚点、分类、置信度等
        map_output,     # 地图分支输出字典，含实例特征、锚点、分类、置信度等
        feature_maps,   # 多尺度多视角图像特征图列表
        metas,          # 元数据字典，含相机参数、时间戳、标定矩阵、全局变换矩阵等
        anchor_encoder, # 检测分支的共享的3D锚点编码器，来自检测头
        mask,           # 时序有效掩码，标记哪些batch的历史帧可用
        anchor_handler, # 检测分支的锚点处理器，提供坐标投影能力
    ):   
        # ============== 1. 检测实例TopK筛选 ==============
        # (1) 提取 Detection 任务输出的实例信息
        instance_feature = det_output["instance_feature"]               # 提取检测分支的实例特征，[B, num_det_anchor=900, embed_dims=256]
        anchor_embed = det_output["anchor_embed"]                       # 提取检测分支的锚点位置嵌入，[B, 900, 256]
        det_classification = det_output["classification"][-1].sigmoid() # 提取最后一层检测分类结果，并经sigmoid转为0~1概率，[B, 900, 10]
        det_anchors = det_output["prediction"][-1]                      # 提取最后一层检测3D框预测结果，[B, 900, 11]

        # (2) 计算每个检测实例的最大置信度（所有类别取最大），[B, 900]
        det_confidence = det_classification.max(dim=-1).values

        # (3) 按置信度取TopK个检测实例，作为交互建模的上下文，同步筛选特征与位置嵌入【规控头会为 900 个检测 Query 都生成运动预测，但进行帧内 Agent 交互时，只取置信度最高的 self.num_det=50 个作为 Key 和 Value】
        _, (instance_feature_selected, anchor_embed_selected) = topk(det_confidence, self.num_det, instance_feature, anchor_embed) # 返回 [B, 50, 256] 和 [B, 50, 256]
        '''lzw
            规控头会为900个检测 Query 都生成运动预测。
            但进行帧内 Agent 交互时，只取置信度最高的50个作为 Key 和 Value。       
            所以需要区分：
                Query：全部900个 Agent + Ego
                Key/Value：Top-50 Agent + Ego
            这是一种稀疏化策略：
                900个 Agent Query 的未来预测轨迹都可以得到更新；
                只有置信度最高的50个 Agent 能被其他实例重点读取交互；
                低置信度 Query 不会成为主要上下文。        
        '''

        # ============== 2. 地图实例TopK筛选 ==============
        # (1) 提取 Map 任务输出的实例信息
        map_instance_feature = map_output["instance_feature"]           # 提取地图分支的实例特征，[B, 100, 256]
        map_anchor_embed = map_output["anchor_embed"]                   # 提取地图分支的锚点位置嵌入，[B, 100, 256]
        map_classification = map_output["classification"][-1].sigmoid() # 提取最后一层地图分类结果，并转概率，[B, 100, 3]
        map_anchors = map_output["prediction"][-1]                      # 提取最后一层地图线预测结果

        # (2) 计算每个地图实例的最大置信度（所有类别取最大），[B, 100]
        map_confidence = map_classification.max(dim=-1).values

        # (3) 按置信度取TopK个地图要素，用于后续交叉注意力引入地图约束【从 100 个地图 Query 中选取置信度最高的 self.num_map=10 个 “Top-10车道线、道路边界、人行横道实例”，作为 Agent-Map Cross Attention 的 Key】
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(map_confidence, self.num_map, map_instance_feature, map_anchor_embed) # 返回 [B, 10, 256] 和 [B, 10, 256]

        # ============== 3. 获取自车实例与时序队列 ==============
        # (1) 获取检测实例的batch大小、数量、特征维度
        B, num_anchor, dim = instance_feature.shape

        # (2) 从实例队列获取当前与历史的智能体/自车状态
        (
            ego_feature,           # 编码前视相机的最后一级特征图得到的当前帧自车特征 [B, 1, embed_dims=256]
            ego_anchor,            # 继承自固定模版并设置其VY速度为上一帧预测VY速度得到当前帧自车锚点 ego_anchor [B, 1, box_dims=11]
            temp_instance_feature, # “时序agent特征” + “时序ego特征”【agent已ID时序对齐】，[B, 900+1, queue_len=4, 256]
            temp_anchor,           # 已转换到当前坐标系的 “时序agent anchor” + “时序ego anchor”【agent已ID时序对齐】，[B, 900+1, queue_len=4, 11]
            temp_mask,             # 时序有效掩码，[B, 900+1, queue_len=4]
        ) = self.instance_queue.get(
            det_output,
            feature_maps,
            metas,
            B,
            mask,
            anchor_handler,
        )

        # (3) 计算各 anchor 的位置嵌入
        ego_anchor_embed = anchor_encoder(ego_anchor)   # 用共享的锚点编码器计算当前帧自车 anchor 的位置嵌入，[B, 1, 256]
        temp_anchor_embed = anchor_encoder(temp_anchor) # 用共享的锚点编码器计算 “时序agent anchor” + “时序ego anchor” 的位置嵌入，[B, 900+1, queue_len=4, 256]

        # (4) 展平，以适配注意力输入格式
        temp_instance_feature = temp_instance_feature.flatten(0, 1) # “时序agent特征” + “时序ego特征” 展平前两维，适配注意力输入格式，[B*901, queue_len=4, 256]
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)         # “时序agent anchor位置嵌入” + “时序ego anchor位置嵌入” 展平前两维，适配注意力输入格式，[B*901, queue_len=4, 256]
        temp_mask = temp_mask.flatten(0, 1)                         # 时序掩码展平前两维，适配注意力输入格式，[B*901, queue_len=4]

        # ============== 4. 初始化他车运动预测模式锚点和自车规划模式锚点 ==============
        # (1) 由 .npy 文件生成他车运动预测的初始轨迹锚点，且已转到全局坐标系，[B, 900, fut_mode=6, fut_ts=12, 2]
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors)
        # (2) 由 .npy 文件生成自车规划的初始规划锚点，且规划锚点在 batch 维度复制，形状 [B, 3, ego_fut_mode=6, ego_fut_ts=6, 2]，其中 3 对应三种驾驶命令：左转、直行、右转
        plan_anchor = torch.tile(self.plan_anchor[None], (B, 1, 1, 1, 1))

        # ============== 5. 模式查询编码 ==============
        # (1) 取运动轨迹最后一个点的坐标，生成正弦位置编码，再经MLP编码为高维模式查询
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :])) # [B, 900, fut_mode=6, 256]
        # (2.1) 取规划轨迹最后一个点的坐标，生成正弦位置编码
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :]) # [B, 3, ego_fut_mode=6, 256]
        # (2.2) 经MLP编码后展平模态维度，扩展维度适配后续广播
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1) # [B, 1, 3*ego_fut_mode=18, 256]，这18个 Query 就分别代表18种自车规划意图

        # ============== 6. 当前帧实例特征拼接：检测+自车 ==============
        # (1.1) 置信度最高的50个检测实例拼接ego特征，作为自注意力的key/value，[B, 50+1=51, 256]【置信度最高的50个检测实例 + 1个Ego】
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1)
        # (1.2) 位置嵌入同步拼接，作为自注意力的key/value位置嵌入，[B, 50+1=51, 256]【置信度最高的50个检测实例 + 1个Ego】
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1)

        # (2.1) 900个检测实例特征拼接ego特征，作为query与后续精修输入，[B, 900+1=901, 256]【900个检测实例 + 1个Ego】
        instance_feature = torch.cat([instance_feature, ego_feature], dim=1)
        # (2.2) 位置嵌入同步拼接，作为query与后续精修输入，[B, 900+1=901, 256]【900个检测实例 + 1个Ego】
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)

        # ============== 7. 解码器层迭代执行 ==============
        motion_classification = []   # 初始化各层运动分类结果列表
        motion_prediction = []       # 初始化各层运动轨迹回归结果列表
        planning_classification = [] # 初始化各层规划分类结果列表
        planning_prediction = []     # 初始化各层规划轨迹回归结果列表
        planning_status = []         # 初始化各层自车状态结果列表

        # 遍历所有解码器操作，逐层执行：sparsedrive_small_stage2.py 中定义的解码器层顺序是 [temp_gnn -> gnn -> norm -> cross_gnn -> norm -> ffn -> norm]*重复3次 → refine
        for i, op in enumerate(self.operation_order):
            # 层为空则跳过
            if self.layers[i] is None:
                continue

            # 7.1 时序GNN【每个 Agent 只看自己的历史】：“当前帧 agent_i” 与 “它自己的多帧历史” 做交叉注意力，融合时序信息
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),       # query: 将 “900个检测实例的当前帧特征” + “ego的当前帧特征” 展平batch与anchor维度，增加序列维度【即[B, 901, 256] → [B*901, 1, 256]】
                    temp_instance_feature,                             # key  : “900个检测实例的时序特征” + “ego时序特征” [B*901, queue_len=4, 256]
                    temp_instance_feature,                             # value: “900个检测实例的时序特征” + “ego时序特征” [B*901, queue_len=4, 256]
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1), # query位置嵌入: 将 “当前帧agent anchor位置嵌入” + “当前帧ego anchor位置嵌入” 同步展平并增加维度【即[B, 901, 256] → [B*901, 1, 256]】
                    key_pos=temp_anchor_embed,                         # key位置嵌入  : “时序agent anchor位置嵌入” + “时序ego anchor位置嵌入” [B*901, queue_len=4, 256]
                    key_padding_mask=temp_mask,                        # 时序padding掩码，标记无效历史帧 [B*901, queue_len=4]
                )
                # 注意力输出恢复为原始形状 [B, 900+1, embed_dims=256]
                instance_feature = instance_feature.reshape(B, num_anchor + 1, dim)

            # 7.2 自注意力GNN【Agent-Agent Interaction】：当前帧所有实例（检测目标+自车）之间做交互建模：Ego读取周围Agent、周围Agent也读取Ego、Agent之间相互读取
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,          # query    : 900个Agent当前帧特征 + Ego当前帧特征
                    instance_feature_selected, # key/value: Top-50 Agent当前帧特征 + Ego当前帧特征，减少计算量
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )

            # 7.3 归一化层 或 前馈网络FFN【FFN：完成 Attention 后的通道维非线性变换：[B, 901, 256] → AsymmetricFFN → [B, 901, 256]】
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature) # 直接调用对应层处理特征

            # 7.4 交叉注意力GNN【Agent-Map Interaction】：实例与地图要素做交叉注意力，引入地图约束：agent读取地图、ego读取地图
            elif op == "cross_gnn":
                instance_feature = self.layers[i](
                    instance_feature,                  # query: 900个Agent当前帧特征 + Ego当前帧特征
                    key=map_instance_feature_selected, # key  : Top-10地图实例的当前帧特征
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )

            # 7.5 精修层：并行输出运动预测与规划的分类、回归结果【见 motion_blocks.py 里的 MotionPlanningRefinementModule 类】
            elif op == "refine":
                # (a.1) 运动查询 = 模式锚点查询 + 实例语义特征（扩展模态维度实现广播相加），即由 “Agent i 的语义和几何信息 + 第m种运动意图” 得到每个目标有6个轨迹 Query，[B, 900, fut_mode=6, 256]
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                # (a.2) 规划查询 = 模式锚点查询 + 自车语义特征（扩展模态维度实现广播相加），即由 “Ego语义和几何信息 + 第m种规划意图” 得到自车的 “3种驾驶指令 × 6个轨迹模态 = 18个规划 Query”，[B, 1, 3*fut_mode=18, 256]
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2) 
                
                # (b) 调用精修层，同时输出运动与规划两路结果
                (
                    motion_cls,  # 运动分类置信度，[B, 900, fut_mode=6]
                    motion_reg,  # 运动轨迹回归，[B, 900, fut_mode=6, fut_ts=12, 2]
                    plan_cls,    # 规划分类置信度，[B, 1, 3*fut_mode=18]
                    plan_reg,    # 规划轨迹回归，[B, 1, 3*ego_fut_mode=18, ego_fut_ts=6, 2]
                    plan_status, # 自车状态预测，[B, 1, 10]
                ) = self.layers[i](
                    motion_query,
                    plan_query,
                    instance_feature[:, num_anchor:], # 自车特征
                    anchor_embed[:, num_anchor:],     # 自车位置嵌入
                )

                # (c) 保存
                motion_classification.append(motion_cls) # 保存该层运动分类结果
                motion_prediction.append(motion_reg)     # 保存该层运动回归结果
                planning_classification.append(plan_cls) # 保存该层规划分类结果
                planning_prediction.append(plan_reg)     # 保存该层规划回归结果
                planning_status.append(plan_status)      # 保存该层自车状态结果
        
        # ============== 8. 更新时序缓存 ==============
        # 缓存当前帧运动实例状态，供下一帧时序建模使用
        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        # 缓存当前帧自车规划状态，供下一帧时序建模使用
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)

        # ============== 9. 组装输出结果 ==============
        # (1) 组装运动预测输出字典
        motion_output = {
            "classification": motion_classification,          # 各层分类置信度列表
            "prediction": motion_prediction,                  # 各层轨迹回归列表
            "period": self.instance_queue.period,             # 每个实例的存活时长
            "anchor_queue": self.instance_queue.anchor_queue, # 历史锚点队列
        }
        # (2) 组装规划输出字典
        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        # (3) 返回两路输出：运动预测结果字典、规划结果字典
        return motion_output, planning_output

    # 7.1 总损失计算入口：分别计算运动预测损失与规划损失，合并后返回 
    def loss(self,
        motion_model_outs,   # 运动预测前向输出字典
        planning_model_outs, # 规划前向输出字典
        data,                # 真值数据字典，含各类监督标签
        motion_loss_cache    # 运动损失缓存，内含检测匹配索引，避免重复做匈牙利匹配
    ):
        # 初始化损失字典
        loss = {}
        
        # (1.1) 计算运动预测分支的所有损失
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        # (1.2) 运动损失合并到总损失字典
        loss.update(motion_loss)

        # (2.1) 计算规划分支的所有损失
        planning_loss = self.loss_planning(planning_model_outs, data)
        # (2.2) 规划损失合并到总损失字典
        loss.update(planning_loss)

        # (3) 返回总损失字典，键为损失名，值为损失标量
        return loss

    # 7.2 运动预测分支损失计算：逐层计算辅助监督，选最优模态计算分类与回归损失
    @force_fp32(apply_to=("model_outs"))
    def loss_motion(
        self, 
        model_outs,       # 运动预测输出字典
        data,             # 真值数据字典
        motion_loss_cache # 损失缓存，含检测匹配索引
    ):
        # 提取各层分类置信度、各层轨迹回归
        cls_scores = model_outs["classification"] # 提取各层分类置信度
        reg_preds = model_outs["prediction"]      # 提取各层轨迹回归

        # 初始化输出损失字典
        output = {}

        # 1. 遍历每一层解码器，逐层计算辅助损失
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            # (1) 调用运动目标采样器，生成训练监督目标
            (
                cls_target, # 最优模态索引，作为分类目标
                cls_weight, # 分类损失权重掩码
                reg_pred,   # 筛选后的最优轨迹预测
                reg_target, # 轨迹回归真值
                reg_weight, # 回归损失权重掩码
                num_pos,    # 正样本数量
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"], # 周围目标未来轨迹真值
                data["gt_agent_fut_masks"], # 轨迹有效掩码
                motion_loss_cache,          # 检测匹配缓存
            )
            # 多卡训练时同步全局正样本数量，取均值，最小为1避免除零
            num_pos = max(reduce_mean(num_pos), 1.0)


            # (2) 计算分类损失
            # (2.a) 展平前两维（batch + 实例数），适配损失函数输入格式
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            # (2.b) 计算分类损失
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)


            # (3) 计算回归损失
            # (3.a) 展平
            reg_weight = reg_weight.flatten(end_dim=1) # 回归权重展平
            reg_pred = reg_pred.flatten(end_dim=1)     # 轨迹预测展平
            reg_target = reg_target.flatten(end_dim=1) # 轨迹真值展平

            # (3.b) 权重增加最后一维，适配坐标维度广播
            reg_weight = reg_weight.unsqueeze(-1)

            # (3.c) 增量轨迹累加为绝对坐标（网络输出增量，损失计算绝对位置）
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)

            # (3.d) 计算回归损失
            reg_loss = self.motion_loss_reg(reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos)

            # (4) 将该层损失存入输出字典，按层编号命名
            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )
        
        # 2. 返回所有层的运动损失
        return output # 返回运动损失字典，包含各层分类、回归损失

    # 7.3 规划分支损失计算：按驾驶命令筛选模态，计算分类、回归、状态三类损失
    @force_fp32(apply_to=("model_outs"))
    def loss_planning(
        self, 
        model_outs, # 规划输出字典
        data        # 真值数据字典
    ):
        # 提取各层规划分类置信度、各层规划轨迹回归、各层自车状态预测
        cls_scores = model_outs["classification"] # 提取各层规划分类置信度
        reg_preds = model_outs["prediction"]      # 提取各层规划轨迹回归
        status_preds = model_outs["status"]       # 提取各层自车状态预测

        # 初始化输出损失字典
        output = {}

        # 1. 遍历每一层解码器，逐层计算辅助损失
        for decoder_idx, (cls, reg, status) in enumerate(zip(cls_scores, reg_preds, status_preds)):
            # (1) 调用规划目标采样器，生成规划监督目标
            (
                cls,        # 筛选后的分类置信度
                cls_target, # 最优模态索引
                cls_weight, # 分类损失权重
                reg_pred,   # 筛选后的最优轨迹预测
                reg_target, # 轨迹真值
                reg_weight, # 回归损失权重
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'], # 自车未来轨迹真值
                data['gt_ego_fut_masks'], # 轨迹有效掩码
                data,                     # 完整数据字典，含驾驶命令
            )

            # (2) 展平维度计算分类损失
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            # (3) 展平维度计算回归损失
            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_loss = self.plan_loss_reg(reg_pred, reg_target, weight=reg_weight)

            # (4) 计算自车状态回归损失，去掉多余的模态维度
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            # (5) 将该层损失存入输出字典
            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )
        
        # 2. 返回所有层的规划损失
        return output # 规划损失字典，包含各层分类、回归、状态损失

    # 8. 后处理入口：调用对应解码器，将原始张量预测转为结构化的可评估结果
    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self, 
        det_output,      # 检测输出字典
        motion_output,   # 运动预测输出字典
        planning_output, # 规划输出字典
        data,            # 输入数据字典
    ):
        # (1) 调用运动解码器，生成运动预测的结构化结果
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        # (2) 调用规划解码器，生成规划的结构化结果（含分层选择、碰撞重打分）
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output, 
            data,
        )
        # (3) 返回两路后处理结果
        return motion_result, planning_result # 【motion_result: 运动预测结构化结果列表，每个元素对应一帧】【planning_result: 规划结构化结果列表，每个元素对应一帧】
