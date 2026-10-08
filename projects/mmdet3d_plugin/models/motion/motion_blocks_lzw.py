import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import Linear, Scale, bias_init_with_prob    # 导入MMCV的线性层、缩放层、偏置初始化工具
from mmcv.runner.base_module import Sequential, BaseModule # 导入MMCV的基础模块基类与序列化容器
from mmcv.cnn import xavier_init                           # 导入xavier初始化工具
from mmcv.cnn.bricks.registry import (PLUGIN_LAYERS)       # 导入插件层注册器

from projects.mmdet3d_plugin.core.box3d import * # 导入3D框维度索引常量
from ..blocks import linear_relu_ln              # 导入线性+ReLU+LayerNorm的基础网络构建工具


# 运动与规划并行精修模块
@PLUGIN_LAYERS.register_module() # 将该类注册到插件层注册器，可通过配置按类名实例化
class MotionPlanningRefinementModule(BaseModule):
    """
    运动与规划并行精修模块
    核心作用：解码器每层的输出头，并行输出运动预测与规划的分类、回归结果，同时预测自车状态
    设计特点：两路任务共享相同的网络结构范式，仅输出维度不同，契合论文并行设计思想
    """

    # 1. 初始化精修模块，构建5个独立的输出分支
    def __init__(
        self,
        embed_dims=256, # 输入特征嵌入维度
        fut_ts=12,      # 周围目标的运动预测的未来时间步数
        fut_mode=6,     # 周围目标的运动预测的轨迹模态数
        ego_fut_ts=6,   # 自车规划的未来时间步数
        ego_fut_mode=3, # 自车单驾驶命令的轨迹模态数
    ):
        super(MotionPlanningRefinementModule, self).__init__() # 调用父类初始化

        # 保存参数  
        self.embed_dims = embed_dims     # 保存特征嵌入维度
        self.fut_ts = fut_ts             # 保存运动预测时间步数
        self.fut_mode = fut_mode         # 保存运动预测模态数
        self.ego_fut_ts = ego_fut_ts     # 保存自车规划时间步数
        self.ego_fut_mode = ego_fut_mode # 保存自车规划单命令模态数

        # ========== 1. 运动预测分类分支 ==========
        # 结构：线性+ReLU+LN的基础块 + 最终线性层，输出每个轨迹模态的置信度
        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2), # 1层线性+ReLU+LN的特征变换
            Linear(embed_dims, 1),             # 最终输出1个值，即该模态的分类logits
        )

        # ========== 2. 运动预测回归分支 ==========
        # 结构：两层MLP，输出所有时间步的xy坐标增量
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), # 第一层线性变换+激活
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims), # 第二层线性变换+激活
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2), # 最终输出 fut_ts * 2 个值，对应所有时间步的xy坐标增量
        )

        # ========== 3. 规划分类分支 ==========
        # 结构与运动分类分支一致，输出自车每个规划模态的置信度
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )

        # ========== 4. 规划回归分支 ==========
        # 结构与运动回归分支一致，输出自车轨迹的xy坐标增量
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

        # ========== 5. 自车状态预测分支 ==========
        # 两层MLP，输出10维自车状态（速度、加速度、航向等）
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10), # 输出10维自车状态向量
        )

    # 2. 权重初始化：对分类分支的最后一层偏置做特殊初始化
    def init_weight(self):
        """
            权重初始化：对分类分支的最后一层偏置做特殊初始化
            原理：用bias_init_with_prob将初始偏置设为对应0.01概率的值，让训练初期负样本主导，更稳定
            无输入参数，无返回值，原地修改模型参数
        """
        # 计算对应0.01概率的偏置值（sigmoid反推）
        bias_init = bias_init_with_prob(0.01)

        # 运动分类分支最后一层偏置初始化
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)

        # 规划分类分支最后一层偏置初始化
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    # 3. 前向传播：并行计算运动预测与规划的所有输出
    def forward(
        self,
        motion_query,     # 运动预测查询，形状 [B, 900, fut_mode=6, embed_dims]
        plan_query,       # 规划查询，形状 [B, 1, 3*ego_fut_mode=18, embed_dims]，3对应三种驾驶命令
        ego_feature,      # 自车语义特征，形状 [B, 1, embed_dims]
        ego_anchor_embed, # 自车锚点位置嵌入，形状 [B, 1, embed_dims]
    ):
        # 获取batch大小和检测实例数量
        bs, num_anchor = motion_query.shape[:2]

        # (1.1) 运动分类分支前向，去掉最后一维的单值维度，得到 [B, 900, fut_mode=6]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        # (1.2) 运动回归分支前向，重塑为 [B, 900, 模态数, 时间步, 2] 的轨迹格式
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)

        # (2.1) 规划分类分支前向，去掉最后一维，得到 [B, 1, 3*ego_fut_mode=18]
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        # (2.2) 规划回归分支前向，重塑为 [B, 1, 总模态数, 时间步, 2] 的轨迹格式
        plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, 3 * self.ego_fut_mode, self.ego_fut_ts, 2)

        # (3) 自车状态预测：自车特征与锚点嵌入相加后输入分支
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)

        # (4) 返回五路输出
        """
            motion_cls      (Tensor): 运动预测分类置信度，形状 [B, 900, fut_mode=6]
            motion_reg      (Tensor): 运动预测轨迹回归，形状 [B, 900, fut_mode=6, fut_ts=12, 2]
            plan_cls        (Tensor): 规划分类置信度，形状 [B, 1, 3*ego_fut_mode=18]
            plan_reg        (Tensor): 规划轨迹回归，形状 [B, 1, 3*ego_fut_mode=18, ego_fut_ts=6, 2]
            planning_status (Tensor): 自车状态预测，形状 [B, 1, 10]
        """
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status
