
from typing import List, Optional, Tuple, Union # 从 typing 模块导入类型标注工具
import warnings                                 # 导入 warnings 模块，warnings 用于输出警告信息，但在当前这份代码中，warnings 实际上没有被使用
import numpy as np                              # 导入 numpy，但在当前这份代码中，np 实际上没有被使用
import torch                                    # 导入 PyTorch，SparseDriveHead 的输入 feature_maps、head 输出等都是 torch.Tensor
import torch.nn as nn                           # 导入 PyTorch 的神经网络模块，nn 常用于定义网络层，但在当前这份代码中，nn 实际上没有被直接使用
from mmcv.runner import BaseModule  # 从 MMCV 中导入 BaseModule：BaseModule 是 MMCV 对 nn.Module 的进一步封装，支持 init_cfg 权重初始化机制
from mmdet.models import HEADS      # 从 MMDetection 中导入 HEADS 注册表：SparseDriveHead 会注册到 HEADS 中，这样配置文件里写 type='SparseDriveHead' 时，框架就可以自动构建该模块
from mmdet.models import build_head # 从 MMDetection 中导入 build_head：build_head 用于根据配置字典构建具体的 head，例如 det_head、map_head、motion_plan_head


# SparseDriveHead 类是 SparseDrive 的总任务头，作用是统一管理 detection head、map head、motion planning head，也就是把多个子任务 head 包装成一个总 head
@HEADS.register_module() # 将 SparseDriveHead 注册到 MMDetection 的 HEADS 注册表中，注册后，配置文件中可以通过 type='SparseDriveHead' 创建该类对象
class SparseDriveHead(BaseModule): # 继承自 BaseModule
    # 1. 初始化函数：保存任务开关配置，根据任务配置构建各个子 head 模块
    def __init__(
        self,                    # self 表示当前 SparseDriveHead 对象
        task_config: dict,       # task_config 是任务开关配置字典，里面通常有：《task_config['with_det']：是否启用 3D 检测任务》《task_config['with_map']：是否启用地图元素检测任务》《task_config['with_motion_plan']：是否启用运动预测和规划任务》
        det_head = dict,         # det_head 是检测头配置，默认值设置为 dict # 注意：这里写成 det_head = dict 不太规范，按逻辑更推荐写 det_head=None，因为 dict 是 Python 内置类型，不是一个具体配置字典
        map_head = dict,         # map_head 是地图头配置，用于构建地图元素检测模块，例如车道线、边界线、人行横道等
        motion_plan_head = dict, # motion_plan_head 是运动预测和规划头配置，用于构建 motion prediction 和 ego planning 相关模块
        init_cfg=None,           # init_cfg 是 MMCV 的初始化配置，用于指定权重初始化策略
        **kwargs,                # **kwargs 接收额外参数，当前代码里没有显式使用 kwargs，主要是为了兼容配置文件里可能传入的其他字段
    ):
        super(SparseDriveHead, self).__init__(init_cfg) # 调用父类 BaseModule 的初始化函数，把 init_cfg 传进去，让 MMCV 管理权重初始化

        # 保存任务配置task_config，后续 forward、loss、post_process 都会根据 task_config 判断启用哪些子任务
        self.task_config = task_config

        # (1) 若任务配置中启用了检测任务
        if self.task_config['with_det']:
            self.det_head = build_head(det_head) # 根据 det_head 配置字典构建检测头【检测头一般负责 3D bounding box、分类、instance query 等】

        # (2) 若任务配置中启用了地图任务
        if self.task_config['with_map']:
            self.map_head = build_head(map_head) # 根据 map_head 配置字典构建地图头【地图头一般负责车道线、路沿、道路边界等矢量地图元素预测】

        # (3) 若任务配置中启用了运动预测和规划任务
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_head(motion_plan_head) # 根据 motion_plan_head 配置字典构建运动预测和规划头【该 head 一般依赖检测结果、地图结果和图像特征】

    # 2. 权重初始化函数 init_weights()：用于初始化各个子模块的权重，MMCVM/MMDetection 在构建模型后通常会调用这个函数：
    def init_weights(self):
        # (1) 若任务配置中启用了检测任务
        if self.task_config['with_det']:
            self.det_head.init_weights() # 初始化检测头权重

        # (2) 若任务配置中启用了地图任务
        if self.task_config['with_map']:          
            self.map_head.init_weights() # 初始化地图头权重

        # (3) 若任务配置中启用了运动预测和规划任务
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights() # 初始化运动预测和规划头权重

    # 3. 前向传播，返回检测输出、地图输出、agent运行预测输出、ego自车规划输出：
    def forward(
        self,                                    # self 表示当前 SparseDriveHead 对象
        feature_maps: Union[torch.Tensor, List], # feature_maps 是 SparseDrive 主干网络提取到的图像特征图列表 list，包含 4 个元素，每个元素是对应 FPN level 的特征图，形状分别是 [B, N=6(个相机), 256, 64, 176]、[B, N=6, 256, 32, 88]、[B, N=6, 256, 16, 44]、[B, N=6, 256, 8, 22]
        metas: dict,                             # metas 是数据字典，里面通常包含图像元信息、相机内参外参、ego 状态、标注信息等
    ):
        # (1) 获取检测头的原始输出
        if self.task_config['with_det']:                    # 若任务配置中启用了检测任务
            det_output = self.det_head(feature_maps, metas) # 调用检测头进行前向传播，输入是图像特征 feature_maps 和元信息 metas，输出 det_output 是检测头的原始输出 # 调用 /home/lzw/SparseDrive/projects/mmdet3d_plugin/models/detection3d/detection3d_head_lzw.py 的 Sparse4DHead 类的 forward()
        else:                                               # 若任务配置中没有启用检测任务
            det_output = None                               # 检测输出设为 None

        # (2) 获取地图头的原始输出
        if self.task_config['with_map']:                    # 若任务配置中启用了地图任务
            map_output = self.map_head(feature_maps, metas) # 调用地图头进行前向传播，输入是图像特征 feature_maps 和元信息 metas，输出 map_output 是地图头的原始输出
        else:                                               # 若任务配置中没有启用地图任务
            map_output = None                               # 地图输出设为 None

        # (3) 获取预测和规划头的原始输出
        if self.task_config['with_motion_plan']:                    # 若任务配置中启用了运动预测和规划任务
            motion_output, planning_output = self.motion_plan_head( # 调用 motion_plan_head 进行运动预测和规划，这里输出两个结果：motion_output 是周围 agent 的未来轨迹预测，planning_output 是 ego vehicle 自车未来规划轨迹
                det_output,                                         # 检测头输出。motion_plan_head 需要知道场景中目标的位置、类别、instance 信息等
                map_output,                                         # 地图头输出。motion_plan_head 需要结合车道线、边界、道路结构等地图信息做规划
                feature_maps,                                       # 图像特征，用于补充场景上下文信息
                metas,                                              # 元信息字典，可能包含相机参数、ego 状态、时间戳、标注等
                self.det_head.anchor_encoder,                       # 检测头中的 anchor_encoder，用于把 anchor 或 instance 表示编码成网络可用的特征
                self.det_head.instance_bank.mask,                   # 检测头 instance_bank 中的 mask【instance_bank 通常用于维护 sparse instance/query，而 mask 通常表示哪些 instance/query 是有效的】
                self.det_head.instance_bank.anchor_handler,         # 检测头 instance_bank 中的 anchor_handler。anchor_handler 通常负责 anchor 的生成、变换、更新或解码
            )
        else:                                           # 若任务配置中没有启用运动预测和规划任务
            motion_output, planning_output = None, None # 运动预测输出和规划输出都设为 None

        # (4) 返回四个子任务的原始输出：检测输出、地图输出、agent运动预测输出、ego自车规划输出
        '''
            核心字段及其形状：
                det_output = {                                   # 检测任务头的一个 Query 表示一个候选动态目标
                    "classification": 6 × [B, 900, 10],          # 6个 refine 层的每个检测框 Query 的 10 类类别 logits
                    "prediction":     6 × [B, 900, 11],          # 6个 refine 层的每个检测框 Query 的 11 维 3D 框状态
                    "quality":        6 × ([B, 900, 2] or None), # 6个 refine 层的每个检测框 Query 的 2 个质量预测值 centerness 和 yawness
                    "instance_feature":   [B, 900, 256],         # 最终 900 个检测框的语义特征
                    "anchor_embed":       [B, 900, 256],         # 最终 900 个检测框的几何位置编码
                    "instance_id":        [B, 900],              # 每个 Query 的跨帧跟踪 ID

                    # ---------- DN：仅训练阶段且开启 DN 时存在 ----------
                    # DN(Denoising)：由 GT 3D box 加噪得到额外 DN Query，
                    # 与普通 900 个 Query 共用 Decoder，训练其恢复原始 GT；推理和后处理阶段不使用 DN。
                    # N_dn 为当前 batch 的 DN Query 数，开启正/负 DN 时通常 N_dn = 2 × num_dn_groups × max_dn_gt。
                    "dn_classification": 6 × [B, N_dn, 10],      # 6个 refine 层的 DN Query 类别 logits
                    "dn_prediction":     6 × [B, N_dn, 11],      # 6个 refine 层的 DN Query 3D 框预测
                    "dn_cls_target":         [B, N_dn],          # 普通 DN 分类目标
                    "dn_reg_target":         [B, N_dn, 11],      # 普通 DN 回归目标，即需要恢复的 GT 3D box
                    "dn_valid_mask":         [B, N_dn],          # DN slot 是否有效，过滤 padding / 无效 DN

                    # ---------- Temporal DN：仅开启时序 DN 时存在 ----------
                    "temp_dn_cls_target":    [B, N_dn],          # Temporal Decoder 使用的 DN 分类目标
                    "temp_dn_reg_target":    [B, N_dn, 11],      # Temporal Decoder 使用的当前帧同一 GT 的回归目标
                    "temp_dn_valid_mask":    [B, N_dn],          # Temporal DN slot 是否有效
                    "dn_id_target":          [B, N_dn],          # GT instance ID，用于跨帧匹配同一个 GT
                }

                map_output = {                           # 地图任务头的一个 Query 表示一条候选地图 polyline
                    "classification": 6 × [B, 100, 3],   # 6个 refine 层的每个地图 polyline Query 的 3 类类别 logits
                    "prediction":     6 × [B, 100, 40],  # 6个 refine 层的每个地图 polyline Query 的 20 个点的 (x,y) 二维坐标预测
                    "quality":        6 × None,          # 地图头通常不启用质量分支
                    "instance_feature":   [B, 100, 256], # 最终 100 条地图 polyline 的语义特征
                    "anchor_embed":       [B, 100, 256], # 最终 100 条地图 polyline 的几何位置编码
                    # 没有 instance_id
                }

                motion_output = {                             # 运动预测任务头的一个 Query 表示基于每个检测 Query 预测多条未来轨迹
                    "classification": 1 × [B, 900, 6],        # 1个 refine 层的每个检测 Query 的 6 个候选轨迹模态的分类 logits
                    "prediction":     1 × [B, 900, 6, 12, 2], # 1个 refine 层的每个检测 Query 的 6 个候选轨迹模态，每个模态预测未来 12 个时间步的 (△x,△y) 二维坐标轨迹增量
                    "period":             [B, 900],           # 每个实例连续有效的时序长度
                    "anchor_queue":       q × [B, 900, 11],   # 对齐到当前帧的历史 Agent Anchor
                }
                
                planning_output = {                         # 规划任务头的一个 Query 表示基于唯一 Ego Query 预测多指令、多模态规划路径
                    "classification": 1 × [B, 1, 18],       # 1个 refine 层的每个 Ego Query 的 18 个候选规划指令 logits
                    "prediction":     1 × [B, 1, 18, 6, 2], # 1个 refine 层的每个 Ego Query 的 18 个候选规划指令，每个指令预测未来 6 个时间步的 (△x,△y) 二维坐标轨迹增量
                    "status":         1 × [B, 1, 10],       # 1个 refine 层的每个 Ego Query 的 10 个状态变量
                    "period":             [B, 1],           # 每个实例连续有效的时序长度
                    "anchor_queue":       q × [B, 1, 11],   # 对齐到当前帧的历史 Ego Anchor
                }
        '''
        return det_output, map_output, motion_output, planning_output

    # 4. 计算训练阶段的所有任务的损失，返回损失字典：
    def loss(self, model_outs, data): # 形参 model_outs 是 forward 返回的四元组；形参 data 是训练数据字典，data 里面包含 GT 标注和元信息
        # (1) 解包模型输出
        det_output, map_output, motion_output, planning_output = model_outs
        '''
            det_output：检测头输出
            map_output：地图头输出
            motion_output：运动预测原始输出
            planning_output：规划原始输出       
        '''

        # (2) 创建一个空字典，用于汇总所有任务的 loss
        losses = dict()

        # (3) 若任务配置中启用了检测任务
        if self.task_config['with_det']:
            # a. 调用检测头自己的 loss 函数
            # loss_det 通常是一个字典，例如：
            # {
            #   "loss_cls": ...,
            #   "loss_box": ...,
            #   ...
            # }
            loss_det = self.det_head.loss(det_output, data)

            # b. 把检测任务 loss_det 合并到总 losses 字典中
            losses.update(loss_det)
        
        # (4) 若任务配置中启用了地图任务
        if self.task_config['with_map']:
            # a. 调用地图头自己的 loss 函数，返回值 loss_map 通常包括地图元素分类 loss、点回归 loss、方向 loss 等
            loss_map = self.map_head.loss(map_output, data)

            # b. 把地图任务 loss_map 合并到总 losses 字典中
            losses.update(loss_map)

        # (5) 若任务配置中启用了运动预测和规划任务
        if self.task_config['with_motion_plan']:
            # a. 构造 motion_plan_head 计算 loss 时需要的缓存信息
            motion_loss_cache = dict(
                # indices 来自检测头 sampler
                # sampler.indices 通常表示检测任务中的匹配结果，例如预测 instance 和 GT instance 之间的匹配关系
                # 运动预测任务可能需要复用检测任务的匹配结果
                indices=self.det_head.sampler.indices, 
            )

            # b. 调用 motion_plan_head 的 loss 函数
            loss_motion = self.motion_plan_head.loss(
                motion_output,    # 运动预测输出
                planning_output,  # 自车规划输出
                data,             # 训练数据字典
                motion_loss_cache # 额外缓存信息，这里主要传入检测头的匹配 indices
            )

            # c. 把运动预测和规划任务 loss_motion 合并到总 losses 字典中
            losses.update(loss_motion)
        
        # (6) 返回总 loss 字典【外层 SparseDrive.forward_train() 会把这个字典交给训练器做反向传播】
        return losses # 返回一个包含所有任务 loss 的字典，例如 {"loss_cls": ..., "loss_box": ..., "loss_map_cls": ..., "loss_motion": ..., ...} 

    # 5. post_process 用于测试阶段的后处理，它把各个子 head 的原始输出转换成最终可评估、可保存、可可视化的结果：
    def post_process(self, model_outs, data):
        # (1) 解包模型输出
        det_output, map_output, motion_output, planning_output = model_outs
        '''
            det_output：检测头输出
            map_output：地图头输出
            motion_output：运动预测原始输出
            planning_output：规划原始输出       
        '''

        # (2) 若任务配置中启用了检测任务，则调用检测头后处理函数获取检测结果
        if self.task_config['with_det']:
            # 调用检测头后处理函数
            # det_result 一般是长度为 batch_size 的 list，每个元素是一个样本的检测结果
            det_result = self.det_head.post_process(det_output)

            # 根据检测结果长度获得 batch_size
            batch_size = len(det_result)
        
        # (3) 若任务配置中启用了地图任务，则调用地图头后处理函数获取地图结果
        if self.task_config['with_map']:
            # 调用地图头后处理函数
            # map_result 一般也是长度为 batch_size 的 list，每个元素是一个样本的地图元素检测结果
            map_result= self.map_head.post_process(map_output)

            # 根据地图结果长度获得 batch_size
            # 若同时启用了 det 和 map，这里会覆盖前面由 det_result 得到的 batch_size
            # 正常情况下二者 batch_size 应该一致，所以问题不大
            batch_size = len(map_result)

        # (4) 若任务配置中启用了运动预测和规划任务，则调用 motion_plan_head 的后处理函数获取周围 agent 的运动预测结果和自车规划结果
        if self.task_config['with_motion_plan']:
            # 调用 motion_plan_head 的后处理函数
            # motion_result：周围 agent 的运动预测结果
            # planning_result：自车规划结果
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output,      # 检测输出。运动预测和规划后处理需要检测结果中的目标信息
                motion_output,   # 运动预测输出
                planning_output, # 规划输出
                data,            # 原始数据字典，可能包含坐标变换、场景 token、ego 状态等
            )

        # (5) 初始化最终结果列表
        # 目标是构造 batch_size 个 dict，每个 dict 存一个样本的最终结果
        # 注意：这里 [dict()] * batch_size 会让所有元素引用同一个 dict，这是一个潜在问题，更安全写法是：results = [dict() for _ in range(batch_size)]
        results = [dict()] * batch_size

        # (6) 遍历 batch 中每一个样本，根据任务配置把各个子任务的结果合并到 results[i] 中：
        for i in range(batch_size):
            if self.task_config['with_det']:          # 若任务配置中启用了检测任务
                results[i].update(det_result[i])      # 把第 i 个样本的检测结果合并到 results[i]
            if self.task_config['with_map']:          # 若任务配置中启用了地图任务
                results[i].update(map_result[i])      # 把第 i 个样本的地图结果合并到 results[i]
            if self.task_config['with_motion_plan']:  # 若任务配置中启用了运动预测和规划任务
                results[i].update(motion_result[i])   # 把第 i 个样本的运动预测结果合并到 results[i]
                results[i].update(planning_result[i]) # 把第 i 个样本的自车规划结果合并到 results[i]

        # (7) 返回最终结果【外层 SparseDrive.simple_test() 会把每个 result 包装成 {"img_bbox": result}】
        return results # results 是长度为 batch_size 的 list，每个元素是一个 dict，包含该样本的所有任务结果，例如 {"det": ..., "map": ..., "motion": ..., "planning": ...}

