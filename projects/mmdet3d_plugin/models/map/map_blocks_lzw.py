# =============================================================================
# map_blocks.py（逐行详细注释版）
# =============================================================================
# 1. 文件作用
#    (1) 本文件定义 SparseDrive map head 中与“地图折线 anchor / map polyline”相关的三个核心模块。
#    (2) SparsePoint3DEncoder：把一条 map polyline 的 20 个二维点坐标编码成 256 维位置特征。
#    (3) SparsePoint3DRefinementModule：根据 instance feature 和 anchor embedding，预测 polyline 坐标残差，并可选预测 map 类别。
#    (4) SparsePoint3DKeyPointsGenerator：把二维 map polyline anchor 变成 3D key points，用于后续多视角图像特征采样。
#
# 2. Map anchor 的基本语义
#    (1) 一个 map instance 通常表示一条 vectorized local map line，例如 lane divider / ped crossing / boundary。
#    (2) 原始 map anchor 常见形状为 [N_map, num_sample, 2]，例如 [100, 20, 2]。
#    (3) 进入 InstanceBank 后常被 flatten 成 [N_map, 40]，其中 40 = 20 * (x, y)。
#    (4) 本文件中的 Encoder / Refine / KeyPointsGenerator 都会把最后 40 维重新理解成 20 个二维点。
#
# 3. 与 detection 分支的差异
#    (1) detection anchor 表示 3D box state，例如 xyz、wlh、yaw、velocity 等。
#    (2) map anchor 表示二维折线点序列，不包含 box size / yaw / velocity。
#    (3) map 的时序投影主要是坐标系变换，不做 detection 那种速度外推。
# =============================================================================

from typing import Optional, List, Tuple  # 导入类型标注工具；当前文件实际只使用 Tuple，Optional/List 属于冗余导入但保留原源码一致性
import torch                     # 导入 PyTorch 主库，用于 Tensor、矩阵乘法、拼接、ones_like 等张量操作
import torch.nn as nn            # 导入 PyTorch 神经网络模块，用于 nn.Sequential 等网络容器
import torch.nn.functional as F  # 导入函数式接口；当前文件未实际使用，属于原源码保留的冗余导入
import numpy as np               # 导入 NumPy，用于把 fix_height 转成 np.array，便于记录固定高度列表

from mmcv.cnn import Linear, Scale, bias_init_with_prob     # Linear 是 MMCV 线性层封装；Scale 是可学习缩放层；bias_init_with_prob 用于按先验概率初始化分类 bias
from mmcv.runner.base_module import Sequential, BaseModule  # BaseModule 是 MMCV 的 nn.Module 封装；Sequential 当前未实际使用，保留原源码一致性
from mmcv.cnn import xavier_init                            # 导入 Xavier 初始化函数，用于初始化 learnable_fc
from mmcv.cnn.bricks.registry import ( # 从 MMCV registry 中导入注册表，SparseDrive 依赖配置字典动态构建模块
    PLUGIN_LAYERS,                     # 插件层注册表：RefinementModule 和 KeyPointsGenerator 会注册到这里
    POSITIONAL_ENCODING,               # 位置编码注册表：SparsePoint3DEncoder 会注册到这里
)
from ..blocks import linear_relu_ln # 导入项目自定义 MLP 构造函数：通常返回 Linear + ReLU + LayerNorm 的层列表


# 一、定义 map polyline anchor 的位置编码器：将锚点的坐标信息编码为神经网络使用的高维位置特征向量，即输入二维点坐标、输出 embed_dims 维 anchor embedding
@POSITIONAL_ENCODING.register_module() # 把 SparsePoint3DEncoder 注册到 POSITIONAL_ENCODING，配置中 type='SparsePoint3DEncoder' 时可自动构建
class SparsePoint3DEncoder(BaseModule): 
    # 1. 初始化位置编码器
    def __init__(  
        self,                  # 当前模块对象
        embed_dims: int = 256, # 输出特征维度，默认 256
        num_sample: int = 20,  # 每条 map line 的采样点数量，默认 20
        coords_dim: int = 2,   # 每个点的坐标维度，默认 2，对应 x/y
    ):
        super(SparsePoint3DEncoder, self).__init__()  # 调用 BaseModule 父类初始化，保证 MMCV init_cfg 等机制可用

        # (1) 保存参数
        self.embed_dims = embed_dims              # 保存 embedding 维度，后续构建 MLP 时使用
        self.input_dims = num_sample * coords_dim # 计算输入维度：采样点数量 × 每个点的坐标维度，即单个锚点的坐标总长度，例如 20 个点 * 2 维坐标 = 40

        # (2) 定义内部函数：根据输入维度构建一个单层嵌入网络 = 线性层 + ReLU + LayerNorm
        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims)) # 构建 Linear/ReLU/LN 组合，将 input_dims 映射到 embed_dims

        # (3) 创建 anchor 坐标位置编码网络：输入坐标展平向量 [B, N, 40]，输出高维位置嵌入 [B, N, 256]
        self.pos_fc = embedding_layer(self.input_dims) 

    # 2. 前向传播：对输入锚点做位置编码，返回位置特征
    def forward(self, anchor: torch.Tensor):
        """
            输入: anchor  : 输入锚点坐标，形状 [batch_size, num_anchor, num_sample*coords_dim] = [B, N, 2*num_sample]，例如 [B, 100, 40]
            返回: pos_feat: 位置特征，形状 [batch_size, num_anchor, embed_dims] = [B, N, embed_dims]，例如 [B, 100, 256]
        """
        pos_feat = self.pos_fc(anchor) # 用 MLP 编码 anchor 坐标得到高维位置特征：本质是把 40 维点序列映射成 256 维几何 embedding
        return pos_feat                # 返回位置特征【供后续 attention / refinement 作为几何条件使用】
    

# 二、定义 map polyline 的 refine 模块：更新 anchor 精修坐标，并可选输出类别 logits
@PLUGIN_LAYERS.register_module() # 把 SparsePoint3DRefinementModule 注册到 PLUGIN_LAYERS，配置中可通过 type 动态构建
class SparsePoint3DRefinementModule(BaseModule):
    # 1. 初始化坐标 refine 分支和分类分支
    def __init__(  
        self,                         # 当前模块对象
        embed_dims: int = 256,        # 特征维度，默认 256
        num_sample: int = 20,         # 每条 map line 的采样点数量，默认 20
        coords_dim: int = 2,          # 每个点的坐标维度，默认 2，对应 x/y
        num_cls: int = 3,             # map 类别数，默认 3
        with_cls_branch: bool = True, # 是否使用分类分支，默认使用
    ):
        super(SparsePoint3DRefinementModule, self).__init__() # 调用 BaseModule 父类初始化

        # (1) 保存参数：
        self.embed_dims = embed_dims              # 保存特征维度，默认 256
        self.num_sample = num_sample              # 保存每条 map line 的采样点数量，默认 20
        self.output_dim = num_sample * coords_dim # 计算输出坐标维度 = 采样点数量 × 坐标维度，即精修后的坐标总长度，如 20 * 2 = 40
        self.num_cls = num_cls                    # 保存 map 类别数，默认 3

        # (2) 构建坐标 refine 分支，用于预测 polyline residual：2层“线性+ReLU+LN” + 输出线性层 + 可学习缩放层
        self.layers = nn.Sequential(  
            *linear_relu_ln(embed_dims, 2, 2),        # 先堆叠2层 “线性+ReLU+LN” 的特征提取，输入输出通道保持 embed_dims 不变
            Linear(self.embed_dims, self.output_dim), # 最终线性层，将特征映射为坐标偏移量，例如 256 -> 40
            Scale([1.0] * self.output_dim),           # 可学习的缩放层：对每个坐标维度加一个单独的可学习缩放，稳定训练，初始缩放为 1.0
        )

        # (3) 构建分类分支，用于输出每个 map query 的类别 logits：1层“线性+ReLU+LN” + 分类输出层
        self.with_cls_branch = with_cls_branch # 记录是否使用分类分支，默认使用
        if with_cls_branch:                    # 如果配置要求输出类别，则创建分类分支
            self.cls_layers = nn.Sequential(   # 构建分类分支，用于输出每个 map query 的类别 logits
                *linear_relu_ln(embed_dims, 1, 2),     # 分类前的轻量 MLP 特征变换
                Linear(self.embed_dims, self.num_cls), # 输出 num_cls=3 个类别 logits，例如 [B, N, 3]
            )

    # 2. 初始化分类分支权重：对分类分支的 bias 做偏置初始化，缓解训练初期正负样本不均衡
    def init_weight(self):  
        """
            初始化权重：
                1. 如果存在分类分支，则把最后一层分类 bias 初始化为低正样本先验。
                2. bias_init_with_prob(0.01) 常用于 FocalLoss 场景，让初始正类概率约为 0.01。
        """
        # 只有存在分类分支时才初始化分类 bias
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)                  # 根据正样本先验概率 0.01 计算分类 bias 初值（Focal Loss常用初始化）
            nn.init.constant_(self.cls_layers[-1].bias, bias_init) # 将分类最后一层 bias 全部设为该初值，缓解训练初期正负样本不平衡

    # 3. 前向传播：根据 instance feature 和 anchor embedding 更新 map anchor【更新 map anchor 的精修坐标和分类】
    def forward(  
        self,                              # 当前模块对象
        instance_feature: torch.Tensor,    # 当前 map query 的实例语义特征，[B, N, C]，N 是 num_anchor
        anchor: torch.Tensor,              # 当前 map anchor 坐标，[B, N, 2*num_sample=40]
        anchor_embed: torch.Tensor,        # 当前 map anchor 的位置编码，[B, N, C]
        time_interval: torch.Tensor = 1.0, # 时间间隔：map 源码未使用，主要是为了对齐 detection 接口，因此预留该参数
        return_cls=True,                   # 是否返回分类结果 logits：训练/推理通常需要分类输出
    ):
        # (1) 使用坐标 refine 分支得到 refine 后的 anchor 坐标
        output = self.layers(instance_feature + anchor_embed) # 将实例语义特征和几何位置特征相加【即融合语义与位置信息】，再经过坐标 refine 分支来预测 40 维坐标残差
        output = output + anchor                              # 残差连接 residual learning：精修后的坐标 = 原 anchor 坐标 + 网络预测的坐标偏移量
 
        # (2) 使用分类分支进行分类得到分类 logits
        if return_cls: # 如果要求返回分类结果
            assert self.with_cls_branch, "Without classification layers !!!" # 若没有创建分类分支但要求分类输出，则直接报错
            cls = self.cls_layers(instance_feature)                          # 用 instance_feature 做分类 ## NOTE anchor embed? 【源码注释 “是否应该加入anchor embed？” 提示这里没有融合 anchor_embed，可能是作者留下的思考点】
        else:          # 如果不需要分类输出
            cls = None # 分类结果置空

        # (3) map 源码未使用质量估计分支，主要是为了对齐 detection 接口，因此预留该参数
        qt = None

        # (4) 返回 refine 后坐标、分类 logits、质量估计占位值 
        """ 
            output: 精修后的锚点坐标，形状与输入 anchor 坐标一致 [B, N=num_anchor, 2*num_sample=40]
            cls   : 分类 logits，形状[B, N=num_anchor, num_cls=3]，不要求返回时为None
            qt    : map 源码未使用质量估计分支，主要是为了对齐 detection 接口，因此预留该参数
        """
        return output, cls, qt


# 三、定义 map polyline 的 3D key points 生成器，用于图像特征采样
@PLUGIN_LAYERS.register_module() # 把 SparsePoint3DKeyPointsGenerator 注册到 PLUGIN_LAYERS，配置中可作为 anchor_handler / key_points_generator 构建
class SparsePoint3DKeyPointsGenerator(BaseModule):
    # 1. 初始化 key points 生成器 learnable_fc
    def __init__(  
        self,                       # 当前模块对象
        embed_dims: int = 256,      # 输入 instance_feature 的特征嵌入维度，默认 256
        num_sample: int = 20,       # 每条 map polyline 的基础采样点数，默认 20
        num_learnable_pts: int = 0, # 每个基础采样点额外生成的可学习偏移点数量，默认 0
        fix_height: Tuple = (0,),   # 固定高度值列表，生成多高度层的关键点，默认只使用 z=0 一层
        ground_height: int = 0,     # 地面基准高度，默认 0
    ):
        super(SparsePoint3DKeyPointsGenerator, self).__init__() # 调用 BaseModule 父类初始化
        self.embed_dims = embed_dims                                    # 保存输入 instance_feature 的特征维度
        self.num_sample = num_sample                                    # 保存基础采样点数量
        self.num_learnable_pts = num_learnable_pts                      # 保存每个基础采样点额外生成的可学习偏移点数量
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts # 计算每条 line 最终用于采样的总关键点数量 = 采样点数 × 高度层数 × 可学习点数，不含 xyz 维度
        if self.num_learnable_pts > 0:                                    # 只有需要生成可学习偏移点时【即如果可学习点数量大于0】，才创建偏移预测层
            self.learnable_fc = Linear(self.embed_dims, self.num_pts * 2) # 定义预测所有偏移点的二维 offset 的线性层：将实例特征映射为 xy 平面的坐标偏移量，每个点有 dx,dy 两个量
        self.fix_height = np.array(fix_height) # 保存固定高度列表为 numpy array，forward 中会转成 tensor
        self.ground_height = ground_height     # 保存地面基准高度，会作为 z 坐标的初始值

    # 2. 对可学习偏移的线性层 learnable_fc 做 xavier 初始化【其 bias 初始化为 0，表示初始偏移接近 0，不会一开始就大幅偏离原始 anchor 点】
    def init_weight(self): 
        if self.num_learnable_pts > 0:                                       # 只有存在 learnable_fc 时才初始化
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0) # 用均匀分布的 Xavier 初始化 offset 预测层，bias 初始化为 0【bias 设为 0，表示初始偏移接近 0，不会一开始就大幅偏离原始 anchor 点】

    # 3. 前向传播：从二维 anchor 生成三维关键点 key points，可选投影到历史帧【称为时序投影】
    def forward( 
        self,                  # 当前模块对象
        anchor,                # 锚点坐标 map polyline anchor，[B, N=100, 2*num_sample=40] = [B, 100, 40]
        instance_feature=None, # 实例特征 map query feature，[B, N, C]，用于预测可学习偏移 offset
        T_cur2temp_list=None,  # 当前帧到历史帧的变换矩阵列表，每个元素是一个 [B, 4, 4] 的变换矩阵
        cur_timestamp=None,    # 当前帧时间戳【当前实现中仅用于判断是否进行 temporal 投影，并未直接参与计算、不做显式时间外推】
        temp_timestamps=None,  # 历史帧时间戳列表【当前实现中仅用于循环数量判断，并未用于速度外推】
    ):
        # 1. 输入校验与基础维度重塑
        assert self.num_learnable_pts > 0, 'No learnable pts' # 断言必须有可学习点 learnable points，否则通过 learnable_fc 生成采样点
        bs, num_anchor, _ = anchor.shape                              # 读取 batch_size、anchor 数量、坐标维度=40
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1) # 把展平的锚点坐标从 [B, N, 40] 还原成 [B, N, 20, 2]，即分离出每条线的 20 个二维点 (x,y)


        # 2. 生成可学习坐标偏移，更新xy平面位置
        # (1) 用实例特征 instance_feature 预测每个基础点周围的可学习二维偏移，再重塑为与关键点匹配的6维结构：[B, 总锚点数=40, 采样点数S, 高度层数H, 可学习点数L, 2]
        offset = (
            self.learnable_fc(instance_feature) # 从 instance feature 预测所有 offset，shape [B, N=100, num_pts*2=40]
            .reshape(bs, num_anchor, self.num_sample, len(self.fix_height), self.num_learnable_pts, 2)  # 还原为 [B, N, S, H, L, 2]，其中 S=num_sample，H=固定高度数量，L=每个点的可学习偏移点数量
        )

        # (2) 将偏移量通过广播加到基础锚点上，得到更新后的xy坐标
        key_points = offset + key_points[..., None, None, :] # 将 offset 加到原始二维点上 # 通过 None 扩展维度实现广播：[B, N, S, 2] 扩展成 [B, N, S, H, L, 2]


        # 3. 补全Z轴维度，生成多高度层3D关键点
        # (1) 拼接z维度：所有点先统一填充地面基准高度 z=0，最终得到的 key_points shape [B, N, S, H, L, 3(x,y,z=0)]
        key_points = torch.cat(  # 
            [
                key_points,                                                                       # 已经带 offset 的二维点(x,y)，[B, N, S, H, L, 2(x,y)]
                key_points.new_full(key_points.shape[:-1] + (1,), fill_value=self.ground_height), # 生成与 xy 形状匹配的全 ground_height=0 张量，作为 z 维度，[B, N, S, H, L, 1(z=0)]
            ],
            dim=-1, # 在最后一维的坐标维度拼接，将二维点 [x,y] 扩展为三维点 [x,y,z]，最终得到的 key_points shape [B, N, S, H, L, 3(x,y,z=0)]
        )

        # (2) 构建高度偏移张量 height_offset：xy偏移为0，仅z轴叠加各固定高度值，[H, 3(0,0,z_offset)]，即每个固定高度对应 [0, 0, z_offset]
        fix_height = key_points.new_tensor(self.fix_height)                     # 将 numpy 的固定高度列表 fix_height 转成与 key_points 同 device 同 dtype 的 tensor
        height_offset = key_points.new_zeros([len(fix_height), 2])              # 创建 [H, 2] 的零偏移，即 xy 偏移为0，用于 xy 两维不变，仅 z 轴有偏移
        height_offset = torch.cat([height_offset, fix_height[:, None]], dim=-1) # 在最后一维的坐标维度拼接高度值，最终拼成 [H, 3] 的高度偏移张量，即每个固定高度对应 [0, 0, z_offset]

        # (3) 将高度偏移广播到所有关键点上【广播到所有batch、锚点、采样点、可学习点】，得到多高度层的3D关键点
        key_points = key_points + height_offset[None, None, None, :, None] # 给第 H 维加上不同固定高度，x/y 不变，z 增加 fix_height


        # 4. 展平维度，输出规整的关键点序列
        key_points = key_points.flatten(2, 4) # 将 “采样点S、高度层H、可学习点L” 三个维度展平为一维，得到最终关键点序列 [B, N, S*H*L, 3]


        # 5. 时序投影分支
        # (0) 若缺少任一时序参数，则直接返回当前帧关键点
        if (                             # 判断是否缺少 temporal 投影所需信息
            cur_timestamp is None        # 当前帧时间戳为空时，不做 temporal 投影
            or temp_timestamps is None   # 历史帧时间戳列表为空时，不做 temporal 投影
            or T_cur2temp_list is None   # 当前到历史的坐标变换矩阵为空时，不做 temporal 投影
            or len(temp_timestamps) == 0 # 历史帧数量为 0 时，不做 temporal 投影
        ):
            return key_points # 只返回当前帧关键点 key points，[B, N, K, 3]

        # (1) 遍历每个历史帧，逐帧做坐标投影
        temp_key_points_list = []                    # 创建历史帧关键点列表，用于保存投影到每个历史帧坐标系下的 key points
        for i, t_time in enumerate(temp_timestamps): # 遍历每一个历史帧时间戳【t_time 当前没有直接用于计算】
            # (2) 当前关键点作为投影基准，获取当前帧到该历史帧的变换矩阵
            temp_key_points = key_points                               # 当前关键点作为投影基准
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype) # 获取当前帧到该历史帧的变换矩阵，转为与关键点 key_points 相同的数据类型 dtype

            # (3) 齐次坐标变换：将3D点补1转为齐次坐标，左乘变换矩阵完成投影，输出 [B,N,K,3,1]
            temp_key_points = (
                T_cur2temp[:, None, None, :3] # 取变换矩阵前三行，扩展成 [B, 1, 1, 3, 4]，用于输出 x/y/z
                @ torch.cat(                  # 构造齐次坐标 (x,y,z,1)
                    [
                        temp_key_points,                           # 原三维点 (x,y,z)
                        torch.ones_like(temp_key_points[..., :1]), # 补齐一维 1
                    ],
                    dim=-1, # 在最后一维拼接，形成齐次坐标，使 (x,y,z) -> (x,y,z,1)
                ).unsqueeze(-1) # 增加矩阵乘法需要的最后一维，shape [B,N,K,4,1]
            ) # 矩阵乘法输出 shape [B,N,K,3,1]

            # (4) 去掉列向量维度，得到变换后的3D坐标
            temp_key_points = temp_key_points.squeeze(-1) # 去掉最后的单维的列向量维度，得到变换后的3D坐标 [B,N,K,3]
            temp_key_points_list.append(temp_key_points)  # 将该历史帧的关键点加入列表：保存第 i 个历史帧坐标系下的 key points
        
        # 6. 返回当前帧关键点 + 所有历史帧投影关键点列表
        """
            key_points          : 当前帧的3D关键点，形状[B, num_anchor, num_pts=num_sample*len(fix_height)*num_learnable_pts, 3]（展平后）
            temp_key_points_list: 各历史帧的3D关键点列表（仅当时序参数齐全时返回）
        """
        return key_points, temp_key_points_list

    # 4. 锚点投影方法：将源坐标系下的2D锚点投影到多个目标坐标系（仅xy平面变换），主要给 InstanceBank 的 temporal cache 对齐使用
    # @staticmethod # 原源码注释掉的 staticmethod；当前方法仍使用 self.num_sample，因此保留实例方法是合理的
    def anchor_projection(
        self,                # 当前模块对象，需要访问 self.num_sample
        anchor,              # 源坐标系下的 map anchor，[B, N, 2*num_sample=40]
        T_src2dst_list,      # 源坐标系到目标坐标系的变换矩阵列表，每个元素是一个 [B, 4, 4] 的变换矩阵
        src_timestamp=None,  # 源时间戳【map anchor 投影当前不使用速度，所以该参数未使用】
        dst_timestamps=None, # 目标时间戳列表【当前实现未使用】
        time_intervals=None, # 时间间隔【map 不做速度外推，因此当前实现未使用】
    ):

        # 1. 创建列表，保存每个目标坐标系下的投影 anchor
        dst_anchors = [] 

        # 2. 遍历每个源到目标的变换矩阵，逐次做变换
        for i in range(len(T_src2dst_list)):
            # (1) 输入预处理：形状规整
            dst_anchor = anchor.clone()       # clone 一份 anchor，避免直接修改输入张量
            bs, num_anchor, _ = anchor.shape  # 读取 batch_size、anchor 数量和坐标维度
            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(1, 2) # 将锚点重塑为 [B,N,40] -> [B,N,20,2] -> [B,N*20,2]，以便批量做坐标变换
            T_src2dst = torch.unsqueeze(T_src2dst_list[i].to(dtype=anchor.dtype), dim=1)       # 给第 i 个变换矩阵增加点维度，得到 [B,1,4,4]，便于广播到所有点

            # (2) 执行2D仿射变换：先做旋转缩放（即矩阵乘法），再加平移向量，得到 dst_anchor 仍为 [B,N*20,2]
            dst_anchor = (  # 对所有二维点执行刚体坐标变换
                torch.matmul(                                     # (a) 取变换矩阵的前2行前2列，与坐标做矩阵乘法：R_2x2 * (x,y)
                    T_src2dst[..., :2, :2], dst_anchor[..., None] # 取 xy 平面的 2x2 旋转矩阵，并把点变成列向量 [B,N*20,2,1]
                ).squeeze(dim=-1)                                 # 去掉列向量最后一维，得到旋转后的 [B,N*20,2]
                + T_src2dst[..., :2, 3]                           # (b) 加上 xy 平移向量（前2行第4列），完成 source -> destination 坐标变换
            )

            # (3) 恢复原始形状 [B,N,40]，存入结果列表
            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(2, 3) # [B,N*20,2] -> [B,N,20,2] -> [B,N,40]
            dst_anchors.append(dst_anchor)                                                     # 将第 i 个目标坐标系下的 anchor 加入结果列表
        
        # 3. 返回所有目标坐标系下的 anchor 列表，每个元素对应一个目标坐标系下的 anchor，单个 shape [B, N, 40]
        return dst_anchors

