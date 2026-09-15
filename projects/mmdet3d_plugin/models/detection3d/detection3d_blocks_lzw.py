
import torch          # 导入 PyTorch。用于张量计算、矩阵乘法、拼接、归一化等
import torch.nn as nn # 导入 PyTorch 神经网络模块。用于 nn.Module、nn.Sequential、nn.Parameter 等
import numpy as np    # 导入 numpy。当前文件中 np 实际没有被使用，属于冗余导入
from mmcv.cnn import Linear, Scale, bias_init_with_prob    # 从 MMCV 中导入常用层和初始化工具：Linear：MMCV 封装的线性层；Scale：可学习缩放层，用于对回归输出做逐维缩放；bias_init_with_prob：根据先验概率初始化分类 bias
from mmcv.runner.base_module import Sequential, BaseModule # 从 MMCV 中导入 Sequential 和 BaseModule：Sequential 类似 torch.nn.Sequential；BaseModule 是 MMCV 对 nn.Module 的封装，支持 init_cfg 等机制
from mmcv.cnn import xavier_init                           # 导入 Xavier 初始化函数。这里用于初始化 learnable keypoints 的线性层
from mmcv.cnn.bricks.registry import (                     # 导入 MMCV 注册表
    PLUGIN_LAYERS,       # PLUGIN_LAYERS 用于注册自定义插件层：SparseBox3DRefinementModule 和 SparseBox3DKeyPointsGenerator 会注册到这里
    POSITIONAL_ENCODING, # POSITIONAL_ENCODING 用于注册位置编码模块：SparseBox3DEncoder 会注册到这里
)

# 导入本地模块
from projects.mmdet3d_plugin.core.box3d import * # 导入 box3d.py 中定义的 box 状态索引常量：如 X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX 等，这些常量用于从 anchor 的最后一维中取对应字段
from ..blocks import linear_relu_ln # 从上一层 models/blocks.py 中导入 linear_relu_ln：这是一个快速构建 Linear + ReLU + LayerNorm 堆叠的小工具函数


# __all__ 控制《from detection3d_blocks_lzw import *》时暴露哪些对象
__all__ = [
    "SparseBox3DRefinementModule",   # 3D box refine 模块
    "SparseBox3DKeyPointsGenerator", # 3D box key points 生成器
    "SparseBox3DEncoder",            # 3D box encoder
]


# 一、SparseBox3DEncoder类：对 3D bbox 的位置、尺寸、朝向角、速度分别进行编码和融合
@POSITIONAL_ENCODING.register_module() # 将 SparseBox3DEncoder 注册到 POSITIONAL_ENCODING 注册表。配置文件中 anchor_encoder=dict(type="SparseBox3DEncoder", ...) 就会构建这个类
class SparseBox3DEncoder(BaseModule):
    '''
        SparseBox3DEncoder 的作用：将 3D box anchor 的几何状态编码成神经网络中的 embedding。
        anchor 原始状态可能类似：[x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy, vz]
        编码后输出：[B, num_anchor, embed_dims] 或在 cat 模式下先拼接多个子 embedding，再可选 output_fc 映射。
    '''

    # 1. 初始化
    def __init__(
        self, # self 表示当前模块对象

        # embed_dims 可以是 int，也可以是 list/tuple
        # 若是 int，例如 256，则各个子分支都使用 256 维
        # 若是 list，例如 [128, 32, 32, 64]，则分别控制位置、尺寸、yaw、速度分支维度
        embed_dims,

        # velocity 维度
        # 默认 vel_dims=3，表示从 VX 开始取 3 维速度状态
        vel_dims=3,

        # 融合模式
        # add：各分支 embedding 相加，要求维度一致
        # cat：各分支 embedding 拼接，允许维度不同
        mode="add",

        output_fc=True, # 是否在最后再接一个 output_fc
        in_loops=1,     # linear_relu_ln 里面每轮 Linear+ReLU 的重复次数
        out_loops=2,    # linear_relu_ln 外层循环次数，每轮最后带 LN
    ):
       
        super().__init__() # 调用 BaseModule 初始化

        # (1) 保存参数：
        assert mode in ["add", "cat"] # 检查 mode 只能是 add 或 cat
        self.embed_dims = embed_dims  # 保存 embedding 配置
        self.vel_dims = vel_dims      # 保存 velocity 维度
        self.mode = mode              # 保存融合模式

        # (2) 定义一个内部函数，用于构建某个 box 子状态的 embedding 网络
        def embedding_layer(input_dims, output_dims):
            # 返回一个 nn.Sequential，其内部结构由 linear_relu_ln 生成：Linear -> ReLU -> LayerNorm 等
            return nn.Sequential(*linear_relu_ln(output_dims, in_loops, out_loops, input_dims)) # 这里 * 表示把 list 展开成 Sequential 的多个模块

        # (3) 若 embed_dims 不是 list/tuple，则把单个 int 扩展成长度为 5 的 list
        if not isinstance(embed_dims, (list, tuple)):
            # 把单个 int 扩展成长度为 5 的 list：前4个分别给 pos、size、yaw、vel 使用，最后1个用于 output_fc
            embed_dims = [embed_dims] * 5

        # (4) 定义编码层
        # (4.1) 位置编码分支
        # 输入 x,y,z 三维
        # 输出 embed_dims[0] 维
        self.pos_fc = embedding_layer(3, embed_dims[0])

        # (4.2) 尺寸编码分支
        # 输入 w,l,h 三维
        # 注意这里一般是 log-space 的 W,L,H
        # 输出 embed_dims[1] 维
        self.size_fc = embedding_layer(3, embed_dims[1])

        # (4.3) 朝向编码分支
        # 输入 sin_yaw, cos_yaw 两维
        # 输出 embed_dims[2] 维
        self.yaw_fc = embedding_layer(2, embed_dims[2])

        # (4.4) 速度编码分支
        if vel_dims > 0: # 若 velocity 维度大于 0
            # 速度编码分支
            # 输入 self.vel_dims 维，vx, vy, vz
            # 输出 embed_dims[3] 维
            self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3])

        # (4.5) 输出层映射分支
        if output_fc:                                                        # 若需要最后输出层
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1]) # 则定义 output_fc 用于对融合后的 embedding 再做一次映射，其输入维度和输出维度都是 embed_dims[-1]
        else:                                                                # 若不需要输出层
            self.output_fc = None                                            # 则输出层 output_fc 直接设为 None

    # 2. 前向传播
    def forward(self, box_3d: torch.Tensor):
        ''' 
            前向传播：
                输入 box_3d shape 通常为 [B, num_anchor, box_dim]
                输出 anchor embedding，shape 通常为 [B, num_anchor, C]
        '''

        # (1) 提取 box_3d 的位置、尺寸、yaw并分别编码：
        pos_feat = self.pos_fc(box_3d[..., [X, Y, Z]])          # 提取位置部分 [x, y, z] 并编码
        size_feat = self.size_fc(box_3d[..., [W, L, H]])        # 提取尺寸部分 [w, l, h] 并编码       
        yaw_feat = self.yaw_fc(box_3d[..., [SIN_YAW, COS_YAW]]) # 提取 yaw 的 sin/cos 表示并编码

        # (2) 融合：
        if self.mode == "add":                                          # 若使用 add 模式
            output = pos_feat + size_feat + yaw_feat                    # 则位置、尺寸、yaw embedding 相加（此时要求三个分支输出维度相同）
        elif self.mode == "cat":                                        # 若使用 cat 模式
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1) # 则位置、尺寸、yaw embedding 在最后一维拼接

        # 若使用速度分支
        if self.vel_dims > 0:
            # (3) 提取 box_3d 的速度并编码：
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims]) # VX : VX + self.vel_dims 表示从 VX 开始取 vel_dims 维

            # (4) 融合：
            if self.mode == "add": # add 模式下速度 embedding 相加
                output = output + vel_feat
            elif self.mode == "cat": # cat 模式下速度 embedding 拼接
                output = torch.cat([output, vel_feat], dim=-1)

        # (5) 对最终 box_3d 编码结果作MLP输出层映射（若有 output_fc），并返回
        if self.output_fc is not None:      # 若有 output_fc
            output = self.output_fc(output) # 对融合后的 embedding 再做一次 MLP 映射
        return output                       # 返回 box anchor 的 embedding


# 二、 SparseBox3DRefinementModule 类：用于 3D 检测 decoder 中的 refine 层，它是 SparseDrive Detection decoder 最后的 3D 检测头 / anchor 迭代细化器，它负责把已有证据转化为“这个 3D 框该往哪里移、大小怎么改、朝向怎么改、速度怎么改，以及它究竟是什么类别”
@PLUGIN_LAYERS.register_module() # 将 SparseBox3DRefinementModule 注册到 PLUGIN_LAYERS。配置文件中 refine_layer=dict(type="SparseBox3DRefinementModule", ...) 就会构建这个类
class SparseBox3DRefinementModule(BaseModule):
    '''
        SparseBox3DRefinementModule 是 3D 检测 decoder 中的 refine 层
        它的作用：
            1. 根据 instance_feature + anchor_embed 预测 box 更新量
            2. 把部分状态加到原 anchor 上，实现 iterative refinement
            3. 输出分类 logits
            4. 可选输出 quality，比如 centerness/yawness
    '''

    # 1. 初始化函数：构建 box 回归分支、分类分支和 quality 估计分支
    def __init__(
        self,                          # self 表示当前模块对象
        embed_dims=256,                # embedding 维度
        output_dim=11,                 # 输出 box 状态维度，默认 11，通常对应 x,y,z,w,l,h,sin_yaw,cos_yaw,vx,vy,vz 等
        num_cls=10,                    # 检测类别数
        normalize_yaw=False,           # 是否对 sin_yaw/cos_yaw 做归一化
        refine_yaw=False,              # 是否细化 yaw：若 True，会把 SIN_YAW、COS_YAW 加入 refine_state
        with_cls_branch=True,          # 是否带分类分支
        with_quality_estimation=False, # 是否带质量估计分支，如输出 centerness、yawness 两维
    ):
        super(SparseBox3DRefinementModule, self).__init__() # 调用 BaseModule 初始化

        # 1. 保存参数：
        self.embed_dims = embed_dims       # 保存 embedding 维度
        self.output_dim = output_dim       # 保存输出状态维度
        self.num_cls = num_cls             # 保存类别数
        self.normalize_yaw = normalize_yaw # 保存是否归一化 yaw 的 sin/cos
        self.refine_yaw = refine_yaw       # 保存是否细化 yaw

        # 2. 设置一个列表 refine_state，表示哪些 box 状态需要做 residual refine
        self.refine_state = [X, Y, Z, W, L, H]      # 默认需要做 residual refine 的状态包括中心位置和尺寸：X,Y,Z,W,L,H
        if self.refine_yaw:                         # 若启用 yaw refinement
            self.refine_state += [SIN_YAW, COS_YAW] # 则也对 sin_yaw 和 cos_yaw 做 residual refine

        # 3.【3个分支都是线性层】构建 box 回归分支、分类分支和 quality 估计分支：
        # (1) 构建 box 回归分支：
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),        # 构建若干 Linear + ReLU + LN 层
            Linear(self.embed_dims, self.output_dim), # 最后一层线性层输出 output_dim 维 box 状态
            Scale([1.0] * self.output_dim),           # Scale 是一个逐维可学习缩放参数，初始每维 scale 都是 1.0，用于调节每个 box 状态维度的输出幅度
        )

        # (2) 构建分类分支：
        self.with_cls_branch = with_cls_branch # 保存是否带分类分支
        if with_cls_branch:                    # 若启用分类分支
            self.cls_layers = nn.Sequential(   # 构建分类分支
                *linear_relu_ln(embed_dims, 1, 2),     # 构建 Linear + ReLU + LN
                Linear(self.embed_dims, self.num_cls), # 输出 num_cls 个类别 logits
            )

        # (3) 构建 quality 估计分支：        
        self.with_quality_estimation = with_quality_estimation # 保存是否带 quality 分支
        if with_quality_estimation:                            # 若启用质量估计
            self.quality_layers = nn.Sequential(               # quality 估计分支
                *linear_relu_ln(embed_dims, 1, 2), # 构建 Linear + ReLU + LN
                Linear(self.embed_dims, 2),        # 输出 2 维质量估计，通常可理解为 centerness、yawness 一类的质量 logits
            )

    # 2. 初始化分类分支的权重【Sparse4DHead.init_weights() 会调用子模块的 init_weight()】
    def init_weight(self):
        if self.with_cls_branch:                                   # 若有分类分支
            bias_init = bias_init_with_prob(0.01)                  # 根据先验正样本概率 0.01 计算 bias 初始化值。常用于 FocalLoss，让训练初期预测偏向背景，稳定训练
            nn.init.constant_(self.cls_layers[-1].bias, bias_init) # 将分类最后一层 bias 初始化成该值

    # 3. 前向传播：根据输入特征和 anchor 预测更新后的 3D box 状态、分类 logits 和质量估计 logits
    def forward(
        self,                              # self 表示当前模块对象
        instance_feature: torch.Tensor,    # 当前 query 的内容特征，shape 为 [B, num_anchor, C]
        anchor: torch.Tensor,              # 当前 query 对应的 box 状态，shape 为 [B, num_anchor, output_dim]
        anchor_embed: torch.Tensor,        # anchor 几何编码，shape 为 [B, num_anchor, C]
        time_interval: torch.Tensor = 1.0, # 当前帧和历史帧的时间间隔，用于把预测的 translation 转成 velocity
        return_cls=True,                   # 是否返回分类结果
    ):
        # 1. 根据输入特征和 anchor 预测更新后的 3D box 状态
        # (1) 根据 instance_feature + anchor_embed 预测 box 更新量
        feature = instance_feature + anchor_embed # 将内容特征和位置编码相加，作为 box 回归分支和 quality 分支的输入，形状 [B, num_anchor, C]
        output = self.layers(feature)             # 通过回归分支预测 output_dim 维 box 状态或增量，形状 [B, num_anchor, output_dim]

        # (2.1) 对 refine_state 中的状态做 residual update，如 new_x = delta_x + anchor_x、new_w = delta_w + anchor_w，注意 W,L,H 通常仍在 log-space 下更新
        output[..., self.refine_state] = (output[..., self.refine_state] + anchor[..., self.refine_state])
        if self.normalize_yaw:                                                                                       # 若启用 yaw 归一化
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(output[..., [SIN_YAW, COS_YAW]], dim=-1) # 则对 sin_yaw/cos_yaw 做 L2 normalize，保证 sin^2 + cos^2 ≈ 1

        # (2.2) 同理，对速度也做 residual update：new_velocity = predicted_velocity + anchor_velocity
        if self.output_dim > 8: # 若输出维度大于 8，说明 box 状态里包含速度相关维度 VX
            if not isinstance(time_interval, torch.Tensor):                # 若 time_interval 不是 Tensor
                time_interval = instance_feature.new_tensor(time_interval) # 转成和 instance_feature 同设备同 dtype 的 Tensor

            # 对速度做 residual update
            translation = torch.transpose(output[..., VX:], 0, -1)         # 取 output 中 VX 之后的部分。这里先把 batch 维转到最后，是为了和 time_interval 做广播
            velocity = torch.transpose(translation / time_interval, 0, -1) # 将预测的位移量除以时间间隔，得到速度
            output[..., VX:] = velocity + anchor[..., VX:]                 # 速度也做 residual update：new_velocity = predicted_velocity + anchor_velocity

        # 2. 根据输入特征和 anchor 预测分类 logits
        if return_cls:                                                       # 若需要返回分类
            assert self.with_cls_branch, "Without classification layers !!!" # 确保当前模块有分类分支
            cls = self.cls_layers(instance_feature)                          # 分类分支只使用 instance_feature、不直接加 anchor_embed，形状 [B, num_anchor, num_cls]
        else:                                                                # 若不返回分类
            cls = None                                                       # 分类输出为 None

        # 3. 根据输入特征和 anchor 预测质量估计 logits
        if return_cls and self.with_quality_estimation: # 若需要分类且启用了质量估计
            quality = self.quality_layers(feature)      # 输入 feature = instance_feature + anchor_embed，预测质量估计 quality [B, num_anchor, 2(centerness,yawness)]
        else:
            quality = None # 否则 quality 为 None

        # 4. 返回：
        '''
            output : 更新后的 box anchor [B, num_anchor, output_dim]
            cls    : 分类 logits [B, num_anchor, num_cls]
            quality: 质量估计 logits [B, num_anchor, 2(centerness,yawness)]
        '''
        return output, cls, quality


# 三、 SparseBox3DKeyPointsGenerator 类：根据 3D box anchor 生成若干 3D key points
@PLUGIN_LAYERS.register_module() # 将 SparseBox3DKeyPointsGenerator 注册到 PLUGIN_LAYERS。配置文件中 kps_generator=dict(type="SparseBox3DKeyPointsGenerator", ...) 或 anchor_handler=dict(type="SparseBox3DKeyPointsGenerator") 都会构建这个类
class SparseBox3DKeyPointsGenerator(BaseModule):
    '''
        SparseBox3DKeyPointsGenerator 的作用：
            1. 根据 3D box anchor 生成若干 3D key points
            2. 这些 key points 会被投影到图像上做 DeformableFeatureAggregation
            3. 同时它也提供 anchor_projection，用于历史 anchor 坐标系变换
    '''

    # 1. 定义一个线性层的可学习关键点预测层 learnable_fc，用于预测可学习关键点的 3D 偏移比例
    def __init__(
        self,                # self 表示当前模块对象
        embed_dims=256,      # embedding 维度。若存在 learnable keypoints，会用 instance_feature 预测可学习点偏移
        num_learnable_pts=0, # 可学习关键点数量
        fix_scale=None,      # 固定关键点尺度：每个元素是相对 box 尺寸的比例，如 [0.45,0,0]
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__() # 调用 BaseModule 初始化

        # 保存 embedding 维度
        self.embed_dims = embed_dims

        # 保存可学习点数量
        self.num_learnable_pts = num_learnable_pts

        # 若没有传入固定尺度，则默认只使用 box 中心点
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),) # 默认只使用 box 中心点

        # 将固定尺度注册成不可训练参数，requires_grad=False 表示不会被优化器更新
        self.fix_scale = nn.Parameter(torch.tensor(fix_scale), requires_grad=False)

        # 总关键点数量 = 固定点数量 + 可学习点数量
        self.num_pts = len(self.fix_scale) + num_learnable_pts

        # 【核心】若有可学习关键点，则定义一个线性层的可学习关键点预测层 learnable_fc，用于预测可学习关键点的 3D 偏移比例
        if num_learnable_pts > 0:
            # 用 instance_feature 预测 num_learnable_pts 个 3D 偏移比例
            # 每个点 3 维，所以输出 num_learnable_pts * 3
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3) 

    # 2. 初始化可学习关键点预测层 learnable_fc 的权重
    def init_weight(self):
        # 若有可学习关键点，则使用 Xavier uniform 初始化 learnable_fc
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0) # 使用 Xavier uniform 初始化 learnable_fc

    # 3. 前向传播：核心作用是
    def forward(
        self,                  # self 表示当前模块对象
        anchor,                # 3D box anchor，shape [B, num_anchor, box_dim]
        instance_feature=None, # 当前 query 特征。若要预测可学习关键点，需要传入它
        T_cur2temp_list=None,  # 当前帧到历史帧的变换矩阵列表，用于生成历史时刻下的 key points
        cur_timestamp=None,    # 当前帧时间戳
        temp_timestamps=None,  # 历史帧时间戳列表
    ):
        # 1. 获取 batch size 和 anchor 数量，将 anchor 的 log_W,log_L,log_H 取 exp 还原为真实尺寸 size
        bs, num_anchor = anchor.shape[:2]         # 获取 batch size 和 anchor 数量
        size = anchor[..., None, [W, L, H]].exp() # 取 box 尺寸 W,L,H，并 exp 还原真实尺寸，size shape [B, num_anchor, 1, 3]。anchor 中 W,L,H 是 log-space，因此使用 .exp() 进行还原


        # 2. 构建关键点张量 key_points：
        # (1) 生成固定关键点【每个固定点 key_points 是 box 局部坐标系下相对中心的偏移】
        key_points = self.fix_scale * size
        '''
            self.fix_scale shape: [num_fix_pts, 3]
            size shape: [B, num_anchor, 1, 3]
            广播后 key_points shape: [B, num_anchor, num_fix_pts, 3]        
        '''

        # (2）先计算出可学习关键点，再拼接到固定关键点后，得到最终的关键点张量 key_points
        if self.num_learnable_pts > 0 and instance_feature is not None: # 若有可学习关键点，并且传入了 instance_feature
            # (a) 用 instance_feature 预测可学习点在 box 局部坐标系下的相对偏移比例
            learnable_scale = (
                self.learnable_fc(instance_feature)                 # Linear 输出 [B, num_anchor, num_learnable_pts * 3]
                .reshape(bs, num_anchor, self.num_learnable_pts, 3) # reshape 成 [B, num_anchor, num_learnable_pts, 3]
                .sigmoid()                                          # sigmoid 映射到 [0,1]，sigmoid 后的结果表示可学习点在 box 局部坐标系下的偏移比例
                - 0.5                                               # 减 0.5 后范围变成 [-0.5, 0.5]，表示可学习点在 box 局部坐标系下的偏移范围为 [-0.5*size, 0.5*size]
            )
            # (b) 将固定关键点和可学习关键点拼接，learnable_scale * size 表示可学习点在 box 局部坐标系下的偏移【通俗讲就是把可学习点的相对偏移比例乘以 box 尺寸，得到实际的偏移量；然后再添加 box 中心点，就得到所有可学习点在全局坐标系下的实际位置】
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)


        # 3. 构建 3x3 旋转矩阵 rotation_mat：
        # (1) 初始化全 0 的旋转矩阵【shape [B, num_anchor, 3, 3]，num_anchor 是 anchor 总数，因此 rotation_mat 表示每个 anchor 都有一个 3x3 的旋转矩阵】
        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])
        # (2) 填入 yaw 旋转矩阵
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]  # 第一行第一列 = cos(yaw)
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW] # 第一行第二列 = -sin(yaw)
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]  # 第二行第一列 = sin(yaw)
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]  # 第二行第二列 = cos(yaw)
        rotation_mat[:, :, 2, 2] = 1                      # z 方向不旋转
        '''
            最终得到的 rotation_mat 是一个 3x3 的旋转矩阵，用于将 box 局部坐标系下的关键点旋转到全局坐标系下，将它写成数学矩阵形式就是：
                [ cos(yaw)  -sin(yaw)   0 ]
                [ sin(yaw)   cos(yaw)   0 ]
                [    0          0       1 ]
        '''


        # 4. 将 box 局部坐标系下的关键点 key_points 转换到全局坐标系下的关键点 key_points
        # (1) 将 box 局部坐标系下的 key_points 旋转到 lidar/ego 坐标系
        key_points = torch.matmul(rotation_mat[:, :, None], key_points[..., None]).squeeze(-1) # 数学公式为：key_points_global = rotation_mat @ key_points_local
        '''
            rotation_mat[:, :, None] shape: [B, N, 1, 3, 3]
            key_points[..., None] shape: [B, N, P, 3, 1]
            输出 shape: [B, N, P, 3]，表示每个 anchor 的 P 个关键点在全局坐标系下的 3D 坐标
        '''
        # (2) 加上 box 中心点，得到全局于当前 lidar 坐标系的 3D key points
        key_points = key_points + anchor[..., None, [X, Y, Z]] # anchor[..., None, [X,Y,Z]] shape: [B, N, 1, 3]
        ''' 
            为啥要加上 anchor 的中心点:
                因为 key_points 是相对于 box 局部坐标系的偏移量，
                而 anchor 的 [X,Y,Z] 是 box 在全局坐标系下的中心点位置，
                所以要加上这个中心点才能得到 key_points 在全局坐标系下的实际位置。

            最终得到的 key_points shape: [B, N, P, 3]，表示每个 anchor 的 P 个关键点在全局坐标系下的 3D 坐标，每个 anchor 的关键点数量 P = num_fix_pts + num_learnable_pts
        '''
        # (3) 若没有提供 temporal 信息，则直接返回当前帧 key points
        if (
            cur_timestamp is None        # 当前时间戳为空
            or temp_timestamps is None   # 或历史时间戳为空
            or T_cur2temp_list is None   # 或当前到历史的变换矩阵为空
            or len(temp_timestamps) == 0 # 或历史时间戳列表长度为 0
        ):
            return key_points # 只返回当前帧 key points


        # 5. 若提供了 temporal 信息，则额外生成历史帧坐标系下的 key points
        temp_key_points_list = []   # 用于保存每个历史帧坐标系下的 key points
        velocity = anchor[..., VX:] # 取 anchor 中速度部分，anchor[..., VX:] 对应 vx, vy, vz 等速度维度，shape [B, num_anchor, vel_dim]
        for i, t_time in enumerate(temp_timestamps): # 遍历每个历史帧时间戳
            # (1) 计算时间间隔 = 当前帧时间戳 - 历史帧时间戳
            time_interval = cur_timestamp - t_time

            # (2) 根据速度 velocity 和时间间隔 time_interval 估计位移 translation
            translation = (velocity * time_interval.to(dtype=velocity.dtype)[:, None, None])
            '''
                velocity shape: [B,N,vel_dim]
                time_interval shape: [B]
                time_interval[:, None, None] 方便广播到 [B,N,vel_dim]            
            '''

            # (3) 将当前 key points 按速度反推到历史时刻，注意这里是 key_points - translation
            temp_key_points = key_points - translation[:, :, None]

            # (4) 取当前帧到第 i 个历史帧的变换矩阵
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)

            # (5) 将 temp_key_points 转成齐次坐标后，用 T_cur2temp 投影到历史坐标系
            temp_key_points = (
                T_cur2temp[:, None, None, :3] # T_cur2temp[:, None, None, :3] shape: [B,1,1,3,4]
                @ torch.cat(                  # 拼接齐次坐标 [x,y,z,1]
                    [
                        temp_key_points,                           # 原始点 
                        torch.ones_like(temp_key_points[..., :1]), # 齐次坐标 1
                    ],
                    dim=-1, # 最后一维拼接
                ).unsqueeze(-1) # 增加最后一维，变成列向量
            )
            temp_key_points = temp_key_points.squeeze(-1) # 去掉最后一维
            temp_key_points_list.append(temp_key_points)  # 加入历史 key points 列表


        # 6. 返回当前帧 key points 和每个历史帧坐标系下的 key points
        return key_points, temp_key_points_list
        '''
            key_points shape: [B, N, P, 3]，表示每个 anchor 的 P 个关键点在当前帧全局坐标系下的 3D 坐标
            temp_key_points_list 是一个列表，长度为 len(temp_timestamps)，每个元素 shape: [B, N, P, 3]，表示每个 anchor 的 P 个关键点在对应历史帧全局坐标系下的 3D 坐标
        '''

    # 4. 【核心】anchor_projection 静态方法：将 anchor 从源坐标系投影到目标坐标系【在 InstanceBank.get() 中会调用 anchor_projection()】
    @staticmethod
    def anchor_projection(
        anchor,              # 源坐标系下的 anchor
        T_src2dst_list,      # 源坐标系到目标坐标系的变换矩阵列表
        src_timestamp=None,  # 源时间戳
        dst_timestamps=None, # 目标时间戳列表
        time_intervals=None, # 显式给定的时间间隔列表
    ):
        dst_anchors = [] # 保存每个目标坐标系下的 anchor

        # 遍历每个目标变换矩阵，对 anchor 做投影
        for i in range(len(T_src2dst_list)):
            # 1. 获取 anchor 的速度、box中心点，获取变换矩阵、获取时间间隔
            # (1) 取速度部分
            vel = anchor[..., VX:]  # anchor[..., VX:] 对应 vx, vy, vz 等速度维度
            vel_dim = vel.shape[-1] # 速度维度，即 vx, vy, vz 等的数量，通常为3

            # (2) 取第 i 个变换矩阵，并在 anchor 数量维前增加一维
            T_src2dst = torch.unsqueeze(T_src2dst_list[i].to(dtype=anchor.dtype), dim=1) # T_src2dst 原 shape [B, 4, 4]，unsqueeze 后 shape [B, 1, 4, 4]，这样可以广播到每个 anchor

            # (3) 取 box 中心点 (x, y, z)
            center = anchor[..., [X, Y, Z]]

            # (4) 计算时间间隔 time_interval：
            if time_intervals is not None:                                              # 若直接给了时间间隔，
                time_interval = time_intervals[i]                                       # 则使用显式 time_intervals；
            elif src_timestamp is not None and dst_timestamps is not None:              # 若没有显式时间间隔，但给了源时间戳和目标时间戳，
                time_interval = (src_timestamp - dst_timestamps[i]).to(dtype=vel.dtype) # 则计算源到目标的时间间隔；
            else:                                                                       # 否则无法计算时间间隔，
                time_interval = None                                                    # 则time_interval 设为 None。


            # 2. 根据速度和时间间隔估计位移，并把中心点反推到目标时间，然后做坐标变换，最终得到从源坐标系投影到目标坐标系的 anchor 
            # (1) 根据速度 vel 和时间间隔 time_interval 估计位移 translation，并把中心点反推到目标时间
            if time_interval is not None: # 若有时间间隔
                translation = vel.transpose(0, -1) * time_interval # 根据速度估计位移，vel.transpose(0, -1) 是为了让 time_interval 广播
                translation = translation.transpose(0, -1)         # 再转回原来的维度顺序
                center = center - translation                      # 根据速度补偿中心点，center - translation 表示把中心点反推到目标时间

            # (2) 对中心点做坐标变换，将其从源坐标系变换到目标坐标系【公式为：p_dst = R @ p_src + t，其中 R 是旋转矩阵 T_src2dst[..., :3, :3]，t 是平移向量 T_src2dst[..., :3, 3]，p_src 是原始中心点坐标 center】
            center = (
                torch.matmul(T_src2dst[..., :3, :3], center[..., None]).squeeze(dim=-1) # 旋转部分 R @ center
                + T_src2dst[..., :3, 3]                                                 # 加上平移部分 t
            )

            # (3) 尺寸部分 W,L,H 保持不变，因为坐标系变换不改变 box 尺寸
            size = anchor[..., [W, L, H]]

            # (4) 对 yaw 方向向量做旋转【anchor 中 yaw 用 [sin, cos] 表示，这里先取 [cos, sin] 组成二维方向向量】
            yaw = torch.matmul(
                T_src2dst[..., :2, :2],                # 只使用 xy 平面的 2x2 旋转
                anchor[..., [COS_YAW, SIN_YAW], None], # 取 [cos_yaw, sin_yaw]
            ).squeeze(-1)
            yaw = yaw[..., [1,0]] # 再把 [cos, sin] 转回 [sin, cos]

            # (5) 对速度向量做旋转：使用变换矩阵左上角 vel_dim x vel_dim 部分
            vel = torch.matmul(T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]).squeeze(-1)

            # (6) 拼接成目标坐标系下的新 anchor，格式仍为 [x,y,z,w,l,h,sin_yaw,cos_yaw,vx,...]
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            dst_anchors.append(dst_anchor) # 加入输出列表

        # 返回已投影到每个目标坐标系下的 anchor
        return dst_anchors

    # 5. 计算 anchor 中心点到原点的 xy 平面距离
    @staticmethod
    def distance(anchor):
        return torch.norm(anchor[..., :2], p=2, dim=-1) # anchor[..., :2] 对应 x,y。p=2 表示 L2 norm。
