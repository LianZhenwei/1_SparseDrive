# ================================ 一、base config ================================
# 基础配置部分：控制数据版本、分布式训练、batch size、训练轮数、日志、checkpoint、fp16 等。

# 1.1 先设置为 mini 版本
# mini 一般用于快速调试，小数据集
version = 'mini'

# 1.2 这里马上又把 version 覆盖成 trainval
# 所以最终实际生效的依旧是 version='trainval' 而不是 version='mini'
version = 'trainval'

# 2. 不同数据版本对应的数据样本数量
# trainval：完整训练验证集信息
# mini：小规模调试集
length = {'trainval': 28130, 'mini': 323}

# 3.1 启用自定义 plugin
# SparseDrive 的很多模块不在 mmdet/mmdet3d 原生库里，需要通过 plugin 注册
plugin = True

# 3.2 指定自定义插件目录
# 框架会去这个目录下导入自定义模型、head、dataset、pipeline、loss 等
plugin_dir = "projects/mmdet3d_plugin/"

# 4. 分布式训练参数
# backend="nccl" 是 NVIDIA GPU 多卡训练常用通信后端
dist_params = dict(backend="nccl")

# 5.1 日志级别
# INFO 表示输出常规训练日志
log_level = "INFO"

# 6. 工作目录
# None 表示不在配置里固定，通常由命令行 --work-dir 指定
work_dir = None

# 7.1 总 batch size
# 表示所有 GPU 加起来的 batch size
total_batch_size = 64

# 7.2 GPU 数量
# 这里按 8 卡训练配置
num_gpus = 8

# 7.3 每张 GPU 的 batch size
# 64 // 8 = 8
batch_size = total_batch_size // num_gpus # lzw: 官方 Stage1 里 8 卡训练时的总 batch_size 是 64、每卡 batch_size 是 8

# 8.1 每个 epoch 的 iteration 数
# length[version] = 28130
# num_gpus * batch_size = 8 * 8 = 64
# 所以 num_iters_per_epoch = 28130 // 64 = 439
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))

# 8.2 epoch数：stage1 训练 100 个 epoch
# 注意后面 runner 是 IterBasedRunner，所以实际用 max_iters 控制
num_epochs = 100

# 8.3 每隔 20 个 epoch 保存一次 checkpoint
checkpoint_epoch_interval = 20

# 8.4 checkpoint 保存配置
checkpoint_config = dict(
    # checkpoint 间隔按 iteration 计算
    # 439 * 20 = 8780 iteration 保存一次
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)

# 5.2 日志配置
log_config = dict(
    # (1) 每隔 51 个 iteration 打印一次日志
    interval=51,

    # (2) 日志 hook 列表
    hooks=[
        # (i) 文本日志 hook
        # by_epoch=False 表示按 iteration 记录
        dict(type="TextLoggerHook", by_epoch=False),

        # (ii) TensorBoard 日志 hook
        # 便于用 tensorboard 查看 loss、lr 等曲线
        dict(type="TensorboardLoggerHook"),
    ],
)

# 8.5 从某个 checkpoint 加载权重
# stage1 这里为 None，表示不加载 SparseDrive 之前阶段的权重
# 但 backbone 内部仍会加载 ResNet-50 预训练权重
load_from = None

# 8.6 是否从中断训练处恢复
# None 表示不恢复 optimizer、lr scheduler、iter 等状态
resume_from = None

# 9. 训练流程
# 只执行 train，每次执行 1 个训练流程
workflow = [("train", 1)]

# 10. fp16 混合精度配置
# loss_scale=32.0 表示固定 loss scale
fp16 = dict(loss_scale=32.0)

# 11. 输入图像尺寸
# 这里写成 input_shape=(W, H)=(704, 256)
# 后面 data_aug_conf 里用 input_shape[::-1] 转成 final_dim=(256, 704)
input_shape = (704, 256)



# ================================ 二、model ================================
# 模型配置部分：定义类别、模型超参数、SparseDrive 主体、det head、map head、motion_plan_head 等。

# 1.1 nuScenes 3D detection 的 10 个类别
class_names = [
    "car",                  # 小汽车
    "truck",                # 卡车
    "construction_vehicle", # 工程车
    "bus",                  # 公交车
    "trailer",              # 拖车
    "barrier",              # 路障、障碍物
    "motorcycle",           # 摩托车
    "bicycle",              # 自行车
    "pedestrian",           # 行人
    "traffic_cone",         # 交通锥
]

# 1.2 3D 检测类别数
num_classes = len(class_names)

# 2.1 地图元素类别
map_class_names = [
    'ped_crossing', # 人行横道
    'divider',      # 车道分隔线
    'boundary',     # 道路边界
]

# 2.2 地图类别数
num_map_classes = len(map_class_names)

# 2.3 局部地图 ROI 范围
# 通常表示自车周围 x/y 范围，例如 30m × 60m
roi_size = (30, 60)

# 2.4 每条地图线采样点数量
# 一条 map vector 会表示成 20 个点
num_sample = 20

# 3.1 agent 未来预测时间步数
# stage1 不训练 motion，但这些变量仍被定义，供 motion_plan_head 配置占位
fut_ts = 12

# 3.2 agent 未来轨迹模态数
fut_mode = 6

# 4.1 ego 自车规划未来时间步数
ego_fut_ts = 6

# 4.2 ego 自车规划模态数
ego_fut_mode = 6

# 4.3 时序队列长度：history + current，通常表示 3 帧历史 + 当前帧
queue_length = 4 # history + current

# 5. 统一 embedding 维度
embed_dims = 256

# 6. attention head 或分组数
num_groups = 8

# 7.1 decoder 层数
num_decoder = 6

# 7.2 det head 中单帧 decoder 层数
# 前 1 层先做单帧建模，然后再引入 temporal
num_single_frame_decoder = 1

# 7.3 map head 中单帧 decoder 层数
num_single_frame_decoder_map = 1

# 8. 是否使用自定义 CUDA deformable aggregation 算子
# 如果为 True，需要先编译 mmdet3d_plugin/ops/setup.py
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed

# 9.1 FPN 多尺度输出的 stride
# 对应输入图像的 1/4、1/8、1/16、1/32 分辨率
strides = [4, 8, 16, 32]

# 9.2 多尺度特征层数量
num_levels = len(strides)

# 9.3 DenseDepthNet 使用的深度预测层数
# 这里使用前三个 FPN level 做深度辅助监督
num_depth_layers = 3

# 10. dropout 概率
drop_out = 0.1

# 11.1 detection 是否使用 temporal instance
temporal = True

# 11.2 map 是否使用 temporal 配置
# 但是 stage1 的 map_head 里 num_temp_instances=0，所以 map 实际不缓存 temporal instance
temporal_map = True

# 12.1 detection attention 是否采用 decouple 方式
# True 时 query feature 和 anchor embedding 拼接，attention 维度变成 2C
decouple_attn = True

# 12.2 map attention 是否采用 decouple 方式
# 这里为 False
decouple_attn_map = False

# 12.3 motion planning attention 是否 decouple
# stage1 不启用 motion_plan，但配置仍保留
decouple_attn_motion = True

# 13. detection refine layer 是否带质量估计分支
# 例如 centerness、yawness 等
with_quality_estimation = True

# 14. 任务开关
task_config = dict(
    with_det=True,          # (1) stage1 启用 3D detection
    with_map=True,          # (2) stage1 启用 map prediction
    with_motion_plan=False, # (3) stage1 不启用 motion prediction 和 planning，这是 stage1 和 stage2 最核心区别之一
)

# 15. 整体模型配置【重点！】
model = dict(
    # (1) 模型类型，对应 projects/mmdet3d_plugin/models/sparsedrive.py 中的 SparseDrive 类
    type="SparseDrive",

    # (2.1) 是否使用 GridMask 图像增强
    use_grid_mask=True,

    # (2.2) 是否使用自定义 deformable aggregation function
    use_deformable_func=use_deformable_func,

    # (2.3) 图像 backbone 配置
    img_backbone=dict(
        type="ResNet",                                # 使用 ResNet
        depth=50,                                     # ResNet-50 # lzw: 网络深度，可选值为 18 / 34 / 50 / 101 / 152，分别对应 ResNet-18 / ResNet-34 / ResNet-50 / ResNet-101 / ResNet-152
        num_stages=4,                                 # ResNet 有 4 个 stage # lzw: num_stages 是 残差阶段总数。标准 ResNet 包含 4 个残差阶段（对应原论文的 conv2_x、conv3_x、conv4_x、conv5_x），默认值就是 4。
        frozen_stages=-1,                             # -1 表示不冻结任何 stage # lzw: frozen_stages 表示冻结的残差阶段数量。-1 表示不冻结任何阶段、全部参与训练，0 表示仅冻结 Stem 部分（初始的 7×7 卷积 + 池化层），1 表示冻结 Stem + 第 1 个残差阶段（stage1），2 表示冻结 Stem + stage1 + stage2，以此类推
        norm_eval=False,                              # norm_eval=False 表示训练时 BN 仍更新均值方差
        style="pytorch",                              # PyTorch 风格 ResNet
        with_cp=True,                                 # with_cp=True 表示使用 checkpoint 节省显存 # 反向传播时会重新计算部分中间激活
        out_indices=(0, 1, 2, 3),                     # 输出四个 stage 的特征
        norm_cfg=dict(type="BN", requires_grad=True), # BN 配置，requires_grad=True 表示 BN 参数参与训练
        pretrained="ckpt/resnet50-19c8e357.pth",      # ResNet-50 ImageNet 预训练权重路径
    ),

    # (2.4) 图像 neck 配置
    img_neck=dict(
        type="FPN",                         # 使用 FPN 作为 neck
        num_outs=num_levels,                # 输出 num_levels=4 个尺度的特征图
        start_level=0,                      # 从 backbone 第 0 个 stage 开始构建 FPN
        out_channels=embed_dims,            # FPN 所有输出特征图的通道数统一为 embed_dims=256
        add_extra_convs="on_output",        # 在 FPN 输出上额外加卷积
        relu_before_extra_convs=True,       # extra conv 前使用 ReLU
        in_channels=[256, 512, 1024, 2048], # ResNet-50 四个 stage 输出通道数
    ),

    # (2.5) 深度辅助分支
    depth_branch=dict(                     # for auxiliary supervision only # 只是为了辅助监督
        type="DenseDepthNet",              # 模块类型，对应 DenseDepthNet
        embed_dims=embed_dims,             # 输入通道数
        num_depth_layers=num_depth_layers, # 使用前三层特征预测 depth
        loss_weight=0.2,                   # 深度 loss 权重
    ),

    # (3) SparseDrive 总 head
    head=dict(
        # a. 对应 /home/lzw/SparseDrive/projects/mmdet3d_plugin/models/sparsedrive_head_lzw.py 中的 SparseDriveHead 类
        type="SparseDriveHead",

        # b. 传入任务开关
        task_config=task_config,

        # c. 【重点】det head 的 3D 检测头配置
        det_head=dict(
            type="Sparse4DHead",         # 使用 /home/lzw/SparseDrive/projects/mmdet3d_plugin/models/detection3d/detection3d_head_lzw.py 里的 Sparse4DHead 类
            cls_threshold_to_reg=0.05,   # 分类分数超过 0.05 的预测才参与某些回归逻辑
            decouple_attn=decouple_attn, # detection 使用 decouple attention

            # instance bank 配置
            instance_bank=dict(
                type="InstanceBank",                                       # 对应 /home/lzw/SparseDrive/projects/mmdet3d_plugin/models/instance_bank.py 里的 InstanceBank 类
                num_anchor=900,                                            # 检测 anchor 数量
                embed_dims=embed_dims,                                     # instance feature 维度
                anchor="data/kmeans/kmeans_det_900.npy",                   # 由 k-means 聚类得到的 900 个 3D anchor
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"), # anchor handler，用于 3D box keypoints 生成和历史 anchor 投影
                num_temp_instances=600 if temporal else -1,                # temporal=True 时缓存 600 个历史 instance，temporal=False 时为 -1
                confidence_decay=0.6,                                      # 历史置信度衰减系数
                feat_grad=False,                                           # instance_feature 不参与梯度，常用于历史缓存稳定处理
            ),

            # anchor 编码器
            anchor_encoder=dict(
                type="SparseBox3DEncoder",                              # 3D box anchor encoder
                vel_dims=3,                                             # velocity 相关维度
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256, # decouple attention 时，各几何部分 embedding 分配为 [128,32,32,64]，否则统一 256
                mode="cat" if decouple_attn else "add",                 # decouple 时拼接，不 decouple 时相加
                output_fc=not decouple_attn,                            # decouple 时不额外输出 FC
                in_loops=1,                                             # 输入侧循环次数
                out_loops=4 if decouple_attn else 2,                    # 输出侧循环次数
            ),

            # 单帧 decoder 数量
            num_single_frame_decoder=num_single_frame_decoder,

            # decoder 操作顺序
            operation_order=(
                [
                    "gnn",        # 当前帧 query 之间的 self-attention
                    "norm",       # LayerNorm
                    "deformable", # 多相机多尺度 deformable 特征聚合
                    "ffn",        # FFN
                    "norm",       # LayerNorm
                    "refine",     # refine 更新 box 和分类
                ]
                * num_single_frame_decoder # num_single_frame_decoder=1：这是第1阶段即单帧发现阶段，该阶段只对单帧 decoder 部分（即第1个decoder）进行 1 次 gnn → norm → deformable → ffn → norm → refine
                + [
                    "temp_gnn",   # temporal attention，融合历史 instance
                    "gnn",        # 当前帧 query self-attention
                    "norm",       # LayerNorm
                    "deformable", # deformable feature aggregation
                    "ffn",        # FFN
                    "norm",       # LayerNorm
                    "refine",     # refine
                ]
                * (num_decoder - num_single_frame_decoder) # num_decoder-num_single_frame_decoder = 6-1 = 5：这是第2阶段即时序精修阶段，该阶段只对剩余的第2个到第6个的时序 decoder 层进行重复 temp_gnn → gnn → norm → deformable → ffn → norm → refine
            )[2:], # [2:] 表示跳过开头的 "gnn" 和 "norm"，因此第1层从 deformable 开始，因此实际上最终真正传入 Sparse4DHead 的 operation_order 为：第 1 个 decoder：deformable → ffn → norm → refine；第 2~6 个 decoder：temp_gnn → gnn → norm → deformable → ffn → norm → refine

            # temporal graph model
            temp_graph_model=dict(
                type="MultiheadFlashAttention",                                 # 使用 FlashAttention 版本多头注意力
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2, # decouple=True 时输入维度为 512
                num_heads=num_groups,                                           # 注意力头数
                batch_first=True,                                               # 输入格式为 [B, N, C]
                dropout=drop_out,                                               # dropout
            )
            if temporal
            else None, # temporal=True 时启用 temp_graph_model = dict(...)，否则 temp_graph_model=None 即不启用 temporal_graph_model

            # 当前帧 graph attention
            graph_model=dict(
                type="MultiheadFlashAttention",                                 # 使用 FlashAttention 版本多头注意力
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2, # decouple=True 时输入维度为 512
                num_heads=num_groups,                                           # 注意力头数
                batch_first=True,                                               # batch first
                dropout=drop_out,                                               # dropout
            ),

            # 归一化层
            norm_layer=dict(
                type="LN",                  # LayerNorm
                normalized_shape=embed_dims # 归一化维度 256
            ),

            # FFN 配置
            ffn=dict(
                type="AsymmetricFFN",                    # 使用 AsymmetricFFN
                in_channels=embed_dims * 2,              # 因为 deformable_model residual_mode="cat"，输出是 2C，所以这里输入 512
                pre_norm=dict(type="LN"),                # 前置 LayerNorm
                embed_dims=embed_dims,                   # 输出维度回到 256
                feedforward_channels=embed_dims * 4,     # 中间层维度 1024
                num_fcs=2,                               # 两层 FC
                ffn_drop=drop_out,                       # dropout
                act_cfg=dict(type="ReLU", inplace=True), # 激活函数 ReLU
            ),

            # deformable feature aggregation 配置
            deformable_model=dict(
                type="DeformableFeatureAggregation",     # 对应 blocks.py 中 DeformableFeatureAggregation
                embed_dims=embed_dims,                   # embedding 维度
                num_groups=num_groups,                   # 分组数
                num_levels=num_levels,                   # 多尺度层数
                num_cams=6,                              # nuScenes 6 个相机
                attn_drop=0.15,                          # attention dropout
                use_deformable_func=use_deformable_func, # 使用自定义 CUDA DAF
                use_camera_embed=True,                   # 使用 camera embedding
                residual_mode="cat",                     # 残差采用 cat，所以输出通道会变成 2 * embed_dims

                # keypoints 生成器
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator", # 3D box keypoints 生成器
                    num_learnable_pts=6,                  # 额外可学习点数量
                    fix_scale=[                           # 固定关键点相对 box 中心的尺度偏移
                        [0, 0, 0],     # 中心
                        [0.45, 0, 0],  # x 正方向
                        [-0.45, 0, 0], # x 负方向
                        [0, 0.45, 0],  # y 正方向
                        [0, -0.45, 0], # y 负方向
                        [0, 0, 0.45],  # z 正方向
                        [0, 0, -0.45], # z 负方向
                    ],
                ),
            ),

            # refine 层配置
            refine_layer=dict(
                type="SparseBox3DRefinementModule",              # 3D box refinement 模块
                embed_dims=embed_dims,                           # embedding 维度
                num_cls=num_classes,                             # 检测类别数
                refine_yaw=True,                                 # 是否细化 yaw
                with_quality_estimation=with_quality_estimation, # 是否输出质量估计
            ),

            # target sampler 配置
            sampler=dict(
                type="SparseBox3DTarget",                      # 3D box target 分配器
                num_dn_groups=0,                               # denoising group 数量。这里为 0，表示 stage1 不使用 DN training
                num_temp_dn_groups=0,                          # temporal denoising group 数
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,          # DN 噪声尺度。前 3 维通常对应 xyz，后 7 维对应尺寸、yaw、速度等
                max_dn_gt=32,                                  # 每张图最多 DN GT 数
                add_neg_dn=True,                               # 是否加入负样本 DN
                cls_weight=2.0,                                # 分类匹配权重
                box_weight=0.25,                               # box 匹配权重
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4, # 匹配时不同回归维度权重

                # 针对特定类别设置不同回归权重
                cls_wise_reg_weights={
                    # traffic_cone 类别特殊
                    # 对某些维度如速度不强监督
                    class_names.index("traffic_cone"): [
                        2.0,
                        2.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                },
            ),

            # 分类 loss
            loss_cls=dict(
                type="FocalLoss", # FocalLoss 适合类别不平衡               
                use_sigmoid=True, # sigmoid 形式
                gamma=2.0,        # focal loss gamma
                alpha=0.25,       # focal loss alpha
                loss_weight=2.0,  # 分类 loss 权重
            ),

            # 回归 loss
            loss_reg=dict(
                type="SparseBox3DLoss",                                          # SparseDrive 自定义 3D box loss
                loss_box=dict(type="L1Loss", loss_weight=0.25),                  # box L1 loss
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True), # centerness loss
                loss_yawness=dict(type="GaussianFocalLoss"),                     # yawness loss
                cls_allow_reverse=[class_names.index("barrier")],                # barrier 方向正反可能等价，所以允许 reverse
            ),

            # 检测 decoder
            decoder=dict(type="SparseBox3DDecoder"),

            # loss 中各回归维度权重
            # 前三维 xyz 权重为 2，其余 7 维为 1
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),

        # d. 【重点】map head 的 地图元素检测头配置
        map_head=dict(
            type="Sparse4DHead",             # map head 也使用 Sparse4DHead
            cls_threshold_to_reg=0.05,       # 分类阈值
            decouple_attn=decouple_attn_map, # map 不使用 decouple attention

            # map instance bank
            instance_bank=dict(
                type="InstanceBank",                                         # InstanceBank
                num_anchor=100,                                              # map anchor 数量
                embed_dims=embed_dims,                                       # embedding 维度
                anchor="data/kmeans/kmeans_map_100.npy",                     # k-means map anchor 文件
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"), # map point anchor handler
                num_temp_instances=0 if temporal_map else -1,                # 注意 stage1 这里是 0，虽然 temporal_map=True，但 num_temp_instances=0，所以 map head 实际不缓存历史 map instance
                confidence_decay=0.6,                                        # 置信度衰减
                feat_grad=True,                                              # map instance feature 参与梯度更新
            ),

            # map anchor encoder
            anchor_encoder=dict(
                type="SparsePoint3DEncoder", # 稀疏点编码器
                embed_dims=embed_dims,       # embedding 维度
                num_sample=num_sample,       # 每条 map line 20 个采样点
            ),

            # map 单帧 decoder 层数
            num_single_frame_decoder=num_single_frame_decoder_map,

            # map decoder 操作顺序
            operation_order=(
                [
                    "gnn",        # 当前 map query 之间 self-attention
                    "norm",       # LayerNorm
                    "deformable", # deformable feature aggregation
                    "ffn",        # FFN
                    "norm",       # LayerNorm
                    "refine",     # refine map line
                ]
                * num_single_frame_decoder_map # num_single_frame_decoder_map=1：这是第1阶段即单帧发现阶段，该阶段只对单帧 decoder 部分（即第1个decoder）进行 1 次 gnn → norm → deformable → ffn → norm → refine
                + [
                    "temp_gnn",   # temporal GNN，但由于 num_temp_instances=0，实际历史缓存为空
                    "gnn",        # 当前 query self-attention
                    "norm",       # LayerNorm
                    "deformable", # deformable
                    "ffn",        # FFN
                    "norm",       # LayerNorm
                    "refine",     # refine
                ]
                * (num_decoder - num_single_frame_decoder_map) # num_decoder-num_single_frame_decoder_map = 6-1 = 5：这是第2阶段即时序精修阶段，该阶段只对剩余的第2个到第6个的时序 decoder 层进行重复 temp_gnn → gnn → norm → deformable → ffn → norm → refine
            )[:], # [:] 表示完整保留【而 detection 任务头中会通过 [2:] 来删除第1个 decoder 中的 gnn 和 norm】

            # map temporal graph model
            temp_graph_model=dict(
                type="MultiheadFlashAttention",                                     # FlashAttention
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2, # decouple_attn_map=False，所以维度是 256
                num_heads=num_groups,                                               # attention head 数
                batch_first=True,                                                   # batch first
                dropout=drop_out,                                                   # dropout
            )
            if temporal_map
            else None, # temporal_map=True 时启用 temp_graph_model=dict(...)，否则 temp_graph_model=None 即不启用 temporal_graph_model

            # map 当前帧 graph model
            graph_model=dict(
                type="MultiheadFlashAttention",                                     # FlashAttention
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2, # embedding 维度
                num_heads=num_groups,                                               # attention head 数
                batch_first=True,                                                   # batch first
                dropout=drop_out,                                                   # dropout
            ),

            # map norm layer
            norm_layer=dict(type="LN", normalized_shape=embed_dims),

            # map FFN
            ffn=dict(
                type="AsymmetricFFN",                    # AsymmetricFFN
                in_channels=embed_dims * 2,              # deformable residual_mode="cat"，所以输入 512
                pre_norm=dict(type="LN"),                # 前置 LN
                embed_dims=embed_dims,                   # 输出 256
                feedforward_channels=embed_dims * 4,     # 中间层 1024
                num_fcs=2,                               # FC 层数量
                ffn_drop=drop_out,                       # dropout
                act_cfg=dict(type="ReLU", inplace=True), # ReLU
            ),

            # map deformable feature aggregation
            deformable_model=dict(
                type="DeformableFeatureAggregation",     # DeformableFeatureAggregation
                embed_dims=embed_dims,                   # embedding 维度
                num_groups=num_groups,                   # group 数
                num_levels=num_levels,                   # level 数
                num_cams=6,                              # 相机数
                attn_drop=0.15,                          # attention dropout
                use_deformable_func=use_deformable_func, # 使用 CUDA DAF
                use_camera_embed=True,                   # 使用 camera embedding
                residual_mode="cat",                     # concat residual

                # map keypoints 生成器
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator", # 稀疏点 3D keypoints 生成器
                    embed_dims=embed_dims,                  # embedding 维度
                    num_sample=num_sample,                  # 每条线 20 个采样点
                    num_learnable_pts=3,                    # 可学习点数量
                    fix_height=(0, 0.5, -0.5, 1, -1),       # 固定高度采样：用不同高度辅助投影到图像特征
                    ground_height=-1.84023,                 # ground height in lidar frame # lidar 坐标系下地面高度
                ),
            ),

            # map refine layer
            refine_layer=dict(
                type="SparsePoint3DRefinementModule", # map point refinement
                embed_dims=embed_dims,                # embedding 维度
                num_sample=num_sample,                # 每条线点数
                num_cls=num_map_classes,              # map 类别数量
            ),

            # map target sampler
            sampler=dict(
                type="SparsePoint3DTarget", # map 点目标分配器
                assigner=dict(              # 匈牙利匹配器
                    type='HungarianLinesAssigner', # 线匹配 assigner
                    cost=dict(                     # 匹配 cost
                        type='MapQueriesCost',                                                   # map query cost
                        cls_cost=dict(type='FocalLossCost', weight=1.0),                         # 分类 cost
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True), # 线回归 cost。permute=True 表示考虑线点序正反排列
                    ),
                ),
                num_cls=num_map_classes, # map 类别数
                num_sample=num_sample,   # 每条线采样点数
                roi_size=roi_size,       # ROI 范围
            ),

            # map 分类 loss
            loss_cls=dict(
                type="FocalLoss", # FocalLoss
                use_sigmoid=True, # sigmoid
                gamma=2.0,        # gamma
                alpha=0.25,       # alpha
                loss_weight=1.0,  # loss 权重
            ),

            # map 回归 loss
            loss_reg=dict(
                type="SparseLineLoss", # 稀疏线 loss
                loss_line=dict(        # line L1 loss
                    type='LinesL1Loss', # LinesL1Loss
                    loss_weight=10.0,   # 权重 10
                    beta=0.01,          # smooth L1 或内部距离参数
                ),
                num_sample=num_sample, # 每条线采样点数量
                roi_size=roi_size,     # ROI 范围
            ),

            # map decoder
            decoder=dict(type="SparsePoint3DDecoder"),

            # map 回归维度权重
            reg_weights=[1.0] * 40, # 20 个点，每个点 x,y 两维，所以 40 维

            # map GT 类别 key
            gt_cls_key="gt_map_labels",

            # map GT 点 key
            gt_reg_key="gt_map_pts",

            # map instance id key
            gt_id_key="map_instance_id",

            # 不输出 map instance id
            with_instance_id=False,

            # loss 名称前缀为 map
            task_prefix='map',
        ),

        # e. 【重点】motion planning head 的 motion planning 头配置
        # 这里虽然配置了 motion_plan_head，但是 task_config.with_motion_plan=False
        # 所以在 SparseDriveHead.__init__ 里不会真正 build 这个 head
        # 也就是说 stage1 不训练 motion prediction / planning
        motion_plan_head=dict(
            type='MotionPlanningHead',                                 # motion planning 模块类型
            fut_ts=fut_ts,                                             # agent 未来预测时间步
            fut_mode=fut_mode,                                         # agent 未来模态数
            ego_fut_ts=ego_fut_ts,                                     # ego 规划时间步
            ego_fut_mode=ego_fut_mode,                                 # ego 规划模态数
            motion_anchor=f'data/kmeans/kmeans_motion_{fut_mode}.npy', # motion anchor 文件
            plan_anchor=f'data/kmeans/kmeans_plan_{ego_fut_mode}.npy', # planning anchor 文件
            embed_dims=embed_dims,                                     # embedding 维度
            decouple_attn=decouple_attn_motion,                        # 是否 decouple attention

            # instance queue 配置
            instance_queue=dict(
                type="InstanceQueue",      # InstanceQueue
                embed_dims=embed_dims,     # embedding 维度
                queue_length=queue_length, # 历史队列长度
                tracking_threshold=0.2,    # tracking 阈值

                # 最后一层 feature map 尺度
                # input_shape[1]/strides[-1] = 256/32 = 8
                # input_shape[0]/strides[-1] = 704/32 = 22
                feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),
            ),

            # motion planning decoder 操作顺序
            operation_order=(
                [
                    "temp_gnn",  # temporal graph
                    "gnn",       # graph attention
                    "norm",      # norm
                    "cross_gnn", # cross graph attention
                    "norm",      # norm
                    "ffn",       # FFN                 
                    "norm",      # norm
                ] * 3 +
                [
                    "refine", # refine 输出 motion/planning
                ]
            ),

            # temporal graph model
            temp_graph_model=dict(
                type="MultiheadAttention",                                             # 普通 MultiheadAttention
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2, # decouple=True 时维度 512
                num_heads=num_groups,                                                  # head 数
                batch_first=True,                                                      # batch first
                dropout=drop_out,                                                      # dropout
            ),

            # graph model
            graph_model=dict(
                type="MultiheadFlashAttention",                                        # FlashAttention
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2, # decouple=True 时维度 512
                num_heads=num_groups,                                                  # head 数
                batch_first=True,                                                      # batch first
                dropout=drop_out,                                                      # dropout
            ),

            # cross graph model
            cross_graph_model=dict(
                type="MultiheadFlashAttention", # FlashAttention
                embed_dims=embed_dims,          # cross attention 使用 256 维
                num_heads=num_groups,           # head 数
                batch_first=True,               # batch first
                dropout=drop_out,               # dropout
            ),

            # norm layer
            norm_layer=dict(type="LN", normalized_shape=embed_dims),

            # motion planning FFN
            ffn=dict(
                type="AsymmetricFFN",                    # AsymmetricFFN
                in_channels=embed_dims,                  # 输入通道 256
                pre_norm=dict(type="LN"),                # 前置 LN
                embed_dims=embed_dims,                   # 输出 256
                feedforward_channels=embed_dims * 2,     # 中间层 512
                num_fcs=2,                               # 两层 FC
                ffn_drop=drop_out,                       # dropout
                act_cfg=dict(type="ReLU", inplace=True), # ReLU
            ),

            # motion planning refine layer
            refine_layer=dict(
                type="MotionPlanningRefinementModule", # refine 模块
                embed_dims=embed_dims,                 # embedding 维度
                fut_ts=fut_ts,                         # agent 时间步
                fut_mode=fut_mode,                     # agent 模态数
                ego_fut_ts=ego_fut_ts,                 # ego 时间步
                ego_fut_mode=ego_fut_mode,             # ego 模态数
            ),

            # motion target sampler
            motion_sampler=dict(
                type="MotionTarget", # motion target
            ),

            # motion 分类 loss
            motion_loss_cls=dict(
                type='FocalLoss', # FocalLoss
                use_sigmoid=True, # sigmoid
                gamma=2.0,        # gamma
                alpha=0.25,       # alpha
                loss_weight=0.2   # loss 权重
            ),

            # motion 回归 loss
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),

            # planning target sampler
            planning_sampler=dict(
                type="PlanningTarget",     # PlanningTarget
                ego_fut_ts=ego_fut_ts,     # ego 时间步
                ego_fut_mode=ego_fut_mode, # ego 模态数
            ),

            # planning 分类 loss
            plan_loss_cls=dict(
                type='FocalLoss', # FocalLoss
                use_sigmoid=True, # sigmoid
                gamma=2.0,        # gamma
                alpha=0.25,       # alpha
                loss_weight=0.5,  # loss 权重
            ),

            # planning 回归 loss
            plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),

            # planning status loss
            plan_loss_status=dict(type='L1Loss', loss_weight=1.0),

            # motion decoder
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),

            # planning decoder
            planning_decoder=dict(
                type="HierarchicalPlanningDecoder", # 分层规划 decoder
                ego_fut_ts=ego_fut_ts,              # ego 时间步
                ego_fut_mode=ego_fut_mode,          # ego 模态数
                use_rescore=True,                   # 是否重打分
            ),

            # 选取 top 50 检测目标用于 motion/planning
            num_det=50,

            # 选取 top 10 map 元素用于 planning
            num_map=10,
        ),
    ),
)



# ================================ 三、data ================================
# 数据配置部分：定义数据集类型、路径、数据增强、训练/测试/评估 pipeline。

# 1. 数据集类型
dataset_type = "NuScenes3DDataset"

# 2.1 nuScenes 数据根目录
data_root = "data/nuscenes/"

# 2.2 info 标注文件目录
# 当前 version=trainval，因此 anno_root="data/infos/"
anno_root = "data/infos/" if version == 'trainval' else "data/infos/mini/"

# 2.3 文件读取后端
# backend="disk" 表示从本地磁盘读取
file_client_args = dict(backend="disk")

# 3. 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], # ImageNet 均值
    std=[58.395, 57.12, 57.375],    # ImageNet 标准差
    to_rgb=True                     # 是否从 BGR 转 RGB
)

# 4. 训练 pipeline
train_pipeline = [
    # 加载多相机图像
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),

    # 加载点云
    # 虽然模型输入 use_lidar=False，但训练时可用点云生成 gt_depth 辅助监督
    dict(
        type="LoadPointsFromFile",         # 从文件读取点云
        coord_type="LIDAR",                # 点云坐标系
        load_dim=5,                        # 原始点云维度
        use_dim=5,                         # 使用的点云维度
        file_client_args=file_client_args, # 文件读取配置
    ),

    # 图像 resize、crop、flip 增强
    dict(type="ResizeCropFlipImage"),

    # 生成多尺度深度图
    dict(
        type="MultiScaleDepthMapGenerator",    # MultiScaleDepthMapGenerator
        downsample=strides[:num_depth_layers], # 下采样尺度为 [4, 8, 16]
    ),

    # 旋转 3D bbox，使其与图像增强保持一致
    dict(type="BBoxRotation"),

    # 多视角图像光度增强
    dict(type="PhotoMetricDistortionMultiViewImage"),

    # 多视角图像归一化
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),

    # 目标距离范围过滤
    dict(
        type="CircleObjectRangeFilter",           # 圆形范围过滤
        class_dist_thred=[55] * len(class_names), # 每个类别保留 55m 范围内目标
    ),

    # 按类别名称过滤实例
    dict(type="InstanceNameFilter", classes=class_names),

    # 矢量化地图
    dict(
        type='VectorizeMap',   # VectorizeMap
        roi_size=roi_size,     # ROI 范围
        simplify=False,        # 训练时不简化线
        normalize=False,       # 不归一化坐标
        sample_num=num_sample, # 每条线采样 20 个点
        permute=True,          # 允许线点序排列
    ),

    # NuScenes 数据适配为 Sparse4D/SparseDrive 需要的格式
    dict(type="NuScenesSparse4DAdaptor"),

    # 收集训练需要的字段
    dict(
        # Collect
        type="Collect",

        # 模型 forward 和 loss 需要的字段
        keys=[
            # 多视角图像
            "img",

            # 时间戳
            "timestamp",

            # 相机投影矩阵
            "projection_mat",

            # 图像宽高
            "image_wh",

            # 深度 GT
            "gt_depth",

            # 焦距
            "focal",

            # 3D box GT
            "gt_bboxes_3d",

            # 3D label GT
            "gt_labels_3d",

            # map label GT
            'gt_map_labels', 

            # map 点序列 GT
            'gt_map_pts',

            # agent 未来轨迹 GT
            # stage1 不训练 motion，但 pipeline 仍然收集该字段
            'gt_agent_fut_trajs',

            # agent 未来轨迹 mask
            'gt_agent_fut_masks',

            # ego 未来轨迹 GT
            'gt_ego_fut_trajs',

            # ego 未来轨迹 mask
            'gt_ego_fut_masks',

            # ego 高层命令
            'gt_ego_fut_cmd',

            # ego 状态
            'ego_status',
        ],

        # 元信息字段
        meta_keys=[
            # 当前帧到 global 的变换
            "T_global",

            # global 到当前帧的变换
            "T_global_inv",

            # 时间戳
            "timestamp",

            # instance id，用于 tracking/temporal association
            "instance_id"
        ],
    ),
]

# 5. 测试 pipeline
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True), # 加载多视角图像
    dict(type="ResizeCropFlipImage"),                          # 图像 resize/crop/flip。测试时一般是确定性处理
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),      # 图像归一化
    dict(type="NuScenesSparse4DAdaptor"),                      # 数据适配器

    # 收集测试需要字段
    dict(
        type="Collect", # Collect

        # 输入字段
        keys=[
            "img",            # 图像
            "timestamp",      # 时间戳
            "projection_mat", # 投影矩阵
            "image_wh",       # 图像宽高
            'ego_status',     # ego 状态
            'gt_ego_fut_cmd', # ego command
        ],

        # 元信息字段
        meta_keys=[
            "T_global",     # 当前到 global
            "T_global_inv", # global 到当前
            "timestamp"     # 时间戳
        ],
    ),
]

# 6. 评估 pipeline
eval_pipeline = [
    # 目标范围过滤
    dict(
        type="CircleObjectRangeFilter",           # 圆形范围过滤
        class_dist_thred=[55] * len(class_names), # 每类 55m
    ),

    # 类别过滤
    dict(type="InstanceNameFilter", classes=class_names),

    # map vector 生成
    dict(
        type='VectorizeMap', # VectorizeMap
        roi_size=roi_size,   # ROI 范围
        simplify=True,       # 评估时简化线
        normalize=False,     # 不归一化
    ),

    # 收集评估需要的字段
    dict(
        type='Collect', # Collect

        # 评估字段
        keys=[
            'vectors',            # 矢量地图 GT
            "gt_bboxes_3d",       # 3D box GT
            "gt_labels_3d",       # 3D label GT
            'gt_agent_fut_trajs', # agent 未来轨迹
            'gt_agent_fut_masks', # agent 未来轨迹 mask
            'gt_ego_fut_trajs',   # ego 未来轨迹
            'gt_ego_fut_masks',   # ego 未来轨迹 mask 
            'gt_ego_fut_cmd',     # ego command
            'fut_boxes'           # future boxes
        ],

        # 元信息
        meta_keys=[
            'token',    # nuScenes token
            'timestamp' # 时间戳
        ]
    ),
]

# 7. 输入模态配置
input_modality = dict(
    use_lidar=False,    # 不把 lidar 作为模型输入
    use_camera=True,    # 使用 camera 作为模型输入
    use_radar=False,    # 不使用 radar
    use_map=False,      # 不使用外部 HD map 作为输入，注意这不代表不预测 map，而是不把 map 当输入
    use_external=False, # 不使用额外外部信息
)

# 8. 数据集基础配置
data_basic_config = dict(
    type=dataset_type,           # 数据集类型
    data_root=data_root,         # 数据根目录
    classes=class_names,         # 3D 检测类别
    map_classes=map_class_names, # map 类别
    modality=input_modality,     # 输入模态
    version="v1.0-trainval",     # nuScenes 版本
)

# 9. 评估配置
eval_config = dict(
    **data_basic_config,                           # 展开基础配置
    ann_file=anno_root + 'nuscenes_infos_val.pkl', # 验证集 ann 文件
    pipeline=eval_pipeline,                        # 评估 pipeline
    test_mode=True,                                # 测试模式
)

# 10. 数据增强配置
data_aug_conf = {
    "resize_lim": (0.40, 0.47),     # resize 比例范围
    "final_dim": input_shape[::-1], # 最终图像尺寸，input_shape=(704,256)，[::-1] 后是 (256,704)
    "bot_pct_lim": (0.0, 0.0),      # 底部裁剪比例
    "rot_lim": (-5.4, 5.4),         # 图像旋转角度范围
    "H": 900,                       # 原始图像高度
    "W": 1600,                      # 原始图像宽度
    "rand_flip": True,              # 是否随机翻转
    "rot3d_range": [0, 0],          # 3D 旋转增强范围，[0,0] 表示不做 3D 旋转增强
}

# 11. DataLoader 和数据集配置
data = dict(
    # 每张 GPU 的样本数，stage1 中 batch_size=8
    samples_per_gpu=batch_size,

    # 每张 GPU 的 worker 数
    workers_per_gpu=batch_size,

    # 训练集配置
    train=dict(
        **data_basic_config,                             # 展开基础配置
        ann_file=anno_root + "nuscenes_infos_train.pkl", # 训练集 info 文件
        pipeline=train_pipeline,                         # 训练 pipeline
        test_mode=False,                                 # 非测试模式
        data_aug_conf=data_aug_conf,                     # 数据增强参数
        with_seq_flag=True,                              # 启用序列标志，temporal 模型需要保持样本序列关系
        sequences_split_num=2,                           # 将序列分成 2 段
        keep_consistent_seq_aug=True,                    # 同一序列保持一致数据增强。对时序模型很重要，避免相邻帧增强不一致破坏几何关系
    ),

    # 验证集配置
    val=dict(
        **data_basic_config,                           # 展开基础配置
        ann_file=anno_root + "nuscenes_infos_val.pkl", # 验证集 info 文件
        pipeline=test_pipeline,                        # 测试 pipeline
        data_aug_conf=data_aug_conf,                   # 数据增强配置
        test_mode=True,                                # 测试模式
        eval_config=eval_config,                       # 评估配置
    ),

    # 测试集配置
    test=dict(
        **data_basic_config,                           # 展开基础配置
        ann_file=anno_root + "nuscenes_infos_val.pkl", # 这里 test 也使用 val info
        pipeline=test_pipeline,                        # 测试 pipeline
        data_aug_conf=data_aug_conf,                   # 数据增强配置
        test_mode=True,                                # 测试模式
        eval_config=eval_config,                       # 评估配置
    ),
)



# ================================ 四、training ================================
# 训练配置：优化器、梯度裁剪、学习率策略、runner。

# 1.1 优化器配置
optimizer = dict(
    type="AdamW",       # AdamW 优化器
    lr=4e-4,            # stage1 基础学习率
    weight_decay=0.001, # weight decay

    # 针对不同参数设置不同学习率
    paramwise_cfg=dict(
        # 自定义参数组
        custom_keys={
            # backbone 使用 0.5 倍学习率
            # 因为 backbone 有 ImageNet 预训练，通常学习率小一点
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)

# 1.2 optimizer hook 配置
optimizer_config = dict(
    # 梯度裁剪
    grad_clip=dict(
        max_norm=25, # 最大梯度范数
        norm_type=2  # L2 norm
    )
)

# 2. 学习率策略
lr_config = dict(
    policy="CosineAnnealing", # 余弦退火
    warmup="linear",          # 线性 warmup
    warmup_iters=500,         # warmup 500 iter
    warmup_ratio=1.0 / 3,     # warmup 初始比例，初始 lr = base_lr * 1/3
    min_lr_ratio=1e-3,        # 最小学习率比例
)

# 3. runner 配置
runner = dict(
    # IterBasedRunner：按 iteration 控制训练，而不是按 epoch
    type="IterBasedRunner",

    # 最大 iteration
    # trainval 下 num_iters_per_epoch=439
    # 100 epoch -> 43900 iter
    max_iters=num_iters_per_epoch * num_epochs,
)



# ================================ 五、eval ================================
# 评估配置：stage1 只评估 detection、tracking、map，不评估 motion/planning。

# 1. 评估模式
eval_mode = dict(
    with_det=True,          # 评估 3D detection
    with_tracking=True,     # 评估 tracking
    with_map=True,          # 评估 map
    with_motion=False,      # stage1 不评估 motion prediction
    with_planning=False,    # stage1 不评估 planning
    tracking_threshold=0.2, # tracking 阈值
    motion_threshhold=0.2,  # motion 阈值【注意这里拼写是 motion_threshhold，不是 motion_threshold，如果评估源码也是这样读取，就没问题】
)

# 2. evaluation hook 配置
evaluation = dict(
    interval=num_iters_per_epoch*checkpoint_epoch_interval, # 每隔 checkpoint_epoch_interval 个 epoch 对应的 iter 评估一次。这里 439 * 20 = 8780 iter
    eval_mode=eval_mode,                                    # 传入评估模式
)
