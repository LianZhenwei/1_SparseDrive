# =============================== 一、base config ===============================
# 这一部分是基础训练配置，包括数据版本、分布式训练、日志、batch size、epoch、fp16 等。

# 1.1 先把 version 设置成 mini
# mini 一般表示 nuScenes mini 小数据集，适合快速调试
version = 'mini'

# 1.2 这里又把 version 覆盖成 trainval
# 所以最终实际生效的是 version = 'trainval'
# 上一行 version = 'mini' 相当于被覆盖掉了
version = 'trainval'

# 2. 定义不同数据版本对应的数据长度
# trainval 有 28130 个训练样本
# mini 有 323 个样本
length = {'trainval': 28130, 'mini': 323}

# 3.1 启用自定义 plugin
# 在 mmdetection / mmdet3d 里，如果模型、数据处理、loss、head 等是自定义的，需要设置 plugin=True
plugin = True

# 3.2 指定自定义插件目录
# SparseDrive 的自定义模块都在 projects/mmdet3d_plugin/ 下面
plugin_dir = "projects/mmdet3d_plugin/"

# 4. 分布式训练通信后端设置
# nccl 是 NVIDIA GPU 多卡训练最常用的通信后端
dist_params = dict(backend="nccl")

# 日志级别
# INFO 表示打印常规训练信息
log_level = "INFO"

# 工作目录
# None 表示不在这里硬编码，可能由命令行参数或默认规则指定
work_dir = None


# 总 batch size
# 注意这是所有 GPU 加起来的总 batch size
total_batch_size = 48

# 使用 GPU 数量
# 这里默认按照 8 卡训练配置
num_gpus = 8

# 每张 GPU 上的 batch size
# 48 // 8 = 6
batch_size = total_batch_size // num_gpus # lzw: 官方 Stage2 里 8 卡训练时的总 batch_size 是 48、每卡 batch_size 是 6

# 每个 epoch 有多少个 iteration
# length[version] 是样本总数
# num_gpus * batch_size = 全局 batch size
# 对 trainval 来说：28130 // 48 = 586
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))

# 训练 epoch 数
# 注意这里虽然写 epoch，但 runner 后面用的是 IterBasedRunner，本质是按 iteration 跑
num_epochs = 10

# 每隔多少个 epoch 保存一次 checkpoint
# 这里是 10，表示 10 个 epoch 保存一次
checkpoint_epoch_interval = 10


# checkpoint 保存配置
checkpoint_config = dict(
    # interval 表示每隔多少个 iteration 保存一次 checkpoint
    # num_iters_per_epoch * checkpoint_epoch_interval = 586 * 10 = 5860
    # 因此基本是整个 stage2 训练结束时保存一次
    interval=num_iters_per_epoch * checkpoint_epoch_interval
)

# 日志配置
log_config = dict(
    # 每隔 51 个 iteration 打印一次日志
    interval=51,

    # 日志 hook 列表
    hooks=[
        # 文本日志 hook
        # by_epoch=False 表示按 iteration 记录，而不是按 epoch
        dict(type="TextLoggerHook", by_epoch=False),

        # TensorBoard 日志 hook
        # 用于在 tensorboard 中可视化 loss、lr 等曲线
        dict(type="TensorboardLoggerHook"),
    ],
)

# 从某个 checkpoint 加载模型权重
# 这里先设置为 None
# 但文件最后又重新设置 load_from = 'ckpt/sparsedrive_stage1.pth'
# 所以最终会加载 stage1 权重
load_from = None

# 是否从中断训练处继续训练
# None 表示不 resume optimizer、iter、scheduler 等状态
resume_from = None

# 训练流程
# [("train", 1)] 表示只执行训练流程，每轮执行 1 次 train
workflow = [("train", 1)]

# fp16 混合精度训练配置
# loss_scale=32.0 表示固定 loss scale，用于避免 fp16 梯度下溢
fp16 = dict(loss_scale=32.0)

# 输入图像尺寸
# 这里写的是宽高形式：input_shape = (W, H) = (704, 256)
# 后面 data_aug_conf 里会用 input_shape[::-1] 变成 final_dim=(256, 704)
input_shape = (704, 256)



# =============================== 二、model ===============================
# 这一部分是模型配置，包括类别、任务开关、backbone、neck、检测头、地图头、运动规划头。


# 3D 检测类别名称
# 这是 nuScenes 3D detection 常用的 10 类
class_names = [
    # 小汽车
    "car",

    # 卡车
    "truck",

    # 施工车辆
    "construction_vehicle",

    # 公交车
    "bus",

    # 挂车、拖车
    "trailer",

    # 障碍物，例如路障
    "barrier",

    # 摩托车
    "motorcycle",

    # 自行车
    "bicycle",

    # 行人
    "pedestrian",

    # 交通锥
    "traffic_cone",
]

# 地图元素类别名称
# SparseDrive 的 map head 预测矢量化地图元素
map_class_names = [
    # 人行横道
    'ped_crossing',

    # 分隔线，例如车道分隔线
    'divider',

    # 道路边界
    'boundary',
]

# 检测类别数量
# len(class_names) = 10
num_classes = len(class_names)

# 地图类别数量
# len(map_class_names) = 3
num_map_classes = len(map_class_names)

# 地图 ROI 范围大小
# 一般表示自车周围局部地图区域
# 这里可以理解成 x 方向 30m，y 方向 60m 的范围
roi_size = (30, 60)


# 每条地图线采样点数量
# map head 将每条矢量线表示为 20 个采样点
num_sample = 20

# agent 未来预测时间步数
# fut_ts=12 表示预测其他交通参与者未来 12 帧/步轨迹
fut_ts = 12

# agent 未来轨迹模态数
# fut_mode=6 表示为每个 agent 预测 6 种可能未来轨迹
fut_mode = 6

# ego 自车未来规划时间步数
# ego_fut_ts=6 表示规划自车未来 6 帧/步轨迹
ego_fut_ts = 6

# ego 自车未来规划模态数
# ego_fut_mode=6 表示自车规划有 6 个候选模态
ego_fut_mode = 6

# 时序队列长度
# history + current 表示包含历史帧和当前帧
# queue_length=4 通常表示使用 3 帧历史 + 1 帧当前
queue_length = 4 # history + current


# 特征维度
# SparseDrive 的 query、anchor embedding、attention embedding 大多使用 256 维
embed_dims = 256

# 多头注意力分组数
# num_groups=8 通常对应 attention 的 head 数或 deformable aggregation 的 group 数
num_groups = 8

# decoder 层数
# det head 和 map head 都会用到
num_decoder = 6

# 单帧 decoder 层数
# 检测任务中，前面多少层不使用 temporal 信息
num_single_frame_decoder = 1

# 地图任务的单帧 decoder 层数
num_single_frame_decoder_map = 1

# 是否使用自定义 deformable aggregation CUDA 算子
# 如果为 True，需要先编译 projects/mmdet3d_plugin/ops/setup.py
use_deformable_func = True  # mmdet3d_plugin/ops/setup.py needs to be executed

# FPN 多尺度特征的 stride
# 对应输入图像下采样 4、8、16、32 倍的特征层
strides = [4, 8, 16, 32]

# 特征层数量
# len(strides)=4
num_levels = len(strides)

# 深度辅助分支使用的特征层数量
# 使用前 3 个尺度生成深度监督
num_depth_layers = 3

# dropout 概率
# 用在 attention、FFN 等模块中，防止过拟合
drop_out = 0.1

# 检测任务是否使用时序信息
temporal = True

# 地图任务是否使用时序信息
temporal_map = True

# 检测 head 是否使用解耦 attention
# True 时，某些 embedding 维度会变成 embed_dims * 2
decouple_attn = True

# map head 是否使用解耦 attention
# 这里为 False
decouple_attn_map = False

# motion planning head 是否使用解耦 attention
decouple_attn_motion = True

# 检测 refine layer 是否输出质量估计
# 例如 centerness、yawness 等质量分支
with_quality_estimation = True


# 任务开关配置
task_config = dict(
    # 启用 3D detection
    with_det=True,

    # 启用 map element detection
    with_map=True,

    # 启用 motion prediction 和 ego planning
    with_motion_plan=True,
)


# 整体模型配置
model = dict(
    # 外层模型类型，对应前面看的 SparseDrive 类
    type="SparseDrive",

    # 是否使用 GridMask 图像增强
    use_grid_mask=True,

    # 是否使用 deformable aggregation 自定义算子
    use_deformable_func=use_deformable_func,

    # 图像 backbone 配置
    img_backbone=dict(
        # backbone 类型为 ResNet
        type="ResNet",

        # ResNet-50
        depth=50,

        # ResNet 有 4 个 stage
        num_stages=4,

        # frozen_stages=-1 表示不冻结任何 stage
        frozen_stages=-1,

        # norm_eval=False 表示 BN 层训练时仍然更新统计量
        norm_eval=False,

        # 使用 PyTorch 风格 ResNet
        style="pytorch",

        # with_cp=True 表示使用 checkpoint 技术节省显存
        # 代价是反向传播时会重新计算部分中间结果
        with_cp=True,

        # 输出 4 个 stage 的特征
        out_indices=(0, 1, 2, 3),

        # BN 配置
        # requires_grad=True 表示 BN 的参数参与训练
        norm_cfg=dict(type="BN", requires_grad=True),

        # ResNet-50 预训练权重路径
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),

    # 图像 neck 配置
    img_neck=dict(
        # 使用 FPN
        type="FPN",

        # 输出特征层数量
        # num_levels=4
        num_outs=num_levels,

        # 从 backbone 第 0 个 stage 开始接入 FPN
        start_level=0,

        # FPN 输出通道数为 256
        out_channels=embed_dims,

        # 在 FPN 输出后额外添加卷积层
        add_extra_convs="on_output",

        # extra conv 前使用 ReLU
        relu_before_extra_convs=True,

        # ResNet-50 四个 stage 的输出通道数
        in_channels=[256, 512, 1024, 2048],
    ),

    # 深度辅助分支
    # 只用于训练阶段的辅助监督，不一定直接用于最终预测
    depth_branch=dict(  # for auxiliary supervision only
        # 深度网络类型
        type="DenseDepthNet",

        # 输入 embedding 维度
        embed_dims=embed_dims,

        # 使用几个尺度的特征做深度预测
        num_depth_layers=num_depth_layers,

        # 深度 loss 权重
        loss_weight=0.2,
    ),

    # SparseDrive 总 head
    head=dict(
        # 总 head 类型，对应 SparseDriveHead
        type="SparseDriveHead",

        # 传入任务开关
        task_config=task_config,

        # ===================== detection head =====================
        # 3D 检测头配置
        det_head=dict(
            # 检测头类型
            # Sparse4DHead 是 SparseDrive/Sparse4D 系列中的稀疏检测头
            type="Sparse4DHead",

            # 分类分数超过该阈值的 query 才参与某些回归逻辑
            cls_threshold_to_reg=0.05,

            # 是否使用解耦 attention
            decouple_attn=decouple_attn,

            # instance bank 配置
            # 用于维护稀疏 3D anchor/query
            instance_bank=dict(
                # instance bank 类型
                type="InstanceBank",

                # 检测 anchor 数量
                # 这里使用 900 个 3D object anchor
                num_anchor=900,

                # 每个 anchor/query 的特征维度
                embed_dims=embed_dims,

                # k-means 聚类得到的检测 anchor 初始化文件
                anchor="data/kmeans/kmeans_det_900.npy",

                # anchor 关键点生成器
                # 用于从 3D box anchor 生成若干采样点
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),

                # 时序保留的 instance 数量
                # temporal=True 时保留 600 个历史 instance
                # 如果不使用 temporal，则为 -1
                num_temp_instances=600 if temporal else -1,

                # 历史 instance 置信度衰减系数
                confidence_decay=0.6,

                # 是否允许 instance feature 回传梯度
                # False 表示可能将历史特征作为缓存使用，避免梯度跨帧传播
                feat_grad=False,
            ),

            # 3D box anchor 编码器
            anchor_encoder=dict(
                # 编码器类型
                type="SparseBox3DEncoder",

                # velocity 维度
                # 一般 3D box 状态中包含 vx、vy 等速度信息
                # 这里 vel_dims=3，说明速度相关编码维度按源码定义为 3 维
                vel_dims=3,

                # 不同部分的 embedding 维度
                # 解耦 attention 时使用 [128, 32, 32, 64]
                # 不解耦时直接用 256
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,

                # 编码融合方式
                # 解耦时 cat 拼接，不解耦时 add 相加
                mode="cat" if decouple_attn else "add",

                # 是否使用输出 FC
                # 解耦时不使用，不解耦时使用
                output_fc=not decouple_attn,

                # 输入侧 MLP/FC 循环次数
                in_loops=1,

                # 输出侧 MLP/FC 循环次数
                out_loops=4 if decouple_attn else 2,
            ),

            # 单帧 decoder 层数
            num_single_frame_decoder=num_single_frame_decoder,

            # detection decoder 的操作顺序
            operation_order=(
                # 单帧 decoder 的操作序列
                [
                    # 图内 query 之间的 GNN/attention
                    "gnn",

                    # 归一化
                    "norm",

                    # 多相机多尺度 deformable feature aggregation
                    "deformable",

                    # FFN 前馈网络
                    "ffn",

                    # 再归一化
                    "norm",

                    # refine 更新 3D box、类别等预测
                    "refine",
                ]
                # 重复 num_single_frame_decoder 次
                * num_single_frame_decoder

                # 后续时序 decoder 的操作序列
                + [
                    # temporal GNN，融合历史 instance
                    "temp_gnn",

                    # 当前帧 query 之间的 GNN/attention
                    "gnn",

                    # 归一化
                    "norm",

                    # 图像特征聚合
                    "deformable",

                    # FFN
                    "ffn",

                    # 归一化
                    "norm",

                    # refine
                    "refine",
                ]
                # 重复剩余 decoder 层数
                * (num_decoder - num_single_frame_decoder)

            # [2:] 表示从第三个操作开始
            # 对于第一层，跳过开头的 "gnn", "norm"
            )[2:],

            # temporal graph model 配置
            # 用于跨帧 instance 交互
            temp_graph_model=dict(
                # 使用 MultiheadFlashAttention
                type="MultiheadFlashAttention",

                # 如果 decouple_attn=True，attention 输入维度是 512
                # 否则是 256
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,

                # attention head 数
                num_heads=num_groups,

                # 输入格式为 [B, N, C]
                batch_first=True,

                # dropout
                dropout=drop_out,
            )
            # temporal=True 时启用
            if temporal
            # temporal=False 时不使用 temporal graph
            else None,

            # 当前帧 query 之间的 graph attention
            graph_model=dict(
                # 使用 FlashAttention 版本的多头注意力
                type="MultiheadFlashAttention",

                # 解耦时维度翻倍
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,

                # attention head 数
                num_heads=num_groups,

                # batch 维在最前
                batch_first=True,

                # dropout 概率
                dropout=drop_out,
            ),

            # 归一化层配置
            norm_layer=dict(
                # LayerNorm
                type="LN",

                # 归一化维度为 256
                normalized_shape=embed_dims
            ),

            # FFN 配置
            ffn=dict(
                # 非对称 FFN
                type="AsymmetricFFN",

                # 输入通道数
                # detection head 这里是 embed_dims * 2
                in_channels=embed_dims * 2,

                # FFN 前先做 LayerNorm
                pre_norm=dict(type="LN"),

                # 输出 embedding 维度
                embed_dims=embed_dims,

                # FFN 中间层维度
                feedforward_channels=embed_dims * 4,

                # FC 层数量
                num_fcs=2,

                # FFN dropout
                ffn_drop=drop_out,

                # 激活函数 ReLU
                act_cfg=dict(type="ReLU", inplace=True),
            ),

            # deformable feature aggregation 配置
            deformable_model=dict(
                # 模块类型
                type="DeformableFeatureAggregation",

                # embedding 维度
                embed_dims=embed_dims,

                # group 数
                num_groups=num_groups,

                # FPN 特征层数量
                num_levels=num_levels,

                # nuScenes 使用 6 个相机
                num_cams=6,

                # attention dropout
                attn_drop=0.15,

                # 是否使用自定义 CUDA deformable function
                use_deformable_func=use_deformable_func,

                # 是否使用 camera embedding
                # 用于区分不同相机视角
                use_camera_embed=True,

                # residual 连接方式
                # cat 表示拼接后再融合
                residual_mode="cat",

                # 关键点生成器
                # 对每个 3D box anchor 生成若干 3D 采样点，再投影到图像特征上取特征
                kps_generator=dict(
                    # 基于 3D box 的关键点生成器
                    type="SparseBox3DKeyPointsGenerator",

                    # 可学习关键点数量
                    num_learnable_pts=6,

                    # 固定关键点相对 box 中心的偏移
                    fix_scale=[
                        # box 中心点
                        [0, 0, 0],

                        # x 正方向偏移
                        [0.45, 0, 0],

                        # x 负方向偏移
                        [-0.45, 0, 0],

                        # y 正方向偏移
                        [0, 0.45, 0],

                        # y 负方向偏移
                        [0, -0.45, 0],

                        # z 正方向偏移
                        [0, 0, 0.45],

                        # z 负方向偏移
                        [0, 0, -0.45],
                    ],
                ),
            ),

            # refine layer 配置
            refine_layer=dict(
                # 3D box refinement 模块
                type="SparseBox3DRefinementModule",

                # embedding 维度
                embed_dims=embed_dims,

                # 类别数量
                num_cls=num_classes,

                # 是否细化 yaw 角
                refine_yaw=True,

                # 是否输出质量估计
                with_quality_estimation=with_quality_estimation,
            ),

            # detection target sampler 配置
            sampler=dict(
                # 3D box target 分配器
                type="SparseBox3DTarget",

                # denoising group 数
                # 这里 stage2 设置为 0，表示不使用 DN 训练
                num_dn_groups=0,

                # temporal denoising group 数
                num_temp_dn_groups=0,

                # DN 噪声尺度
                # 前 3 维一般对应 xyz，后面对应尺寸、yaw、速度等
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,

                # 每个样本最多加入多少个 GT 做 DN
                max_dn_gt=32,

                # 是否加入负样本 DN
                add_neg_dn=True,

                # 分类匹配权重
                cls_weight=2.0,

                # box 匹配权重
                box_weight=0.25,

                # 各个回归维度的匹配权重
                # 前 3 维位置权重大，后 3 维尺寸或角度较小，最后 4 维不参与匹配
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,

                # 按类别单独设置回归权重
                cls_wise_reg_weights={
                    # traffic_cone 类别比较特殊
                    # 对速度等维度不强监督
                    class_names.index("traffic_cone"): [
                        # x 权重
                        2.0,

                        # y 权重
                        2.0,

                        # z 权重
                        2.0,

                        # 尺寸/角度等权重
                        1.0,
                        1.0,
                        1.0,

                        # 某些维度不参与，例如速度
                        0.0,
                        0.0,

                        # 其他维度权重
                        1.0,
                        1.0,
                    ],
                },
            ),

            # 检测分类 loss
            loss_cls=dict(
                # FocalLoss 常用于类别不平衡检测任务
                type="FocalLoss",

                # sigmoid 多标签形式
                use_sigmoid=True,

                # focal loss gamma
                gamma=2.0,

                # focal loss alpha
                alpha=0.25,

                # 分类 loss 权重
                loss_weight=2.0,
            ),

            # 检测回归 loss
            loss_reg=dict(
                # SparseDrive 自定义 3D box loss
                type="SparseBox3DLoss",

                # box L1 loss
                loss_box=dict(type="L1Loss", loss_weight=0.25),

                # centerness loss
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),

                # yawness loss
                loss_yawness=dict(type="GaussianFocalLoss"),

                # barrier 类别允许反向
                # 因为 barrier 方向正反可能等价
                cls_allow_reverse=[class_names.index("barrier")],
            ),

            # 检测 decoder
            # 将网络输出解码成 3D box 结果
            decoder=dict(type="SparseBox3DDecoder"),

            # 训练回归 loss 时各维度权重
            # 前 3 维位置权重 2.0，其余 7 维权重 1.0
            reg_weights=[2.0] * 3 + [1.0] * 7,
        ),

        # ===================== map head =====================
        # 地图元素检测头配置
        map_head=dict(
            # 地图头也复用 Sparse4DHead
            type="Sparse4DHead",

            # 分类阈值
            cls_threshold_to_reg=0.05,

            # map head 不使用 decouple attention
            decouple_attn=decouple_attn_map,

            # map instance bank
            instance_bank=dict(
                # instance bank 类型
                type="InstanceBank",

                # 地图 anchor 数量
                num_anchor=100,

                # embedding 维度
                embed_dims=embed_dims,

                # 地图 kmeans anchor 文件
                anchor="data/kmeans/kmeans_map_100.npy",

                # 地图点关键点生成器
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),

                # 时序保留的地图 instance 数
                # temporal_map=True 时为 33
                num_temp_instances=33 if temporal_map else -1,

                # 历史置信度衰减
                confidence_decay=0.6,

                # map 特征是否保留梯度
                feat_grad=True,
            ),

            # 地图 anchor 编码器
            anchor_encoder=dict(
                # 稀疏点编码器
                type="SparsePoint3DEncoder",

                # embedding 维度
                embed_dims=embed_dims,

                # 每条线的采样点数
                num_sample=num_sample,
            ),

            # 地图单帧 decoder 层数
            num_single_frame_decoder=num_single_frame_decoder_map,

            # map decoder 操作顺序
            operation_order=(
                # 单帧层操作
                [
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * num_single_frame_decoder_map

                # 时序层操作
                + [
                    "temp_gnn",
                    "gnn",
                    "norm",
                    "deformable",
                    "ffn",
                    "norm",
                    "refine",
                ]
                * (num_decoder - num_single_frame_decoder_map)

            # [:] 表示保留完整列表
            )[:],

            # map temporal graph model
            temp_graph_model=dict(
                # FlashAttention
                type="MultiheadFlashAttention",

                # map 不解耦时就是 256
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,

                # attention head 数
                num_heads=num_groups,

                # batch first
                batch_first=True,

                # dropout
                dropout=drop_out,
            )
            # temporal_map=True 时启用
            if temporal_map
            # 否则不使用
            else None,

            # map 当前帧 graph model
            graph_model=dict(
                # FlashAttention
                type="MultiheadFlashAttention",

                # embedding 维度
                embed_dims=embed_dims if not decouple_attn_map else embed_dims * 2,

                # head 数
                num_heads=num_groups,

                # batch first
                batch_first=True,

                # dropout
                dropout=drop_out,
            ),

            # LayerNorm
            norm_layer=dict(type="LN", normalized_shape=embed_dims),

            # map FFN
            ffn=dict(
                # 非对称 FFN
                type="AsymmetricFFN",

                # 输入通道
                in_channels=embed_dims * 2,

                # 前置 LN
                pre_norm=dict(type="LN"),

                # 输出 embedding 维度
                embed_dims=embed_dims,

                # FFN 中间维度
                feedforward_channels=embed_dims * 4,

                # FC 层数量
                num_fcs=2,

                # dropout
                ffn_drop=drop_out,

                # ReLU 激活
                act_cfg=dict(type="ReLU", inplace=True),
            ),

            # map deformable feature aggregation
            deformable_model=dict(
                # deformable 聚合模块
                type="DeformableFeatureAggregation",

                # embedding 维度
                embed_dims=embed_dims,

                # group 数
                num_groups=num_groups,

                # feature level 数
                num_levels=num_levels,

                # nuScenes 6 相机
                num_cams=6,

                # attention dropout
                attn_drop=0.15,

                # 使用自定义 deformable function
                use_deformable_func=use_deformable_func,

                # 使用 camera embedding
                use_camera_embed=True,

                # residual 拼接模式
                residual_mode="cat",

                # 地图点关键点生成器
                kps_generator=dict(
                    # 稀疏 3D 点关键点生成器
                    type="SparsePoint3DKeyPointsGenerator",

                    # embedding 维度
                    embed_dims=embed_dims,

                    # 每条地图线采样 20 个点
                    num_sample=num_sample,

                    # 可学习点数量
                    num_learnable_pts=3,

                    # 固定高度采样
                    # 因为地图线在地面附近，但为了投影到图像，会在不同高度采样增强鲁棒性
                    fix_height=(0, 0.5, -0.5, 1, -1),

                    # lidar 坐标系下的地面高度
                    ground_height=-1.84023, # ground height in lidar frame
                ),
            ),

            # map refine layer
            refine_layer=dict(
                # 稀疏点 refinement 模块
                type="SparsePoint3DRefinementModule",

                # embedding 维度
                embed_dims=embed_dims,

                # 采样点数量
                num_sample=num_sample,

                # 地图类别数
                num_cls=num_map_classes,
            ),

            # map target sampler
            sampler=dict(
                # 稀疏点目标分配器
                type="SparsePoint3DTarget",

                # 匈牙利匹配 assigner
                assigner=dict(
                    # 线实例匹配器
                    type='HungarianLinesAssigner',

                    # 匹配代价
                    cost=dict(
                        # 地图 query 的综合 cost
                        type='MapQueriesCost',

                        # 分类代价
                        cls_cost=dict(type='FocalLossCost', weight=1.0),

                        # 线回归代价
                        # permute=True 表示考虑点序排列的等价性
                        reg_cost=dict(type='LinesL1Cost', weight=10.0, beta=0.01, permute=True),
                    ),
                ),

                # 地图类别数
                num_cls=num_map_classes,

                # 每条线采样点数量
                num_sample=num_sample,

                # ROI 范围
                roi_size=roi_size,
            ),

            # map 分类 loss
            loss_cls=dict(
                # FocalLoss
                type="FocalLoss",

                # sigmoid 形式
                use_sigmoid=True,

                # gamma
                gamma=2.0,

                # alpha
                alpha=0.25,

                # loss 权重
                loss_weight=1.0,
            ),

            # map 回归 loss
            loss_reg=dict(
                # 稀疏线 loss
                type="SparseLineLoss",

                # 线 L1 loss
                loss_line=dict(
                    # 线段点序列 L1 loss
                    type='LinesL1Loss',

                    # loss 权重
                    loss_weight=10.0,

                    # smooth L1 beta 或线 loss 内部参数
                    beta=0.01,
                ),

                # 每条线采样点数
                num_sample=num_sample,

                # ROI 范围
                roi_size=roi_size,
            ),

            # map decoder
            decoder=dict(type="SparsePoint3DDecoder"),

            # map 回归权重
            # 20 个点，每个点 x,y 两维，所以 40 个回归量
            reg_weights=[1.0] * 40,

            # 地图 GT 类别字段名
            gt_cls_key="gt_map_labels",

            # 地图 GT 点字段名
            gt_reg_key="gt_map_pts",

            # 地图实例 ID 字段名
            gt_id_key="map_instance_id",

            # 是否使用 instance id
            with_instance_id=False,

            # 任务前缀
            # loss 或结果字段可能会加 map 前缀
            task_prefix='map',
        ),

        # ===================== motion and planning head =====================
        # 运动预测和自车规划头配置
        motion_plan_head=dict(
            # 模块类型
            type='MotionPlanningHead',

            # agent 未来预测时间步
            fut_ts=fut_ts,

            # agent 轨迹模态数
            fut_mode=fut_mode,

            # ego 未来规划时间步
            ego_fut_ts=ego_fut_ts,

            # ego 规划模态数
            ego_fut_mode=ego_fut_mode,

            # agent motion anchor 文件
            # 例如 data/kmeans/kmeans_motion_6.npy
            motion_anchor=f'data/kmeans/kmeans_motion_{fut_mode}.npy',

            # ego planning anchor 文件
            # 例如 data/kmeans/kmeans_plan_6.npy
            plan_anchor=f'data/kmeans/kmeans_plan_{ego_fut_mode}.npy',

            # embedding 维度
            embed_dims=embed_dims,

            # motion planning 是否使用解耦 attention
            decouple_attn=decouple_attn_motion,

            # instance queue 配置
            # 用于缓存历史检测 instance，供 motion prediction 使用
            instance_queue=dict(
                # 队列类型
                type="InstanceQueue",

                # embedding 维度
                embed_dims=embed_dims,

                # 队列长度
                queue_length=queue_length,

                # tracking 阈值
                # 低于这个分数的目标可能不加入跟踪/运动预测
                tracking_threshold=0.2,

                # 特征图尺度
                # input_shape[1]/strides[-1] = 256/32 = 8
                # input_shape[0]/strides[-1] = 704/32 = 22
                # 所以最后层特征图大约是 8 x 22
                feature_map_scale=(input_shape[1]/strides[-1], input_shape[0]/strides[-1]),
            ),

            # motion planning decoder 操作顺序
            operation_order=(
                [
                    # temporal GNN，融合历史 instance
                    "temp_gnn",

                    # agent/query 间交互
                    "gnn",

                    # LayerNorm
                    "norm",

                    # cross_gnn，通常用于 agent、map、ego 或不同 query 间交叉交互
                    "cross_gnn",

                    # LayerNorm
                    "norm",

                    # FFN
                    "ffn",                    

                    # LayerNorm
                    "norm",
                ] * 3 +

                [
                    # 最后 refine 输出 motion 和 planning 结果
                    "refine",
                ]
            ),

            # temporal graph model
            temp_graph_model=dict(
                # 这里使用普通 MultiheadAttention，不是 FlashAttention
                type="MultiheadAttention",

                # 解耦时输入维度为 512
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,

                # attention head 数
                num_heads=num_groups,

                # batch first
                batch_first=True,

                # dropout
                dropout=drop_out,
            ),

            # graph model
            graph_model=dict(
                # 使用 FlashAttention
                type="MultiheadFlashAttention",

                # 解耦时输入维度为 512
                embed_dims=embed_dims if not decouple_attn_motion else embed_dims * 2,

                # head 数
                num_heads=num_groups,

                # batch first
                batch_first=True,

                # dropout
                dropout=drop_out,
            ),

            # cross graph model
            cross_graph_model=dict(
                # 使用 FlashAttention
                type="MultiheadFlashAttention",

                # cross attention 这里使用 256 维
                embed_dims=embed_dims,

                # head 数
                num_heads=num_groups,

                # batch first
                batch_first=True,

                # dropout
                dropout=drop_out,
            ),

            # LayerNorm
            norm_layer=dict(type="LN", normalized_shape=embed_dims),

            # motion planning FFN
            ffn=dict(
                # 非对称 FFN
                type="AsymmetricFFN",

                # 输入通道为 256
                in_channels=embed_dims,

                # 前置 LN
                pre_norm=dict(type="LN"),

                # 输出 embedding 维度
                embed_dims=embed_dims,

                # FFN 中间维度是 2 倍 embedding
                feedforward_channels=embed_dims * 2,

                # FC 层数量
                num_fcs=2,

                # dropout
                ffn_drop=drop_out,

                # ReLU
                act_cfg=dict(type="ReLU", inplace=True),
            ),

            # motion planning refinement 模块
            refine_layer=dict(
                # refinement 类型
                type="MotionPlanningRefinementModule",

                # embedding 维度
                embed_dims=embed_dims,

                # agent 预测时间步
                fut_ts=fut_ts,

                # agent 模态数
                fut_mode=fut_mode,

                # ego 规划时间步
                ego_fut_ts=ego_fut_ts,

                # ego 规划模态数
                ego_fut_mode=ego_fut_mode,
            ),

            # motion target sampler
            motion_sampler=dict(
                # 运动预测目标分配器
                type="MotionTarget",
            ),

            # motion 分类 loss
            # 用于预测多个 future mode 中哪个更接近 GT
            motion_loss_cls=dict(
                # FocalLoss
                type='FocalLoss',

                # sigmoid
                use_sigmoid=True,

                # gamma
                gamma=2.0,

                # alpha
                alpha=0.25,

                # loss 权重
                loss_weight=0.2
            ),

            # motion 轨迹回归 loss
            motion_loss_reg=dict(type='L1Loss', loss_weight=0.2),

            # planning target sampler
            planning_sampler=dict(
                # 自车规划目标分配器
                type="PlanningTarget",

                # ego 未来时间步
                ego_fut_ts=ego_fut_ts,

                # ego 轨迹模态数
                ego_fut_mode=ego_fut_mode,
            ),

            # planning 分类 loss
            # 用于选择自车规划候选模态
            plan_loss_cls=dict(
                # FocalLoss
                type='FocalLoss',

                # sigmoid
                use_sigmoid=True,

                # gamma
                gamma=2.0,

                # alpha
                alpha=0.25,

                # loss 权重
                loss_weight=0.5,
            ),

            # planning 轨迹回归 loss
            plan_loss_reg=dict(type='L1Loss', loss_weight=1.0),

            # planning status loss
            # ego_status 可能包括速度、加速度、命令状态等
            plan_loss_status=dict(type='L1Loss', loss_weight=1.0),

            # agent motion decoder
            motion_decoder=dict(type="SparseBox3DMotionDecoder"),

            # ego planning decoder
            planning_decoder=dict(
                # 分层规划 decoder
                type="HierarchicalPlanningDecoder",

                # ego 时间步
                ego_fut_ts=ego_fut_ts,

                # ego 模态数
                ego_fut_mode=ego_fut_mode,

                # 是否重新打分
                use_rescore=True,
            ),

            # 用于 motion/planning 的检测目标数量
            # 通常取 top 50 detected agents
            num_det=50,

            # 用于 planning 的地图元素数量
            # 通常取 top 10 map elements
            num_map=10,
        ),
    ),
)



# =============================== 三、data ===============================
# 这一部分配置数据集类型、数据路径、训练/测试 pipeline、数据增强、DataLoader 等。


# 数据集类型
# 对应 SparseDrive 自定义或继承的 NuScenes3DDataset
dataset_type = "NuScenes3DDataset"

# nuScenes 数据根目录
data_root = "data/nuscenes/"

# 标注信息目录
# 如果 version 是 trainval，使用 data/infos/
# 如果 version 是 mini，使用 data/infos/mini/
# 当前 version 最终 trainval，使用 data/infos/
# 如果 version 是 mini，使用 data/infos为 trainval，所以 anno_root = "data/infos/"
anno_root = "data/infos/" if version == 'trainval' else "data/infos/mini/"

# 文件读取后端
# backend="disk" 表示从本地磁盘读取
file_client_args = dict(backend="disk")


# 图像归一化配置
img_norm_cfg = dict(
    # 图像均值
    # 常见 ImageNet RGB 均值
    mean=[123.675, 116.28, 103.53],

    # 图像标准差
    std=[58.395, 57.12, 57.375],

    # 是否把 BGR 转成 RGB
    to_rgb=True
)

# 训练数据处理流水线
train_pipeline = [
    # 加载多相机图像
    # to_float32=True 表示转成 float32
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),

    # 加载点云
    # 虽然模型输入主要是 camera，但训练时可能需要点云生成深度监督或辅助信息
    dict(
        # 从文件读取点云
        type="LoadPointsFromFile",

        # 点云坐标系为 LIDAR
        coord_type="LIDAR",

        # 原始点云每个点加载 5 维
        load_dim=5,

        # 实际使用 5 维
        use_dim=5,

        # 文件读取后端
        file_client_args=file_client_args,
    ),

    # 图像 resize、crop、flip 数据增强
    dict(type="ResizeCropFlipImage"),

    # 多尺度深度图生成器
    # 用点云投影到图像生成深度监督
    dict(
        # 生成多尺度 depth map
        type="MultiScaleDepthMapGenerator",

        # 下采样尺度
        # strides[:num_depth_layers] = [4, 8, 16]
        downsample=strides[:num_depth_layers],
    ),

    # 旋转 3D bbox
    # 用于保持图像增强和 3D 标注一致
    dict(type="BBoxRotation"),

    # 多视角图像光度增强
    # 如亮度、对比度、饱和度变化
    dict(type="PhotoMetricDistortionMultiViewImage"),

    # 图像归一化
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),

    # 基于距离范围过滤 3D 目标
    dict(
        # 圆形范围过滤
        type="CircleObjectRangeFilter",

        # 每个类别的最大距离阈值都是 55m
        class_dist_thred=[55] * len(class_names),
    ),

    # 过滤类别，只保留 class_names 中定义的类别
    dict(type="InstanceNameFilter", classes=class_names),

    # 矢量化地图
    dict(
        # 地图矢量化模块
        type='VectorizeMap',

        # ROI 范围
        roi_size=roi_size,

        # 训练时不简化线
        simplify=False,

        # 不归一化坐标
        normalize=False,

        # 每条地图线采样 20 个点
        sample_num=num_sample,

        # 是否允许点序排列变换
        # 对线段来说，正序和反序可能等价
        permute=True,
    ),

    # Sparse4D/SparseDrive 数据适配器
    # 把 nuScenes 数据整理成模型需要的字段格式
    dict(type="NuScenesSparse4DAdaptor"),

    # 收集模型训练需要的字段
    dict(
        # Collect 是 MMDetection 数据 pipeline 中的字段收集器
        type="Collect",

        # 放入 data batch 的主要字段
        keys=[
            # 多视角图像
            "img",

            # 时间戳
            "timestamp",

            # 相机投影矩阵
            "projection_mat",

            # 图像宽高
            "image_wh",

            # 深度 GT，用于 DenseDepthNet 辅助监督
            "gt_depth",

            # 相机焦距
            "focal",

            # 3D bbox GT
            "gt_bboxes_3d",

            # 3D bbox 类别 GT
            "gt_labels_3d",

            # 地图类别 GT
            'gt_map_labels', 

            # 地图点序列 GT
            'gt_map_pts',

            # 其他 agent 未来轨迹 GT
            'gt_agent_fut_trajs',

            # agent 未来轨迹 mask
            # 表示哪些未来时间步有效
            'gt_agent_fut_masks',

            # ego 自车未来轨迹 GT
            'gt_ego_fut_trajs',

            # ego 自车未来轨迹 mask
            'gt_ego_fut_masks',

            # ego 高层驾驶命令
            # 例如左转、右转、直行等
            'gt_ego_fut_cmd',

            # ego 状态
            # 例如速度、加速度、角速度等，具体取决于数据适配器实现
            'ego_status',
        ],

        # 元信息字段
        meta_keys=[
            # 从当前帧到全局坐标系的变换
            "T_global",

            # 从全局坐标系到当前帧的逆变换
            "T_global_inv",

            # 时间戳
            "timestamp",

            # 实例 ID，用于 tracking 或 temporal association
            "instance_id"
        ],
    ),
]

# 测试 pipeline
test_pipeline = [
    # 加载多视角图像
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),

    # resize、crop、flip
    # 测试时一般是确定性增强，不是随机增强
    dict(type="ResizeCropFlipImage"),

    # 图像归一化
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),

    # 数据适配器
    dict(type="NuScenesSparse4DAdaptor"),

    # 收集测试所需字段
    dict(
        # Collect
        type="Collect",

        # 测试输入字段
        keys=[
            # 图像
            "img",

            # 时间戳
            "timestamp",

            # 投影矩阵
            "projection_mat",

            # 图像宽高
            "image_wh",

            # ego 状态
            'ego_status',

            # ego 高层命令
            'gt_ego_fut_cmd',
        ],

        # 测试元信息
        meta_keys=[
            # 当前到全局
            "T_global",

            # 全局到当前
            "T_global_inv",

            # 时间戳
            "timestamp"
        ],
    ),
]

# 评估 pipeline
# 用于把 GT 转换成评估需要的格式
eval_pipeline = [
    # 范围过滤目标
    dict(
        # 圆形范围过滤
        type="CircleObjectRangeFilter",

        # 55m 范围
        class_dist_thred=[55] * len(class_names),
    ),

    # 过滤类别
    dict(type="InstanceNameFilter", classes=class_names),

    # 矢量化地图
    dict(
        # VectorizeMap
        type='VectorizeMap',

        # ROI 范围
        roi_size=roi_size,

        # 评估时 simplify=True，可能对地图线进行简化
        simplify=True,

        # 不归一化
        normalize=False,
    ),

    # 收集评估字段
    dict(
        # Collect
        type='Collect', 

        # 评估所需 keys
        keys=[
            # 矢量地图 GT
            'vectors',

            # 3D bbox GT
            "gt_bboxes_3d",

            # 3D label GT
            "gt_labels_3d",

            # agent 未来轨迹
            'gt_agent_fut_trajs',

            # agent 未来 mask
            'gt_agent_fut_masks',

            # ego 未来轨迹
            'gt_ego_fut_trajs',

            # ego 未来 mask
            'gt_ego_fut_masks', 

            # ego 未来 command
            'gt_ego_fut_cmd',

            # 未来 boxes
            'fut_boxes'
        ],

        # 评估元信息
        meta_keys=[
            # nuScenes sample token
            'token',

            # 时间戳
            'timestamp'
        ]
    ),
]


# 输入模态配置
input_modality = dict(
    # 不使用 lidar 作为模型输入
    use_lidar=False,

    # 使用 camera 作为模型输入
    use_camera=True,

    # 不使用 radar
    use_radar=False,

    # 不使用外部地图作为输入
    # 注意这里的 use_map=False 不代表不做 map prediction
    # 它表示不把外部 HD map 作为输入模态
    use_map=False,

    # 不使用其他外部信息
    use_external=False,
)


# 数据集基础配置
data_basic_config = dict(
    # 数据集类型
    type=dataset_type,

    # 数据根目录
    data_root=data_root,

    # 检测类别
    classes=class_names,

    # 地图类别
    map_classes=map_class_names,

    # 输入模态
    modality=input_modality,

    # nuScenes 版本
    version="v1.0-trainval",
)

# 评估配置
eval_config = dict(
    # 展开基础数据集配置
    **data_basic_config,

    # 验证集 info 文件
    ann_file=anno_root + 'nuscenes_infos_val.pkl',

    # 评估 pipeline
    pipeline=eval_pipeline,

    # 测试模式
    test_mode=True,
)

# 数据增强配置
data_aug_conf = {
    # resize 缩放比例范围
    "resize_lim": (0.40, 0.47),

    # 最终图像尺寸
    # input_shape=(704,256)，[::-1] 后为 (256,704)
    # pipeline 里通常使用 final_dim=(H,W)
    "final_dim": input_shape[::-1],

    # bottom crop 百分比范围
    # 0 表示不从底部裁剪
    "bot_pct_lim": (0.0, 0.0),

    # 图像旋转角度范围，单位通常是 degree
    "rot_lim": (-5.4, 5.4),

    # 原始图像高度
    "H": 900,

    # 原始图像宽度
    "W": 1600,

    # 是否随机水平翻转
    "rand_flip": True,

    # 3D 旋转增强范围
    # [0,0] 表示不做 3D 旋转增强
    "rot3d_range": [0, 0],
}


# DataLoader 和 train/val/test 数据集配置
data = dict(
    # 每张 GPU 的 batch size
    # 前面算出来是 6
    samples_per_gpu=batch_size,

    # 每张 GPU 的 dataloader worker 数
    # 这里也设为 6
    workers_per_gpu=batch_size,

    # 训练集配置
    train=dict(
        # 展开基础配置
        **data_basic_config,

        # 训练集 info 文件
        ann_file=anno_root + "nuscenes_infos_train.pkl",

        # 训练 pipeline
        pipeline=train_pipeline,

        # 训练模式
        test_mode=False,

        # 数据增强配置
        data_aug_conf=data_aug_conf,

        # 是否使用序列标志
        # temporal 模型需要知道样本序列关系
        with_seq_flag=True,

        # 把序列切成几段
        # 用于多卡训练或数据采样时减少长序列依赖
        sequences_split_num=2,

        # 是否保持同一序列内的数据增强一致
        # 对 temporal 模型很重要，否则历史帧和当前帧增强不一致会破坏时序几何关系
        keep_consistent_seq_aug=True,
    ),

    # 验证集配置
    val=dict(
        # 展开基础配置
        **data_basic_config,

        # 验证集 info 文件
        ann_file=anno_root + "nuscenes_infos_val.pkl",

        # 测试 pipeline
        pipeline=test_pipeline,

        # 数据增强配置
        data_aug_conf=data_aug_conf,

        # 验证时 test_mode=True
        test_mode=True,

        # 评估配置
        eval_config=eval_config,
    ),

    # 测试集配置
    test=dict(
        # 展开基础配置
        **data_basic_config,

        # 这里测试也用 val info
        ann_file=anno_root + "nuscenes_infos_val.pkl",

        # 测试 pipeline
        pipeline=test_pipeline,

        # 数据增强配置
        data_aug_conf=data_aug_conf,

        # 测试模式
        test_mode=True,

        # 评估配置
        eval_config=eval_config,
    ),
)



# =============================== 四、training ===============================
# 这一部分配置优化器、梯度裁剪、学习率策略和 runner。


# 优化器配置
optimizer = dict(
    # 使用 AdamW
    type="AdamW",

    # 基础学习率
    lr=3e-4,

    # weight decay
    # 用于正则化权重，抑制过拟合
    weight_decay=0.001,

    # 按参数名设置不同学习率
    paramwise_cfg=dict(
        # 自定义参数规则
        custom_keys={
            # 对 img_backbone 使用 0.1 倍学习率
            # backbone 有 ImageNet 预训练权重，所以学习率小一点，避免破坏预训练特征
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
)

# optimizer hook 配置
optimizer_config = dict(
    # 梯度裁剪
    grad_clip=dict(
        # 梯度范数最大为 25
        max_norm=25,

        # 使用 L2 norm
        norm_type=2
    )
)

# 学习率配置
lr_config = dict(
    # 使用余弦退火学习率
    policy="CosineAnnealing",

    # warmup 采用线性 warmup
    warmup="linear",

    # warmup iteration 数
    warmup_iters=500,

    # warmup 初始学习率比例
    # 初始 lr = base_lr * 1/3
    warmup_ratio=1.0 / 3,

    # 最小学习率比例
    # 最低 lr = base_lr * 1e-3
    min_lr_ratio=1e-3,
)

# runner 配置
runner = dict(
    # 使用按 iteration 训练的 runner
    type="IterBasedRunner",

    # 最大 iteration 数
    # trainval 下 num_iters_per_epoch=586
    # num_epochs=10
    # max_iters=5860
    max_iters=num_iters_per_epoch * num_epochs,
)



# =============================== 五、eval ===============================
# 这一部分配置评估任务开关和评估间隔。


# 评估模式
eval_mode = dict(
    # 评估 detection
    with_det=True,

    # 评估 tracking
    with_tracking=True,

    # 评估 map
    with_map=True,

    # 评估 motion prediction
    with_motion=True,

    # 评估 planning
    with_planning=True,

    # tracking 分数阈值
    tracking_threshold=0.2,

    # motion 分数阈值
    # 注意这里变量名写成了 motion_threshhold，多了一个 h
    # 如果评估代码里也是按这个字段读取，就没有问题
    # 如果代码期望 motion_threshold，则可能读不到
    motion_threshhold=0.2,
)

# evaluation hook 配置
evaluation = dict(
    # 评估间隔
    # num_iters_per_epoch * checkpoint_epoch_interval = 5860
    # 也就是训练结束时评估一次
    interval=num_iters_per_epoch*checkpoint_epoch_interval,

    # 传入评估任务开关
    eval_mode=eval_mode,
)

# =============================== pretrained model ===============================
# 预训练模型配置

# 加载 stage1 训练好的权重作为 stage2 初始化
# 这说明 sparsedrive_small_stage2.py 是第二阶段训练配置
# stage2 不是从 ImageNet ResNet 直接开始，而是在 SparseDrive stage1 基础上继续训练
load_from = 'ckpt/sparsedrive_stage1.pth'