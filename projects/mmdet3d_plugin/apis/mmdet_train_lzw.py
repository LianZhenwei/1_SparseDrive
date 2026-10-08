# ---------------------------------------------
# OpenMMLab 版权声明
# 表示这个文件最初来自 OpenMMLab/MMDetection 体系
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

# 表示这个文件被 Zhiqi Li 修改过
# Zhiqi Li 是 BEVFormer 等项目相关作者之一，很多 SparseDrive 代码风格继承自 BEVFormer
#  Modified by Zhiqi Li
# ---------------------------------------------


# 导入 random
# 当前文件里没有实际使用，属于冗余导入或历史遗留
import random

# 导入 warnings
# 用于发出警告，例如配置里缺少 runner 或 imgs_per_gpu 已过时
import warnings


# 导入 numpy
# 当前文件中没有实际使用，属于冗余导入
import numpy as np

# 导入 PyTorch
# 用于模型放到 GPU、分布式训练等
import torch

# 导入 torch.distributed
# 当前文件里没有直接使用 dist，属于冗余导入
import torch.distributed as dist

# 从 MMCV 并行模块中导入两个模型包装器
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

# 从 MMCV runner 模块中导入训练相关组件
from mmcv.runner import (
    # HOOKS 是 MMCV 的 hook 注册表
    # 自定义 hook 会通过 build_from_cfg(hook_cfg, HOOKS) 构建
    HOOKS,

    # 分布式训练时，每个 epoch 设置 sampler seed 的 hook
    # 作用是让每个 epoch 的 shuffle 不一样，但各 GPU 之间又能正确同步
    DistSamplerSeedHook,

    # 基于 epoch 的 runner 类型
    # 用于判断是否需要注册 DistSamplerSeedHook
    EpochBasedRunner,

    # fp16 混合精度训练的 optimizer hook
    # SparseDrive 配置里如果有 fp16，就会用它
    Fp16OptimizerHook,

    # 普通 fp32 训练的 optimizer hook
    OptimizerHook,

    # 根据 cfg.optimizer 构建优化器
    # 例如 AdamW、SGD 等
    build_optimizer,

    # 根据 cfg.runner 构建 runner
    # runner 是训练循环控制器
    build_runner,

    # 获取分布式 rank/world_size
    # 当前文件里没有实际使用
    get_dist_info,
)

# 从 MMCV 导入 build_from_cfg
# 根据配置字典和注册表构建对象
# 这里用于构建 custom_hooks
from mmcv.utils import build_from_cfg


# 从 MMDetection 导入普通 EvalHook
# 单卡验证时使用
from mmdet.core import EvalHook


# 从 MMDetection 数据集模块导入 build_dataset 和 replace_ImageToTensor
# build_dataset 当前文件里没有用到，属于冗余导入
# replace_ImageToTensor 用于 val batch_size > 1 时替换 pipeline 组件
from mmdet.datasets import build_dataset, replace_ImageToTensor

# 获取根 logger
# 用于打印训练日志
from mmdet.utils import get_root_logger

# 导入 time
# 用于生成验证结果 jsonfile_prefix 的时间戳
import time

# 导入 os.path 并命名为 osp
# 用于拼接路径
import os.path as osp

# 导入 SparseDrive 自定义 dataloader 构建函数
# 注意：这里不用 mmdet.datasets.build_dataloader，而用项目自定义版本
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

# 导入 SparseDrive 自定义分布式评估 hook
# 分布式验证时使用 CustomDistEvalHook
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import (
    CustomDistEvalHook,
)

# 导入 SparseDrive 自定义 dataset 构建函数
# 用于构建验证集 val_dataset
from projects.mmdet3d_plugin.datasets import custom_build_dataset


def custom_train_detector(
    # model：已经构建好的模型
    # 在外层 tools/train.py 中通常由 build_detector(cfg.model) 得到
    model,

    # dataset：训练数据集
    # 可以是单个 dataset，也可以是 list/tuple
    dataset,

    # cfg：完整配置文件对象
    # 里面包含 model、data、optimizer、runner、lr_config、checkpoint_config 等
    cfg,

    # distributed：是否分布式训练
    # True：多 GPU DDP
    # False：单机普通 DataParallel
    distributed=False,

    # validate：是否在训练过程中做验证
    # 通常由外层 not args.no_validate 决定
    validate=False,

    # timestamp：外层生成的时间戳
    # 用于让 .log 和 .log.json 文件名一致
    timestamp=None,

    # meta：保存环境信息、seed、config 等元信息
    # checkpoint 里也可能保存这些内容
    meta=None,
):
    # 获取 logger
    # cfg.log_level 通常是 "INFO"
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    # 准备 dataloader


    # 如果 dataset 已经是 list 或 tuple，就保持不变
    # 如果只是单个 dataset，就包装成 list
    # 因为后面统一用 for ds in dataset 来构建 data_loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # 原作者可能想 assert len(dataset)==1，但写错/注释掉了
    # assert len(dataset)==1s


    # 如果配置里使用了旧字段 imgs_per_gpu
    if "imgs_per_gpu" in cfg.data:

        # 打印警告：imgs_per_gpu 在 MMDet V2.0 中已经废弃
        # 推荐使用 samples_per_gpu
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )

        # 如果配置里同时有 samples_per_gpu
        if "samples_per_gpu" in cfg.data:

            # 打印警告：两个都写了，但本实验会优先使用 imgs_per_gpu
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )

        # 如果配置里没有 samples_per_gpu
        else:

            # 打印警告：自动把 samples_per_gpu 设置成 imgs_per_gpu
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )

        # 兼容旧配置：
        # 把 cfg.data.samples_per_gpu 设置成 cfg.data.imgs_per_gpu
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu


    # 如果配置文件里有 runner 字段
    if "runner" in cfg:

        # 取 runner 类型
        # SparseDrive 通常是 IterBasedRunner
        # 也可能是 EpochBasedRunner
        runner_type = cfg.runner["type"]

    # 如果配置文件里没有 runner
    else:

        # 默认使用 EpochBasedRunner
        runner_type = "EpochBasedRunner"


    # 构建训练 dataloader 列表
    # dataset 可能有一个，也可能有多个
    data_loaders = [
        build_dataloader(
            # 当前 dataset
            ds,

            # 每张 GPU 的 batch size
            cfg.data.samples_per_gpu,

            # 每张 GPU 的 dataloader worker 数
            cfg.data.workers_per_gpu,

            # GPU 数量
            # 注释说：如果是 distributed，这个参数会被忽略
            # 非分布式时用于确定总 GPU 数
            len(cfg.gpu_ids),

            # 是否分布式训练
            dist=distributed,

            # 随机 seed
            seed=cfg.seed,

            # 非 shuffle sampler 配置
            # 这里指定 DistributedSampler
            # 在分布式场景下保证不同 rank 拿到不同数据
            nonshuffler_sampler=dict(
                type="DistributedSampler"
            ),  # dict(type='DistributedSampler'),

            # runner 类型
            # dataloader 可能根据 IterBasedRunner/EpochBasedRunner 做不同处理
            runner_type=runner_type,
        )

        # 对 dataset 列表里的每个 ds 构建 dataloader
        for ds in dataset
    ]


    # put model on gpus
    # 把模型放到 GPU 上，并根据是否分布式进行包装

    # 如果是分布式训练
    if distributed:

        # 从 cfg 中读取 find_unused_parameters
        # 如果模型里有些参数在某些 forward 中没有参与 loss，需要设 True
        # 但 True 会稍微降低 DDP 性能
        find_unused_parameters = cfg.get("find_unused_parameters", False)

        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        # 用 MMDistributedDataParallel 包装模型
        model = MMDistributedDataParallel(
            # 先把模型放到当前 GPU
            model.cuda(),

            # 当前进程使用的 GPU
            # 分布式训练一般是一个进程对应一张 GPU
            device_ids=[torch.cuda.current_device()],

            # 不广播 buffer
            # BN 的 running_mean/running_var 这类 buffer 不在每次 forward 广播
            broadcast_buffers=False,

            # 是否查找未使用参数
            find_unused_parameters=find_unused_parameters,
        )

    # 如果不是分布式训练
    else:

        # 用普通 MMDataParallel 包装模型
        model = MMDataParallel(
            # 把模型放到第一张 GPU
            model.cuda(cfg.gpu_ids[0]),

            # 使用 cfg.gpu_ids 指定的 GPU
            device_ids=cfg.gpu_ids
        )


    # build runner
    # 构建优化器和 runner

    # 根据 cfg.optimizer 构建优化器
    # 例如：
    # optimizer = AdamW(model.parameters(), lr=..., weight_decay=...)
    optimizer = build_optimizer(model, cfg.optimizer)


    # 如果配置里没有 runner 字段
    if "runner" not in cfg:

        # 兼容老版本配置：
        # 根据 cfg.total_epochs 自动构造一个 EpochBasedRunner 配置
        cfg.runner = {
            "type": "EpochBasedRunner",
            "max_epochs": cfg.total_epochs,
        }

        # 发出警告，提示新版本配置应显式写 runner
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )

    # 如果配置里有 runner
    else:

        # 如果同时还有 total_epochs 字段
        if "total_epochs" in cfg:

            # 要求 total_epochs 和 runner.max_epochs 一致
            # 防止配置冲突
            assert cfg.total_epochs == cfg.runner.max_epochs


    # 根据 cfg.runner 构建 runner
    # runner 是 MMCV 训练循环的核心控制器
    runner = build_runner(
        # runner 配置
        cfg.runner,

        # 默认参数
        default_args=dict(
            # 包装后的模型
            model=model,

            # 优化器
            optimizer=optimizer,

            # 工作目录
            work_dir=cfg.work_dir,

            # logger
            logger=logger,

            # 元信息
            meta=meta,
        ),
    )


    # an ugly workaround to make .log and .log.json filenames the same
    # 一个不太优雅的 workaround：
    # 让 runner.timestamp 使用外层传入的 timestamp
    # 这样普通 log 和 json log 文件名能对齐
    runner.timestamp = timestamp


    # fp16 setting
    # 处理 fp16 混合精度训练配置

    # 从配置里读取 fp16 配置
    fp16_cfg = cfg.get("fp16", None)

    # 如果配置了 fp16
    if fp16_cfg is not None:

        # 构建 fp16 optimizer hook
        # 它会负责 loss scale、反向传播、梯度裁剪、optimizer.step 等
        optimizer_config = Fp16OptimizerHook(
            # cfg.optimizer_config 里可能有 grad_clip 等配置
            **cfg.optimizer_config,

            # fp16_cfg 里可能有 loss_scale 等配置
            **fp16_cfg,

            # 是否分布式
            distributed=distributed
        )

    # 如果没有 fp16，并且是分布式训练，同时 optimizer_config 里没写 type
    elif distributed and "type" not in cfg.optimizer_config:

        # 构建普通 OptimizerHook
        optimizer_config = OptimizerHook(**cfg.optimizer_config)

    # 其他情况
    else:

        # 直接使用 cfg.optimizer_config
        optimizer_config = cfg.optimizer_config


    # register hooks
    # 注册训练过程中要用的 hook

    runner.register_training_hooks(
        # 学习率策略
        # 例如 CosineAnnealing、step lr、warmup 等
        cfg.lr_config,

        # optimizer hook
        # 控制 backward、梯度裁剪、参数更新、fp16 等
        optimizer_config,

        # checkpoint 保存策略
        # 例如 interval=8780
        cfg.checkpoint_config,

        # 日志打印策略
        # 例如 TextLoggerHook、TensorboardLoggerHook
        cfg.log_config,

        # momentum 策略
        # 一般 AdamW 不一定用
        cfg.get("momentum_config", None),
    )


    # register profiler hook
    # 下面是性能 profiler hook 的示例代码，被注释掉了
    # 如果打开，可以做 PyTorch/TensorBoard 性能分析

    # trace_config = dict(type='tb_trace', dir_name='work_dir')
    # profiler_config = dict(on_trace_ready=trace_config)
    # runner.register_profiler_hook(profiler_config)


    # 如果是分布式训练
    if distributed:

        # 如果 runner 是 EpochBasedRunner
        if isinstance(runner, EpochBasedRunner):

            # 注册 DistSamplerSeedHook
            # 它会在每个 epoch 开始时设置 sampler 的随机种子
            # 保证分布式 shuffle 正确
            runner.register_hook(DistSamplerSeedHook())


    # register eval hooks
    # 注册验证 hook

    # 如果需要训练中验证
    if validate:

        # Support batch_size > 1 in validation
        # 从 cfg.data.val 里弹出 samples_per_gpu
        # 如果没有，就默认 1
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)

        # 如果验证 batch size > 1
        if val_samples_per_gpu > 1:

            # 这里直接 assert False
            # 说明当前 SparseDrive 自定义验证逻辑暂不支持 val batch_size > 1
            assert False

            # 下面这段理论上是 MMDet 原始逻辑：
            # batch_size > 1 时，需要把 ImageToTensor 替换成 DefaultFormatBundle
            # 但因为 assert False，实际不会执行
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline
            )

        # 构建验证集 dataset
        # custom_build_dataset 是 SparseDrive 自定义 dataset 构建函数
        # dict(test_mode=True) 表示验证时使用测试模式
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))


        # 构建验证 dataloader
        val_dataloader = build_dataloader(
            # 验证数据集
            val_dataset,

            # 验证 batch size
            samples_per_gpu=val_samples_per_gpu,

            # worker 数
            workers_per_gpu=cfg.data.workers_per_gpu,

            # 是否分布式
            dist=distributed,

            # 验证不 shuffle
            shuffle=False,

            # 分布式验证 sampler
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )

        # 获取 evaluation 配置
        # 例如 interval、metric、save_best 等
        eval_cfg = cfg.get("evaluation", {})

        # 设置 evaluation 是按 epoch 还是按 iteration
        # 如果 runner 不是 IterBasedRunner，就 by_epoch=True
        # 如果 runner 是 IterBasedRunner，就 by_epoch=False
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"

        # 设置验证结果 json 文件前缀
        # time.ctime() 会生成当前时间字符串
        # replace 是为了替换空格和冒号，避免路径不合法
        eval_cfg["jsonfile_prefix"] = osp.join(
            "val",
            cfg.work_dir,
            time.ctime().replace(" ", "_").replace(":", "_"),
        )

        # 如果是分布式训练，使用 SparseDrive 自定义 CustomDistEvalHook
        # 如果不是分布式，使用 MMDetection 原生 EvalHook
        eval_hook = CustomDistEvalHook if distributed else EvalHook

        # 注册验证 hook
        # runner 会在训练到指定 interval 时自动执行验证
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))


    # user-defined hooks
    # 注册用户自定义 hooks

    # 如果配置里有 custom_hooks
    if cfg.get("custom_hooks", None):

        # 取出 custom_hooks 配置
        custom_hooks = cfg.custom_hooks

        # 要求 custom_hooks 必须是 list
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"

        # 遍历每一个 hook 配置
        for hook_cfg in cfg.custom_hooks:

            # 每个 hook 配置必须是 dict
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )

            # 拷贝 hook 配置，避免修改原始 cfg
            hook_cfg = hook_cfg.copy()

            # 取出 priority
            # 如果没写，默认 NORMAL
            priority = hook_cfg.pop("priority", "NORMAL")

            # 根据 hook_cfg 和 HOOKS 注册表构建 hook 对象
            hook = build_from_cfg(hook_cfg, HOOKS)

            # 把自定义 hook 注册到 runner
            runner.register_hook(hook, priority=priority)


    # 如果配置中指定了 resume_from
    if cfg.resume_from:

        # 从某个 checkpoint 恢复训练
        # 注意：resume 会恢复模型参数、optimizer 状态、runner 当前 iter/epoch 等
        runner.resume(cfg.resume_from)

    # 否则，如果配置中指定了 load_from
    elif cfg.load_from:

        # 只加载模型权重
        # 不恢复 optimizer 状态和当前 iter/epoch
        runner.load_checkpoint(cfg.load_from)

    # 真正启动训练
    # data_loaders：训练/验证 dataloader 列表
    # cfg.workflow：训练流程，例如 [('train', 1)] 或 [('train', 1), ('val', 1)]
    runner.run(data_loaders, cfg.workflow)
