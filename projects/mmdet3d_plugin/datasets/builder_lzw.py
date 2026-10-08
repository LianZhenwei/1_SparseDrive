# 导入 copy
# 当前文件中没有实际使用，属于冗余导入或历史遗留
import copy

# 导入 platform
# 后面会用 platform.system() 判断当前系统是不是 Windows
import platform

# 导入 random
# worker_init_fn() 中会用 random.seed() 设置 Python 随机种子
import random

# 从 functools 中导入 partial
# partial 可以提前固定函数的一部分参数
# 后面用于：
#   collate_fn=partial(collate, samples_per_gpu=samples_per_gpu)
#   worker_init_fn=partial(worker_init_fn, ...)
from functools import partial


# 导入 numpy
# worker_init_fn() 中会用 np.random.seed() 设置 numpy 随机种子
import numpy as np

# 从 MMCV 并行模块中导入 collate
# collate 负责把 dataset 返回的多个 sample 合并成一个 batch
# 它比 PyTorch 默认 collate 更适合 MMDetection 的 DataContainer 数据结构
from mmcv.parallel import collate

# 从 MMCV runner 中导入 get_dist_info
# 用于获取当前进程 rank 和总进程数 world_size
from mmcv.runner import get_dist_info

# 从 MMCV 中导入 Registry 和 build_from_cfg
# Registry 用于注册模块
# build_from_cfg 用于根据配置 dict 构建对象
# 注意：这里前半部分导入了这两个，后半部分又重复导入了一次
from mmcv.utils import Registry, build_from_cfg

# 从 PyTorch 中导入 DataLoader
# DataLoader 负责多进程读取数据、组成 batch、送入训练循环
from torch.utils.data import DataLoader


# 从 mmdet 中导入 GroupSampler
# 非分布式训练且 shuffle=True 时使用
# GroupSampler 会根据图片宽高比例等 flag 把相似样本分在一起，提高 batch 内尺寸一致性
from mmdet.datasets.samplers import GroupSampler

# 从 SparseDrive 自定义 samplers 中导入采样器
from projects.mmdet3d_plugin.datasets.samplers import (
    # GroupInBatchSampler：
    # SparseDrive 自定义 batch sampler，IterBasedRunner 时使用
    # 它可能保证一个 batch 内样本满足某些时序/分组要求
    GroupInBatchSampler,

    # DistributedGroupSampler：
    # 分布式训练且 shuffle=True 时使用
    # 让每张 GPU 拿到不同数据，同时尽量保持同组样本在一起
    DistributedGroupSampler,

    # DistributedSampler：
    # 分布式测试/验证或不 shuffle 时使用
    # 每个 rank/GPU 拿连续或指定部分数据
    DistributedSampler,

    # build_sampler：
    # 根据配置 dict 构建具体 sampler
    build_sampler
)


"""Build PyTorch DataLoader.

函数作用：
    根据 dataset、batch size、worker 数、是否分布式、是否 shuffle、
    runner 类型等配置，构建一个 PyTorch DataLoader。

在分布式训练中：
    每个 GPU / 每个进程都会有自己的 dataloader。
    每个 dataloader 只读取当前 rank 负责的那部分数据。

在非分布式训练中：
    通常只有一个 dataloader，负责所有 GPU 或单 GPU 的数据。

Args:
    dataset (Dataset):
        PyTorch dataset。每次 __getitem__ 返回一个 sample。

    samples_per_gpu (int):
        每张 GPU 的 batch size。

    workers_per_gpu (int):
        每张 GPU 用多少个子进程读取数据。

    num_gpus (int):
        GPU 数量。主要用于非分布式训练。

    dist (bool):
        是否分布式训练或测试。

    shuffle (bool):
        每个 epoch 是否打乱数据顺序。

    seed:
        随机种子，用于 sampler 和 worker 初始化。

    shuffler_sampler:
        shuffle=True 时的 sampler 配置。

    nonshuffler_sampler:
        shuffle=False 时的 sampler 配置。

    runner_type:
        runner 类型。IterBasedRunner 时会使用 GroupInBatchSampler。

    kwargs:
        其他传给 DataLoader 的参数。

Returns:
    DataLoader:
        构建好的 PyTorch dataloader。
"""
def build_dataloader(
    # dataset：PyTorch Dataset 对象
    # 例如 NuScenes3DDataset / SparseDrive 自定义 dataset
    dataset,

    # samples_per_gpu：每张 GPU 上的 batch size
    # 比如 8 GPU，每卡 batch=6，则 samples_per_gpu=6
    samples_per_gpu,

    # workers_per_gpu：每张 GPU 的 dataloader worker 数量
    # worker 是后台读取数据的子进程
    workers_per_gpu,

    # num_gpus：GPU 数量
    # 主要在非分布式训练时使用
    num_gpus=1,

    # dist：是否分布式训练/测试
    # True 表示每张 GPU 一个进程
    # False 表示非分布式
    dist=True,

    # shuffle：是否打乱数据
    # 训练通常 True，验证/测试通常 False
    shuffle=True,

    # seed：随机种子
    # 用于 worker_init_fn 和 sampler
    seed=None,

    # shuffler_sampler：shuffle=True 时自定义 sampler 配置
    # 如果不传，则默认用 DistributedGroupSampler
    shuffler_sampler=None,

    # nonshuffler_sampler：shuffle=False 时自定义 sampler 配置
    # 如果不传，则默认用 DistributedSampler
    nonshuffler_sampler=None,

    # runner_type：runner 类型
    # 例如 "EpochBasedRunner" 或 "IterBasedRunner"
    # SparseDrive 训练中经常用 IterBasedRunner
    runner_type="EpochBasedRunner",

    # 额外参数
    # 会原样传给 PyTorch DataLoader
    **kwargs
):
    # 获取当前分布式信息
    # rank：当前进程编号
    # world_size：总进程数，也就是总 GPU 数
    rank, world_size = get_dist_info()

    # 初始化 batch_sampler
    # batch_sampler 和 sampler 二者通常不能同时控制 batch 逻辑
    batch_sampler = None

    # 如果 runner 是 IterBasedRunner
    # SparseDrive 训练配置常见是 IterBasedRunner，因为它按 iteration 控制训练长度
    if runner_type == 'IterBasedRunner':

        # 打印提示：当前使用 GroupInBatchSampler
        print("Use GroupInBatchSampler !!!")

        # 构建 GroupInBatchSampler
        # 它直接生成一个 batch 的 indices
        # 所以 DataLoader 里 batch_size 要设为 1
        batch_sampler = GroupInBatchSampler(
            # 数据集
            dataset,

            # 每张 GPU 的 batch size
            samples_per_gpu,

            # 总进程数
            world_size,

            # 当前 rank
            rank,

            # 随机种子
            seed=seed,
        )

        # 因为 batch_sampler 已经负责“一个 batch 取哪些样本”
        # 所以这里 DataLoader 的 batch_size 设置为 1
        batch_size = 1

        # 使用 batch_sampler 时，不再使用普通 sampler
        sampler = None

        # 每个进程的 worker 数
        num_workers = workers_per_gpu

    # 如果不是 IterBasedRunner，但处于分布式模式
    elif dist:

        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group

        # 如果需要 shuffle
        if shuffle:

            # 打印提示：当前使用 DistributedGroupSampler
            print("Use DistributedGroupSampler !!!")

            # 构建分布式 shuffle sampler
            sampler = build_sampler(
                # 如果传了 shuffler_sampler，就用用户指定的
                # 否则默认使用 DistributedGroupSampler
                shuffler_sampler
                if shuffler_sampler is not None
                else dict(type="DistributedGroupSampler"),

                # sampler 构造参数
                dict(
                    # 数据集
                    dataset=dataset,

                    # 每张 GPU 的 batch size
                    samples_per_gpu=samples_per_gpu,

                    # 总进程数
                    num_replicas=world_size,

                    # 当前 rank
                    rank=rank,

                    # 随机种子
                    seed=seed,
                ),
            )

        # 如果不需要 shuffle
        else:

            # 构建分布式 non-shuffle sampler
            sampler = build_sampler(
                # 如果传了 nonshuffler_sampler，就用用户指定的
                # 否则默认使用 DistributedSampler
                nonshuffler_sampler
                if nonshuffler_sampler is not None
                else dict(type="DistributedSampler"),

                # sampler 构造参数
                dict(
                    # 数据集
                    dataset=dataset,

                    # 总进程数
                    num_replicas=world_size,

                    # 当前 rank
                    rank=rank,

                    # 是否 shuffle
                    shuffle=shuffle,

                    # 随机种子
                    seed=seed,
                ),
            )

        # 分布式模式下，每个进程 DataLoader 的 batch_size 就是每张 GPU 的 batch size
        batch_size = samples_per_gpu

        # 每个进程 DataLoader 的 worker 数就是 workers_per_gpu
        num_workers = workers_per_gpu

    # 如果不是分布式模式
    else:

        # 原作者提示：非分布式模式主要用于测试推理速度
        # 训练上可能不完全支持
        # assert False, 'not support in bevformer'
        print("WARNING!!!!, Only can be used for obtain inference speed!!!!")

        # 如果 shuffle=True，则使用 mmdet 的 GroupSampler
        # 如果 shuffle=False，则不使用 sampler
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None

        # 非分布式模式下，一个 DataLoader 服务 num_gpus 张卡
        # 所以总 batch_size = num_gpus * samples_per_gpu
        batch_size = num_gpus * samples_per_gpu

        # worker 总数 = num_gpus * workers_per_gpu
        num_workers = num_gpus * workers_per_gpu


    # 构造 worker 初始化函数
    # 如果 seed 不为 None，就给每个 worker 设置不同随机种子
    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )


    # 真正构建 PyTorch DataLoader
    data_loader = DataLoader(
        # 数据集
        dataset,

        # batch_size
        # 如果使用 batch_sampler，这里是 1
        # 如果普通 sampler，这里是 samples_per_gpu 或 num_gpus*samples_per_gpu
        batch_size=batch_size,

        # 普通 sampler
        sampler=sampler,

        # batch_sampler
        # 如果非 None，则它直接生成一个 batch 的 indices
        batch_sampler=batch_sampler,

        # worker 数
        num_workers=num_workers,

        # collate_fn 负责把多个 sample 合并成一个 batch
        # 使用 mmcv.parallel.collate，而不是 PyTorch 默认 collate
        # 因为 MMDetection 中数据常被 DataContainer 包装
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),

        # 是否固定内存
        # pin_memory=True 可能提升 CPU 到 GPU 拷贝速度
        # 这里设 False
        pin_memory=False,

        # worker 初始化函数
        # 用来设置每个 worker 的随机种子
        worker_init_fn=init_fn,

        # 其他 DataLoader 参数
        **kwargs
    )

    # 返回构建好的 dataloader
    return data_loader


'''
    # worker 初始化函数
    #
    # DataLoader 会启动多个 worker 子进程读取数据。
    # 如果所有 worker 随机种子一样，那么数据增强随机性可能重复。
    # 所以这里为每个 worker 设置一个不同 seed。
    #
    # 参数说明：
    #   worker_id：当前 worker 在本进程内的编号
    #   num_workers：本进程 worker 总数
    #   rank：当前分布式进程编号
    #   seed：用户指定的基础随机种子
'''
def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed

    # 计算当前 worker 的随机种子
    # 不同 rank、不同 worker_id 都会得到不同 seed
    worker_seed = num_workers * rank + worker_id + seed

    # 设置 numpy 随机种子
    np.random.seed(worker_seed)

    # 设置 Python random 随机种子
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
# 下面开始是 dataset 构建相关代码
# 这个文件前半部分构建 dataloader，后半部分构建 dataset


# 再次导入 platform
# 前面已经导入过，这里重复导入，属于冗余
import platform

# 再次导入 Registry 和 build_from_cfg
# 前面已经导入过，这里重复导入，属于冗余
from mmcv.utils import Registry, build_from_cfg


# 从 mmdet.datasets 中导入 DATASETS 注册表
# 所有 dataset 类会注册到 DATASETS 中
# 例如 NuScenes3DDataset、自定义 SparseDrive dataset 等
from mmdet.datasets import DATASETS

# 导入 MMDetection 的 _concat_dataset 工具
# 当 cfg.ann_file 是 list/tuple 时，用它把多个 ann_file 拼成一个 ConcatDataset
from mmdet.datasets.builder import _concat_dataset


# 如果当前系统不是 Windows
if platform.system() != "Windows":

    # https://github.com/pytorch/pytorch/issues/973
    # 导入 resource 模块
    # 用于调整进程可打开文件数量上限
    import resource

    # 获取当前进程文件句柄数量限制
    # rlimit[0] 是 soft limit
    # rlimit[1] 是 hard limit
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)

    # 当前 soft limit
    base_soft_limit = rlimit[0]

    # 当前 hard limit
    hard_limit = rlimit[1]

    # 设置新的 soft limit
    # 至少 4096，但不能超过 hard limit
    soft_limit = min(max(4096, base_soft_limit), hard_limit)

    # 应用新的文件句柄限制
    # 这样 DataLoader 多 worker 读图像/标注时不容易报 too many open files
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


# 定义 Object sampler 注册表
# 用于数据库采样器等对象采样模块
# 当前文件中只是定义了这个 Registry，本文件没有继续使用它
OBJECTSAMPLERS = Registry("Object sampler")



'''
    # 自定义 dataset 构建函数
    #
    # 作用：
    # 根据配置 cfg 构建 dataset。
    # 它兼容多种 dataset wrapper：
    #   1. list/tuple -> ConcatDataset
    #   2. ConcatDataset
    #   3. RepeatDataset
    #   4. ClassBalancedDataset
    #   5. CBGSDataset
    #   6. ann_file 是 list/tuple 时的自动 concat
    #   7. 普通 dataset -> build_from_cfg(cfg, DATASETS, default_args)
    #
    # 参数：
    #   cfg：dataset 配置字典
    #   default_args：构建 dataset 时的默认参数，例如 dict(test_mode=True)
    #
    # 返回：
    #   dataset 对象
'''
def custom_build_dataset(cfg, default_args=None):
    # 尝试从 mmdet3d 导入 CBGSDataset
    # CBGS = Class-Balanced Grouping and Sampling
    # 常用于 3D 检测，让长尾类别采样更均衡
    try:
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset

    # 如果导入失败
    except:
        # 说明当前环境可能没有 mmdet3d 的 CBGSDataset
        # 设为 None，避免直接报错
        CBGSDataset = None

    # 从 mmdet 导入常见 dataset wrapper
    from mmdet.datasets.dataset_wrappers import (
        # ClassBalancedDataset：
        # 按类别频率进行过采样，缓解类别不平衡
        ClassBalancedDataset,

        # ConcatDataset：
        # 拼接多个 dataset
        ConcatDataset,

        # RepeatDataset：
        # 重复一个 dataset 多次
        RepeatDataset,
    )


    # 如果 cfg 本身是 list 或 tuple
    if isinstance(cfg, (list, tuple)):

        # 对每个子配置递归调用 custom_build_dataset
        # 然后用 ConcatDataset 拼接起来
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg]
        )

    # 如果 cfg 的 type 是 ConcatDataset
    elif cfg["type"] == "ConcatDataset":

        # 构建每个子 dataset，再拼接成 ConcatDataset
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg["datasets"]],

            # separate_eval 表示是否分别对每个子 dataset 做 eval
            # 默认 True
            cfg.get("separate_eval", True),
        )

    # 如果 cfg 的 type 是 RepeatDataset
    elif cfg["type"] == "RepeatDataset":

        # 先构建内部 dataset
        # 再用 RepeatDataset 重复 cfg["times"] 次
        dataset = RepeatDataset(
            custom_build_dataset(cfg["dataset"], default_args), cfg["times"]
        )

    # 如果 cfg 的 type 是 ClassBalancedDataset
    elif cfg["type"] == "ClassBalancedDataset":

        # 先构建内部 dataset
        # 再用 ClassBalancedDataset 做类别均衡采样包装
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg["dataset"], default_args),

            # oversample_thr 控制过采样阈值
            cfg["oversample_thr"],
        )

    # 如果 cfg 的 type 是 CBGSDataset
    elif cfg["type"] == "CBGSDataset":

        # 先构建内部 dataset
        # 再用 CBGSDataset 包装
        dataset = CBGSDataset(
            custom_build_dataset(cfg["dataset"], default_args)
        )

    # 如果 cfg["ann_file"] 是 list 或 tuple
    elif isinstance(cfg.get("ann_file"), (list, tuple)):

        # 使用 mmdet 的 _concat_dataset 自动构建拼接 dataset
        dataset = _concat_dataset(cfg, default_args)

    # 其他普通情况
    else:

        # 根据 cfg 从 DATASETS 注册表中构建 dataset
        # 例如 cfg = dict(type="NuScenes3DDataset", ...)
        # 就会去 DATASETS 注册表中找 NuScenes3DDataset 类并实例化
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    # 返回构建好的 dataset
    return dataset
