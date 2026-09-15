'''
    本文件只需搜索《lzw》看重点即可
'''
# Copyright (c) OpenMMLab. All rights reserved.
# 版权声明：这份 train.py 最初来自 OpenMMLab / MMDetection 体系，SparseDrive 在其基础上做了插件化修改。

from __future__ import division
# 兼容 Python2 的除法行为。
# 在 Python3 中，1 / 2 默认就是 0.5；这行在现在的 Python3 环境里基本没实际影响。
# 但 OpenMMLab 很多老代码会保留这类兼容写法。

import sys # 导入 sys，用于访问 Python 解释器路径、命令行参数等系统级信息。
import os  # 导入 os，用于处理环境变量、路径、文件夹等操作。
print(sys.executable, os.path.abspath(__file__)) # 打印当前使用的 Python 解释器路径，以及当前 train.py 文件的绝对路径。这在服务器 / conda / LSF / Slurm 环境下非常有用。例如你想确认当前是不是用的 sparsedrive conda 环境，而不是系统 Python。

# import init_paths # for conda pkgs submitting method # 这行被注释掉了。某些集群提交任务时，需要手动把项目路径加入 sys.path。init_paths 通常就是做这类路径初始化的。这里 SparseDrive 没有启用它。
import copy         # copy 用于深拷贝配置。后面构建 val_dataset 时会用 copy.deepcopy，避免直接修改原始 cfg.data.val。
import mmcv         # 导入 MMCV。MMCV 是 OpenMMLab 的基础库，提供 Config、Runner、Hook、日志、文件操作等功能。
import time         # time 用于生成 timestamp，比如 20260610_123456.log。
import torch        # 导入 PyTorch，用于模型训练、CUDA、分布式等。
import warnings     # warnings 用于发出警告。后面 --options 被弃用时会用 warnings.warn。

from mmcv import Config, DictAction                  # Config：MMCV 的配置文件读取类，可以读取 .py config。DictAction：argparse 的自定义 action，用于解析 --cfg-options 这种 key=value 参数。
from mmcv.runner import get_dist_info, init_dist     # get_dist_info：获取当前分布式训练中的 rank 和 world_size。init_dist：初始化分布式训练环境，比如 pytorch / slurm / mpi。
from os import path as osp                           # 把 os.path 简写成 osp。后面会用 osp.join、osp.basename、osp.splitext 等路径操作函数。
from mmdet import __version__ as mmdet_version       # 获取当前安装的 mmdet 版本。后面保存 checkpoint meta 信息时会记录版本，方便复现实验。
from mmdet.apis import train_detector                # MMDetection 原生训练入口。如果没有 SparseDrive 自定义 plugin，就会调用这个函数。
from mmdet.datasets import build_dataset             # 根据 cfg.data.train / cfg.data.val 构建数据集对象。SparseDrive 的 NuScenes 数据集类会通过 Registry 注册进来。
from mmdet.models import build_detector              # 根据 cfg.model 构建模型。SparseDrive 模型类也会通过 Registry 注册进来。
from mmdet.utils import collect_env, get_root_logger # collect_env：收集当前环境信息，比如 Python、CUDA、PyTorch、MMCV、MMDetection 版本。get_root_logger：创建日志记录器，把训练日志写到 work_dir 里。
from mmdet.apis import set_random_seed               # 设置随机种子。控制 Python、NumPy、PyTorch 等随机性，方便复现实验。
from torch import distributed as dist                # PyTorch 分布式训练模块。后面的 mpi_nccl 分支会直接调用 dist.init_process_group。
from datetime import timedelta                       # timedelta 用于设置分布式初始化的 timeout。SparseDrive 这里设置为 3600 秒，防止大模型初始化时间过长导致超时。

import cv2           # 导入 OpenCV。数据预处理、图像读取、图像增强等流程中可能会用到 OpenCV。
cv2.setNumThreads(8) # 设置 OpenCV 内部最多使用 8 个线程。防止 OpenCV 默认开太多线程，导致 dataloader worker 和 OpenCV 线程互相抢 CPU。在服务器多进程训练时，这个设置有助于稳定性能。


import argparse     # argparse 用于解析命令行参数。比如 python tools/train.py xxx.py --work-dir xxx --launcher pytorch。
# 一、命令行参数解析函数：负责把用户输入的命令行参数转换成 args 对象
def parse_args():
    # ========================================== (1) 创建 argparse 解析器：description 表示这个脚本的用途是训练检测器。虽然 SparseDrive 不只是 detector，但它沿用了 mmdet 的训练脚本结构。 ==========================================
    parser = argparse.ArgumentParser(description="Train a detector")


    # ========================================== (2) 以下是命令行参数定义。每个参数都有 help 描述，方便用户查看。 ==========================================
    # 必填位置参数：config。
    # 例如：python tools/train.py projects/configs/sparsedrive_small_stage2.py。这里的 config 就是 sparse_drive 的训练配置文件路径。
    parser.add_argument("config", help="train config file path")

    # 可选参数：--work-dir。
    # 用于指定训练日志、checkpoint、config 备份等保存目录。如果用户不传，则后面会根据 config 文件名自动生成 work_dirs/xxx。
    parser.add_argument("--work-dir", help="the dir to save logs and models")

    # 可选参数：--resume-from。
    # 用于从某个 checkpoint 恢复训练。
    # 注意：resume 是恢复训练状态，包括模型权重、optimizer、lr scheduler、iteration 等。
    # 它和 load_from 不一样，load_from 通常只是加载模型权重。
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")

    # 可选参数：--no-validate。
    # 如果传了这个参数，就表示训练过程中不做验证集评估。
    # action="store_true" 表示只要命令行里出现这个参数，args.no_validate 就是 True。
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )

    # 创建一个互斥参数组：互斥的意思是“这个组里的参数不能同时使用”，在这里 --gpus 和 --gpu-ids 不能同时传。
    group_gpus = parser.add_mutually_exclusive_group()

    # 非分布式训练时使用的 GPU 数量。
    # 例如 --gpus 1。
    # 注意：这个参数只适用于 launcher=none 的单进程训练。
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use "
        "(only applicable to non-distributed training)",
    )

    # 非分布式训练时指定具体 GPU id。
    # 例如 --gpu-ids 0 或 --gpu-ids 0 1。
    # nargs="+" 表示可以接收一个或多个整数。
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use "
        "(only applicable to non-distributed training)",
    )

    # 随机种子，默认是 0：训练深度学习模型时，seed 会影响数据打乱、初始化、增强随机性等。
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # 是否让 CUDNN 尽量使用确定性算法。
    # 传了 --deterministic 后，实验更容易复现。
    # 但有时会降低速度，并且不是所有算子都能完全确定。
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )

    # 旧版参数：--options。
    # 用于在命令行里覆盖 config 中的某些字段。
    # 例如 --options optimizer.lr=0.0002。但这个参数已经被废弃，推荐用 --cfg-options。
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    # 新版参数：--cfg-options。
    # 作用是在不修改配置文件的情况下，临时覆盖 config 字段。
    # 例如：--cfg-options data.samples_per_gpu=1 optimizer.lr=1e-4。对调参和 debug 很有用。
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    # 分布式初始化地址。
    # 例如 tcp://localhost:8000。
    # 对 mpi_nccl 模式有用。
    # 默认是 auto。
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )

    # 每台机器有多少张 GPU。
    # 默认 8 张。
    # mpi_nccl 分支里会用它设置 CUDA_VISIBLE_DEVICES 和当前进程对应 GPU。
    parser.add_argument("--gpus-per-machine", type=int, default=8)

    # 指定训练启动方式。
    # none：不使用分布式，一般单卡 debug。
    # pytorch：用 PyTorch 分布式启动，一般 torchrun 或 tools/dist_train.sh。
    # slurm：通过 Slurm 集群启动。
    # mpi：通过 MPI 启动。
    # mpi_nccl：代码里额外写的 MPI + NCCL 初始化方式。
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi", "mpi_nccl"],
        default="none",
        help="job launcher",
    )

    # 当前进程的本地 GPU rank。
    # PyTorch 分布式启动时常用 local_rank 指定当前进程使用哪张卡。
    # 例如一台机器 4 张卡，就会有 local_rank 0、1、2、3。
    parser.add_argument("--local_rank", type=int, default=0)

    # 是否根据 GPU 数量自动缩放学习率。
    # 原理是 linear scaling rule。
    # 例如原配置按 8 卡设置，如果只用 4 卡，则 lr 会乘 4/8。
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )


    # ========================================== (3) 解析命令行参数 ==========================================
    # 真正解析命令行参数：返回 args 对象，后面通过 args.config、args.work_dir 等访问。
    args = parser.parse_args()

    # 如果环境变量里没有 LOCAL_RANK，就用命令行参数 args.local_rank 设置：这是为了兼容不同版本的分布式启动方式，有的启动器会自动注入 LOCAL_RANK，有的不会。
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    # 如果用户同时传了 --options 和 --cfg-options，就直接报错：因为二者功能重复，而且 --options 已经被废弃。
    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, "
            "--options is deprecated in favor of --cfg-options"
        )

    # 如果用户只传了老参数 --options：则给出警告，然后把 args.options 赋值给 args.cfg_options，保证后续统一处理。
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options


    # ========================================== (4) 返回解析好的命令行参数 ==========================================
    return args
    

# 二、主函数：训练的核心流程从这里开始
def main():
    # ========================================== (1) 解析命令行参数：例如 args.config、args.work_dir、args.launcher、args.seed 等 ==========================================
    args = parse_args()


    # ========================================== (2) 解析配置文件里的参数：从配置文件中读取 cfg 并解析成一个 Config 对象，如 cfg.model、cfg.data、cfg.optimizer、cfg.runner 等都来自这个配置文件 ==========================================
    cfg = Config.fromfile(args.config) # lzw重点1.1：读取 .py 配置文件（sparsedrive_small_stage1.py或sparse_drive_small_stage2.py）并解析为mmcv的 Config 对象


    # ========================================== (3) 处理命令行参数 args 和配置文件里的参数 cfg ==========================================
    # 如果命令行传了 --cfg-options，就把这些 key=value 合并进 cfg。
    # 命令行覆盖优先级高于 config 文件本身。
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 根据 config 里的 custom_imports 导入额外模块：
    if cfg.get("custom_imports", None):                      # 只有配置里存在 custom_imports 时才导入这个 custom_imports() 函数
        from mmcv.utils import import_modules_from_strings   # import_modules_from_strings 可以根据字符串模块路径动态 import
        import_modules_from_strings(**cfg["custom_imports"]) # 按 config 中 custom_imports 的设置导入模块。这一步通常用于注册自定义组件到 Registry。

    # 这段是 SparseDrive 关键逻辑：导入 projects/mmdet3d_plugin，使自定义模块注册进 mmdet/mmcv 的 Registry
    if hasattr(cfg, "plugin"): # 判断 config 里是否有 plugin 字段
        if cfg.plugin:         # 如果 plugin=True，就启用插件导入逻辑【SparseDrive 的配置通常会设置 plugin=True】
            # importlib 用于动态导入 Python 模块。例如根据字符串 "projects.mmdet3d_plugin" 导入模块。
            import importlib

            # 如果 config 里指定了 plugin_dir，就从 plugin_dir 推导要导入的模块路径
            if hasattr(cfg, "plugin_dir"):
                # 读取插件目录。SparseDrive 通常是类似 "projects/mmdet3d_plugin/"。
                plugin_dir = cfg.plugin_dir

                # 取 plugin_dir 的目录名。例如 plugin_dir="projects/mmdet3d_plugin/"，则 os.path.dirname 得到 "projects/mmdet3d_plugin"。
                _module_dir = os.path.dirname(plugin_dir)

                # 按 "/" 切分路径。"projects/mmdet3d_plugin" -> ["projects", "mmdet3d_plugin"]。
                _module_dir = _module_dir.split("/")

                # 初始化模块路径。此时 _module_path = "projects"。
                _module_path = _module_dir[0]

                # 遍历后续路径部分。例如 "mmdet3d_plugin"。
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m # 把文件路径转换成 Python 模块路径。["projects", "mmdet3d_plugin"] -> "projects.mmdet3d_plugin"

                # 打印最终要 import 的模块路径。例如 projects.mmdet3d_plugin。
                print(_module_path)

                # 动态导入插件模块。
                # 这一步非常关键。
                # 导入 projects.mmdet3d_plugin 后，它里面的 Dataset、Model、Head、Loss、Hook 会通过注册器注册。
                # 后面 build_detector / build_dataset 才能找到 SparseDrive 自定义类。
                plg_lib = importlib.import_module(_module_path)

            # 如果 config 里没有 plugin_dir，则默认从 config 文件所在目录推导模块路径：
            else:
                # 取 config 文件所在目录。例如 projects/configs。
                _module_dir = os.path.dirname(args.config)

                # 按 "/" 切分路径。例如 ["projects", "configs"]。
                _module_dir = _module_dir.split("/")

                # 初始化模块路径。例如 "projects"。
                _module_path = _module_dir[0]

                # 遍历后续路径片段
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m # 拼接成 Python import 路径。例如 "projects.configs"。
                    
                # 打印模块路径
                print(_module_path)
                
                # 动态导入该模块。不过对 SparseDrive 来说，通常更推荐显式 plugin_dir 指向 projects/mmdet3d_plugin。
                plg_lib = importlib.import_module(_module_path)

            # 从 SparseDrive 插件里导入自定义训练函数 custom_train_model。
            # 这个函数会进一步调用 SparseDrive 改过的训练逻辑。
            # 如果不走 plugin，就会使用 mmdet.apis.train_detector。
            from projects.mmdet3d_plugin.apis.train import custom_train_model

    # 设置 CUDNN benchmark
    if cfg.get("cudnn_benchmark", False):
        # 如果 config 里 cudnn_benchmark=True，就开启 CUDNN benchmark。
        # 当输入图像尺寸固定时，它可以自动选择最快卷积算法，提高训练速度。
        # 如果输入尺寸变化很大，可能反而不稳定或不划算。
        torch.backends.cudnn.benchmark = True

    # 【工作目录】设置 work_dir
    '''
        work_dir 的优先级：CLI > segment in file > filename，即：
            1. 命令行 --work-dir
            2. config 文件里的 work_dir
            3. 根据 config 文件名自动生成    
    '''
    if args.work_dir is not None:    # 如果命令行显式指定了 --work-dir，
        cfg.work_dir = args.work_dir # 则用命令行传入的 work_dir 覆盖 config 中的 work_dir；
    elif cfg.get("work_dir", None) is None:                                                # 如果命令行没有传 --work-dir，并且 config 里也没有 work_dir，
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0]) # 则根据 config 文件名自动创建默认 work_dir：例如 config 是 sparse_drive_small_stage2.py，work_dir 就是 ./work_dirs/sparse_drive_small_stage2。
        
    # 【恢复训练】如果命令行传了 --resume-from，就写入 cfg.resume_from。后续 Runner 会根据这个字段恢复训练。
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # 设置 GPU 设备 id：
    if args.gpu_ids is not None:   # 如果用户传了 --gpu-ids，
        cfg.gpu_ids = args.gpu_ids # 就用这些指定 GPU id。
    else:
        # 如果用户没传 --gpu-ids、也没传 --gpus，就默认使用 range(1)，也就是单卡。
        # 如果传了 --gpus N，就使用 range(N)，也就是 GPU 0 到 GPU N-1。
        # 注意：分布式训练时后面会重新设置 cfg.gpu_ids。
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # 【自动缩放学习率】如果启用自动缩放学习率，则按 GPU 数量线性缩放学习
    if args.autoscale_lr:
        # 按 GPU 数量线性缩放学习率。
        # 默认假设原 config 是按 8 卡设置的。
        # 如果当前 len(cfg.gpu_ids)=4，学习率变为原来的 4/8。
        # 如果当前 len(cfg.gpu_ids)=1，学习率变为原来的 1/8。
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    # 先初始化分布式环境。因为 logger 会根据 rank 决定哪些进程输出日志。
    if args.launcher == "none":       # 如果 launcher 是 none，
        distributed = False           # 则说明不使用分布式训练，一般是单卡 debug；
    elif args.launcher == "mpi_nccl": # 如果 launcher 是 mpi_nccl，
        distributed = True            # 则启用分布式训练；

        # 这是代码里额外写的一套 MPI + NCCL 初始化逻辑。

        # 导入 mpi4py。用 MPI 获取当前进程 rank 和总进程数。
        import mpi4py.MPI as MPI

        # 获取全局 MPI 通信器。
        comm = MPI.COMM_WORLD

        # 获取当前 MPI 进程的 rank。
        # 注意这里变量名叫 local_rank，但 comm.Get_rank() 通常是 global rank。
        mpi_local_rank = comm.Get_rank()

        # 获取总进程数量。
        # 例如 4 卡训练时 world_size=4。
        mpi_world_size = comm.Get_size()

        # 打印当前 MPI rank 和 world_size，方便调试。
        print("MPI local_rank=%d, world_size=%d" % (mpi_local_rank, mpi_world_size))

        # 这行被注释掉了。原本可能想动态获取当前机器 GPU 数量。
        # num_gpus = torch.cuda.device_count()

        # 生成当前机器可见 GPU id 列表。
        # 如果 gpus_per_machine=8，则得到 [0,1,2,3,4,5,6,7]。
        device_ids_on_machines = list(range(args.gpus_per_machine))

        # 把 GPU id 转成字符串。例如 ["0","1","2","3","4","5","6","7"]。
        str_ids = list(map(str, device_ids_on_machines))

        # 设置 CUDA_VISIBLE_DEVICES。
        # 例如 "0,1,2,3,4,5,6,7"。
        # 这样当前进程能看到这些 GPU。
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)

        # 设置当前进程使用哪张 GPU。
        # 例如 mpi_local_rank=3，gpus_per_machine=8，就使用 cuda:3。
        # 如果多机，每台 8 卡，则通过取模映射到本机 GPU。
        torch.cuda.set_device(mpi_local_rank % args.gpus_per_machine)

        # 初始化 PyTorch 分布式进程组
        dist.init_process_group(
            backend="nccl",                  # backend="nccl"：GPU 分布式训练最常用的通信后端。
            init_method=args.dist_url,       # init_method=args.dist_url：指定进程之间如何发现彼此。
            world_size=mpi_world_size,       # world_size：总进程数。
            rank=mpi_local_rank,             # rank：当前进程编号。
            timeout=timedelta(seconds=3600), # timeout：初始化和通信超时时间，这里是 1 小时。
        )

        # 分布式情况下，gpu_ids 设置成 world_size 范围。
        # 例如 4 卡就是 range(0,4)。
        cfg.gpu_ids = range(mpi_world_size)

        # 打印当前 cfg.gpu_ids，方便检查
        print("cfg.gpu_ids:", cfg.gpu_ids)
    else:                             # 除 none 和 mpi_nccl 外，
        distributed = True            # 则其他 launcher 都视为分布式训练，例如 pytorch、slurm、mpi。
        
        # 使用 MMCV 的 init_dist 初始化分布式训练。
        # args.launcher 指定启动方式。
        # cfg.dist_params 通常在 config 里，例如 dict(backend='nccl')。
        # timeout 设置为 1 小时，避免大任务初始化超时。
        init_dist(args.launcher, timeout=timedelta(seconds=3600), **cfg.dist_params)

        # 获取当前分布式环境信息。
        # 第一个返回值是 rank，这里用 _ 忽略。
        # 第二个返回值是 world_size，也就是总 GPU / 进程数量。
        _, world_size = get_dist_info()

        # 分布式训练时重新设置 gpu_ids。
        # 例如 8 卡训练时 cfg.gpu_ids = range(0,8)。
        cfg.gpu_ids = range(world_size)

    # 【工作目录】创建 work_dir 文件夹。
    # 如果已经存在，则不会报错。
    # 所有日志、checkpoint、配置备份都会放在这里。
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # 【保存配置】把当前最终使用的 config 保存一份到 work_dir。
    # 注意：这里保存的是合并过命令行 cfg-options 之后的配置。这对实验复现非常重要。
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config))) # lzw重点1.2：保存config到work_dir

    # 生成时间戳字符串。例如 20260610_151230。
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # 【日志文件】日志文件路径。
    # 例如 work_dirs/sparsedrive_stage2/20260610_151230.log。
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")

    # 指定 logger 名称。
    # 如果仍然使用默认的 mmdet logger，某些输出可能被过滤，无法写入 log_file。

    # TODO: ugly workaround to judge whether we are training det or seg model
    # 原作者注释：这是一个不太优雅的 workaround。
    # 用于判断当前训练的是检测模型还是分割模型。

    # 创建 root logger。
    # 日志会同时输出到终端和 log_file。
    # cfg.log_level 通常是 "INFO"。
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # 初始化 meta 字典。
    # meta 会记录环境信息、config、seed、实验名等。
    # 后面 checkpoint 里也可能保存这些 meta 信息。
    meta = dict()

    # 收集环境信息。
    # 包括系统、Python、CUDA、PyTorch、MMCV、MMDetection 等版本。
    env_info_dict = collect_env()

    # 把环境信息字典格式化成多行字符串
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])

    # 构造分割线，让日志更清晰
    dash_line = "-" * 60 + "\n"

    # 把环境信息写入日志。
    # 以后排查版本问题时非常有用。
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)

    # 把环境信息保存进 meta。
    meta["env_info"] = env_info

    # 把完整 config 文本保存进 meta。
    meta["config"] = cfg.pretty_text

    # 记录当前是否使用分布式训练。
    logger.info(f"Distributed training: {distributed}")

    # 把完整 config 打印到日志中。
    # 训练开始时日志里通常会看到非常长的一段 config，就是这里输出的。
    logger.info(f"Config:\n{cfg.pretty_text}")

    # 如果用户设置了随机种子。
    # 默认 seed=0，所以通常都会进入这里。
    if args.seed is not None:
        # 记录随机种子和 deterministic 设置
        logger.info(
            f"Set random seed to {args.seed}, "
            f"deterministic: {args.deterministic}"
        )

        # 设置随机种子。
        # deterministic=True 时，会设置 CUDNN 确定性选项。
        set_random_seed(args.seed, deterministic=args.deterministic)

    # 把 seed 写进 cfg。
    # 后续 dataloader 或 runner 可能会读取 cfg.seed。
    cfg.seed = args.seed

    # 把 seed 记录进 meta
    meta["seed"] = args.seed

    # 把实验名记录进 meta。
    # 这里用 config 文件名作为实验名。
    meta["exp_name"] = osp.basename(args.config)

    # 【重点】根据 cfg.model 构建模型：
    # SparseDrive 中这里会构建 SparseDrive 类。
    # cfg.model 里会指定：
    # type='SparseDrive'
    # img_backbone
    # img_neck
    # head
    # depth_branch
    # 等等。
    # build_detector 的背后是 Registry 机制：
    # 先根据 type 找到注册过的模型类，再把 config 参数传入构造函数。
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")) # lzw重点2：构建model（build_detector()函数来自于mmdet库）【cfg.model 就是 sparsedrive_small_stage1.py（或 sparsedrive_small_stage2.py）里定义的模型 model = dict(type="SparseDrive", ...)】

    # 初始化模型权重：
    # 对 SparseDrive 来说：
    # 1. ResNet 可能加载 ImageNet 预训练权重。
    # 2. Anchor encoder、head、FFN 等模块会初始化。
    # 3. 如果 config 里有 load_from，后面 Runner 阶段会加载 checkpoint。
    # 注意：init_weights 不是 resume_from，也不是 load_from。
    model.init_weights()

    # 把模型结构打印进日志。
    # 这会很长，但非常有用，可以确认 det_head、map_head、motion_plan_head 有没有构建出来。
    logger.info(f"Model:\n{model}")

    # 【训练集的工作目录】把 work_dir 写入训练数据集配置。
    # SparseDrive 的某些 dataset / pipeline / evaluator 可能需要知道 work_dir。
    # 例如保存中间结果、可视化结果或评估文件。
    cfg.data.train.work_dir = cfg.work_dir

    # 【验证集的工作目录】把 work_dir 写入验证数据集配置。
    cfg.data.val.work_dir = cfg.work_dir

    # 【重点】根据 cfg.data.train 构建训练数据集。
    # 返回的数据集对象会被放进列表 datasets。
    # SparseDrive 中这里通常构建的是 NuScenes 相关自定义 Dataset。
    # 数据集会读取 data/infos 里的 pkl 文件，并使用 config 中的 pipeline 处理多相机图像、标注、地图、轨迹等。
    datasets = [build_dataset(cfg.data.train)] # lzw重点4.1：构建训练数据集（build_dataset()函数来自于mmdet3d库）

    # 如果 workflow 长度是 2，说明训练流程里不仅有 train，还有 val。
    # 例如 workflow=[('train', 1), ('val', 1)]。
    # 大多数情况下 SparseDrive 训练可能只用 [('train', 1)]。
    if len(cfg.workflow) == 2:
        # 深拷贝验证集配置。这样修改 val_dataset 不会影响原始 cfg.data.val。
        val_dataset = copy.deepcopy(cfg.data.val)

        # 如果训练数据集外面套了 dataset wrapper，就要特殊处理 pipeline
        if "dataset" in cfg.data.train:                            # 如果 cfg.data.train 里面包含 dataset 字段，说明它可能是 RepeatDataset、ClassBalancedDataset 等 wrapper，
            val_dataset.pipeline = cfg.data.train.dataset.pipeline # 这时真正的 pipeline 在 cfg.data.train.dataset.pipeline；
        else:                                                      # 如果没有 wrapper，
            val_dataset.pipeline = cfg.data.train.pipeline         # 则直接把训练 pipeline 赋给 val_dataset。
            # 这里的目的：workflow 里的 val 更像训练过程中的 val step，不是最终 test_mode 评估。所以它使用训练风格 pipeline。

        # 设置 val_dataset.test_mode=False。
        # 这里是为了让 val step 也能返回训练需要的数据格式。
        # 它不会影响后面真正的 AP/AR 评估。
        val_dataset.test_mode = False

        # 构建验证数据集，并加入 datasets 列表。
        # 此时 datasets = [train_dataset, val_dataset]。
        datasets.append(build_dataset(val_dataset)) # lzw重点4.2：构建验证数据集（build_dataset()函数来自于mmdet3d库）

    # 【Checkpoint】如果配置里启用了 checkpoint 保存。
    if cfg.checkpoint_config is not None:
        # 设置 checkpoint 的 meta 信息：在 checkpoint 里保存 mmdet 版本、config 内容、类别名等 meta 信息
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES, # datasets[0].CLASSES 是训练数据集类别名。对 nuScenes detection 来说，CLASSES 可能是 car、truck、pedestrian 等类别。保存这些信息后，测试和可视化时更容易恢复类别含义。
        )

    # 给 model 添加 CLASSES 属性。
    # 这不是模型前向传播必须的，而是为了测试、评估、可视化方便。
    model.CLASSES = datasets[0].CLASSES

    # 【重点】如果 config 里有 plugin 字段，说明使用 SparseDrive 自定义插件训练逻辑
    if hasattr(cfg, "plugin"):
        # 调用 SparseDrive 自定义训练入口。
        # 这个函数会继续进入 projects/mmdet3d_plugin/apis/train.py。
        # 最终会构建 dataloader、optimizer、runner，并开始训练。
        # validate=(not args.no_validate)：如果没传 --no-validate，就会启用训练过程验证。
        custom_train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        ) # lzw重点5.1：调用SparseDrive自定义训练入口函数 custom_train_model()（函数custom_train_model()在文件 projects/mmdet3d_plugin/apis/train.py 里）

    # 【重点】如果 config 没有 plugin 字段，则走 MMDetection 原生训练逻辑
    else:
        # 调用 mmdet.apis.train_detector。
        # 普通 MMDetection 模型会走这里。
        # SparseDrive 一般不会走这里，而是走 custom_train_model。
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta,
        ) # lzw重点5.2：调用MMDetection训练入口函数 train_detector()（函数train_detector()在文件 projects/mmdet3d_plugin/apis/train.py 里）【但 SparseDrive 一般不会走这里，而是走 custom_train_model】


# 三、Python 脚本入口
if __name__ == "__main__": 
    # 设置 PyTorch multiprocessing 的启动方式为 fork。
    # 这样 dataloader 的 workers_per_gpu 可以大于 1。
    # fork 的好处是启动快、能继承父进程状态。
    # 但在某些 CUDA / 多线程场景下也可能带来隐患。
    # SparseDrive 这里显式使用 fork，是为了兼容多 worker 数据加载。
    torch.multiprocessing.set_start_method("fork") # use fork workers_per_gpu can be > 1

    # 调用主函数，正式开始训练流程
    main()
