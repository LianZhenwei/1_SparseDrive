# Copyright (c) OpenMMLab. All rights reserved.
# 版权声明：这份 test.py 继承自 OpenMMLab / MMDetection 的测试脚本风格，
# SparseDrive 在其基础上加入了自己的 dataloader 和 multi_gpu_test 逻辑。

import argparse
# argparse：用于解析命令行参数。
# 例如：
# python tools/test.py config.py checkpoint.pth --eval bbox

import mmcv
# mmcv：OpenMMLab 基础库。
# 这里会用到 Config、load/dump 文件、mkdir_or_exist 等工具。

import os
# os：用于读取环境变量、处理路径等。

from os import path as osp
# 把 os.path 简写为 osp。
# 后面会用 osp.join、osp.basename、osp.splitext 等函数处理路径。

import torch
# 导入 PyTorch。
# 用于模型推理、CUDA、分布式、multiprocessing 等。

import warnings
# warnings：用于输出弃用参数的警告。
# 例如 --options 已弃用，推荐 --eval-options。

from mmcv import Config, DictAction
# Config：用于读取 .py 配置文件。
# DictAction：用于解析命令行里的 key=value 格式参数。
# 例如 --cfg-options data.test.samples_per_gpu=1

from mmcv.cnn import fuse_conv_bn
# fuse_conv_bn：把 Conv 和 BatchNorm 融合。
# 测试时可以略微提升推理速度。

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# MMDataParallel：单机单进程 / 单卡测试时包装模型。
# MMDistributedDataParallel：分布式多卡测试时包装模型。

from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
# get_dist_info：获取当前分布式 rank 和 world_size。
# init_dist：初始化分布式环境。
# load_checkpoint：加载模型 checkpoint。
# wrap_fp16_model：把模型包装成 fp16 推理 / 训练模式。

from mmdet.apis import single_gpu_test, multi_gpu_test, set_random_seed
# single_gpu_test：MMDetection 原生单卡测试函数。
# multi_gpu_test：MMDetection 原生多卡测试函数。
# set_random_seed：设置随机种子。
# 注意：这里 multi_gpu_test 被导入了，但后面实际没有用到。
# SparseDrive 多卡测试用的是 custom_multi_gpu_test。

from mmdet.datasets import replace_ImageToTensor, build_dataset
# replace_ImageToTensor：当 samples_per_gpu > 1 时，把 ImageToTensor 替换成 DefaultFormatBundle。
# build_dataset：根据 cfg.data.test 构建测试数据集。

from mmdet.datasets import build_dataloader as build_dataloader_origin
# 导入 MMDetection 原生 dataloader，并改名为 build_dataloader_origin。
# 后面非分布式测试时使用它。

from mmdet.models import build_detector
# build_detector：根据 cfg.model 构建检测器 / SparseDrive 模型。
# 本质依赖 Registry 机制。

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
# 导入 SparseDrive / mmdet3d_plugin 自定义 dataloader。
# 后面分布式测试时会使用这个版本。

from projects.mmdet3d_plugin.apis.test import custom_multi_gpu_test
# 导入 SparseDrive 自定义多卡测试函数。
# SparseDrive 多卡测试不使用 mmdet 原生 multi_gpu_test，而是用这个 custom_multi_gpu_test。


def parse_args():
    # 定义命令行参数解析函数。
    # 它会把用户从终端传入的 config、checkpoint、--eval 等参数解析成 args。

    parser = argparse.ArgumentParser(
        description="MMDet test (and eval) a model"
    )
    # 创建参数解析器。
    # description 表示这个脚本用于测试和评估模型。

    parser.add_argument("config", help="test config file path")
    # 必填参数 1：config。
    # 表示测试配置文件路径。
    # 例如：
    # projects/configs/sparsedrive_small_stage2.py

    parser.add_argument("checkpoint", help="checkpoint file")
    # 必填参数 2：checkpoint。
    # 表示要加载的模型权重文件。
    # 例如：
    # work_dirs/sparsedrive_stage2/latest.pth

    parser.add_argument("--out", help="output result file in pickle format")
    # 可选参数：--out。
    # 用于把模型推理结果保存成 .pkl / .pickle 文件。
    # 例如：
    # --out results.pkl

    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    # 可选参数：--fuse-conv-bn。
    # 如果传入这个参数，就会在测试前把 Conv + BN 融合。
    # 一般用于部署或加速推理。
    # action="store_true" 表示：只要命令里出现它，args.fuse_conv_bn 就是 True。

    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    # 可选参数：--format-only。
    # 只格式化输出结果，不计算评估指标。
    # 常用于提交到官方 benchmark server，例如 nuScenes test server。
    # 比如只生成 submission json，而不在本地算 NDS/mAP。

    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    # 可选参数：--eval。
    # 指定评估指标。
    # nargs="+" 表示可以接收一个或多个字符串。
    # 对 SparseDrive / nuScenes 来说，常见可能是：
    # --eval bbox
    # 具体可用指标取决于 dataset.evaluate() 的实现。

    parser.add_argument("--show", action="store_true", help="show results")
    # 可选参数：--show。
    # 是否直接显示预测结果。
    # 对服务器无图形界面的环境一般不常用。

    parser.add_argument(
        "--show-dir", help="directory where results will be saved"
    )
    # 可选参数：--show-dir。
    # 指定可视化结果保存目录。
    # 例如：
    # --show-dir vis_results

    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    # 可选参数：--gpu-collect。
    # 多卡测试时，各 GPU 都会得到一部分推理结果。
    # 最后需要把所有结果汇总。
    # 如果传了 --gpu-collect，就用 GPU 通信收集结果。
    # 否则通常通过 tmpdir 文件系统收集。

    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    # 可选参数：--tmpdir。
    # 多卡测试时，如果不用 GPU 收集结果，就需要一个临时目录保存各 rank 的结果。
    # 最后 rank0 再从 tmpdir 汇总。
    # 这个参数只有在不使用 --gpu-collect 时有用。

    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # 可选参数：--seed。
    # 设置随机种子，默认 0。
    # 测试阶段一般随机性较少，但涉及某些数据处理或模型模块时仍有意义。

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    # 可选参数：--deterministic。
    # 如果传入，则尽量让 CUDNN 使用确定性算法。
    # 有利于复现，但可能牺牲速度。

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
    # 可选参数：--cfg-options。
    # 用于在命令行临时覆盖 config 配置。
    # 例如：
    # --cfg-options data.test.samples_per_gpu=1 model.use_grid_mask=False
    # 这样不用修改原始 config 文件。

    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    # 可选参数：--options。
    # 老版本参数，用于给 dataset.evaluate() 传额外参数。
    # 已弃用，推荐使用 --eval-options。

    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    # 可选参数：--eval-options。
    # 用于给 dataset.evaluate() 或 format_results() 传额外参数。
    # 例如：
    # --eval-options jsonfile_prefix=work_dirs/results

    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    # 可选参数：--launcher。
    # 指定分布式启动方式。
    # none：不使用分布式。
    # pytorch：PyTorch 分布式，dist_test.sh 默认传这个。
    # slurm：Slurm 集群启动。
    # mpi：MPI 启动。

    parser.add_argument("--local_rank", type=int, default=0)
    # 可选参数：--local_rank。
    # 当前进程在本机上的 rank。
    # 分布式启动器通常会传入这个参数或环境变量 LOCAL_RANK。

    parser.add_argument("--result_file", type=str, default=None)
    # 可选参数：--result_file。
    # 如果已经有保存好的推理结果文件，可以直接加载它，不重新跑模型。
    # 例如：
    # --result_file results.pkl --eval bbox
    # 这样会直接读取 results.pkl，然后调用 dataset.evaluate()。

    parser.add_argument("--show_only", action="store_true")
    # 可选参数：--show_only。
    # 只做可视化展示，不做正式 evaluation / format_only。
    # 后面会调用 dataset.show(outputs, show=True, **eval_kwargs)。

    args = parser.parse_args()
    # 解析命令行参数，得到 args 对象。

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    # 如果环境变量里没有 LOCAL_RANK，就用 args.local_rank 设置。
    # 这是为了兼容不同版本的分布式启动方式。

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    # 如果用户同时传了 --options 和 --eval-options，直接报错。
    # 因为这两个参数功能重复，而且 --options 已经过时。

    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    # 如果用户只传了旧参数 --options，就给出 warning。
    # 然后把 args.options 转成 args.eval_options，后续统一处理。

    return args
    # 返回解析好的参数对象。


def main():
    # 主函数。
    # 测试 / 评估主逻辑从这里开始。

    args = parse_args()
    # 首先解析命令行参数。

    assert (
        args.out or args.eval or args.format_only or args.show or args.show_dir
    ), (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )
    # 断言：用户至少要指定一种操作。
    # 也就是测试完之后，你到底要干什么？
    # 你必须至少传下面之一：
    # --out          保存 pkl 结果
    # --eval         计算指标
    # --format-only  只格式化结果
    # --show         显示结果
    # --show-dir     保存可视化结果
    #
    # 注意：--result_file 不在这个 assert 里面。
    # 所以即使你传了 --result_file，也还需要传 --eval / --show-dir / --format-only 等操作之一。

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")
    # --eval 和 --format-only 不能同时使用。
    # 因为一个是计算指标，一个是只格式化结果。
    # 二者语义冲突。

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")
    # 如果指定了 --out，则输出文件必须以 .pkl 或 .pickle 结尾。
    # 因为这里用 mmcv.dump 保存 pickle 格式结果。

    cfg = Config.fromfile(args.config)
    # 从配置文件读取 cfg。
    # 例如：
    # cfg = Config.fromfile("projects/configs/sparsedrive_small_stage2.py")
    # 读取后 cfg.model、cfg.data.test、cfg.evaluation 等字段都可以访问。

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # 如果命令行传入 --cfg-options，则把这些设置合并进 cfg。
    # 命令行参数优先级高于 config 文件本身。

    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        # 如果 config 里有 custom_imports，就导入这个工具函数。

        import_modules_from_strings(**cfg["custom_imports"])
        # 根据字符串路径动态 import 一些模块。
        # 目的是让自定义类注册进 OpenMMLab 的 Registry。

    # import modules from plguin/xx, registry will be updated
    # 注释里的 plguin 是 plugin 拼写错误。
    # 下面这段是 SparseDrive 的关键逻辑：导入自定义插件模块。
    if hasattr(cfg, "plugin"):
        # 如果 config 里有 plugin 字段。

        if cfg.plugin:
            # 如果 plugin=True，说明需要导入自定义项目模块。

            import importlib
            # importlib 用于动态导入 Python 模块。

            if hasattr(cfg, "plugin_dir"):
                # 如果 config 里指定了 plugin_dir，就从 plugin_dir 推导模块路径。

                plugin_dir = cfg.plugin_dir
                # 读取插件目录。
                # SparseDrive 通常类似：
                # plugin_dir = "projects/mmdet3d_plugin/"

                _module_dir = os.path.dirname(plugin_dir)
                # 取 plugin_dir 的目录名。
                # 如果 plugin_dir = "projects/mmdet3d_plugin/"，
                # os.path.dirname(plugin_dir) = "projects/mmdet3d_plugin"

                _module_dir = _module_dir.split("/")
                # 按 "/" 切分路径。
                # "projects/mmdet3d_plugin" -> ["projects", "mmdet3d_plugin"]

                _module_path = _module_dir[0]
                # 初始化模块路径。
                # 此时 _module_path = "projects"

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                # 把文件路径转换成 Python import 路径。
                # ["projects", "mmdet3d_plugin"] -> "projects.mmdet3d_plugin"

                print(_module_path)
                # 打印将要导入的模块路径。
                # 通常输出：
                # projects.mmdet3d_plugin

                plg_lib = importlib.import_module(_module_path)
                # 真正导入 SparseDrive 插件。
                # 这一步会触发很多 __init__.py，使自定义模型、dataset、head、loss 等注册进 Registry。
                # 如果不导入，build_detector / build_dataset 可能找不到 SparseDrive 自定义类。

            else:
                # 如果 config 中没有 plugin_dir，就用 config 文件所在目录推导模块路径。

                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                # 获取 config 文件所在目录。
                # 例如：
                # projects/configs

                _module_dir = _module_dir.split("/")
                # 切分路径。
                # "projects/configs" -> ["projects", "configs"]

                _module_path = _module_dir[0]
                # 初始化模块路径。
                # _module_path = "projects"

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                # 拼成 Python import 路径。
                # "projects.configs"

                print(_module_path)
                # 打印模块路径。

                plg_lib = importlib.import_module(_module_path)
                # 导入该模块。
                # 但对 SparseDrive 来说，一般 config 里会有 plugin_dir，
                # 也就是更常见地导入 projects.mmdet3d_plugin。

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    # 如果 config 里 cudnn_benchmark=True，则开启 CUDNN benchmark。
    # 当输入图像尺寸固定时，可以提升卷积速度。
    # 测试时通常图像尺寸固定，因此有可能加速。

    cfg.model.pretrained = None
    # 测试时不需要加载 backbone 预训练权重。
    # 因为后面会加载完整 checkpoint。
    # 如果不设为 None，有些模型构建时可能会额外加载 pretrained，造成冲突或浪费时间。

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    # 默认测试时每张 GPU 每次处理 1 个 sample。
    # 对自动驾驶多相机 3D 感知模型来说，单样本显存很大，因此 samples_per_gpu 通常为 1。

    if isinstance(cfg.data.test, dict):
        # 如果 cfg.data.test 是一个 dict，说明只有一个测试数据集配置。

        cfg.data.test.test_mode = True
        # 设置测试集为 test_mode=True。
        # 这样 dataset 会走测试逻辑，不会返回训练时需要的 GT loss 字段。

        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        # 从 cfg.data.test 里取出 samples_per_gpu。
        # 如果没有设置，则默认 1。
        # pop 的意思是取出来并从 cfg.data.test 删除这个 key。

        if samples_per_gpu > 1:
            # 如果每张 GPU 一次处理多个 sample。

            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline
            )
            # 把 pipeline 中的 ImageToTensor 替换为 DefaultFormatBundle。
            # 原因：ImageToTensor 通常只适合 samples_per_gpu=1。
            # 当 batch size > 1 时，需要 DefaultFormatBundle 正确打包 batch。

    elif isinstance(cfg.data.test, list):
        # 如果 cfg.data.test 是 list，说明可能有多个测试数据集拼接。

        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        # 给每个测试数据集配置都设置 test_mode=True。

        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        # 从每个测试数据集配置里取 samples_per_gpu。
        # 如果多个数据集设置不同，就取最大值。

        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
        # 如果 samples_per_gpu > 1，
        # 则给每个测试数据集的 pipeline 都替换 ImageToTensor。

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    # 如果 launcher 是 none，说明不使用分布式。
    # 例如直接：
    # python tools/test.py config.py ckpt.pth --eval bbox

    else:
        distributed = True
        # 如果 launcher 不是 none，则启用分布式。
        # dist_test.sh 会传 --launcher pytorch，因此通常会走这里。

        init_dist(args.launcher, **cfg.dist_params)
        # 初始化分布式环境。
        # args.launcher 可以是 pytorch / slurm / mpi。
        # cfg.dist_params 通常是 dict(backend="nccl")。

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
    # 设置随机种子。
    # 如果 deterministic=True，则同时设置 CUDNN 确定性选项。

    # set work dir
    if cfg.get('work_dir', None) is None:
        # 如果 config 里没有设置 work_dir。

        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
        # 根据 config 文件名自动生成 work_dir。
        # 例如：
        # config = projects/configs/sparsedrive_small_stage2.py
        # cfg.work_dir = ./work_dirs/sparsedrive_small_stage2

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # 创建 work_dir 文件夹。
    # 如果已经存在，不会报错。

    cfg.data.test.work_dir = cfg.work_dir
    # 把 work_dir 写入测试 dataset 配置。
    # SparseDrive 的 dataset / evaluator / show / format_results 可能需要用 work_dir 保存结果。

    print('work_dir: ', cfg.work_dir)
    # 打印当前 work_dir。

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # 根据 cfg.data.test 构建测试数据集。
    # SparseDrive 中这里通常是 nuScenes 相关 dataset。
    # 它会读取 info pkl、多相机图像路径、标注、地图信息、轨迹信息等。

    print("distributed:", distributed)
    # 打印当前是否为分布式测试。

    if distributed:
        # 如果是分布式测试。

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
        # 使用 SparseDrive 自定义 build_dataloader。
        # 参数说明：
        # dataset：刚刚构建的测试集。
        # samples_per_gpu：每张 GPU 的 batch size。
        # workers_per_gpu：每张 GPU 对应多少个 dataloader worker。
        # dist=True：分布式模式。
        # shuffle=False：测试不能打乱顺序，否则结果和 sample 对不上。
        # nonshuffler_sampler=dict(type="DistributedSampler")：
        #   指定非 shuffle 的分布式采样器。
        #   每个 rank 负责测试集的一部分，最后再汇总。

    else:
        # 如果不是分布式测试。

        data_loader = build_dataloader_origin(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        # 使用 MMDetection 原生 build_dataloader。
        # 单卡测试不需要 SparseDrive 自定义分布式 sampler。
        # shuffle=False 仍然很重要，保证输出顺序稳定。

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # 测试时不需要 train_cfg。
    # 这可以避免构建训练专用模块或使用训练配置。
    # 模型只需要 test_cfg。

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    # 根据 cfg.model 构建模型。
    # 在 SparseDrive 中通常构建的是 SparseDrive 模型：
    # img_backbone + img_neck + SparseDriveHead + depth_branch 等。
    # 这里只传 test_cfg，不传 train_cfg。

    # model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # 被注释掉的备用写法。
    # 有些 OpenMMLab 项目使用 build_model。
    # 这里实际使用的是 build_detector。

    fp16_cfg = cfg.get("fp16", None)
    # 从 config 中读取 fp16 配置。
    # 如果 config 里有 fp16=dict(...)，说明可能要用半精度。

    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 如果启用了 fp16，就把模型包装成 fp16 模式。
    # 注意：这只是模型层面的 fp16 包装。
    # 具体是否稳定要看模型和 checkpoint 是否支持。

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # 加载 checkpoint 权重到 model。
    # map_location="cpu" 表示先把权重加载到 CPU，再由后续 DataParallel/DDP 移到 GPU。
    # args.checkpoint 就是命令行第二个参数。

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # 如果传了 --fuse-conv-bn，就融合 Conv 和 BN。
    # 通常只在测试 / 部署阶段使用。
    # 训练阶段不能这样做，因为 BN 还需要更新统计量。

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # 兼容旧版本 checkpoint。
    # 有些旧 checkpoint 的 meta 里没有保存 CLASSES。
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    # 如果 checkpoint meta 里有 CLASSES，就使用 checkpoint 里的类别名。

    else:
        model.CLASSES = dataset.CLASSES
    # 如果 checkpoint 里没有 CLASSES，就使用 dataset.CLASSES。
    # 这样后处理、评估、可视化时能知道类别名称。

    # palette for visualization in segmentation tasks
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    # 如果 checkpoint meta 里有 PALETTE，就使用它。
    # PALETTE 主要是分割任务可视化用的颜色表。

    elif hasattr(dataset, "PALETTE"):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    # 如果 checkpoint 没有 PALETTE，但 dataset 有 PALETTE，就从 dataset 读取。
    # SparseDrive 主要不是语义分割模型，这部分更多是继承自 MMDetection 通用脚本。

    if args.result_file is not None:
        # 如果用户传了 --result_file，说明已有推理结果文件。

        # outputs = torch.load(args.result_file)
        # 原本可以用 torch.load 读结果，但这里被注释掉。

        outputs = mmcv.load(args.result_file)
        # 使用 mmcv.load 读取结果文件。
        # 这样可以直接评估已有 outputs，不重新跑模型推理。
        # 适合：
        # 1. 推理很慢，不想重复跑；
        # 2. 只想换 eval-options；
        # 3. 只想重新 format/show。

    elif not distributed:
        # 如果没有 result_file，并且不是分布式测试。

        model = MMDataParallel(model, device_ids=[0])
        # 用 MMDataParallel 包装模型。
        # device_ids=[0] 表示使用 GPU 0。
        # 这是单卡测试常用包装方式。

        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        # 调用 MMDetection 原生单卡测试函数。
        # 它会遍历 data_loader，执行 model(return_loss=False, rescale=True, ...)
        # 最后返回 outputs。
        # 如果 args.show=True 或 args.show_dir 不为空，还会处理可视化。

    else:
        # 如果没有 result_file，并且是分布式多卡测试。

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        # 把模型移动到当前 GPU，并用 MMDistributedDataParallel 包装。
        # torch.cuda.current_device() 是当前 rank 对应的 GPU。
        # broadcast_buffers=False 表示不广播 BN 等 buffer。
        # 测试阶段通常不需要同步这些 buffer。

        outputs = custom_multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
        )
        # 调用 SparseDrive 自定义多卡测试函数。
        # 每个 GPU / rank 处理测试集的一部分。
        # 最后通过 tmpdir 或 GPU 通信汇总结果。
        # 这里不用 mmdet.apis.multi_gpu_test，而是用 custom_multi_gpu_test。

    rank, _ = get_dist_info()
    # 获取当前进程 rank 和 world_size。
    # rank=0 通常是主进程。
    # 这里只关心 rank，所以 world_size 用 _ 忽略。

    if rank == 0:
        # 只有主进程负责保存、评估、打印结果。
        # 避免多卡每个进程都重复写文件、重复打印指标。

        if args.out:
            print(f"\nwriting results to {args.out}")
            # 打印提示：正在写结果文件。

            mmcv.dump(outputs, args.out)
            # 把 outputs 保存为 pkl / pickle 文件。
            # 之后可以用 --result_file 重新读取，避免重复推理。

        kwargs = {} if args.eval_options is None else args.eval_options
        # 处理 eval-options。
        # 如果用户没传 --eval-options，则 kwargs 为空字典。
        # 如果传了，则作为额外参数传给 dataset.evaluate / dataset.format_results / dataset.show。

        if args.show_only:
            # 如果传了 --show_only，则只做可视化。

            eval_kwargs = cfg.get("evaluation", {}).copy()
            # 从 cfg.evaluation 复制一份评估配置。
            # copy 是为了不要修改原始 cfg。

            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            # 删除 EvalHook 专用参数。
            # 这些参数是训练过程中 EvalHook 使用的，
            # 但 dataset.show() 不需要，甚至可能因为多余参数报错。
            #
            # pop(key, None) 表示：
            # 如果 key 存在就删除；
            # 如果不存在也不报错。

            eval_kwargs.update(kwargs)
            # 把命令行 --eval-options 传入的额外参数合并进去。

            dataset.show(outputs, show=True, **eval_kwargs)
            # 调用 dataset.show 做可视化。
            # show=True 表示显示结果。
            # 对无显示器服务器环境可能不方便。

        elif args.format_only:
            dataset.format_results(outputs, **kwargs)
            # 如果传了 --format-only，则只格式化结果，不计算指标。
            # 常见用途：生成 nuScenes 官方提交格式文件。

        elif args.eval:
            # 如果传了 --eval，则计算评估指标。

            eval_kwargs = cfg.get("evaluation", {}).copy()
            # 从 cfg.evaluation 复制评估配置。

            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            # 删除训练 EvalHook 专用参数。
            # dataset.evaluate() 不需要这些字段。

            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            # 把 metric 和 eval-options 合并进 eval_kwargs。
            # metric=args.eval，比如 ["bbox"]。

            print(eval_kwargs)
            # 打印最终传给 dataset.evaluate 的参数。
            # 有助于确认到底评估了什么 metric。

            results_dict = dataset.evaluate(outputs, **eval_kwargs)
            # 调用 dataset.evaluate 计算指标。
            # 对 SparseDrive / nuScenes 来说，这里会进入自定义 dataset 的 evaluate。
            # 可能计算 detection、tracking、map、motion、planning 等相关指标，
            # 具体取决于 dataset.evaluate 的实现和传入 metric。

            print(results_dict)
            # 打印最终评估结果字典。


if __name__ == "__main__":
    # Python 脚本入口。
    # 只有直接运行 test.py 时，才会进入这里。

    torch.multiprocessing.set_start_method(
        "fork"
    )  # use fork workers_per_gpu can be > 1
    # 设置 PyTorch 多进程启动方式为 fork。
    # 这样 dataloader 的 workers_per_gpu 可以大于 1。
    # fork 启动快，但在某些 CUDA / 多线程场景下可能需要注意稳定性。
    # SparseDrive 这里和 train.py 一样，显式使用 fork。

    main()
    # 调用主函数，开始测试 / 评估流程。