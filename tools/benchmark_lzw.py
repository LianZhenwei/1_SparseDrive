# OpenMMLab 版权声明
# 说明这个脚本来源于 OpenMMLab/MMDetection 系列工程风格
# Copyright (c) OpenMMLab. All rights reserved.


# 导入 argparse
# 用于解析命令行参数，例如 config、checkpoint、samples 等
import argparse

# 导入 time
# 用于统计推理耗时，例如 time.perf_counter()
import time

# 导入 PyTorch
# 用于模型推理、CUDA 同步、显存统计等
import torch

# 从 mmcv 导入 Config
# Config.fromfile() 可以读取 .py 配置文件
from mmcv import Config

# MMDataParallel 是 MMCV 的单机单卡/多卡 DataParallel 封装
# 这里 benchmark 只用 device_ids=[0]，即单 GPU 测 FPS
from mmcv.parallel import MMDataParallel

# load_checkpoint：加载模型权重
# wrap_fp16_model：让模型支持 fp16 混合精度推理
from mmcv.runner import load_checkpoint, wrap_fp16_model

# 导入 sys
# 用于修改 Python 模块搜索路径
import sys

# 将当前目录加入 Python 搜索路径
# 这样可以 import projects.mmdet3d_plugin 里的自定义模块
sys.path.append('.')

# 导入 SparseDrive 自定义 dataloader 构建函数
# 和 mmdet 原生 build_dataloader 不同，这个可能适配了时序/自定义 sampler 等逻辑
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

# 导入 SparseDrive 自定义 dataset 构建函数
# 用于构建 NuScenes3DDataset 等自定义数据集
from projects.mmdet3d_plugin.datasets import custom_build_dataset

# 从 mmdet 导入模型构建函数
# 根据 cfg.model 构建 SparseDrive 模型
from mmdet.models import build_detector

# 导入 MMCV 的 FLOPs 统计工具
# add_flops_counting_methods 会给模型注册统计 FLOPs 的 hook/method
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods

# scatter 用于把 CPU 上的数据分发到指定 GPU
# 这里在 get_flops_params() 中把一个 batch 的 data 放到 gpu_id=0 上
from mmcv.parallel import scatter


def parse_args():
    # 定义命令行参数解析函数

    # 创建 argparse 参数解析器
    # description 是命令行帮助信息里的脚本说明
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')

    # 位置参数：配置文件路径
    # 例如 projects/configs/sparsedrive_small_stage2.py
    parser.add_argument('config', help='test config file path')

    # 可选参数：checkpoint 权重路径
    # 如果不传，则只构建模型，不加载训练权重
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')

    # 可选参数：benchmark 用多少个样本
    # 默认 1000
    # 注意：这里没有写 type=int，所以如果命令行显式传入 --samples 100，
    # args.samples 会是字符串 "100"，后面和 int 比较会出问题
    parser.add_argument('--samples', default=1000, help='samples to benchmark')

    # 可选参数：每隔多少个样本打印一次 FPS 日志
    # 默认 50
    # 注意：这里也没有写 type=int，显式传参时会变成字符串
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')

    # 可选参数：是否融合 Conv 和 BN
    # action='store_true' 表示命令行出现 --fuse-conv-bn 时为 True，否则为 False
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')

    # 解析命令行参数
    args = parser.parse_args()

    # 返回解析结果
    return args


def get_max_memory(model):
    # 获取当前模型所在设备的 CUDA 最大显存占用，单位 MB

    # 从 MMDataParallel 模型中尝试读取 output_device 属性
    # 如果没有这个属性，就返回 None
    device = getattr(model, 'output_device', None)

    # 获取当前 CUDA 设备上已经分配过的最大显存，单位是 byte
    mem = torch.cuda.max_memory_allocated(device=device)

    # 将 byte 转换成 MB
    # 1024 * 1024 byte = 1 MB
    # 这里包成 torch.tensor，dtype=torch.int，并放到同一个 device 上
    mem_mb = torch.tensor([mem / (1024 * 1024)],
        dtype=torch.int,
        device=device)

    # 返回 Python int
    return mem_mb.item()


def main():
    # 主函数

    # 解析命令行参数
    args = parse_args()

    # 先统计模型 FLOPs 和参数量
    get_flops_params(args)

    # 再统计模型推理显存和 FPS
    get_mem_fps(args)


def get_mem_fps(args):
    # 统计模型推理速度 FPS 和最大显存占用

    # 从配置文件读取 cfg
    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    # 如果配置文件中 cudnn_benchmark=True，则开启 cudnn benchmark
    # 对固定输入尺寸可能加速卷积
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 测试/benchmark 时不需要 backbone 的预训练权重路径
    # 因为如果提供 checkpoint，会从 checkpoint 加载完整模型权重
    cfg.model.pretrained = None

    # 将 test 数据集设置为 test_mode
    # test_mode=True 时，dataset 通常不会读取训练标签或不会做训练增强
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # 打印测试集配置，方便确认 benchmark 用的是哪个数据集和 pipeline
    print(cfg.data.test)

    # 根据 cfg.data.test 构建测试数据集
    dataset = custom_build_dataset(cfg.data.test)

    # 构建 dataloader
    data_loader = build_dataloader(
        # 数据集对象
        dataset,

        # 每张 GPU 一个样本
        # 这个 benchmark 脚本默认只支持 samples_per_gpu=1
        samples_per_gpu=1,

        # 每张 GPU 的 dataloader worker 数量
        workers_per_gpu=cfg.data.workers_per_gpu,

        # 非分布式 benchmark
        dist=False,

        # 不打乱顺序
        shuffle=False)

    # build the model and load checkpoint
    # 推理时不需要 train_cfg
    cfg.model.train_cfg = None

    # 根据 cfg.model 构建检测器
    # 对 SparseDrive 来说，这里会构建 SparseDrive 模型
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # 获取 fp16 配置
    fp16_cfg = cfg.get('fp16', None)

    # 如果配置文件中有 fp16
    if fp16_cfg is not None:
        # 将模型包装成支持 fp16 的形式
        wrap_fp16_model(model)

    # 如果传入了 checkpoint
    if args.checkpoint is not None:
        # 加载 checkpoint 权重
        # map_location='cpu' 表示先加载到 CPU，再由后续 model.cuda/DataParallel 放到 GPU
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 如果启用 Conv-BN 融合，理论上这里应该 fuse
    # 但源码中这两行被注释掉了，所以 --fuse-conv-bn 实际不会生效
    # if args.fuse_conv_bn:
    #     model = fuse_module(model)

    # 用 MMDataParallel 包装模型
    # device_ids=[0] 表示使用第 0 张 GPU
    model = MMDataParallel(model, device_ids=[0])

    # 设置为 eval 模式
    # 关闭 dropout，BN 使用 running mean/var
    model.eval()

    # the first several iterations may be very slow so skip them
    # 前几次推理可能包含 CUDA kernel 初始化、缓存建立等额外耗时
    # 所以跳过前 5 次 warmup，不计入 FPS
    num_warmup = 5

    # 纯推理时间累计
    pure_inf_time = 0

    # benchmark with several samples and take the average
    # 最大显存记录
    max_memory = 0

    # 遍历 dataloader
    for i, data in enumerate(data_loader):

        # 这里本来可以在开始计时前同步 CUDA
        # 但源码注释掉了
        # torch.cuda.synchronize()

        # 推理阶段关闭梯度计算，减少显存和加速
        with torch.no_grad():

            # 记录开始时间
            start_time = time.perf_counter()

            # 执行一次模型推理
            # return_loss=False 表示测试/推理模式
            # rescale=True 表示结果通常会映射回原图尺度
            # **data 把 batch 数据字典展开传给模型 forward
            model(return_loss=False, rescale=True, **data)

            # CUDA 是异步执行的
            # synchronize 保证 GPU 推理真正结束后再计时
            torch.cuda.synchronize()

            # 计算当前样本推理耗时
            elapsed = time.perf_counter() - start_time

            # 更新最大显存占用
            max_memory = max(max_memory, get_max_memory(model))

        # 如果已经过了 warmup 阶段
        if i >= num_warmup:

            # 累计纯推理时间
            pure_inf_time += elapsed

            # 每隔 log_interval 打印一次速度
            if (i + 1) % args.log_interval == 0:

                # FPS = 有效样本数 / 有效推理时间
                fps = (i + 1 - num_warmup) / pure_inf_time

                # 打印当前处理进度、FPS、最大显存
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s, '
                      f"gpu mem: {max_memory} M")

        # 如果已经达到指定 benchmark 样本数
        if (i + 1) == args.samples:

            # 注意：这里又把当前 elapsed 加了一次
            # 如果 i >= num_warmup，那么上面已经加过一次 elapsed
            # 因此这里可能会重复累计最后一次耗时
            pure_inf_time += elapsed

            # 计算总 FPS
            fps = (i + 1 - num_warmup) / pure_inf_time

            # 打印总 FPS
            print(f'Overall fps: {fps:.1f} img / s')

            # 结束循环
            break


def get_flops_params(args):
    # 统计模型各部分 FLOPs 和参数量

    # 指定使用第 0 张 GPU
    gpu_id = 0

    # 读取配置文件
    cfg = Config.fromfile(args.config)

    # 构建验证集 dataset
    # 这里用 cfg.data.val，而不是 cfg.data.test
    # 因为 FLOPs 统计只需要取一个样本
    dataset = custom_build_dataset(cfg.data.val)

    # 构建 dataloader
    dataloader = build_dataloader(
        # 验证集
        dataset,

        # 每次一个样本
        samples_per_gpu=1,

        # FLOPs 统计不需要多 worker
        workers_per_gpu=0,

        # 非分布式
        dist=False,

        # 不打乱
        shuffle=False,
    )

    # 创建 dataloader 迭代器
    data_iter = dataloader.__iter__()

    # 取第一个 batch
    data = next(data_iter)

    # 将数据 scatter 到 GPU 0
    # scatter 返回 list，这里取第 0 个 GPU 上的数据
    data = scatter(data, [gpu_id])[0]

    # 推理时不需要 train_cfg
    cfg.model.train_cfg = None

    # 构建模型
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # 如果配置中有 fp16
    fp16_cfg = cfg.get('fp16', None)

    # 包装 fp16 模型
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # 如果提供 checkpoint
    if args.checkpoint is not None:
        # 加载 checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    # 将模型放到 GPU 0
    model = model.cuda(gpu_id)

    # 设置 eval 模式
    model.eval()

    # 手动估计 bilinear grid sampling 的 FLOPs 系数
    # 每个采样点大约按 11 次操作估算
    bilinear_flops = 11

    # 计算 detection deformable aggregation 的关键点数量
    # det key points = 可学习点数量 + 固定点数量
    num_key_pts_det = (
        cfg.model["head"]['det_head']["deformable_model"]["kps_generator"]["num_learnable_pts"]
        + len(cfg.model["head"]['det_head']["deformable_model"]["kps_generator"]["fix_scale"])
    )

    # 手动估计 detection 分支 deformable aggregation 的 FLOPs
    deformable_agg_flops_det = (
        # decoder 层数
        cfg.num_decoder

        # 每个点的 embedding 维度
        * cfg.embed_dims

        # FPN level 数量
        * cfg.num_levels

        # det anchor 数量，例如 900
        * cfg.model["head"]['det_head']["instance_bank"]["num_anchor"]

        # 相机数量，例如 6
        * cfg.model["head"]['det_head']["deformable_model"]["num_cams"]

        # 每个 box 的关键点数量
        * num_key_pts_det

        # 每个双线性采样点估算 FLOPs
        * bilinear_flops
    )

    # 计算 map deformable aggregation 的关键点数量
    # map key points = (可学习高度点数量 + 固定高度数量) * 每条线采样点数量
    num_key_pts_map = (
        cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["num_learnable_pts"]
        + len(cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["fix_height"])
    ) * cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["num_sample"]

    # 手动估计 map 分支 deformable aggregation 的 FLOPs
    deformable_agg_flops_map = (
        # decoder 层数
        cfg.num_decoder

        # embedding 维度
        * cfg.embed_dims

        # FPN level 数
        * cfg.num_levels

        # map anchor 数量，例如 100
        * cfg.model["head"]['map_head']["instance_bank"]["num_anchor"]

        # 相机数量
        * cfg.model["head"]['map_head']["deformable_model"]["num_cams"]

        # map 每条线的总关键点数量
        * num_key_pts_map

        # 双线性采样 FLOPs 估计系数
        * bilinear_flops
    )

    # detection + map 的 deformable aggregation FLOPs
    deformable_agg_flops = deformable_agg_flops_det + deformable_agg_flops_map

    # 依次统计整个模型和各个子模块的 FLOPs/参数量
    for module in ["total", "img_backbone", "img_neck", "head"]:

        # 如果不是 total
        if module != "total":
            # 取出模型对应子模块，并添加 FLOPs 统计方法
            flops_model = add_flops_counting_methods(getattr(model, module))

        # 如果是 total
        else:
            # 对整个模型添加 FLOPs 统计方法
            flops_model = add_flops_counting_methods(model)

        # 设置 eval 模式
        flops_model.eval()

        # 开始统计 FLOPs
        flops_model.start_flops_count()
        
        # 如果统计 img_backbone
        if module == "img_backbone":
            # 只前向 backbone
            # data["img"] 原始通常是 [B, num_cam, C, H, W]
            # flatten(0, 1) 变成 [B*num_cam, C, H, W]
            flops_model(data["img"].flatten(0, 1))

        # 如果统计 img_neck
        elif module == "img_neck":
            # 先用 backbone 提取特征，再统计 neck 的 FLOPs
            flops_model(model.img_backbone(data["img"].flatten(0, 1)))

        # 如果统计 head
        elif module == "head":
            # 先提取图像特征，再统计 head 前向
            # model.extract_feat(data["img"], metas=data) 输出多尺度特征
            # 第二个参数 data 是 metas
            flops_model(model.extract_feat(data["img"], metas=data), data)

        # 如果统计 total
        else:
            # 整个模型完整前向
            flops_model(**data)

        # 计算平均 FLOPs 和参数量
        flops_count, params_count = flops_model.compute_average_flops_cost()

        # 乘以 batch_counter
        # 因为 compute_average_flops_cost 通常给的是平均单 batch 统计，
        # 这里乘回来得到当前实际前向累计 FLOPs
        flops_count *= flops_model.__batch_counter__

        # 停止 FLOPs 统计
        flops_model.stop_flops_count()

        # 如果统计的是 head 或 total
        if module == "head" or module == "total":
            # 手动加上 deformable aggregation 的 FLOPs
            # 因为自定义 CUDA op / grid_sample 可能无法被 mmcv flops_counter 正确统计
            flops_count += deformable_agg_flops

        # 如果统计的是 total
        if module == "total":
            # 保存总 FLOPs，用于后续计算各模块占比
            total_flops = flops_count

            # 保存总参数量，用于后续计算各模块占比
            total_params = params_count

        # 打印当前模块复杂度
        print(
            f"{module:<13} complexity: "
            f"FLOPs={flops_count/ 10.**9:>8.4f} G / {flops_count/total_flops*100:>6.2f}%, "
            f"Params={params_count/10**6:>8.4f} M / {params_count/total_params*100:>6.2f}%."
        )

# Python 文件入口
if __name__ == '__main__':
    # 设置 multiprocessing 启动方式为 fork
    # 注释说明：使用 fork 时 workers_per_gpu 可以大于 1
    torch.multiprocessing.set_start_method(
        "fork"
    )  # use fork workers_per_gpu can be > 1

    # 执行主函数
    main()