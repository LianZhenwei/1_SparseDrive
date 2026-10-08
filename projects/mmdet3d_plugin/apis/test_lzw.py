# ---------------------------------------------
# OpenMMLab 版权声明
# 表示该文件基于 OpenMMLab / MMDetection 系列工程
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

# 表示该文件被 Zhiqi Li 修改过
# SparseDrive / BEVFormer 这类项目很多自定义 API 都继承了类似代码风格
#  Modified by Zhiqi Li
# ---------------------------------------------


# 导入 os.path，并简写为 osp
# 用于路径拼接，例如 osp.join(tmpdir, f"part_{rank}.pkl")
import os.path as osp

# 导入 pickle
# 当前文件里没有直接使用 pickle，属于冗余导入
# mmcv.dump/mmcv.load 内部可能会用 pickle，但这里没有直接调用 pickle
import pickle

# 导入 shutil
# 用于删除临时文件夹，例如 shutil.rmtree(tmpdir)
import shutil

# 导入 tempfile
# 用于创建临时目录 tempfile.mkdtemp()
import tempfile

# 导入 time
# 用于 sleep
# 这里 time.sleep(2) 用来规避某些分布式场景下的死锁问题
import time


# 导入 mmcv
# 用于：
# 1. ProgressBar 进度条
# 2. mkdir_or_exist 创建目录
# 3. dump/load 保存和读取 pkl 文件
import mmcv

# 导入 PyTorch
# 用于 no_grad、cuda tensor、分布式通信辅助等
import torch

# 导入 torch.distributed，并简写为 dist
# 用于分布式通信，例如 dist.broadcast() 和 dist.barrier()
import torch.distributed as dist

# 从 mmcv.image 导入 tensor2imgs
# 当前文件中没有实际使用，属于冗余导入
# 在原始 MMDetection single_gpu_test 中经常用于把 tensor 图像转回 numpy 图像可视化
from mmcv.image import tensor2imgs

# 从 mmcv.runner 导入 get_dist_info
# 用于获取当前进程 rank 和总进程数 world_size
from mmcv.runner import get_dist_info


# 从 mmdet.core 导入 encode_mask_results
# 当前文件没有实际使用，因为作者自己写了 custom_encode_mask_results()
# 属于冗余导入
from mmdet.core import encode_mask_results


# 再次导入 mmcv
# 前面已经 import mmcv 了，这里重复导入，属于冗余
import mmcv

# 导入 numpy
# custom_encode_mask_results() 中用于把 mask 转成 np.array
import numpy as np

# 导入 pycocotools 的 mask 工具
# 用于把二值 mask 编码成 RLE 格式
# RLE = Run-Length Encoding，常用于 COCO 格式的 mask 保存
import pycocotools.mask as mask_util


def custom_encode_mask_results(mask_results):
    # 自定义 mask 编码函数
    #
    # 作用：
    # 把语义分割/实例分割得到的 bitmap mask 编码成 RLE 格式。
    #
    # 注意：
    # SparseDrive 主要是 3D 检测、tracking、map、motion、planning，
    # 正常情况下你可能很少用到 mask_results。
    # 这段更像从 OpenMMLab 通用测试代码继承/改造而来。

    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """

    # 这里直接把传入的 mask_results 赋给 cls_segms
    # cls_segms 可以理解成每个类别对应的 mask 结果
    cls_segms = mask_results

    # mask 类别数量
    # 当前变量 num_classes 后面没有实际使用
    num_classes = len(cls_segms)

    # 用于保存编码后的 mask 结果
    encoded_mask_results = []

    # 遍历每一类 mask
    for i in range(len(cls_segms)):

        # 将第 i 类 mask 编码成 RLE
        encoded_mask_results.append(
            mask_util.encode(
                # 将 mask 转成 numpy array
                # cls_segms[i] 原本可能是 [H, W]
                # cls_segms[i][:, :, np.newaxis] 变成 [H, W, 1]
                #
                # order="F" 表示 Fortran 内存顺序
                # pycocotools 的 RLE 编码通常要求 Fortran contiguous
                #
                # dtype="uint8" 表示二值 mask 使用 uint8
                np.array(
                    cls_segms[i][:, :, np.newaxis], order="F", dtype="uint8"
                )
            )[0]
        )  # encoded with RLE

    # 返回编码后的 mask
    # 外面再套一层 list，格式上和 mmdet 的 mask result 结构保持兼容
    return [encoded_mask_results]


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    # 自定义多 GPU 测试函数
    #
    # 作用：
    # 1. 多 GPU 上分别跑模型推理
    # 2. 每个 GPU 得到自己负责的数据结果
    # 3. 把所有 GPU 的结果收集到 rank 0
    # 4. 返回完整验证集/测试集的预测结果
    #
    # 这个函数会在 tools/test.py 中被调用：
    # outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    # 将模型切换到 eval 模式
    # eval 模式下：
    # 1. Dropout 关闭
    # 2. BN 使用 running_mean/running_var
    # 3. 模型进入推理状态
    model.eval()

    # 用于保存 bbox 类结果
    # 对 SparseDrive 来说，bbox_results 通常会包含 det/map/motion/planning 等最终输出
    bbox_results = []

    # 用于保存 mask 类结果
    # SparseDrive 正常不太用 mask，但保留这个接口
    mask_results = []

    # 从 dataloader 中取出 dataset
    # 后面需要 len(dataset) 来知道总样本数量
    dataset = data_loader.dataset

    # 获取当前分布式进程信息
    # rank：当前进程编号
    # world_size：总进程数，也就是总 GPU 数
    rank, world_size = get_dist_info()

    # 只有 rank 0 负责显示进度条
    # 否则每个 GPU 都打印进度条会乱
    if rank == 0:
        # 创建进度条，总长度为 dataset 样本数
        prog_bar = mmcv.ProgressBar(len(dataset))

    # 暂停 2 秒
    # 注释说：某些情况下可以防止 deadlock
    # 分布式程序里，各进程启动速度不同，sleep 有时能避免同步问题
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    # 标记结果中是否真的有 mask
    have_mask = False

    # 遍历 data_loader
    # 每个 rank/GPU 只会拿到自己负责的一部分数据
    for i, data in enumerate(data_loader):

        # 推理时不需要梯度
        # 可以减少显存占用，加快速度
        with torch.no_grad():

            # 执行模型推理
            #
            # return_loss=False：
            #   表示测试模式，不计算训练 loss
            #
            # rescale=True：
            #   表示把预测结果恢复到原始图像/坐标尺度
            #
            # **data：
            #   data 是 dataloader 输出的字典
            #   展开后传入模型，例如 img、metas、gt 等
            result = model(return_loss=False, rescale=True, **data)

            # encode mask results
            # 下面开始处理模型输出结果

            # 如果模型输出是 dict
            if isinstance(result, dict):

                # 如果 dict 中有 bbox_results
                if "bbox_results" in result.keys():

                    # 取出 bbox_results
                    bbox_result = result["bbox_results"]

                    # 当前 batch size
                    # bbox_results 通常是 list，每个元素对应 batch 中一个样本
                    batch_size = len(result["bbox_results"])

                    # 将当前 batch 的 bbox 结果加入总列表
                    bbox_results.extend(bbox_result)

                # 如果 dict 中有 mask_results，并且不是 None
                if (
                    "mask_results" in result.keys()
                    and result["mask_results"] is not None
                ):

                    # 对 mask 结果做 RLE 编码
                    mask_result = custom_encode_mask_results(
                        result["mask_results"]
                    )

                    # 将 mask 结果加入总列表
                    mask_results.extend(mask_result)

                    # 标记当前测试结果中包含 mask
                    have_mask = True

            # 如果模型输出不是 dict
            else:

                # 对普通 MMDetection 模型来说，result 通常是 list
                # 每个元素对应 batch 中一个样本
                batch_size = len(result)

                # 把结果加入 bbox_results
                bbox_results.extend(result)


        # rank 0 更新进度条
        if rank == 0:

            # 每个 rank 都处理了 batch_size 个样本
            # world_size 个 rank 一共推进 batch_size * world_size 个样本
            for _ in range(batch_size * world_size):

                # 更新一次进度条
                prog_bar.update()


    # collect results from all ranks
    # 所有 GPU 都测试完自己那部分数据后，需要收集结果

    # 如果使用 GPU collect
    if gpu_collect:

        # 用 GPU 通信收集 bbox_results
        bbox_results = collect_results_gpu(bbox_results, len(dataset))

        # 如果有 mask
        if have_mask:
            # 用 GPU 通信收集 mask_results
            mask_results = collect_results_gpu(mask_results, len(dataset))

        # 如果没有 mask
        else:
            # mask_results 设为 None
            mask_results = None

    # 如果使用 CPU collect
    else:

        # 用 CPU 文件方式收集 bbox_results
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)

        # mask 结果使用另一个临时目录
        # 避免和 bbox 的 part_x.pkl 混在一起
        tmpdir = tmpdir + "_mask" if tmpdir is not None else None

        # 如果有 mask
        if have_mask:

            # 用 CPU 文件方式收集 mask_results
            mask_results = collect_results_cpu(
                mask_results, len(dataset), tmpdir
            )

        # 如果没有 mask
        else:

            # mask_results 设为 None
            mask_results = None


    # 如果没有 mask 结果
    if mask_results is None:

        # 只返回 bbox_results
        return bbox_results

    # 如果有 mask 结果
    # 返回一个 dict，同时包含 bbox 和 mask
    return {"bbox_results": bbox_results, "mask_results": mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    # 使用 CPU 文件系统收集多 GPU 测试结果
    #
    # 每个 rank/GPU 会把自己的 result_part 保存成：
    #   tmpdir/part_0.pkl
    #   tmpdir/part_1.pkl
    #   ...
    #
    # rank 0 等所有 rank 保存完后，再把它们全部读回来，拼成 ordered_results。

    # 获取当前 rank 和总进程数
    rank, world_size = get_dist_info()

    # create a tmp dir if it is not specified
    # 如果没有指定临时目录
    if tmpdir is None:

        # 临时目录字符串最大长度
        MAX_LEN = 512

        # 32 is whitespace
        # 创建一个长度为 512 的 uint8 tensor，全部填充 ASCII 空格 32
        # 这个 tensor 放在 CUDA 上，用于通过 dist.broadcast 广播临时目录路径
        dir_tensor = torch.full(
            (MAX_LEN,), 32, dtype=torch.uint8, device="cuda"
        )

        # 如果当前是 rank 0
        if rank == 0:

            # 创建 .dist_test 目录
            mmcv.mkdir_or_exist(".dist_test")

            # 在 .dist_test 下创建一个唯一临时目录
            tmpdir = tempfile.mkdtemp(dir=".dist_test")

            # 将 tmpdir 字符串编码成 bytearray，再转成 uint8 CUDA tensor
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
            )

            # 把路径内容写入 dir_tensor 前 len(tmpdir) 个位置
            dir_tensor[: len(tmpdir)] = tmpdir

        # rank 0 把 dir_tensor 广播给所有其他 rank
        # 这样每个 rank 都知道临时目录路径
        dist.broadcast(dir_tensor, 0)

        # 所有 rank 把 dir_tensor 转回字符串路径
        # cpu().numpy().tobytes().decode()：tensor -> bytes -> string
        # rstrip()：去掉末尾空格
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()

    # 如果用户指定了 tmpdir
    else:

        # 确保该目录存在
        mmcv.mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    # 每个 rank 把自己的结果保存成 part_rank.pkl
    mmcv.dump(result_part, osp.join(tmpdir, f"part_{rank}.pkl"))

    # 分布式同步屏障
    # 确保所有 rank 都写完 pkl 文件后，再继续往下走
    dist.barrier()

    # collect all parts
    # 收集所有 part 结果

    # 如果当前不是 rank 0
    if rank != 0:

        # 非 rank 0 不负责读取和合并结果，直接返回 None
        return None

    # 如果当前是 rank 0
    else:

        # load results of all parts from tmp dir
        # 用于保存每个 rank 的结果
        part_list = []

        # 遍历所有 rank
        for i in range(world_size):

            # 当前 rank 的 pkl 文件路径
            part_file = osp.join(tmpdir, f"part_{i}.pkl")

            # 读取该 rank 的结果，并加入 part_list
            part_list.append(mmcv.load(part_file))

        # sort the results
        # 保存合并后的有序结果
        ordered_results = []

        """
        bacause we change the sample of the evaluation stage to make sure that
        each gpu will handle continuous sample,
        """

        # 原始 MMDetection 常见写法是：
        # for res in zip(*part_list):
        #     ordered_results.extend(list(res))
        #
        # 这种写法适用于各 rank 间交错处理样本的情况：
        # rank0: sample 0, 2, 4
        # rank1: sample 1, 3, 5
        #
        # 但这里 SparseDrive 修改了 evaluation 阶段的采样方式，
        # 保证每个 GPU 处理连续样本：
        # rank0: sample 0, 1, 2
        # rank1: sample 3, 4, 5
        #
        # 所以不能 zip 交错合并，而是直接按 rank 顺序拼接。
        # for res in zip(*part_list):

        # 按 rank 顺序拼接每个 rank 的结果
        for res in part_list:

            # 把当前 rank 的结果转成 list 后加入 ordered_results
            ordered_results.extend(list(res))

        # the dataloader may pad some samples
        # dataloader 在分布式测试时可能为了凑齐 batch/rank 数量补了一些样本
        # 所以最终结果要裁剪回真实 dataset 长度 size
        ordered_results = ordered_results[:size]

        # remove tmp dir
        # 删除临时目录
        shutil.rmtree(tmpdir)

        # 返回完整结果
        return ordered_results


def collect_results_gpu(result_part, size):
    # GPU 收集结果函数
    #
    # 注意：这里实际上没有真正实现 GPU collect。
    # 它只是调用 collect_results_cpu(result_part, size)
    # 而且没有 return。
    #
    # 所以这个函数目前是一个不完整/有问题的实现。

    collect_results_cpu(result_part, size)

