# OpenMMLab 版权声明
# 说明该脚本来源于 OpenMMLab 工程体系
# Copyright (c) OpenMMLab. All rights reserved.


# 导入 argparse
# 用于解析命令行参数，例如 config、checkpoint、out
import argparse


# 导入 PyTorch
# 用于张量运算、创建参数、处理 Conv/BN 权重等
import torch

# 从 mmcv.runner 导入 save_checkpoint
# 用于把融合后的模型保存成新的 checkpoint 文件
from mmcv.runner import save_checkpoint

# 从 torch 中导入 nn，并命名为 nn
# 用于判断模块类型，例如 nn.Conv2d、nn.BatchNorm2d、nn.Identity
from torch import nn as nn


# 从 mmdet3d.apis 导入 init_model
# init_model 会根据 config 构建模型，并加载 checkpoint 权重
from mmdet3d.apis import init_model


def fuse_conv_bn(conv, bn):
    # 定义 Conv + BN 融合函数
    # 输入：
    #   conv：一个 nn.Conv2d 层
    #   bn：紧跟在 conv 后面的 BatchNorm2d 或 SyncBatchNorm 层
    #
    # 输出：
    #   融合 BN 参数后的新 conv
    #
    # 融合后可以把 BN 层替换成 Identity，从而推理时少算一个 BN

    """During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures."""

    # 取出卷积层的权重
    # conv_w shape 通常是 [out_channels, in_channels, kernel_h, kernel_w]
    conv_w = conv.weight

    # 取出卷积层 bias
    # 如果 conv 本身有 bias，就使用 conv.bias
    # 如果 conv 没有 bias，则创建一个和 bn.running_mean 同形状的 0 向量
    #
    # 注意：
    # 很多 Conv + BN 结构中的 Conv 会设置 bias=False，
    # 因为后面的 BN 有 beta 偏置项。
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    # 计算 BN 融合因子
    #
    # BN 推理公式：
    #   y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
    #
    # 其中：
    #   gamma = bn.weight
    #   beta = bn.bias
    #
    # factor = gamma / sqrt(running_var + eps)
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)

    # 融合卷积权重
    #
    # 原 Conv：
    #   z = conv_w * input + conv_b
    #
    # BN：
    #   y = factor * (z - running_mean) + bn.bias
    #
    # 展开：
    #   y = factor * conv_w * input + factor * (conv_b - running_mean) + bn.bias
    #
    # 所以融合后的卷积权重：
    #   fused_w = conv_w * factor
    #
    # factor shape 是 [out_channels]
    # reshape 成 [out_channels, 1, 1, 1]，方便广播到 conv_w
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))

    # 融合卷积 bias
    #
    # fused_b = (conv_b - running_mean) * factor + bn.bias
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)

    # 返回融合后的 conv
    return conv


def fuse_module(m):
    # 递归遍历一个模块 m，把其中相邻的 Conv2d + BN 融合
    #
    # 输入：
    #   m：任意 nn.Module，例如整个模型、backbone、某个 stage、某个 block
    #
    # 输出：
    #   融合后的 m

    # 用于记录最近一次遇到的 Conv2d 模块
    last_conv = None

    # 用于记录最近一次遇到的 Conv2d 模块名称
    # 后面需要通过 m._modules[last_conv_name] 替换它
    last_conv_name = None


    # 遍历当前模块 m 的直接子模块
    # named_children() 只遍历一层，不会递归到孙子模块
    for name, child in m.named_children():

        # 如果当前 child 是 BatchNorm2d 或 SyncBatchNorm
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):

            # 如果前面没有紧邻的 Conv2d
            if last_conv is None:  # only fuse BN that is after Conv
                # 说明这个 BN 不能和前面的 Conv 融合
                # 例如 BN 前面可能不是 Conv，或者已经递归进入了其他结构
                continue

            # 将最近的 Conv2d 和当前 BN 融合
            fused_conv = fuse_conv_bn(last_conv, child)

            # 用融合后的 Conv 替换原来的 Conv
            m._modules[last_conv_name] = fused_conv

            # 为了尽量少改变网络结构，不删除 BN
            # 而是把 BN 替换成 Identity
            # Identity 前向时直接返回输入，相当于什么都不做
            # To reduce changes, set BN as Identity instead of deleting it.
            m._modules[name] = nn.Identity()

            # 清空 last_conv
            # 避免一个 Conv 被错误地和后面的其他 BN 再次融合
            last_conv = None

        # 如果当前 child 是 Conv2d
        elif isinstance(child, nn.Conv2d):

            # 记录这个 Conv2d
            # 如果下一个 child 是 BN，就可以进行融合
            last_conv = child

            # 记录 Conv2d 在当前模块中的名字
            last_conv_name = name

        # 如果当前 child 既不是 BN，也不是 Conv
        else:
            # 递归进入这个子模块内部继续查找 Conv + BN
            fuse_module(child)

    # 返回融合后的模块
    return m


def parse_args():
    # 定义命令行参数解析函数

    # 创建 argparse 参数解析器
    # description 是命令行帮助信息
    parser = argparse.ArgumentParser(
        description='fuse Conv and BN layers in a model')

    # 第一个位置参数：config 文件路径
    # 例如 projects/configs/sparsedrive_small_stage2.py
    parser.add_argument('config', help='config file path')

    # 第二个位置参数：checkpoint 文件路径
    # 例如 ckpt/sparsedrive_stage2.pth
    parser.add_argument('checkpoint', help='checkpoint file path')

    # 第三个位置参数：输出 checkpoint 路径
    # 例如 ckpt/sparsedrive_stage2_fuse.pth
    parser.add_argument('out', help='output path of the converted model')

    # 解析命令行参数
    args = parser.parse_args()

    # 返回参数对象
    return args


def main():
    # 主函数

    # 解析命令行参数
    args = parse_args()

    # build the model from a config file and a checkpoint file
    # 根据 config 构建模型，并加载 checkpoint 权重
    #
    # init_model 会做几件事：
    #   1. 读取 config
    #   2. 构建模型结构
    #   3. 加载 checkpoint
    #   4. 通常把模型设成 eval 模式
    model = init_model(args.config, args.checkpoint)

    # fuse conv and bn layers of the model
    # 递归融合模型中的 Conv2d + BN
    fused_model = fuse_module(model)

    # 保存融合后的模型 checkpoint
    save_checkpoint(fused_model, args.out)


# Python 脚本入口
# 只有直接运行这个文件时，才会执行 main()
# 如果被其他文件 import，则不会自动执行
if __name__ == '__main__':
    main()


