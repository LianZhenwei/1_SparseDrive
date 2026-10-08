# ---------------------------------------------
# OpenMMLab 版权声明
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

# 表示该文件被 Zhiqi Li 修改过
#  Modified by Zhiqi Li
# ---------------------------------------------


from .mmdet_train import custom_train_detector # 从同目录下的 mmdet_train.py 中导入 custom_train_detector，这个函数是 SparseDrive 自定义训练流程的核心
# from mmseg.apis import train_segmentor       # 原本可能支持 mmseg 的 train_segmentor，但这里被注释掉了
from mmdet.apis import train_detector          # 导入 MMDetection 原生 train_detector，train_model() 会直接调用它


# 一、【/home/lzw/SparseDrive/tools/train_lzw.py 调用的是该函数】该函数是一个训练入口包装器，它的目的：根据模型类型，选择不同训练流程
def custom_train_model(
    model,             # model：模型对象
    dataset,           # dataset：训练数据集
    cfg,               # cfg：配置文件对象
    distributed=False, # distributed：是否分布式训练
    validate=False,    # validate：是否训练过程中验证
    timestamp=None,    # timestamp：时间戳
    meta=None,         # meta：环境信息、seed、config 等元信息
):
    # 如果模型类型是 EncoderDecoder3D【这可能原本是为 3D segmentation 或其他任务预留的】，当前代码直接 assert False【表示这个分支暂不支持】
    if cfg.model.type in ["EncoderDecoder3D"]:
        assert False # 当前代码直接 assert False，表示这个分支暂不支持

    # 否则，默认走 detection 类型训练流程
    else:
        # 调用 SparseDrive 自定义 detection 训练函数 custom_train_detector()【见 /home/lzw/SparseDrive/projects/mmdet3d_plugin/apis/mmdet_train.py】
        custom_train_detector(
            model,                   # 模型
            dataset,                 # 数据集
            cfg,                     # 配置文件
            distributed=distributed, # 是否分布式
            validate=validate,       # 是否验证
            timestamp=timestamp,     # 时间戳
            meta=meta,               # 元信息
        )


# 二、【/home/lzw/SparseDrive/tools/train_lzw.py 通常不会调用该函数】该函数是普通 MMDetection 训练入口包装器：它不走 SparseDrive 自定义 custom_train_detector，而是直接调用 mmdet.apis.train_detector
def train_model(
    model,             # model：模型对象
    dataset,           # dataset：训练数据集
    cfg,               # cfg：配置文件
    distributed=False, # distributed：是否分布式
    validate=False,    # validate：是否验证
    timestamp=None,    # timestamp：时间戳
    meta=None,         # meta：元信息
):
    train_detector(
        model,                   # 模型
        dataset,                 # 数据集
        cfg,                     # 配置文件
        distributed=distributed, # 是否分布式
        validate=validate,       # 是否验证
        timestamp=timestamp,     # 时间戳
        meta=meta,               # 元信息
    )
