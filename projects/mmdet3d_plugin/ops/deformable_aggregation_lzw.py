import torch
# 导入 PyTorch。
# 这里主要用于：
# 1. 创建梯度张量 torch.zeros_like；
# 2. 保证自定义 CUDA 算子与 PyTorch autograd 计算图衔接；
# 3. 在 backward 中处理 grad_output 等张量。

from torch.autograd.function import Function, once_differentiable
# Function：PyTorch 自定义 autograd 函数的基类。
# 如果某个操作不是纯 PyTorch 算子，而是 C++/CUDA 扩展，就可以继承 Function，
# 手动定义 forward() 和 backward()，让 PyTorch 知道如何前向和反向传播。
#
# once_differentiable：装饰 backward() 的工具。
# 表示这个 backward 只支持一阶梯度，不支持对 backward 再求梯度，即不支持二阶梯度。
# 自定义 CUDA 算子常见这种写法，因为通常只实现训练需要的一阶反传。

from . import deformable_aggregation_ext
# 从当前 ops 包中导入编译好的 C++/CUDA 扩展模块。
# deformable_aggregation_ext 不是普通 .py 文件，而是 setup.py 编译生成的扩展模块。
# 里面暴露了两个关键函数：
# 1. deformable_aggregation_forward：CUDA 前向聚合；
# 2. deformable_aggregation_backward：CUDA 反向传播。


class DeformableAggregationFunction(Function):
    """
    PyTorch 自定义自动求导函数：Deformable Aggregation。

    作用：
        把 SparseDrive 中“多相机、多尺度 feature map 上的可变形采样聚合”
        封装成一个 PyTorch autograd 可识别的算子。

    为什么需要这个类：
        SparseDrive/Sparse4D 的 sparse query 会在 3D 空间生成 key points，
        再投影到多个相机、多个 FPN level 上采样图像特征。
        这个操作涉及大量不规则索引、双线性采样、加权求和。
        如果全用 Python/PyTorch 写，速度可能较慢；
        因此项目把核心计算写成 C++/CUDA 扩展，
        再用这个 Function 类接入 PyTorch 的 forward/backward。

    forward 输入：
        ctx:
            PyTorch 自动求导上下文，用于保存 backward 需要的张量。
        mc_ms_feat:
            multi-camera multi-scale feature 的展平格式。
            通常来自 feature_maps_format() 的 col_feats。
            形状大致为 [B, total_spatial_positions, C]。
        spatial_shape:
            每个相机、每个 FPN level 的空间尺寸。
            形状通常是 [num_cams, num_levels, 2]，例如 [6, 4, 2]。
            最后一维是 [H, W]。
        scale_start_index:
            每个相机、每个 FPN level 在 mc_ms_feat 展平序列中的起始下标。
            形状通常是 [num_cams, num_levels]，例如 [6, 4]。
        sampling_location:
            采样点投影到图像/FPN 特征上的位置。
            在 SparseDrive 中通常来自 3D key points 投影。
        weights:
            对不同相机、level、point、group 的聚合权重。

    forward 输出：
        output:
            聚合后的 query/anchor 图像特征。
            原代码注释写作 [bs, num_pts, num_embeds]；
            结合 SparseDrive 调用处，更直观地理解为：
            [B, num_anchor, embed_dims] 或可 reshape 成这个形状。

    backward 返回：
        对 forward 每个输入的梯度：
            grad_mc_ms_feat：图像特征的梯度，需要反传到 FPN/ResNet；
            None：spatial_shape 是元信息，不需要梯度；
            None：scale_start_index 是元信息，不需要梯度；
            grad_sampling_location：采样位置的梯度，若采样点由网络预测则可学习；
            grad_weights：聚合权重的梯度，需要反传到 weights_fc 等模块。

    注意：
        这个 Python 文件不实现真正的采样聚合数学细节；
        真正计算在 deformable_aggregation_ext 的 C++/CUDA 代码中。
    """

    @staticmethod
    # PyTorch 自定义 Function 的 forward 必须写成 staticmethod。
    # 调用时不是 DeformableAggregationFunction().forward(...)，
    # 而是 DeformableAggregationFunction.apply(...)。

    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        """
        前向传播：调用 CUDA 扩展完成 deformable aggregation。

        主要步骤：
            1. 把输入张量转成 CUDA 扩展更容易处理的 contiguous + 指定 dtype；
            2. 调用 deformable_aggregation_ext.deformable_aggregation_forward；
            3. 保存 backward 需要的输入张量；
            4. 返回聚合后的 output。

        参数说明：
            ctx:
                autograd 上下文，用于 ctx.save_for_backward(...)。
            mc_ms_feat:
                展平后的多相机多尺度特征。
            spatial_shape:
                每个相机、每个 level 的 [H, W]。
            scale_start_index:
                每个相机、每个 level 的展平起始位置。
            sampling_location:
                采样点坐标。
            weights:
                聚合权重。

        返回：
            output:
                CUDA 前向计算得到的聚合特征。
        """

        # output: [bs, num_pts, num_embeds]
        # 源码原注释。
        # 更结合 SparseDrive 使用场景可理解为：输出最终会被 reshape 成 [B, num_anchor, embed_dims]。

        mc_ms_feat = mc_ms_feat.contiguous().float()
        # 确保 mc_ms_feat 内存连续，并转换为 float32。
        # contiguous() 很重要：
        #   CUDA kernel 通常按连续内存指针访问数据；
        #   如果张量是 permute/transpose 后的非连续布局，直接传给底层 kernel 可能出错或性能差。
        # float() 很重要：
        #   该 CUDA 扩展通常按 float32 计算；
        #   即使外部用了 fp16，这里也会转成 fp32 以保证数值和 kernel 类型匹配。

        spatial_shape = spatial_shape.contiguous().int()
        # 确保 spatial_shape 连续，并转为 int32。
        # spatial_shape 是形状/索引类元信息，不参与梯度；
        # CUDA 中通常用 int 读取 H、W。

        scale_start_index = scale_start_index.contiguous().int()
        # 确保 scale_start_index 连续，并转为 int32。
        # 它告诉 CUDA：某个相机某个 FPN level 的特征在展平特征表中从哪里开始。

        sampling_location = sampling_location.contiguous().float()
        # 确保 sampling_location 连续，并转为 float32。
        # 采样位置是连续坐标，因此使用浮点数。

        weights = weights.contiguous().float()
        # 确保 weights 连续，并转为 float32。
        # weights 是加权融合系数，需要参与乘法和反向传播。

        output = deformable_aggregation_ext.deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        # 调用 C++/CUDA 扩展的前向函数。
        # 这一步完成真正的“在多相机、多尺度特征图上按 sampling_location 采样，
        # 再按 weights 加权聚合”的高性能计算。

        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        # 保存 backward 需要使用的张量。
        # backward 中需要原始 feature、shape、start index、sampling location、weights
        # 来计算对 feature/location/weights 的梯度。

        return output
        # 返回 CUDA 算子前向输出。
        # PyTorch 会把这个 output 接入计算图；
        # 后续 loss.backward() 时会自动调用下面的 backward()。

    @staticmethod
    # backward 同样必须是 staticmethod。

    @once_differentiable
    # 声明该 backward 不支持二阶求导。
    # 也就是可以 loss.backward()，
    # 但不支持对这个 backward 过程再求 gradient-of-gradient。

    def backward(ctx, grad_output):
        """
        反向传播：调用 CUDA 扩展计算输入梯度。

        参数：
            ctx:
                forward() 中保存过张量的上下文。
            grad_output:
                loss 对 forward 输出 output 的梯度。
                它由 PyTorch autograd 从后续网络层自动传回来。

        返回：
            一个 tuple，长度必须与 forward 的输入数量一致：
                grad_mc_ms_feat:
                    loss 对 mc_ms_feat 的梯度。
                None:
                    spatial_shape 不需要梯度。
                None:
                    scale_start_index 不需要梯度。
                grad_sampling_location:
                    loss 对 sampling_location 的梯度。
                grad_weights:
                    loss 对 weights 的梯度。
        """

        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        # 从 forward 的 ctx.save_for_backward(...) 中取回保存的张量。
        # backward 需要这些张量来复现前向采样关系，并计算梯度。

        mc_ms_feat = mc_ms_feat.contiguous().float()
        # 再次确保 mc_ms_feat 是连续 float32。
        # 虽然 forward 中已经处理过，但这里再处理一次更稳妥。

        spatial_shape = spatial_shape.contiguous().int()
        # 确保 spatial_shape 是连续 int32。
        # shape 元信息不会求导，只是传给 CUDA kernel 用于定位。

        scale_start_index = scale_start_index.contiguous().int()
        # 确保 scale_start_index 是连续 int32。
        # 用于 CUDA backward 中找到各 level 的特征起始位置。

        sampling_location = sampling_location.contiguous().float()
        # 确保采样坐标是连续 float32。
        # 反向中要计算 loss 对采样位置的梯度。

        weights = weights.contiguous().float()
        # 确保权重是连续 float32。
        # 反向中要计算 loss 对聚合权重的梯度。

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        # 创建与 mc_ms_feat 形状相同的全 0 张量。
        # CUDA backward 会把 loss 对特征表的梯度写到这里。
        # 这个梯度会继续反传给 FPN 和 ResNet。

        grad_sampling_location = torch.zeros_like(sampling_location)
        # 创建与 sampling_location 形状相同的全 0 张量。
        # CUDA backward 会把 loss 对采样点位置的梯度写到这里。
        # 如果采样点位置由网络预测，这个梯度就能训练相关预测模块。

        grad_weights = torch.zeros_like(weights)
        # 创建与 weights 形状相同的全 0 张量。
        # CUDA backward 会把 loss 对聚合权重的梯度写到这里。
        # 这个梯度会继续反传给预测 weights 的网络层。

        deformable_aggregation_ext.deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        # 调用 C++/CUDA 扩展的反向函数。
        # 输入：
        #   前向保存的张量 + grad_output；
        # 输出：
        #   不是通过 return 返回，而是直接写入 grad_mc_ms_feat / grad_sampling_location / grad_weights。
        #
        # 注意 grad_output.contiguous()：
        #   确保从后续层传回来的梯度内存连续，便于 CUDA kernel 读取。

        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )
        # backward 的返回值必须和 forward 的输入一一对应：
        # forward(ctx, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
        # 因此返回：
        # 1. mc_ms_feat 的梯度；
        # 2. spatial_shape 的梯度：None，因为它只是 H/W 元信息；
        # 3. scale_start_index 的梯度：None，因为它只是索引元信息；
        # 4. sampling_location 的梯度；
        # 5. weights 的梯度。

