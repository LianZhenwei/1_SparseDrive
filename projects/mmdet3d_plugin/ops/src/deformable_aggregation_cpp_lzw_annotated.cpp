// ============================================================================
// 文件：deformable_aggregation.cpp
// 位置：~/SparseDrive/projects/mmdet3d_plugin/ops/src/deformable_aggregation.cpp
// 作用：PyTorch C++ Extension 的绑定层 / 包装层。
//
// 这个文件不写真正的 CUDA kernel 细节，真正的 GPU 并行计算在
// deformable_aggregation_cuda.cu 中实现。
//
// 本文件负责：
//   1. 接收 Python/PyTorch 传进来的 at::Tensor；
//   2. 从 Tensor 中读取 shape 和 data_ptr；
//   3. 创建 output / 接收 grad Tensor；
//   4. 调用 .cu 文件中实现的 CUDA launcher 函数；
//   5. 用 PYBIND11_MODULE 把 C++ 函数暴露给 Python。
//
// Python 调用链大致是：
//   deformable_aggregation.py
//       -> deformable_aggregation_ext.deformable_aggregation_forward(...)
//       -> 本文件 deformable_aggregation_forward(...)
//       -> .cu 文件 deformable_aggregation(...)
//       -> .cu 文件 deformable_aggregation_kernel<<<...>>>(...)
// ============================================================================

#include <torch/extension.h>
// 引入 PyTorch C++ Extension 头文件。
// 作用：
//   1. 提供 at::Tensor 类型；
//   2. 提供 pybind11 与 PyTorch 扩展集成；
//   3. 让 C++ 函数可以被编译成 Python 可 import 的扩展模块。

#include <c10/cuda/CUDAGuard.h>
// 引入 CUDA 设备保护相关工具。
// 多 GPU 训练时，Tensor 可能位于 cuda:0、cuda:1 等不同设备。
// CUDAGuard 可以确保后续 CUDA kernel 在输入 Tensor 所在的正确 GPU 上执行。


/**
 * @brief CUDA 前向 launcher 函数声明。
 *
 * 这个函数的真正定义在 deformable_aggregation_cuda.cu 中。
 * 本文件只声明它，方便 deformable_aggregation_forward() 调用。
 *
 * @param output 输出特征指针，形状逻辑为 [batch_size, num_anchors, num_embeds]。
 * @param mc_ms_feat multi-camera multi-scale feature 展平特征表，形状为 [B, num_feat, C]。
 * @param spatial_shape 每个 camera、每个 scale 的二维尺寸 [H, W]，形状为 [num_cams, num_scale, 2]。
 * @param scale_start_index 每个 camera、每个 scale 在 mc_ms_feat 中的起始 index，形状为 [num_cams, num_scale]。
 * @param sample_location 采样位置，形状为 [B, num_anchor, num_pts, num_cams, 2]，最后一维通常是归一化 [w, h]。
 * @param weights 聚合权重，形状为 [B, num_anchor, num_pts, num_cams, num_scale, num_groups]。
 * @param batch_size batch size。
 * @param num_cams 相机数量，nuScenes/SparseDrive 通常是 6。
 * @param num_feat 展平后的总特征位置数量，例如 89760。
 * @param num_embeds embedding/channel 维度，例如 256。
 * @param num_scale FPN level 数量，例如 4。
 * @param num_anchors query/anchor 数量，例如 det head 是 900。
 * @param num_pts 每个 anchor 的采样点数量。
 * @param num_groups 通道分组数，例如 8。
 */
void deformable_aggregation(
  float* output,
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
);


// feat: bs, num_feat, c
// 说明 _mc_ms_feat 的逻辑形状：[batch_size, num_feat, num_embeds]。
// bs 是 batch size；num_feat 是所有相机所有尺度展平后的空间位置数；c 是通道数。

// _spatial_shape: cam, scale, 2
// 说明 _spatial_shape 的逻辑形状：[num_cams, num_scale, 2]。
// 最后一维 2 表示 [H, W]。

// _scale_start_index: cam, scale
// 说明 _scale_start_index 的逻辑形状：[num_cams, num_scale]。
// 每个位置记录对应 camera/scale 在展平特征表里的起始 offset。

// _sampling_location: bs, anchor, pts, cam, 2
// 说明 _sampling_location 的逻辑形状：[B, num_anchor, num_pts, num_cams, 2]。
// 最后一维 2 通常是归一化图像坐标 [w, h]。

// _weights: bs, anchor, pts, cam, scale, group
// 说明 _weights 的逻辑形状：[B, num_anchor, num_pts, num_cams, num_scale, num_groups]。
// 权重用于融合不同相机、不同尺度、不同采样点、不同通道 group 的特征。

// output: bs, anchor, c
// 说明输出形状：[B, num_anchor, num_embeds]。

// kernel: bs, anchor, pts, c
// 原作者备注：kernel 维度会按 batch、anchor、point、channel 等展开。
// 实际 CUDA kernel 中还会展开 camera 和 scale。


/**
 * @brief Deformable Aggregation 前向传播的 C++ 包装函数。
 *
 * 这是 Python 扩展模块暴露出来的前向函数。
 * Python 侧调用 deformable_aggregation_ext.deformable_aggregation_forward(...)
 * 时，会进入这里。
 *
 * 本函数做的事情：
 *   1. 使用 CUDAGuard 设置正确 GPU；
 *   2. 从输入 Tensor 中读取各维度大小；
 *   3. 取出底层裸指针 data_ptr；
 *   4. 创建输出 Tensor；
 *   5. 调用 .cu 文件中的 deformable_aggregation(...) launcher；
 *   6. 返回 output Tensor。
 *
 * @param _mc_ms_feat 展平后的多相机多尺度特征 Tensor，[B, num_feat, C]。
 * @param _spatial_shape 每个 camera/scale 的 [H, W]，[num_cams, num_scale, 2]。
 * @param _scale_start_index 每个 camera/scale 的起始 index，[num_cams, num_scale]。
 * @param _sampling_location 采样坐标，[B, num_anchor, num_pts, num_cams, 2]。
 * @param _weights 聚合权重，[B, num_anchor, num_pts, num_cams, num_scale, num_groups]。
 * @return at::Tensor 聚合后的特征，[B, num_anchor, C]。
 */
at::Tensor deformable_aggregation_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  // 设置当前设备为 _mc_ms_feat 所在设备。
  // 例如 _mc_ms_feat 在 cuda:1，那么后面创建 output 和 launch kernel 都应在 cuda:1 上。

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  // 进一步用 OptionalCUDAGuard 保护 CUDA 设备上下文。
  // 这在多卡训练/多线程场景更稳妥，避免 kernel 发到错误 GPU。

  int batch_size = _mc_ms_feat.size(0);
  // 从 _mc_ms_feat 第 0 维读取 batch size。
  // _mc_ms_feat shape: [B, num_feat, C]。

  int num_feat = _mc_ms_feat.size(1);
  // 从 _mc_ms_feat 第 1 维读取展平后的总特征位置数。
  // 例如 6 个相机 × 4 个 FPN level 展平后可能是 89760。

  int num_embeds = _mc_ms_feat.size(2);
  // 从 _mc_ms_feat 第 2 维读取 embedding/channel 数。
  // SparseDrive 中通常是 256。

  int num_cams = _spatial_shape.size(0);
  // 从 spatial_shape 第 0 维读取相机数量。
  // nuScenes/SparseDrive 通常是 6。

  int num_scale = _spatial_shape.size(1);
  // 从 spatial_shape 第 1 维读取 FPN level 数量。
  // SparseDrive 通常是 4。

  int num_anchors = _sampling_location.size(1);
  // 从 sampling_location 第 1 维读取 anchor/query 数量。
  // detection head 可能是 900，map head 可能是 100。

  int num_pts = _sampling_location.size(2);
  // 从 sampling_location 第 2 维读取每个 anchor 的采样点数量。

  int num_groups = _weights.size(5);
  // 从 weights 第 5 维读取通道分组数量。
  // 通常 num_embeds=256, num_groups=8，则每组 32 通道。

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  // 获取 _mc_ms_feat 的 float 指针，传给 CUDA kernel。
  // 注意 Python 包装层已经把 Tensor 转成 float32 contiguous。

  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  // 获取 spatial_shape 的 int 指针。
  // Python 包装层已经把它转成 int32 contiguous。

  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  // 获取 scale_start_index 的 int 指针。

  const float* sampling_location = _sampling_location.data_ptr<float>();
  // 获取 sampling_location 的 float 指针。

  const float* weights = _weights.data_ptr<float>();
  // 获取 weights 的 float 指针。

  auto output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());
  // 创建输出 Tensor，形状 [B, num_anchor, C]。
  // 使用 _mc_ms_feat.options() 保持 dtype/device/layout 一致。
  // 这里初始化为 0，因为 CUDA kernel 中会用 atomicAdd 累加多个 point/camera/scale 的贡献。

  deformable_aggregation(
    output.data_ptr<float>(),
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
  // 调用 .cu 文件中的 CUDA launcher。
  // launcher 内部会启动 deformable_aggregation_kernel<<<blocks, threads>>>(...)。

  return output;
  // 返回聚合后的特征 Tensor 给 Python。
}


/**
 * @brief CUDA 反向传播 launcher 函数声明。
 *
 * 真正定义在 deformable_aggregation_cuda.cu 中。
 * 它会启动 CUDA backward kernel，计算：
 *   1. grad_mc_ms_feat；
 *   2. grad_sampling_location；
 *   3. grad_weights。
 *
 * @param mc_ms_feat 前向输入特征。
 * @param spatial_shape 空间形状元信息。
 * @param scale_start_index 起始索引元信息。
 * @param sample_location 前向采样位置。
 * @param weights 前向聚合权重。
 * @param grad_output loss 对前向 output 的梯度。
 * @param grad_mc_ms_feat 输出参数：loss 对 mc_ms_feat 的梯度。
 * @param grad_sampling_location 输出参数：loss 对 sampling_location 的梯度。
 * @param grad_weights 输出参数：loss 对 weights 的梯度。
 * 后续 int 参数含义同 forward launcher。
 */
void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
);


/**
 * @brief Deformable Aggregation 反向传播的 C++ 包装函数。
 *
 * Python 侧的 DeformableAggregationFunction.backward(...) 会先创建三个全 0 梯度 Tensor：
 *   _grad_mc_ms_feat、_grad_sampling_location、_grad_weights。
 * 然后把它们传到这里。
 * 本函数把 Tensor 转成裸指针，再调用 CUDA backward launcher，把梯度写进去。
 *
 * 注意：本函数返回 void，因为梯度 Tensor 是作为可写引用传入，并在 CUDA kernel 中原地填充。
 *
 * @param _mc_ms_feat 前向输入特征，[B, num_feat, C]。
 * @param _spatial_shape 空间形状，[num_cams, num_scale, 2]。
 * @param _scale_start_index 起始索引，[num_cams, num_scale]。
 * @param _sampling_location 采样坐标，[B, num_anchor, num_pts, num_cams, 2]。
 * @param _weights 聚合权重，[B, num_anchor, num_pts, num_cams, num_scale, num_groups]。
 * @param _grad_output loss 对 output 的梯度，[B, num_anchor, C]。
 * @param _grad_mc_ms_feat 待写入的 mc_ms_feat 梯度 Tensor。
 * @param _grad_sampling_location 待写入的 sampling_location 梯度 Tensor。
 * @param _grad_weights 待写入的 weights 梯度 Tensor。
 */
void deformable_aggregation_backward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_grad_output,
  at::Tensor &_grad_mc_ms_feat,
  at::Tensor &_grad_sampling_location,
  at::Tensor &_grad_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  // 设置当前设备为 _mc_ms_feat 所在 GPU。

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  // 进一步保护 CUDA 设备上下文，避免多 GPU 时设备错乱。

  int batch_size = _mc_ms_feat.size(0);
  // batch size。

  int num_feat = _mc_ms_feat.size(1);
  // 展平后的总 feature 位置数。

  int num_embeds = _mc_ms_feat.size(2);
  // 通道/embedding 数。

  int num_cams = _spatial_shape.size(0);
  // 相机数量。

  int num_scale = _spatial_shape.size(1);
  // FPN level 数量。

  int num_anchors = _sampling_location.size(1);
  // anchor/query 数量。

  int num_pts = _sampling_location.size(2);
  // 每个 anchor 的采样点数量。

  int num_groups = _weights.size(5);
  // 通道 group 数量。

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  // 获取前向 feature 指针。

  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  // 获取空间形状指针。

  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  // 获取 level 起始 index 指针。

  const float* sampling_location = _sampling_location.data_ptr<float>();
  // 获取采样坐标指针。

  const float* weights = _weights.data_ptr<float>();
  // 获取权重指针。

  const float* grad_output = _grad_output.data_ptr<float>();
  // 获取从后续网络/损失传回来的 output 梯度指针。

  float* grad_mc_ms_feat = _grad_mc_ms_feat.data_ptr<float>();
  // 获取待写入的 feature 梯度指针。

  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  // 获取待写入的 sampling_location 梯度指针。

  float* grad_weights = _grad_weights.data_ptr<float>();
  // 获取待写入的 weights 梯度指针。

  deformable_aggregation_grad(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
  // 调用 .cu 文件中的 backward launcher。
  // CUDA kernel 会用 atomicAdd 把多线程计算得到的梯度累加到三个梯度 Tensor 中。
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 定义 Python 扩展模块入口。
  // TORCH_EXTENSION_NAME 是 PyTorch 编译扩展时自动传入的模块名宏。
  // setup.py 中 name="deformable_aggregation_ext"，所以 Python 里 import 的模块就是它。

  m.def(
    "deformable_aggregation_forward",
    &deformable_aggregation_forward,
    "deformable_aggregation_forward"
  );
  // 把 C++ 函数 deformable_aggregation_forward 绑定到 Python 名字
  // deformable_aggregation_forward。
  // Python 侧可以这样调用：
  //   deformable_aggregation_ext.deformable_aggregation_forward(...)

  m.def(
    "deformable_aggregation_backward",
    &deformable_aggregation_backward,
    "deformable_aggregation_backward"
  );
  // 把 C++ 函数 deformable_aggregation_backward 绑定到 Python 名字
  // deformable_aggregation_backward。
  // Python 侧可以这样调用：
  //   deformable_aggregation_ext.deformable_aggregation_backward(...)
}
