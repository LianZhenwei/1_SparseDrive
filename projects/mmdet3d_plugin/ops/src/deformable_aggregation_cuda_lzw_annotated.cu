// ============================================================================
// 文件：deformable_aggregation_cuda.cu
// 位置：~/SparseDrive/projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu
// 作用：Deformable Aggregation 自定义 CUDA kernel 的真正实现。
//
// 这个文件负责：
//   1. 在 CUDA device 上实现双线性采样 bilinear_sampling；
//   2. 在 CUDA device 上实现双线性采样的反向 bilinear_sampling_grad；
//   3. 实现前向 CUDA kernel：deformable_aggregation_kernel；
//   4. 实现反向 CUDA kernel：deformable_aggregation_grad_kernel；
//   5. 提供 C++ 可调用的 launcher：deformable_aggregation / deformable_aggregation_grad。
//
// 小白理解：
//   SparseDrive 需要对每个 3D anchor/query 生成若干 3D 采样点，投影到 6 个相机、4 个 FPN level 上，
//   从图像特征图中取特征，再按权重融合。
//   这个操作有大量循环：batch × anchor × point × camera × scale × channel。
//   CUDA kernel 就是把这些循环拆成很多 GPU 线程并行算。
// ============================================================================

#include <ATen/ATen.h>
// 引入 ATen，PyTorch C++ 后端张量库。
// 本 .cu 文件主要用裸指针计算，ATen 头文件提供必要的 PyTorch/CUDA 集成类型。

#include <ATen/cuda/CUDAContext.h>
// 引入 PyTorch CUDA 上下文相关工具。
// 当前文件中没有显式使用很多 API，但 PyTorch CUDA 扩展常规会包含它。

#include <cuda.h>
// CUDA Driver API 基础头文件。

#include <cuda_runtime.h>
// CUDA Runtime API 头文件。
// 提供 kernel launch、线程/block 索引等运行时支持。

#include <THC/THCAtomics.cuh>
// PyTorch THC 里的 atomic 操作头文件。
// 这里使用 atomicAdd 做并发累加。
// 注意：多个线程可能同时给同一个 output 或 gradient 位置加值，所以必须 atomicAdd。

#include <iostream>
// C++ 标准输入输出头文件。
// 当前文件没有实际使用 cout，属于常见冗余 include。

#include <stdlib.h>
// C 标准库头文件。
// 当前文件主要可能为了通用函数/宏保留，实际使用很少。


/**
 * @brief CUDA device 函数：对单个通道上的一个连续坐标做双线性采样。
 *
 * 这个函数运行在 GPU device 上，只能被 kernel 或其他 __device__ 函数调用。
 *
 * 输入背景：
 *   mc_ms_feat 在 feature_maps_format 后形状为 [B, num_feat, C]，内存布局相当于：
 *      batch -> flattened spatial position -> channel
 *   对某个 batch、camera、scale、channel，base_ptr 已经指向当前 level 的第一个像素位置和当前 channel。
 *   后续通过 h/w offset 找到四邻域像素的该 channel 值。
 *
 * 双线性采样公式：
 *   给定浮点坐标 (h_im, w_im)，取四个邻居：
 *      左上 v1: (h_low,  w_low)
 *      右上 v2: (h_low,  w_high)
 *      左下 v3: (h_high, w_low)
 *      右下 v4: (h_high, w_high)
 *   根据距离计算权重：
 *      w1=(1-lh)(1-lw), w2=(1-lh)lw, w3=lh(1-lw), w4=lh*lw
 *   返回：w1*v1 + w2*v2 + w3*v3 + w4*v4。
 *
 * @param bottom_data 输入特征表指针，即 mc_ms_feat。
 * @param height 当前 FPN level 的高 H。
 * @param width 当前 FPN level 的宽 W。
 * @param num_embeds 通道数 C。
 * @param h_im 浮点采样纵坐标，单位是当前 feature map 像素坐标。
 * @param w_im 浮点采样横坐标，单位是当前 feature map 像素坐标。
 * @param base_ptr 当前 batch、camera、scale、channel 在 bottom_data 中的基础偏移。
 * @return float 双线性插值得到的单通道特征值。
 */
__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  const int h_low = floorf(h_im);
  // h_im 的下边界整数坐标。
  // 例如 h_im=10.3，则 h_low=10。

  const int w_low = floorf(w_im);
  // w_im 的左边界整数坐标。

  const int h_high = h_low + 1;
  // h_im 的上/下一个邻居坐标。
  // 双线性插值需要 h_low 和 h_high 两行。

  const int w_high = w_low + 1;
  // w_im 的右邻居坐标。

  const float lh = h_im - h_low;
  // h 方向距离下边界的比例。
  // 例如 h_im=10.3，则 lh=0.3。

  const float lw = w_im - w_low;
  // w 方向距离左边界的比例。

  const float hh = 1 - lh, hw = 1 - lw;
  // hh 是距离上边界的互补权重，hw 是距离右边界的互补权重。
  // 命名上 hh/hw 有点抽象，可以理解成：
  //   hh = 1 - lh；
  //   hw = 1 - lw。

  const int w_stride = num_embeds;
  // 同一行中，相邻 w 像素之间的内存跨度是 C。
  // 因为布局是 [..., H, W, C] 展平后的 [position, channel]。

  const int h_stride = width * w_stride;
  // 相邻 h 行之间的内存跨度是 W*C。

  const int h_low_ptr_offset = h_low * h_stride;
  // h_low 行相对于当前 level 起点的偏移。

  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  // h_high 行相对于当前 level 起点的偏移。

  const int w_low_ptr_offset = w_low * w_stride;
  // w_low 列相对于当前行起点的偏移。

  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  // w_high 列相对于当前行起点的偏移。

  float v1 = 0;
  // 左上邻居值，默认 0。
  // 如果采样点越界，就保持 0，相当于 zero padding。

  if (h_low >= 0 && w_low >= 0) {
    // 判断左上点是否在合法范围的下/左边界内。
    // 这里没有显式判断 h_low <= height-1 / w_low <= width-1，
    // 是因为外层 loc_w/loc_h 被限制在 (0,1)，h_low/w_low 通常不会超过最大边界。

    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    // 计算左上点在 bottom_data 中的绝对偏移。

    v1 = bottom_data[ptr1];
    // 读取左上点特征值。
  }

  float v2 = 0;
  // 右上邻居值。

  if (h_low >= 0 && w_high <= width - 1) {
    // 判断右上点是否合法。
    // 需要 w_high 不超过 width-1。

    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    // 计算右上点偏移。

    v2 = bottom_data[ptr2];
    // 读取右上点特征值。
  }

  float v3 = 0;
  // 左下邻居值。

  if (h_high <= height - 1 && w_low >= 0) {
    // 判断左下点是否合法。
    // 需要 h_high 不超过 height-1。

    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    // 计算左下点偏移。

    v3 = bottom_data[ptr3];
    // 读取左下点特征值。
  }

  float v4 = 0;
  // 右下邻居值。

  if (h_high <= height - 1 && w_high <= width - 1) {
    // 判断右下点是否合法。

    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    // 计算右下点偏移。

    v4 = bottom_data[ptr4];
    // 读取右下点特征值。
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  // 计算四个邻居的双线性插值权重。
  // v1 左上权重：离左上越近越大。
  // v2 右上权重：横向靠右越大。
  // v3 左下权重：纵向靠下越大。
  // v4 右下权重：横向靠右且纵向靠下越大。

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  // 加权求和，得到采样值。

  return val;
  // 返回单个 channel 的双线性采样结果。
}


/**
 * @brief CUDA device 函数：双线性采样的反向传播。
 *
 * 它计算三类梯度：
 *   1. loss 对输入特征 bottom_data/mc_ms_feat 的梯度；
 *   2. loss 对 sampling_location 的梯度；
 *   3. loss 对 aggregation weight 的梯度。
 *
 * 为什么要 atomicAdd：
 *   一个 feature 像素可能被很多 anchor/point/camera/scale/channel 线程同时采样，
 *   多个线程会同时给同一个 grad_mc_ms_feat 位置累加梯度。
 *   所以必须用 atomicAdd 保证并发加法安全。
 *
 * @param bottom_data 前向输入特征表。
 * @param weight 当前采样位置对应的聚合权重。
 * @param height 当前 level 高。
 * @param width 当前 level 宽。
 * @param num_embeds 通道数。
 * @param h_im feature map 像素坐标系下的 h 坐标。
 * @param w_im feature map 像素坐标系下的 w 坐标。
 * @param base_ptr 当前 batch/camera/scale/channel 的基础偏移。
 * @param grad_output loss 对当前 output 通道的梯度。
 * @param grad_mc_ms_feat 待累加的 feature 梯度。
 * @param grad_sampling_location 待累加的采样坐标梯度，长度 2，对应 [w, h]。
 * @param grad_weights 待累加的 weight 梯度。
 */
__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  const int h_low = floorf(h_im);
  // h 方向下边界。

  const int w_low = floorf(w_im);
  // w 方向左边界。

  const int h_high = h_low + 1;
  // h 方向上边界。

  const int w_high = w_low + 1;
  // w 方向右边界。

  const float lh = h_im - h_low;
  // h 方向小数部分。

  const float lw = w_im - w_low;
  // w 方向小数部分。

  const float hh = 1 - lh, hw = 1 - lw;
  // 插值互补权重。

  const int w_stride = num_embeds;
  // 相邻列跨度 C。

  const int h_stride = width * w_stride;
  // 相邻行跨度 W*C。

  const int h_low_ptr_offset = h_low * h_stride;
  // h_low 行偏移。

  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  // h_high 行偏移。

  const int w_low_ptr_offset = w_low * w_stride;
  // w_low 列偏移。

  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  // w_high 列偏移。

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  // 四邻域插值权重。

  const float top_grad_mc_ms_feat = grad_output * weight;
  // output = weight * sampled_value。
  // 所以 loss 对 sampled_value 的梯度 = grad_output * weight。

  float grad_h_weight = 0, grad_w_weight = 0;
  // 这两个变量累计 sampled_value 对 h_im/w_im 的偏导。
  // 后面会乘 top_grad_mc_ms_feat，得到 loss 对 h_im/w_im 的梯度。

  float v1 = 0;
  // 左上邻居值。

  if (h_low >= 0 && w_low >= 0) {
    // 如果左上邻居合法。

    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    // 左上邻居偏移。

    v1 = bottom_data[ptr1];
    // 读取左上邻居值。

    grad_h_weight -= hw * v1;
    // 对 h_im 的偏导贡献。
    // w1=(1-lh)*(1-lw)，对 lh 的导数是 -(1-lw)。

    grad_w_weight -= hh * v1;
    // 对 w_im 的偏导贡献。
    // w1=(1-lh)*(1-lw)，对 lw 的导数是 -(1-lh)。

    atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
    // loss 对 v1 的梯度 = 插值权重 w1 * loss 对 sampled_value 的梯度。
    // 多线程可能写同一 ptr1，所以 atomicAdd。
  }

  float v2 = 0;
  // 右上邻居值。

  if (h_low >= 0 && w_high <= width - 1) {
    // 如果右上邻居合法。

    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    // 右上邻居偏移。

    v2 = bottom_data[ptr2];
    // 读取右上邻居值。

    grad_h_weight -= lw * v2;
    // w2=(1-lh)*lw，对 lh 的导数是 -lw。

    grad_w_weight += hh * v2;
    // w2=(1-lh)*lw，对 lw 的导数是 +(1-lh)。

    atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
    // 累加 v2 的梯度。
  }

  float v3 = 0;
  // 左下邻居值。

  if (h_high <= height - 1 && w_low >= 0) {
    // 如果左下邻居合法。

    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    // 左下邻居偏移。

    v3 = bottom_data[ptr3];
    // 读取左下邻居值。

    grad_h_weight += hw * v3;
    // w3=lh*(1-lw)，对 lh 的导数是 +(1-lw)。

    grad_w_weight -= lh * v3;
    // w3=lh*(1-lw)，对 lw 的导数是 -lh。

    atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
    // 累加 v3 的梯度。
  }

  float v4 = 0;
  // 右下邻居值。

  if (h_high <= height - 1 && w_high <= width - 1) {
    // 如果右下邻居合法。

    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    // 右下邻居偏移。

    v4 = bottom_data[ptr4];
    // 读取右下邻居值。

    grad_h_weight += lw * v4;
    // w4=lh*lw，对 lh 的导数是 +lw。

    grad_w_weight += lh * v4;
    // w4=lh*lw，对 lw 的导数是 +lh。

    atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
    // 累加 v4 的梯度。
  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  // 重新计算前向采样值 sampled_value。
  // 因为 weight 的梯度需要 sampled_value：
  // output = weight * sampled_value，所以 dL/dweight = dL/doutput * sampled_value。

  atomicAdd(grad_weights, grad_output * val);
  // 累加 weight 梯度。
  // 多个 channel 可能共享同一个 group weight，所以需要 atomicAdd。

  atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
  // 累加采样位置 w 方向的梯度。
  // 注意 sample_location 存的是归一化坐标 loc_w，前向中 w_im = loc_w * width - 0.5。
  // 所以 d w_im / d loc_w = width，因此要乘 width。

  atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
  // 累加采样位置 h 方向的梯度。
  // h_im = loc_h * height - 0.5，所以要乘 height。
}


/**
 * @brief Deformable Aggregation 前向 CUDA kernel。
 *
 * 并行粒度：
 *   一个 CUDA 线程负责一个组合：
 *      batch × anchor × point × camera × scale × channel
 *   也就是每个线程算“某个 query 的某个采样点，在某个相机某个尺度某个通道上的贡献”。
 *
 * 前向数学：
 *   output[b, anchor, c] +=
 *       weight[b, anchor, pt, cam, scale, group(c)]
 *       * bilinear_sample(feature[b, cam, scale, :, :, c], location[b, anchor, pt, cam])
 *
 * @param num_kernels 总线程任务数。
 * @param output 输出特征，[B, num_anchor, C]。
 * @param mc_ms_feat 展平特征表，[B, num_feat, C]。
 * @param spatial_shape [num_cams, num_scale, 2]。
 * @param scale_start_index [num_cams, num_scale]。
 * @param sample_location [B, num_anchor, num_pts, num_cams, 2]。
 * @param weights [B, num_anchor, num_pts, num_cams, num_scale, num_groups]。
 * 后续 int 参数为各维度大小。
 */
__global__ void deformable_aggregation_kernel(
    const int num_kernels,
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算当前线程的一维全局索引。
    // blockIdx.x：当前 block 编号。
    // blockDim.x：每个 block 的线程数。
    // threadIdx.x：当前 block 内线程编号。

    if (idx >= num_kernels) return;
    // 如果线程索引超过任务总数，直接返回。
    // 因为 grid/block 数量通常会向上取整，可能多启动一些空线程。

    const float weight = *(weights + idx / (num_embeds / num_groups));
    // 读取当前线程对应的聚合权重。
    // num_embeds / num_groups 是每个 group 包含的通道数。
    // 多个 channel 共享一个 group 权重，所以 channel 维要除以 group_dims。

    const int channel_index = idx % num_embeds;
    // 从一维 idx 中解析出 channel 维。

    idx /= num_embeds;
    // 去掉 channel 维，继续解析更高维。

    const int scale_index = idx % num_scale;
    // 解析 FPN scale/level 维。

    idx /= num_scale;
    // 去掉 scale 维。

    const int cam_index = idx % num_cams;
    // 解析 camera 维。

    idx /= num_cams;
    // 去掉 camera 维。

    const int pts_index = idx % num_pts;
    // 解析 point 维，即当前 anchor 的第几个采样点。

    idx /= num_pts;
    // 去掉 point 维。

    int anchor_index = idx % num_anchors;
    // 解析 anchor/query 维。

    idx /= num_anchors;
    // 去掉 anchor 维。

    const int batch_index = idx % batch_size;
    // 解析 batch 维。

    idx /= batch_size;
    // 这里继续除已经没实际用途，只是保留解析模板。

    anchor_index = batch_index * num_anchors + anchor_index;
    // 把 [batch, anchor] 合并成一维 anchor_index。
    // 后续访问 output 和 sample_location 时更方便。

    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
    // 计算 sample_location 的偏移。
    // sample_location 逻辑形状：[B, anchor, pts, cam, 2]。
    // anchor_index 已经合并了 B 和 anchor。
    // 最后一维长度为 2，所以用 << 1 等价于乘以 2。
    // loc_offset 指向 [loc_w, loc_h] 中的 loc_w。

    const float loc_w = sample_location[loc_offset];
    // 读取归一化横坐标 loc_w，范围通常应该在 [0,1]。

    if (loc_w <= 0 || loc_w >= 1) return;
    // 如果横坐标在图像外或边界上，直接不采样。
    // 这里用严格 (0,1)，边界点被忽略。

    const float loc_h = sample_location[loc_offset + 1];
    // 读取归一化纵坐标 loc_h。

    if (loc_h <= 0 || loc_h >= 1) return;
    // 如果纵坐标在图像外或边界上，直接返回。

    int cam_scale_index = cam_index * num_scale + scale_index;
    // 把 camera 和 scale 合并，用于索引 spatial_shape 和 scale_start_index。
    // spatial_shape 逻辑形状 [num_cams, num_scale, 2]。

    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;
    // 计算当前 batch、camera、scale、channel 在 mc_ms_feat 中的基础偏移。
    // batch_index * num_feat：跳到当前 batch 的展平特征表。
    // scale_start_index[cam_scale_index]：跳到当前 camera/scale 的起始空间位置。
    // * num_embeds + channel_index：跳到当前 channel。

    cam_scale_index = cam_scale_index << 1;
    // spatial_shape 最后一维是 2，即 [H, W]。
    // 所以索引乘以 2，指向当前 camera/scale 的 H。

    const int h = spatial_shape[cam_scale_index];
    // 当前 feature level 的高度 H。

    const int w = spatial_shape[cam_scale_index + 1];
    // 当前 feature level 的宽度 W。

    const float h_im = loc_h * h - 0.5;
    // 把归一化坐标 loc_h 转成 feature map 像素坐标。
    // -0.5 是 align_corners=False 风格的常见坐标对齐方式。

    const float w_im = loc_w * w - 0.5;
    // 把归一化坐标 loc_w 转成 feature map 像素坐标。

    atomicAdd(
        output + anchor_index * num_embeds + channel_index,
        bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight
    );
    // 对当前 output[b, anchor, channel] 累加贡献。
    // 贡献 = 当前 camera/scale/point/channel 上双线性采样值 × 对应权重。
    // 多个 point/camera/scale 线程会写同一个 output 位置，所以必须 atomicAdd。
}


/**
 * @brief Deformable Aggregation 反向 CUDA kernel。
 *
 * 并行粒度与 forward kernel 一样：
 *   一个线程对应 batch × anchor × point × camera × scale × channel 的一个组合。
 *
 * 反向需要计算：
 *   1. grad_mc_ms_feat：采样到的四邻域像素特征的梯度；
 *   2. grad_sampling_location：采样位置坐标梯度；
 *   3. grad_weights：聚合权重梯度。
 */
__global__ void deformable_aggregation_grad_kernel(
    const int num_kernels,
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 当前线程的一维全局任务索引。

    if (idx >= num_kernels) return;
    // 空线程直接返回。

    const int weights_ptr = idx / (num_embeds / num_groups);
    // 计算当前线程对应的 weights 指针位置。
    // 因为一个 group 权重对应多个 channel，所以 channel 维需要除以 group_dims。

    const int channel_index = idx % num_embeds;
    // 解析 channel 维。

    idx /= num_embeds;
    // 去掉 channel 维。

    const int scale_index = idx % num_scale;
    // 解析 scale 维。

    idx /= num_scale;
    // 去掉 scale 维。

    const int cam_index = idx % num_cams;
    // 解析 camera 维。

    idx /= num_cams;
    // 去掉 camera 维。

    const int pts_index = idx % num_pts;
    // 解析 point 维。

    idx /= num_pts;
    // 去掉 point 维。

    int anchor_index = idx % num_anchors;
    // 解析 anchor 维。

    idx /= num_anchors;
    // 去掉 anchor 维。

    const int batch_index = idx % batch_size;
    // 解析 batch 维。

    idx /= batch_size;
    // 后续不再使用。

    anchor_index = batch_index * num_anchors + anchor_index;
    // 合并 batch 和 anchor，方便索引 output/sample_location。

    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;
    // 计算 sample_location 中当前 [b, anchor, point, camera] 的坐标偏移。

    const float loc_w = sample_location[loc_offset];
    // 读取归一化横坐标。

    if (loc_w <= 0 || loc_w >= 1) return;
    // 越界则没有前向贡献，因此也没有梯度。

    const float loc_h = sample_location[loc_offset + 1];
    // 读取归一化纵坐标。

    if (loc_h <= 0 || loc_h >= 1) return;
    // 越界则返回。

    const float grad = grad_output[anchor_index*num_embeds + channel_index];
    // 读取 loss 对 output[b, anchor, channel] 的梯度。

    int cam_scale_index = cam_index * num_scale + scale_index;
    // 合并 camera 和 scale 索引。

    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;
    // 当前 batch/camera/scale/channel 在 mc_ms_feat 中的基础偏移。

    cam_scale_index = cam_scale_index << 1;
    // 转成 spatial_shape 的扁平索引位置。

    const int h = spatial_shape[cam_scale_index];
    // 当前 level 高度。

    const int w = spatial_shape[cam_scale_index + 1];
    // 当前 level 宽度。

    const float h_im = loc_h * h - 0.5;
    // 归一化 h 坐标转 feature map 像素坐标。

    const float w_im = loc_w * w - 0.5;
    // 归一化 w 坐标转 feature map 像素坐标。

    /* atomicAdd( */
    /*     output + anchor_index * num_embeds + channel_index, */
    /*     bilinear_sampling(mc_ms_feat, h, w, num_embeds, h_im, w_im, value_offset) * weight */
    /* ); */
    // 这是原作者保留的 forward 逻辑注释。
    // backward 正是在对这条 forward 累加公式求导。

    const float weight = weights[weights_ptr];
    // 读取当前 group 对应的聚合权重。

    float *grad_weights_ptr = grad_weights + weights_ptr;
    // 当前 weight 的梯度地址。

    float *grad_location_ptr = grad_sampling_location + loc_offset;
    // 当前 sampling_location 的梯度地址，包含 [grad_w, grad_h] 两个 float。

    bilinear_sampling_grad(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset,
        grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
    // 调用双线性采样反向函数，累加 feature/location/weight 的梯度。
}


/**
 * @brief 前向 CUDA launcher，被 C++ 文件 deformable_aggregation.cpp 调用。
 *
 * 它本身在 CPU 侧运行，负责计算 kernel 数量，并启动真正的 GPU kernel。
 *
 * @param output 输出指针。
 * @param mc_ms_feat 输入展平特征指针。
 * @param spatial_shape 空间形状指针。
 * @param scale_start_index 起始 index 指针。
 * @param sample_location 采样位置指针。
 * @param weights 聚合权重指针。
 * 后续 int 参数为各维度大小。
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
) {
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    // 计算总线程任务数。
    // 每个线程对应一个：B × point × channel × anchor × camera × scale 的组合。
    // 维度乘积很大，适合 GPU 并行。

    deformable_aggregation_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128>>>(
        num_kernels, output,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    // 启动 CUDA kernel。
    // <<<grid, block>>> 是 CUDA kernel launch 语法：
    //   blockDim.x = 128：每个 block 128 个线程；
    //   gridDim.x = ceil(num_kernels / 128)：block 数向上取整。
    // kernel 中每个线程用 idx 判断是否越界。
}


/**
 * @brief 反向 CUDA launcher，被 C++ 文件 deformable_aggregation.cpp 调用。
 *
 * 负责启动 deformable_aggregation_grad_kernel，计算三个梯度：
 *   grad_mc_ms_feat、grad_sampling_location、grad_weights。
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
) {
    const int num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;
    // 反向 kernel 的线程任务数与前向一致。
    // 每个线程处理一个前向贡献项的梯度。

    deformable_aggregation_grad_kernel
        <<<(int)ceil(((double)num_kernels/128)), 128>>>(
        num_kernels,
        mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );
    // 启动反向 CUDA kernel。
    // 同样使用每 block 128 线程。
}
