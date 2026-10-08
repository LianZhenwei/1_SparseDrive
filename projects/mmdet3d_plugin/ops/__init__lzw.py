import torch # 导入 PyTorch。本文件主要用 torch 做张量 reshape、cat、split、unflatten、permute 等格式转换操作。

from .deformable_aggregation import DeformableAggregationFunction
# 从同目录的 deformable_aggregation.py 中导入自定义 autograd Function。
# 这个 Function 内部会调用编译好的 C++/CUDA 扩展 deformable_aggregation_ext。


# 一、对外暴露的 deformable aggregation() 调用函数【在 blocks.py 的 DeformableFeatureAggregation类中被调用】
def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    """
    对外暴露的 deformable aggregation 调用函数。

    作用：
        把 DeformableAggregationFunction.apply(...) 包装成一个普通函数，
        让其他模块调用时更简洁。

    背景：
        PyTorch 自定义 autograd Function 不能像普通 nn.Module 那样直接实例化调用，
        必须通过 ClassName.apply(...) 进入 autograd 机制。
        所以这里提供一个薄封装：
            deformable_aggregation_function(...)
        内部实际调用：
            DeformableAggregationFunction.apply(...)

    参数：
        feature_maps:
            已展平的多相机多尺度特征，通常就是 feature_maps_format() 返回的 col_feats。
            形状大致为 [B, total_camera_level_pixels, C]。
        spatial_shape:
            每个相机、每个 FPN level 的空间尺寸。
            形状通常是 [num_cams, num_levels, 2]。
        scale_start_index:
            每个相机、每个 FPN level 在展平特征中的起始位置。
            形状通常是 [num_cams, num_levels]。
        sampling_location:
            采样位置，由 3D key points 投影到图像/FPN 特征空间得到。
        weights:
            对不同 camera、level、point、group 的聚合权重。

    返回：
        聚合后的 query/anchor 图像特征。
        具体形状由 CUDA 扩展输出约定，SparseDrive 调用处通常会 reshape 成 [B, num_anchor, embed_dims]。
    """

    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )
    # 调用 PyTorch 自定义 Function。
    # 这样 forward 会进入 DeformableAggregationFunction.forward，
    # backward 会进入 DeformableAggregationFunction.backward。


# 二、多相机、多尺度 feature maps 的格式转换函数【在 sparsedrive.py 的 extract_feat() 函数中提取图像特征时使用到】
def feature_maps_format(feature_maps, inverse=False):
    """
    多相机、多尺度 feature maps 的格式转换函数。

    作用：
        在 SparseDrive 中，FPN 输出通常是 list 格式：
            [
                [B, N, C, H0, W0],
                [B, N, C, H1, W1],
                [B, N, C, H2, W2],
                [B, N, C, H3, W3],
            ]
        其中：
            B = batch size；
            N = num_cams，nuScenes 中通常是 6；
            C = embed_dims，通常是 256；
            H/W = 不同 FPN level 的空间尺寸。

        但是自定义 CUDA deformable aggregation 算子更适合使用一个连续展平的大特征表：
            col_feats: [B, N * sum_l(H_l * W_l), C]

        因此 inverse=False 时，本函数负责把普通 list feature maps 转成：
            [col_feats, spatial_shape, scale_start_index]

        inverse=True 时，本函数做反向转换：
            [col_feats, spatial_shape, scale_start_index]
            -> 原始嵌套多相机多尺度 feature map 格式。

    参数：
        feature_maps:
            inverse=False:
                可以是普通多尺度 feature list；
                也可以是嵌套 list/tuple，用于处理相机形状分组。
            inverse=True:
                应该是 [col_feats, spatial_shape, scale_start_index]。
        inverse:
            False 表示普通 feature maps -> CUDA 展平格式；
            True 表示 CUDA 展平格式 -> 普通 feature maps。

    返回：
        inverse=False:
            [col_feats, spatial_shape, scale_start_index]
        inverse=True:
            还原后的 multi-camera multi-scale feature map list。

    常见 SparseDrive 形状例子：
        输入：
            [
                [B, 6, 256, 64, 176],
                [B, 6, 256, 32, 88],
                [B, 6, 256, 16, 44],
                [B, 6, 256, 8, 22],
            ]
        输出：
            col_feats: [B, 89760, 256]
                因为 64*176 + 32*88 + 16*44 + 8*22 = 14960，
                6 个相机就是 89760。
            spatial_shape: [6, 4, 2]
            scale_start_index: [6, 4]
    """

    # 1. 如果 inverse=True，说明要把展平格式还原成原始多相机、多尺度 feature maps。
    if inverse:
        # (1) 解包特征图列表feature_maps，得到
        col_feats, spatial_shape, scale_start_index = feature_maps
        '''
            解包输入：
                col_feats：展平后的特征表，形状大致 [B, total_positions, C]。
                spatial_shape：每个 camera/level 的 [H, W]，形状 [num_cams, num_levels, 2]。
                scale_start_index：每个 camera/level 的起始 index，形状 [num_cams, num_levels]。
            注意：当前 inverse 代码主要依赖 spatial_shape 来 split/unflatten，scale_start_index 解包了但没有显式使用。        
        '''

        # (2) 获取相机数和 FPN level 数
        num_cams, num_levels = spatial_shape.shape[:2] # 对 SparseDrive 来说 num_cams=6, num_levels=4。

        # (3) 计算每个相机、每个 level 展平后的像素/网格数量 H*W。
        split_size = spatial_shape[..., 0] * spatial_shape[..., 1] # spatial_shape 的 shape 是: [num_cams, num_levels]，每个元素是对应 level 的 H*W。例如每个相机是 [11264, 2816, 704, 176]。
        split_size = split_size.cpu().numpy().tolist()             # 把 split_size 从 tensor 转成 Python list。torch.split 更容易接收 Python list 作为每段长度。

        # (4) 初始化 idx。
        idx = 0 # 注意：这个变量后续没有被使用，属于冗余变量。不影响程序运行。可调试用

        # (5) 根据 spatial_shape 来分组相机。相机分组的原则是：如果相邻两个相机的 FPN level 空间尺寸完全相同，就把它们放在同一组；如果不同，就开启新组。
        cam_split = [1]                       # cam_split 用于记录连续相机分组中每组包含多少个相机。初始认为第一个分组包含 1 个相机。
        cam_split_size = [sum(split_size[0])] # cam_split_size 用于记录每个相机分组在 col_feats dim=1 上需要切分的长度。sum(split_size[0]) 是第 0 个相机所有 level 的总位置数。

        for i in range(num_cams - 1):
            # 遍历相邻相机：比较第 i 个相机和第 i+1 个相机的 spatial_shape 是否相同。
            # 如果相机之间 feature map 尺寸相同，就可以放在同一组；
            # 如果不同，就开启新分组。

            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                # 如果相邻两个相机的多尺度空间形状不完全相同，
                # 说明它们不能用同一套 split_size 还原，需要新建一组。

                cam_split.append(0)
                # 新增一个相机分组，初始相机数为 0。
                # 后面 cam_split[-1] += 1 会把 i+1 号相机加入该组。

                cam_split_size.append(0)
                # 新增一个分组长度计数器，初始为 0。

            cam_split[-1] += 1
            # 把第 i+1 个相机计入当前分组。
            # 如果 shape 没变，就是加入已有组；
            # 如果 shape 变了，就是加入刚创建的新组。

            cam_split_size[-1] += sum(split_size[i + 1])
            # 把第 i+1 个相机所有 level 的总 H*W 加到当前分组长度中。

        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]
        # 先按相机分组总长度切分 col_feats。
        # col_feats.split(cam_split_size, dim=1)：
        #   沿着展平位置维 dim=1 切成若干相机组。
        #
        # x.unflatten(1, (cam_split[i], -1))：
        #   把该组的展平维度拆回 [组内相机数, 该组每个相机的总位置数]。
        #
        # 得到 mc_feat：
        #   一个 list，每个元素大致形状是 [B, group_num_cams, sum_level_pixels, C]。

        spatial_shape = spatial_shape.cpu().numpy().tolist()
        # 把 spatial_shape 转成 Python list，方便后续作为 unflatten 的 shape 参数。

        mc_ms_feat = []
        # 用于保存最终还原出来的 multi-camera multi-scale feature maps。

        shape_index = 0
        # 当前相机组对应的 spatial_shape 起始索引。
        # 如果前面一组有多个相机，下一个组要跳过这些相机。

        for i, feat in enumerate(mc_feat):
            # 遍历每一个相机分组。
            # feat 形状大致是 [B, group_num_cams, sum_level_pixels, C]。

            feat = list(feat.split(split_size[shape_index], dim=2))
            # 按当前相机组的每个 level 的 H*W 长度切分。
            # dim=2 是每个相机内部的展平空间维。
            # 切完后 feat 是 list：
            #   每个元素对应一个 FPN level，形状大致 [B, group_num_cams, H_l*W_l, C]。

            for j, f in enumerate(feat):
                # 遍历当前相机组的每个 FPN level。

                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                # 把展平空间维 H_l*W_l 还原成 [H_l, W_l]。
                # f 原来大致 [B, group_num_cams, H_l*W_l, C]。
                # unflatten 后大致 [B, group_num_cams, H_l, W_l, C]。

                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
                # 调整维度顺序：
                # [B, group_num_cams, H_l, W_l, C]
                # -> [B, group_num_cams, C, H_l, W_l]。
                # 这就是 CNN/FPN 常见的 feature map 格式。

            mc_ms_feat.append(feat)
            # 把当前相机组还原出的多尺度特征加入结果。
            # feat 是 list，包含该相机组的多个 FPN level。

            shape_index += cam_split[i]
            # shape_index 前进当前组包含的相机数量，
            # 让下一组使用正确的 spatial_shape。

        return mc_ms_feat
        # 返回还原后的 feature maps。

    if isinstance(feature_maps[0], (list, tuple)):
        # 如果 feature_maps[0] 本身还是 list/tuple，
        # 说明输入是嵌套结构，例如按相机组组织的 feature maps。
        # 这时递归处理每个子组，再把结果拼起来。

        formated = [feature_maps_format(x) for x in feature_maps]
        # 对每个子组递归调用 feature_maps_format。
        # 每个 x 会被转换成 [col_feats, spatial_shape, scale_start_index]。

        col_feats = torch.cat([x[0] for x in formated], dim=1)
        # 把所有子组的 col_feats 沿展平位置维 dim=1 拼接起来。
        # 得到总的 col_feats。

        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        # 把所有子组的 spatial_shape 沿相机维 dim=0 拼接起来。
        # 得到完整 [num_cams, num_levels, 2]。

        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        # 把所有子组的 scale_start_index 沿相机维 dim=0 拼接。
        # 注意：这里简单 cat，假设子组内部 index 已经按自身 col_feats 组织。

        return [col_feats, spatial_shape, scale_start_index]
        # 返回合并后的 CUDA 友好格式。

    bs, num_cams = feature_maps[0].shape[:2]
    # 普通情况：feature_maps 是 FPN 多尺度 list。
    # 每个元素形状一般是 [B, num_cams, C, H, W]。
    # 这里从第一个 level 读取 B 和 num_cams。

    spatial_shape = []
    # 用 list 保存每个 FPN level 的空间尺寸 [H, W]。

    col_feats = []
    # 用 list 保存每个 level 展平后的特征。

    for i, feat in enumerate(feature_maps):
        # 遍历每个 FPN level。
        # feat 形状一般是 [B, num_cams, C, H_l, W_l]。

        spatial_shape.append(feat.shape[-2:])
        # 保存当前 level 的空间尺寸 [H_l, W_l]。

        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )
        # 把当前 level 的 H_l 和 W_l 展平成 H_l*W_l。
        # 原 shape:
        #   [B, num_cams, C, H_l, W_l]
        # 新 shape:
        #   [B, num_cams, C, H_l*W_l]
        #
        # feat.shape[2] 是通道数 C，SparseDrive FPN 后通常是 256。

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    # 第一步 torch.cat(col_feats, dim=-1)：
    #   把所有 level 沿空间展平维拼接。
    #   list 中每个元素是 [B, num_cams, C, H_l*W_l]。
    #   拼接后是 [B, num_cams, C, sum_l(H_l*W_l)]。
    #
    # 第二步 permute(0, 1, 3, 2)：
    #   把通道维放到最后。
    #   [B, num_cams, C, total_pixels]
    #   -> [B, num_cams, total_pixels, C]。
    #
    # 第三步 flatten(1, 2)：
    #   把相机维和空间维合并。
    #   [B, num_cams, total_pixels, C]
    #   -> [B, num_cams*total_pixels, C]。
    #
    # 对 256x704 输入、4 层 FPN：
    #   total_pixels = 64*176 + 32*88 + 16*44 + 8*22 = 14960。
    #   num_cams=6 时，num_cams*total_pixels=89760。
    #   所以 col_feats 常见形状是 [B, 89760, 256]。

    spatial_shape = [spatial_shape] * num_cams
    # 把同一套 FPN level 空间尺寸复制 num_cams 份。
    # 因为通常 6 个相机图像经过同一个 backbone/FPN，特征图尺寸相同。
    # 此时 spatial_shape 从：
    #   [[64,176], [32,88], [16,44], [8,22]]
    # 变成：
    #   [
    #     [[64,176], [32,88], [16,44], [8,22]],
    #     ... 重复 num_cams 次
    #   ]

    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    # 把 spatial_shape 转成 tensor，并放到和 col_feats 相同的设备上。
    # 形状为 [num_cams, num_levels, 2]。
    # dtype=int64，但后续传给 CUDA Function 时会再转 int32。

    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    # 计算每个相机、每个 level 的 H*W。
    # 形状 [num_cams, num_levels]。
    # 例如每行是 [11264, 2816, 704, 176]。

    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    # 先 flatten 成 [num_cams*num_levels]，
    # 再做累计和 cumsum。
    # 这一步得到的是每段结束位置的累计偏移。
    # 例如 [11264, 14080, 14784, 14960, ...]。

    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    # 把“结束位置偏移”转换成“起始位置偏移”。
    # 第一个 level 的起始位置是 0；
    # 后面每个 level 的起始位置是前一段的累计结束位置。

    scale_start_index = scale_start_index.reshape(num_cams, -1)
    # 重新 reshape 成 [num_cams, num_levels]。
    # 对 nuScenes 通常是 [6, 4]。
    # 每一行代表一个相机的 4 个 FPN level 起始 index。

    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    # 打包成 deformable_aggregation_function 需要的格式：
    # 1. col_feats：连续展平特征表；
    # 2. spatial_shape：每个 camera/level 的空间尺寸；
    # 3. scale_start_index：每个 camera/level 的展平起始位置。

    return feature_maps
    # 返回 [col_feats, spatial_shape, scale_start_index]。

