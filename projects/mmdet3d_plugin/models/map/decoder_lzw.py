# 文件整体作用：
#    (1) 本文件定义 SparseDrive map 分支的 decoder。
#    (2) 这里的 decoder 不是 Transformer decoder，而是“后处理解码器”。
#    (3) 它把网络输出的分类 logits 和点坐标预测，整理成评估/可视化需要的结果格式。


from typing import Optional, List               # 从 typing 导入类型标注工具
import torch                                    # 导入 PyTorch
from mmdet.core.bbox.builder import BBOX_CODERS # 导入 MMDetection 的 BBOX_CODERS 注册器。虽然 map 输出不是传统 bbox，但 MMDetection 仍常用 BBOX_CODERS 注册“解码器”。配置文件里可以通过 type='SparsePoint3DDecoder' 构建该类。


# SparsePoint3DDecoder 类：稀疏3D点线解码器，作用是将模型原始输出的 “分类 logits 预测和点坐标预测” 解码为结构化的线要素检测结果，供评估和可视化用【该类是普通 object，不继承 nn.Module，因为它没有可学习参数。它只负责推理阶段的结果解码和格式转换】
@BBOX_CODERS.register_module() # 将 SparsePoint3DDecoder 注册到 BBOX_CODERS。这样它能被 MMDetection / MMCV 的 config 系统自动构建。SparseDrive 的 map 分支在测试/推理时会调用它来整理输出。
class SparsePoint3DDecoder(object):
    # 1. 初始化函数
    def __init__(
        self,
        coords_dim: int = 2,                     # 单个采样点的坐标维度，默认2为BEV平面xy坐标
        score_threshold: Optional[float] = None, # 置信度阈值，低于该值的低分检测结果会被过滤，设为 None 表示不过滤低分预测
    ):
        super(SparsePoint3DDecoder, self).__init__() # 调用父类初始化（object类，此处为框架规范写法）
        self.score_threshold = score_threshold # 保存分数阈值：如果该值不为 None，就会过滤 scores < threshold 的预测；如果为 None，则保留 topk 选出的全部预测。
        self.coords_dim = coords_dim           # 保存点坐标维度：对 map polyline 来说，coords_dim 通常是 2【decode() 中会把展平坐标 reshape 成 [..., num_points, coords_dim]】

    # 2. 解码函数：对模型原始预测做后处理，输出最终检测结果
    def decode(
        self,
        cls_scores,       # 每层解码器的分类logits列表，每个元素形状 [B, num_pred, num_cls]
        pts_preds,        # 每层解码器的点坐标预测列表，每个元素形状 [B, num_pred, num_sample*coords_dim]
        instance_id=None, # 预留参数：实例ID，map 并不使用，只是为了对齐 detection 接口而预留
        quality=None,     # 预留参数：质量分数，map 并不使用，只是为了对齐 detection 接口而预留
        output_idx=-1,    # 选取的解码层索引：指定使用第几层解码输出进行解码，默认-1即最后一层输出 output_idx ，当前代码固定用 [-1]
    ):
        '''
            函数作用：将 SparseDrive map head 的原始输出解码成评估与可视化格式
            具体实现步骤：
                Step1. 对分类 logits 做 sigmoid，得到每个 query 对每个类别的置信度。
                Step2. 在 query-class 展平空间中选 top num_pred 个预测。
                Step3. 根据 topk 结果取出对应类别 id、分数和 polyline 点坐标。
        '''
        # 从最后一层分类预测中获取维度：batch大小、预测query数量、map类别数量
        bs, num_pred, num_cls = cls_scores[-1].shape

        # 1. 提取模型最后一层输出的 “分类logits” 和 “点坐标预测”，并整理
        # (1) 选取最后一层的分类logits，通过sigmoid映射为0~1的置信度概率
        cls_scores = cls_scores[-1].sigmoid()
        # (2) 选取最后一层的点坐标预测，重塑为 [bs, num_pred, num_sample, coords_dim]【目的是将展平的坐标向量恢复为「采样点×坐标维度」的结构化形状】
        pts_preds = pts_preds[-1].reshape(bs, num_pred, -1, self.coords_dim)

        # 2. 《flatten() + topk() + % num_cls》得到每个 query 的最大类别预测，并选出 top num_pred 个预测
        # (a) 将分类分数从第1维开始展平，得到形状 [bs, num_pred * num_cls] 即把每个query的所有类别分数展开为一维，再全局取TopK个置信度最高的候选
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(num_pred, dim=1) # topk的k值为num_pred，即保留与query数量相同的候选数量

        # (b) 计算每个候选对应的类别ID：展平索引对类别数取余，得到类别下标
        cls_ids = indices % num_cls
        '''
            flatten() + topk() + % num_cls 其实就是在求该 polyline 的最大类别 max()：
                1. flatten() 将 [num_pred, num_cls] 展平为 [num_pred * num_cls]，即把每个 query 的所有类别分数展开为一维。
                2. topk() 在展平后的分数中选出前 num_pred 个最高分的候选。
                3. % num_cls 将展平索引映射回原始的类别索引，得到每个候选对应的类别ID。
            这样就能得到每个候选的类别ID、分数和对应的点坐标预测，供后续整理输出。
        '''

        # 3. 如果设置了置信度阈值，则生成布尔掩码，标记分数达标的候选
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        # 4. 解码
        # (1) 初始化输出列表，每个元素对应一个batch样本的检测结果
        output = []

        # (2) 逐帧（逐batch样本）处理检测结果
        for i in range(bs):
            # (a) 获取当前帧所有候选的类别ID
            category_ids = cls_ids[i]

            # (b) 获取当前帧所有候选的置信度分数
            scores = cls_scores[i]

            # (c) 根据候选索引找到对应的点坐标：索引整除类别数得到原始query的下标
            pts = pts_preds[i, indices[i] // num_cls]

            # (d) 如果启用了分数阈值过滤，则过滤类别ID、过滤分数、过滤点坐标
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]] # 过滤类别ID
                scores = scores[mask[i]]             # 过滤分数
                pts = pts[mask[i]]                   # 过滤点坐标

            # (e) 将当前帧结果组装为字典，转为numpy格式（脱离计算图，便于后续处理）
            output.append(
                {
                    "vectors": [vec.detach().cpu().numpy() for vec in pts], # 每条线的点坐标，已逐个转为numpy数组、并依次存入列表，形状 [num_sample, coords_dim]
                    "scores": scores.detach().cpu().numpy(),                # 每条线的置信度分数，已转为numpy数组，形状 [num_det]
                    "labels": category_ids.detach().cpu().numpy(),          # 每条线的类别ID，已转为numpy数组，形状 [num_det]
                }
            )

        # (3) 返回所有batch的解码结果
        return output # list[dict]，长度等于batch_size，每个字典对应一帧的检测结果，字典包含 "vectors" "scores" "labels"
