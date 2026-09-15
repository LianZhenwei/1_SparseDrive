# =============================================================================
# decoder_lzw.py —— SparseDrive 3D 检测“最终解码 / 后处理”模块
# =============================================================================
#
# 【本文件在整条检测链中的位置】
#
#   Sparse4DHead.forward()
#       └─ 每个 refine stage 输出：
#            classification : [B, N, K] 的分类 logits
#            prediction     : [B, N, D] 的内部 3D box state
#            quality        : [B, N, 2] 的 centerness / yawness logits（可选）
#       └─ Sparse4DHead.post_process()
#            └─ SparseBox3DDecoder.decode()  【本文件核心】
#                 (1) sigmoid 分类 logits；
#                 (2) top-k 选高分候选；
#                 (3) 可选用 centerness 重标定最终分数；
#                 (4) threshold 过滤；
#                 (5) decode_box() 将内部 box state 还原为评测 / 可视化格式；
#                 (6) 输出 boxes_3d / scores_3d / labels_3d / instance_ids。
#
# 【Detection 标准 shape】
#   B : batch size
#   N : normal detection query 数，官方 SparseDrive-S 通常 N=900
#   K : nuScenes 类别数，通常 K=10
#   D : 内部 box state 维度，通常 D=11
#       [x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz]
#   Q : quality 维度，Detection 通常 Q=2，即 [centerness, yawness]
#
# 【一个重要区别】
#   - 不带 tracking ID：top-k 在 “query × class” 上进行，一个 query 理论上可因多个类别候选进入 top-k。
#   - 带 tracking ID：先对每个 query 只保留最大类别，之后 top-k 在 “query” 上进行；
#     这样每个 query / instance_id 只会有一个输出类别，便于 tracking 评估。
# =============================================================================

from typing import Optional                      # 从 typing 中导入 Optional
import torch                                     # 导入 PyTorch。用于 tensor 操作，例如 sigmoid、topk、gather、sort、cat、atan2 等
from mmdet.core.bbox.builder import BBOX_CODERS  # 导入 BBOX_CODERS 注册表：SparseBox3DDecoder 会注册到 BBOX_CODERS 中，这样配置文件中 decoder=dict(type="SparseBox3DDecoder") 时可以自动构建
from projects.mmdet3d_plugin.core.box3d import * # 导入 box3d 中定义的常量索引：如 X、Y、Z、W、L、H、SIN_YAW、COS_YAW、VX、CNS 等，这些常量用于从 box tensor 的最后一维中取出对应字段


# 工具函数 decode_box() 用于把网络输出的 box 编码格式转换成真实 3D box 格式：内部 box state → 最终可读 3D box
def decode_box(box):
    '''
        decode_box 用于把网络输出的 box 编码格式转换成真实 3D box 格式
        
        1. 输入 box 是 refinement module 输出的 raw box prediction：[x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy, vz]
                其中：
                    x, y, z 通常直接预测
                    w, l, h 通常预测的是 log 尺寸，需要 exp 还原
                    yaw 不是直接预测角度，而是预测 sin_yaw 和 cos_yaw
                    velocity 等其他维度直接保留
                内部这样存储有两个训练上的好处：
                    (1) log(w/l/h) 保证解码后的尺寸 exp(.) 永远为正；
                    (2) sin/cos 避开 yaw 在 -pi / pi 交界处的角度不连续问题。
        2. 输出 box 是最终可读的 3D box：[x, y, z, w, l, h, yaw, vx, vy, vz]
    '''

    # (1) 将预测的 [sin(yaw), cos(yaw)] 恢复连续角度 yaw
    yaw = torch.atan2(box[..., SIN_YAW], box[..., COS_YAW]) # atan2(sin, cos) 输出范围为 [-pi, pi] 的弧度 rad

    # (2) 沿最后一维拼接最终可读的 box 字段
    box = torch.cat(
        [
            box[..., [X, Y, Z]],       # 取 x, y, z 中心坐标
            box[..., [W, L, H]].exp(), # 取 w, l, h 尺寸。由于网络预测的是 log-space 尺寸，所以用 exp() 还原成正数尺寸
            yaw[..., None],            # yaw 角增加一个最后维度。yaw 原 shape 是 [...]，yaw[..., None] 后变成 [..., 1]，便于与其他字段拼接
            box[..., VX:],             # 从 VX 开始的后续其他尾部状态量直接保留。从 VX 开始的后续其他尾部状态量通常包括 vx, vy, vz 等速度相关字段，这些状态量不需要额外解码，直接保留
        ],
        dim=-1, # 在 state 的最后一维拼接
    )

    # (3) 返回解码后的 box，格式：[x, y, z, w, l, h, yaw, vx, vy, vz]
    return box # 返回评测、保存、可视化所用的最终 3D box


# =============================================================================
# SparseBox3DDecoder：SparseDrive Detection 的最终结果解码器
# =============================================================================
# 【类作用】
#   它不再执行 attention、deformable aggregation 或 box refinement；
#   它只负责把最后一个（或指定）decoder stage 的“原始预测张量”变成最终检测结果。
#
# 【核心工作流】
#   (1) 选择 output_idx 指定的 refine stage；默认 -1，即最后一层。
#   (2) 分类 logits → sigmoid 概率。
#   (3) top-k 选候选：
#       (3.1) 非 tracking：在 N×K 个 query-class 对上选 top-k。
#       (3.2) tracking：每个 query 先取最大类别，再在 N 个 query 上选 top-k。
#   (4) 可选乘 centerness：final_score = cls_score × sigmoid(centerness)。
#   (5) 可选 score threshold 过滤。
#   (6) internal box → decoded 3D box。
#   (7) 输出每个 batch 样本的 dict。
#
# 【它不做什么】
#   本类没有显式 3D NMS、IoU 去重、Hungarian matching 或 ReID merge；
#   它主要依赖 query-based 检测器本身已经学习到的集合预测与 top-k 筛选。
# =============================================================================
@BBOX_CODERS.register_module() # 将 SparseBox3DDecoder 注册到 BBOX_CODERS 注册表中，配置文件里写 decoder=dict(type="SparseBox3DDecoder") 时，会实例化这个类
class SparseBox3DDecoder(object):
    # SparseBox3DDecoder 是 SparseDrive 3D 检测结果解码器
    # 它负责：
    # 1. 取最后一层或指定层 decoder 输出
    # 2. 对分类 logits 做 sigmoid 得到置信度
    # 3. 选取得分最高的 top-k 预测
    # 4. 可选用 quality/centerness 重新调整分数
    # 5. 把 raw box prediction 解码成真实 3D box
    # 6. 输出 boxes_3d、scores_3d、labels_3d、instance_ids 等字段

    # 1. 保存后处理超参数
    def __init__(
        self,                                    # self 表示当前 decoder 对象
        num_output: int = 300,                   # 每个 batch 样本最多输出多少个预测框结果：默认 300，也就是每个样本最多保留 top 300 个检测框
        score_threshold: Optional[float] = None, # 分数阈值：None 表示不额外过滤，若不是 None，则只保留 score >= score_threshold 的预测
        sorted: bool = True,                     # topk 是否按分数排序的开关：True 表示输出按照置信度从高到低排列
    ):
        super(SparseBox3DDecoder, self).__init__() # 调用父类 object 初始化
        self.num_output = num_output               # 保存最大输出数量
        self.score_threshold = score_threshold     # 保存分数阈值
        self.sorted = sorted                       # 保存 topk 是否排序的排序开关


    # 2. decode()：分类 / quality / box / ID → 最终检测结果 list[dict]
    def decode(
        self,             # self 表示当前 decoder 对象
        cls_scores,       # cls_scores 是 Sparse4DHead 输出的 classification list，单个元素 cls_scores[decoder_idx] shape 为 [B, 900, num_cls=10]
        box_preds,        # box_preds  是 Sparse4DHead 输出的 prediction list，单个元素 box_preds[decoder_idx] shape 为 [B, 900, box_dim=11]
        instance_id=None, # instance id：可选，若用于 tracking，会传入 shape [B, 900]
        quality=None,     # 质量估计：可选，例如 centerness、yawness 等，quality[decoder_idx] shape 通常为 6 × [B, 900, quality_dim=2]
        output_idx=-1,    # 使用哪一层 decoder 输出：默认 -1，表示使用最后一层 decoder 的预测
    ):
        '''
            【输入】
                cls_scores[stage] : [B, N=query数, K=类别数]，各 refine stage 的分类 logits。
                box_preds[stage]  : [B, N=query数, D]，各 refine stage 的内部 box state。
                instance_id       : None 或 [B, N]；Detection tracking 时传入。
                quality[stage]    : None 或 [B, N, 2]；通常 [centerness, yawness]。
                output_idx        : 要使用的 decoder stage；-1 代表最终 refine stage。
            
            【输出】
                output : 长度 B 的 list；每一项是：
                    {
                        "boxes_3d"    : [M, D_decoded]
                        "scores_3d"   : [M]
                        "labels_3d"   : [M]
                        "cls_scores"  : [M]，仅 quality 存在时额外保存的原分类分数
                        "instance_ids": [M]，仅 tracking 时存在
                    }
            其中 M <= num_output，若设 score_threshold 则还会进一步减少。
        '''

        # 1. 获取当前 refine stage 的分类分数和内部 box state
        # (1) 决定是否进入 tracking-aware 的 “每 query 只保留一个类别” 路径：
        squeeze_cls = instance_id is not None
        '''
            squeeze_cls 表示是否把类别维压缩掉：
                若 instance_id 不为 None，说明当前需要 tracking id，
                此时一个 slot 不能同时以多个类别输出，否则同一个 ID 会对应多个检测结果，
                因而每个 query 先取最大分类分数和对应类别：代码会先对每个 query 取最大类别分数，而不是把所有 query-class 展平。
        '''

        # (2) 获取 decoder 最后一层 refine 的最大类别分数 cls_scores 和最大类别分数的类别编号 cls_ids
        # (2.a) 读取第 output_idx=-1 层 decoder 的分类 logits，并做 sigmoid 得到独立类别概率
        cls_scores = cls_scores[output_idx].sigmoid() # lzw: output_idx=-1，所以是取最后一层 decoder 的分类输出，并做 sigmoid 得到每个类别的置信度（从 logits 变成概率式 score），返回 [B, 900, num_cls=10]
        # (2.b) Tracking 路径：每个 query 仅保留其最可能的一类
        if squeeze_cls: # 若有 instance_id
            cls_scores, cls_ids = cls_scores.max(dim=-1) # 对每个 query 在类别维上取最大分数和对应类别 id【cls_scores：[B, 900, 10] -> [B, 900]】【cls_ids：[B, 900]，记录每个 query 取最大分数时的类别编号】
            cls_scores = cls_scores.unsqueeze(dim=-1)    # 恢复一个长度为 1 的“伪类别维”【即把 cls_scores 扩展回 [B, 900, 1]】：这样后面 flatten + topk 可以与非 tracking 路径复用同一套代码
            
        # (3) 获取 decoder 最后一层 refine 的内部 box state，并取得基本维度
        box_preds = box_preds[output_idx]        # 取第 output_idx=-1 层 decoder 的 box 输出 box_preds，shape 为 [B, 900, box_dim=11]
        bs, num_pred, num_cls = cls_scores.shape # 读取 batch size、预测 query 数、类别数【若为 tracking 路径即若有 instance_id 即若 squeeze_cls=True，则 num_cls=1；若为非 tracking 路径，则 num_cls=真实类别数 K】


        # 2. 进行第一轮 top-300：从分类分数中选出最高 num_output=300 个候选，返回从高到低排序的 top-300 的分数 cls_scores [B, 300]]
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output,   # 每个样本最多保留 num_output=300 个候选结果
            dim=1,             # 在展平后的预测维度（即 query-class 维或 query 维）上执行 topk
            sorted=self.sorted # 是否排序：默认 True 表示输出按 score 从高到低排列
        )
        '''
            tracking 路径的 top-300：
                原 cls_scores shape: [B, 900, num_cls=1]
                flatten(start_dim=1) 展平后 shape: [B, 900*1=900]
                然后取 top-300 个最高分
            即 cls_scores：[B, 900, 1] → flatten → [B, 900]，一个 index 直接对应一个 query_id → top-300 → [B, 300]，一个 index 对应一个 query_id
        
            非 tracking 路径的 top-300：
                原 cls_scores shape: [B, 900, num_cls=10]
                flatten(start_dim=1) 展平后 shape: [B, 900*10]
                然后取 top-300 个最高分
            即 cls_scores：[B, 900, 10] → flatten → [B, 900*10]，一个 index 对应一个 (query_id, class_id) → top-300 → [B, 300]，一个 index 对应一个 (query_id, class_id)
        '''


        # 3. 非 tracking 路径：根据展平索引还原类别 id
        if not squeeze_cls:             # 若没有 squeeze 类别
            cls_ids = indices % num_cls # 则根据展平索引还原类别 id：由于 indices = query_id * num_cls + cls_id，所以 cls_id = indices % num_cls


        # 4. 可选：基于 “第一轮纯分类分数” 构造 threshold mask
        # 注意：quality 存在时，之后分数会再乘 centerness 并重排；
        # 原代码在乘 centerness 前创建 mask，之后仅同步重排该 mask，而没有重新按最终 score 计算 mask。
        if self.score_threshold is not None:          # 若设置了分数阈值
            mask = cls_scores >= self.score_threshold # 生成分数 mask，shape [B, num_output]，True 表示当前候选通过初始分类分数阈值


        # 5. 判断指定 refine stage 是否实际提供了 quality 分支输出
        # 【quality 在 Detection 时通常是 list，quality[output_idx] 为 [B, N, 2]；Map 任务或关闭 quality estimation 时，该位置可能为 None】
        if quality[output_idx] is None: # 若当前 decoder 层没有 quality 输出
            quality = None              # 统一令整体 quality=None，下面直接跳过 quality re-ranking


        # 6. 可选质量重排序：final_score = cls_score × sigmoid(centerness) 即用 centerness 重新加权分类分数
        if quality is not None: # 若存在 quality
            # (1) 取出 centerness 质量分支：提取每个 query 的 centerness logit，centerness shape [B, N]，CNS 是 box3d.py 中定义的 quality 维度索引
            centerness = quality[output_idx][..., CNS]

            # (2) 为 top-k 候选收集对应 query 的 centerness：
            centerness = torch.gather(centerness, 1, indices // num_cls)
            '''
                根据 topk 的 query 索引取对应 query 的 centerness：
                    indices 是 flatten 后的 query-class 索引或 query 索引
                        非 tracking：query_id * K + class_id；
                        tracking：query_id * 1 + 0。
                    所以 indices // num_cls 均可还原 query_id。
            '''

            # (3) 保存纯分类分数，供输出字段 "cls_scores" 使用
            cls_scores_origin = cls_scores.clone() # 保存一份原始分类分数，后面输出 cls_scores 字段时会用到

            # (4) 将 centerness 从 logit 转成 [0,1] 概率后乘入分类分数，最终 score = P(class) × P(center quality)
            cls_scores *= centerness.sigmoid() # 用 centerness 重新加权分类分数：centerness 是 logits，所以先 sigmoid，最终 score = class_score * centerness_score
            
            # (5) 乘质量后，重新按最终 score 降序排列 cls_scores，同时同步排列 cls_ids、mask、indices
            # (5.1) 乘质量后，重新按最终 score 降序排列
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True) # 对重新加权后的分数重新排序

            # (5.2) 非 tracking 时，同步重排类别 id
            if not squeeze_cls:                         # 若没有 squeeze 类别
                cls_ids = torch.gather(cls_ids, 1, idx) # 根据新的排序 idx，同步重排类别 id

            # (5.3) 若有 threshold mask，也同步到重排后的候选顺序           
            if self.score_threshold is not None:  # 若设置了 score threshold
                mask = torch.gather(mask, 1, idx) # 同步重排 mask

            # (5.4) 同步重排原始 flatten indices【后续 box / ID 均依赖这些索引】
            indices = torch.gather(indices, 1, idx) # 同步重排 indices，后面要用新的 indices 取 box


        output = [] # 初始化输出列表，每个 batch 样本一个 dict
        # 7. 逐 batch 样本整理最终输出 dict
        for i in range(bs): # 遍历 batch 中每个样本，i 表示 batch 中第 i 个样本
            # (1) 得到当前样本每个候选的类别 id
            # (1.a) 取第 i 个样本的类别 id
            category_ids = cls_ids[i]

            # (1.b) tracking 路径下 cls_ids 还保留在 [B, N] 的 “每 query 最大类别” 布局，此时 indices[i] 已是 query_id，因此需要按 top-k query 重新 gather
            if squeeze_cls:                             # 若 squeeze_cls=True，则是 tracking 路径
                category_ids = category_ids[indices[i]] # 这里 category_ids 原本 shape 是 [num_query]，而 indices[i] 是 topk 后的 query 索引，因为 num_cls=1，所以根据 indices 取出 topk 对应的类别 id
                
            # (2) 取最终分数与对应内部 box state
            scores = cls_scores[i]                    # 取第 i 个样本的 topk 分数
            box = box_preds[i, indices[i] // num_cls] # 取第 i 个样本的 topk box：box 是按 query 存储的，不按 query-class 存储，因而必须先从 flatten index 还原 query_id = indices // num_cls

            # (3) 可选 threshold 过滤：类别、分数、box 必须使用同一 mask 同步筛选
            if self.score_threshold is not None:     # 若设置了分数阈值
                category_ids = category_ids[mask[i]] # 根据 mask 过滤类别 id
                scores = scores[mask[i]]             # 根据 mask 过滤分数
                box = box[mask[i]]                   # 根据 mask 过滤 box

            # (4) quality 存在时，同步取未乘 centerness 的纯分类分数
            if quality is not None:                        # 若使用了 quality，
                scores_origin = cls_scores_origin[i]       # 则取第 i 个样本原始分类分数，
                if self.score_threshold is not None:       # 若设置了分数阈值，
                    scores_origin = scores_origin[mask[i]] # 则同步过滤原始分类分数。

            # (5) 解码 box：由内部 box state → 最终 [x,y,z,w,l,h,yaw,vx,...] 格式
            box = decode_box(box)

            # (6) 将基础 detection 字段写入当前样本输出 output
            output.append(
                {
                    "boxes_3d": box.cpu(),           # 3D boxes。.cpu()：转到 CPU，即评测、保存、可视化通常在 CPU 上消费这些结果
                    "scores_3d": scores.cpu(),       # 3D 检测分数
                    "labels_3d": category_ids.cpu(), # 3D 检测类别标签
                }
            ) # 将当前样本结果加入 output

            # (7) quality 存在时额外保存纯分类分数，便于分析分数重标定效果
            if quality is not None:                            # 若有 quality
                output[-1]["cls_scores"] = scores_origin.cpu() # 额外保存原始分类分数：scores_3d 是乘了 centerness 后的分数【scores_3d = cls_score × centerness】，cls_scores 是未乘 centerness 前的原始分类分数
                
            # (8) tracking 时按相同 top-k indices 取 instance ID
            if instance_id is not None: # 若传入了 instance_id
                # (a) 根据 topk indices 取对应 instance id：tracking 路径 num_cls=1，因此 indices[i] 即 query_id
                ids = instance_id[i, indices[i]]

                # (b) 与类别 / score / box 保持完全相同的 threshold 过滤
                if self.score_threshold is not None: # 若设置了分数阈值
                    ids = ids[mask[i]]               # 同步过滤 instance id

                # (c) 保存 instance id【instance_ids 不调用 .cpu()，保留原始源码行为】
                output[-1]["instance_ids"] = ids

        # (9) 返回长度为 B 的结果 list
        return output # 返回 batch 级输出结果

