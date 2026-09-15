# =============================================================================
# attention.py（带层次化中文注释版）
# =============================================================================
#
# 【文件总览】
# 本文件实现 SparseDrive 的 FlashAttention 注意力模块，整体分为四层：
#     (1) _in_projection_packed()：将输入 Q / K / V 分别做线性投影；
#     (2) FlashAttention：直接调用外部 flash-attn CUDA kernel，计算多头缩放点积注意力；
#     (3) FlashMHA：负责多头拆分、K/V 打包、输出投影；
#     (4) MultiheadFlashAttention：注册到 MMCV ATTENTION registry，兼容
#         MMDetection / MMCV 风格的 query、key、value、identity、query_pos、key_pos 调用接口。
#
# 【SparseDrive 中的实际用途】
# detection3d_head.py 中：
#     - gnn      ：当前帧 N 个 sparse instance 的 self-attention；
#     - temp_gnn ：当前帧 instance 读取历史 temporal instances 的 cross-attention。
#
# 两者最终调用的都是本文件注册的 MultiheadFlashAttention；
# 差别只在于传入的 Q / K / V 是否来自同一组 instance。
# =============================================================================

import warnings # 导入 warnings 模块，用于在参数弃用、位置编码缺失等情况下发出警告
import math     # 导入 math 模块，后面 gen_sineembed_for_position() 中会用到 math.pi
import torch
import torch.nn as nn

# 从 torch.nn.functional 中导入 linear 函数
# 这里用于手动实现 q/k/v 的线性投影
from torch.nn.functional import linear

# 导入权重初始化函数
# xavier_uniform_ 用于初始化 in_proj_weight
# constant_ 用于把 bias 初始化为 0
from torch.nn.init import xavier_uniform_, constant_


# MMCV 的弃用 API 警告装饰器
# 用于提示 residual 参数已经改名为 identity
from mmcv.utils import deprecated_api_warning

# MMCV 的 fp16 自动转换装饰器
# FlashAttention CUDA kernel 通常要求 fp16 或 bf16 输入
from mmcv.runner import auto_fp16

# MMCV 的 BaseModule
# MultiheadFlashAttention 继承它，方便接入 MMCV 初始化和注册机制
from mmcv.runner.base_module import BaseModule

# 构建 dropout 层的工具函数
# 可以根据配置 dict 构建 Dropout、DropPath 等
from mmcv.cnn.bricks.drop import build_dropout

# ATTENTION 注册表
# MultiheadFlashAttention 会注册到这里
# 这样配置文件里写 type="MultiheadFlashAttention" 时可以自动构建
from mmcv.cnn.bricks.registry import ATTENTION

# 导入 torch checkpoint 工具
# 当前文件中 cp 实际没有使用，属于冗余导入或历史遗留
import torch.utils.checkpoint as cp


# einops 的 rearrange，用于清晰地 reshape / permute 多头注意力中的张量布局，方便地重排张量维度，比如 [B, S, H, D] <-> [B*S, H, D]
from einops import rearrange


# =============================================================================
# 【FlashAttention 外部库接口兼容】
# flash-attn 不同版本对“变长 / 去 padding K-V packed attention”函数命名不同。
# 这里优先尝试旧名；失败时退化为新名并统一别名，后续代码只需调用一个名字。
# =============================================================================
# 尝试导入旧版本 flash-attn 接口
try:
    # flash_attn_unpadded_kvpacked_func 是 FlashAttention 的变长/去 padding 版本接口
    # kvpacked 表示 key 和 value 会被打包在一个张量里
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func

    # 如果导入成功，打印当前使用的接口名
    print('Use flash_attn_unpadded_kvpacked_func')

# 如果旧接口导入失败
except:
    # 导入新版本 flash-attn 中的 varlen 接口，并重命名成旧接口名，这样后面代码不用改
    from flash_attn.flash_attn_interface import  flash_attn_varlen_kvpacked_func as flash_attn_unpadded_kvpacked_func

    # 打印当前使用的接口名
    print('Use flash_attn_varlen_kvpacked_func')


# 导入 flash-attn 的 padding 工具
# unpad_input：移除 K/V 中的 padding token，并返回有效 token 与累计长度信息；
# pad_input：把去 padding 后的结果恢复成原 shape，当前文件没有使用
# index_first_axis：按第一维索引，当前文件没有使用
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis



# 工具函数 _in_projection_packed：手动实现 MultiheadAttention 里的 q/k/v 三路线性投影
def _in_projection_packed(q, k, v, w, b = None):
    '''
        【函数作用】
            将同一份拼接参数 w / b 拆成 Q、K、V 三组参数，随后分别执行：
                Q_proj = q @ W_q^T + b_q
                K_proj = k @ W_k^T + b_k
                V_proj = v @ W_v^T + b_v

        【典型输入 shape】
            q : [B, L_q, C]
            k : [B, L_k, C]
            v : [B, L_k, C]
            w : 拼接后的投影权重，[3C, C]，等价于沿第 0 维拼接 [W_q; W_k; W_v]
            b : 拼接后的 bias，[3C] 或 None，等价于拼接 [b_q; b_k; b_v]

        【输出 shape】
            q_proj : [B, L_q, C]
            k_proj : [B, L_k, C]
            v_proj : [B, L_k, C]
    '''

    # (1) 沿第 0 维把 [3C, C] 的总投影权重 w 均分为 Q、K、V 三份，权重 w_q、w_k、w_v 的形状均为 [C, C]
    w_q, w_k, w_v = w.chunk(3)

    # (2) 根据是否配置 bias，准备三组投影偏置
    if b is None: # (a) 若无 bias 时，三个函数式 linear() 调用都传入 None 偏置
        b_q = b_k = b_v = None # q/k/v 的 bias 都设为 None
    else: # (b) 若有 bias 时，将 [3C] 的总偏置 bias 均分为 Q、K、V 三份， 偏置向量 b_q, b_k, b_v 的形状均为 [C]
        b_q, b_k, b_v = b.chunk(3)

    # (3) 分别对 Q、K、V 执行线性投影：linear(x, W, b) 等价于 x @ weight.T + bias
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# 一、FlashAttention：最底层 FlashAttention CUDA kernel 调用包装器
class FlashAttention(nn.Module):
    '''
        【类作用】
            本类不负责 Q/K/V 线性投影，也不负责多头拆分；
            它只接收已经拆成多头的 Q、K、V，并调用 flash-attn 的高效 CUDA kernel 计算缩放点积注意力：
                Attention(Q, K, V) = Softmax(QK^T / sqrt(d)) V

            与普通 PyTorch attention 相比，FlashAttention 的核心优势是：
                (1) 不显式存储巨大的 [L_q, L_k] attention matrix；
                (2) 更少的 HBM 显存读写；
                (3) 更快且更省显存。

        【输入格式】
                q  : [B, L_q, H, D_h]
                kv : [B, L_k, 2, H, D_h]，其中 kv[:,:,0] 是 K、kv[:,:,1] 是 V
                FlashAttention 是最底层的 attention 计算模块
                它假设：输入 q 已经被拆成多头形式 [B, T, H, D]，k/v 已经被打包成 [B, S, 2, H, D]，其中 2 表示 key 和 value
    '''

    # 1. 初始化底层 FlashAttention 包装器
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__() # 调用 nn.Module 初始化，注册当前模块

        # (1) 保存 attention softmax 的缩放系数：如果 softmax_scale=None 时，FlashAttention 内部使用默认 1/sqrt(head_dim)
        self.softmax_scale = softmax_scale

        # (2) 保存 attention dropout 概率【仅训练阶段实际传入非零 dropout】
        self.dropout_p = attention_dropout

        # (3) 标记支持 fp16，即标记当前模块可参与 MMCV 的 fp16 自动混合精度流程【MMCV 的 auto_fp16 会参考该属性】
        self.fp16_enabled = True

    # 2. 前向传播 forward()：调用 flash-attn CUDA kernel 计算注意力
    # 自动将 q 和 kv 转成 fp16，并要求装饰器将函数输出恢复到 fp32【out_fp32=True 表示输出转回 fp32】
    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
    def forward(self, q, kv, causal=False, key_padding_mask=None):
        '''
            【函数作用】
                根据 K/V 是否含有 padding，选择两种 flash-attn 数据准备路径：
                    (1) 无 padding：直接将 batch 与序列维展平，然后调用变长 attention kernel；
                    (2) 有 padding：先移除无效 K/V token，再调用变长 attention kernel。

            【参数】
                q                : [B, L_q, H, D_h]，已经完成 Q 投影与多头拆分。
                kv               : [B, L_k, 2, H, D_h]，已经完成 K/V 投影、多头拆分并打包。
                causal           : 是否启用因果 mask；SparseDrive 的 gnn/temp_gnn 通常为 False。
                key_padding_mask : [B, L_k] 或 None；用于指示 K/V 中需要处理的 padding 位置。

            【返回】
                output : [B, L_q, H, D_h]
                None   : flash-attn 默认不显式返回完整 attention 权重，以节省显存。
        '''
        # ====================== 一、准备工作 ========================
        # (1) 断言检查
        # (a) FlashAttention kernel 只接受 fp16 或 bf16 输入；auto_fp16 装饰器负责正常训练时的类型转换
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16] # FlashAttention 要求输入数据类型是 fp16 或 bf16

        # (b) FlashAttention 是 CUDA kernel，因此 Q/K/V 必须位于 GPU
        assert q.is_cuda and kv.is_cuda # FlashAttention 要求输入在 CUDA GPU 上

        # (c) 检查 batch size、head 数、每个 head 的维度 head_dim 是否一致，否则 QK^T 无法计算
        # q.shape[0] == kv.shape[0]：batch size 一致
        # q.shape[-2] == kv.shape[-2]：num_heads 一致
        # q.shape[-1] == kv.shape[-1]：head_dim 一致
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1] # 检查 batch size、head 数、head_dim 是否一致

        # (2) 读取 batch size、 query 序列长度【记为 L_q】、key/value 序列长度【记为 L_k 和 L_v】
        batch_size = q.shape[0]                      # 从 Q 的第 0 维读取 batch size B
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1] # query 序列长度和 key/value 序列长度


        # ====================== 【核心】二、根据是否存在 K/V padding mask，选择不同的数据准备方式 ======================
        # 1. 无 key padding mask 路径：如果没有 key padding mask，也就是每个 batch 中序列长度都是固定一样的，则没有 padding token 需要忽略，所有 token 都有效
        if key_padding_mask is None:
            # (1) 合并 batch 维与序列维：
            # q  : [B, L_q, H, D_h]    -> [B*L_q, H, D_h]
            # kv : [B, L_k, 2, H, D_h] -> [B*L_k, 2, H, D_h]
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')

            # (2) 记录每个样本的最大 Q / K 序列长度；无 padding 时就是原序列长度
            max_sq = seqlen_q # query 最大长度
            max_sk = seqlen_k # key 最大长度

            # (3.1) 构造 query 累积序列长度【cu_seqlens_q shape 为 [B+1]】：如 B=2、L_q=4 时的 query 累积序列长度 cu_seqlens_q 为 [0, 4, 8]
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)

            # (3.2) 构造 key/value 累积序列长度【cu_seqlens_k shape 为 [B+1]】：如 B=2、L_k=6 时的 key/value 累积序列长度 cu_seqlens_k 为 [0, 6, 12]
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=kv.device)

            # (4) 调用 flash-attn CUDA kernel，得到展平后的多头 attention 输出 [B*L_q, H, D_h]
            output = flash_attn_unpadded_kvpacked_func(   # 调用 FlashAttention 核心函数
                q,                                        # query，shape [B*T, H, D]
                kv,                                       # key/value 打包，shape [B*S, 2, H, D]
                cu_seqlens_q,                             # query 累积长度
                cu_seqlens_k,                             # key 累积长度
                max_sq,                                   # query 最大长度
                max_sk,                                   # key 最大长度
                self.dropout_p if self.training else 0.0, # 训练时使用 dropout，测试时 dropout=0
                softmax_scale=self.softmax_scale,         # softmax 缩放因子
                causal=causal                             # 是否使用 causal mask：causal=True 常用于自回归语言模型，SparseDrive 里一般不是 causal attention
            )

            # (5) 将输出还原回 batch-first 的多头格式：[B*L_q, H, D_h] -> [B, L_q, H, D_h]
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size) # 将输出从 [B*S, H, D] 恢复成 [B, S, H, D]

        # 2. 有 key padding mask 路径：只移除 K/V 中的 padding token，避免无效 token 占计算量
        else:
            # (1) 获取 key/value 的 attention head 数，记为 H
            nheads = kv.shape[-2] # 从 kv 的倒数第二维读取 attention head 数 H

            # (2) q 仍然展平成 [B*L_q, H, D_h]【Q 不在此分支中去 padding，只合并 B、L_q 两维：[B, L_q, H, D_h] -> [B*L_q, H, D_h]】
            q = rearrange(q, 'b s ... -> (b s) ...')

            # (3) query 最大长度：Q 的最大序列长度仍是原始 L_q
            max_sq = seqlen_q

            # (4) 构造 query 累积序列长度：该形式供变长 attention kernel 识别 batch 边界
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)

            # (5) 先把 kv 从三维 [B, S, 2, H, D_h] 压成一维 [B, S, 2*H*D_h]，方便 flash-attn 的 unpad_input() 处理
            x = rearrange(kv, 'b s two h d -> b s (two h d)')

            # (6) 根据 key_padding_mask 移除 K/V 的无效 token：
            # x_unpad      ：所有有效 K/V token 拼接后的张量
            # indices      ：有效 token 的索引（本函数后续未使用）
            # cu_seqlens_k ：每个 batch 有效 K/V 长度的累积和
            # max_sk       ：最大有效 key 长度【即batch 中最大的有效 K/V 序列长度】
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)

            # (7) 将去 padding 后的 K/V（即 x_unpad）恢复为 kernel 所需的 kv packed 格式：[有效 token 数, 2*H*D_h] -> [有效 token 数, 2, H, D_h]
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)

            # (8) 调用 FlashAttention：使用去 padding 后 K/V 调用 flash-attn CUDA kernel，输出为展平的 [B*L_q, H, D_h]
            output_unpad = flash_attn_unpadded_kvpacked_func(
                q,                                        # query 没有去 padding，因为 q 这里假设没有 query padding
                x_unpad,                                  # 去 padding 后的 kv
                cu_seqlens_q,                             # query 累积长度
                cu_seqlens_k,                             # key 累积长度
                max_sq,                                   # query 最大长度
                max_sk,                                   # key 最大有效长度
                self.dropout_p if self.training else 0.0, # dropout
                softmax_scale=self.softmax_scale,         # softmax 缩放
                causal=causal                             # causal 开关
            )

            # (9) 将输出恢复为 batch-first 多头格式：[B*L_q, H, D_h] -> [B, L_q, H, D_h]
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size) # 将输出从 [B*S, H, D] 恢复成 [B, S, H, D]

        # 3. 返回 attention 多头输出和 attention weights【FlashAttention 通常不显式保存 attention weight，所以第二个返回 None】
        return output, None # output shape: [B, L_q, H, D_h]，attn_weights shape: None


# 二、FlashMHA：多头注意力的中间包装层
class FlashMHA(nn.Module):
    # 它负责：
    # 1. 输入 q/k/v 是 [B, S, C]
    # 2. 线性投影成多头 q/k/v
    # 3. 调用 FlashAttention 做注意力
    # 4. 输出投影回 [B, S, C]
    '''
        【类作用】
            FlashMHA 是一个多头注意力模块，本类负责标准 Multi-Head Attention 的除核心 attention kernel 外的所有准备工作，它负责：
                (1) 用合并参数对输入分别生成 Q/K/V；
                (2) 线性投影将通道 C 拆为 H 个 attention heads 的多头 Q/K/V；
                (3) 将 K/V 打包为 FlashAttention 需要的 kv 格式；
                (4) 调用底层 FlashAttention 做注意力；
                (5) 合并多头并输出投影

        【典型 shape】
            输入 q/k/v : [B, L, C]
            多头后     : [B, L, H, D_h]，其中 C = H * D_h
            输出       : [B, L_q, C]
    '''

    # 1. 初始化多头注意力所需的参数与子模块
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0, causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first                                  # 当前实现只支持 batch_first=True，即所有外部输入均要求必须是 [B, L, C]
        factory_kwargs = {'device': device, 'dtype': dtype} # 整理 device、dtype 工厂参数，用于传给 FlashAttention
        super().__init__()                                  # 调用 nn.Module 初始化

        # (1) 保存 embedding 维度、是否使用 causal attention、是否使用 bias
        self.embed_dim = embed_dim # 总 embedding 维度 C
        self.causal = causal       # 是否使用 causal attention【SparseDrive 的 gnn / temp_gnn 默认并不使用因果约束】
        self.bias = bias           # 是否使用 bias【是否为 Q/K/V 投影和输出投影使用 bias】

        # (2) 保存头数 self.num_heads 和每个 head 的维度 self.head_dim，并检查
        self.num_heads = num_heads # 保存 attention head 数量 H
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads" # 检查 embed_dim 必须能被 num_heads 整除
        self.head_dim = self.embed_dim // num_heads # 每个单 head 的维度 D_h = C / H
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8" # FlashAttention 对每头维度 head_dim 有约束，这里要求每头维度 head_dim 能被 8 整除且不超过 128

        # (3) 创建拼接式 Q/K/V 投影参数 [3C, C]，逻辑上等价于分别存 W_q、W_k、W_v 三个 [C,C] 矩阵
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim))) # q/k/v 三个投影的合并权重，shape [3*embed_dim, embed_dim]

        # (4) 根据 bias 开关创建或注册 Q/K/V 的拼接偏置 [3C]
        if bias:                                                         # 如果使用 bias
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) # 使用 bias 时，创建可训练偏置参数（即创建 q/k/v 三个投影的合并 bias），shape [3*embed_dim]
        else:                                                            # 如果不使用 bias
            self.register_parameter('in_proj_bias', None)                # 不使用 bias 时，注册一个 None 参数，保持 state_dict 与 module 结构逻辑一致

        # (5) 创建底层 FlashAttention 模块，它只接收已经多头拆分并 K/V packed 的张量
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)

        # (6) 创建输出投影层：将拼接后的多头输出 [B, L_q, C] 再做一次 C->C 的线性融合
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias) # 将多头拼接后的结果再投影回 embed_dim

        # (7) 初始化参数：初始化 Q/K/V 与输出投影相关参数
        self._reset_parameters()

    # 2. 初始化 FlashMHA 内部可训练参数：
    def _reset_parameters(self) -> None:
        # (1) 初始化 q/k/v 投影权重：对拼接式 Q/K/V 权重 [3C,C] 使用 Xavier uniform 初始化
        xavier_uniform_(self.in_proj_weight)

        # (2) 只有配置 bias 时才初始化偏置：
        if self.in_proj_bias is not None:     # 如果有 bias
            constant_(self.in_proj_bias, 0.)  # 则将 Q/K/V 投影偏置 bias 初始化为 0
            constant_(self.out_proj.bias, 0.) # 且将输出投影 out_proj 的 bias 初始化为 0；out_proj.weight 沿用 nn.Linear 默认初始化
        
    # 3. 完成 Q/K/V 投影、多头拆分、FlashAttention 调用与输出投影
    def forward(self, q, k, v, key_padding_mask=None):
        '''
            【函数作用】
                输入 batch-first 的 Q/K/V，计算标准多头 attention：
                    (1) Q/K/V 线性投影；
                    (2) 通道维拆分为多头；
                    (3) 打包 K/V；
                    (4) 调用 FlashAttention；
                    (5) 拼回多头并输出投影。

            【输入】
                q : [B, L_q, C]
                k : [B, L_k, C]
                v : [B, L_k, C]
        '''
        # (1) 使用拼接权重分别对输入 Q/K/V 执行线性投影；shape 分别保持 [B, L_q, C]、[B, L_k, C]、[B, L_k, C]
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)

        # (2) 拆头：
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads) # 将 Q 的最后一维 C 拆成 H 个 head、每头 D_h 维：[B, L_q, C] -> [B, L_q, H, D_h]
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads) # 将 K 的最后一维 C 拆成多头：[B, L_k, C] -> [B, L_k, H, D_h]
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads) # 将 V 的最后一维 C 拆成多头：[B, L_k, C] -> [B, L_k, H, D_h]

        # (3) 将 K 与 V 在新建的第 2 维打包在一起：[B, L_k, H, D_h]×2 -> [B, L_k, 2, H, D_h]
        kv = torch.stack([k, v], dim=2)
        
        # (4) 调用底层 FlashAttention：返回 context [B, L_q, H, D_h] 与 None 的 attention weights
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)

        # (5) 合并多头 [B, L_q, H, D_h] -> [B, L_q, C]，再执行输出线性投影，并返回结果及 None
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights # 将 context 从 [B, L_q, H, D_h] 合并回 [B, L_q, C]，然后经过输出投影


# 三、MultiheadFlashAttention：MMCV / SparseDrive 可配置注意力模块
@ATTENTION.register_module() # 将 MultiheadFlashAttention 注册到 MMCV ATTENTION 注册表，这样 SparseDrive 配置文件中的 type="MultiheadFlashAttention" 就会构建这个类
class MultiheadFlashAttention(BaseModule):
    '''
        【类作用】
            这是 SparseDrive 实际配置和 detection3d_head.py 直接调用的注意力层。
            它对 FlashMHA 再包一层，以兼容 MMCV 标准 attention 接口：
                query、key、value、identity、query_pos、key_pos、key_padding_mask。

            其实际计算链为：
                query/key 加位置编码
                -> FlashMHA(Q,K,V)
                -> output dropout
                -> residual identity 相加

        【SparseDrive 对照】
            (1) gnn：调用时 Q=K=V=当前 instance_feature，属于 self-attention；
            (2) temp_gnn：Q=当前 instance_feature，K/V=历史 temp_instance_feature，属于 cross-attention；
            (3) query_pos/key_pos：通常由当前 / 历史 anchor 的几何 embedding 提供。
    '''

    # 1. 初始化 MMCV 兼容的 FlashAttention 多头注意力层：
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        '''
            MultiheadFlashAttention 是对 FlashMHA 的 MMCV 风格封装
            主要为了兼容 mmdet/mmcv transformer layer 的调用方式：
            query, key, value, identity, query_pos, key_pos 等        
        '''
        
        # (1) 调用 MMCV BaseModule 初始化，并把 init_cfg 交给 MMCV 权重初始化机制
        super(MultiheadFlashAttention, self).__init__(init_cfg) # 调用 BaseModule 初始化

        # (2) 兼容旧版 MMCV 参数 dropout；现在建议分别使用 attn_drop、proj_drop 与 dropout_layer
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning) # 发出弃用警告
            attn_drop = kwargs['dropout']                         # 将旧的 dropout 参数作为 attention dropout
            dropout_layer['drop_prob'] = kwargs.pop('dropout')    # 同时设置 residual dropout_layer 的 drop_prob

        # (3) 保存 embedding 维度、head 数，并强制 batch_first=True
        self.embed_dims = embed_dims # 保存总 embedding 维度 C
        self.num_heads = num_heads   # 保存 attention head 数 H
        self.batch_first = True      # 当前实现强制使用 batch_first=True 格式 [B, L, C]，因此传入 batch_first 参数不会改变该赋值

        # (4) 构建 FlashMHA【内部要求 CUDA + fp16/bf16，因此这里明确传 device='cuda'、dtype=torch.float16】
        self.attn = FlashMHA(
            embed_dim=embed_dims,        # embedding 维度
            num_heads=num_heads,         # head 数量
            attention_dropout=attn_drop, # attention dropout
            dtype=torch.float16,         # 指定 dtype 为 float16。注意 FlashAttention 要求输入为 fp16 或 bf16
            device='cuda',               # 指定 device 为 cuda，这也说明该模块默认依赖 GPU
            **kwargs                     # 其他参数，例如 causal
        )

        # (5) 创建 attention 输出后的投影 dropout
        self.proj_drop = nn.Dropout(proj_drop) # 输出后的 dropout

        # (6) 构建 residual 分支上的 dropout layer：如果 dropout_layer 不为空，则根据配置构建，否则使用恒等映射 Identity
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    # 2. 兼容旧 API：兼容旧版 residual 参数名，如果外部传 residual，会提示改用，则将 residual 自动重命名为 identity
    @deprecated_api_warning({'residual': 'identity'}, cls_name='MultiheadAttention')
    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None, **kwargs):
        '''
            【函数作用】
                按 MMCV attention 标准接口组织 self-attention / cross-attention：
                    (1) 补全 key、value、identity 的默认值；
                    (2) 为 query / key 加入几何位置编码；
                    (3) 调用 FlashMHA；
                    (4) 做 output dropout 与 residual connection。

            【典型 SparseDrive shape】
                gnn：
                    query=key=value=[B,N=900,C=512]（decouple attention 内部维度）
                temp_gnn：
                    query=[B,N=900,C=512]
                    key/value=[B,T=600,C=512]

            【注意】
                本 FlashAttention 包装当前明确不支持 attn_mask；若传入非 None 会直接 assert。
        '''

        # (1) 当前 FlashAttention 实现暂不支持二维 attention mask 的代码路径，因此要求 attn_mask 必须为 None，如果传入 attn_mask 会直接报错
        assert attn_mask is None, 'attn mask not supported now.'

        # (2) 设置默认 Key、Value、Identity：
        '''
            默认 Key              : 若未传 key     ，则使用 query，形成 self-attention
            默认 Value            : 若未传 value   ，则使用 key
            默认 residual identity: 若未传 identity，则使用尚未加位置编码的原始 query
        '''
        if key is None:      # 如果 key 为空
            key = query      # self-attention 情况下 key=query
        if value is None:    # 如果 value 为空，
            value = key      # value 默认等于 key
        if identity is None: # 如果 identity 为空
            identity = query # residual 分支默认使用 query

        # (3) 处理 query/key 的位置编码：
        # (3.1) 处理 key 的位置编码默认值：没有显式给 key_pos 时，只有 query_pos 存在时，才尝试复用 query_pos 作为 key_pos
        if key_pos is None:           # 如果没有显式给 key_pos
            if query_pos is not None: # 但给了 query_pos
                if query_pos.shape == key.shape:                                                         # 如果 query_pos 和 key 的形状一样
                    key_pos = query_pos                                                                  # 直接复用 query_pos 作为 key_pos【self-attention 中 query 与 key 的 shape 一般一致，因此可共用同一份位置编码】
                else:                                                                                    # 如果形状不一样
                    warnings.warn(f'position encoding of key is missing in {self.__class__.__name__}.')  # 给出警告：key 的位置编码缺失【cross-attention 中 Q/K 长度可能不同】

        # (3.2) 若提供 Query 的位置编码，则将它加到 Query 上【SparseDrive 中通常是当前 anchor 的 geometry embedding】
        if query_pos is not None:     # 如果 query_pos 不为空
            query = query + query_pos # 将 query 位置编码加到 query 上

        # (3.3) 若提供 Key 的位置编码，则将它加到 Key 上【temp_gnn 中通常是历史 temp_anchor 的 geometry embedding】
        if key_pos is not None: # 如果 key_pos 不为空
            key = key + key_pos # 将 key 位置编码加到 key 上

        # (4) FlashAttention 内部约定 batch-first [B, L, C]【由于 __init__() 强制 self.batch_first=True，正常 SparseDrive 路径不会进入该兼容分支；但仍保留原始代码，以便与 MMVC 接口的历史格式兼容】
        if not self.batch_first: # 如果 batch_first=False
            # Query：[L_q,B,C] -> [B,L_q,C]。
            # Key  ：[L_k,B,C] -> [B,L_k,C]。
            # Value：[L_k,B,C] -> [B,L_k,C]。
            query = query.transpose(0, 1) # 将 query 从 [L_q, B, C] 转成 [B, L_q, C]
            key = key.transpose(0, 1)     # 将 key   从 [L_k, B, C] 转成 [B, L_k, C]
            value = value.transpose(0, 1) # 将 value 从 [L_k, B, C] 转成 [B, L_k, C]
        
        # (5) 调用 FlashMHA，取其第一个返回值 attention output：[B, L_q, C]
        out = self.attn(
            q=query,                              # query
            k=key,                                # key
            v=value,                              # value
            key_padding_mask=key_padding_mask)[0] # key padding mask

        # (6) 若采用非 batch-first 外部格式，则将输出还原：[B, L_q, C] -> [L_q, B, C]
        if not self.batch_first:      # 如果输入原本不是 batch_first
            out = out.transpose(0, 1) # 将输出从 [B, N, C] 转回 [N, B, C]

        # (7) 先对 attention 输出做 proj dropout、residual dropout，再与 identity 相加，形成标准残差连接
        return identity + self.dropout_layer(self.proj_drop(out)) # residual 连接：identity + dropout(proj_drop(out))


# 工具函数：生成二维位置的 sine/cosine 位置编码
def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    '''
        Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/

        【函数作用】
            将二维位置 (x,y) 编码为 hidden_dim 维的固定 sin/cos positional embedding。实现基本沿用 DAB-DETR 的位置编码写法。

        【输入】
            pos_tensor : [...,2]，最后一维为2依次为 x、y。
            hidden_dim : 输出位置编码维度，默认 256，且通常应为偶数。

        【输出】
            pos : [...,hidden_dim]，前一半来自 y 的 sin/cos 编码，后一半来自 x 的 sin/cos 编码，最后一维是 hidden_dim
    '''

    # ====================== 一、准备工作 ========================
    # (1) 将总维度一分为二：一半用于编码 y，另一半用于编码 x
    half_hidden_dim = hidden_dim // 2

    # (2) 使用 2π 缩放坐标，使输入位置映射到一个完整圆周尺度
    scale = 2 * math.pi

    # (3) 创建频率维度索引 [0,1,...,half_hidden_dim-1]，并保持 float32 以确保 sin/cos 编码数值稳定
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device) # dim_t shape: [half_hidden_dim]

    # (4) 构造 Transformer 风格的正弦曲线位置编码的多频率分母：10000^(2 floor(i/2) / half_hidden_dim)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim) # 10000 ** (2i / d)

    # ====================== 二、计算二维 sin/cos 位置编码 ========================
    # (1) 将 pos_tensor 的最后一维拆分为 x、y 坐标，分别乘以 2π 缩放
    x_embed = pos_tensor[..., 0] * scale # 读取 x 坐标并乘 2π；shape 由 [...,2] 变为 [...]
    y_embed = pos_tensor[..., 1] * scale # 读取 y 坐标并乘 2π；shape 由 [...,2] 变为 [...]

    # (2) 将 x、y 坐标除以不同频率尺度 dim_t，增加一个频率维度，得到 [...,half_hidden_dim] 的编码 
    pos_x = x_embed[..., None] / dim_t # x 坐标除以不同频率尺度：将 x 与每个频率分母相除，增加一个频率维后 shape 为 [...,half_hidden_dim]
    pos_y = y_embed[..., None] / dim_t # y 坐标除以不同频率尺度：将 y 与每个频率分母相除，增加一个频率维后 shape 为 [...,half_hidden_dim]

    # (3) 对 x、y 编码分别使用 sin/cos 交错编码：
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2) # x 编码：对 x 的偶数频率维使用 sin，奇数频率维使用 cos，然后交错堆叠并展平回 [...,half_hidden_dim]
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2) # y 编码：对 y 的偶数频率维使用 sin，奇数频率维使用 cos，然后交错堆叠并展平回 [...,half_hidden_dim]

    # (4) 将 y 编码和 x 编码拼接（y 在前，x 在后），得到最终 [...,hidden_dim] 位置编码
    pos = torch.cat((pos_y, pos_x), dim=-1) # shape [..., hidden_dim]
    return pos # 返回固定二维 sin/cos 位置编码

