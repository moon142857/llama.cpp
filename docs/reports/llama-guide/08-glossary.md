# llama.cpp 术语表与核心概念详解

> **目标**：快速查阅 llama.cpp 和 LLM 推理领域的专业术语
> **版本**：llama.cpp build 9010 (commit d05fe1d7d)
> **日期**：2026-05-04

---

# 附录 A：专业术语速查表

> 本表遍历报告中所有专业名词，用一句话概括其含义，方便初学者快速查阅。

| 术语 | 英文全称 | 一句话解释 |
|------|---------|-----------|
| **GGUF** | GGML Universal Format | llama.cpp 专用的模型文件格式，包含元数据和张量数据 |
| **GGML** | — | llama.cpp 底层的张量计算库，定义了张量类型、算子和计算图 |
| **Tensor** | 张量 | 多维数组，是深度学习的基本数据单元（0D=标量，1D=向量，2D=矩阵，3D+=张量） |
| **Quantization** | 量化 | 将浮点权重（f32/f16）压缩为低精度整数（如 4bit/8bit），减少模型体积和内存带宽 |
| **q4_K** | 4-bit K-quant | 一种 4-bit 量化格式，每 256 个权重为一组 super-block，实际等效约 4.5 bits |
| **q5_K** | 5-bit K-quant | 类似 q4_K，但用 5-bit 存储权重，等效约 5.5 bits |
| **q8_0** | 8-bit quant | 每 32 个权重一组 block，用 int8 存储，精度较高 |
| **Backend** | 后端 | 执行张量运算的硬件/软件抽象，如 CPU、Metal、CUDA、Vulkan |
| **Scheduler** | 调度器 | `ggml_backend_sched`，负责将计算图中的算子分配到不同后端执行 |
| **Graph** | 计算图 | 由张量和算子构成的有向无环图（DAG），描述一次前向传播的全部计算 |
| **Node** | 节点 | 计算图中的算子（如 ADD、MUL_MAT、RMS_NORM），每个节点消费输入张量、产出输出张量 |
| **View** | 视图 | 对已有张量的零拷贝切片（如 `ggml_view_2d`），不复制数据，只修改形状/步长 |
| **Offload** | 卸载 | 将模型层从 CPU 内存转移到 GPU 显存执行，减少 CPU-GPU 数据传输 |
| **Embedding** | 嵌入 | 将离散 token ID 映射为连续向量（`token_embd.weight`），是模型的输入层 |
| **Hidden Dim** | 隐藏维度 | `n_embd`，Transformer 层内部的特征维度，本模型为 2048 |
| **Layer** | 层 | Transformer 的基本重复单元，本模型共 40 层 |
| **Attention** | 注意力机制 | 让模型在生成每个 token 时，能“关注”输入序列中相关的其他 token |
| **GQA** | Grouped Query Attention | 分组查询注意力，K/V 的头数少于 Q，减少 KV cache 内存（本模型 16:2） |
| **QKV** | Query/Key/Value | Attention 的三个输入投影：Q（查询）、K（键）、V（值） |
| **QKV Fusion** | QKV 融合 | 将 Q/K/V 的三个权重矩阵合并为一个 `attn_qkv.weight`，减少内存访问次数 |
| **Flash Attention** | — | 一种内存高效的 Attention 算法，通过分块计算避免存储完整的 N×N 注意力矩阵 |
| **KV Cache** | Key-Value 缓存 | 存储已生成 token 的 K/V 向量，避免重复计算，是推理加速的核心 |
| **RoPE** | Rotary Position Embedding | 旋转位置编码，通过旋转 Q/K 向量来注入位置信息，替代绝对位置编码 |
| **RMSNorm** | Root Mean Square Norm | 均方根归一化，LayerNorm 的简化版，只除 RMS 不加 bias，计算更快 |
| **SwiGLU** | Swish-Gated Linear Unit | 一种前馈网络激活函数：`SiLU(gate) * up`，比 ReLU/GELU 效果更好 |
| **SiLU** | Sigmoid Linear Unit | 激活函数 `x * sigmoid(x) = x / (1 + e^(-x))`，SwiGLU 的组成部分 |
| **GLU** | Gated Linear Unit | 门控线性单元，将输入分成 gate 和 up 两部分，gate 控制 up 的通过 |
| **MoE** | Mixture of Experts | 混合专家模型，每层的 FFN 由多个“专家”组成，路由网络决定激活哪些专家 |
| **Expert** | 专家 | MoE 中的子网络，通常是一个标准 FFN，本模型每层有 256 个专家 |
| **Routing** | 路由 | MoE 中选择专家的机制，输入经 gate 网络计算每个专家的权重，选 top-k |
| **MUL_MAT_ID** | Matrix Multiply with ID | 带索引的矩阵乘法，只计算选中的专家，是 MoE 的核心算子 |
| **ARGSORT** | Argument Sort | 返回排序后的索引（而非值），MoE 路由中用于选出 top-k 专家 |
| **Shared Expert** | 共享专家 | 所有 token 都会激活的公共 FFN，与 routed expert 相加，提升基础能力 |
| **SSM** | State Space Model | 状态空间模型，用卷积+门控网络替代标准 Attention，长序列效率更高 |
| **SSM_CONV** | SSM Convolution | SSM 中的局部卷积操作，对输入做 1D 卷积提取局部特征 |
| **GATED_DELTA_NET** | Gated Delta Net | 一种 SSM 变体，用门控机制控制状态更新 |
| **Sliding Window** | 滑动窗口注意力 | 只关注最近 N 个 token，降低长序列的 Attention 计算量，本模型关闭 |
| **BPE** | Byte-Pair Encoding | 字节对编码，一种子词分词算法，将文本拆成常见子词单元 |
| **Token** | 词元 | 模型处理的最小文本单位，可以是字、词或子词，本模型词表 248320 个 |
| **Vocab** | 词表 | 模型认识的所有 token 的集合 |
| **Sampler** | 采样器 | 从模型输出的 logits 中选择下一个 token 的模块 |
| **Logit** | 对数几率 | 模型输出层经过 softmax 前的原始分数，越高表示该 token 越可能被选中 |
| **Softmax** | — | 将 logits 转换为概率分布（和为 1），公式 `exp(x_i) / sum(exp(x_j))` |
| **Temperature** | 温度 | 控制采样随机性的参数，越高分布越平坦（越随机），越低越尖锐（越确定） |
| **Top-K** | — | 只保留概率最高的 K 个 token，其余置零，减少低概率噪声 |
| **Top-P** | Nucleus Sampling | 只保留累积概率达到 P 的 token，动态调整保留数量 |
| **Min-P** | — | 只保留概率 >= max_prob * P 的 token，进一步过滤低概率候选 |
| **Greedy** | 贪心采样 | 总是选择概率最高的 token，完全确定无随机性 |
| **Detokenize** | 反分词 | 将 token ID 序列还原为人类可读的文本 |
| **Ubatch** | Micro-batch | 一次前向传播处理的最小批次单元 |
| **Slot** | 槽位 | KV cache 中分配给某个序列的连续 cell 区域 |
| **Cell** | 单元 | KV cache 中最小存储单位，对应一个 token 的 K/V 向量 |
| **Stream** | 流 | 并行执行的序列通道，单序列场景下只有 1 个 stream |
| **Head** | 头 | Attention 中的并行注意力通道，本模型 Q 有 16 头，K/V 各有 2 头 |
| **Head Dim** | 头维度 | 每个注意力头的向量维度，本模型 `n_embd_head = 256 / 16 = 128`（注：实际为 256） |
| **Alibi** | Attention with Linear Biases | 用线性偏置替代位置编码，本模型未使用 |
| **Fused Op** | 融合算子 | 将多个连续操作合并为一个内核（如 RMS_NORM + MUL），减少内存访问 |

---

# 附录 B：核心概念代码详解

> 本附录结合源码，对每个核心概念进行“通俗解释 → 代码位置 → 数据流转”三级拆解，目标是让初学者能跟着代码理解数据是如何被一步步转换的。

---

## B.1 GQA（Grouped Query Attention）

### 通俗解释

标准 Multi-Head Attention（MHA）中，Q、K、V 各有 `n_head` 个头。但 K 和 V 的注意力头可以共享——**多个 Q 头共用同一组 K/V 头**，这就是 GQA。

本模型 `n_head=16, n_head_kv=2`，意味着：
- 16 个 Q 头各自独立计算查询向量
- 但 K/V 只有 2 个头，每 8 个 Q 头共享 1 个 K 头和 1 个 V 头
- KV cache 内存从 `40 * 16 * 256 * 262144 * 2` 降到 `40 * 2 * 256 * 262144 * 2`，**节省了 8 倍**

### 代码实现

**① 模型加载时创建融合 QKV 张量**（`src/llama-model.cpp:2518-2533`）

```cpp
void llama_model_base::create_tensor_qkv(llama_layer & layer, int bid,
        int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
        int flags) {
    // n_embd_q_ = n_embd_head * n_head     = 128 * 16 = 2048
    // n_embd_k_ = n_embd_head * n_head_kv  = 128 * 2  = 256
    // n_embd_v_ = n_embd_head * n_head_kv  = 128 * 2  = 256
    const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;  // = 2560
    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid),
                               {n_embd_, n_embd_qkv}, ...);
    // 如果不存在融合张量，则分别创建 wq/wk/wv
    if (!layer.wqkv) {
        layer.wq = create_tensor(..., {n_embd_, n_embd_q_}, ...);
        layer.wk = create_tensor(..., {n_embd_, n_embd_k_}, ...);
        layer.wv = create_tensor(..., {n_embd_, n_embd_v_}, ...);
    }
}
```

**② 图构建时从融合张量中切出 Q/K/V**（`src/llama-graph.cpp:1076-1100`）

```cpp
const int64_t n_embd_q  = n_embd_head * n_head;     // = 2048
const int64_t n_embd_kv = n_embd_head * n_head_kv;  // = 256

if (layer.wqkv) {
    // 融合路径：一次矩阵乘得到 QKV，再用 view 切片
    ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
    // Q: [head_dim=128, n_head=16, n_tokens]
    Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head, n_tokens,
        ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
    // K: [head_dim=128, n_head_kv=2, n_tokens]，偏移 n_embd_q
    Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
        ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
        ggml_row_size(qkv->type, n_embd_q));
    // V: [head_dim=128, n_head_kv=2, n_tokens]，偏移 n_embd_q + n_embd_kv
    Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
        ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
        ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
}
```

### 数据流转

```
输入: cur [n_embd=2048, n_tokens]
    │
    ▼ 矩阵乘: wqkv [2048, 2560]
    │
qkv [2560, n_tokens]  (2560 = 2048 + 256 + 256)
    │
    ├─► Qcur = view(qkv, offset=0)        [128, 16, n_tokens]
    ├─► Kcur = view(qkv, offset=2048)     [128, 2,  n_tokens]
    └─► Vcur = view(qkv, offset=2304)     [128, 2,  n_tokens]
```

**关键洞察**：`ggml_view_3d` 是零拷贝操作——Qcur/Kcur/Vcur 都指向 qkv 的同一块内存，只是形状和偏移不同。这意味着 **一次内存加载就能得到 Q/K/V 三个张量**，大幅减少了访存带宽。

---

## B.2 QKV Fusion（QKV 权重融合）

### 通俗解释

传统 Transformer 分别用 `wq`、`wk`、`wv` 三个矩阵做投影。QKV Fusion 将它们拼接成一个大矩阵 `wqkv`，**一次矩阵乘法完成三个投影**。

为什么这样做？因为矩阵乘法的瓶颈通常是内存带宽（加载权重），而非计算。把三个小矩阵乘法合并成一个大矩阵乘法，**权重只需要从内存加载一次**。

### 代码实现

**① 张量注册**（`src/llama-arch.cpp:382`）

```cpp
{ LLM_TENSOR_ATTN_QKV, "blk.%d.attn_qkv" },
```

**② 本模型融合 QKV 的形状**（`src/models/qwen35moe.cpp:68`）

```cpp
create_tensor_qkv(layer, i, n_embd,
    n_embd_head_k * n_head * 2,   // n_embd_q = 128 * 16 = 2048
    n_embd_k_gqa,                  // n_embd_k = 128 * 2 = 256
    n_embd_v_gqa,                  // n_embd_v = 128 * 2 = 256
    0);
```

**③ 切分逻辑**（同 B.1 中的 `src/llama-graph.cpp:1081-1100`）

注意 `ggml_row_size(qkv->type, n_embd_q)` 这个偏移量：
- `n_embd_q = 2048`（Q 部分占 2048 个元素）
- K 从偏移 2048 开始，占 256 个元素
- V 从偏移 2304 开始，占 256 个元素

如果模型没有融合 QKV（如早期模型），代码会走 `else` 分支：

```cpp
} else {
    // 分别做三次矩阵乘
    Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
    Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
}
```

### 数据流转

```
未融合路径（3 次加载权重）:
  cur ──► wq [2048,2048] ──► Qcur    (加载 wq)
  cur ──► wk [2048, 256] ──► Kcur    (加载 wk)
  cur ──► wv [2048, 256] ──► Vcur    (加载 wv)

融合路径（1 次加载权重）:
  cur ──► wqkv [2048,2560] ──► qkv   (加载 wqkv 一次)
              │
              ├─► view ──► Qcur
              ├─► view ──► Kcur
              └─► view ──► Vcur
```

---

## B.3 Flash Attention

### 通俗解释

标准 Attention 的计算顺序是：`Q × K^T → Softmax → × V`。其中 `Q × K^T` 会产生一个 `[n_tokens, n_kv]` 的注意力分数矩阵。当上下文很长时（如 256K），这个矩阵巨大（256K × 256K = 640 亿个元素），内存无法容纳。

Flash Attention 的核心思想是 **分块（tiling）**：
1. 将 Q、K、V 切成小块
2. 每次只加载一小块 K/V 到高速缓存（SRAM/Shared Memory）
3. 在 SRAM 内完成 softmax 和加权求和
4. 只输出最终结果，不存储中间的注意力矩阵

这样**时间和内存复杂度都从 O(N²) 降到了 O(N)**（实际实现中）。

### 代码实现

**① 图构建中的条件选择**（`src/llama-graph.cpp:1960-1982`）

```cpp
const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
if (use_flash_attn) {
    GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");
    // V cache 如果是转置的，需要先转回来
    if (v_trans) {
        v = ggml_transpose(ctx0, v);
    }
    // Flash Attention 要求 K/V 是 f16 类型
    if (k->type == GGML_TYPE_F32) {
        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
    }
    if (v->type == GGML_TYPE_F32) {
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);
    }
    // 调用 Flash Attention 扩展算子
    cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale,
        hparams.f_max_alibi_bias,
        hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
    ggml_flash_attn_ext_add_sinks(cur, sinks);
    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
} else {
    // 标准 Attention 路径（内存密集型）
    ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);  // [n_kv, n_tokens]
    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
    cur = ggml_mul_mat(ctx0, v, kq);  // [head_dim, n_tokens]
}
```

**② GGML 算子定义**（`ggml/src/ggml.c:5329-5371`）

```cpp
struct ggml_tensor * ggml_flash_attn_ext(...) {
    // 输出形状由 V 决定: [head_dim, n_head, n_tokens, batch]
    int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    float params[] = { scale, max_bias, logit_softcap };
    ggml_set_op_params(result, params, sizeof(params));
    result->op     = GGML_OP_FLASH_ATTN_EXT;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;
    return result;
}
```

**③ CUDA 后端多内核选择**（`ggml/src/ggml-cuda/fattn.cu:532-550`）

```cpp
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (ggml_cuda_get_best_fattn_kernel(...)) {
        case BEST_FATTN_KERNEL_TILE:      ggml_cuda_flash_attn_ext_tile(ctx, dst);      break;
        case BEST_FATTN_KERNEL_VEC:       ggml_cuda_flash_attn_ext_vec(ctx, dst);       break;
        case BEST_FATTN_KERNEL_WMMA_F16:  ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);  break;
        case BEST_FATTN_KERNEL_MMA_F16:   ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);   break;
    }
}
```

### 数据流转

```
标准 Attention（内存瓶颈）:
  q [128,16,1] ──┐
                 ├──► mul_mat ──► kq [262144,1] ──► softmax ──► attn_probs [262144,1]
  k [128,2,262144] ┘                                               │
                                                                   ▼
  v [128,2,262144] ─────────────────────────────────────────────► mul_mat ──► out [128,16,1]

Flash Attention（分块计算）:
  q [128,16,1] ──┐
  k [128,2,262144] ├─► flash_attn_ext_tile (分块在 SRAM 内计算)
  v [128,2,262144] ┘   └──► out [128,16,1]  (直接输出，不存中间矩阵)
```

**关键洞察**：Flash Attention 不是算法层面的改变（数学结果和标准 Attention 完全一致），而是**工程层面的内存优化**——通过分块和重计算避免存储巨大的中间注意力矩阵。

---

## B.4 RoPE（Rotary Position Embedding）

### 通俗解释

模型需要知道“第 5 个词在第 5 个位置”这个信息，否则 `"A 打了 B"` 和 `"B 打了 A"` 对它来说是一样的。

传统做法是给每个位置一个唯一的位置向量（绝对位置编码），加到输入上。RoPE 的做法更巧妙：**把 Q/K 向量在二维平面上旋转一个角度**，旋转角度随位置变化。

想象每个注意力头的向量是一对 `(x, y)`。对于位置 `pos`，RoPE 把它旋转 `pos × θ` 度：
- `x' = x·cos(pos·θ) - y·sin(pos·θ)`
- `y' = x·sin(pos·θ) + y·cos(pos·θ)`

这样**位置信息就被编码进了向量的方向中**，而且天然支持相对位置计算（因为旋转角度差就是位置差）。

### 代码实现

**① GGML 算子创建**（`ggml/src/ggml.c:4130-4185`）

```cpp
static struct ggml_tensor * ggml_rope_impl(...) {
    // a: 输入张量 [head_dim, n_head, n_tokens]
    // b: 位置索引 [n_tokens]（每个 token 在序列中的位置）
    // n_dims: 旋转的维度数（通常等于 head_dim）
    // freq_base: 基础频率（如 10000.0）
    // ...
    result->op     = GGML_OP_ROPE;
    result->src[0] = a;  // 输入 Q/K
    result->src[1] = b;  // 位置索引
    result->src[2] = c;  // 可选的频率因子
    return result;
}
```

**② CPU 内核实现**（`ggml/src/ggml-cpu/ops.cpp:5761-5787`）

```cpp
template<typename T>
static void rotate_pairs(const int64_t n, const int64_t n_offset,
                         const float * cache, const T * src_data, T * dst_data,
                         const int scale = 2) {
    for (int64_t i0 = 0; i0 < n; i0 += 2) {
        const int64_t ic = i0 / scale;
        const float cos_theta = cache[i0 + 0];  // 预计算的 cos
        const float sin_theta = cache[i0 + 1];  // 预计算的 sin
        const float x0 = type_conversion_table<T>::to_f32(src[0]);
        const float x1 = type_conversion_table<T>::to_f32(src[n_offset]);
        // 旋转公式
        dst[0]        = type_conversion_table<T>::from_f32(x0 * cos_theta - x1 * sin_theta);
        dst[n_offset] = type_conversion_table<T>::from_f32(x0 * sin_theta + x1 * cos_theta);
    }
}
```

**③ 模型中的调用**（`src/models/qwen35moe.cpp:262-273`）

```cpp
// Qwen3.5 MoE 使用 multi-section RoPE（支持多模态的不同部分）
Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
    ext_factor, attn_factor, beta_fast, beta_slow);
Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
    ext_factor, attn_factor, beta_fast, beta_slow);
```

### 数据流转

```
输入:
  Qcur [128, 16, n_tokens]        (未编码位置的 Q)
  inp_pos [n_tokens]              (每个 token 的绝对位置，如 [0, 1, 2, ...])
    │
    ▼ RoPE 内核
    │
  1. 根据位置 pos 和维度 i 计算旋转角度 θ(pos, i)
     θ = pos / (base^(2i/head_dim))   // base 通常是 10000
    │
  2. 预计算 cos(θ) 和 sin(θ) 到 cache
    │
  3. 对每对维度 (i, i+1) 执行旋转:
     [x_i,   x_{i+1}]  ──► [x_i·cos - x_{i+1}·sin, x_i·sin + x_{i+1}·cos]
    │
输出:
  Qcur' [128, 16, n_tokens]       (编码位置后的 Q)
  Kcur' [128, 2, n_tokens]        (编码位置后的 K)
```

**关键洞察**：RoPE 的巧妙之处在于**旋转操作保持了内积的相对位置特性**：`<RoPE(q, m), RoPE(k, n)>` 只依赖于 `q, k` 和位置差 `(m-n)`，这正好符合 Attention 需要相对位置信息的需求。

---

## B.5 RMSNorm

### 通俗解释

神经网络层之间需要进行归一化，防止数值爆炸或消失。LayerNorm 的公式是：
```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

RMSNorm 是 LayerNorm 的简化版，**去掉 mean 和 beta**，只保留：
```
y = x / sqrt(mean(x²) + eps) * gamma
```

为什么可以去掉 mean？因为深度学习研究发现，对于 Transformer，**均方根（RMS）已经足够稳定训练**，而且少了一次统计量计算，更快。

### 代码实现

**① GGML 算子定义**（`ggml/include/ggml.h:506-507`）

```cpp
GGML_OP_RMS_NORM,
GGML_OP_RMS_NORM_BACK,  // 反向传播用
```

**② 公共 API**（`ggml/include/ggml.h:1370-1374`）

```cpp
GGML_API struct ggml_tensor * ggml_rms_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 eps);  // 如 1e-6
```

**③ 算子创建**（`ggml/src/ggml.c:3115-3143`）

```cpp
static struct ggml_tensor * ggml_rms_norm_impl(...) {
    result->op     = GGML_OP_RMS_NORM;
    result->src[0] = a;
    ggml_set_op_params_f32(result, 0, eps);
    return result;
}
```

**④ CPU 内核实现**（`ggml/src/ggml-cpu/ops.cpp:3723-3787`）

```cpp
template <ggml_rms_norm_fuse_op FUSE_OP>
static void ggml_compute_forward_rms_norm_f32(...) {
    for (int64_t i01 = 0; i01 < ne01; i01++) {  // 遍历每个样本
        // 计算均方根: sqrt(mean(x²))
        ggml_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {  // 遍历每个元素
            sum += (ggml_float)(x[i00] * x[i00]);
        }
        const float mean  = sum / ne00;
        const float scale = 1.0f / sqrtf(mean + eps);
        // 归一化并乘以 gamma（如果有的话）
        ggml_vec_scale_f32(ne00, y, scale);
        // 融合操作（如 RMS_NORM + MUL）
        FUSE_OP(...);
    }
}
```

**⑤ 模型中的调用**（`src/llama-graph.cpp:1033-1066`）

```cpp
switch (norm_type) {
    case LLM_NORM_RMS:
        cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);
        break;
    case LLM_NORM_LAYER:
        cur = ggml_norm(ctx0, cur, hparams.f_norm_rms_eps);
        break;
}
// 乘以归一化权重
if (layer.ffn_norm) {
    cur = ggml_mul(ctx0, cur, layer.ffn_norm);
}
```

### 数据流转

```
输入: cur [n_embd=2048, n_tokens]
    │
    ▼ RMSNorm
    │
  1. 对每个 token 的 2048 维向量，计算 sum(x_i²)
  2. rms = sqrt(sum / 2048 + eps)
  3. 每个元素除以 rms
    │
    ▼ 乘以 gamma 权重
    │
输出: cur [2048, n_tokens]  (归一化后的结果)
```

**关键洞察**：RMSNorm 比 LayerNorm 少了一次求 mean 和一次加 beta 的操作。在 40 层 Transformer 中，每层有 2-3 次归一化（输入归一化、Attention 后归一化、FFN 后归一化），累积起来节省了不少计算。

---

## B.6 SwiGLU / SiLU

### 通俗解释

FFN（前馈网络）的作用是给模型增加非线性表达能力。传统 FFN 是：
```
FFN(x) = max(0, x·W1 + b1)·W2 + b2   (ReLU)
```

SwiGLU 是目前 LLM 中最流行的 FFN 结构，它引入了“门控”机制：
```
SwiGLU(x) = (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down
```

分解来看：
1. **Gate 分支**：`x·W_gate` 经过 SiLU 激活，产生一个 0~1 的门控信号
2. **Up 分支**：`x·W_up` 产生原始特征
3. **门控乘法**：`SiLU(gate) * up`，门控信号决定哪些特征可以通过
4. **Down 投影**：`(gate * up)·W_down` 投影回原始维度

SiLU（Swish）激活函数的公式是 `x / (1 + e^(-x))`，它在 x>0 时近似线性，x<0 时平滑趋近于 0，比 ReLU 更平滑。

### 代码实现

**① GGML GLU 算子定义**（`ggml/include/ggml.h:584, 616-625`）

```cpp
GGML_OP_GLU,

enum ggml_glu_op {
    GGML_GLU_OP_REGLU,
    GGML_GLU_OP_GEGLU,
    GGML_GLU_OP_SWIGLU,      // 本模型使用
    GGML_GLU_OP_SWIGLU_OAI,
    // ...
};
```

**② 算子创建**（`ggml/src/ggml.c:2867-2892`）

```cpp
static struct ggml_tensor * ggml_glu_impl(...) {
    // 如果 b 为 NULL，将输入 a 沿第 0 维分成两半
    // 前半是 gate，后半是 up
    int64_t ne[GGML_MAX_DIMS] = { a->ne[0] / 2 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, GGML_MAX_DIMS,
                                                        b ? a->ne : ne, NULL, 0);
    ggml_set_op_params_i32(result, 0, (int32_t) op);   // SWIGLU
    ggml_set_op_params_i32(result, 1, (int32_t) swapped);
    result->op = GGML_OP_GLU;
    result->src[0] = a;  // gate+up 合并输入
    result->src[1] = b;  // 可选的外部 up 输入
    return result;
}
```

**③ SiLU 标量实现**（`ggml/src/ggml-cpu/vec.h:1064-1070`）

```cpp
inline static float ggml_silu_f32(float x) {
    return x / (1.0f + expf(-x));
}
```

**④ SwiGLU 向量实现**（`ggml/src/ggml-cpu/vec.cpp:433-469`）

```cpp
void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g) {
    // x = gate 输入（已投影）
    // g = up 输入（已投影）
    // y = SiLU(x) * g
    for (int i = 0; i < n; i++) {
        y[i] = ggml_silu_f32(x[i]) * g[i];
    }
}
```

**⑤ CPU 内核**（`ggml/src/ggml-cpu/ops.cpp:3131-3190`）

```cpp
static void ggml_compute_forward_swiglu_f32(...) {
    const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;  // 输出通道数
    if (!src1) {
        // 输入被分成两半
        src0_p += swapped ? nc : 0;   // gate 半区
        src1_p += swapped ? 0 : nc;   // up 半区
    }
    ggml_vec_swiglu_f32(nc, dst_p, src0_p, src1_p);
}
```

### 数据流转

```
情况 1: gate 和 up 是分开的张量（MoE 中常见）
  cur [2048] ──► gate_proj [2048, 512] ──► gate [512]
                                           │
                                           ▼ SiLU
                                           │
  cur [2048] ──► up_proj   [2048, 512] ──► up   [512]
                                           │
                                           ▼ 逐元素乘
                                           │
                                        gate * up [512]
                                           │
                                           ▼ down_proj [512, 2048]
                                           │
                                        out [2048]

情况 2: gate 和 up 合并为一个张量（节省内存）
  cur [2048] ──► gate_up [2048, 1024] ──► 分成两半
                                            ├─► gate [512] ──► SiLU ──┐
                                            └─► up   [512] ──────────┤
                                                                     ▼
                                                                  gate*up [512]
                                                                     │
                                                                     ▼ down_proj
                                                                     │
                                                                  out [2048]
```

**关键洞察**：SwiGLU 的“门控”机制让网络可以**选择性激活特征**。SiLU 激活在负值区域不完全截断（不像 ReLU 直接变 0），保留了一定的梯度流，这对深层网络的训练稳定性很重要。

---

## B.7 MoE Routing（混合专家路由）

### 通俗解释

传统 Transformer 的 FFN 层是“ dense（密集）”的——每个 token 都要经过同一个大网络。MoE 把 FFN 拆成很多小网络（experts），每个 token **只经过少数几个专家**。

这样做的好处：
- **总参数量大**（256 个专家 = 256 倍 FFN 容量），但**每次只激活几个**（本模型 n_expert_used 未知，通常是 4-8 个）
- 活跃参数量约 3B（而非总参数量 35B），推理速度和 3B 模型相近，但效果接近大模型

路由过程：
1. **Gate 网络**：输入 token 的 hidden state 经过一个线性层，输出 `n_expert` 个分数
2. **Top-K 选择**：用 ARGSORT 选出分数最高的 K 个专家
3. **加权聚合**：每个选中的专家处理输入，结果按 gate 分数加权求和
4. **Shared Expert**：所有 token 还会经过一个共享专家，保证基础能力

### 代码实现

**① Qwen3.5 MoE 张量创建**（`src/models/qwen35moe.cpp:88-98`）

```cpp
// 路由门控: [n_embd=2048, n_expert=256]
layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i),
                                   { n_embd, n_expert }, 0);
// 256 个专家各自的权重
layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                                    { n_ff_exp, n_embd, n_expert }, 0);
create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, 0);
// 共享专家
layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i),
                                         { n_embd }, 0);
layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i),
                                     { n_embd, n_ff_shexp }, 0);
```

**② 路由核心逻辑**（`src/llama-graph.cpp:1432-1460`）

```cpp
// 1. 计算路由分数: input [2048] × gate_inp [2048, 256] ──► scores [256]
ggml_tensor * selection_probs = build_lora_mm(layer.ffn_gate_inp, cur);
// 经过 softmax/sigmoid 归一化

// 2. DeepSeek V3 风格的分组路由（可选）
if (hparams.n_expert_groups > 1 && n_tokens > 0) {
    ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs,
        n_exp_per_group, hparams.n_expert_groups, n_tokens);
    ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2);
    ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used);
    selection_probs = ggml_get_rows(...);  // 只保留选中组的专家
}

// 3. 选出 top-k 专家
// selected_experts 形状: [n_expert_used, n_tokens]，存的是专家索引（int32）
ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
```

**③ MUL_MAT_ID 执行专家计算**（`src/llama-graph.cpp:1517-1540`）

```cpp
if (gate_up_exps) {
    // 合并 gate+up 路径: 一次 mul_mat_id 得到 gate 和 up
    // gate_up 形状: [n_ff*2, n_expert_used, n_tokens]
    ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts);
    // 切成 gate 和 up 两半
    ggml_tensor * gate = ggml_view_3d(..., 0);
    ggml_tensor * up   = ggml_view_3d(..., n_ff);
    cur = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
} else {
    // 分开路径
    up = build_lora_mm_id(up_exps, cur, selected_experts);
    cur = build_lora_mm_id(gate_exps, cur, selected_experts);
    cur = ggml_mul(ctx0, ggml_silu(ctx0, cur), up);
}
// Down 投影
experts = build_lora_mm_id(down_exps, cur, selected_experts);  // [n_embd, n_expert_used, n_tokens]
```

**④ GGML MUL_MAT_ID 定义**（`ggml/src/ggml.c:3277-3315`）

```cpp
// as  -> [cols, rows, n_expert]          (所有专家的合并权重)
// b   -> [cols, n_expert_used, n_tokens] (输入)
// ids -> [n_expert_used, n_tokens]       (i32，选中的专家索引)
// c   -> [rows, n_expert_used, n_tokens] (输出)
struct ggml_tensor * ggml_mul_mat_id(...) {
    // ... 验证形状 ...
    result->op = GGML_OP_MUL_MAT_ID;
    result->src[0] = as;  // 专家权重
    result->src[1] = b;   // 输入
    result->src[2] = ids; // 路由索引
    return result;
}
```

**⑤ Shared Expert + Routed Expert 融合**（`src/models/qwen35moe.cpp:494-527`）

```cpp
// Routed experts 输出
ggml_tensor * moe_out = build_moe_ffn(cur,
    model.layers[il].ffn_gate_inp,
    model.layers[il].ffn_up_exps,
    model.layers[il].ffn_gate_exps,
    model.layers[il].ffn_down_exps,
    nullptr, n_expert, n_expert_used,
    LLM_FFN_SILU, true,
    hparams.expert_weights_scale,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il, ...);

// Shared expert（所有 token 都走）
if (model.layers[il].ffn_up_shexp != nullptr) {
    ggml_tensor * ffn_shexp = build_ffn(cur,
        model.layers[il].ffn_up_shexp, NULL, ...,
        model.layers[il].ffn_gate_shexp, NULL, ...,
        model.layers[il].ffn_down_shexp, NULL, ...,
        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    // 用 sigmoid gate 控制共享专家的强度
    ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
    shared_gate = ggml_sigmoid(ctx0, shared_gate);  // 0~1 之间的标量
    ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
    // 相加: routed + shared
    cur = ggml_add(ctx0, moe_out, ffn_shexp);
}
```

### 数据流转

```
输入: cur [2048, n_tokens]
    │
    ▼ 1. Gate 网络
    │
  scores [256, n_tokens]    (每个 token 对 256 个专家的分数)
    │
    ▼ 2. ARGSORT (top-k)
    │
  selected_experts [K, n_tokens]   (K 个专家索引，如 [5, 12, 33, 89])
  expert_weights   [K, n_tokens]   (对应的 softmax 权重)
    │
    ▼ 3. MUL_MAT_ID（对每个 token 分别做）
    │
  token_0: 用专家 5 计算  ──┐
  token_1: 用专家 12 计算 ──┤
  token_2: 用专家 33 计算 ──┼──► 按权重加权求和
  ...                       ──┘
    │
  moe_out [2048, n_tokens]
    │
    ▼ 4. Shared Expert
    │
  ffn_shexp [2048, n_tokens]
    │
    ▼ 5. 相加
    │
  cur [2048, n_tokens]
```

**关键洞察**：`MUL_MAT_ID` 是 MoE 能高效工作的核心——它不是把 256 个专家全部算一遍再 mask，而是**只加载和计算选中的 K 个专家**。这要求后端支持**按索引的条件矩阵乘法**。

---

## B.8 SSM（State Space Model）

### 通俗解释

标准 Attention 的复杂度是 O(N²)（序列长度的平方），当序列很长时（如 256K）计算量巨大。

SSM（State Space Model）提供了一种 O(N) 复杂度的替代方案。它的核心思想是：
1. **用一个隐藏状态（state）来压缩历史信息**，而不是存储所有 token 的 K/V
2. 新 token 到来时，**更新状态**而不是和所有历史做 Attention
3. 输出由当前输入和状态共同决定

SSM 家族包括 Mamba、RWKV、RetNet 等。本模型使用的 `SSM_CONV` + `GATED_DELTA_NET` 是其中一种变体。

### 代码实现

**① SSM_CONV 算子定义**（`ggml/src/ggml.c:5481-5505`）

```cpp
struct ggml_tensor * ggml_ssm_conv(struct ggml_context * ctx,
                                    struct ggml_tensor * sx,  // 输入 [seq_len, d_inner, n_seqs]
                                    struct ggml_tensor * c) { // 卷积核 [d_conv, d_inner]
    const int64_t d_conv  = c->ne[0];   // 卷积核长度（如 4）
    const int64_t d_inner = c->ne[1];   // 内部维度
    const int64_t n_t     = sx->ne[0] - d_conv + 1;  // 输出序列长度
    const int64_t n_s     = sx->ne[2];  // 序列数
    struct ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);
    result->op = GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;
    return result;
}
```

**② GATED_DELTA_NET 算子定义**（`ggml/src/ggml.c:6182-6229`）

```cpp
struct ggml_tensor * ggml_gated_delta_net(struct ggml_context * ctx,
    struct ggml_tensor * q,      // 查询 [head_dim, n_head, n_tokens]
    struct ggml_tensor * k,      // 键
    struct ggml_tensor * v,      // 值
    struct ggml_tensor * g,      // 门控
    struct ggml_tensor * beta,   // 衰减因子
    struct ggml_tensor * state) { // 历史状态
    // 输出包含两部分: 当前输出 + 更新后的状态
    const int64_t ne[4] = { S_v * H, n_tokens * n_seqs + S_v * n_seqs, 1, 1 };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    result->op = GGML_OP_GATED_DELTA_NET;
    result->src[0] = q; result->src[1] = k; result->src[2] = v;
    result->src[3] = g; result->src[4] = beta; result->src[5] = state;
    return result;
}
```

**③ Qwen3.5 中的 SSM 使用**（`src/models/qwen35moe.cpp:384-388`）

```cpp
// SSM 卷积提取局部特征
ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
// SiLU 激活
ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
```

**④ Delta Net Base 中的 Gated Delta Net**（`src/models/delta-net-base.cpp:400-420`）

```cpp
ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s);
// 结果分为两部分: 输出和状态
if (n_tokens == 1) {
    cb(result, LLAMA_TENSOR_NAME_FGDN_AR, il);  // 自回归模式
} else {
    cb(result, LLAMA_TENSOR_NAME_FGDN_CH, il);  // 块模式
}
// 切片提取输出和状态
ggml_tensor * output = ggml_view_4d(ctx0, result, S_v, H_v, n_tokens, n_seqs, ...);
ggml_tensor * new_state = ggml_view_4d(ctx0, result, S_v, S_v, H_v, n_seqs, ...);
```

### 数据流转

```
标准 Attention (O(N²)):
  每个新 token 要和所有历史 token 做 dot product
  内存: 存储所有 K/V [n_layer, n_head_kv, head_dim, n_ctx]

SSM (O(N)):
  输入 x_t ──► SSM_CONV ──► 局部特征提取
                │
                ▼
           GATED_DELTA_NET ──► 用门控更新状态 state_t
                │
                ├─► 输出 y_t
                └─► 新状态 state_{t+1} (传给下一个 token)

  内存: 只存储状态 [state_dim]，不存历史 K/V
  计算: 每个 token 只和固定大小的状态交互
```

**关键洞察**：SSM 和 Attention 不是互斥的——本模型是**混合架构**，部分层用 Attention，部分层用 SSM。Attention 擅长捕获任意距离的长程依赖，SSM 擅长线性复杂度的长序列处理，两者互补。

---

## B.9 Quantization（q4_K / q5_K / q8_0）

### 通俗解释

大模型的权重通常用 f32（32-bit 浮点）或 f16（16-bit 浮点）存储。一个 35B 参数的模型用 f16 需要 **70GB 内存**，这远超大多数设备的容量。

**量化（Quantization）** 用更少的 bit 来表示权重：
- **q8_0**：8-bit，每 32 个权重一组，精度损失很小
- **q5_K**：5-bit，每 256 个权重一个 super-block，中等压缩
- **q4_K**：4-bit，每 256 个权重一个 super-block，压缩率最高

量化不是简单地把每个权重单独截断，而是**分组统计分布**，用 scale 和 min 来还原：
```
原始值 = scale × quantized_value + min
```

### 代码实现

**① 常量定义**（`ggml/src/ggml-common.h:87-90`）

```cpp
#define QK_K 256        // super-block 大小（K-quant 的核心单位）
#define K_SCALE_SIZE 12 // scale/min 的压缩存储大小
```

**② q8_0 块结构**（`ggml/src/ggml-common.h:241-246`）

```cpp
#define QK8_0 32        // 每 32 个权重一组

typedef struct {
    ggml_half d;       // delta（scale），2 bytes
    int8_t  qs[QK8_0]; // 32 个 int8 量化值，32 bytes
} block_q8_0;
// 总大小: 2 + 32 = 34 bytes
// 压缩率: 34 / (32 × 4) = 26.5%（相比 f32）
```

**③ q4_K 块结构**（`ggml/src/ggml-common.h:317-328`）

```cpp
// 4-bit 量化，每 256 个权重一个 super-block
// 内部细分为 8 个 block（每 block 32 个权重）
// 权重还原公式: x = d × q + dmin

typedef struct {
    ggml_half d;                 // super-block 的 scale，2 bytes
    ggml_half dmin;              // super-block 的 min，2 bytes
    uint8_t scales[K_SCALE_SIZE]; // 8 个 block 的 scale/min，用 6-bit 压缩存储，12 bytes
    uint8_t qs[QK_K/2];          // 256 个 4-bit 量化值，128 bytes
} block_q4_K;
// 总大小: 2 + 2 + 12 + 128 = 144 bytes
// 压缩率: 144 / (256 × 4) = 14%（相比 f32），等效 4.5 bits
```

**④ q5_K 块结构**（`ggml/src/ggml-common.h:330-346`）

```cpp
// 5-bit 量化，结构与 q4_K 类似

typedef struct {
    ggml_half d;                 // super-block scale，2 bytes
    ggml_half dmin;              // super-block min，2 bytes
    uint8_t scales[K_SCALE_SIZE]; // block scale/min，12 bytes
    uint8_t qh[QK_K/8];          // 高 1 bit（256 个权重各 1 bit），32 bytes
    uint8_t qs[QK_K/2];          // 低 4 bit，128 bytes
} block_q5_K;
// 总大小: 2 + 2 + 12 + 32 + 128 = 176 bytes
// 压缩率: 176 / (256 × 4) = 17.2%（相比 f32），等效 5.5 bits
```

### 数据流转

```
原始权重 (f32):
  [w0, w1, w2, ..., w255]  共 256 × 4 = 1024 bytes

q4_K 编码过程:
  1. 找到 256 个权重的 min 和 max
  2. 计算 scale = (max - min) / 15   (因为 4-bit 只能表示 0~15)
  3. 每个权重: q_i = round((w_i - min) / scale)，存 4-bit
  4. scale 和 min 进一步用 6-bit 压缩存储到 scales 数组
  5. 输出: block_q4_K (144 bytes)

q4_K 解码过程:
  1. 从 d 和 dmin 还原 super-block 的 scale 和 min
  2. 从 scales 数组还原 8 个 block 各自的微调 scale/min
  3. 每个量化值 q_i 还原: w_i = d × q_i + dmin
  4. 得到近似的原始权重

推理时的矩阵乘法:
  输入 x (f16/f32) × 权重 W (q4_K)
    │
    ▼ 在 kernel 内部实时反量化
    │
  x × W_dequantized ──► 输出
```

**关键洞察**：量化权重在推理时**不是一次性全部反量化回 f32**，而是在矩阵乘法的内核中**逐 block 实时反量化**。这样只需要少量额外寄存器/共享内存，不会增加全局内存占用。

---

## B.10 Sampler Chain（采样链）

### 通俗解释

模型输出的是每个 token 的“分数”（logits），但直接选最高分太死板，完全随机又太混乱。Sampler Chain 是一系列**过滤和变换操作**，逐步缩小候选范围，最终选一个 token。

本模型的采样链（10 步）：
```
penalties → dry → top-n-sigma → top-k → typical → top-p → min-p → xtc → temp-ext → dist
```

每一步的作用：
1. **penalties**：惩罚重复出现的 token，避免模型说车轱辘话
2. **dry**：另一种重复惩罚策略
3. **top-n-sigma**：基于标准差的过滤
4. **top-k**：只保留概率最高的 40 个 token
5. **typical**：Typical Sampling，保留“典型”的 token
6. **top-p**：只保留累积概率达到 95% 的 token
7. **min-p**：只保留概率 >= max_prob × 5% 的 token
8. **xtc**：XTC 采样
9. **temp-ext**：温度缩放，控制随机性
10. **dist**：从最终的概率分布中随机采样一个 token

### 代码实现

**① 核心数据结构**（`include/llama.h:202-215`）

```cpp
typedef struct llama_token_data {
    llama_token id;   // token ID（如 90700）
    float logit;      // 原始分数（如 22.43）
    float p;          // 概率（softmax 后，如 0.6866）
} llama_token_data;

typedef struct llama_token_data_array {
    llama_token_data * data;   // token 数组
    size_t size;               // 当前数组大小（从 248320 逐步缩小）
    int64_t selected;          // 最终选中的数组下标
    bool sorted;               // 是否已排序
} llama_token_data_array;
```

**② Sampler Chain 结构**（`src/llama-sampler.h:12-34`）

```cpp
struct llama_sampler_chain {
    struct info {
        bool is_backend;       // 是否在后端（GPU）执行
        llama_sampler * ptr;   // 具体的 sampler
    };
    std::vector<info> samplers;  // 10 个 sampler 的数组
    std::vector<llama_token_data> cur;  // 预分配的缓冲区
};
```

**③ 入口函数：logits → token**（`src/llama-sampler.cpp:837-915`）

```cpp
llama_token llama_sampler_sample(struct llama_sampler * smpl,
                                  struct llama_context * ctx, int32_t idx) {
    const int n_vocab = llama_vocab_n_tokens(vocab);  // 248320

    // 使用预分配缓冲区避免重复 malloc
    std::vector<llama_token_data> * cur_ptr;
    if (smpl->iface == &llama_sampler_chain_i) {
        auto * chain = (llama_sampler_chain *) smpl->ctx;
        cur_ptr = &chain->cur;
    }
    auto & cur = *cur_ptr;

    // 从模型获取 logits，构建 token 数组
    const auto * logits = llama_get_logits_ith(ctx, idx);
    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    // 包装成 cur_p 并执行采样链
    llama_token_data_array cur_p = {
        .data = cur.data(), .size = cur.size(),
        .selected = -1, .sorted = false,
    };
    llama_sampler_apply(smpl, &cur_p);
    auto token = cur_p.data[cur_p.selected].id;
    return token;
}
```

**④ Chain Apply：逐个执行 sampler**（`src/llama-sampler.cpp:663-693`）

```cpp
static void llama_sampler_chain_apply(struct llama_sampler * smpl,
                                       llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;
    for (auto & smpl : chain->samplers) {
        if (smpl.ptr->iface->apply == nullptr) continue;
        llama_sampler_apply(smpl.ptr, cur_p);  // 递归调用每个 sampler
    }
}
```

**⑤ Temperature 实现**（`src/llama-sampler.cpp:265-294`）

```cpp
static void llama_sampler_temp_impl(llama_token_data_array * cur_p, float temp) {
    // temp=0.8: logit 除以 0.8 = 乘以 1.25，拉大差距
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;
    }
}
```

**⑥ Softmax 实现**（`src/llama-sampler.cpp:296-330`）

```cpp
static void llama_sampler_softmax_impl(llama_token_data_array * cur_p, bool do_sort) {
    if (do_sort && !cur_p->sorted) {
        llama_token_data_array_partial_sort_inplace(cur_p, cur_p->size);
    }
    float max_l = cur_p->data[0].logit;  // 找到最大 logit（数值稳定性）
    float cum_sum = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = expf(cur_p->data[i].logit - max_l);  // exp(logit - max)
        cur_p->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= cum_sum;  // 归一化为概率
    }
}
```

**⑦ Top-K 实现**（`src/llama-sampler.cpp:317` 附近）

```cpp
// 保留概率最高的 k 个 token，其余丢弃
// 实现：用 partial_sort 找到 top-k，然后 resize 数组
```

**⑧ Top-P 实现**（`src/llama-sampler.cpp:1390` 附近）

```cpp
// 按概率从高到低排序，累积概率直到 >= p（0.95）
// 保留这些 token，其余的丢弃
```

**⑨ Min-P 实现**（`src/llama-sampler.cpp:1551` 附近）

```cpp
// 只保留概率 >= max_prob * p 的 token
// 如 max_prob=0.6866, p=0.05，则阈值 = 0.0343
// 只保留概率 >= 3.43% 的 token
```

### 数据流转

```
输入: logits [248320] float
    │
    ▼ penalties / dry
    │
  logits' [248320]  (重复 token 分数降低)
    │
    ▼ top-k (k=40)
    │
  candidates [40]  (只保留 top 40)
    │
    ▼ softmax
    │
  candidates [40]  (每个有 .p 概率值)
    │
    ▼ top-p (p=0.95)
    │
  candidates [3]   (累积概率达 95%)
    │
    ▼ min-p (p=0.05)
    │
  candidates [2]   (概率 >= max*0.05)
    │
    ▼ temp (0.8)
    │
  candidates [2]   (logit 缩放)
    │
    ▼ dist (随机采样)
    │
  selected = 1  →  token_id = candidates[1].id
```

**关键洞察**：采样链的精髓在于**逐步过滤**。不是所有 token 都需要 softmax——先在 248320 个中快速筛选到 40 个，再对这 40 个做 softmax，计算量降低了 6200 倍。

---

## B.11 KV Cache 内部机制

### 通俗解释

生成第 N 个 token 时，模型需要“看”到前面所有 N-1 个 token。标准 Attention 需要计算 `Q_N × [K_1, K_2, ..., K_N]`。

如果每次重新计算所有 K/V，生成速度会慢到无法使用。KV Cache 的做法是：**第一次计算后，把 K/V 向量存起来，后续只计算新 token 的 K/V，然后从 cache 中读取旧的**。

### 代码实现

**① KV Cache 类定义**（`src/llama-kv-cache.h:20-307`）

```cpp
class llama_kv_cache : public llama_memory_i {
public:
    struct slot_info {
        uint32_t s0, s1;                    // stream 范围
        std::vector<llama_seq_id> strm;     // 每个 stream 的序列 ID
        std::vector<idx_vec_t>    idxs;     // 每个 stream 的 cell 索引
    };

private:
    std::vector<uint32_t> v_heads;           // 每个 stream 的写入头位置
    std::vector<llama_kv_cells> v_cells;     // 占用标记 [stream][cell]
    std::vector<uint32_t> seq_to_stream;     // seq_id → stream_id 映射
    std::vector<kv_layer> layers;            // 每层的 K/V 张量
};
```

**② 初始化**（`src/llama-kv-cache.cpp:134-150`）

```cpp
v_heads.resize(n_stream);
for (uint32_t s = 0; s < n_stream; ++s) {
    v_heads[s] = 0;  // 初始写入位置为 0
}

v_cells.resize(n_stream);
for (uint32_t s = 0; s < n_stream; ++s) {
    v_cells[s].resize(kv_size);  // 每个 stream 262144 个 cell
}
```

**③ 创建 K/V 张量**（`src/llama-kv-cache.cpp:221-241`）

```cpp
// K 张量: [n_embd_k_gqa=512, kv_size=262144, n_stream=1]
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
// V 张量: [n_embd_v_gqa=512, kv_size=262144, n_stream=1]
ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);

// 为每个 stream 创建 view
for (uint32_t s = 0; s < n_stream; ++s) {
    k_stream.push_back(ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size,
                                    k->nb[1], s * k->nb[2]));
    v_stream.push_back(ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size,
                                    v->nb[1], s * v->nb[2]));
}
```

**④ get_k：读取 K cache 供 Attention 使用**（`src/llama-kv-cache.cpp:1215-1235`）

```cpp
ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il,
                                     uint32_t n_kv, const slot_info & sinfo) const {
    auto * k = layers[ikv].k;
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    // 创建 4D view: [head_dim, n_head_kv, n_kv, ns]
    return ggml_view_4d(ctx, k,
        hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
        ggml_row_size(k->type, hparams.n_embd_head_k(il)),
        ggml_row_size(k->type, n_embd_k_gqa),
        ggml_row_size(k->type, n_embd_k_gqa * kv_size),
        ggml_row_size(k->type, n_embd_k_gqa * kv_size) * sinfo.s0);
}
```

**⑤ cpy_k：将新 K 写入 cache**（`src/llama-kv-cache.cpp:1273-1311`）

```cpp
ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur,
                                     ggml_tensor * k_idxs, int32_t il,
                                     const slot_info & sinfo) const {
    // k_cur: [n_embd_gqa=512, n_tokens]
    // k_idxs: [n_tokens] int32，写入位置的索引
    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);
    // 使用 ggml_set_rows 按索引写入
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}
```

### 数据流转

```
Prompt 阶段（并行处理）:
  Token_1 ──► K_1, V_1 ──► 写入 cache[0]
  Token_2 ──► K_2, V_2 ──► 写入 cache[1]
  Token_3 ──► K_3, V_3 ──► 写入 cache[2]
  ...
  cache 状态: [K_1, K_2, K_3, ..., K_N] [V_1, V_2, V_3, ..., V_N]

Generation 阶段（逐个生成）:
  Token_{N+1} ──► K_{N+1}, V_{N+1} ──► 写入 cache[N]
    │
    ▼ Attention
    │
  Q_{N+1} × [K_1..K_{N+1}] ──► 预测 Token_{N+2}
    │
    ▼ 写入
    │
  cache 状态: [K_1, ..., K_{N+1}] + [V_1, ..., V_{N+1}]
```

**关键洞察**：KV Cache 是**自回归生成能实时进行的前提**。没有 KV Cache，生成每个新 token 都要重新计算所有历史 token 的 Attention，复杂度是 O(N³)；有了 KV Cache，复杂度降到 O(N²)（只对新的 Q 和全部 K/V 做 Attention）。Flash Attention 进一步降到 O(N)。

---

## B.12 BPE Tokenization

### 通俗解释

模型不能直接读汉字或英文单词，它需要数字。Tokenization 就是把文本变成数字序列的过程。

BPE（Byte-Pair Encoding）的工作方式：
1. 从字符级别开始（每个字符是一个 token）
2. 统计文本中哪些字符对最常一起出现
3. 把最常见的字符对合并成一个新的 token
4. 重复步骤 2-3，直到词表达到目标大小（本模型 248320）

例如：
- 初始：`['H', 'e', 'l', 'l', 'o']`
- 发现 `'l', 'l'` 经常一起出现 → 合并成 `'ll'`
- 发现 `'H', 'e'` 经常一起出现 → 合并成 `'He'`
- 最终：`['He', 'l', 'l', 'o']` 或 `['H', 'e', 'll', 'o']`

### 代码实现

**① Token 属性定义**（`include/llama.h`）

```cpp
enum llama_token_attr {
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,   // 未知 token
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,   // 普通文本 token
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,   // 控制 token（如 BOS/EOS）
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,   // 字节 token（用于编码未知字符）
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,   // 去除前导空格
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
};
```

**② token_to_piece 核心逻辑**（`src/llama-vocab.cpp:3262` 附近）

```cpp
int llama_vocab::impl::token_to_piece(...) {
    // 1. 查缓存
    const auto & token_data = cache_token_to_piece[token];

    // 2. 根据属性决定处理方式
    switch (token_data.attr) {
        case LLAMA_TOKEN_ATTR_NORMAL:
            // 普通 token：直接输出对应文本
            result = token_data.text;
            if (lstrip) result = ltrim(result);  // 去除前导空格
            break;
        case LLAMA_TOKEN_ATTR_CONTROL:
            // 控制 token（如 <s>, </s>）：通常输出空字符串
            result = "";
            break;
        case LLAMA_TOKEN_ATTR_BYTE:
            // 字节 token：解码为原始字节
            result = token_to_byte(token);
            break;
        case LLAMA_TOKEN_ATTR_UNKNOWN:
            result = "\xEF\xBF\xBD";  // UTF-8 替换字符
            break;
    }

    // 3. 写入输出缓冲区
    memcpy(buf, result.data(), result.size());
    return result.size();
}
```

**③ 本模型 BPE 示例**（从 trace 日志提取）

```
Token 0:  attr=NORMAL  text='!'  bytes=[0x21]
Token 1:  attr=NORMAL  text='"'  bytes=[0x22]
...
Token 248044:  attr=CONTROL  text='<|im_start|>'  (BOS)
Token 248046:  attr=CONTROL  text='<|im_end|>'    (EOS)
```

### 数据流转

```
文本: "Hello, world!"
    │
    ▼ BPE Tokenize
    │
  1. 从字符开始: ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']
  2. 按词表查找最长匹配:
     'Hello' -> token 15496
     ',' -> token 11
     ' world' -> token 995
     '!' -> token 0
    │
输出: [15496, 11, 995, 0]

反向（Detokenize）:
  [15496, 11, 995, 0]
    │
    ▼ 逐个查 token_to_piece
    │
  token 15496 -> "Hello"
  token 11    -> ","
  token 995   -> " world"
  token 0     -> "!"
    │
输出: "Hello, world!"
```

**关键洞察**：BPE 的词表是**训练出来的**，不是人工定义的。高频词（如 "the", "ing"）会被合并成单个 token，低频词或专有名词会被拆成子词或字符。这保证了任何文本都能被编码，同时常用词只用 1 个 token 表示，效率很高。

---

*附录补充时间：2026-05-08*
*基于 Agent 搜索结果和源代码阅读*
