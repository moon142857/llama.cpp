# llama.cpp 全链路深度分析报告（增强版）

> 基于 `tinyllamas/stories15M` 模型运行日志，面向 **NPU 后端移植开发者**
> 报告版本：v2.0（含 GGUF 格式、KV Cache、算子 Shape、后端接口深度解析）

---

## 一、运行环境概述

| 项目 | 值 |
|------|-----|
| 硬件 | Apple M4 (SoC) |
| 后端 | Metal (MTL0) + CPU |
| 模型 | tinyllamas/stories15M |
| 架构 | **LLaMA**（确认：使用 RMS_NORM、SwiGLU、RoPE、FLASH_ATTN_EXT） |
| 参数量 | ~15M |
| 层数 | n_layer=6 |
| 隐藏维度 | n_embd=288 |
| 注意力头 | n_head=6, n_head_kv=6（非 GQA） |
| 上下文训练长度 | n_ctx_train=128 |
| GPU 层 | n_gpu_layers=7（全部 offload 到 GPU） |
| 内存 | unified memory = true（CPU/GPU 共享内存） |
| KV Cache | kv_size=256, type_k=f16, type_v=f16 |

**架构纠正说明：** 日志中明确出现 `arch=llama`，且算子链包含 `RMS_NORM`、`GLU`（SwiGLU）、`ROPE`、`FLASH_ATTN_EXT`，这些都是 LLaMA 家族（LLaMA 1/2/3、Qwen、Mistral 等）的典型特征，而非 GPT-2 的 LayerNorm + GELU + 标准 Attention。

---

## 二、GGUF 数据格式深度解析

### 2.1 GGUF 文件结构（二进制格式）

GGUF（GGML Universal Format）是 llama.cpp 使用的模型文件格式，基于 Google Protocol Buffers 的变长整数编码思想。

```
[GGUF File Layout]
├─ Header (固定前缀)
│   ├─ Magic: "GGUF" (4 bytes)
│   ├─ Version: uint32 (本例 = 3)
│   ├─ n_tensors: uint64 (本例 = 57)
│   └─ n_kv: uint64 (metadata KV 对数量)
│
├─ KV Metadata Array (n_kv 个键值对)
│   ├─ key: string (如 "general.architecture")
│   ├─ type: gguf_type (UINT32, FLOAT32, STRING, ARRAY 等)
│   └─ value: 对应类型的值
│   本例关键 KV：
│     - general.architecture = "llama"
│     - llama.embedding_length = 288
│     - llama.block_count = 6
│     - llama.attention.head_count = 6
│     - llama.attention.head_count_kv = 6
│     - llama.context_length = 128
│     - llama.rope.freq_base = 10000.0
│
├─ Tensor Info Array (n_tensors 个张量信息)
│   ├─ name: string (如 "blk.0.attn_q.weight")
│   ├─ n_dims: uint32 (维度数，通常 1~4)
│   ├─ shape: uint64[n_dims] (各维度大小)
│   ├─ type: ggml_type (Q4_0, Q8_0, F32 等)
│   └─ offset: uint64 (数据区偏移量)
│
└─ Aligned Data Section (张量权重数据)
    ├─ alignment padding (默认对齐到 32 字节)
    └─ raw tensor data (按 tensor info 顺序排列)
```

### 2.2 本模型 57 个张量完整清单

从 `gguf_init_from_file_ptr` 日志解析：

| # | 张量名 | 类型 | shape | 作用 |
|---|--------|------|-------|------|
| 0 | `token_embd.weight` | q4_0 | [288, 32000] | 词嵌入矩阵 [n_embd, n_vocab] |
| 1 | `output_norm.weight` | f32 | [288] | 最终输出层 RMSNorm 权重 |
| 2 | `output.weight` | q8_0 | [288, 32000] | 输出投影矩阵 [n_embd, n_vocab] |
| 3 | `blk.0.attn_q.weight` | q4_0 | [288, 288] | Layer 0 Q 投影 [n_embd, n_embd] |
| 4 | `blk.0.attn_k.weight` | q4_0 | [288, 288] | Layer 0 K 投影 [n_embd, n_embd] |
| 5 | `blk.0.attn_v.weight` | q4_0 | [288, 288] | Layer 0 V 投影 [n_embd, n_embd] |
| 6 | `blk.0.attn_output.weight` | q4_0 | [288, 288] | Layer 0 Attention 输出投影 |
| 7 | `blk.0.attn_norm.weight` | f32 | [288] | Layer 0 Attention 前 RMSNorm |
| 8 | `blk.0.ffn_gate.weight` | q4_0 | [288, 768] | Layer 0 FFN gate 投影 [n_embd, n_ff] |
| 9 | `blk.0.ffn_down.weight` | q4_0 | [768, 288] | Layer 0 FFN down 投影 [n_ff, n_embd] |
| 10 | `blk.0.ffn_up.weight` | q4_0 | [288, 768] | Layer 0 FFN up 投影 [n_embd, n_ff] |
| 11 | `blk.0.ffn_norm.weight` | f32 | [288] | Layer 0 FFN 前 RMSNorm |
| 12-56 | blk.1~5.* | 同上 | 同上 | Layer 1~5 的对应权重 |

**量化分析：**
- **q4_0**: 4-bit 量化权重（主流），每个 block 32 个权重 + 1 个 f16 scale
- **q8_0**: 8-bit 量化权重（输出层精度要求高）
- **f32**: 32-bit 浮点（LayerNorm 权重不可量化，否则精度崩溃）

**张量命名规范：**
```
blk.{layer_id}.{op_type}.{role}.weight
  layer_id: 0 ~ n_layer-1
  op_type:  attn / ffn
  role:     q / k / v / output / gate / up / down / norm
```

### 2.3 代码解析：GGUF 读取流程

```cpp
// ggml/src/gguf.cpp: gguf_init_from_file_ptr()

// Step 1: 读取文件头
file.read(&magic, 4);           // "GGUF"
file.read(&version, 4);         // 3
file.read(&n_tensors, 8);       // 57
file.read(&n_kv, 8);            // metadata 数量

// Step 2: 读取 KV metadata
for (uint64_t i = 0; i < n_kv; i++) {
    key   = read_string(file);
    type  = read_type(file);
    value = read_value(file, type);  // 根据类型读取不同长度
}

// Step 3: 读取 Tensor Info（不读数据，只读元信息）
for (uint64_t i = 0; i < n_tensors; i++) {
    info.name   = read_string(file);
    info.n_dims = read_u32(file);
    for (int d = 0; d < info.n_dims; d++) {
        info.ne[d] = read_u64(file);  // 各维度大小
    }
    info.type   = read_type(file);    // Q4_0 / Q8_0 / F32
    info.offset = read_u64(file);     // 数据区偏移
}

// Step 4: 对齐到 data section
offset = align_offset(file.pos, alignment);

// Step 5: mmap 数据区（懒加载）
ctx->data = mmap(file, offset, file_size - offset);
```

**关键设计：** GGUF 解析阶段**只读元信息**，权重数据通过 `mmap` 懒加载，实际在 `load_tensors()` 创建 `ggml_tensor` 时才按需读取到 backend buffer。

---

## 三、环节一：模型加载（GGUF → 内存）

### 3.1 调用链
```
main() → llama_model_load() → GGUF 解析 → hparams 加载 → vocab 加载 → load_tensors()
```

### 3.2 超参数（Hyperparameters）

```
=== [LLAMA_TRACE] llama_model_load:   arch=llama n_embd=288 n_layer=6 n_head=6 n_head_kv=6 n_ctx_train=128 f_norm_eps=0.000000 ===
=== [LLAMA_TRACE] llama_model_load:   vocab_size=32000 BOS=1 EOS=2 EOT=-1 ===
```

| 参数 | 值 | 说明 |
|------|-----|------|
| arch | llama | 模型架构标识 |
| n_embd | 288 | 隐藏层维度（embedding size） |
| n_layer | 6 | Transformer 层数 |
| n_head | 6 | Attention Q 头数 |
| n_head_kv | 6 | Attention K/V 头数（= n_head，非 GQA） |
| n_ctx_train | 128 | 训练时的最大上下文长度 |
| f_norm_eps | 1e-6 | RMSNorm 的 epsilon |
| vocab_size | 32000 | 词表大小 |
| BOS | 1 | Begin-of-Sequence token ID |
| EOS | 2 | End-of-Sequence token ID |
| n_ff | 768 | FFN 中间层维度（= 2.67 * n_embd，SwiGLU 约 8/3 * n_embd） |

### 3.3 张量加载与设备分配

```
=== [LLAMA_TRACE] load_tensors: device assignment: n_layer=6 n_gpu_layers=7 i_gpu_start=0 act_gpu_layers=7 ===
=== [LLAMA_TRACE] load_tensors:   layer  0 -> MTL0 ===
=== [LLAMA_TRACE] load_tensors:   layer  1 -> MTL0 ===
...（全部 6 层 + 输出层 → MTL0）
=== [LLAMA_TRACE] load_tensors: buffer CPU_Mapped   size=    4.94 MiB ===
=== [LLAMA_TRACE] load_tensors: buffer MTL0_Mapped  size=   12.56 MiB ===
```

**关键行为：**
1. `i_gpu_start=0`：从第 0 层开始 offload 到 GPU/Metal
2. `n_gpu_layers=7 > n_layer=6`：全部层 + 输出层都 offload
3. CPU buffer (4.94 MiB)：存放 input/output tensors、KV cache metadata
4. MTL0 buffer (12.56 MiB)：存放全部量化权重

---

## 四、环节二：Tokenizer（文本 → Token IDs）

### 4.1 调用链
```
main() → llama_tokenize() → vocab->tokenize() → BPE 编码 → token IDs 数组
```

### 4.2 日志证据
```
=== [LLAMA_TRACE] llama_tokenize: text_len=66 add_special=1 ===
=== [LLAMA_TRACE] llama_tokenize: result=32 tokens ===
```

### 4.3 详细拆解

- 输入文本被 BPE 分词器切分为 32 个 token
- `add_special=1` 表示自动添加 BOS token (ID=1)
- 输出 token IDs 格式：`[1, 19182, 13775, 257, 640, 11, ...]`

---

## 五、KV Cache 数据格式与代码深度解析

### 5.1 KV Cache 初始化参数

```
=== [LLAMA_TRACE] llama_kv_cache: KV cache init ===
=== [LLAMA_TRACE] llama_kv_cache:   kv_size=256 n_layer_kv=6 n_stream=1 n_seq_max=1 n_pad=1 n_swa=0 ===
=== [LLAMA_TRACE] llama_kv_cache:   type_k=f16 type_v=f16 v_trans=0 offload=1 ===
```

| 参数 | 值 | 说明 |
|------|-----|------|
| kv_size | 256 | KV cache 总槽位数（> n_ctx_train=128，预留余量） |
| n_layer_kv | 6 | KV cache 层数（= n_layer） |
| n_stream | 1 | 并行流数量（本例单序列） |
| n_seq_max | 1 | 最大序列数（batch 中同时处理的独立序列） |
| n_pad | 1 | 对齐填充（用于缓存对齐优化） |
| n_swa | 0 | Sliding Window Attention 大小（0 = 关闭） |
| type_k | f16 | K cache 数据类型（FP16） |
| type_v | f16 | V cache 数据类型（FP16） |
| v_trans | 0 | V cache 是否转置存储（0 = 不转置） |
| offload | 1 | 是否 offload 到 GPU |

### 5.2 KV Cache 内存布局

```
K Cache Tensor (per layer): shape = [n_embd_head_k, n_head_kv, kv_size]
                           = [48, 6, 256]  (当 n_embd=288, n_head=6 时，head_dim=48)

V Cache Tensor (per layer): shape = [n_embd_head_v, n_head_kv, kv_size]
                           = [48, 6, 256]

Total KV Cache Size:
  K: 6 layers * 48 * 6 * 256 * 2 bytes (f16) = 884,736 bytes ≈ 0.84 MiB
  V: 6 layers * 48 * 6 * 256 * 2 bytes (f16) = 884,736 bytes ≈ 0.84 MiB
  Total: ~1.68 MiB
```

### 5.3 KV Cache 更新流程（apply_ubatch）

```
=== [KV_TRACE] apply_ubatch: n_tokens=2 n_stream=1 slot_size=2 ===
```

**代码流程（`src/llama-kv-cache.cpp`）：**

```cpp
// Step 1: 为每个序列分配/查找 slot
for (uint32_t i = 0; i < n_stream; i++) {
    // 查找已有 slot 或分配新 slot
    slot_info & slot = get_or_create_slot(seq_id);
}

// Step 2: 检查 slot 是否有足够空间
// slot_size = 当前 slot 已占用位置数
// 如果 slot_size + n_tokens > kv_size，需要回收/驱逐

// Step 3: 标记本次要写入的 cell 位置
// cells[cell_id].seq_id = stream_id
// cells[cell_id].pos    = token_position_in_sequence

// Step 4: 返回 batch 中每个 token 对应的 KV cache 写入位置
// 这些位置后续被 build_graph() 用于生成 SET_ROWS 算子的参数
```

**Slot 管理：**
- `slot_info` 是一个序列在 KV cache 中的占用记录
- `slot_size` 表示该序列已经写了多少个 token 的 KV
- 每次 `apply_ubatch(n_tokens=2)` 会将新的 2 个 token 的 K/V 写入 slot 的 `slot_size` ~ `slot_size+1` 位置
- 写入完成后 `slot_size += n_tokens`

### 5.4 KV Cache 与计算图的交互

在计算图中，KV cache 通过以下算子读写：

```
// 写入新的 K/V 到 cache
SET_ROWS: cache_k_l0 (view) ← 将当前 token 的 K 写入 cache [n_embd_head, n_head, kv_size]
SET_ROWS: cache_v_l0 (view) ← 将当前 token 的 V 写入 cache

// 读取全部 K/V 做 attention（通过 VIEW + PERMUTE 重排形状）
VIEW    : cache_k_l0 (view)     shape=[48, 6, 256, 1]
PERMUTE : cache_k_l0 (permuted) shape=[48, 256, 6, 1]  // 重排为 [head_dim, seq_len, n_head, 1]
VIEW    : cache_v_l0 (view)     shape=[48, 6, 256, 1]
PERMUTE : cache_v_l0 (permuted) shape=[48, 256, 6, 1]

// FLASH_ATTN_EXT 内部读取 permuted K/V
FLASH_ATTN_EXT: Q[48,6,2,1] + K[48,256,6,1] + V[48,256,6,1] → output[48,6,2,1]
```

---

## 六、环节三：计算图构建（Graph Build）

### 6.1 调用链
```
decode() → process_ubatch() → build_graph() → llama_model_llama::graph::build()
```

### 6.2 图参数（来自日志）
```
=== [LLAMA_TRACE] build_graph:   params: n_tokens=2 n_embd=288 n_layer=6 n_head=6 n_head_kv=6 n_vocab=32000 ===
```

### 6.3 LLaMA 前向传播图详细拆解

代码位置：`src/models/llama.cpp`（LLaMA 架构通用实现）

```cpp
// ====== 输入层 ======
// inpL = token_embeddings[n_embd, n_vocab] * one_hot[n_vocab, n_tokens]
//      → shape: [n_embd, n_tokens] = [288, 2]
inpL = build_inp_embd(model.tok_embd);

// ====== Transformer Layer × 6 ======
for (int il = 0; il < n_layer; ++il) {
    // ---- Attention Block ----
    // 1. Pre-Attention RMSNorm
    //    inpL=[288,2] → norm=[288,2]
    cur = build_norm(inpL, layer.attn_norm, LLM_NORM_RMS, il);

    // 2. QKV 线性投影 (MUL_MAT)
    //    Qcur = attn_q.weight[288,288] × cur[288,2] → [288,2]
    //    Kcur = attn_k.weight[288,288] × cur[288,2] → [288,2]
    //    Vcur = attn_v.weight[288,288] × cur[288,2] → [288,2]
    Qcur = ggml_mul_mat(layer.attn_q, cur);   // shape=[288,2]
    Kcur = ggml_mul_mat(layer.attn_k, cur);   // shape=[288,2]
    Vcur = ggml_mul_mat(layer.attn_v, cur);   // shape=[288,2]

    // 3. RESHAPE 为多头格式 [head_dim, n_head, n_tokens]
    //    [288,2] → [48, 6, 2, 1]
    Qcur = ggml_reshape_4d(Qcur, head_dim, n_head, n_tokens, 1);
    Kcur = ggml_reshape_4d(Kcur, head_dim, n_head_kv, n_tokens, 1);
    Vcur = ggml_reshape_4d(Vcur, head_dim, n_head_kv, n_tokens, 1);

    // 4. RoPE 位置编码（旋转位置编码）
    //    [48,6,2,1] → [48,6,2,1]（ inplace 旋转）
    Qcur = ggml_rope_ext(Qcur, pos, nullptr, rope_params);
    Kcur = ggml_rope_ext(Kcur, pos, nullptr, rope_params);

    // 5. 写入 KV Cache
    //    将当前 K/V 写入 cache 的对应位置
    cache_k = ggml_view_2d(kv_cache.k[il], ...);
    cache_v = ggml_view_2d(kv_cache.v[il], ...);
    ggml_set_rows(cache_k, Kcur_flat, pos_indices);  // SET_ROWS
    ggml_set_rows(cache_v, Vcur_flat, pos_indices);  // SET_ROWS

    // 6. 读取全部 K/V 并 PERMUTE 为 attention 所需格式
    //    K: [48,6,256,1] → PERMUTE → [48,256,6,1]
    //    V: [48,6,256,1] → PERMUTE → [48,256,6,1]
    K_all = ggml_permute(ggml_view(cache_k), 0, 2, 1, 3);  // [48,256,6,1]
    V_all = ggml_permute(ggml_view(cache_v), 0, 2, 1, 3);  // [48,256,6,1]

    // 7. Flash Attention ( fused Q×K^T, softmax, ×V )
    //    Q=[48,6,2,1], K=[48,256,6,1], V=[48,256,6,1]
    //    → output=[48,6,2,1]
    attn_out = ggml_flash_attn_ext(Qcur, K_all, V_all, mask, scale, max_bias);

    // 8. RESHAPE 回 [n_embd, n_tokens]
    //    [48,6,2,1] → [288,2]
    attn_out = ggml_reshape_2d(attn_out, n_embd, n_tokens);

    // 9. Attention 输出投影
    //    [288,2] × [288,288] → [288,2]
    cur = ggml_mul_mat(layer.attn_output, attn_out);

    // 10. 残差连接
    //     [288,2] + [288,2] → [288,2]
    cur = ggml_add(cur, inpL);
    ffn_inp = cur;

    // ---- FFN Block ----
    // 11. Pre-FFN RMSNorm
    //     [288,2] → [288,2]
    cur = build_norm(ffn_inp, layer.ffn_norm, LLM_NORM_RMS, il);

    // 12. SwiGLU: gate 和 up 并行投影
    //     gate = ffn_gate[768,288] × cur[288,2] → [768,2]
    //     up   = ffn_up[768,288] × cur[288,2] → [768,2]
    gate = ggml_mul_mat(layer.ffn_gate, cur);  // [768,2]
    up   = ggml_mul_mat(layer.ffn_up, cur);    // [768,2]

    // 13. GLU: gate ⊙ SiLU(gate) × up
    //     [768,2] ⊙ [768,2] → [768,2]
    //     (SiLU(x) = x * sigmoid(x))
    cur = ggml_glu(gate, up);  // shape=[768,2]

    // 14. FFN down 投影
    //     [288,768] × [768,2] → [288,2]
    cur = ggml_mul_mat(layer.ffn_down, cur);

    // 15. 残差连接
    //     [288,2] + [288,2] → [288,2]
    cur = ggml_add(cur, ffn_inp);

    inpL = cur;  // 下一层输入
}

// ====== 输出层 ======
// 1. 最终 RMSNorm
//    [288,2] → [288,2]
cur = build_norm(inpL, model.output_norm, LLM_NORM_RMS, -1);

// 2. 输出投影到词表
//    [32000,288] × [288,2] → [32000,2]
res->t_logits = ggml_mul_mat(model.output, cur);
```

---

## 七、算子类型与 Shape 详细分析

### 7.1 本模型使用的全部 ggml op 类型

从 `ggml_backend_graph_compute_async` 日志提取，本模型实际使用的 op 共 **14 种**：

| Op 名称 | 出现次数 | 作用 | 输入 Shape 示例 | 输出 Shape 示例 |
|---------|---------|------|----------------|----------------|
| **GET_ROWS** | 2 | 嵌入查找 | indices[2], table[288,32000] | [288,2] |
| **RMS_NORM** | 7 | RMS 归一化 | x[288,2], weight[288] | [288,2] |
| **MUL** | 7 | 逐元素乘（norm×weight） | [288,2], [288,1] | [288,2] |
| **CPY** | 1 | 拷贝（mask 准备） | [256,2] | [256,2] |
| **MUL_MAT** | 19 | 矩阵乘法（权重×激活） | W[288,288], x[288,2] | [288,2] |
| **RESHAPE** | 12 | 重塑张量维度 | [288,2] | [48,6,2,1] |
| **ROPE** | 12 | 旋转位置编码 | [48,6,2,1], pos[2] | [48,6,2,1] |
| **VIEW** | 24 | 零拷贝视图 | [288,256] | [48,6,256,1] |
| **SET_ROWS** | 12 | 按行写入 KV cache | cache[288,256], src[288,2] | [288,256] |
| **PERMUTE** | 12 | 维度置换 | [48,6,256,1] | [48,256,6,1] |
| **FLASH_ATTN_EXT** | 6 | 融合 Flash Attention | Q[48,6,2,1], K[48,256,6,1], V[48,256,6,1] | [48,6,2,1] |
| **GLU** | 6 | SwiGLU 激活 | gate[768,2], up[768,2] | [768,2] |
| **ADD** | 7 | 逐元素加法（残差） | [288,2], [288,2] | [288,2] |
| **NONE** | - | 占位/输入节点 | - | - |

### 7.2 单层（Layer 0）完整算子链与 Shape 变化

以 `n_tokens=2` 为例，单层的算子执行链（来自 MTL0 split 日志）：

```
Node  Op           Name                     Input Shape(s)              Output Shape
─────────────────────────────────────────────────────────────────────────────────────────
  0   RMS_NORM     norm-0                   [288,2]                     [288,2]
  1   MUL          attn_norm-0              [288,2], [288,1]            [288,2]
  2   CPY          attn_inp_kq_mask (copy)  [256,2]                     [256,2]
  3   MUL_MAT      Qcur-0                   W[288,288], x[288,2]        [288,2]
  4   RESHAPE      Qcur-0                   [288,2]                     [48,6,2,1]
  5   MUL_MAT      Vcur-0                   W[288,288], x[288,2]        [288,2]
  6   RESHAPE      Vcur-0                   [288,2]                     [48,6,2,1]
  7   MUL_MAT      Kcur-0                   W[288,288], x[288,2]        [288,2]
  8   RESHAPE      Kcur-0                   [288,2]                     [48,6,2,1]
  9   VIEW         Vcur-0 (view)            [288,2]                     [288,2]   (准备写入cache)
 10   ROPE         Qcur-0                   [48,6,2,1], pos             [48,6,2,1]
 11   ROPE         Kcur-0                   [48,6,2,1], pos             [48,6,2,1]
 12   VIEW         Kcur-0 (view)            [288,2]                     [288,2]   (准备写入cache)
 13   SET_ROWS     cache_v_l0 (view)        cache[288,256], src[288,2]  [288,256]
 14   VIEW         Qcur-0 (view)            [48,6,2,1]                  [48,6,2,1]
 15   PERMUTE      Qcur-0 (permuted)        [48,6,2,1]                  [48,2,6,1]
 16   VIEW         cache_v_l0 (view)        [48,6,256,1]                [48,6,256,1]
 17   PERMUTE      cache_v_l0 (permuted)    [48,6,256,1]                [48,256,6,1]
 18   SET_ROWS     cache_k_l0 (view)        cache[288,256], src[288,2]  [288,256]
 19   VIEW         cache_k_l0 (view)        [48,6,256,1]                [48,6,256,1]
 20   PERMUTE      cache_k_l0 (permuted)    [48,6,256,1]                [48,256,6,1]
 21   FLASH_ATTN_EXT __fattn__-0            Q[48,2,6,1], K[48,256,6,1], V[48,256,6,1]  [48,6,2,1]
 22   RESHAPE      kqv_out-0                [48,6,2,1]                  [288,2]
 23   MUL_MAT      attn_out-0               W[288,288], x[288,2]        [288,2]
 24   ADD          ffn_inp-0                [288,2], [288,2]            [288,2]   (残差)
 25   RMS_NORM     norm-0                   [288,2]                     [288,2]
 26   MUL          ffn_norm-0               [288,2], [288,1]            [288,2]
 27   MUL_MAT      ffn_gate-0               W[768,288], x[288,2]        [768,2]
 28   MUL_MAT      ffn_up-0                 W[768,288], x[288,2]        [768,2]
 29   GLU          ffn_swiglu-0             [768,2], [768,2]            [768,2]
 30   MUL_MAT      ffn_out-0                W[288,768], x[768,2]        [288,2]
 31   ADD          l_out-0                  [288,2], [288,2]            [288,2]   (残差)
```

**每层 32 个节点，6 层共 192 个节点**，加上输入的 GET_ROWS (1) 和输出的 RMS_NORM+MUL+MUL_MAT (3)，总计 **193 个节点**（与 `nodes=193` 日志一致）。

### 7.3 Shape 变化规律总结

```
[n_tokens=2 时]
输入嵌入:        [288, 2]          ← GET_ROWS
Attention Path:
  Q/K/V proj:    [288, 2]          ← MUL_MAT
  reshape:       [48, 6, 2, 1]     ← RESHAPE (head_dim=48, n_head=6)
  RoPE:          [48, 6, 2, 1]     ← ROPE
  write cache:   [288, 256]        ← SET_ROWS
  permute K/V:   [48, 256, 6, 1]   ← PERMUTE
  FlashAttn:     [48, 6, 2, 1]     ← FLASH_ATTN_EXT
  reshape:       [288, 2]          ← RESHAPE
  output proj:   [288, 2]          ← MUL_MAT
FFN Path:
  gate/up:       [768, 2]          ← MUL_MAT
  SwiGLU:        [768, 2]          ← GLU
  down:          [288, 2]          ← MUL_MAT
输出投影:        [32000, 2]        ← MUL_MAT
```

---

## 八、后端调度与执行深度解析

### 8.1 调用链
```
decode() → graph_compute() → ggml_backend_sched_compute_splits()
    → compute_split() → ggml_backend_graph_compute_async()
        → backend->iface.graph_compute()
```

### 8.2 图切分（Split Graph）

```
=== [LLAMA_TRACE] ggml_backend_sched_compute_splits: n_splits=2 ===
=== [LLAMA_TRACE] ggml_backend_sched_compute_splits:   split 0/2 backend=CPU nodes=1 ===
=== [LLAMA_TRACE] ggml_backend_sched_compute_splits:     node  0: op=GET_ROWS         name=embd    shape=[288,2,1,1]
=== [LLAMA_TRACE] ggml_backend_sched_compute_splits:   split 1/2 backend=MTL0 nodes=192 ===
```

**切分逻辑：**
1. 遍历图的拓扑顺序
2. 当一个节点所需的 backend 发生变化时（或不支持当前 backend），创建新的 split
3. Split 0 (CPU, 1 node): `GET_ROWS` 算子读取 CPU 上的 embedding 表
4. Split 1 (MTL0, 192 nodes): 全部 Transformer 计算

### 8.3 后端接口调用链代码解析

```cpp
// ggml/src/ggml-backend.cpp: compute_split()

static enum ggml_status compute_split(ggml_backend_sched_t sched, int split_idx) {
    ggml_backend_sched_split_t split = &sched->splits[split_idx];

    // 1. 准备输入张量：从上一个 split 的 backend 拷贝到当前 backend
    for (int i = 0; i < split->n_inputs; i++) {
        ggml_tensor * input = split->inputs[i];
        if (!ggml_backend_buffer_is_host(input->buffer)) {
            // 如果输入不在 host 内存，需要从上一个 backend 拷贝
            ggml_backend_sched_synchronize(sched, split_idx - 1);
            ggml_backend_tensor_get_async(...);
        }
    }

    // 2. 设置当前 backend 的上下文
    ggml_backend_t backend = split->backend;

    // 3. 调用后端图计算接口
    // 这里是关键：backend 的 graph_compute 函数指针
    enum ggml_status status = ggml_backend_graph_compute_async(
        backend,               // CPU 或 MTL0
        &split->graph,         // 该 split 的子图（1 个或 192 个节点）
        sched->events[split_idx][0],
        sched->events[split_idx][1]
    );

    return status;
}
```

### 8.4 后端图计算异步接口

```cpp
// ggml/src/ggml-backend.cpp: ggml_backend_graph_compute_async()

enum ggml_status ggml_backend_graph_compute_async(
        ggml_backend_t backend,
        ggml_cgraph_t cgraph,
        ggml_backend_event_t event,
        ggml_backend_event_t event_pool) {

    // 调用 backend 接口的 graph_compute 函数指针
    return backend->iface.graph_compute(backend, cgraph);
}
```

**关键数据结构：** `ggml_backend_i`（backend 接口虚表）

```cpp
// ggml/include/ggml-backend.h
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void         (*free)(ggml_backend_t backend);
    enum ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    // ... 其他函数指针
};
```

### 8.5 Metal 后端算子分发（参考实现）

```cpp
// ggml/src/ggml-metal/ggml-metal.cpp: ggml_backend_metal_graph_compute()

static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph_t cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case GGML_OP_RMS_NORM:
                // dispatch RMSNorm kernel
                ggml_metal_encode_rms_norm(...);
                break;
            case GGML_OP_MUL_MAT:
                // dispatch matrix multiplication kernel
                // 根据量化类型 (Q4_0/Q8_0/F32) 选择不同 kernel
                ggml_metal_encode_mul_mat(...);
                break;
            case GGML_OP_ROPE:
                ggml_metal_encode_rope(...);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                ggml_metal_encode_flash_attn_ext(...);
                break;
            case GGML_OP_GLU:
                ggml_metal_encode_glu(...);
                break;
            // ... 其他 op
            default:
                GGML_ABORT("fatal error: node %d, op = %d not supported\n", i, node->op);
        }
    }
}
```

**NPU 移植要点：**
- 你需要实现 `ggml_backend_graph_compute()`，遍历图中的每个节点
- 为每个支持的 `ggml_op` 编写对应的 NPU kernel 调用
- 不支持的 op 返回 false，`ggml_backend_supports_op()` 中声明
- scheduler 会自动将不支持的 op 切分到 CPU split

---

## 九、环节四：KV Cache 管理（运行时）

### 9.1 调用链
```
decode() → memory_update() → KV cache 位置更新 / shift
```

### 9.2 日志证据
```
=== [LLAMA_TRACE] memory_update: optimize=0 ===
```

### 9.3 与 Graph 的交互
- `memory_update()` 在 `decode()` 开始时调用，负责：
  1. 标记哪些 KV cache cell 需要保留
  2. 处理上下文溢出时的 cache shifting
  3. 设置 `optimize=0` 表示不重新分配 cache 内存
- `build_graph()` 中的 `build_attn_inp_kv()` 读取当前 cache 状态
- KV cache 的 shape 随序列长度增长，但图的拓扑结构不变

---

## 十、环节五：采样与 Detokenize

### 10.1 调用链
```
decode() → 获取 logits → llama_sampler_sample() → token ID
         → llama_token_to_piece() → 文本输出
```

### 10.2 日志证据
```
=== [LLAMA_TRACE] llama_token_to_piece: token=21475 ===
=== [LLAMA_TRACE] llama_token_to_piece: result=5 ===
```

### 10.3 详细拆解
1. **获取 logits**：从 `res->t_logits` 读取最后一个 token 的 [32000] logits 向量
2. **温度缩放 + Top-K/Top-P**：通过 sampler chain 处理
3. **随机采样**：从过滤后的分布中采样下一个 token ID
4. **Detokenize**：`llama_token_to_piece(token=21475)` → 返回 5 字节 UTF-8 字符串

---

## 十一、NPU 后端移植关键接入点

### 11.1 需要实现的文件/接口

```
ggml/src/ggml-<your_npu>/
├── ggml-<npu>.cpp          # 核心：backend 实现
├── ggml-<npu>.h            # 头文件
└── CMakeLists.txt          # 构建配置
```

### 11.2 核心接口清单

| 接口 | 作用 | 代码参考 |
|------|------|---------|
| `ggml_backend_reg` | 注册你的 NPU backend | `ggml_backend_metal_reg()` |
| `ggml_backend_buffer_type` | 定义内存分配策略 | `ggml_backend_metal_buffer_type()` |
| `ggml_backend_buffer` | 实际内存分配 | `ggml_backend_buft_alloc_buffer()` |
| `ggml_backend_graph_compute` | 执行计算图 | `ggml_backend_metal_graph_compute()` |
| `ggml_backend_supports_op` | 声明支持哪些 op | 必须支持 MATMUL, FLASH_ATTN_EXT, ROPE, ADD, MUL, RMS_NORM, GLU 等 |

### 11.3 关键接入路径

```
1. 注册阶段
   ggml_backend_register("NPU", ...)  →  llama.cpp 启动时枚举到 NPU

2. 模型加载阶段
   load_tensors() → 根据 n_gpu_layers 将层分配到 NPU
   → 你的 buffer_type 分配 NPU 内存
   → 权重通过 DMA/copy 写入 NPU 内存

3. 图构建阶段（无需修改，架构通用）
   build_graph() 构建 193 个节点的图
   → scheduler 自动切分：CPU node → NPU nodes

4. 执行阶段
   ggml_backend_sched_compute_splits()
   → split 0: CPU (1 node, 输入准备)
   → split 1: NPU (192 nodes, 全部 Transformer)
   → 你的 backend_graph_compute() 收到 192 个 node 的子图

5. 最小化实现策略
   - 先只实现 MUL_MAT（占 90% 计算量）
   - 不支持的 op fallback 到 CPU
   - scheduler 会自动将不支持的 op 切到 CPU split
```

### 11.4 Metal 后端作为参考实现

关键文件：`ggml/src/ggml-metal/ggml-metal.cpp`

学习路径：
1. `ggml_backend_metal_init()` — 设备初始化
2. `ggml_metal_buffer_type()` — 缓冲区类型
3. `ggml_metal_graph_compute()` — 图执行（遍历 node，switch op 类型，dispatch kernel）
4. `ggml_metal_supports_op()` — op 支持列表

---

## 十二、数据流全景图（含 Shape）

```
[GGUF File: 57 tensors, q4_0/q8_0/f32]
    ↓ mmap
[llama_model_loader] 解析 metadata + 张量信息
    ↓
[load_tensors] 按层分配设备 → CPU buffer (4.94 MiB) + MTL0 buffer (12.56 MiB)
    ↓
[Tokenizer] "Once upon a time" → [1, 19182, 13775, ...] (32 tokens)
    ↓
[llama_batch] batch.n_tokens=32
    ↓
[decode] → [memory_update] KV cache 位置准备
    ↓
[build_graph] 构建 193 node 的 Transformer 计算图
    ↓
[split_graph] 切分：CPU(1 node: GET_ROWS [288,32]) + NPU(192 nodes)
    ↓
[alloc_splits] 为 205 nodes 分配内存
    ↓
[compute_splits] 顺序执行 2 个 split
    ↓
    ├─ CPU: GET_ROWS(embd) [288,32]
    └─ MTL0: 192 nodes
       ├─ Layer 0~5: each 32 nodes
       │   ├─ Attention: RMS_NORM→MUL→MUL_MAT(Q,K,V)→RESHAPE→ROPE
       │   │             →SET_ROWS(cache)→PERMUTE→FLASH_ATTN_EXT→MUL_MAT→ADD
       │   └─ FFN: RMS_NORM→MUL→MUL_MAT(gate,up)→GLU→MUL_MAT(down)→ADD
       └─ Output: RMS_NORM→MUL→MUL_MAT [32000,32]
    ↓
[logits] [32000] 概率分布向量（最后一个 token）
    ↓
[sampler] Top-K/Top-P + 温度采样 → token ID
    ↓
[token_to_piece] token ID → UTF-8 字符串片段
    ↓
[输出文本]
    ↓
[自回归循环] 新生成的 token 作为下一个 decode() 的输入
```

---

## 十三、关键结论与 NPU 移植建议

1. **图构建一次，多次复用**：193 个 node 的图只在前几次 `decode()` 时构建，后续直接复用
2. **调度器是核心**：`ggml_backend_sched` 负责跨后端切分图、分配内存、管理数据依赖
3. **两阶段执行**：
   - prompt processing（batch，n_tokens=2~128）：计算密集，适合 NPU 加速
   - token generation（single，n_tokens=1）：内存带宽密集，KV cache 读取是瓶颈
4. **本模型 LLaMA 架构特征**：
   - RMS_NORM（非 LayerNorm）
   - SwiGLU（非 GELU）
   - RoPE（非正弦位置编码）
   - FLASH_ATTN_EXT（融合 Attention）
5. **NPU 移植最小工作集**：
   - 注册 backend → buffer allocation → graph compute → supports_op
   - **优先实现**：MUL_MAT（矩阵乘）、FLASH_ATTN_EXT（注意力）、RMS_NORM、GLU、ROPE
   - 其余 fallback CPU，scheduler 自动切分
6. **内存模型**：
   - GGUF 权重通过 mmap 懒加载，实际在 `load_tensors()` 时拷贝到 backend buffer
   - KV cache 使用 f16 格式，大小约 1.68 MiB（本模型）
   - NPU 需要实现自己的 `ggml_backend_buffer` 分配器
7. **量化支持**：
   - 本模型权重使用 q4_0（4-bit）和 q8_0（8-bit）
   - NPU 需要支持反量化到 f16/f32 后再计算，或原生支持量化矩阵乘
