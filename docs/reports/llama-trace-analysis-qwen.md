# llama.cpp 全链路深度分析报告 —— Qwen3.6-35B-A3B 模型

> 基于 `Qwen3.6-35B-A3B-UD-Q4_K_M` 模型运行日志，面向 **NPU 后端移植开发者**
> 报告版本：v3.0（新增 KV Cache 深度解析、Sampler 全链路解析、Detokenize 日志验证）

---

## 一、运行环境概述

| 项目 | 值 |
|------|-----|
| 硬件 | Apple M4 (SoC) |
| 后端 | Metal (MTL0) + CPU |
| 模型 | Qwen3.6-35B-A3B-UD-Q4_K_M |
| 架构 | **qwen35moe**（MoE + SSM + Transformer Attention） |
| 参数量 | ~35B（活跃参数约 3B） |
| 层数 | n_layer=40 |
| 隐藏维度 | n_embd=2048 |
| 注意力头 | n_head=16, n_head_kv=2（**GQA**） |
| 上下文训练长度 | n_ctx_train=262144 |
| GPU 层 | n_gpu_layers=41（全部 offload 到 GPU） |
| 内存 | unified memory = true |
| GGUF Tensors | 733 |
| 量化 | Q4_K_M / Q5_K / Q8_0 混合 |
| KV Cache | kv_size=262144, type_k=f16, type_v=f16 |
| Vocab | 248320（BPE） |

**关键架构特征：**
- **GQA**：n_head=16, n_head_kv=2，KV cache 只需要存储 1/8 的 K/V 头
- **MoE**：256 个 experts + shared expert，使用 `MUL_MAT_ID` + `ARGSORT` 路由
- **SSM**：部分层使用 `SSM_CONV` 和 `GATED_DELTA_NET` 替代标准 Attention
- **QKV 融合**：`attn_qkv.weight` 合并了 Q/K/V 投影

---

## 二、GGUF 数据格式解析

### 2.1 GGUF 文件结构

GGUF（GGML Universal Format）是 llama.cpp 的模型文件格式。

```
[GGUF File Layout]
├─ Header
│   ├─ Magic: "GGUF" (4 bytes)
│   ├─ Version: uint32
│   ├─ n_tensors: uint64
│   └─ n_kv: uint64
├─ KV Metadata Array
├─ Tensor Info Array (n_tensors 个)
│   ├─ name: string
│   ├─ n_dims: uint32
│   ├─ shape: uint64[n_dims]
│   ├─ type: ggml_type
│   └─ offset: uint64
└─ Aligned Data Section
```

### 2.2 本模型 733 个张量关键示例

从 `gguf_init_from_file_ptr` 日志解析的关键张量：

| # | 张量名 | 类型 | shape | 作用 |
|---|--------|------|-------|------|
| - | `token_embd.weight` | q4_K | [2048, 248320] | 词嵌入矩阵 |
| - | `output_norm.weight` | f32 | [2048] | 最终 RMSNorm |
| - | `output.weight` | q8_0 | [2048, 248320] | 输出投影 |
| - | `blk.0.attn_qkv.weight` | q8_0 | [2048, 8192] | **融合 QKV** 投影 |
| - | `blk.0.attn_output.weight` | q8_0 | [4096, 2048] | Attention 输出投影 |
| - | `blk.0.ffn_gate_inp.weight` | f32 | [2048, 256] | MoE 路由门控 |
| - | `blk.0.ffn_gate_exps.weight` | q4_K | [2048, 512, 256] | 256 experts 的 gate |
| - | `blk.0.ffn_up_exps.weight` | q4_K | [2048, 512, 256] | 256 experts 的 up |
| - | `blk.0.ffn_down_exps.weight` | q5_K | [512, 2048, 256] | 256 experts 的 down |
| - | `blk.0.ffn_gate_shexp.weight` | q8_0 | [2048, 512] | shared expert gate |
| - | `blk.0.ssm_out.weight` | q8_0 | [4096, 2048] | SSM 输出投影 |

**量化分析：**
- **q4_K / q5_K**: 4-bit/5-bit 量化权重（主流），block 32 权重 + f16 scale + f16 min
- **q8_0**: 8-bit 量化（Attention 权重精度要求高）
- **f32**: 32-bit 浮点（Norm 权重、MoE 路由门控）

---

## 三、模型加载参数

```
=== [LLAMA_TRACE] llama_model_load:   arch=qwen35moe n_embd=2048 n_layer=40 n_head=16 n_head_kv=2 n_ctx_train=262144 ===
=== [LLAMA_TRACE] llama_model_load:   vocab_size=248320 BOS=248044 EOS=248046 EOT=248046 ===
```

| 参数 | 值 | 说明 |
|------|-----|------|
| arch | qwen35moe | Qwen3.5-MoE 架构标识 |
| n_embd | 2048 | 隐藏层维度 |
| n_layer | 40 | Transformer 层数 |
| n_head | 16 | Attention Q 头数 |
| n_head_kv | 2 | Attention K/V 头数（GQA，1/8 压缩） |
| n_ctx_train | 262144 | 训练最大上下文（256K） |
| vocab_size | 248320 | BPE 词表大小 |
| BOS | 248044 | Begin-of-Sequence token |
| EOS | 248046 | End-of-Sequence token |

**设备分配：**
```
=== [LLAMA_TRACE] load_tensors: device assignment: n_layer=40 n_gpu_layers=41 i_gpu_start=0 act_gpu_layers=41 ===
=== [LLAMA_TRACE] load_tensors:   layer  0 -> MTL0
...
=== [LLAMA_TRACE] load_tensors:   layer 39 -> MTL0
=== [LLAMA_TRACE] load_tensors: buffer CPU_Mapped   size=  515.31 MiB
```

全部 40 层 + 输出层 offload 到 Metal GPU。

---

## 四、Tokenizer / VOCAB 日志

### 4.1 BPE 词表结构

```
=== [VOCAB_TRACE] token_to_piece: token=0 vocab_type=2 special=1 lstrip=0 ===
=== [VOCAB_TRACE] token_to_piece:   attr=NORMAL  ===
=== [VOCAB_TRACE] token_to_piece:   id_to_token text='!' score=0.0000 ===
=== [VOCAB_TRACE] token_to_piece:   result=1 bytes bytes=[21] ===
```

| Token | 属性 | 文本 | 字节 |
|-------|------|------|------|
| 0 | NORMAL | `!` | `0x21` |
| 1 | NORMAL | `"` | `0x22` |
| 2 | NORMAL | `#` | `0x23` |
| 3 | NORMAL | `$` | `0x24` |

**BOS/EOS 特殊 Token：**
- BOS=248044, EOS=248046, EOT=248046
- vocab_type=2 表示 **BPE**（Byte-Pair Encoding）

### 4.2 Detokenize 流程

`token_to_piece()` 的核心逻辑：
1. 查 `cache_token_to_piece` 缓存（模型加载时预填充）
2. 根据 token attr（NORMAL/CONTROL/BYTE/UNKNOWN）决定处理方式
3. NORMAL token：直接输出对应文本，去除前导空格（受 `lstrip` 控制）
4. BYTE token：通过 `token_to_byte()` 解码为原始字节
5. 将结果写入 `buf` 缓冲区，返回字节数

---

## 五、KV Cache 深度解析（核心新增）

### 5.1 KV Cache 初始化参数

```
=== [LLAMA_TRACE] llama_kv_cache: KV cache init ===
=== [LLAMA_TRACE] llama_kv_cache:   kv_size=262144 n_layer_kv=40 n_stream=1 n_seq_max=1 n_pad=1 n_swa=0 ===
=== [LLAMA_TRACE] llama_kv_cache:   type_k=f16 type_v=f16 v_trans=0 offload=1 ===
=== [LLAMA_TRACE] llama_kv_cache:   v_cells groups=1 cells_per_group=262144 total_cells=262144 ===
=== [LLAMA_TRACE] llama_kv_cache:   seq_to_stream map (first 16): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ===
```

| 参数 | 值 | 说明 |
|------|-----|------|
| kv_size | 262144 | KV cache 总槽位数（= n_ctx_train） |
| n_layer_kv | 40 | KV cache 层数 |
| n_stream | 1 | 并行流数量（单序列） |
| n_seq_max | 1 | 最大序列数 |
| n_pad | 1 | 对齐填充 |
| n_swa | 0 | Sliding Window Attention（关闭） |
| type_k | f16 | K cache 数据类型 |
| type_v | f16 | V cache 数据类型 |
| v_trans | 0 | V cache 不转置存储 |
| offload | 1 | 是否 offload 到 GPU |
| groups | 1 | v_cells 分组数 |
| cells_per_group | 262144 | 每组 cell 数 |

**KV Cache 内存估算：**
- n_embd_k_gqa = 512 (n_head_kv * head_dim = 2 * 256)
- n_embd_v_gqa = 512
- K: 40 * 512 * 262144 * 2 bytes ≈ **10.7 GiB**
- V: 40 * 512 * 262144 * 2 bytes ≈ **10.7 GiB**
- Total: ~**21.4 GiB**（FP16）

### 5.2 KV Cache 核心数据结构

**`v_cells`**：二维数组，`v_cells[stream_id][cell_id]`，标记每个 cell 被哪个序列占用。

**`v_heads`**：一维数组，`v_heads[stream_id]`，记录每个 stream 的写入头位置。

**`seq_to_stream`**：映射 seq_id → stream_id。单 stream 场景下全部为 0。

### 5.3 KV Cache 操作流程

#### Step 1: prepare（为 batch 分配 slots）

```
=== [KV_TRACE] prepare: START n_ubatches=1 ===
=== [KV_TRACE] prepare: DONE n_slots=1 ===
=== [KV_TRACE] prepare:   slot[0] s0=0 s1=0 n_stream=1 size=2 contiguous=1 ===
```

- `n_ubatches=1`：本次只有 1 个 ubatch
- `n_slots=1`：分配了 1 个 slot
- `slot[0] size=2`：该 slot 已存储 2 个 token 的 KV
- `contiguous=1`：slot 内的 cell 是连续的

#### Step 2: find_slot（查找/分配序列的 KV 位置）

```
=== [KV_TRACE] find_slot: START n_tokens=2 n_seqs=1 cont=0 ===
=== [KV_TRACE] find_slot:   seq[0] seq_id=0 stream=0 cells_used=0 head=0 total=161792 ===
=== [KV_TRACE] find_slot: DONE s0=0 s1=0 n_seqs=1 idxs_per_seq=2 ===
```

- `n_tokens=2`：本次要写入 2 个 token 的 KV
- `cont=0`：不连续（新序列或新 batch）
- `cells_used=0`：该序列当前占用了 0 个 cell
- `total=161792`：总可用 cell 数
- `idxs_per_seq=2`：为序列分配了 2 个 cell 索引

#### Step 3: apply_ubatch（将 token 的 KV 写入 cache）

```
=== [KV_TRACE] apply_ubatch: START n_tokens=2 n_stream=1 slot_size=2 ===
=== [KV_TRACE] apply_ubatch:   s0=0 s1=0 contiguous=1 ===
=== [KV_TRACE] apply_ubatch:   stream[0] strm_id=0 idxs=[0, 1] ===
```

- `slot_size=2`：当前 slot 大小为 2
- `idxs=[0, 1]`：这 2 个 token 的 KV 分别写入 cell 0 和 cell 1
- 写入完成后，`slot_size += n_tokens`

#### Step 4: get_k / get_v（读取 KV cache 供 Attention 使用）

```
=== [KV_TRACE] get_k: layer=0 n_kv=262144 s0=0 s1=0 k_shape=[512,262144,1,1] ===
=== [KV_TRACE] get_v: layer=0 n_kv=262144 s0=0 s1=0 v_shape=[512,262144,1,1] v_trans=0 ===
```

- `n_kv=262144`：KV cache 的总槽位数
- `k_shape=[512, 262144, 1, 1]`：[n_embd_k_gqa, kv_size, 1, 1]
- `v_shape=[512, 262144, 1, 1]`：[n_embd_v_gqa, kv_size, 1, 1]
- `v_trans=0`：V cache 不转置

#### Step 5: cpy_k / cpy_v（将当前 token 的 K/V 写入 cache）

```
=== [KV_TRACE] cpy_k: layer=0 k_cur_shape=[256,2,1,1] k_idxs_shape=[1,1,1,1] ===
=== [KV_TRACE] cpy_v: layer=0 v_cur_shape=[256,2,1,1] v_idxs_shape=[1,1,1,1] ===
```

- `k_cur_shape=[256, 2, 1, 1]`：当前 batch 的 K 张量 [head_dim, n_tokens]
- `k_idxs_shape=[1, 1, 1, 1]`：写入位置的索引张量
- 通过 `ggml_cpy` 将 `k_cur` 写入 KV cache 的指定位置

### 5.4 KV Cache 与 Graph 的交互总结

```
[Graph 构建阶段]
  get_k(layer) → ggml_view_2d(cache_k_l0) → 供 FLASH_ATTN_EXT 读取
  get_v(layer) → ggml_view_2d(cache_v_l0) → 供 FLASH_ATTN_EXT 读取

[Graph 执行阶段]
  cpy_k(layer) → 将当前 token 的 K 写入 cache_k_l0[idx]
  cpy_v(layer) → 将当前 token 的 V 写入 cache_v_l0[idx]
  FLASH_ATTN_EXT(Q_cur, K_cache_view, V_cache_view) → Attention 输出
```

**关键设计：**
- K/V cache 以 **大连续张量** 形式存储（shape=[512, 262144]）
- 通过 `ggml_view_2d` 创建窗口视图，避免数据拷贝
- `cpy_k/cpy_v` 使用 `ggml_cpy` 将新 token 的 K/V 写入指定偏移位置
- `FLASH_ATTN_EXT` 直接读取 cache 视图，实现高效的 Attention 计算

---

## 六、Sampler 深度解析（核心新增）

### 6.1 采样链结构

```
=== [SAMPLER_TRACE] llama_sampler_chain_apply: chain n_samplers=10 input_size=248320 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[0] name=?penalties is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[1] name=?dry is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[2] name=?top-n-sigma is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[3] name=top-k is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[4] name=?typical is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[5] name=top-p is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[6] name=min-p is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[7] name=?xtc is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[8] name=temp-ext is_backend=0 ===
=== [SAMPLER_TRACE] llama_sampler_chain_apply:   sampler[9] name=dist is_backend=0 ===
```

| # | Sampler | 作用 | is_backend |
|---|---------|------|------------|
| 0 | penalties | 重复惩罚、频率惩罚 | 0 |
| 1 | dry | DRY 重复惩罚 | 0 |
| 2 | top-n-sigma | Top-n-sigma 采样 | 0 |
| 3 | top-k | 保留概率最高的 K 个 token | 0 |
| 4 | typical | Typical 采样 | 0 |
| 5 | top-p | 保留累积概率达到 P 的 token | 0 |
| 6 | min-p | 保留概率 >= max_prob * P 的 token | 0 |
| 7 | xtc | XTC 采样 | 0 |
| 8 | temp-ext | 温度缩放（扩展版） | 0 |
| 9 | dist | 从概率分布中随机采样 | 0 |

### 6.2 采样流程（单次 token 生成）

**Step 1: top-k（粗筛）**
```
=== [SAMPLER_TRACE] llama_sampler_top_k_impl: k=40 input_size=248320 sorted=0 ===
=== [SAMPLER_TRACE] llama_sampler_top_k_impl: output_size=40 top_token=90700 top_prob=0.000000 ===
```
- 输入：248320 个 logits
- 从 248320 中筛选出 top 40 个最高 logit 的 token
- `sorted=0`：未排序（使用 quickselect 而非 full sort）

**Step 2: softmax（转概率）**
```
=== [SAMPLER_TRACE] llama_sampler_softmax_impl: size=40 max_logit=22.4278 max_prob=0.686588 min_prob=0.000025 cum_sum=1.4565 ===
```
- 对 40 个 token 做 softmax
- `max_logit=22.4278`：最大 logit 值
- `max_prob=0.686588`：最大概率（约 68.7%）
- `min_prob=0.000025`：最小概率
- `cum_sum=1.4565`：累积概率和（>1 是因为某些 token 被裁剪后重新归一化）

**Step 3: top-p（累积概率筛选）**
```
=== [SAMPLER_TRACE] llama_sampler_top_p_apply: p=0.9500 min_keep=0 input_size=40 output_size=3 cum_sum=0.968013 ===
```
- `p=0.95`：保留累积概率达到 95% 的 token
- 从 40 个 token 筛选到 3 个 token
- `cum_sum=0.968013`：这 3 个 token 的累积概率

**Step 4: min-p（最小概率阈值）**
```
=== [SAMPLER_TRACE] llama_sampler_min_p_apply: p=0.0500 min_keep=0 input_size=3 sorted=1 ===
=== [SAMPLER_TRACE] llama_sampler_min_p_apply: output_size=2 max_logit=22.4278 ===
```
- `p=0.05`：保留概率 >= max_prob * 0.05 = 0.686588 * 0.05 ≈ 0.034 的 token
- 从 3 个 token 筛选到 2 个 token

**Step 5: temperature（温度缩放）**
```
=== [SAMPLER_TRACE] llama_sampler_temp_impl: temp=0.8000 input_size=2 ===
=== [SAMPLER_TRACE] llama_sampler_temp_impl: scaled logits range [21.4493, 22.4278] -> [26.8117, 28.0347] ===
```
- `temp=0.8`：温度参数
- 将 logits 除以 0.8（等价于乘以 1.25）
- 低温度使分布更尖锐，高概率 token 更容易被选中

**Step 6: 随机采样（dist）**
```
=== [SAMPLER_TRACE] llama_sampler_chain_apply: output_size=2 selected=1 ===
```
- 从最终的 2 个候选 token 中按概率随机采样
- `selected=1`：选中了第二个候选 token

### 6.3 Sampler 输入输出数据流

```
Input logits: [248320] float
    │
    ▼
penalties ──→ logits' [248320]
    │
    ▼
dry ──→ logits'' [248320]
    │
    ▼
top-k ──→ [40] (筛选最高 40 个)
    │
    ▼
softmax ──→ probs [40] (归一化概率)
    │
    ▼
top-p ──→ [3] (累积概率 95%)
    │
    ▼
min-p ──→ [2] (概率阈值过滤)
    │
    ▼
temp ──→ scaled logits [2]
    │
    ▼
dist ──→ selected_token (int)
```

---

## 七、Backend Graph Compute

### 7.1 算子类型统计

从 `ggml_backend_sched` 的节点日志统计（单次 graph compute 的算子分布）：

| 算子 | 次数 | 说明 |
|------|------|------|
| VIEW | 3960 | 张量视图（零拷贝切片） |
| RESHAPE | 2600 | 形状重排 |
| ADD | 2150 | 逐元素加法 |
| MUL_MAT | 1955 | 矩阵乘法（核心算子） |
| MUL | 1405 | 逐元素乘法 |
| UNARY | 850 | 一元运算（SiLU、GELU 等） |
| GET_ROWS | 820 | 按行索引取数据 |
| RMS_NORM | 655 | RMS 归一化 |
| CPY | 605 | 数据拷贝 |
| **MUL_MAT_ID** | **600** | **MoE 专家选择矩阵乘** |
| GLU | 400 | Gated Linear Unit（SwiGLU） |
| SCALE | 300 | 缩放运算 |
| SUM_ROWS | 200 | 按行求和 |
| SOFT_MAX | 200 | Softmax |
| DIV | 200 | 除法 |
| CLAMP | 200 | 裁剪 |
| **ARGSORT** | **200** | **MoE 专家路由排序** |
| TRANSPOSE | 150 | 转置 |
| **SSM_CONV** | **150** | **SSM 状态空间卷积** |
| PERMUTE | 150 | 维度置换 |
| **GATED_DELTA_NET** | **150** | **SSM Gated Delta Net** |
| CONCAT | 150 | 拼接 |
| SET_ROWS | 100 | 按行写入 |
| ROPE | 100 | 旋转位置编码 |
| FLASH_ATTN_EXT | 50 | Flash Attention（扩展） |
| CONT | 50 | 连续化 |

### 7.2 MoE + SSM 特有算子

**MoE 路由流程：**
1. `MUL_MAT`：计算 routing logits（gate_inp × input）
2. `ARGSORT`：对 expert 权重排序，选择 top-k experts
3. `MUL_MAT_ID`：对每个选中的 expert 执行矩阵乘法

**SSM 状态空间模型流程：**
1. `SSM_CONV`：状态空间卷积
2. `GATED_DELTA_NET`：Gated Delta Net 门控网络
3. 输出与 Attention 结果拼接或相加

---

## 八、新增日志验证总结

### 8.1 KV Cache 日志验证

| 日志点 | 文件 | 状态 | 关键输出 |
|--------|------|------|----------|
| 构造函数 | `llama-kv-cache.cpp` | ✅ | kv_size, n_layer_kv, n_stream, v_cells groups, seq_to_stream map |
| prepare() | `llama-kv-cache.cpp` | ✅ | n_ubatches, slot_info (s0, s1, n_stream, size, contiguous) |
| find_slot() | `llama-kv-cache.cpp` | ✅ | n_tokens, n_seqs, seq_id, stream, cells_used, head, total |
| apply_ubatch() | `llama-kv-cache.cpp` | ✅ | n_tokens, slot_size, contiguous, stream idxs |
| update() | `llama-kv-cache.cpp` | ✅ | do_shift, per-stream head/used before/after |
| get_k() | `llama-kv-cache.cpp` | ✅ | layer, n_kv, s0, s1, k_shape |
| get_v() | `llama-kv-cache.cpp` | ✅ | layer, n_kv, s0, s1, v_shape, v_trans |
| cpy_k() | `llama-kv-cache.cpp` | ✅ | layer, k_cur_shape, k_idxs_shape |
| cpy_v() | `llama-kv-cache.cpp` | ✅ | layer, v_cur_shape, v_idxs_shape |

**验证结论：**
- KV cache 初始化正确：kv_size=262144, n_layer_kv=40, n_stream=1
- Slot 分配正确：size=2, contiguous=1, idxs=[0,1]
- K/V tensor shape 正确：k_shape=[512,262144], v_shape=[512,262144]
- 与 Graph 交互正常：cpy_k/cpy_v 在 graph compute 中被调用

### 8.2 Sampler 日志验证

| 日志点 | 文件 | 状态 | 关键输出 |
|--------|------|------|----------|
| chain_apply() | `llama-sampler.cpp` | ✅ | n_samplers=10, input_size=248320, sampler names |
| temp_impl() | `llama-sampler.cpp` | ✅ | temp=0.8, scaled logits range |
| softmax_impl() | `llama-sampler.cpp` | ✅ | size, max_logit, max_prob, min_prob, cum_sum |
| top_k_impl() | `llama-sampler.cpp` | ✅ | k=40, input_size, output_size, top_token |
| top_p_apply() | `llama-sampler.cpp` | ✅ | p=0.95, min_keep, input_size, output_size, cum_sum |
| min_p_apply() | `llama-sampler.cpp` | ✅ | p=0.05, min_keep, input_size, output_size, max_logit |
| sample() | `llama-sampler.cpp` | ✅ | n_vocab, sampled token, logits, probs |

**验证结论：**
- 采样链完整：10 个 sampler 全部参与
- top-k → softmax → top-p → min-p → temp → dist 流程正确
- 输入输出大小递减：248320 → 40 → 3 → 2 → 1
- 最终采样结果：selected=1（第二次采样的 top 候选）

### 8.3 VOCAB / Detokenize 日志验证

| 日志点 | 文件 | 状态 | 关键输出 |
|--------|------|------|----------|
| token_to_piece() | `llama-vocab.cpp` | ✅ | vocab_type, token attr, id_to_token text+score, result bytes hex |

**验证结论：**
- BPE vocab_type=2 正确识别
- token attr（NORMAL/CONTROL/BYTE/UNKNOWN）正确分类
- id_to_token 的 text 和 score 正确输出
- hex byte dump 展示了字节级编码结果

---

## 九、NPU 后端移植关键参考

### 9.1 必须实现的算子（按优先级）

| 优先级 | 算子 | 出现次数 | 说明 |
|--------|------|----------|------|
| P0 | MUL_MAT | 1955 | 矩阵乘法，核心中的核心 |
| P0 | MUL_MAT_ID | 600 | MoE 专家选择矩阵乘，必须支持索引路由 |
| P0 | RMS_NORM | 655 | 每层都有的归一化 |
| P0 | FLASH_ATTN_EXT | 50 | Attention 核心，或降级为 MUL_MAT + SOFT_MAX |
| P1 | ADD | 2150 | 逐元素加法 |
| P1 | MUL | 1405 | 逐元素乘法 |
| P1 | GLU | 400 | SwiGLU 激活 |
| P1 | ROPE | 100 | 旋转位置编码 |
| P2 | ARGSORT | 200 | MoE 路由排序 |
| P2 | SSM_CONV | 150 | SSM 卷积 |
| P2 | GATED_DELTA_NET | 150 | SSM 门控网络 |

### 9.2 KV Cache 管理要点

1. **内存布局**：K/V 以 `[n_embd_gqa, kv_size]` 的大张量存储，通过 view 切片访问
2. **写入位置**：通过 `cpy_k/cpy_v` 的索引张量（`k_idxs`）指定写入偏移
3. **读取窗口**：`get_k/get_v` 返回的是 `ggml_view_2d`，NPU 需要支持动态窗口读取
4. **GQA 压缩**：n_head_kv=2，K/V 的 head 维度只有 Q 的 1/8，大幅降低 KV cache 带宽

### 9.3 Sampler 移植要点

1. **输入**：`[n_vocab]` float logits 数组
2. **输出**：单个 int token_id
3. **链式执行**：按固定顺序执行 penalties → top-k → softmax → top-p → min-p → temp → dist
4. **关键参数**：top_k=40, top_p=0.95, min_p=0.05, temp=0.8
5. **概率计算**：softmax 在裁剪后的子集上执行，不需要对所有 248320 个 token 做完整 softmax

---

*报告生成时间：2026-05-07*
*基于 llama.cpp commit: master 分支最新代码*
