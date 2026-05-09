# llama.cpp 推理全流程详解

> **目标**：从模型加载到 token 生成的完整技术路径
> **版本**：llama.cpp build 9010 (commit d05fe1d7d)
> **日期**：2026-05-04

---

## 5. 模型加载全流程

### 5.1 阶段1：GGUF 文件解析

**入口函数**：`llama_model_load_from_file()` → `llama_model_load_from_file_impl()`

```cpp
// 步骤1：创建 loader
llama_model_loader loader(fname, use_mmap);

// 步骤2：读取元数据
loader.load_meta_data();
//   - 读取 magic "GGUF"
//   - 读取 version (uint32)
//   - 读取 n_tensors (int64)
//   - 读取 n_kv (int64)
//   - 遍历所有 KV 对：
//       key: "general.architecture", value: "qwen35moe"
//       key: "general.name", value: "Qwen3.6-35B-A3B"
//       key: "qwen35moe.block_count", value: 40
//       key: "qwen35moe.context_length", value: 262144
//       ... 共 54 个 KV 对

// 步骤3：读取张量信息
loader.load_tensor_infos();
//   - 对每个张量：
//       name: "token_embd.weight"
//       type: GGML_TYPE_Q4_K
//       dims: [248320, 2048]
//       offset: 在 data blob 中的偏移
```

### 5.2 阶段2：超参数加载

**入口**：`llama_model::load_hparams()`

```cpp
// 从 GGUF KV 中读取架构特定参数
hparams.n_layer = get_key(LLM_KV_BLOCK_COUNT);           // 40
hparams.n_embd  = get_key(LLM_KV_EMBEDDING_LENGTH);      // 2048
hparams.n_head  = get_key(LLM_KV_ATTENTION_HEAD_COUNT);  // 16
hparams.n_head_kv = get_key(LLM_KV_ATTENTION_HEAD_COUNT_KV); // 2
hparams.n_expert = get_key(LLM_KV_EXPERT_COUNT);         // 256
hparams.n_expert_used = get_key(LLM_KV_EXPERT_USED_COUNT); // 8
hparams.ssm_d_inner = get_key(LLM_KV_SSM_INNER_SIZE);    // 4096
hparams.ssm_d_state = get_key(LLM_KV_SSM_STATE_SIZE);    // 128
hparams.ssm_dt_rank = get_key(LLM_KV_SSM_TIME_STEP_RANK); // 32
hparams.ssm_n_group = get_key(LLM_KV_SSM_GROUP_COUNT);   // 16
// ...
```

### 5.3 阶段3：张量创建与权重映射

**入口**：`llama_model::load_tensors()`

```cpp
// 对每个层 i (0 ~ n_layer-1)
for (int i = 0; i < n_layer; i++) {
    // 创建 Attention 权重张量
    layer[i].attn_q = ggml_new_tensor_2d(ctx, type, n_embd, n_head * head_k_dim);
    layer[i].attn_k = ggml_new_tensor_2d(ctx, type, n_embd, n_head_kv * head_k_dim);
    layer[i].attn_v = ggml_new_tensor_2d(ctx, type, n_embd, n_head_kv * head_v_dim);
    layer[i].attn_o = ggml_new_tensor_2d(ctx, type, n_head * head_k_dim, n_embd);
    
    // Q/K RMS Norm（Qwen 特有）
    layer[i].attn_q_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_k_dim);
    layer[i].attn_k_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_k_dim);
    
    // Attention Gate（Qwen3.5 特有）
    layer[i].attn_gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    
    // Delta Net SSM 参数（Qwen3.5 特有）
    layer[i].ssm_conv1d = ggml_new_tensor_2d(ctx, type, ssm_d_inner, ssm_d_conv);
    layer[i].ssm_alpha  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ssm_n_group);
    layer[i].ssm_beta   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ssm_n_group);
    layer[i].ssm_a      = ggml_new_tensor_2d(ctx, type, ssm_d_state, ssm_d_inner);
    layer[i].ssm_dt     = ggml_new_tensor_1d(ctx, type, ssm_d_inner);
    
    // MoE FFN 参数
    layer[i].ffn_gate_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_embd);
    layer[i].ffn_gate_exps = ggml_new_tensor_3d(ctx, type, n_embd, n_ff_exp, n_expert);
    layer[i].ffn_up_exps   = ggml_new_tensor_3d(ctx, type, n_embd, n_ff_exp, n_expert);
    layer[i].ffn_down_exps = ggml_new_tensor_3d(ctx, type, n_ff_exp, n_embd, n_expert);
    
    // Shared Expert
    layer[i].ffn_gate_shexp = ggml_new_tensor_2d(ctx, type, n_embd, n_ff_shexp);
    layer[i].ffn_up_shexp   = ggml_new_tensor_2d(ctx, type, n_embd, n_ff_shexp);
    layer[i].ffn_down_shexp = ggml_new_tensor_2d(ctx, type, n_ff_shexp, n_embd);
}
```

**权重映射方式：**

| 方式 | 说明 | 使用场景 |
|------|------|----------|
| **mmap** | `mmap()` 映射文件到虚拟内存，按需加载 | 默认，加载最快 |
| **direct_io** | `read()` 直接读取到内存 | `--direct-io` 启用 |
| **copy** | 从 mmap 复制到后端专用缓冲区 | GPU offload 时需要 |

**mmap 工作流程：**

```
1. open(model.gguf, O_RDONLY)
2. mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0)
3. 张量 data 指针直接指向 mmap 区域中的对应偏移
4. 操作系统按需分页加载（page fault 时才从磁盘读入物理内存）
5. 优势：
   - 加载几乎瞬时完成（只需映射，不用复制数据）
   - 不使用的页面可被 OS 换出（节省物理内存）
   - 多进程共享同一物理页面
```

### 5.4 阶段4：GPU Offload 决策

**入口**：`llama_context` 创建时的 `common_params_fit_impl()`

```cpp
// 分析设备可用内存
for each device:
    total_mem = device.get_memory()
    free_mem  = device.get_free_memory()
    
// 计算模型内存需求
model_size = sum(tensor_sizes)  // ~21GB for Qwen3.6 Q4_K

// 根据 -ngl 参数决定 offload 层数
if ngl == 0:
    // 全部放在 CPU
    offload_layers = 0
elif ngl == "auto":
    // 自动计算：能放下多少层就放多少层
    while (free_mem - model_size_per_layer * layers > margin):
        offload_layers++
else:
    offload_layers = ngl  // 用户指定

// 分配张量到后端
for each layer i:
    if i < offload_layers:
        layer[i].*->buffer = gpu_backend->alloc_buffer()
    else:
        layer[i].*->buffer = cpu_backend->alloc_buffer()

// 拷贝权重数据（如果需要）
if mmap && offload to GPU:
    backend->tensor_set(layer[i].attn_q, mmap_data + offset)
```

### 5.5 阶段5：上下文初始化

**入口**：`llama_new_context_with_model()`

```cpp
// 创建 llama_context
ctx = new llama_context();

// 1. 初始化后端调度器
ctx->sched = ggml_backend_sched_new(backends, backend_bufs, n_backends, max_nodes);

// 2. 初始化 KV Cache
ctx->kv_cache = llama_kv_cache_init(ctx, hparams);
//   - 分配 K/V 缓冲区
//   - n_embd_k_gqa * n_ctx * n_layers 字节
//   - n_embd_v_gqa * n_ctx * n_layers 字节

// 3. 预分配计算图缓冲区
ctx->buf_compute[0] = ggml_backend_buffer_alloc(ctx->sched, ...);
ctx->buf_compute[1] = ggml_backend_buffer_alloc(ctx->sched, ...);
//   双缓冲：一个用于当前计算，一个用于下一次

// 4. 分配 logits 输出缓冲区
ctx->logits = ggml_backend_buffer_alloc(ctx->sched, n_vocab * sizeof(float));
```

### 5.6 量化格式与转换工具链

llama.cpp 支持从 HuggingFace 模型转换为 GGUF 格式，并提供多种量化精度选择。

#### 5.6.1 量化格式对比

| 格式 | 位宽 | 块大小 | 精度 | 速度 | 适用场景 |
|------|------|--------|------|------|----------|
| `f32` | 32-bit | - | 最高 | 最慢 | 训练/调试 |
| `f16` | 16-bit | - | 高 | 慢 | 高精度推理 |
| `bf16` | 16-bit | - | 中高 | 慢 | 兼容训练格式 |
| `q8_0` | 8-bit | 32 | 很高 | 快 |  Attention 权重推荐 |
| `q5_K` | 5-bit | 256 | 高 | 很快 | 平衡选择 |
| `q4_K` | 4-bit | 256 | 中高 | 最快 | 通用推理推荐 |
| `q3_K` | 3-bit | 256 | 中 | 最快 | 低显存设备 |
| `q2_K` | 2-bit | 256 | 较低 | 最快 | 极限压缩 |
| `iq4_nl` | 4-bit | 32 | 高 | 很快 | 非均匀量化 |

**混合量化策略：**

实际 GGUF 模型通常对不同张量使用不同精度：

```
Qwen3.6-35B-A3B-UD-Q4_K_M.gguf 的量化分配：
├── token_embd.weight         → q4_K    (词汇表大，量化收益高)
├── output.weight             → q8_0    (输出层对精度敏感)
├── attn_qkv.weight (每层)    → q8_0    (Attention 权重需高精度)
├── attn_output.weight (每层) → q8_0    (Attention 输出投影)
├── ffn_gate_exps.weight      → q4_K    (MoE 专家数量多)
├── ffn_up_exps.weight        → q4_K
├── ffn_down_exps.weight      → q5_K    (down 投影精度影响更大)
├── ssm_conv1d.weight         → q8_0    (SSM 权重精度敏感)
└── norm 权重                  → f32     (归一化权重绝对不准)
```

#### 5.6.2 转换工具链

**主转换脚本：** `convert_hf_to_gguf.py`

```bash
# 1. 安装依赖
pip install -e ./gguf-py

# 2. 基本转换（自动选择量化）
python convert_hf_to_gguf.py \
  /path/to/hf_model \
  --outfile model-f16.gguf \
  --outtype f16

# 3. 转换为 Q4_K_M（推荐通用格式）
python convert_hf_to_gguf.py \
  /path/to/hf_model \
  --outfile model-q4_k_m.gguf \
  --outtype q4_k_m

# 4. 使用 imatrix 提升量化质量（需要校准数据）
python convert_hf_to_gguf.py \
  /path/to/hf_model \
  --outfile model-imat.gguf \
  --outtype q4_k_s \
  --imatrix imatrix.dat
```

**`gguf-py` 包结构：**

```
gguf-py/gguf/
├── __init__.py
├── gguf_reader.py        # GGUF 文件读取
├── gguf_writer.py        # GGUF 文件写入
├── constants.py          # MODEL_ARCH, MODEL_TENSORS 枚举定义
├── tensor_mapping.py     # HF 张量名 → GGUF 张量名映射
├── quants.py             # 量化/反量化实现
├── utility.py            # 辅助函数
└── pyproject.toml        # 包配置（PEP 621 / Poetry）
```

**关键常量定义**（`gguf-py/gguf/constants.py`）：

```python
# 模型架构枚举
class MODEL_ARCH(IntEnum):
    LLAMA        = auto()
    QWEN         = auto()
    QWEN2        = auto()
    QWEN2MOE     = auto()
    QWEN3        = auto()
    QWEN3MOE     = auto()
    QWEN35       = auto()
    QWEN35MOE    = auto()   # ← Qwen3.5/3.6 使用的架构
    # ... 共 100+ 种

# 张量名称枚举
class TENSOR_NAMES(IntEnum):
    TOKEN_EMBD        = auto()
    OUTPUT_NORM       = auto()
    OUTPUT            = auto()
    ATTN_Q            = auto()
    ATTN_K            = auto()
    ATTN_V            = auto()
    ATTN_QKV          = auto()    # ← Qwen3.5 融合 QKV
    ATTN_OUTPUT       = auto()
    FFN_GATE_INP      = auto()    # ← MoE 路由
    FFN_GATE_EXPS     = auto()
    FFN_UP_EXPS       = auto()
    FFN_DOWN_EXPS     = auto()
    FFN_GATE_SHEXP    = auto()    # ← Shared Expert
    SSM_CONV1D        = auto()    # ← Delta Net
    # ...
```

**张量映射**（`gguf-py/gguf/tensor_mapping.py`）：

```python
# HF 权重名 → GGUF 张量名的映射
class TensorNameMap:
    mapping = {
        MODEL_ARCH.QWEN35MOE: {
            "model.embed_tokens.weight":                    TENSOR_NAMES.TOKEN_EMBD,
            "model.layers.{bid}.self_attn.qkv_proj.weight": TENSOR_NAMES.ATTN_QKV,
            "model.layers.{bid}.mlp.gate.weight":           TENSOR_NAMES.FFN_GATE_INP,
            "model.layers.{bid}.mlp.experts.{eid}.gate_proj.weight": TENSOR_NAMES.FFN_GATE_EXPS,
            "model.layers.{bid}.mlp.experts.{eid}.up_proj.weight":   TENSOR_NAMES.FFN_UP_EXPS,
            "model.layers.{bid}.mlp.experts.{eid}.down_proj.weight": TENSOR_NAMES.FFN_DOWN_EXPS,
            "model.layers.{bid}.ssm.conv1d.weight":         TENSOR_NAMES.SSM_CONV1D,
            # ...
        }
    }
```

#### 5.6.3 量化工具 `llama-quantize`

```bash
# 对已有 GGUF 进行重新量化
./build/bin/llama-quantize \
  model-f16.gguf \
  model-q4_k_m.gguf \
  q4_k_m

# 使用 imatrix（重要性矩阵）优化量化
# 1. 先生成 imatrix
./build/bin/llama-imatrix \
  -m model-f16.gguf \
  -f calibration.txt \
  -o imatrix.dat

# 2. 再用 imatrix 量化
./build/bin/llama-quantize \
  --imatrix imatrix.dat \
  model-f16.gguf \
  model-q4_k_m_imat.gguf \
  q4_k_m
```

**imatrix 原理：**
- 在代表数据上运行模型，统计每个权重对输出的影响
- 对"重要"权重分配更多 bit，对"不重要"权重分配更少 bit
- 通常可将量化损失（perplexity 增加）从 ~5% 降到 ~1%

---

## 6. Tokenization 流程

### 6.1 分词器架构

`src/llama-vocab.cpp` 实现了多种分词算法：

| 类型 | 说明 | 模型示例 |
|------|------|----------|
| `SPM` (SentencePiece) | BPE + Unigram 混合 | Llama, Qwen1 |
| `BPE` (Byte-Pair Encoding) | 最常用 | GPT-2/3/4, Qwen2/3, Mistral |
| `WPM` (WordPiece) | BERT 风格 | BERT, T5 |
| `UGM` (Unigram) | SentencePiece 纯 Unigram | XLM-R |
| `RWKV` | RWKV 专用 | RWKV |
| `Plamo2` | Plamo 专用 | Plamo |

Qwen3.6 使用 **BPE** 分词器，预处理方式为 `LLAMA_VOCAB_PRE_TYPE_QWEN35`。

### 6.2 BPE 分词流程

```cpp
// 入口：llama_tokenize()
// 输入："你好，请简单自我介绍一下"
// 输出：token ID 数组

步骤1：文本预处理
  "你好，请简单自我介绍一下"
  → 添加前缀（如果需要）
  → Unicode 规范化（NFKC）
  → Byte-fallback 处理

步骤2：初始切分（Pre-tokenization）
  按正则表达式切分成"词"：
  ["你好", "，", "请", "简单", "自我", "介绍", "一下"]

步骤3：BPE 合并
  对每"词"应用 BPE 合并规则：
  
  "你好" → 查词表
    如果 "你好" 在词表中 → [token_id_你好]
    否则拆成子词 → ["你", "好"] → [token_id_你, token_id_好]
  
  "自我" → 可能拆成 ["自", "我"] 或找到子词组合

步骤4：后处理
  添加特殊 token（BOS/EOS）
  最终输出：[bos_id, id_你, id_好, id_，, id_请, id_简, id_单, id_自, id_我, id_介, id_绍, id_一, id_下, eos_id]
```

### 6.3 Qwen3.6 词表结构

```
n_vocab: 248,320
n_merges: 247,587

特殊 Token：
- BOS (248044): <|endoftext|>
- EOS (248046): <|im_end|>
- PAD (248055): <|vision_pad|>
- FIM_PREFIX (248060): <|fim_prefix|>
- FIM_MIDDLE (248061): <|fim_middle|>
- FIM_SUFFIX (248062): <|fim_suffix|>
- FIM_PAD (248063): <|fim_pad|>
- REPO_NAME (248064): <|repo_name|>
- FILE_SEP (248065): <|file_sep|>
```

---

## 7. 推理引擎：从计算图到 Attention

### 7.1 llama_decode() 总览

`llama_decode()` 是推理的核心入口。它处理一个 batch 的 token，执行前向传播，产生 logits。

```cpp
int llama_decode(llama_context * ctx, llama_batch batch) {
    // 1. 参数校验
    // 2. 将 batch 拆分为 ubatch (micro-batch)
    // 3. 对每个 ubatch：
    //      a. 构建计算图
    //      b. 执行计算图
    //      c. 提取 logits
    // 4. 合并结果
}
```

### 7.2 Batch 到 UBatch 的拆分

```cpp
llama_batch batch;
batch.n_tokens = N;           // 输入 token 数量
batch.token = token_ids;      // token ID 数组
batch.pos = positions;        // 位置编码
batch.n_seq_id = ...;         // 序列 ID（多序列并行）

// 拆分为 micro-batch（受限于物理 batch size -ub）
llama_ubatch ubatch;
ubatch.n_tokens = min(N, max_ubatch);
ubatch.token = &batch.token[0];
ubatch.pos = &batch.pos[0];
```

### 7.3 计算图构建

**入口**：`llama_model::build_graph()` → `llm_build_qwen35moe()`

```cpp
// 对于 Qwen3.5 MoE，计算图构建在 src/models/qwen35moe.cpp

class llm_build_qwen35moe : public llm_graph_context {
public:
    ggml_cgraph * build() {
        // 1. 输入嵌入
        ggml_tensor * inpL = build_inp_embd();  // token → embedding
        
        // 2. 遍历每一层
        for (int il = 0; il < n_layer; il++) {
            // 判断该层是 Full Attention 还是 Linear Attention
            if (is_recurrent_layer(il)) {
                // Linear Attention (Gated Delta Net)
                inpL = build_layer_attn_linear(ctx, inpL, il);
            } else {
                // Full Self-Attention
                inpL = build_layer_attn(ctx, inpL, il);
            }
            
            // FFN (MoE)
            inpL = build_layer_ffn(ctx, inpL, il);
        }
        
        // 3. 最终 LayerNorm
        inpL = build_norm(ctx, inpL, model.output_norm, ...);
        
        // 4. 输出投影（logits）
        ggml_tensor * logits = build_inp_out_logits(ctx, inpL);
        
        return graph;
    }
};
```

### 7.4 Full Self-Attention 层（标准 Transformer）

```
输入: x (n_tokens, n_embd)

步骤1: 计算 Q, K, V
  q = x @ W_q^T    → (n_tokens, n_head * head_k_dim)
  k = x @ W_k^T    → (n_tokens, n_head_kv * head_k_dim)
  v = x @ W_v^T    → (n_tokens, n_head_kv * head_v_dim)

步骤2: Q/K RMS Norm（Qwen 特有）
  q = rms_norm(q) * W_q_norm
  k = rms_norm(k) * W_k_norm

步骤3: RoPE 位置编码
  q = rope(q, positions)    # 应用旋转位置编码
  k = rope(k, positions)

步骤4: KV Cache 存储
  k_cache[layer][position] = k
  v_cache[layer][position] = v

步骤5: Attention 计算
  scores = q @ k^T / sqrt(head_k_dim)   # (n_tokens, n_past + n_tokens)
  scores = softmax(scores, mask)         # 应用因果掩码
  out = scores @ v                       # (n_tokens, n_head_kv * head_v_dim)

步骤6: 输出投影
  out = out @ W_o^T      # (n_tokens, n_embd)
  
步骤7: Attention Gate（Qwen3.5 特有）
  gate = sigmoid(x @ W_gate)
  out = gate * out       # 逐元素门控

步骤8: 残差连接
  x = x + out
```

### 7.5 Gated Delta Net Linear Attention（Qwen3.5 特有）

对于部分层，Qwen3.5 使用**线性注意力**替代标准自注意力，以降低计算复杂度：

```
输入: x (n_tokens, n_embd)

步骤1: QKV 混合投影
  qkv = x @ W_qkv^T     # (n_tokens, 3 * ssm_d_inner)
  q, k, v = split(qkv)  # 各 (n_tokens, ssm_d_inner)

步骤2: 1D 卷积（局部上下文）
  q = conv1d(q, W_conv) # 沿序列维度的一维卷积
  k = conv1d(k, W_conv)

步骤3: L2 归一化
  q = l2_norm(q)
  k = l2_norm(k)

步骤4: Delta Net 递推
  # 维护一个隐藏状态 S (ssm_d_state, ssm_d_inner)
  # 对每个 token t：
    delta = softplus(W_dt @ x[t] + b_dt)    # 时间步衰减因子
    alpha = W_alpha * delta                  # 门控参数
    beta  = W_beta  * delta
    
    S = alpha * S + outer(k[t], v[t])        # 状态更新
    out[t] = q[t] @ S                        # 输出

步骤5: 门控 RMS Norm
  z = sigmoid(x @ W_z)     # z-gate
  out = rms_norm(out) * z  # 门控归一化

步骤6: 残差连接
  x = x + out
```

**复杂度对比：**
- 标准 Attention: O(n² · d) — 与序列长度平方成正比
- Linear Attention: O(n · d²) — 与序列长度线性成正比

对于长序列（262K 上下文），线性注意力显著更快。

### 7.6 MoE FFN 层（Qwen3.5 MoE 特有）

```
输入: x (n_tokens, n_embd)

步骤1: Router 计算路由权重
  router_logits = x @ W_gate_inp^T     # (n_tokens, n_expert)
  router_probs = softmax(router_logits) # (n_tokens, n_expert)
  
  # Top-K 专家选择
  top_k_probs, top_k_indices = topk(router_probs, n_expert_used)  # K=8
  
步骤2: 对每个 token，只计算选中的 K 个专家
  for each token t:
    out_t = 0
    for each selected expert e in top_k_indices[t]:
      # SwiGLU: gate = silu(x @ W_gate[e]^T) * (x @ W_up[e]^T)
      gate = silu(x[t] @ W_gate_exps[e]^T)
      up   = x[t] @ W_up_exps[e]^T
      ffn_out = (gate * up) @ W_down_exps[e]^T
      out_t += top_k_probs[t][e] * ffn_out

步骤3: Shared Expert（共享专家）
  shared_gate = sigmoid(x @ W_gate_inp_shexp)
  shared_out = SwiGLU(x, W_gate_shexp, W_up_shexp, W_down_shexp)
  out += shared_gate * shared_out

步骤4: 残差连接
  x = x + out
```

### 7.7 KV Cache 管理

```cpp
// KV Cache 结构
struct llama_kv_cache {
    ggml_tensor * k_l[LLAMA_MAX_LAYERS];  // 每层 K 缓存
    ggml_tensor * v_l[LLAMA_MAX_LAYERS];  // 每层 V 缓存
    
    int32_t head;        // 当前写入位置
    int32_t size;        // 总容量
    int32_t used;        // 已使用单元数
    
    // 每个 cell 对应一个 token 位置
    struct llama_kv_cell {
        llama_pos pos;           // 位置编码
        llama_seq_id seq_id;     // 序列 ID
        bool has_logits;         // 是否已计算 logits
    } cells[];
};

// Prompt Processing（首次）
//   输入: [t1, t2, t3, ..., tN]
//   K_cache[0:N] = K(t1:tN)
//   V_cache[0:N] = V(t1:tN)

// Token Generation（后续）
//   输入: [t_new]
//   K_cache[N] = K(t_new)
//   V_cache[N] = V(t_new)
//   Attention 只计算 t_new 与 cache[0:N] 的关系（不重新计算旧 token）
```

### 7.8 添加新模型架构的端到端流程

llama.cpp 支持 140+ 种模型架构。当需要添加一种新架构时，需要在 Python 转换端和 C++ 推理端同时修改。

#### Step 1：Python 端 — 定义架构和张量映射

**修改 `gguf-py/gguf/constants.py`：**

```python
class MODEL_ARCH(IntEnum):
    # ... 已有架构 ...
    MYMODEL = auto()   # ← 新增

class MODEL_TENSOR(IntEnum):
    # ... 已有张量 ...
    ATTN_Q_NORM = auto()   # ← 如果新架构有特殊张量

# 为每个架构定义其包含的张量列表
MODEL_TENSOR_MAP: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    # ...
    MODEL_ARCH.MYMODEL: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_Q_NORM,   # ← 新增的特殊张量
        MODEL_TENSOR.ATTN_OUTPUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_DOWN,
    ],
}
```

**修改 `gguf-py/gguf/tensor_mapping.py`：**

```python
class TensorNameMap:
    mapping = {
        # ... 已有映射 ...
        MODEL_ARCH.MYMODEL: {
            "model.embed_tokens.weight":              TENSOR_NAMES.TOKEN_EMBD,
            "model.layers.{bid}.self_attn.q_proj.weight": TENSOR_NAMES.ATTN_Q,
            "model.layers.{bid}.self_attn.k_proj.weight": TENSOR_NAMES.ATTN_K,
            "model.layers.{bid}.self_attn.v_proj.weight": TENSOR_NAMES.ATTN_V,
            "model.layers.{bid}.self_attn.q_norm.weight": TENSOR_NAMES.ATTN_Q_NORM,  # ← 新张量
            "model.layers.{bid}.self_attn.o_proj.weight": TENSOR_NAMES.ATTN_OUTPUT,
            "model.layers.{bid}.mlp.gate_proj.weight":    TENSOR_NAMES.FFN_GATE,
            "model.layers.{bid}.mlp.up_proj.weight":      TENSOR_NAMES.FFN_UP,
            "model.layers.{bid}.mlp.down_proj.weight":    TENSOR_NAMES.FFN_DOWN,
        },
    }
```

**修改 `convert_hf_to_gguf.py`：**

```python
@ModelBase.register("MyModelForCausalLM")
class MyModel(ModelBase):
    model_arch = gguf.MODEL_ARCH.MYMODEL

    def set_vocab(self):
        # 定义分词器类型和特殊 token
        self._set_vocab_bpe()  # 或 _set_vocab_spm(), _set_vocab_wpm()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_context_length(hparams.get("max_position_embeddings", 4096))
        self.gguf_writer.add_embedding_length(hparams["hidden_size"])
        self.gguf_writer.add_block_count(hparams["num_hidden_layers"])
        self.gguf_writer.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams.get("num_key_value_heads", hparams["num_attention_heads"]))
        # 如果有自定义参数：
        self.gguf_writer.add_key_value("mymodel.q_norm_eps", hparams.get("q_norm_eps", 1e-6))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # 处理权重的特殊转换（如转置、拆分融合张量）
        # 返回 [(gguf_tensor_name, tensor_data), ...]
        return [(self.map_tensor_name(name), data_torch)]
```

#### Step 2：C++ 端 — 注册架构

**修改 `src/llama-arch.h`：**

```cpp
enum llm_arch {
    LLM_ARCH_UNKNOWN,
    LLM_ARCH_LLAMA,
    // ... 已有架构 ...
    LLM_ARCH_MYMODEL,   // ← 新增
    LLM_ARCH_COUNT
};
```

**修改 `src/llama-arch.cpp`：**

```cpp
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_UNKNOWN,  "unknown"  },
    { LLM_ARCH_LLAMA,    "llama"    },
    // ...
    { LLM_ARCH_MYMODEL,  "mymodel"  },  // ← 新增（与 GGUF 中 general.architecture 对应）
};

// 定义该架构的超参数键名
static const std::map<llm_arch, std::map<llm_kv, const char *>> LLM_KV_NAMES = {
    {
        LLM_ARCH_MYMODEL,
        {
            { LLM_KV_ATTENTION_HEAD_COUNT,          "%s.attention.head_count" },
            { LLM_KV_ATTENTION_HEAD_COUNT_KV,       "%s.attention.head_count_kv" },
            { LLM_KV_ATTENTION_LAYERNORM_EPS,       "%s.attention.layer_norm_epsilon" },
            // ...
            { LLM_KV_MYMODEL_Q_NORM_EPS,            "%s.q_norm_eps" },  // ← 自定义参数
        },
    },
};

// 定义张量名称模板
static const std::map<llm_arch, std::map<llm_tensor, const char *>> LLM_TENSOR_NAMES = {
    {
        LLM_ARCH_MYMODEL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },  // ← 新张量
            { LLM_TENSOR_ATTN_OUTPUT,     "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
};
```

#### Step 3：C++ 端 — 实现图构建器

**创建 `src/models/mymodel.cpp`：**

```cpp
// mymodel.cpp
#include "llama-model.h"
#include "llama-graph.h"

// 加载超参数
static void llm_load_hparams_mymodel(llama_model & model) {
    auto & hparams = model.hparams;
    hparams.n_layer = model.find_key(LLM_KV_BLOCK_COUNT).get_i32();
    hparams.n_embd  = model.find_key(LLM_KV_EMBEDDING_LENGTH).get_i32();
    hparams.n_head  = model.find_key(LLM_KV_ATTENTION_HEAD_COUNT).get_i32();
    hparams.n_head_kv = model.find_key(LLM_KV_ATTENTION_HEAD_COUNT_KV).get_i32();
    // 读取自定义参数
    hparams.f_q_norm_eps = model.find_key(LLM_KV_MYMODEL_Q_NORM_EPS).get_f32(1e-6f);
}

// 加载张量
static void llm_load_tensors_mymodel(llama_model & model) {
    auto & ctx = model.ctx;
    auto & hparams = model.hparams;
    const int n_layer = hparams.n_layer;

    model.tok_embd = create_tensor(ctx, tn(LLM_TENSOR_TOKEN_EMBD), {n_embd, n_vocab});
    model.output_norm = create_tensor(ctx, tn(LLM_TENSOR_OUTPUT_NORM), {n_embd});
    model.output = create_tensor(ctx, tn(LLM_TENSOR_OUTPUT), {n_embd, n_vocab});

    for (int i = 0; i < n_layer; i++) {
        auto & layer = model.layers[i];
        layer.attn_q = create_tensor(ctx, tn(LLM_TENSOR_ATTN_Q, i), {n_embd, n_embd});
        layer.attn_k = create_tensor(ctx, tn(LLM_TENSOR_ATTN_K, i), {n_embd, n_embd_k_gqa});
        layer.attn_v = create_tensor(ctx, tn(LLM_TENSOR_ATTN_V, i), {n_embd, n_embd_v_gqa});
        layer.attn_q_norm = create_tensor(ctx, tn(LLM_TENSOR_ATTN_Q_NORM, i), {head_dim});  // ← 新张量
        layer.attn_o = create_tensor(ctx, tn(LLM_TENSOR_ATTN_OUTPUT, i), {n_embd, n_embd});
        layer.ffn_gate = create_tensor(ctx, tn(LLM_TENSOR_FFN_GATE, i), {n_embd, n_ff});
        layer.ffn_up = create_tensor(ctx, tn(LLM_TENSOR_FFN_UP, i), {n_embd, n_ff});
        layer.ffn_down = create_tensor(ctx, tn(LLM_TENSOR_FFN_DOWN, i), {n_ff, n_embd});
    }
}

// 构建计算图
class llm_build_mymodel : public llm_graph_context {
public:
    ggml_cgraph * build() {
        ggml_tensor * inpL = build_inp_embd();  // 输入嵌入

        for (int il = 0; il < n_layer; il++) {
            // 1. Self-Attention
            ggml_tensor * cur = build_norm(inpL, layer[il].attn_norm, hparams.f_norm_eps);

            ggml_tensor * Qcur = build_lora_mm(layer[il].attn_q, cur);
            ggml_tensor * Kcur = build_lora_mm(layer[il].attn_k, cur);
            ggml_tensor * Vcur = build_lora_mm(layer[il].attn_v, cur);

            // ← 新架构的特殊处理：Q Norm
            if (layer[il].attn_q_norm) {
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_q_norm_eps);
                Qcur = ggml_mul(ctx0, Qcur, layer[il].attn_q_norm);
            }

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

            // KV Cache 更新
            ggml_tensor * k = kv_cache.get_k(ctx0, il, n_kv);
            ggml_tensor * v = kv_cache.get_v(ctx0, il, n_kv);
            ggml_tensor * kq = ggml_mul_mat(ctx0, k, Qcur);
            kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
            ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
            ggml_tensor * cur_attn = build_lora_mm(layer[il].attn_o, kqv);

            inpL = ggml_add(ctx0, inpL, cur_attn);  // 残差连接

            // 2. FFN
            cur = build_norm(inpL, layer[il].ffn_norm, hparams.f_norm_eps);
            ggml_tensor * ffn_gate = build_lora_mm(layer[il].ffn_gate, cur);
            ggml_tensor * ffn_up = build_lora_mm(layer[il].ffn_up, cur);
            ffn_gate = ggml_silu(ctx0, ffn_gate);
            ffn_gate = ggml_mul(ctx0, ffn_gate, ffn_up);
            ggml_tensor * ffn_out = build_lora_mm(layer[il].ffn_down, ffn_gate);
            inpL = ggml_add(ctx0, inpL, ffn_out);
        }

        // 输出层
        inpL = build_norm(inpL, model.output_norm, hparams.f_norm_eps);
        ggml_tensor * logits = build_inp_out_logits(ctx0, inpL);
        return graph;
    }
};
```

#### Step 4：注册到模型分发系统

**修改 `src/llama-model.cpp` 中的分发逻辑：**

```cpp
// 在 llama_model::build_graph() 中添加分发 case
switch (arch) {
    case LLM_ARCH_LLAMA:    return std::make_unique<llm_build_llama>(*this);
    case LLM_ARCH_QWEN35MOE: return std::make_unique<llm_build_qwen35moe>(*this);
    case LLM_ARCH_MYMODEL:   return std::make_unique<llm_build_mymodel>(*this);  // ← 新增
    default:
        throw std::runtime_error("unsupported architecture");
}
```

#### Step 5：验证

```bash
# 1. 编译
cmake -B build && cmake --build build -j

# 2. 转换模型
python convert_hf_to_gguf.py /path/to/mymodel --outfile mymodel.gguf --outtype q4_k_m

# 3. 基础推理测试
./build/bin/llama-cli -m mymodel.gguf -p "Hello" -n 32

# 4. Perplexity 对比（与参考实现对比）
./build/bin/llama-perplexity -m mymodel.gguf -f test.txt

# 5. 输出一致性检查（与 transformers 库对比）
# 使用相同 seed，对比 llama.cpp 和 transformers 的 token 输出是否一致

# 6. 后端兼容性测试
./build/bin/test-backend-ops  # 确保新架构使用的算子在所有后端正确执行
```

**常见问题排查：**

| 问题 | 排查方向 |
|------|----------|
| 模型加载失败 `unknown architecture` | 检查 `LLM_ARCH_NAMES` 中名称是否与 GGUF 中 `general.architecture` 一致 |
| 张量未找到 `tensor not found` | 检查 `LLM_TENSOR_NAMES` 中的命名是否与 GGUF 中一致 |
| 输出结果与 transformers 不一致 | 检查 RMSNorm eps、RoPE 参数（freq_base）、注意力缩放因子 |
| Shape mismatch | 检查 `create_tensor` 的维度是否与 GGUF 中张量 shape 匹配 |
| 量化后 perplexity 激增 | 检查 attention 权重是否用了 q4_K（应至少 q5_K 或 q8_0）|

---

## 8. 后端执行系统

### 8.1 后端调度器（ggml_backend_sched）

调度器负责将计算图的节点分发到合适的后端执行：

```cpp
ggml_backend_sched_graph_compute_async(sched, graph);

// 内部流程：
// 1. 分割图（split graph）
//    - 根据张量所在后端，将图切分为多个子图
//    - 在子图边界插入 ggml_cpy 节点
//
// 2. 按拓扑顺序执行子图
//    for each split in splits:
//        backend = split->backend
//        ggml_backend_graph_compute_async(backend, &split->graph)
//        if not last split:
//            ggml_backend_synchronize(backend)  // 等待完成
//
// 3. 如果是最后一个 split，可能异步返回（不等待）
```

### 8.2 CPU 后端执行

```cpp
// ggml/src/ggml-cpu/ggml-cpu.c

void ggml_graph_compute(ggml_cgraph * graph) {
    // 1. 初始化线程池
    thread_pool_init(n_threads);
    
    // 2. 遍历计算节点
    for (int i = 0; i < graph->n_nodes; i++) {
        ggml_tensor * node = graph->nodes[i];
        
        // 3. 分发到具体算子实现
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_compute_forward_mul_mat(node);
                break;
            case GGML_OP_ADD:
                ggml_compute_forward_add(node);
                break;
            case GGML_OP_NORM:
                ggml_compute_forward_norm(node);
                break;
            case GGML_OP_ROPE:
                ggml_compute_forward_rope(node);
                break;
            // ... 100+ 算子
        }
    }
}

// MUL_MAT 在 CPU 上的实现（ARM NEON）
void ggml_compute_forward_mul_mat(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // weight
    const ggml_tensor * src1 = dst->src[1];  // input
    
    // 反量化（如果是量化张量）
    if (ggml_is_quantized(src0->type)) {
        // 使用 NEON 指令反量化 block
        // 每个 block 32 个权重，共享 scale
        for each block:
            scale = block_scale
            for i in 0..31:
                float_val[i] = scale * dequant(qweight[i])
    }
    
    // 矩阵乘法
    // 使用 NEON intrinsics 加速
    // 4x4 或 8x1 分块
    for m in 0..M:
        for n in 0..N:
            sum = 0
            for k in 0..K:
                sum += src0[m][k] * src1[k][n]
            dst[m][n] = sum
}
```

### 8.3 Metal 后端执行

```cpp
// ggml/src/ggml-metal/ggml-metal.cpp

void ggml_metal_graph_compute(ggml_backend * backend, ggml_cgraph * graph) {
    // 1. 编码计算图到 Metal Command Buffer
    id<MTLCommandBuffer> cmd_buf = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    
    // 2. 遍历节点，编码每个算子
    for each node in graph:
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                // 选择 Metal kernel
                // 根据矩阵大小、数据类型选择最优 kernel
                if (node->src[0]->type == GGML_TYPE_Q4_0)
                    kernel = pipeline_q4_0_mul_mat
                else if (node->src[0]->type == GGML_TYPE_Q4_K)
                    kernel = pipeline_q4_k_mul_mat
                // ...
                
                // 设置参数
                [encoder setComputePipelineState:kernel];
                [encoder setBuffer:weight_buffer offset:0 atIndex:0];
                [encoder setBuffer:input_buffer offset:0 atIndex:1];
                [encoder setBuffer:output_buffer offset:0 atIndex:2];
                
                // 设置线程组大小
                MTLSize grid = MTLSizeMake(M, N, 1);
                MTLSize threadgroup = MTLSizeMake(32, 1, 1);  // SIMD 宽度
                [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
                break;
                
            case GGML_OP_SOFT_MAX:
                kernel = pipeline_soft_max;
                // ...
                break;
        }
    
    // 3. 提交 Command Buffer
    [encoder endEncoding];
    [cmd_buf commit];
    
    // 4. 等待完成（或异步返回）
    [cmd_buf waitUntilCompleted];
}
```

### 8.4 BLAS 后端（Accelerate）

对于大矩阵的 FP32 乘法，GGML 会调用 Accelerate 框架的 `cblas_sgemm`：

```cpp
// ggml/src/ggml-blas/ggml-blas.cpp

if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
    // 当矩阵足够大时，使用 BLAS 比手写 NEON 更快
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                src0->data, K,
                src1->data, N,
                0.0f,
                dst->data, N);
}
```

### 8.5 权重重排优化（Repack）

```cpp
// --repack 启用时，量化权重被重新排列以提高 SIMD 效率

// 原始布局（Q4_K）:
// [block0][block1][block2]...
// 每个 block: [scale][min][32x4bit_weights]

// Repack 后（interleaved）:
// [scale0, scale1, scale2, scale3]  // 4 个 scale 连续
// [min0, min1, min2, min3]
// [w0_0, w0_1, w0_2, w0_3]          // 4 个 block 的第 0 个权重连续
// ...

// 这样 SIMD 可以一次加载 4 个 block 的 scale，减少指令数
```

### 8.6 核心算子 Kernel 实现深度对比

#### MUL_MAT（矩阵乘法）的实现路径

MUL_MAT 是 LLM 推理中耗时最高的算子（占 60-80% 总时间）。不同后端有截然不同的实现策略：

**CPU 后端 — NEON 优化（`ggml/src/ggml-cpu/ops.cpp`）**

```cpp
// 针对 ARM NEON 的 Q4_K 反量化 + 矩阵乘
static void ggml_compute_forward_mul_mat_q4_k_f32(...) {
    const int nb = qk / QK4_K;  // 每个 block 32 个权重
    
    // 使用 NEON 的 128-bit 寄存器
    // 一次处理 4 个 block（128 个权重）
    for (int i = 0; i < nb; i += 4) {
        // 加载 4 个 scale（float32x4_t）
        float32x4_t scales = vld1q_f32(scale_ptr + i);
        
        // 加载量化权重（int8x16_t × 2）
        uint8x16_t qs0 = vld1q_u8(qs_ptr + i * 16 + 0);
        uint8x16_t qs1 = vld1q_u8(qs_ptr + i * 16 + 16);
        
        // 将 4-bit 权重解包为 8-bit
        // 使用表查找 + 移位
        int8x16_t q0 = vreinterpretq_s8_u8(vandq_u8(qs0, vdupq_n_u8(0x0F)));
        int8x16_t q1 = vreinterpretq_s8_u8(vshrq_n_u8(qs0, 4));
        
        // 反量化为 float 并累加
        float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(q0))))), scales);
        // ... 累加到结果
    }
}

// 大矩阵（M,N,K > 阈值）走 BLAS 路径
if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && ne0 >= 128) {
    cblas_sgemm(...);  // 调用 Accelerate / OpenBLAS
}
```

**Metal 后端 — 着色器选择（`ggml/src/ggml-metal/ggml-metal.metal`）**

```metal
// Q4_K 矩阵乘 kernel
kernel void kernel_mul_mat_q4_K_f32(
    device const  void * src0,       // 量化权重 [K/32 blocks, M]
    device const float * src1,       // 输入 [K, N]
    device       float * dst,        // 输出 [M, N]
    constant   int64_t & ne00,       // K
    constant   int64_t & ne01,       // M
    constant   int64_t & ne10,       // K
    constant   int64_t & ne11,       // N
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]
) {
    // 分块策略：
    // - 每个 threadgroup 处理 32×32 的输出块
    // - 每个 thread 处理一行
    // - K 维度循环，每次加载 256 个权重（8 个 Q4_K block）
    
    const int row = tgpig.y * 32 + tpitg.y;
    const int col = tgpig.x * 32 + tpitg.x;
    
    float sum = 0.0f;
    for (int k = 0; k < ne00; k += 256) {
        // 从 device memory 加载 8 个 block 到 threadgroup memory
        // 反量化在寄存器中完成
        // 使用 SIMD shuffle 进行 warp 内归约
        ...
    }
    dst[row * ne11 + col] = sum;
}
```

**Metal 的 kernel 选择逻辑：**

```cpp
// ggml/src/ggml-metal/ggml-metal.cpp
static id<MTLComputePipelineState> get_kernel_for_op(...) {
    switch (op) {
        case GGML_OP_MUL_MAT:
            if (src0->type == GGML_TYPE_Q4_0) return pipeline_mul_mat_q4_0_f32;
            if (src0->type == GGML_TYPE_Q4_K) return pipeline_mul_mat_q4_K_f32;
            if (src0->type == GGML_TYPE_F16)  {
                // 根据矩阵大小选择不同 kernel
                if (ne00 >= 512 && ne11 >= 512) return pipeline_mul_mat_f16_f32_large;
                else return pipeline_mul_mat_f16_f32_small;
            }
            ...
    }
}
```

**CUDA 后端 — Warp-level 优化（`ggml/src/ggml-cuda/mul_mat.cu`）**

```cuda
// Q4_K 反量化 + 矩阵乘
// 使用 __shared__ memory 缓存反量化后的权重
// 每个 warp 处理 32×N 的输出块

__global__ void mul_mat_q4_K_f32_cuda(...) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // shared memory: 缓存 B 矩阵的 32×32 块
    __shared__ float sB[32][32];
    
    float acc = 0.0f;
    for (int k = 0; k < K; k += 32) {
        // 协同加载 B 到 shared memory
        sB[threadIdx.y][threadIdx.x] = B[(blockIdx.y * 32 + threadIdx.y) * K + k + threadIdx.x];
        __syncthreads();
        
        // 从 global memory 加载量化权重，在寄存器中反量化
        const block_q4_K * b = (block_q4_K *)A + (blockIdx.x * 32 + threadIdx.y) * (K/256) + k/256;
        float scale = b->d;
        uint8_t qs = b->qs[lane_id];
        float w = scale * ((qs & 0x0F) - 8);  // 反量化
        
        // 使用 warp shuffle 进行累加
        acc += w * sB[lane_id][threadIdx.x];
        __syncthreads();
    }
    
    // warp 内归约
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    
    if (lane_id == 0) {
        C[(blockIdx.x * 32 + threadIdx.y) * N + blockIdx.y * 32 + threadIdx.x] = acc;
    }
}
```

#### Flash Attention 的多后端实现

**标准 Attention vs Flash Attention 的内存差异：**

```
标准 Attention（内存密集型）：
  Q [n_embd, n_tokens] × K^T [n_tokens, n_embd] 
    → 中间矩阵 S [n_tokens, n_tokens]  （O(N²) 内存）
  S 经过 Softmax → P [n_tokens, n_tokens]
  P × V [n_embd, n_tokens] → O [n_embd, n_tokens]
  
Flash Attention（分块计算）：
  将 Q/K/V 分成小块（如 64×64）
  每次只加载一块到 SRAM/Shared Memory
  在 SRAM 内完成：Q_block × K_block^T → softmax → × V_block
  只输出最终结果，不存储中间矩阵
```

**CPU 后端 Flash Attention：**

```cpp
// ggml/src/ggml-cpu/ops.cpp
// 使用 AVX2/NEON 进行 16×16 分块
// 关键优化：避免存储 N×N 注意力矩阵
static void ggml_compute_forward_flash_attn_ext_f16(...) {
    const int block_size = 256;  // 分块大小
    
    for (int i = 0; i < n_tokens; i += block_size) {
        for (int j = 0; j < n_kv; j += block_size) {
            // 加载 Q_block, K_block, V_block 到 L1 cache
            // 计算 attention score
            // 在线进行 softmax 归一化（增量算法）
            // 累加输出
        }
    }
}
```

**CUDA Flash Attention — 多级分块（`ggml/src/ggml-cuda/fattn.cu`）**

```cuda
// 使用 CUDA 的 thread block 层次结构
// - 每个 block 处理一个 attention head
// - 每个 warp 处理一个 query token
// - K/V 缓存到 shared memory，Q 缓存到 registers

template<int D, int block_size>
__global__ void flash_attn_ext_f16(...) {
    constexpr int warps_per_block = 4;
    constexpr int threads_per_warp = 32;
    
    __shared__ half sK[block_size][D];   // Shared memory for K cache
    __shared__ half sV[block_size][D];   // Shared memory for V cache
    
    float q[D/2];  // Query vector in registers (使用 float 累加避免精度损失)
    float m = -INFINITY;  // Running max for softmax
    float l = 0.0f;       // Running sum for softmax
    float acc[D/2] = {0}; // Accumulated output
    
    // 加载 Q 到寄存器
    #pragma unroll
    for (int i = 0; i < D/2; i++) {
        q[i] = __half2float(Q[threadIdx.x * D/2 + i]);
    }
    
    // 分块遍历 K/V
    for (int kv_start = 0; kv_start < n_kv; kv_start += block_size) {
        // 协同加载 K_block, V_block
        ...
        
        // 计算 Q × K^T（单个 warp 内）
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < D/2; i++) {
            score += q[i] * __half2float(sK[threadIdx.y][i]);
        }
        score *= scale;  // 除以 sqrt(dim)
        
        // Online softmax（增量更新）
        float m_new = max(m, score);
        float l_new = l * expf(m - m_new) + expf(score - m_new);
        
        // 更新输出累加
        #pragma unroll
        for (int i = 0; i < D/2; i++) {
            acc[i] = acc[i] * (l * expf(m - m_new) / l_new) 
                   + __half2float(sV[threadIdx.y][i]) * (expf(score - m_new) / l_new);
        }
        
        m = m_new;
        l = l_new;
    }
    
    // 写回结果
    #pragma unroll
    for (int i = 0; i < D/2; i++) {
        O[threadIdx.x * D/2 + i] = __float2half(acc[i]);
    }
}
```

**Metal Flash Attention — SIMD 组优化：**

```metal
// Metal 使用 SIMD-group（32 threads）进行 warp 内通信
kernel void flash_attn_ext_f16(
    ...
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Metal 的 threadgroup memory 相当于 CUDA shared memory
    // simd_shuffle 相当于 CUDA __shfl_sync
    
    float score = 0.0f;
    for (int d = 0; d < head_dim; d += 32) {
        score += q[d + simd_lane_id] * k[d + simd_lane_id];
    }
    
    // SIMD-group 内归约
    score = simd_sum(score);  // 比 CUDA 的 shuffle 更简洁
    
    ...
}
```

#### 量化反量化的逐 block 策略

```cpp
// 推理时不是一次性全部反量化，而是逐 block 实时反量化

// Q4_K block 结构（256 个权重）
struct block_q4_K {
    ggml_half d;      // super-block scale
    ggml_half dmin;   // super-block min
    uint8_t scales[12];  // 8 个 sub-block 的 scale/min（6-bit 压缩）
    uint8_t qs[128];     // 256 个 4-bit 量化值
};

// CPU 反量化（NEON）
static inline float32x4_t dequantize_q4_K_block(const block_q4_K * b, int idx) {
    // 从压缩的 scales 数组还原 8 个 sub-block 的 scale
    float scale = decode_scale(b->scales, idx / 32);
    float min   = decode_min(b->scales, idx / 32);
    
    // 从 qs 数组提取 4-bit 值
    uint8_t q = (idx % 2 == 0) ? (b->qs[idx/2] & 0x0F) : (b->qs[idx/2] >> 4);
    
    // 还原：value = d * (q - min_d)
    return vdupq_n_f32(scale * (q - 8));  // NEON 向量
}
```

---

## 9. 采样与生成

### 9.1 采样器链

`src/llama-sampler.cpp` 实现了完整的采样器链：

```cpp
// 默认采样器链顺序：
// penalties → dry → top_n_sigma → top_k → typ_p → top_p → min_p → xtc → temperature

struct llama_sampler_chain {
    std::vector<llama_sampler *> samplers;
    
    llama_token sample(const llama_token_data_array * candidates) {
        for each sampler in chain:
            sampler->apply(candidates);
        return candidates[0].id;  // 选择概率最高的 token
    }
};
```

### 9.2 各采样器详解

| 采样器 | 作用 | 参数 |
|--------|------|------|
| **Penalties** | 重复惩罚 | `repeat_penalty`, `presence_penalty`, `frequency_penalty` |
| **DRY** | Don't Repeat Yourself | `dry_multiplier`, `dry_base`, `dry_allowed_length` |
| **Top-N-Sigma** | 基于标准差的截断 | `top_n_sigma` |
| **Top-K** | 只保留概率最高的 K 个 token | `top_k` (默认 40) |
| **Typical-P** | 局部典型采样 | `typical_p` (默认 1.0=禁用) |
| **Top-P** | 累积概率截断 | `top_p` (默认 0.95) |
| **Min-P** | 最低相对概率阈值 | `min_p` (默认 0.05) |
| **XTC** | eXtreme Token Compression | `xtc_probability`, `xtc_threshold` |
| **Temperature** | 概率分布平滑 | `temperature` (默认 0.8) |
| **Mirostat** | 自适应困惑度控制 | `mirostat` (0=禁用, 1/2) |

### 9.3 采样流程

```
输入: logits (n_vocab 个浮点数)

步骤1: 转换为概率分布
  probs = softmax(logits / temperature)

步骤2: 应用惩罚
  for each token in recent_history:
      probs[token] *= repeat_penalty  // 降低重复 token 的概率

步骤3: Top-K 截断
  保留概率最高的 top_k 个 token，其余设为 0

步骤4: Top-P 截断
  按概率从高到低累加，保留累积概率 <= top_p 的最小集合

步骤5: Min-P 截断
  删除概率 < (max_prob * min_p) 的 token

步骤6: 重新归一化
  probs = normalize(probs)

步骤7: 采样
  token_id = multinomial_sample(probs)
  
输出: 选中的 token_id
```

---


---

## 11. Qwen3.6-35B-A3B 运行全流程拆解

现在，我们将以上所有知识串联起来，完整追踪 **"用户在 llama-cli 中输入 '你好，请简单自我介绍一下'"** 的每一步技术细节。

### 11.1 阶段0：程序启动

```bash
./build/bin/llama-cli \
  -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -p "你好，请简单自我介绍一下" \
  -n 256 --temp 0.7 -ngl 0 -c 4096 \
  --reasoning off --single-turn
```

**执行流程：**

```
1. main() 入口 (tools/cli/cli.cpp)
   
2. 解析命令行参数 (common/arg.cpp)
   - 识别 -m → model_path
   - 识别 -p → prompt = "你好，请简单自我介绍一下"
   - 识别 -n → n_predict = 256
   - 识别 --temp → temperature = 0.7
   - 识别 -ngl 0 → n_gpu_layers = 0
   - 识别 --reasoning off → reasoning = false
   
3. 初始化日志系统
   
4. 加载模型
   llama_model_load_from_file()
   
5. 创建上下文
   llama_new_context_with_model()
   
6. Tokenize 提示词
   llama_tokenize()
   
7. 推理循环
   llama_decode() → llama_sampler_sample() → 输出 token
   
8. Detokenize 输出
   llama_detokenize()
   
9. 打印结果并退出
```

### 11.2 阶段1：模型加载（~30-60 秒）

```
[llama_model_load_from_file_impl]
│
├─▶ [llama_model_loader] 打开 Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
│   ├─▶ 读取 magic "GGUF"，version=3
│   ├─▶ 读取 54 个 KV 元数据
│   │   ├─▶ general.architecture = "qwen35moe"
│   │   ├─▶ general.name = "Qwen3.6-35B-A3B"
│   │   ├─▶ qwen35moe.block_count = 40
│   │   ├─▶ qwen35moe.context_length = 262144
│   │   ├─▶ qwen35moe.expert_count = 256
│   │   ├─▶ qwen35moe.expert_used_count = 8
│   │   └─▶ ... (共 54 个)
│   ├─▶ 读取 733 个张量信息
│   │   ├─▶ token_embd.weight: Q4_K, [248320, 2048]
│   │   ├─▶ blk.0.attn_q.weight: Q4_K, [2048, 4096]
│   │   ├─▶ blk.0.attn_k.weight: Q4_K, [2048, 512]
│   │   ├─▶ blk.0.ssm_conv1d.weight: Q4_K, [4096, 4]
│   │   ├─▶ blk.0.ffn_gate_exps.weight: Q4_K, [2048, 512, 256]
│   │   └─▶ ... (共 733 个)
│   └─▶ mmap 文件到虚拟内存 (~21GB)
│
├─▶ [llama_model::load_hparams]
│   └─▶ 解析超参数到 llama_hparams 结构
│
├─▶ [llama_model::load_tensors]
│   └─▶ 遍历 40 层，为每层创建 ggml_tensor
│       ├─▶ Attention: W_q, W_k, W_v, W_o, q_norm, k_norm, gate
│       ├─▶ Delta Net: ssm_conv1d, ssm_alpha, ssm_beta, ssm_a, ssm_dt
│       ├─▶ MoE FFN: gate_inp, gate_exps, up_exps, down_exps
│       └─▶ Shared Expert: gate_shexp, up_shexp, down_shexp
│
├─▶ [内存分配]
│   └─▶ 由于 -ngl 0，所有张量分配到 CPU 后端缓冲区
│       ├─▶ CPU_Mapped buffer: 20700.80 MiB (mmap 映射)
│       └─▶ CPU_REPACK buffer: 20483.60 MiB (重排优化)
│
└─▶ 模型加载完成
```

### 11.3 阶段2：上下文创建

```
[llama_new_context_with_model]
│
├─▶ 创建 ggml_backend_sched (调度器)
│   └─▶ 注册后端：CPU (主), Metal (辅助矩阵运算)
│
├─▶ 创建 KV Cache
│   ├─▶ K cache: 40 层 × 4096 ctx × 512 dim × 2 bytes = 160 MB (f16)
│   ├─▶ V cache: 40 层 × 4096 ctx × 512 dim × 2 bytes = 160 MB (f16)
│   └─▶ 单元数组: 4096 个 llama_kv_cell
│
├─▶ 预分配计算缓冲区
│   ├─▶ buf_compute[0]: ~500 MB
│   └─▶ buf_compute[1]: ~500 MB (双缓冲)
│
└─▶ 上下文创建完成
```

### 11.4 阶段3：Tokenization

```
输入: "你好，请简单自我介绍一下"

[llama_tokenize]
│
├─▶ 预处理
│   └─▶ 无需添加 BOS（add_bos_token = false）
│
├─▶ BPE 分词
│   ├─▶ "你" → token 23168
│   ├─▶ "好" → token 294
│   ├─▶ "，" → token 15
│   ├─▶ "请" → token 16175
│   ├─▶ "简" → token 105380
│   ├─▶ "单" → token 102704
│   ├─▶ "自" → token 104038
│   ├─▶ "我" → token 400
│   ├─▶ "介" → token 104898
│   ├─▶ "绍" → token 101024
│   ├─▶ "一" → token 644
│   └─▶ "下" → token 569
│
└─▶ 输出 token IDs: [23168, 294, 15, 16175, 105380, 102704, 104038, 400, 104898, 101024, 644, 569]
    (共 12 个 token)
```

### 11.5 阶段4：Prompt Processing（首次 llama_decode）

```
[llama_decode]
│
├─▶ 构建 batch
│   └─▶ batch.token = [23168, 294, 15, ..., 569]
│   └─▶ batch.n_tokens = 12
│   └─▶ batch.pos = [0, 1, 2, ..., 11]
│
├─▶ 拆分为 ubatch（12 < 512，无需拆分）
│
├─▶ [build_graph] → llm_build_qwen35moe::build()
│   │
│   ├─▶ Input Embedding
│   │   └─▶ inpL = tok_embd[batch.token]  # (12, 2048)
│   │
│   ├─▶ Layer 0 (假设是 Full Attention 层)
│   │   ├─▶ Pre-Norm: inpL = rms_norm(inpL)
│   │   ├─▶ Q = inpL @ W_q.T  # (12, 4096)
│   │   ├─▶ K = inpL @ W_k.T  # (12, 512)
│   │   ├─▶ V = inpL @ W_v.T  # (12, 512)
│   │   ├─▶ Q/K RMS Norm
│   │   ├─▶ RoPE 位置编码 (positions 0-11)
│   │   ├─▶ KV Cache 写入: K_cache[0][0:11] = K, V_cache[0][0:11] = V
│   │   ├─▶ Attention: Q @ K.T / sqrt(dim) → softmax → @ V  # (12, 512)
│   │   ├─▶ Output Projection: @ W_o.T  # (12, 2048)
│   │   ├─▶ Sigmoid Gate: gate * attn_out
│   │   └─▶ Residual: inpL = inpL + attn_out
│   │
│   │   ├─▶ MoE FFN
│   │   │   ├─▶ Router: logits = inpL @ W_gate_inp.T  # (12, 256)
│   │   │   ├─▶ Top-8: 为每个 token 选 8 个专家
│   │   │   ├─▶ 对每个 token 计算 8 个专家的 SwiGLU
│   │   │   ├─▶ Shared Expert: gate_shexp * SwiGLU_shared
│   │   │   └─▶ Residual: inpL = inpL + ffn_out
│   │
│   ├─▶ Layer 1-39（类似，交替 Full Attention 和 Linear Attention）
│   │   ├─▶ Linear Attention 层使用 Gated Delta Net
│   │   │   ├─▶ 1D Conv → L2 Norm → Delta Net 递推
│   │   │   └─▶ 复杂度 O(n·d²) 而非 O(n²·d)
│   │   └─▶ MoE FFN（同 Layer 0）
│   │
│   ├─▶ Final RMS Norm
│   │   └─▶ inpL = rms_norm(inpL)  # (12, 2048)
│   │
│   └─▶ Output Projection (Logits)
│       └─▶ logits = inpL @ output.weight.T  # (12, 248320)
│
├─▶ [执行计算图]
│   └─▶ ggml_backend_sched_graph_compute_async()
│       └─▶ 由于 -ngl 0，全部在 CPU 后端执行
│           ├─▶ CPU 线程池启动 4 个线程
│           ├─▶ 对每个节点：
│           │   ├─▶ 反量化 Q4_K → F32（NEON SIMD）
│           │   ├─▶ 矩阵乘法（NEON 优化）
│           │   ├─▶ Softmax（NEON 优化）
│           │   └─▶ ...
│           └─▶ 全部节点执行完成
│
└─▶ 提取 logits
    └─▶ logits = [..., ..., ...]  # 最后一个位置 (pos=11) 的 logits
```

**Prompt Processing 性能：**
- 12 tokens → ~500ms → **24.7 tok/s**

### 11.6 阶段5：Token Generation（循环）

```
循环 i = 0, 1, 2, ... 直到生成 EOS 或达到 max_tokens

第 i 次迭代:
│
├─▶ [采样]
│   └─▶ llama_sampler_sample(logits)
│       ├─▶ temperature = 0.7
│       ├─▶ top_k = 40
│       ├─▶ top_p = 0.95
│       └─▶ 选中 token_id = 23169 ("你")
│
├─▶ [Detokenize]
│   └─▶ "你" → 添加到输出缓冲区
│
├─▶ [llama_decode] 处理单个 token
│   └─▶ batch.token = [23169]
│   └─▶ batch.pos = [12 + i]
│   │
│   ├─▶ [build_graph]
│   │   ├─▶ Input Embedding: tok_embd[23169]  # (1, 2048)
│   │   ├─▶ Layer 0
│   │   │   ├─▶ Q = inpL @ W_q.T  # (1, 4096)
│   │   │   ├─▶ K = inpL @ W_k.T  # (1, 512)
│   │   │   ├─▶ V = inpL @ W_v.T  # (1, 512)
│   │   │   ├─▶ KV Cache 追加: K_cache[0][12+i] = K, V_cache[0][12+i] = V
│   │   │   ├─▶ Attention: Q @ K_cache[0][0:12+i].T  # 只计算新 token 与所有历史的关系
│   │   │   │   ⚡ 关键优化：不重新计算旧 token 的 K/V
│   │   │   └─▶ ...（同 Prompt Processing）
│   │   └─▶ ...（Layer 1-39）
│   │   └─▶ Logits: (1, 248320)
│   │
│   └─▶ [执行] CPU 后端，单 token 图
│
└─▶ 如果 token != EOS，继续循环
```

**Token Generation 性能：**
- 1 token → ~123ms → **8.1 tok/s**

### 11.7 阶段6：输出与清理

```
最终输出（77 tokens）：

"你好！我是通义千问（Qwen），由阿里巴巴集团通义实验室独立研发的大语言模型。

我的目标是成为你的思考伙伴和得力助手，能够协助你高效地解决各种问题。
我具备以下核心能力：

*   全栈代码能力：支持复杂代码的生成、理解与调试
*   深度逻辑推理：擅长处理数学、自然科学及复杂的逻辑分析问题
*   超长文本处理：原生支持超长上下文窗口
*   多语言与视觉分析：支持全球100多种语言
*   智能体自主规划：可以自主规划并完成需要多步协同的复杂任务

请问今天有什么我可以帮你的吗？"

[性能统计]
Prompt: 12 tokens @ 24.7 t/s = 486 ms
Generate: 77 tokens @ 8.1 t/s = 9500 ms
Total: 89 tokens @ 9.4 t/s = 9986 ms

[内存清理]
├─▶ llama_free(ctx)     # 释放上下文（KV Cache + 计算缓冲区）
├─▶ llama_free_model(model)  # 释放模型
│   ├─▶ munmap() 解除文件映射
│   └─▶ 释放 CPU 缓冲区
└─▶ 程序退出
```

---

