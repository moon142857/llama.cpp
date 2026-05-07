# llama.cpp 全链路深度分析报告

> 基于 `tinyllamas/stories15M` 模型运行日志，面向 NPU 后端移植开发者

---

## 一、运行环境概述

| 项目 | 值 |
|------|-----|
| 硬件 | Apple M4 (SoC) |
| 后端 | Metal (MTL0) + CPU |
| 模型 | tinyllamas/stories15M (GPT-2 架构) |
| 参数量 | ~15M |
| 层数 | n_layer=6 |
| GPU 层 | n_gpu_layers=7 (全部 offload 到 GPU) |
| 内存 | unified memory = true (CPU/GPU 共享内存) |

---

## 二、环节一：模型加载（GGUF → 内存）

### 2.1 调用链
```
main() → llama_model_load() → GGUF 解析 → hparams 加载 → vocab 加载 → load_tensors()
```

### 2.2 日志证据
```
=== [LLAMA_TRACE] llama_model_load: START model load ===
=== [LLAMA_TRACE] llama_model_load: GGUF parsed, creating model object ===
=== [LLAMA_TRACE] llama_model_load: hyperparameters loaded ===
=== [LLAMA_TRACE] llama_model_load: vocabulary loaded ===
=== [LLAMA_TRACE] llama_model_load: loading tensors... ===
```

### 2.3 详细拆解

#### Step 1: GGUF 文件解析
- `llama_model_loader` 读取 `.gguf` 文件，解析 KV metadata（超参数、架构名称、RoPE 配置等）
- 根据 `general.architecture` 字段确定模型架构，本例为 `gpt2`
- 解析张量信息（name, shape, type, offset），但**不立即读取权重数据**

#### Step 2: 超参数加载
- `load_arch_hparams()` 架构相关参数：n_layer, n_embd, n_head, n_vocab, f_norm_eps 等
- `llama_model_load()` 通用参数：上下文长度、RoPE 频率、GQA 配置等

#### Step 3: 词表加载
- 从 GGUF 读取 tokenizer 配置（BPE/SentencePiece/WordPiece）
- 加载 merge rules、special tokens、token → piece 映射表

#### Step 4: 张量加载（核心）
```
=== [LLAMA_TRACE] load_tensors: device assignment: n_layer=6 n_gpu_layers=7 i_gpu_start=0 ===
=== [LLAMA_TRACE] load_tensors: creating backend buffers for 2 contexts ===
=== [LLAMA_TRACE] load_tensors: buffer CPU_Mapped   size=    4.94 MiB ===
=== [LLAMA_TRACE] load_tensors: buffer MTL0_Mapped  size=   12.56 MiB ===
```

**关键行为：**
1. **逐层分配设备**：`i_gpu_start=0` 表示从第 0 层开始 offload 到 GPU
2. **后端缓冲区创建**：`ml.ctx_map` 按设备分桶，为每个 backend 创建 `ggml_backend_buffer`
3. **内存映射**：
   - `CPU_Mapped (4.94 MiB)`：无法 offload 到 GPU 的张量（如输入/输出处理相关）
   - `MTL0_Mapped (12.56 MiB)`：GPU 上的权重张量
4. **张量创建**：`create_tensor()` 在对应 backend buffer 上分配内存，**mmap 懒加载权重**

**NPU 移植要点：**
- 你需要在 `ggml_backend_reg` 中注册自己的 NPU backend
- `load_tensors` 会根据 `n_gpu_layers` 决定将哪些层放到你的 NPU 上
- 关键是实现 `ggml_backend_buffer_type` 和 `ggml_backend_buffer` 接口

---

## 三、环节二：Tokenizer（文本 → Token IDs）

### 3.1 调用链
```
main() → llama_tokenize() → vocab->tokenize() → BPE 编码 → token IDs 数组
```

### 3.2 日志证据
```
=== [LLAMA_TRACE] llama_tokenize: text_len=66 add_special=1 ===
=== [LLAMA_TRACE] llama_tokenize: result=32 tokens ===
```

### 3.3 详细拆解

#### 输入处理
- `text="Once upon a time, there was a little girl named..."`（共 66 字节）
- `add_special=1`：自动添加 BOS (Beginning of Sequence) token

#### BPE 编码过程
1. **预分词**：按空格/标点切分
2. **子词切分**：使用 BPE merge table 将单词切分为 subword tokens
3. **特殊 token**：在开头添加 BOS，结尾可选 EOS

#### 输出
- 32 个 token IDs：`[1, 19182, 13775, 257, 640, 11, 612, 373, 257, 1310, 1848, ...]`
- 这些 token IDs 将作为 `llama_batch` 的输入，进入 `decode()`

---

## 四、环节三：计算图构建（Graph Build）

### 4.1 调用链
```
decode() → process_ubatch() → build_graph() → llama_model_gpt2::graph::graph()
```

### 4.2 日志证据
```
=== [LLAMA_TRACE] build_graph: START building graph ===
=== [LLAMA_TRACE] build_graph: DONE graph built ===
=== [LLAMA_TRACE] ggml_backend_sched_split_graph: nodes=193 leafs=76 ===
```

### 4.3 详细拆解：GPT-2 前向传播图

代码位置：`src/models/gpt2.cpp:57-155`

```cpp
// 1. 输入嵌入
inpL = build_inp_embd(model.tok_embd);     // [n_vocab, n_embd] → [n_tokens, n_embd]
pos  = ggml_get_rows(model.pos_embd, inp_pos); // 位置编码
inpL = ggml_add(inpL, pos);                // 嵌入 + 位置编码

// 2. 循环 n_layer 次（本例 n_layer=6）
for (int il = 0; il < n_layer; ++il) {
    // 2.1 LayerNorm
    cur = build_norm(inpL, attn_norm, attn_norm_b, LLM_NORM, il);

    // 2.2 Self-Attention
    auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv, il);
    // QKV shape: [n_embd, n_embd + 2*n_embd_gqa] 的线性投影

    cur = build_attn(inp_attn, layer.wo, layer.wo_b, layer.wo_s,
                     Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                     1.0f/sqrtf(float(n_embd_head)), il);
    // attention scale = 1/sqrt(head_dim)

    // 2.3 残差连接
    ffn_inp = ggml_add(cur, inpL);

    // 2.4 FFN (Feed Forward Network)
    cur = build_norm(ffn_inp, ffn_norm, ffn_norm_b, LLM_NORM, il);
    cur = build_ffn(cur,
        layer.ffn_up,   layer.ffn_up_b,   NULL,  // up-projection
        NULL,           NULL,              NULL,  // gate (GPT-2 没有 Gated FFN)
        layer.ffn_down, layer.ffn_down_b, NULL,  // down-projection
        NULL,
        LLM_FFN_GELU, LLM_FFN_SEQ, il);          // GELU 激活

    // 2.5 残差连接
    cur = ggml_add(cur, ffn_inp);

    inpL = cur;  // 下一层的输入
}

// 3. 输出层
 cur = build_norm(inpL, output_norm, output_norm_b, LLM_NORM, -1);
res->t_embd = cur;

cur = build_lora_mm(model.output, cur);  // [n_embd, n_vocab] 投影到词表
res->t_logits = cur;
```

### 4.4 图结构统计
- **193 个 nodes**：每个 ggml op 是一个 node
- **76 个 leafs**：输入张量和权重张量（没有依赖其他节点的节点）
- **图拓扑**：DAG（有向无环图）， Metal backend 可以并行执行无依赖的节点

### 4.5 图复用优化
```
// 第一次 decode（warmup）
process_ubatch: building graph
build_graph: START → DONE
alloc_splits: allocation done

// 第二次及以后（相同拓扑）
process_ubatch: setting inputs  ← 跳过 build_graph 和 alloc_splits
```

**关键发现：** 当 batch size 和 token 数相同时，图结构不变，直接复用。只有在 KV cache 增长导致 shapes 变化时才需要重建。

---

## 五、环节四：后端调度与执行

### 5.1 调用链
```
decode() → graph_compute() → ggml_backend_sched_compute_splits()
    → split_graph() → alloc_splits() → 逐 split 执行
```

### 5.2 日志证据
```
=== [LLAMA_TRACE] ggml_backend_sched_compute_splits: n_splits=2 ===
=== [LLAMA_TRACE] ggml_backend_sched_compute_splits:   split 0/2 backend=CPU nodes=1 ===
=== [LLAMA_TRACE] ggml_backend_sched_compute_splits:   split 1/2 backend=MTL0 nodes=192 ===
=== [LLAMA_TRACE] ggml_backend_sched_compute_splits: ALL 2 splits computed OK ===
```

### 5.3 详细拆解

#### Step 1: 图切分（Split Graph）
- 遍历图的拓扑顺序，按 backend 支持能力切分
- **Split 0 (CPU, 1 node)**：输入节点（`build_inp_embd` 相关），因为某些输入张量只在 CPU 上
- **Split 1 (MTL0, 192 nodes)**：所有 Transformer 计算（LayerNorm, Attention, FFN）

#### Step 2: 内存分配（Alloc Splits）
```
=== [LLAMA_TRACE] ggml_backend_sched_alloc_splits: n_nodes=205 n_leafs=76 ===
```
- 为每个 split 的 intermediate tensors 分配内存
- n_nodes 从 193 变成 205：调度器可能插入了 copy/transpose 节点

#### Step 3: 算子分发执行
- **Split 0**：CPU 执行 1 个 node（输入嵌入准备）
- **Split 1**：Metal 执行 192 个 nodes（全部 Transformer 计算）
- CPU → Metal 之间有隐式数据拷贝（unified memory 上只是指针传递）

#### 两阶段执行特征

**Prompt Processing（批量处理）**：
```
decode: n_tokens=32          ← 一次性处理 32 个 token
graph_compute: batched=1     ← batched mode
```

**Token Generation（自回归）**：
```
decode: n_tokens=1           ← 每次只处理 1 个 token
graph_compute: batched=0     ← 单 token mode
```

---

## 六、环节五：KV Cache 管理

### 6.1 调用链
```
decode() → memory_update() → KV cache 位置更新 / shift
```

### 6.2 日志证据
```
=== [LLAMA_TRACE] memory_update: optimize=0 ===
```

### 6.3 详细拆解

#### KV Cache 结构
- K cache: [n_layer, n_head_kv, n_ctx, n_embd_head_k]
- V cache: [n_layer, n_head_kv, n_ctx, n_embd_head_v]
- `memory_update()` 负责：
  1. 标记哪些位置需要保留
  2. 处理 KV cache 的 shifting（当超过上下文窗口时）
  3. 设置 `optimize=0` 表示不进行 aggressive optimization（保持已有分配）

#### 与 Graph 的交互
- `build_attn_inp_kv()` 创建 KV cache 输入节点
- `build_attn()` 中 Kcur/Vcur 被写入 KV cache，同时读取之前的 K/V 做 attention
- 图复用的关键：KV cache 的 shape 随序列长度增长，但 topology 不变

---

## 七、环节六：采样与 Detokenize

### 7.1 调用链
```
decode() → 获取 logits → llama_sampler_sample() → token ID
         → llama_token_to_piece() → 文本输出
```

### 7.2 日志证据
```
=== [LLAMA_TRACE] llama_token_to_piece: token=6716 ===
=== [LLAMA_TRACE] llama_token_to_piece: result=3 ===
```

### 7.3 详细拆解

#### 采样流程
1. **获取 logits**：从 `res->t_logits` 读取最后一个 token 的 [n_vocab]  logits 向量
2. **温度缩放 + Top-K/Top-P**：通过 sampler chain 处理
3. **随机采样**：从过滤后的分布中采样下一个 token ID

#### Detokenize
- `llama_token_to_piece(token=6716)` → 返回 3 字节 → 解码为 UTF-8 字符串
- 生成的 token 被追加到输出文本，同时作为下一个 `decode()` 的输入

#### 自回归循环
```
decode(n_tokens=1) → sample → detokenize → print
      ↓______________________________________|
```
本例中循环 12 次，每次生成 1 个 token。

---

## 八、NPU 后端移植关键接入点

### 8.1 需要实现的文件/接口

```
ggml/src/ggml-<your_npu>/
├── ggml-<npu>.cpp          # 核心：backend 实现
├── ggml-<npu>.h            # 头文件
└── CMakeLists.txt          # 构建配置
```

### 8.2 核心接口清单

| 接口 | 作用 | 代码参考 |
|------|------|---------|
| `ggml_backend_reg` | 注册你的 NPU backend | `ggml_backend_metal_reg()` |
| `ggml_backend_buffer_type` | 定义内存分配策略 | `ggml_backend_metal_buffer_type()` |
| `ggml_backend_buffer` | 实际内存分配 | `ggml_backend_buft_alloc_buffer()` |
| `ggml_backend_graph_compute` | 执行计算图 | `ggml_backend_metal_graph_compute()` |
| `ggml_backend_supports_op` | 声明支持哪些 op | 必须支持 MATMUL, SOFT_MAX, ROPE, ADD, MUL, NORM 等 |

### 8.3 关键接入路径

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
   - 先只实现 MATMUL（占 90% 计算量）
   - 不支持的 op fallback 到 CPU
   - scheduler 会自动将不支持的 op 切到 CPU split
```

### 8.4 Metal 后端作为参考实现

关键文件：`ggml/src/ggml-metal/ggml-metal.cpp`

学习路径：
1. `ggml_backend_metal_init()` — 设备初始化
2. `ggml_metal_buffer_type()` — 缓冲区类型
3. `ggml_metal_graph_compute()` — 图执行（查找 kernel 并 dispatch）
4. `ggml_metal_supports_op()` — op 支持列表

---

## 九、数据流全景图

```
[GGUF File]
    ↓ mmap
[llama_model_loader] 解析 metadata + 张量信息
    ↓
[load_tensors] 按层分配设备 → CPU buffer + NPU buffer
    ↓
[Tokenizer] "Once upon a time" → [1, 19182, 13775, ...] (32 tokens)
    ↓
[llama_batch] batch.n_tokens=32
    ↓
[decode] → [memory_update] KV cache 位置准备
    ↓
[build_graph] 构建 193 node 的 Transformer 计算图
    ↓
[split_graph] 切分：CPU(1 node) + NPU(192 nodes)
    ↓
[alloc_splits] 为 205 nodes 分配内存
    ↓
[compute_splits] 顺序执行 2 个 split
    ↓
[logits] [n_vocab] 概率分布向量
    ↓
[sampler] Top-K/Top-P + 温度采样 → token ID
    ↓
[token_to_piece] token ID → " Once" / " upon" / ...
    ↓
[输出文本] "Once upon a time there was a little girl named..."
    ↓
[自回归循环] 新生成的 token 作为下一个 decode() 的输入
```

---

## 十、关键结论

1. **图构建一次，多次复用**：193 个 node 的图只在前几次 `decode()` 时构建，后续直接复用
2. **调度器是核心**：`ggml_backend_sched` 负责跨后端切分图、分配内存、管理数据依赖
3. **两阶段执行**：prompt processing（batch）vs token generation（single），NPU 优化重点在单 token latency
4. **NPU 移植最小工作集**：
   - 注册 backend → buffer allocation → graph compute → supports_op
   - 从只支持 MATMUL 开始，其余 fallback CPU
   - 参考 `ggml-metal.cpp` 的 2000+ 行实现
5. **内存模型**：llama.cpp 通过 `ggml_backend_buffer` 抽象设备内存，NPU 需要实现自己的分配器
