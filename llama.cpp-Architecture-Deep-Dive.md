# llama.cpp 整体架构与 Qwen3.6-35B-A3B 运行全流程技术详解

> **研究目标**：从源码层面彻底拆解 llama.cpp 的架构设计，并逐行追踪 Qwen3.6-35B-A3B 模型从"用户输入提示词"到"模型输出生成 token"的完整技术路径。  
> **代码版本**：llama.cpp build 9010 (commit d05fe1d7d)  
> **研究日期**：2026-05-04

---

## 目录

1. [项目架构总览](#1-项目架构总览)
2. [编译系统详解](#2-编译系统详解)
3. [GGML 张量计算引擎](#3-ggml-张量计算引擎)
4. [核心库架构：llama](#4-核心库架构llama)
5. [模型加载全流程](#5-模型加载全流程)
6. [Tokenization 流程](#6-tokenization-流程)
7. [推理引擎：从计算图到 Attention](#7-推理引擎从计算图到-attention)
8. [后端执行系统](#8-后端执行系统)
9. [采样与生成](#9-采样与生成)
10. [Server 架构](#10-server-架构)
11. [Qwen3.6-35B-A3B 运行全流程拆解](#11-qwen36-35b-a3b-运行全流程拆解)

---

## 1. 项目架构总览

### 1.1 顶层目录结构

```
llama.cpp/
├── ggml/                    # 张量计算引擎（核心）
│   ├── include/             # GGML 公共头文件
│   │   ├── ggml.h           # 主 API（~107KB，2829 行）
│   │   ├── ggml-backend.h   # 后端抽象层
│   │   ├── ggml-cpu.h       # CPU 后端
│   │   ├── ggml-metal.h     # Metal 后端
│   │   └── gguf.h           # GGUF 文件格式
│   └── src/                 # GGML 实现
│       ├── ggml.c           # 核心张量库（~248KB）
│       ├── ggml-backend.cpp # 后端调度器（~93KB）
│       ├── ggml-quants.c    # 量化/反量化（~223KB）
│       ├── ggml-alloc.c     # 内存分配器
│       ├── gguf.cpp         # GGUF 读写（~53KB）
│       ├── ggml-cpu/        # CPU 后端（SIMD 优化）
│       ├── ggml-metal/      # Apple Metal 后端
│       ├── ggml-blas/       # BLAS/Accelerate
│       ├── ggml-cuda/       # NVIDIA CUDA
│       ├── ggml-vulkan/     # Vulkan
│       └── ...              # SYCL, HIP, RPC 等
│
├── src/                     # llama 核心库
│   ├── llama.cpp            # C API 实现（~19KB）
│   ├── llama-model.cpp      # 模型加载与架构分发（~546KB）
│   ├── llama-model-loader.cpp  # GGUF 加载器（~71KB）
│   ├── llama-context.cpp    # 上下文与推理（~127KB）
│   ├── llama-graph.cpp      # 计算图构建（~101KB）
│   ├── llama-batch.cpp      # Batch 处理
│   ├── llama-arch.cpp       # 架构注册表（~63KB）
│   ├── llama-kv-cache.cpp   # KV 缓存
│   ├── llama-sampler.cpp    # 采样器链（~132KB）
│   ├── llama-vocab.cpp      # 分词器（~167KB）
│   └── models/              # 各架构计算图构建器
│       ├── qwen35moe.cpp    # Qwen3.5 MoE（~17KB）
│       ├── qwen35.cpp       # Qwen3.5 Dense（~15KB）
│       ├── qwen3next.cpp    # Qwen3-Next（~23KB）
│       └── ...              # 140+ 架构
│
├── common/                  # 共享工具库
│   ├── arg.cpp/h            # 命令行参数解析
│   ├── common.cpp/h         # 共享参数、系统信息
│   ├── chat.cpp/h           # Chat 模板引擎
│   ├── sampling.cpp/h       # 高级采样管道
│   ├── jinja/               # Jinja2 模板引擎
│   └── ...
│
├── tools/                   # 可执行工具
│   ├── cli/                 # llama-cli
│   ├── server/              # llama-server
│   ├── llama-bench/         # 基准测试
│   ├── quantize/            # 量化工具
│   └── ...
│
├── include/                 # 公共 C API 头文件
│   ├── llama.h              # 主 C API（~80KB，1565 行）
│   └── llama-cpp.h          # C++ 包装器
│
├── examples/                # 示例程序
├── tests/                   # 测试套件
├── gguf-py/                 # Python GGUF 包
└── vendor/                  # 第三方库
```

### 1.2 核心数据流

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  用户输入    │────▶│  llama_tokenize  │────▶│  llama_decode   │
│  (文本)     │     │  (分词器)        │     │  (推理引擎)      │
└─────────────┘     └─────────────────┘     └────────┬────────┘
                                                      │
                      ┌───────────────────────────────┘
                      ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  文本输出    │◀────│ llama_sampler    │◀────│  ggml_cgraph    │
│             │     │  (采样器)        │     │  (计算图执行)    │
└─────────────┘     └─────────────────┘     └─────────────────┘
                                                      ▲
                      ┌───────────────────────────────┘
                      │
               ┌──────┴──────┐     ┌─────────────┐
               │ GGUF 模型文件 │────▶│ 后端执行器   │
               │ (weights)   │     │ (Metal/CPU) │
               └─────────────┘     └─────────────┘
```

---

## 2. 编译系统详解

### 2.1 CMake 目标层级

```
llama.cpp (根 CMakeLists.txt)
│
├── ggml/CMakeLists.txt
│   ├── ggml-base (静态库)          # 核心张量操作
│   ├── ggml-cpu (静态库)           # CPU 后端
│   ├── ggml-metal (静态库)         # Metal 后端
│   ├── ggml-blas (静态库)          # BLAS 后端
│   └── ggml (共享/静态库)          # 聚合库 + 后端注册
│
├── src/CMakeLists.txt
│   └── llama (共享/静态库)         # 模型加载 + 推理 + 采样
│       依赖: PUBLIC ggml
│
├── common/CMakeLists.txt
│   ├── llama-common-base (静态库)
│   └── llama-common (共享库)       # 参数解析 + Chat 模板
│       依赖: llama + cpp-httplib
│
└── tools/*/CMakeLists.txt
    ├── llama-cli                   # 交互式 CLI
    ├── llama-server                # HTTP API 服务
    ├── llama-bench                 # 基准测试
    └── ...
```

### 2.2 后端编译机制

`ggml/src/CMakeLists.txt` 定义了两个关键函数：

```cmake
# 添加后端库
ggml_add_backend_library(name sources...)

# 注册后端
ggml_add_backend(name)
```

后端可以以两种方式构建：
- **静态链接**（默认）：后端库直接链接到 `ggml` 主库
- **动态加载**（`GGML_BACKEND_DL=ON`）：每个后端编译为 `.so`/`.dylib`，运行时通过 `dlopen` 加载

### 2.3 Apple Silicon 编译路径

```
CMakeLists.txt
  └─▶ add_subdirectory(ggml)
        └─▶ ggml_add_backend(METAL)
              └─▶ ggml-metal/CMakeLists.txt
                    ├─▶ 编译 ggml-metal.cpp (C++)
                    ├─▶ 编译 ggml-metal-ops.cpp (C++)
                    ├─▶ 编译 ggml-metal-context.m (Objective-C)
                    ├─▶ 编译 ggml-metal-device.m (Objective-C)
                    └─▶ 处理 ggml-metal.metal (着色器源码)
                          └─▶ 嵌入二进制：xxd -i → ggml-metal-embed.s
        └─▶ ggml_add_backend(BLAS)
              └─▶ 链接 Accelerate.framework
        └─▶ ggml_add_backend(CPU)
              └─▶ 检测 ARM 特性 (NEON, SVE, SME)
                    ├─▶ 编译 ggml-cpu.c (通用)
                    ├─▶ 编译 arch/arm/quants.c (ARM 专用量化)
                    └─▶ 编译 arch/arm/repack.cpp (权重重排)
```

---

## 3. GGML 张量计算引擎

GGML 是 llama.cpp 的底层张量计算库。它定义了统一的张量抽象，并支持多种计算后端。

### 3.1 核心抽象

#### ggml_tensor（张量）

```c
struct ggml_tensor {
    enum ggml_type type;        // 数据类型 (F32, Q4_0, Q4_K, etc.)
    enum ggml_type src0_type;   // 第一个输入的类型
    
    struct ggml_backend_buffer * buffer;  // 内存缓冲区
    
    int64_t ne[GGML_MAX_DIMS];  // 各维度大小 (number of elements)
    size_t  nb[GGML_MAX_DIMS];  // 各维度步长 (number of bytes)
    
    enum ggml_op op;            // 操作类型 (MUL_MAT, ADD, NORM, etc.)
    enum ggml_op_params op_params;  // 操作参数
    
    int32_t flags;              // 标志位 (permuted, etc.)
    
    struct ggml_tensor * src[GGML_MAX_SRC];  // 输入张量
    struct ggml_tensor * view_src;           // 视图源
    int64_t view_offs;                       // 视图偏移
    
    void * data;                // 数据指针
    char name[GGML_MAX_NAME];   // 张量名称
    void * extra;               // 后端专用数据
};
```

#### 数据类型系统

| 类型 | 位宽 | 说明 |
|------|------|------|
| `GGML_TYPE_F32` | 32-bit | 单精度浮点 |
| `GGML_TYPE_F16` | 16-bit | 半精度浮点 |
| `GGML_TYPE_BF16` | 16-bit | Brain 浮点 |
| `GGML_TYPE_Q4_0` | 4-bit | 每 block 32 个权重，共享 1 个 scale |
| `GGML_TYPE_Q4_1` | 4-bit | 每 block 32 个权重，共享 scale + min |
| `GGML_TYPE_Q4_K` | 4-bit | K-quant，双级量化，质量更好 |
| `GGML_TYPE_Q5_K` | 5-bit | K-quant 变体 |
| `GGML_TYPE_Q6_K` | 6-bit | K-quant 变体 |
| `GGML_TYPE_Q8_0` | 8-bit | 每 block 32 个权重，共享 scale |
| `GGML_TYPE_IQ4_NL` | 4-bit | Importance 加权量化 |
| `GGML_TYPE_Q1_0` ~ `Q3_K` | 1-3 bit | 更低精度 |
| `GGML_TYPE_MXFP4` | 4-bit | NVIDIA 微缩放格式 |

### 3.2 计算图（ggml_cgraph）

GGML 使用**计算图**来表示整个前向传播过程：

```c
struct ggml_cgraph {
    int n_nodes;                    // 节点数量
    int n_leafs;                    // 叶子节点数量
    struct ggml_tensor ** nodes;    // 计算节点（有序）
    struct ggml_tensor ** leafs;    // 输入/常量节点
    enum ggml_cgraph_eval_order order;  // 执行顺序
};
```

**构建计算图的流程：**

```
1. 创建 ggml_context（张量分配器）
2. 定义输入张量（leafs）
3. 通过 ggml_op 函数构建计算链：
   ggml_mul_mat(ctx, weight, input) → ggml_add(ctx, bias, result) → ggml_norm(ctx, ...)
4. 调用 ggml_build_forward_expand(graph, output_tensor)
5. 图构建完成， nodes[] 中存储了拓扑排序后的计算节点
```

### 3.3 后端抽象层（ggml_backend）

后端抽象层是 GGML 的多设备核心：

```
ggml_backend_t                    # 后端实例（如 Metal、CPU）
  ├── name
  ├── device
  └── interface (函数指针表)
      ├── alloc_buffer()
      ├── get_alignment()
      ├── tensor_set/get()
      ├── graph_compute()
      └── ...

ggml_backend_sched_t              # 调度器（跨后端自动分配）
  ├── 管理多个后端（主后端 + 辅助后端）
  ├── 分析计算图，决定每个张量存哪个后端
  ├── 自动插入 cross-backend copy 节点
  └── 按拓扑顺序调度执行
```

**调度器工作原理：**

```
1. 分析计算图中每个节点的输入/输出张量
2. 根据 -ngl 参数决定哪些层的主权重放在 GPU
3. 对于 GPU 上的层：节点在 GPU 后端执行
4. 对于 CPU 上的层：节点在 CPU 后端执行
5. 如果某节点输入在 GPU、输出需要在 CPU：
   自动插入 ggml_cpy 节点（GPU → CPU 内存拷贝）
6. 按拓扑顺序执行所有节点
```

---

## 4. 核心库架构：llama

### 4.1 公共 C API（include/llama.h）

llama.h 定义了约 200 个函数，分为以下几组：

| 函数组 | 说明 |
|--------|------|
| `llama_model_load_from_file*()` | 模型加载 |
| `llama_new_context_with_model()` | 上下文创建 |
| `llama_decode()` / `llama_encode()` | 推理（解码/编码） |
| `llama_get_logits*()` | 获取 logits |
| `llama_get_embeddings*()` | 获取嵌入 |
| `llama_sampler_*()` | 采样器 API |
| `llama_kv_cache_*()` | KV 缓存操作 |
| `llama_tokenize()` / `llama_detokenize()` | 分词/反分词 |
| `llama_chat_apply_template()` | Chat 模板应用 |
| `llama_state_*()` | 状态保存/恢复 |

### 4.2 核心类结构

```
llama_model                       # 模型（只读，可共享）
  ├── hparams                     # 超参数（n_layer, n_embd, n_head 等）
  ├── vocab                       # 词表（tokenizer）
  ├── arch                        # 架构枚举（LLM_ARCH_QWEN35MOE）
  ├── layers[]                    # 各层张量指针
  │   ├── attn_q, attn_k, attn_v  # Attention 权重
  │   ├── attn_o                  # Attention 输出投影
  │   ├── ffn_gate, ffn_up, ffn_down  # FFN 权重
  │   └── ...                     # MoE、SSM 等额外权重
  ├── tok_embd                    # Token 嵌入矩阵
  ├── output_norm                 # 最终 LayerNorm
  ├── output                      # 输出投影（logits）
  └── tensors                     # 所有张量的映射表

llama_context                     # 上下文（每个会话一个）
  ├── model*                      # 指向共享模型
  ├── kv_cache                    # KV 缓存
  ├── sched                       # 后端调度器
  ├── graph                       # 当前计算图
  ├── buf_compute[]               # 计算缓冲区
  ├── logits                      # 输出 logits
  └── embeddings                  # 输出嵌入
```

### 4.3 架构注册系统

`src/llama-arch.cpp` 维护了一个巨大的注册表：

```cpp
// 架构枚举（~140 种）
enum llm_arch {
    LLM_ARCH_UNKNOWN,
    LLM_ARCH_LLAMA,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_QWEN2MOE,
    LLM_ARCH_QWEN3,
    LLM_ARCH_QWEN3MOE,
    LLM_ARCH_QWEN35,       // Qwen3.5
    LLM_ARCH_QWEN35MOE,    // Qwen3.5 MoE ← 我们的模型
    LLM_ARCH_QWEN3NEXT,
    LLM_ARCH_DEEPSEEK,
    LLM_ARCH_DEEPSEEK2,
    LLM_ARCH_GEMMA,
    // ... 共约 140 种
};

// GGUF 键名枚举（~200 个）
enum llm_kv {
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_NAME,
    LLM_KV_GENERAL_QUANTIZATION_VERSION,
    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_EXPERT_COUNT,       // MoE
    LLM_KV_EXPERT_USED_COUNT,  // MoE
    // ...
};

// 张量名枚举（~150 个）
enum llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_OUTPUT_NORM,
    LLM_TENSOR_OUTPUT,
    LLM_TENSOR_ATTN_Q,
    LLM_TENSOR_ATTN_K,
    LLM_TENSOR_ATTN_V,
    LLM_TENSOR_ATTN_Q_NORM,    // Q/K RMS Norm（Qwen 特有）
    LLM_TENSOR_ATTN_K_NORM,
    LLM_TENSOR_ATTN_GATE,      // Attention Gate（Qwen3.5 特有）
    LLM_TENSOR_FFN_GATE_INP,   // MoE Router
    LLM_TENSOR_FFN_GATE_EXPS,  // MoE Gate
    LLM_TENSOR_FFN_UP_EXPS,    // MoE Up
    LLM_TENSOR_FFN_DOWN_EXPS,  // MoE Down
    LLM_TENSOR_SSM_CONV1D,     // Delta Net Conv（Qwen3.5 特有）
    LLM_TENSOR_SSM_ALPHA,      // Delta Net Alpha
    LLM_TENSOR_SSM_BETA,       // Delta Net Beta
    // ...
};
```

**注册流程：**

```
1. GGUF 文件中的 general.architecture = "qwen35moe"
2. llama_model_loader 读取该字符串
3. llm_arch_from_string("qwen35moe") → 返回 LLM_ARCH_QWEN35MOE
4. llama-model.cpp 根据 arch 分发：
   - 调用 qwen35moe 的 load_hparams() 加载超参数
   - 调用 qwen35moe 的 load_tensors() 创建张量
   - 后续 build_graph() 使用 llm_build_qwen35moe
```

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

## 10. Server 架构

### 10.1 多线程架构

```
主线程                    工作线程 (N 个)
  │                         │
  ▼                         ▼
HTTP Listener ──▶ Task Queue ──▶ Worker Loop
  (cpp-httplib)     (mutex+cond)   (server_context)
                         │              │
                         │              ▼
                         │         模型加载/推理
                         │              │
                         ▼              ▼
                    Result Queue  ◀── 完成通知
                         │
                         ▼
                    HTTP Response
                    (SSE 流式传输)
```

### 10.2 Slot 系统

Server 使用 **Slot** 来管理并发请求：

```cpp
struct server_slot {
    int id;                    // Slot ID
    llama_seq_id seq_id;       // 序列 ID（对应 KV Cache 中的序列）
    
    // 状态
    enum SERVER_SLOT_STATE {
        IDLE,      // 空闲
        PROCESSING // 正在处理
    } state;
    
    // 数据
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> generated_tokens;
    
    // 参数
    common_params_sampling sparams;
    int n_ctx;                 // 上下文窗口
    bool stream;               // 是否流式输出
};

// 默认 4 个 slot（-np 参数）
server_slot slots[4];
```

### 10.3 请求处理流程

```
1. 客户端 POST /v1/chat/completions
   
2. HTTP 层解析 JSON → server_task
   {
       type: COMPLETION,
       data: {
           messages: [...],
           max_tokens: 256,
           temperature: 0.7,
           stream: true/false
       }
   }

3. Task Queue 接收任务

4. Worker 线程获取任务
   
5. 查找空闲 Slot
   for slot in slots:
       if slot.state == IDLE:
           slot.state = PROCESSING
           break

6. 应用 Chat Template
   text = chat_template.render(messages)
   tokens = tokenizer.encode(text)

7. 推理循环
   for i in 0..max_tokens:
       llama_decode(ctx, batch)      // 前向传播
       logits = llama_get_logits(ctx)
       token = llama_sampler_sample(sampler, logits)  // 采样
       
       if token == EOS:
           break
       
       if stream:
           发送 SSE chunk: {"choices": [{"delta": {"content": "字"}}]}
       
       batch = {token}  // 下一个 token

8. 如果不是流式，构造完整响应 JSON

9. Slot 状态重置为 IDLE

10. 返回 HTTP 响应
```

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

## 附录：关键源码文件速查表

| 功能 | 文件 | 大小 |
|------|------|------|
| GGML 核心 | `ggml/src/ggml.c` | ~248KB |
| GGML 后端调度 | `ggml/src/ggml-backend.cpp` | ~93KB |
| GGML 量化 | `ggml/src/ggml-quants.c` | ~223KB |
| CPU 后端 | `ggml/src/ggml-cpu/ops.cpp` | ~382KB |
| Metal 后端 | `ggml/src/ggml-metal/ggml-metal.metal` | ~447KB |
| 模型加载 | `src/llama-model.cpp` | ~546KB |
| 上下文/推理 | `src/llama-context.cpp` | ~127KB |
| 计算图构建 | `src/llama-graph.cpp` | ~101KB |
| 架构注册 | `src/llama-arch.cpp` | ~63KB |
| Qwen3.5 MoE 图 | `src/models/qwen35moe.cpp` | ~17KB |
| KV 缓存 | `src/llama-kv-cache.cpp` | ~82KB |
| 采样器 | `src/llama-sampler.cpp` | ~132KB |
| 分词器 | `src/llama-vocab.cpp` | ~167KB |
| C API 头 | `include/llama.h` | ~80KB |
| Server | `tools/server/server-context.cpp` | ~179KB |
| CLI | `tools/cli/cli.cpp` | ~15KB |
| 参数解析 | `common/arg.cpp` | ~189KB |
| Chat 模板 | `common/chat.cpp` | ~103KB |

---

*文档生成时间：2026-05-04*  
*基于 llama.cpp commit d05fe1d7d (build 9010)*
