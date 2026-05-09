# llama.cpp 系统架构总览

> **目标**：理解 llama.cpp 的模块分层、数据流和核心抽象
> **版本**：llama.cpp build 9010 (commit d05fe1d7d)
> **日期**：2026-05-04

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

### 2.4 后端动态加载机制 (`GGML_BACKEND_DL`)

默认情况下后端**静态链接**到 `ggml` 主库。开启 `-DGGML_BACKEND_DL=ON` 后，每个后端编译为独立的动态库：

```
# 动态加载模式下的构建产物
build/bin/
├── libggml-cpu.so        # CPU 后端
├── libggml-metal.so      # Metal 后端
├── libggml-cuda.so       # CUDA 后端
├── libggml-vulkan.so     # Vulkan 后端
└── ...
```

**运行时加载流程**（`ggml/src/ggml-backend.cpp`）：

```cpp
// 1. 扫描 build/bin/ 目录下的 libggml-*.so
std::vector<std::string> ggml_backend_load_all() {
    auto dir = get_executable_path();  // 查找与可执行文件同目录
    for each file in dir:
        if filename matches "libggml-*.so":
            ggml_backend_load(file);  // dlopen
}

// 2. 每个后端导出一个注册函数
typedef void (*ggml_backend_register_t)();

// 例如 libggml-cuda.so 导出：
extern "C" void ggml_backend_cuda_reg() {
    ggml_backend_register(&ggml_backend_cuda);
}
```

**动态加载的优势：**
- 运行时按需加载后端，减少启动内存
- 驱动/库缺失时优雅降级（如没有 CUDA 驱动则不加载 CUDA 后端）
- 支持后端热插拔（开发调试时重新编译后端无需重链主程序）

**注意事项：**
- 后端动态库必须与主程序 ABI 兼容（同一编译器、同一 GGML 头版本）
- Windows 上需注意 DLL 导出符号（`__declspec(dllexport)`）

### 2.5 CMake 宏解析：`ggml_add_backend()`

`ggml/src/CMakeLists.txt` 中定义的关键宏：

```cmake
# 添加后端库（静态或动态）
function(ggml_add_backend_library TARGET SOURCES...)
    add_library(${TARGET} ${GGML_BACKEND_TYPE} ${SOURCES})
    # GGML_BACKEND_TYPE = STATIC (默认) 或 SHARED (GGML_BACKEND_DL=ON)
    target_link_libraries(${TARGET} PRIVATE ggml-base)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

# 注册后端到全局列表
function(ggml_add_backend NAME)
    set(GGML_BACKENDS ${GGML_BACKENDS} ${NAME} PARENT_SCOPE)
    # 生成 backend 注册代码
    set(GGML_BACKEND_REGS "${GGML_BACKEND_REGS}extern void ggml_backend_${NAME}_reg();\n")
    set(GGML_BACKEND_INITS "${GGML_BACKEND_INITS}    ggml_backend_${NAME}_reg();\n")
endfunction()
```

**多后端链接顺序处理：**

当同时启用多个后端时，CMake 需要处理符号冲突。例如 CPU 后端和 BLAS 后端都实现了 `ggml_compute_forward_mul_mat`，但通过**弱符号**和**函数指针表**机制避免冲突：

```cpp
// CPU 后端注册时将自己的接口填入全局表
ggml_backend_register(&ggml_backend_cpu) {
    .graph_compute = ggml_cpu_graph_compute,
    .supports_op   = ggml_cpu_supports_op,
}

// BLAS 后端注册时同样填入，但优先级不同
// 调度器根据张量所在 buffer 的类型选择具体后端
```

### 2.6 调试构建配置

```bash
# Debug 构建（单配置生成器，如 Unix Makefiles、Ninja）
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j

# Debug 构建（多配置生成器，如 Xcode）
cmake -B build -G "Xcode"
cmake --build build --config Debug

# 启用 AddressSanitizer
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address"

# 启用 ThreadSanitizer
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=thread"
```

**Debug 构建的关键差异：**
- `GGML_DEBUG` 宏开启，启用额外的断言和日志
- 量化 kernel 使用参考实现而非优化版本（方便比对结果）
- 计算图执行时打印每个节点的 shape 和类型
- 后端缓冲区分配时填充哨兵值（检测越界写入）

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

