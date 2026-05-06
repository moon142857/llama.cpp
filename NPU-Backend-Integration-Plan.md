# 自研 NPU 接入 llama.cpp 技术方案

> **目标**：将自研 NPU 作为新后端接入 llama.cpp，使 Qwen3.6-35B-A3B 等模型能在 NPU 上高效运行  
> **参考实现**：CANN (华为昇腾)、Hexagon (高通)、OpenCL、Metal 后端  
> **版本**：llama.cpp build 9010

---

## 目录

1. [架构总览：后端抽象层](#1-架构总览后端抽象层)
2. [方案选型：静态链接 vs 动态加载](#2-方案选型静态链接-vs-动态加载)
3. [核心接口实现清单](#3-核心接口实现清单)
4. [目录结构与文件组织](#4-目录结构与文件组织)
5. [Step-by-Step 接入流程](#5-step-by-step-接入流程)
6. [算子实现策略](#6-算子实现策略)
7. [内存管理与数据传输](#7-内存管理与数据传输)
8. [CMake 集成](#8-cmake-集成)
9. [测试与验证方案](#9-测试与验证方案)
10. [性能优化建议](#10-性能优化建议)
11. [常见问题与排坑指南](#11-常见问题与排坑指南)

---

## 1. 架构总览：后端抽象层

llama.cpp 的后端系统分三层：

```
┌─────────────────────────────────────────────┐
│           llama 核心库 (src/)                │
│    llama_decode → build_graph → ggml_cgraph │
└─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────┐
│       GGML 后端调度器 (ggml-backend.cpp)      │
│  - 分析计算图，分割子图                        │
│  - 决定每个张量/节点放在哪个后端               │
│  - 自动插入 cross-backend copy 节点           │
└─────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌──────────┐
   │  CPU    │    │  Metal  │    │ 你的 NPU │
   │ 后端    │    │ 后端    │    │ 后端     │
   └─────────┘    └─────────┘    └──────────┘
```

**你的 NPU 后端只需要对接 GGML 后端抽象层**，上层（llama 模型、计算图构建）完全无需改动。

---

## 2. 方案选型：静态链接 vs 动态加载

### 方案 A：静态链接（推荐用于开发阶段）

```
ggml/src/ggml-npu/
├── CMakeLists.txt
├── ggml-npu.cpp          # 后端主实现
├── ggml-npu.h            # 对外头文件
└── kernels/              # NPU 算子实现
    ├── matmul.cpp
    ├── softmax.cpp
    └── ...
```

**优点**：调试方便，编译时链接，无运行时依赖问题  
**缺点**：二进制体积增大，升级 NPU 驱动需重新编译

### 方案 B：动态加载（推荐用于生产部署）

```
ggml/src/ggml-npu/
├── CMakeLists.txt
├── ggml-npu.cpp          # 后端实现
└── ...

# 编译产出
build/bin/libggml-npu.so (Linux)
build/bin/libggml-npu.dylib (macOS)
```

**优点**：驱动升级无需重新编译 llama.cpp；可按需加载  
**缺点**：需要处理运行时符号解析、版本兼容

### 建议

| 阶段 | 推荐方案 |
|------|---------|
| 原型验证 | 静态链接 |
| 内部测试 | 静态链接 |
| 对外发布 | 动态加载 |

---

## 3. 核心接口实现清单

你的 NPU 后端必须实现 `ggml_backend_i` 接口（函数指针表）：

```cpp
// ggml/include/ggml-backend.h

struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void         (*free)(ggml_backend_t backend);
    
    // 缓冲区管理
    ggml_backend_buffer_type_t (*get_default_buffer_type)(ggml_backend_t backend);
    
    // 张量数据拷贝
    void (*set_tensor_async)(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor_async)(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
    
    // 内存同步
    void (*synchronize)(ggml_backend_t backend);
    
    // 计算图执行（最关键）
    ggml_status (*graph_compute)(ggml_backend_t backend, ggml_cgraph * cgraph);
    
    // 可选：异步执行
    ggml_status (*graph_compute_async)(ggml_backend_t backend, ggml_cgraph * cgraph);
    
    // 可选：是否支持某操作
    bool (*supports_op)(ggml_backend_t backend, const ggml_tensor * op);
};
```

### 必须实现的函数（按优先级）

| # | 函数 | 优先级 | 说明 |
|---|------|--------|------|
| 1 | `get_name` | P0 | 返回 "NPU" |
| 2 | `free` | P0 | 释放 NPU 上下文 |
| 3 | `get_default_buffer_type` | P0 | 返回 NPU 内存缓冲区类型 |
| 4 | `graph_compute` | P0 | **核心：执行计算图** |
| 5 | `set_tensor_async` | P0 | Host → NPU 拷入数据 |
| 6 | `get_tensor_async` | P0 | NPU → Host 拷出数据 |
| 7 | `synchronize` | P0 | 等待 NPU 完成 |
| 8 | `supports_op` | P1 | 判断是否支持某算子（不支持则 fallback 到 CPU） |
| 9 | `graph_compute_async` | P1 | 异步执行（性能优化） |

### 缓冲区类型接口

```cpp
struct ggml_backend_buffer_type_i {
    const char * (*get_name)(ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_buffer_type_t buft, size_t size);
    size_t (*get_alignment)(ggml_backend_buffer_type_t buft);
    size_t (*get_max_size)(ggml_backend_buffer_type_t buft);
    // ...
};
```

**关键**：`alloc_buffer` 需要分配 **NPU 设备内存**（而非 CPU 内存）。

---

## 4. 目录结构与文件组织

参考 CANN 和 Hexagon 的简洁结构：

```
ggml/src/ggml-npu/
├── CMakeLists.txt              # 构建配置
├── ggml-npu.cpp                # 后端主实现 (~2000-5000 行)
├── ggml-npu.h                  # 内部头文件
├── npu_device.cpp              # NPU 设备管理（初始化/释放）
├── npu_device.h
├── npu_buffer.cpp              # 缓冲区管理（分配/释放/拷贝）
├── npu_buffer.h
├── npu_ops.cpp                 # 算子分发与实现
├── npu_ops.h
└── kernels/                    # NPU Kernel 实现（可选单独目录）
    ├── npu_matmul.cpp          # 矩阵乘法
    ├── npu_matmul.h
    ├── npu_norm.cpp            # LayerNorm/RMSNorm
    ├── npu_norm.h
    ├── npu_softmax.cpp         # Softmax
    ├── npu_softmax.h
    ├── npu_rope.cpp            # RoPE 位置编码
    ├── npu_rope.h
    ├── npu_moe.cpp             # MoE 路由与 FFN
    ├── npu_moe.h
    └── npu_ssm.cpp             # Delta Net SSM（Qwen3.5 需要）
        └── npu_ssm.h
```

---

## 5. Step-by-Step 接入流程

### Step 1：创建后端骨架（1-2 天）

复制 `ggml/src/ggml-cann/` 或 `ggml/src/ggml-hexagon/` 作为模板：

```bash
cp -r ggml/src/ggml-cann ggml/src/ggml-npu
```

修改 `ggml-npu.cpp` 中的函数名和逻辑，实现最小可用后端：
- `ggml_backend_npu_init()` — 初始化 NPU 设备
- `ggml_backend_npu_free()` — 释放资源
- `ggml_backend_npu_buffer_type()` — 返回缓冲区类型
- `ggml_backend_npu_graph_compute()` — 遍历图并打印节点（先不执行）

验证：编译通过，运行 `llama-cli -ngl 999` 能看到 "NPU" 后端名称。

### Step 2：实现缓冲区管理（2-3 天）

实现 `npu_buffer.cpp`：

```cpp
// 分配 NPU 设备内存
static ggml_backend_buffer_t npu_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * dev_ptr = npu_malloc(size);  // 你的 NPU 内存分配 API
    return ggml_backend_buffer_init(buft, npu_buffer_interface, dev_ptr, size);
}

// Host → NPU 拷贝
static void npu_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, 
                                   const void * data, size_t offset, size_t size) {
    npu_memcpy_h2d(tensor->data + offset, data, size);  // 你的 H2D API
}

// NPU → Host 拷贝
static void npu_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                   void * data, size_t offset, size_t size) {
    npu_memcpy_d2h(data, tensor->data + offset, size);  // 你的 D2H API
}
```

验证：运行 `llama-cli -ngl 0`，模型能在 CPU 加载；运行 `-ngl 999`，模型权重应被分配到 NPU 内存。

### Step 3：实现核心算子（1-2 周）

按优先级逐个实现算子：

```cpp
ggml_status ggml_backend_npu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                npu_op_mul_mat(node);  // 矩阵乘法（最重要）
                break;
            case GGML_OP_ADD:
                npu_op_add(node);      // 逐元素加法
                break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
                npu_op_norm(node);     // LayerNorm / RMSNorm
                break;
            case GGML_OP_SOFT_MAX:
                npu_op_soft_max(node); // Softmax
                break;
            case GGML_OP_ROPE:
                npu_op_rope(node);     // RoPE 位置编码
                break;
            case GGML_OP_MUL:
                npu_op_mul(node);      // 逐元素乘法
                break;
            case GGML_OP_SILU:
                npu_op_silu(node);     // SiLU 激活
                break;
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // 这些只是改变张量视图，无需实际计算
                break;
            case GGML_OP_NONE:
                break;
            default:
                // 不支持的算子，fallback 到 CPU
                fprintf(stderr, "NPU: unsupported op %s, falling back to CPU\n", 
                        ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}
```

### Step 4：实现 MUL_MAT（最关键，占 80% 计算量）

MUL_MAT（矩阵乘法）是 LLM 推理的核心，占 **~80%** 的总计算时间。

```cpp
static void npu_op_mul_mat(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // weight (quantized)
    const ggml_tensor * src1 = dst->src[1];  // input (f32/f16)
    
    // src0 通常是量化权重，需要反量化后乘
    // src1 是输入激活值
    
    // 调用你的 NPU 矩阵乘法 API
    // 示例伪代码：
    npu_matmul_desc_t desc;
    desc.A = src0->data;      // M x K
    desc.B = src1->data;      // K x N
    desc.C = dst->data;       // M x N
    desc.M = dst->ne[1];
    desc.N = dst->ne[0];
    desc.K = src0->ne[0];
    desc.dtype_A = npu_type_from_ggml(src0->type);  // Q4_K → NPU quantized type
    desc.dtype_B = NPU_TYPE_F16;
    desc.dtype_C = NPU_TYPE_F16;
    
    npu_submit_matmul(&desc);
}
```

**关键挑战**：
- 输入权重是 Q4_K / Q5_K 等 GGML 量化格式
- 你需要：**在 NPU 上直接支持量化格式**，或 **在拷贝到 NPU 时反量化为 F16**
- 建议：如果 NPU 硬件支持 INT4/INT8 运算，实现 Q4_K → NPU INT4 的格式转换

### Step 5：实现 MoE 专用算子（Qwen3.5 需要）

Qwen3.5 MoE 需要额外的算子：

```cpp
case GGML_OP_MUL_MAT_ID:   // MoE 路由后的条件矩阵乘法
    npu_op_mul_mat_id(node);
    break;
    
case GGML_OP_SSM_CONV:     // Delta Net 1D 卷积
    npu_op_ssm_conv(node);
    break;
    
case GGML_OP_SSM_SCAN:     // Delta Net 递推扫描
    npu_op_ssm_scan(node);
    break;
```

### Step 6：CMake 集成（见第 8 节）

### Step 7：测试验证（见第 9 节）

---

## 6. 算子实现策略

### 6.1 算子优先级矩阵

| 算子 | Prompt Processing | Token Generation | 优先级 | Qwen3.5 需要 |
|------|------------------|------------------|--------|-------------|
| MUL_MAT | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | P0 | ✅ |
| MUL_MAT_ID | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | P0 | ✅ (MoE) |
| ADD | ⭐⭐ | ⭐⭐ | P1 | ✅ |
| RMS_NORM | ⭐⭐⭐ | ⭐⭐⭐ | P1 | ✅ |
| SOFT_MAX | ⭐⭐⭐ | ⭐⭐⭐ | P1 | ✅ |
| ROPE | ⭐⭐⭐ | ⭐⭐⭐ | P1 | ✅ |
| SILU | ⭐⭐ | ⭐⭐ | P1 | ✅ |
| SSM_CONV | ⭐⭐ | ⭐⭐ | P2 | ✅ (Delta Net) |
| SSM_SCAN | ⭐⭐ | ⭐⭐ | P2 | ✅ (Delta Net) |
| PERMUTE/TRANSPOSE | ⭐ | ⭐ | P3 | ✅ |
| CPY | ⭐⭐ | ⭐⭐ | P1 | ✅ (跨后端拷贝) |

### 6.2 量化权重处理策略

**策略 A：NPU 原生支持 GGML 量化格式（最佳性能）**

```
GGUF 文件中的 Q4_K 权重
        │
        ▼
   [直接传入 NPU]
        │
        ▼
   NPU 硬件解码 Q4_K 并执行 INT4/INT8 运算
```

- 优点：零拷贝，最低内存占用，最高性能
- 缺点：需要 NPU 驱动/编译器支持 GGML 的 block 布局

**策略 B：运行时在 CPU 反量化后传入 NPU（最快实现）**

```
GGUF 文件中的 Q4_K 权重
        │
        ▼
   [CPU 反量化] → F16 缓冲区
        │
        ▼
   [拷贝到 NPU 内存]
        │
        ▼
   NPU 执行 F16 运算
```

- 优点：实现简单，通用性强
- 缺点：NPU 内存占用翻倍（21GB → ~42GB），拷贝耗时

**策略 C：预反量化缓存（折中方案）**

```
GGUF 文件中的 Q4_K 权重
        │
        ▼
   [首次使用时 CPU 反量化] → F16
        │
        ▼
   [缓存到 NPU 内存，后续复用]
```

- 推荐作为**第一版实现**，后续再优化到策略 A

### 6.3 矩阵乘法实现要点

MUL_MAT 的核心是 `Y = X @ W.T`，其中：
- `W` 是权重矩阵（量化，静态）
- `X` 是输入激活（F16/F32，动态）
- `Y` 是输出

**Prompt Processing（batch > 1）**：
```
X: (n_tokens, n_embd)     # 多个 token 一起处理
W: (n_embd, n_out)        # 权重
Y: (n_tokens, n_out)
```

**Token Generation（batch = 1）**：
```
X: (1, n_embd)            # 单个 token
W: (n_embd, n_out)
Y: (1, n_out)
```

**优化建议**：
- Prompt Processing：利用 batch 并行，尽可能并行计算多个 token
- Token Generation：优化单 token 延迟（latency 敏感），使用权重预加载、算子融合

---

## 7. 内存管理与数据传输

### 7.1 内存层级

```
┌─────────────────────────────────────────────┐
│              CPU Host Memory                 │
│  ├─ 模型权重（mmap 映射 GGUF 文件）           │
│  └─ 输入/输出缓冲区                            │
└─────────────────────────────────────────────┘
                        │
                        ▼ H2D 拷贝
┌─────────────────────────────────────────────┐
│              NPU Device Memory               │
│  ├─ 权重张量（反量化后或原生量化）             │
│  ├─ KV Cache（K/V 缓存）                     │
│  ├─ 激活值（中间计算结果）                    │
│  └─ 输出 logits                              │
└─────────────────────────────────────────────┘
```

### 7.2 权重加载策略

```cpp
// 在模型加载阶段，将权重拷贝到 NPU
void npu_load_weights(llama_model * model) {
    for each layer i:
        for each weight tensor in layer[i]:
            if (tensor->buffer->buft != npu_buffer_type) {
                // 在 NPU 上分配内存
                ggml_backend_buffer_t npu_buf = 
                    ggml_backend_buft_alloc_buffer(npu_buffer_type, tensor->nelements * bytes_per_elem);
                
                // 反量化（如果需要）
                if (ggml_is_quantized(tensor->type)) {
                    void * temp_f16 = malloc(tensor->nelements * sizeof(uint16_t));
                    ggml_dequantize_row(tensor->type, tensor->data, temp_f16, tensor->nelements);
                    npu_memcpy_h2d(npu_buf->data, temp_f16, ...);
                    free(temp_f16);
                } else {
                    npu_memcpy_h2d(npu_buf->data, tensor->data, tensor->size);
                }
                
                tensor->buffer = npu_buf;
                tensor->data = npu_buf->data;
            }
}
```

### 7.3 KV Cache 管理

KV Cache 需要在 NPU 内存中分配：

```cpp
// KV Cache 大小计算（以 Qwen3.6-35B-A3B 为例）
// n_layers = 40, n_ctx = 4096, n_embd_k_gqa = 512, n_embd_v_gqa = 512
// K cache: 40 * 4096 * 512 * 2 bytes (f16) = 160 MB
// V cache: 40 * 4096 * 512 * 2 bytes (f16) = 160 MB
// Total: 320 MB

// 在 NPU 上分配 KV Cache
kv_cache.k = ggml_backend_buft_alloc_tensor(npu_buffer_type, ...);
kv_cache.v = ggml_backend_buft_alloc_tensor(npu_buffer_type, ...);
```

### 7.4 减少数据传输

**关键优化**：避免每轮迭代都拷贝输入/输出

```
坏做法（每次 token 都 H2D + D2H）：
  token → CPU buffer → [H2D] → NPU → compute → [D2H] → CPU buffer → token

好做法（输入/输出常驻 NPU）：
  token_id → [H2D, 4 bytes] → NPU → compute → logits → [D2H, 4 bytes] → token_id
  权重和 KV Cache 始终留在 NPU
```

---

## 8. CMake 集成

### 8.1 创建 `ggml/src/ggml-npu/CMakeLists.txt`

```cmake
# ggml/src/ggml-npu/CMakeLists.txt

# 检测 NPU 环境
check_language(CXX)

# 查找 NPU SDK
find_path(NPU_SDK_INCLUDE_DIR npu_runtime.h
    PATHS 
        /usr/local/npu/include
        /opt/npu/include
        $ENV{NPU_SDK}/include
)

find_library(NPU_SDK_LIBRARY npu_runtime
    PATHS
        /usr/local/npu/lib
        /opt/npu/lib
        $ENV{NPU_SDK}/lib
)

if (NOT NPU_SDK_INCLUDE_DIR OR NOT NPU_SDK_LIBRARY)
    message(STATUS "NPU SDK not found, skipping NPU backend")
    return()
endif()

# 定义源文件
set(GGML_NPU_SOURCES
    ggml-npu.cpp
    npu_device.cpp
    npu_buffer.cpp
    npu_ops.cpp
    kernels/npu_matmul.cpp
    kernels/npu_norm.cpp
    kernels/npu_softmax.cpp
    kernels/npu_rope.cpp
    kernels/npu_moe.cpp
    kernels/npu_ssm.cpp
)

# 添加后端库
ggml_add_backend_library(ggml-npu ${GGML_NPU_SOURCES})

# 链接 NPU SDK
target_include_directories(ggml-npu PRIVATE ${NPU_SDK_INCLUDE_DIR})
target_link_libraries(ggml-npu PRIVATE ${NPU_SDK_LIBRARY})

# 添加编译选项
target_compile_definitions(ggml-npu PRIVATE GGML_NPU_DEBUG=$<CONFIG:Debug>)

# 注册后端
ggml_add_backend(npu)
```

### 8.2 修改根 CMakeLists.txt（如果需要）

通常不需要修改根 CMakeLists.txt，因为 `ggml_add_backend(npu)` 会自动注册。

但如果你的 NPU SDK 路径特殊，可以添加选项：

```cmake
# 在根 CMakeLists.txt 或 cmake/common.cmake 中添加
option(GGML_NPU "Enable NPU backend" OFF)
set(NPU_SDK_PATH "" CACHE PATH "Path to NPU SDK")
```

### 8.3 编译命令

```bash
# 启用 NPU 后端编译
cmake -B build \
  -DGGML_NPU=ON \
  -DNPU_SDK_PATH=/opt/npu/sdk \
  -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF

cmake --build build --config Release -j
```

---

## 9. 测试与验证方案

### 9.1 单元测试：后端接口

```cpp
// tests/test-backend-npu.cpp

TEST(npu_backend, init_free) {
    ggml_backend_t backend = ggml_backend_npu_init();
    ASSERT_NE(backend, nullptr);
    ASSERT_STREQ(ggml_backend_name(backend), "NPU");
    ggml_backend_free(backend);
}

TEST(npu_backend, buffer_alloc) {
    ggml_backend_t backend = ggml_backend_npu_init();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, 1024 * 1024);
    ASSERT_NE(buf, nullptr);
    
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
}

TEST(npu_backend, tensor_copy) {
    // Host → NPU → Host 拷贝验证数据一致性
}

TEST(npu_backend, matmul) {
    // 验证 MUL_MAT 结果与 CPU 参考实现一致
}
```

### 9.2 集成测试：单算子验证

使用 `examples/simple/` 或自定义测试：

```bash
# 测试单个 MUL_MAT
./build/bin/test-backend-ops --backend NPU --op MUL_MAT

# 测试全部算子
./build/bin/test-backend-ops --backend NPU
```

### 9.3 模型测试：输出一致性验证

```bash
# 1. CPU 模式生成参考输出
./build/bin/llama-cli -m model.gguf -p "你好" -n 32 -ngl 0 --seed 42 > cpu_output.txt

# 2. NPU 模式生成输出
./build/bin/llama-cli -m model.gguf -p "你好" -n 32 -ngl 999 --seed 42 > npu_output.txt

# 3. 对比输出（应该完全一致）
diff cpu_output.txt npu_output.txt

# 4. 计算 Perplexity
./build/bin/llama-perplexity -m model.gguf -f test.txt -ngl 999
# 与 CPU 的 ppl 对比，差异应 < 1%
```

### 9.4 性能基准

```bash
# 使用 llama-bench 测试 NPU 性能
./build/bin/llama-bench -m model.gguf -ngl 999 -p 512 -n 128

# 对比 CPU 和 NPU
./build/bin/llama-bench -m model.gguf -ngl 0  -p 512 -n 128  # CPU
./build/bin/llama-bench -m model.gguf -ngl 999 -p 512 -n 128  # NPU
```

---

## 10. 性能优化建议

### 10.1 算子融合

将多个小算子合并为一个大 kernel，减少 kernel launch 开销：

```
未融合（6 个 kernel）：
  RMSNorm → MulMat → Add → RMSNorm → MulMat → Add

融合后（2 个 kernel）：
  Fused_RMSNorm_MulMat_Add → Fused_RMSNorm_MulMat_Add
```

### 10.2 权重预加载

在模型加载阶段一次性将所有权重拷贝到 NPU，避免运行时拷贝：

```cpp
// 模型加载时
for each tensor:
    if (tensor on NPU):
        npu_prefetch(tensor->data);  // 预加载到 NPU 高速缓存
```

### 10.3 异步执行

```cpp
ggml_status ggml_backend_npu_graph_compute_async(...) {
    // 提交计算图到 NPU 队列
    npu_submit_graph_async(graph);
    
    // 立即返回，不等待
    return GGML_STATUS_SUCCESS;
}

// 在需要结果时同步
void ggml_backend_npu_synchronize(ggml_backend_t backend) {
    npu_wait_for_completion();
}
```

### 10.4 量化支持

如果 NPU 支持 INT8/INT4：

```cpp
// 在模型加载时将 GGML Q4_K 转换为 NPU 原生 INT4
void npu_convert_weight_q4k_to_npu_int4(ggml_tensor * src, npu_tensor_t * dst) {
    // GGML Q4_K block: 256 weights, 6-bit scales, min values
    // NPU INT4 block: 你的硬件格式
    
    for each block in src:
        decode_ggml_q4k_block(block, &npu_block);
        encode_npu_int4_block(&npu_block, dst);
}
```

### 10.5 内存池

避免频繁分配/释放 NPU 内存：

```cpp
class npu_memory_pool {
    std::vector<void *> free_blocks;
    
    void * allocate(size_t size) {
        // 先从池中查找合适的空闲块
        for each block in free_blocks:
            if (block.size >= size):
                return block.ptr;
        
        // 没有则分配新内存
        return npu_malloc(size);
    }
    
    void free(void * ptr, size_t size) {
        free_blocks.push_back({ptr, size});
    }
};
```

---

## 11. 常见问题与排坑指南

### Q1: 编译时找不到 NPU SDK

**解决**：在 CMake 命令中显式指定路径：
```bash
cmake -B build -DGGML_NPU=ON -DCMAKE_PREFIX_PATH=/opt/npu/sdk
```

### Q2: 模型输出与 CPU 不一致

**排查步骤**：
1. 检查权重拷贝是否正确（H2D 拷贝是否完整）
2. 检查量化格式转换是否正确（Q4_K → F16 是否有精度损失）
3. 逐个算子对比（先测试 MUL_MAT，再测试 ADD，逐步排查）
4. 检查数据类型（F32 vs F16 精度差异）
5. 使用 `--verbose` 打印中间结果

### Q3: NPU 内存不足

**解决**：
```bash
# 减少 offload 层数
./llama-cli -m model.gguf -ngl 20  # 只 offload 前 20 层

# 或减小上下文长度
./llama-cli -m model.gguf -c 2048
```

### Q4: 性能比 CPU 还慢

**排查**：
1. 检查数据传输是否成为瓶颈（H2D/D2H 太频繁）
2. 检查 kernel launch 开销（小算子太多）
3. 检查是否充分使用了 NPU 并行度
4. 使用 profiler 分析 hotspot

### Q5: 某些算子 NPU 不支持

**解决**：实现 `supports_op` 函数，让调度器自动 fallback 到 CPU：

```cpp
bool ggml_backend_npu_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
            return true;
        default:
            return false;  // 不支持的算子 fallback 到 CPU
    }
}
```

---

## 附录：参考后端代码量

| 后端 | 核心代码量 | 复杂度 | 参考价值 |
|------|-----------|--------|----------|
| **Hexagon** (高通) | ~1-2KB | ⭐⭐ | 最简单，适合入门参考 |
| **CANN** (华为昇腾) | ~3KB | ⭐⭐⭐ | 结构清晰，推荐作为模板 |
| **Metal** (Apple) | ~15KB | ⭐⭐⭐⭐ | 功能完整，着色器编译参考 |
| **OpenCL** | ~14KB | ⭐⭐⭐⭐ | 通用 GPU，调度逻辑参考 |
| **CUDA** | ~50KB+ | ⭐⭐⭐⭐⭐ | 最复杂，算子实现参考 |

**建议**：以 **CANN** 或 **Hexagon** 为模板开始，逐步实现到 Metal 的复杂度。

---

*文档生成时间：2026-05-04*  
*基于 llama.cpp commit d05fe1d7d*
