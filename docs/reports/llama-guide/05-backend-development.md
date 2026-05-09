# llama.cpp 后端开发指南

> **目标**：为 llama.cpp 开发新后端（NPU/GPU/ASIC）
> **版本**：llama.cpp build 9010 (commit d05fe1d7d)
> **日期**：2026-05-04

---

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

---

## 核心算子 Kernel 实现参考

以下从 llama.cpp 现有后端（CPU/Metal/CUDA）中提取 MUL_MAT 和 Flash Attention 的实现细节，供新后端开发参考。

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

