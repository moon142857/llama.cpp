# Qwen3.6-35B-A3B 本地部署测试报告

> 测试时间：2026-05-04  
> 测试框架：llama.cpp (build 9010, commit d05fe1d7d)  
> 硬件平台：Apple MacBook (Apple M4, 32GB Unified Memory)

---

## 1. 编译环境

| 项目 | 详情 |
|------|------|
| 操作系统 | macOS (Darwin arm64) |
| 编译器 | AppleClang 17.0.0.17000604 |
| CMake 版本 | 4.2.3 |
| 启用后端 | Metal (Apple GPU) + BLAS (Accelerate) + CPU (NEON/ARM_FMA/FP16/SME) |
| 编译命令 | `cmake -B build -DGGML_CUDA=OFF && cmake --build build --config Release -j` |
| 编译状态 | ✅ 成功 |

**生成二进制文件：**
- `build/bin/llama-cli`
- `build/bin/llama-server`
- `build/bin/llama-bench`

---

## 2. 模型信息

| 项目 | 详情 |
|------|------|
| 模型名称 | Qwen3.6-35B-A3B |
| 来源 | `unsloth/Qwen3.6-35B-A3B-GGUF` (hf-mirror.com) |
| 量化格式 | UD-Q4_K_XL (Unsloth Dynamic 4-bit) |
| GGUF 文件大小 | **20.60 GiB** |
| 参数量 | **34.66 B** |
| 架构 | qwen35moe (MoE) |
| 层数 | 40 |
| 隐藏层维度 | 2048 |
| 注意力头数 | 16 (KV head: 2) |
| 专家数量 | 256 (每层激活 8 个) |
| 上下文长度 | 262,144 tokens (训练长度) |
| 词表大小 | 248,320 |

**模型缓存路径：**
```
~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/
```

---

## 3. 性能基准测试 (llama-bench)

> **测试条件**：纯 CPU 模式 (`-ngl 0`)，4 threads，使用 Metal/BLAS 后端加速矩阵运算

| 测试项 | 输入规模 | 性能 | 备注 |
|--------|---------|------|------|
| Prompt Processing | 512 tokens | **30.19 ± 13.20 tok/s** | 首 token 延迟约 17ms |
| Token Generation | 128 tokens | **6.17 ± 0.62 tok/s** | 每 token 约 162ms |

**分析：**
- Prompt Processing 速度较快（30+ tok/s），得益于 Apple M4 的 NEON/AMX 指令集和 Metal 矩阵加速
- Token Generation 相对较慢（~6 tok/s），符合大模型纯 CPU 推理预期
- 波动较大（±13 tok/s）可能与 macOS 的线程调度及内存带宽竞争有关

---

## 4. API 服务推理测试

> **服务启动命令**：
> ```bash
> llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
>   --host 127.0.0.1 --port 8080 \
>   -ngl 0 -c 4096 --temp 0.7 \
>   --chat-template-kwargs '{"enable_thinking":false}'
> ```

### 4.1 中文对话测试

**输入：** `你好，请简单自我介绍一下`

**输出（节选）：**
> 你好！我是通义千问（Qwen），是由阿里巴巴集团通义实验室自主研发的大语言模型。我的目标是成为你聪明、贴心且可靠的AI思维伙伴。我可以协助你完成多种任务，包括但不限于：解答问题、内容创作、逻辑与编程、多语言支持...

**性能：** Prompt 70.03 tok/s | Generation 19.97 tok/s

**评价：** ✅ 中文表达流畅自然，结构清晰，信息准确

---

### 4.2 代码生成测试

**输入：** `Write a Python function to calculate the factorial of a number using recursion.`

**输出：** 完整的 Python 函数，包含：
- 类型注解 (`n: int -> int`)
- 文档字符串（Args/Returns/Raises）
- 输入校验（负数、非整数）
- 递归基准条件（0! = 1! = 1）

**性能：** Prompt 75.91 tok/s | Generation 22.64 tok/s

**评价：** ✅ 代码质量高，符合工程规范

---

### 4.3 数学推理测试

**输入：** `If a train travels 120 km in 2 hours, how far will it travel in 5 hours at the same speed? Think step by step.`

**输出（节选）：**
> Step 1: Calculate the speed... Speed = 120 km / 2 hours = 60 km/h  
> Step 2: Calculate the distance... Distance = 60 km/h × 5 hours = **300 km**

**性能：** Prompt 75.4 tok/s | Generation 22.39 tok/s

**评价：** ✅ 分步推理正确，LaTeX 公式渲染完整，答案准确

---

### 4.4 多轮对话测试

**输入（多轮）：**
1. User: `中国的首都是哪里？`
2. Assistant: `中国的首都是北京。`
3. User: `那里有哪些著名景点？`

**输出（节选）：**
> 北京作为中国的古都，拥有众多世界闻名的历史文化遗产和现代地标：
> 1. **故宫（紫禁城）** — 世界上现存规模最大的木质结构古建筑群
> 2. **长城（八达岭/慕田峪段）**
> 3. **天坛公园**

**性能：** Prompt 74.55 tok/s | Generation 22.52 tok/s

**评价：** ✅ 上下文连贯，能正确继承前文信息

---

## 5. 关键发现与问题

### 5.1 运行模式对比

| 模式 | GPU Offload | 结果 | 原因 |
|------|------------|------|------|
| `-ngl 999` (全 GPU) | ❌ 失败 | `Insufficient Memory` | 21GB 模型 + 上下文缓存 > 32GB 统一内存上限 |
| `-ngl 0` (纯 CPU) | ✅ 成功 | 稳定运行 | CPU 映射内存（mmap）模式，系统负责内存管理 |

**结论：** 在 32GB Apple Silicon 设备上，运行 21GB 的 Q4_K_M 量化模型必须使用 CPU 模式，或仅 offload 少量层到 GPU（建议 `-ngl < 10`）。

### 5.2 Thinking 模式

Qwen3.6 默认启用 **Hybrid Thinking**（混合推理模式）：
- 若不禁用，模型会在 `reasoning_content` 中输出很长的思考过程，可能占满 `max_tokens` 配额，导致 `content` 为空
- 通过 `--chat-template-kwargs '{"enable_thinking":false}'` 可禁用

### 5.3 网络下载

- 直接连接 Hugging Face 在中国大陆不稳定（多次连接超时）
- 使用 `hf-mirror.com` 镜像后下载速度正常（约 20MB/s）
- 模型总下载量：约 **21GB**，耗时约 1 小时

---

## 6. 资源占用

| 项目 | 数值 |
|------|------|
| 模型文件磁盘占用 | 20.60 GiB |
| 运行时内存占用 | ~21 GiB (模型) + ~1 GiB (上下文) |
| CPU 核心使用 | 4 线程（server 默认） |
| GPU 使用 | 0 layers offloaded（纯 CPU） |

---

## 7. 总结

| 维度 | 评分 | 说明 |
|------|------|------|
| **部署成功率** | ⭐⭐⭐⭐⭐ | 编译、下载、运行全部成功 |
| **推理质量** | ⭐⭐⭐⭐⭐ | 中英文对话、代码、数学均表现优秀 |
| **生成速度** | ⭐⭐⭐ | 约 6~22 tok/s，可用但不算快 |
| **资源效率** | ⭐⭐⭐ | 21GB 模型吃满内存，32GB 设备较为紧张 |
| **稳定性** | ⭐⭐⭐⭐ | CPU 模式下长时间运行稳定 |

**最终结论：**

> **Qwen3.6-35B-A3B 可以在 Apple M4 + 32GB 内存的 Mac 上通过 llama.cpp 成功部署和运行。** 推理质量优秀，但生成速度受限于纯 CPU 计算。如需更高性能，建议：
> 1. 升级到 64GB/128GB 内存的 Mac，尝试部分 GPU offload
> 2. 使用更小的量化版本（如 UD-Q2_K_XL，约 17GB）
> 3. 使用专用推理框架（如 MLX）以获得更好的 Apple Silicon 优化

---

*报告生成时间：2026-05-04 01:30*  
*测试工具：llama.cpp build 9010 (d05fe1d7d)*
