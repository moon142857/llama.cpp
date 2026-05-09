# llama.cpp 快速上手指南

> **目标**：5 分钟完成编译、下载模型、运行推理
> **适用平台**：macOS (Apple Silicon) / Linux / Windows
> **版本**：llama.cpp build 9010 (commit d05fe1d7d)

---

## 1. 编译

### 1.1 克隆仓库

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### 1.2 基本编译（CPU 模式）

```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

### 1.3 Apple Silicon（自动启用 Metal）

```bash
cmake -B build -DGGML_CUDA=OFF
cmake --build build --config Release -j
```

### 1.4 启用特定后端

| 后端 | CMake 选项 |
|------|-----------|
| NVIDIA CUDA | `-DGGML_CUDA=ON` |
| AMD HIP | `-DGGML_HIP=ON` |
| Intel SYCL | `-DGGML_SYCL=ON` |
| Vulkan | `-DGGML_VULKAN=ON` |
| BLAS (OpenBLAS) | `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` |

### 1.5 编译产出

```
build/bin/
├── llama-cli       # 命令行交互/推理
├── llama-server    # OpenAI 兼容 API 服务
├── llama-bench     # 性能基准测试
├── llama-quantize  # 模型量化
└── test-*          # 测试套件
```

---

## 2. 下载模型

### 2.1 从 Hugging Face 下载（推荐）

```bash
# 设置镜像（中国大陆）
export HF_ENDPOINT=https://hf-mirror.com

# 使用 llama-cli 直载
./build/bin/llama-cli \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M \
  -p "你好" -n 128

# 或手动下载后使用
# https://hf-mirror.com/unsloth/Qwen3.6-35B-A3B-GGUF
```

### 2.2 模型文件说明

| 文件后缀 | 含义 |
|---------|------|
| `-f16.gguf` | 16-bit 浮点，精度最高 |
| `-q8_0.gguf` | 8-bit 量化，精度很好 |
| `-q4_K_M.gguf` | 4-bit K-quant 混合，推荐通用 |
| `-q4_K_S.gguf` | 4-bit K-quant 小模型，精度略低 |
| `-q2_K.gguf` | 2-bit 量化，极限压缩 |

---

## 3. 运行推理

### 3.1 最简命令

```bash
./build/bin/llama-cli -m model.gguf -p "你好" -n 128
```

### 3.2 交互式对话

```bash
./build/bin/llama-cli \
  -m model.gguf \
  --conversation \
  --temp 0.7 \
  -ngl 0
```

### 3.3 关键参数速查

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `-m` | 模型路径 | `model.gguf` |
| `-p` | 提示词 | `"你好"` |
| `-n` | 最大生成 token 数 | `128` |
| `-c` | 上下文长度 | `4096` |
| `--temp` | 采样温度 | `0.7` |
| `--top-p` | Top-P 采样 | `0.8` |
| `--top-k` | Top-K 采样 | `20` |
| `-ngl` | GPU 卸载层数 | `0`(纯CPU) / `999`(全GPU) |
| `-t` | CPU 线程数 | `4` |
| `--reasoning` | Thinking 模式 | `off` / `on` / `auto` |

### 3.4 GPU 层数选择

```bash
# 纯 CPU（内存不足时）
-ngl 0

# 全部 GPU 卸载（显存充足时）
-ngl 999

# 部分卸载（平衡方案）
-ngl 20  # 卸载前 20 层到 GPU
```

---

## 4. 性能基准测试

```bash
# 标准测试
./build/bin/llama-bench -m model.gguf -p 512 -n 128

# 纯 CPU
./build/bin/llama-bench -m model.gguf -ngl 0 -p 512 -n 128

# 全 GPU
./build/bin/llama-bench -m model.gguf -ngl 999 -p 512 -n 128
```

**输出示例：**

```
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |       4 |           pp512 |        30.19 ± 13.20 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |       4 |           tg128 |          6.17 ± 0.62 |
```

**指标说明：**
- `pp512`：Prompt Processing（512 tokens 的预填充速度）
- `tg128`：Token Generation（128 tokens 的生成速度）
- 单位 `t/s`：tokens per second

---

## 5. 快速参考卡

### 一键启动 Server

```bash
export HF_ENDPOINT=https://hf-mirror.com
./build/bin/llama-server \
  -m model.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 0 -c 4096 --temp 0.7 --reasoning off
```

### 一键命令行对话

```bash
./build/bin/llama-cli \
  -m model.gguf \
  --conversation --reasoning off -ngl 0
```

### 一键性能测试

```bash
./build/bin/llama-bench -m model.gguf -ngl 0 -p 512 -n 128
```

### curl 测试 API

```bash
# 健康检查
curl http://127.0.0.1:8080/health

# 对话生成
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model.gguf",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128
  }'
```

---

## 6. 常见问题

| 问题 | 解决 |
|------|------|
| `Insufficient Memory` | 使用 `-ngl 0` 或减小 `-c` |
| 下载超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| Thinking 模式输出为空 | 添加 `--reasoning off` |
| 编译错误 | 确保 CMake ≥ 3.14，C++ 编译器支持 C++17 |

---

*基于 llama.cpp build 9010 (commit d05fe1d7d)*
