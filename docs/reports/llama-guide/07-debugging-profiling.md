# llama.cpp 开发调试与性能分析

> **目标**：掌握 llama.cpp 的调试方法、测试框架和性能分析工具
> **版本**：llama.cpp build 9010 (commit d05fe1d7d)
> **日期**：2026-05-04

---

## 12. 开发调试与性能分析

### 12.1 GDB 调试核心流程

**常用断点位置：**

```bash
# 启动 GDB
gdb --args ./build/bin/llama-cli -m model.gguf -p "test" -n 10

# 关键断点
(gdb) break llama_decode                    # 推理入口
(gdb) break llama_model_load_from_file      # 模型加载
(gdb) break llama_sampler_sample            # 采样
(gdb) break ggml_backend_sched_graph_compute_async  # 计算图执行

# 打印张量信息
(gdb) break ggml_compute_forward_mul_mat
(gdb) command
> print *node->src[0]       # 打印输入张量0
> print *node->src[1]       # 打印输入张量1
> print *dst                 # 打印输出张量
> continue
> end

# 打印计算图结构
(gdb) break llama_context::build_graph
(gdb) command
> call ggml_graph_print(graph)    # 打印图的拓扑结构
> continue
> end
```

**观察 KV Cache 状态：**

```bash
(gdb) break llama_kv_cache::find_slot
(gdb) print kv_cache->v_heads[0]       # 当前写入位置
(gdb) print kv_cache->cells[0].pos     # 第一个 cell 的位置
(gdb) print kv_cache->k_l[0]->ne[1]    # 第0层 K cache 的容量
```

### 12.2 计算图调试（`ggml_graph_dump` / `ggml_graph_print`）

```cpp
// 在代码中插入调试输出
// src/llama-context.cpp 中 build_graph 之后：

// 打印图结构到控制台
ggml_graph_print(graph);

// 将图导出为 Graphviz DOT 格式（可用 Graphviz 可视化）
FILE * dot = fopen("graph.dot", "w");
ggml_graph_dump_dot(graph, NULL, dot);
fclose(dot);
// 然后运行: dot -Tpng graph.dot -o graph.png

// 验证图的合法性（检查循环依赖、shape 一致性）
ggml_graph_compute_with_ctx(ctx, graph, n_threads);  // 会打印每个节点的执行时间
```

**计算图节点信息解读：**

```
===== GRAPH =====
n_nodes = 1256
- node   0: [     2048,     4096,        1,        1]       MUL_MAT
  src 0: [     2048,     2048]          Q4_0    attn_q.weight
  src 1: [     2048,        1]          F32     inp_embd
- node   1: [     2048,     1024,        1,        1]       MUL_MAT
- node   2: [     2048,     4096,        1,        1]       ADD
...
```

### 12.3 测试框架使用

**后端算子测试（`test-backend-ops`）：**

```bash
# 编译测试（默认已编译）
cmake --build build --target test-backend-ops

# 测试所有后端的所有算子
./build/bin/test-backend-ops

# 只测试 CPU 后端的 MUL_MAT
./build/bin/test-backend-ops --backend CPU --op MUL_MAT

# 只测试 Metal 后端的量化类型
./build/bin/test-backend-ops --backend Metal --type Q4_0 --type Q4_K

# 测试特定矩阵大小
./build/bin/test-backend-ops --backend CPU --op MUL_MAT --ne0 4096 --ne1 4096 --ne2 1

# 输出详细对比（发现不一致时打印参考值和实际值）
./build/bin/test-backend-ops --backend Metal --verbose
```

**Tokenizer 测试：**

```bash
# 测试 tokenizer（验证分词结果与参考一致）
./build/bin/test-tokenizer-0 ./models/ggml-vocab-llama-spm.gguf

# 测试 chat template
./build/bin/test-chat-template ./models/ggml-vocab-qwen.gguf
```

**完整 CI 本地运行：**

```bash
mkdir -p tmp/results tmp/mnt

# CPU-only CI
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# 带 CUDA
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# 带 SYCL（需要 oneAPI 环境）
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

### 12.4 AddressSanitizer / ThreadSanitizer

```bash
# ASan 构建（检测内存越界、use-after-free）
cmake -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
cmake --build build-asan -j

# 运行（ASan 会在检测到错误时打印详细堆栈）
./build-asan/bin/llama-cli -m model.gguf -p "test"

# TSan 构建（检测数据竞争）
cmake -B build-tsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="-fsanitize=thread" \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread"
cmake --build build-tsan -j
```

**常见 ASan 报错与排查：**

| 报错类型 | 含义 | 常见原因 |
|----------|------|----------|
| `heap-buffer-overflow` | 堆内存越界 | 张量 shape 计算错误、量化 block 大小不匹配 |
| `stack-buffer-overflow` | 栈内存越界 | 局部数组太小、字符串操作越界 |
| `use-after-free` | 使用已释放内存 | KV Cache cell 被复用后仍被引用 |
| `memory-leak` | 内存泄漏 | `ggml_context` 未释放、后端 buffer 泄漏 |

### 12.5 性能分析工具

**llama.cpp 内置性能计时：**

```bash
# 启用详细计时
./build/bin/llama-cli -m model.gguf -p "test" --verbose

# Server 模式下的 token 级计时
./build/bin/llama-server -m model.gguf --show-timings
```

输出示例：
```
prompt eval time =    1234.56 ms /    64 tokens (   19.29 ms per token,    51.84 tokens per second)
       eval time =    5678.90 ms /   128 tokens (   44.37 ms per token,    22.54 tokens per second)
      total time =    6913.46 ms /   192 tokens
```

**系统级性能分析：**

```bash
# perf（Linux）- 查看热点函数
perf record -g ./build/bin/llama-cli -m model.gguf -p "test" -n 128
perf report --sort=dso,symbol

# Instruments（macOS）- Time Profiler
time_profiler ./build/bin/llama-cli -m model.gguf -p "test" -n 128

# Tracy 帧分析（如果编译时启用 -DLLAMA_TRACY）
# 需要运行 Tracy Profiler GUI 连接
```

**Metal GPU 性能分析（macOS）：**

```bash
# 使用 Xcode Instruments 的 Metal System Trace
xcrun xctrace record --template "Metal System Trace" \
  --launch -- ./build/bin/llama-cli -m model.gguf -p "test"

# 或使用 Metal GPU Counters
xcrun xctrace record --template "Metal GPU Counters" \
  --launch -- ./build/bin/llama-cli -m model.gguf -p "test"
```

**CUDA Nsight：**

```bash
# Nsight Systems（时间线分析）
nsys profile -o llama_profile ./build/bin/llama-cli -m model.gguf -p "test"

# Nsight Compute（kernel 级详细分析）
ncu -o llama_kernel ./build/bin/llama-cli -m model.gguf -p "test"
```

### 12.6 调试日志系统

llama.cpp 使用宏控制日志级别，可在运行时通过环境变量或参数调整：

```bash
# 命令行参数控制
./llama-cli -m model.gguf -v              # 最高详细级别
./llama-cli -m model.gguf --verbosity 4   # 调试级别

# 环境变量控制
export GGML_LOG_LEVEL=debug
./llama-cli -m model.gguf

# 日志写入文件
./llama-cli -m model.gguf --log-file debug.log --log-timestamps
```

**自定义 trace 日志（开发时使用）：**

```cpp
// 在源码中添加临时 trace
#define MY_TRACE(fmt, ...) \
    fprintf(stderr, "[MY_TRACE] %s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

// 使用示例
MY_TRACE("kv_cache size=%d head=%d", kv_cache.size, kv_cache.head);
MY_TRACE("graph n_nodes=%d n_leafs=%d", graph->n_nodes, graph->n_leafs);
```

### 12.7 常见问题排查手册

#### 问题1：输出结果与 transformers/HF 不一致

**排查步骤：**
1. 确认 GGUF 转换正确：`test-tokenizer-0` 验证分词一致性
2. 确认超参数一致：对比 `hparams` 中的 `rms_norm_eps`, `rope_theta`, `attention_scale`
3. 禁用采样：`--temp 0 --top-k 1 --top-p 1`（贪心解码，消除随机性）
4. 检查量化精度：用 `f16` 模型对比，排除量化误差
5. 逐层对比中间结果：在 `llama_decode` 后打印每层输出的 hash

```bash
# 逐层对比脚本思路
./llama-cli -m model.gguf -p "test" --dump-layer-output 0 > layer0.txt
./llama-cli -m model.gguf -p "test" --dump-layer-output 1 > layer1.txt
# 与 transformers 的对应层输出对比
```

#### 问题2：特定后端崩溃（Metal/CUDA）

```bash
# 1. 先用 CPU 后端确认模型本身正确
./llama-cli -m model.gguf -p "test" -ngl 0

# 2. 逐步增加 offload 层数，定位崩溃层
for i in 1 5 10 20 40; do
    echo "Testing ngl=$i"
    ./llama-cli -m model.gguf -p "test" -ngl $i -n 10 || break
done

# 3. 检查后端缓冲区分配
./llama-cli -m model.gguf -p "test" -ngl 999 --verbose 2>&1 | grep "buffer"
```

#### 问题3：内存不足（OOM）

```bash
# 查看模型内存需求
./llama-cli -m model.gguf -p "test" -ngl 0 --verbose 2>&1 | grep -E "(buffer|size|MiB)"

# 减少上下文长度
./llama-cli -m model.gguf -p "test" -c 2048   # 默认可能是 4096 或更大

# 使用 mmap（默认开启，允许 OS 换出）
./llama-cli -m model.gguf -p "test" --mmap

# 禁用 mlock（避免强制驻留）
./llama-cli -m model.gguf -p "test" --no-mlock

# 减少 batch size（Server 模式）
./llama-server -m model.gguf --ctx-size 2048 --parallel 2
```

#### 问题4：性能比预期慢

```bash
# 1. 确认后端正确加载
./llama-cli -m model.gguf -p "test" --verbose 2>&1 | grep -i "backend\|metal\|cuda\|blas"

# 2. 确认线程数
./llama-cli -m model.gguf -p "test" -t 8   # 显式指定线程数

# 3. 检查是否触发 fallback（某算子不支持当前后端）
./llama-cli -m model.gguf -p "test" --verbose 2>&1 | grep "fallback\|CPU"

# 4. 使用 llama-bench 获取标准化数据
./llama-bench -m model.gguf -p 512 -n 128
```

### 12.8 提交代码前的检查清单

根据 `CONTRIBUTING.md` 和 CI 要求：

```bash
# 1. 代码格式化
git-clang-format HEAD~1   # 检查最近 commit 的格式
# 或手动格式化修改的文件
clang-format -i src/models/mymodel.cpp src/llama-arch.cpp

# 2. Python 代码检查
cd gguf-py && flake8

# 3. 编译通过（多后端）
cmake -B build -DGGML_CUDA=ON -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j

# 4. 运行核心测试
cd build && ctest -L main --verbose --timeout 900

# 5. 运行 tokenizer 测试
ctest -R test-tokenizer --verbose

# 6. Perplexity 验证（与参考值对比）
./build/bin/llama-perplexity -m model.gguf -f wiki.test.raw
# 对比未修改版本的 ppl，差异应 < 1%

# 7. 本地完整 CI（如果修改影响广泛）
bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

---

