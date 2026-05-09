# llama.cpp 新增模型架构指南

> **目标**：将新的大模型架构接入 llama.cpp
> **版本**：llama.cpp build 9010 (commit d05fe1d7d)
> **日期**：2026-05-04

---

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

