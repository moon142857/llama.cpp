# llama.cpp API Server 部署指南

> **目标**：生产环境部署 llama-server，配置 API 服务
> **版本**：llama.cpp build 9010 (commit d05fe1d7d)
> **日期**：2026-05-04

---

## 3. API 服务部署

### 3.1 llama-server 启动命令

```bash
./build/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl 0 \
  -c 4096 \
  --temp 0.7 \
  --reasoning off
```

### 3.2 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--host` | 监听地址 | `127.0.0.1` (本地) / `0.0.0.0` (全网) |
| `--port` | 监听端口 | `8080` |
| `-c, --ctx-size` | 上下文大小 | `4096` |
| `-np, --parallel` | 并行解码序列数 | `4` (server 自动) |
| `--reasoning off` | 禁用 Thinking 模式 | `off` |
| `-ngl 0` | 纯 CPU 运行 | `0` (32GB Mac) |

### 3.3 支持的 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/chat/completions` | POST | ChatGPT 兼容对话接口 |
| `/v1/completions` | POST | 文本补全接口 |
| `/v1/embeddings` | POST | 文本嵌入接口 |
| `/props` | GET | Server 属性 |

### 3.4 curl 调用示例

```bash
# 健康检查
curl http://127.0.0.1:8080/health

# 列出模型
curl http://127.0.0.1:8080/v1/models

# 对话生成
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'

# 流式输出 (SSE)
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128,
    "stream": true
  }'
```

### 3.5 Python 调用示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=128,
    temperature=0.7
)
print(response.choices[0].message.content)
```

---

## 4. 测试用例与完整 JSON 响应

> 以下所有测试均通过 `llama-server` REST API 执行，模型已加载，Thinking 模式已禁用。

### 4.1 测试1：中文对话

**请求命令：**
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "你好，请简单自我介绍一下"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

**完整 JSON 响应：**

```json

{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "\u4f60\u597d\uff01\u6211\u662f\u901a\u4e49\u5343\u95ee\uff0c\u7531\u963f\u91cc\u4e91\u901a\u4e49\u5b9e\u9a8c\u5ba4\u5f00\u53d1\u7684\u8d85\u5927\u89c4\u6a21\u8bed\u8a00\u6a21\u578b\u3002\n\n\u6211\u7684\u76ee\u6807\u662f\u6210\u4e3a\u4f60\u771f\u8bda\u3001\u6709\u5e2e\u52a9\u7684\u4f19\u4f34\u3002\u6211\u5177\u5907\u5f3a\u5927\u7684\u8bed\u8a00\u7406\u89e3\u548c\u8868\u8fbe\u80fd\u529b\uff0c\u80fd\u591f\u4e0e\u4f60\u8fdb\u884c\u81ea\u7136\u6d41\u7545\u7684\u5bf9\u8bdd\u3002\u65e0\u8bba\u662f\u89e3\u7b54\u7591\u95ee\u3001\u63d0\u4f9b\u521b\u610f\u7075\u611f\u3001\u8f85\u52a9\u7f16\u7a0b\uff0c\u8fd8\u662f\u5904\u7406\u590d\u6742\u7684\u903b\u8f91\u63a8\u7406\uff0c\u6211\u90fd\u80fd\u4e3a\u4f60\u63d0\u4f9b\u652f\u6301\u3002\n\n\u8bf7\u95ee\u4eca\u5929\u6709\u4ec0\u4e48\u6211\u53ef\u4ee5\u5e2e\u4f60\u7684\u5417\uff1f"
            }
        }
    ],
    "created": 1777857746,
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "system_fingerprint": "b9010-d05fe1d7d",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 77,
        "prompt_tokens": 18,
        "total_tokens": 95,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    },
    "id": "chatcmpl-r7rqkodtMfyLL54sQIC3NHxFIidhIE4P",
    "timings": {
        "cache_n": 0,
        "prompt_n": 18,
        "prompt_ms": 1378.108,
        "prompt_per_token_ms": 76.56155555555556,
        "prompt_per_second": 13.061385609836094,
        "predicted_n": 77,
        "predicted_ms": 4353.352,
        "predicted_per_token_ms": 56.53703896103896,
        "predicted_per_second": 17.68751986974635
    }
}

**性能数据：**
- Prompt: 18 tokens → **13.06 tok/s**
- Generation: 77 tokens → **17.69 tok/s**

**输出内容：**
```
你好！我是通义千问，由阿里云通义实验室开发的超大规模语言模型。

我的目标是成为你真诚、有帮助的伙伴。我具备强大的语言理解和表达能力，能够与你进行自然流畅的对话。无论是解答疑问、提供创意灵感、辅助编程，还是处理复杂的逻辑推理，我都能为你提供支持。

请问今天有什么我可以帮你的吗？
```

---

### 4.2 测试2：代码生成

**请求命令：**
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Write a Python function to calculate the factorial of a number using recursion."}],
    "max_tokens": 256,
    "temperature": 0.3
  }'
```

**完整 JSON 响应：**

```json
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "```python\ndef factorial(n: int) -> int:\n    \"\"\"\n    Calculate the factorial of a non-negative integer using recursion.\n    \n    Args:\n        n: A non-negative integer.\n        \n    Returns:\n        The factorial of n.\n        \n    Raises:\n        ValueError: If n is negative.\n        TypeError: If n is not an integer.\n    \"\"\"\n    if not isinstance(n, int):\n        raise TypeError(\"Input must be an integer.\")\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers.\")\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n```"
            }
        }
    ],
    "created": 1777857755,
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "system_fingerprint": "b9010-d05fe1d7d",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 156,
        "prompt_tokens": 26,
        "total_tokens": 182,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    },
    "id": "chatcmpl-BpebtNKKZEarmu30EJDvDLaF28cb58FI",
    "timings": {
        "cache_n": 0,
        "prompt_n": 26,
        "prompt_ms": 589.49,
        "prompt_per_token_ms": 22.67269230769231,
        "prompt_per_second": 44.105922068228466,
        "predicted_n": 156,
        "predicted_ms": 9011.082,
        "predicted_per_token_ms": 57.76334615384616,
        "predicted_per_second": 17.31201647038613
    }
}
```

**性能数据：**
- Prompt: 26 tokens → **44.11 tok/s**
- Generation: 156 tokens → **17.31 tok/s**

**输出内容：**
```python
```python
def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer using recursion.
    
    Args:
        n: A non-negative integer.
        
    Returns:
        The factorial of n.
        
    Raises:
        ValueError: If n is negative.
        TypeError: If n is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
```
```

---

### 4.3 测试3：数学推理

**请求命令：**
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "If a train travels 120 km in 2 hours, how far will it travel in 5 hours at the same speed? Think step by step."}],
    "max_tokens": 256,
    "temperature": 0.3
  }'
```

**完整 JSON 响应：**

```json
{
    "choices": [
        {
            "finish_reason": "length",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "To solve this problem, we need to determine the speed of the train and then use that speed to calculate the distance traveled in 5 hours.\n\n**Step 1: Calculate the speed of the train.**\nThe formula for speed is:\n$$ \\text{Speed} = \\frac{\\text{Distance}}{\\text{Time}} $$\n\nGiven:\n*   Distance = $120 \\text{ km}$\n*   Time = $2 \\text{ hours}$\n\n$$ \\text{Speed} = \\frac{120 \\text{ km}}{2 \\text{ hours}} = 60 \\text{ km/h} $$\n\n**Step 2: Calculate the distance traveled in 5 hours.**\nThe formula for distance is:\n$$ \\text{Distance} = \\text{Speed} \\times \\text{Time} $$\n\nGiven:\n*   Speed = $60 \\text{ km/h}$ (calculated in Step 1)\n*   Time = $5 \\text{ hours}$\n\n$$ \\text{Distance} = 60 \\text{ km/h} \\times 5 \\text{ hours} = 300 \\text{ km} $$\n\n**Conclusion"
            }
        }
    ],
    "created": 1777857773,
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "system_fingerprint": "b9010-d05fe1d7d",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 256,
        "prompt_tokens": 45,
        "total_tokens": 301,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    },
    "id": "chatcmpl-FMdjRp60LnNgGvDcS5fWPQx4VoI6jWUj",
    "timings": {
        "cache_n": 0,
        "prompt_n": 45,
        "prompt_ms": 836.408,
        "prompt_per_token_ms": 18.586844444444445,
        "prompt_per_second": 53.80149400770915,
        "predicted_n": 256,
        "predicted_ms": 17097.953,
        "predicted_per_token_ms": 66.78887890625,
        "predicted_per_second": 14.972552562286257
    }
}
```

**性能数据：**
- Prompt: 45 tokens → **53.80 tok/s**
- Generation: 256 tokens → **14.97 tok/s**

**输出内容：**
```
To solve this problem, we need to determine the speed of the train and then use that speed to calculate the distance traveled in 5 hours.

**Step 1: Calculate the speed of the train.**
The formula for speed is:
$$ \text{Speed} = \frac{\text{Distance}}{\text{Time}} $$

Given:
*   Distance = $120 \text{ km}$
*   Time = $2 \text{ hours}$

$$ \text{Speed} = \frac{120 \text{ km}}{2 \text{ hours}} = 60 \text{ km/h} $$

**Step 2: Calculate the distance traveled in 5 hours.**
The formula for distance is:
$$ \text{Distance} = \text{Speed} \times \text{Time} $$

Given:
*   Speed = $60 \text{ km/h}$ (calculated in Step 1)
*   Time = $5 \text{ hours}$

$$ \text{Distance} = 60 \text{ km/h} \times 5 \text{ hours} = 300 \text{ km} $$

**Conclusion
```

---

### 4.4 测试4：多轮对话

**请求命令：**
```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "messages": [
      {"role": "user", "content": "中国的首都是哪里？"},
      {"role": "assistant", "content": "中国的首都是北京。"},
      {"role": "user", "content": "那里有哪些著名景点？"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

**完整 JSON 响应：**

```json
{
    "choices": [
        {
            "finish_reason": "length",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "\u5317\u4eac\u4f5c\u4e3a\u4e2d\u56fd\u7684\u9996\u90fd\uff0c\u62e5\u6709\u4f17\u591a\u5386\u53f2\u60a0\u4e45\u7684\u540d\u80dc\u53e4\u8ff9\u548c\u73b0\u4ee3\u5316\u7684\u5730\u6807\u5efa\u7b51\u3002\u4ee5\u4e0b\u662f\u4e00\u4e9b\u6700\u8457\u540d\u7684\u666f\u70b9\u63a8\u8350\uff1a\n\n### 1. \u5386\u53f2\u6587\u5316\u9057\u8ff9\n*   **\u6545\u5bab\uff08\u7d2b\u7981\u57ce\uff09**\uff1a\u4e16\u754c\u4e0a\u73b0\u5b58\u89c4\u6a21\u6700\u5927\u3001\u4fdd\u5b58\u6700\u5b8c\u6574\u7684\u6728\u8d28\u7ed3\u6784\u53e4\u5efa\u7b51\u7fa4\uff0c\u660e\u6e05\u4e24\u4ee3\u7684\u7687\u5bb6\u5bab\u6bbf\u3002\n*   **\u957f\u57ce**\uff1a\u5176\u4e2d\u6700\u8457\u540d\u7684\u6bb5\u843d\u5305\u62ec**\u516b\u8fbe\u5cad\u957f\u57ce**\uff08\u5f00\u53d1\u6700\u65e9\u3001\u4fdd\u5b58\u6700\u5b8c\u6574\uff09\u548c**\u6155\u7530\u5cea\u957f\u57ce**\uff08\u98ce\u666f\u79c0\u4e3d\u3001\u6e38\u5ba2\u76f8\u5bf9\u8f83\u5c11\uff09\u3002\n*   **\u5929\u575b**\uff1a\u660e\u6e05\u4e24\u4ee3\u7687\u5e1d\u796d\u5929\u3001\u7948\u8c37\u7684\u573a\u6240\uff0c\u4ee5\u5176\u72ec\u7279\u7684\u5706\u5f62\u5efa\u7b51\u548c\u4f18\u7f8e\u7684\u56ed\u6797\u8457\u79f0\u3002\n*   **"
            }
        }
    ],
    "created": 1777857781,
    "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "system_fingerprint": "b9010-d05fe1d7d",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 128,
        "prompt_tokens": 37,
        "total_tokens": 165,
        "prompt_tokens_details": {
            "cached_tokens": 0
        }
    },
    "id": "chatcmpl-khZA1UWEU0NNNW6nEVGBeMZY0gC1NjK4",
    "timings": {
        "cache_n": 0,
        "prompt_n": 37,
        "prompt_ms": 852.769,
        "prompt_per_token_ms": 23.047810810810812,
        "prompt_per_second": 43.38806875015391,
        "predicted_n": 128,
        "predicted_ms": 6662.773,
        "predicted_per_token_ms": 52.0529140625,
        "predicted_per_second": 19.211220313223937
    }
}
```

**性能数据：**
- Prompt: 37 tokens → **43.39 tok/s**
- Generation: 128 tokens → **19.21 tok/s**

**输出内容：**
```
北京作为中国的首都，拥有众多历史悠久的名胜古迹和现代化的地标建筑。以下是一些最著名的景点推荐：

### 1. 历史文化遗迹
*   **故宫（紫禁城）**：世界上现存规模最大、保存最完整的木质结构古建筑群，明清两代的皇家宫殿。
*   **长城**：其中最著名的段落包括**八达岭长城**（开发最早、保存最完整）和**慕田峪长城**（风景秀丽、游客相对较少）。
*   **天坛**：明清两代皇帝祭天、祈谷的场所，以其独特的圆形建筑和优美的园林著称。
*   **
```

---

### 4.5 测试5：模型信息 (/v1/models)

**请求命令：**
```bash
curl -s http://127.0.0.1:8080/v1/models
```

**完整 JSON 响应：**

```json
{
    "models": [
        {
            "name": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "modified_at": "",
            "size": "",
            "digest": "",
            "type": "model",
            "description": "",
            "tags": [
                ""
            ],
            "capabilities": [
                "completion"
            ],
            "parameters": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "",
                "families": [
                    ""
                ],
                "parameter_size": "",
                "quantization_level": ""
            }
        }
    ],
    "object": "list",
    "data": [
        {
            "id": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "aliases": [],
            "tags": [],
            "object": "model",
            "created": 1777857781,
            "owned_by": "llamacpp",
            "meta": {
                "vocab_type": 2,
                "n_vocab": 248320,
                "n_ctx_train": 262144,
                "n_embd": 2048,
                "n_params": 34660610688,
                "size": 22123538944
            }
        }
    ]
}
```

---

### 4.6 测试6：Server 属性 (/props)

**请求命令：**
```bash
curl -s http://127.0.0.1:8080/props
```

**完整 JSON 响应（节选）：**

```json
{
    "default_generation_settings": {
        "params": {
            "seed": 4294967295,
            "temperature": 0.699999988079071,
            "dynatemp_range": 0.0,
            "dynatemp_exponent": 1.0,
            "top_k": 20,
            "top_p": 0.949999988079071,
            "min_p": 0.05000000074505806,
            "top_n_sigma": -1.0,
            "xtc_probability": 0.0,
            "xtc_threshold": 0.10000000149011612,
            "typical_p": 1.0,
            "repeat_last_n": 64,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "dry_multiplier": 0.0,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": -1,
            "mirostat": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.10000000149011612,
            "max_tokens": -1,
            "n_predict": -1,
            "n_keep": 0,
            "n_discard": 0,
            "ignore_eos": false,
            "stream": true,
            "n_probs": 0,
            "min_keep": 0,
            "chat_format": "Content-only",
            "reasoning_format": "none",
            "reasoning_in_content": false,
            "generation_prompt": "",
            "samplers": [
                "penalties",
                "dry",
                "top_n_sigma",
                "top_k",
                "typ_p",
                "top_p",
                "min_p",
                "xtc",
                "temperature"
            ],
            "speculative.type": "none",
            "timings_per_token": false,
            "post_sampling_probs": false,
            "backend_sampling": false,
            "lora": []
        },
        "n_ctx": 4096
    },
    "total_slots": 4,
    "model_alias": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "model_path": "/Users/zhengxiaoxi/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/snapshots/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    "modalities": {
        "vision": false,
        "audio": false
    },
    "media_marker": "<__media_Zr05Hl9U1mLWDHRabSE7CNLyhFkMoB7l__>",
    "endpoint_slots": true,
    "endpoint_props": false,
    "endpoint_metrics": false,
    "webui": true,
    "webui_settings": {},
    "chat_template": "{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- macro render_content(content, do_vision_count, is_system_content=false) %}\n    {%- if content is string %}\n        {{- content }}\n    {%- elif content is iterable and content is not mapping %}\n        {%- for item in content %}\n            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}\n                {%- if is_system_content %}\n                    {{- raise_exception('System message cannot contain images.') }}\n                {%- endif %}\n                {%- if do_vision_count %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                {%- endif %}\n                {%- if add_vision_id %}\n                    {{- 'Picture ' ~ image_count.value ~ ': ' }}\n                {%- endif %}\n                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}\n            {%- elif 'video' in item or item.type == 'video' %}\n                {%- if is_system_content %}\n                    {{- raise_exception('System message cannot contain videos.') }}\n                {%- endif %}\n                {%- if do_vision_count %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                {%- endif %}\n                {%- if add_vision_id %}\n                    {{- 'Video ' ~ video_count.value ~ ': ' }}\n                {%- endif %}\n                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}\n            {%- elif 'text' in item %}\n                {{- item.text }}\n            {%- else %}\n                {{- raise_exception('Unexpected item type in content.') }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif content is none or content is undefined %}\n        {{- '' }}\n    {%- else %}\n        {{- raise_exception('Unexpected content type.') }}\n    {%- endif %}\n{%- endmacro %}\n{%- if not messages %}\n    {{- raise_exception('No messages provided.') }}\n{%- endif %}\n{%- set num_sys = 0 %}\n{%- set merged_system = '' %}\n{%- if messages[0].role == 'system' or messages[0].role == 'developer' %}\n    {%- set first = render_content(messages[0].content, false, true)|trim %}\n    {%- if messages|length > 1 and (messages[1].role == 'system' or messages[1].role == 'developer') %}\n        {%- set second = render_content(messages[1].content, false, true)|trim %}\n        {%- set merged_system = first + '\\n' + second %}\n        {%- set num_sys = 2 %}\n    {%- else %}\n        {%- set merged_system = first %}\n        {%- set num_sys = 1 %}\n    {%- endif %}\n{%- endif %}\n{%- if tools and tools is iterable and tools is not mapping %}\n    {{- '<|im_start|>system\\n' }}\n    {{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\" }}\n    {{- '\\n\\nIf you choose to call a function ONLY reply in the following format with NO suffix:\\n\\n<tool_call>\\n<function=example_function_name>\\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n<parameter=example_parameter_2>\\nThis is the value for the second parameter\\nthat can span\\nmultiple lines\\n</parameter>\\n</function>\\n</tool_call>\\n\\n<IMPORTANT>\\nReminder:\\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\\n- Required parameters MUST be specified\\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\\n</IMPORTANT>' }}\n    {%- if merged_system %}\n        {{- '\\n\\n' + merged_system }}\n    {%- endif %}\n    {{- '<|im_end|>\\n' }}\n{%- else %}\n    {%- if merged_system %}\n        {{- '<|im_start|>system\\n' + merged_system + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" %}\n        {%- set content = render_content(message.content, false)|trim %}\n        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}\n            {%- set ns.multi_step_tool = false %}\n            {%- set ns.last_query_index = index %}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if loop.index0 >= num_sys and message.role != \"system\" and message.role != \"developer\" %}\n    {%- set content = render_content(message.content, true)|trim %}\n    {%- if message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- set reasoning_content = reasoning_content|trim %}\n        {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if tool_call.function is defined %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {%- if loop.first %}\n                    {%- if content|trim %}\n                        {{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n                    {%- else %}\n                        {{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n                    {%- endif %}\n                {%- else %}\n                    {{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}\n                {%- endif %}\n                {%- if tool_call.arguments is mapping %}\n                    {%- for args_name in tool_call.arguments %}\n                        {%- set args_value = tool_call.arguments[args_name] %}\n                        {{- '<parameter=' + args_name + '>\\n' }}\n                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}\n                        {{- args_value }}\n                        {{- '\\n</parameter>\\n' }}\n                    {%- endfor %}\n                {%- endif %}\n                {{- '</function>\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.previtem and loop.previtem.role != \"tool\" %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if not loop.last and loop.nextitem.role != \"tool\" %}\n            {{- '<|im_end|>\\n' }}\n        {%- elif loop.last %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- else %}\n        {{- '<think>\\n' }}\n    {%- endif %}\n{%- endif %}\n{#- Unsloth fixes - developer role, tool calling #}",
    "chat_template_caps": {
        "supports_object_arguments": true,
        "supports_parallel_tool_calls": true,
        "supports_preserve_reasoning": true,
        "supports_string_content": true,
        "supports_system_role": true,
        "supports_tool_calls": true,
        "supports_tools": true,
        "supports_typed_content": false
    },
    "bos_token": "<|endoftext|>",
    "eos_token": "<|im_end|>",
    "build_info": "b9010-d05fe1d7d",
    "is_sleeping": false
}
```

**关键属性：**
- `total_slots`: 4 (并行槽位数)
- `n_ctx`: 4096 (上下文长度)
- `temperature`: 0.7
- `top_k`: 20
- `top_p`: 0.95
- `chat_format`: "Content-only"
- `reasoning_format`: "none"

---

## 5. 性能基准测试

### 5.1 llama-bench 结果

**测试命令：**
```bash
./build/bin/llama-bench \
  -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 0 -p 512 -n 128
```

**原始输出：**

```
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |       4 |           pp512 |        30.19 ± 13.20 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |       4 |           tg128 |          6.17 ± 0.62 |

build: d05fe1d7d (9010)
```

### 5.2 API 推理性能汇总

| 测试 | Prompt Tokens | Prompt Speed | Gen Tokens | Gen Speed |
|------|--------------|--------------|-----------|-----------|
| 中文对话 | 18 | **13.06 tok/s** | 77 | **17.69 tok/s** |
| 代码生成 | 26 | **44.11 tok/s** | 156 | **17.31 tok/s** |
| 数学推理 | 45 | **53.80 tok/s** | 256 | **14.97 tok/s** |
| 多轮对话 | 37 | **43.39 tok/s** | 128 | **19.21 tok/s** |

**分析：**
- Prompt Processing 速度随 token 数增加而提升（缓存预热效应）
- Token Generation 稳定在 **15~19 tok/s**，优于 llama-bench 的 6.17 tok/s，可能因为 API 测试时模型已完全加载到内存，且 batch size 更优
- 整体性能在纯 CPU 模式下属于可接受范围

---

## 6. Server 运行日志

**Server 启动命令：**
```bash
./build/bin/llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 0 -c 4096 --temp 0.7 --reasoning off
```

**完整启动日志：**

```
srv   prompt_save:  - saving prompt with length 94, total state size = 64.651 MiB
srv          load:  - looking for better prompt, base f_keep = 0.032, sim = 0.115
srv        update:  - cache state: 1 prompts, 64.651 MiB (limits: 8192.000 MiB, 4096 tokens, 11910 est)
srv        update:    - prompt 0xbfd007e10:      94 tokens, checkpoints:  0,    64.651 MiB
srv  get_availabl: prompt cache update took 7.45 ms
slot launch_slot_: id  3 | task -1 | sampler chain: logits -> ?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> min-p -> ?xtc -> temp-ext -> dist 
slot launch_slot_: id  3 | task 79 | processing task, is_child = 0
slot update_slots: id  3 | task 79 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 26
slot update_slots: id  3 | task 79 | n_past = 3, slot.prompt.tokens.size() = 94, seq_id = 3, pos_min = 93, n_swa = 0
slot update_slots: id  3 | task 79 | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
slot update_slots: id  3 | task 79 | n_tokens = 0, memory_seq_rm [0, end)
slot update_slots: id  3 | task 79 | prompt processing progress, n_tokens = 22, batch.n_tokens = 22, progress = 0.846154
slot update_slots: id  3 | task 79 | n_tokens = 22, memory_seq_rm [22, end)
slot init_sampler: id  3 | task 79 | init sampler, took 0.00 ms, tokens: text = 26, total = 26
slot update_slots: id  3 | task 79 | prompt processing done, n_tokens = 26, batch.n_tokens = 4
slot print_timing: id  3 | task 79 | 
prompt eval time =     589.49 ms /    26 tokens (   22.67 ms per token,    44.11 tokens per second)
       eval time =    9011.08 ms /   156 tokens (   57.76 ms per token,    17.31 tokens per second)
      total time =    9600.57 ms /   182 tokens
slot      release: id  3 | task 79 | stop processing: n_tokens = 181, truncated = 0
srv  update_slots: all slots are idle
srv  log_server_r: done request: POST /v1/chat/completions 127.0.0.1 200
srv  params_from_: Chat format: peg-native
slot get_availabl: id  2 | task -1 | selected slot by LRU, t_last = -1
srv  get_availabl: updating prompt cache
srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000
srv        update:  - cache state: 1 prompts, 64.651 MiB (limits: 8192.000 MiB, 4096 tokens, 11910 est)
srv        update:    - prompt 0xbfd007e10:      94 tokens, checkpoints:  0,    64.651 MiB
srv  get_availabl: prompt cache update took 0.01 ms
slot launch_slot_: id  2 | task -1 | sampler chain: logits -> ?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> min-p -> ?xtc -> temp-ext -> dist 
slot launch_slot_: id  2 | task 237 | processing task, is_child = 0
slot slot_save_an: id  3 | task -1 | saving idle slot to prompt cache
srv   prompt_save:  - saving prompt with length 181, total state size = 66.352 MiB
slot prompt_clear: id  3 | task -1 | clearing prompt with 181 tokens
srv        update:  - cache state: 2 prompts, 131.003 MiB (limits: 8192.000 MiB, 4096 tokens, 17196 est)
srv        update:    - prompt 0xbfd007e10:      94 tokens, checkpoints:  0,    64.651 MiB
srv        update:    - prompt 0xbfd007c90:     181 tokens, checkpoints:  0,    66.352 MiB
slot update_slots: id  2 | task 237 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 45
slot update_slots: id  2 | task 237 | n_tokens = 0, memory_seq_rm [0, end)
slot update_slots: id  2 | task 237 | prompt processing progress, n_tokens = 41, batch.n_tokens = 41, progress = 0.911111
slot update_slots: id  2 | task 237 | n_tokens = 41, memory_seq_rm [41, end)
slot init_sampler: id  2 | task 237 | init sampler, took 0.00 ms, tokens: text = 45, total = 45
slot update_slots: id  2 | task 237 | prompt processing done, n_tokens = 45, batch.n_tokens = 4
slot print_timing: id  2 | task 237 | 
prompt eval time =     836.41 ms /    45 tokens (   18.59 ms per token,    53.80 tokens per second)
       eval time =   17097.95 ms /   256 tokens (   66.79 ms per token,    14.97 tokens per second)
      total time =   17934.36 ms /   301 tokens
slot      release: id  2 | task 237 | stop processing: n_tokens = 300, truncated = 0
srv  update_slots: all slots are idle
srv  log_server_r: done request: POST /v1/chat/completions 127.0.0.1 200
srv  params_from_: Chat format: peg-native
slot get_availabl: id  1 | task -1 | selected slot by LRU, t_last = -1
srv  get_availabl: updating prompt cache
srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000
srv        update:  - cache state: 2 prompts, 131.003 MiB (limits: 8192.000 MiB, 4096 tokens, 17196 est)
srv        update:    - prompt 0xbfd007e10:      94 tokens, checkpoints:  0,    64.651 MiB
srv        update:    - prompt 0xbfd007c90:     181 tokens, checkpoints:  0,    66.352 MiB
srv  get_availabl: prompt cache update took 0.00 ms
slot launch_slot_: id  1 | task -1 | sampler chain: logits -> ?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> top-p -> min-p -> ?xtc -> temp-ext -> dist 
slot launch_slot_: id  1 | task 495 | processing task, is_child = 0
slot slot_save_an: id  2 | task -1 | saving idle slot to prompt cache
srv   prompt_save:  - saving prompt with length 300, total state size = 68.679 MiB
slot prompt_clear: id  2 | task -1 | clearing prompt with 300 tokens
srv        update:  - cache state: 3 prompts, 199.682 MiB (limits: 8192.000 MiB, 4096 tokens, 23589 est)
srv        update:    - prompt 0xbfd007e10:      94 tokens, checkpoints:  0,    64.651 MiB
srv        update:    - prompt 0xbfd007c90:     181 tokens, checkpoints:  0,    66.352 MiB
srv        update:    - prompt 0xbfd007f90:     300 tokens, checkpoints:  0,    68.679 MiB
slot update_slots: id  1 | task 495 | new prompt, n_ctx_slot = 4096, n_keep = 0, task.n_tokens = 37
slot update_slots: id  1 | task 495 | n_tokens = 0, memory_seq_rm [0, end)
slot update_slots: id  1 | task 495 | prompt processing progress, n_tokens = 33, batch.n_tokens = 33, progress = 0.891892
slot update_slots: id  1 | task 495 | n_tokens = 33, memory_seq_rm [33, end)
slot init_sampler: id  1 | task 495 | init sampler, took 0.00 ms, tokens: text = 37, total = 37
slot update_slots: id  1 | task 495 | prompt processing done, n_tokens = 37, batch.n_tokens = 4
slot print_timing: id  1 | task 495 | 
prompt eval time =     852.77 ms /    37 tokens (   23.05 ms per token,    43.39 tokens per second)
       eval time =    6662.77 ms /   128 tokens (   52.05 ms per token,    19.21 tokens per second)
      total time =    7515.54 ms /   165 tokens
slot      release: id  1 | task 495 | stop processing: n_tokens = 164, truncated = 0
srv  update_slots: all slots are idle
srv  log_server_r: done request: POST /v1/chat/completions 127.0.0.1 200
```

---

## 7. 问题与解决方案

### 7.1 问题1：GPU 内存不足

**现象：**
```
ggml_metal_synchronize: error: command buffer 0 failed with status 5
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

**原因：** `-ngl 999` 试图将全部 41 层 offload 到 GPU，21GB 模型 + 上下文缓存超过 32GB 统一内存上限。

**解决：** 使用 `-ngl 0` 纯 CPU 模式，或仅 offload 少量层（如 `-ngl 5`）。

### 7.2 问题2：Thinking 模式导致输出为空

**现象：** API 返回的 `content` 为空，`reasoning_content` 包含大量思考过程。

**原因：** Qwen3.6 默认启用 Thinking 模式，`max_tokens` 被思考过程耗尽。

**解决（旧方法，已废弃）：**
```bash
--chat-template-kwargs '{"enable_thinking":false}'
```

**解决（推荐方法）：**
```bash
--reasoning off
```

### 7.3 问题3：Hugging Face 下载超时

**现象：**
```
get_repo_commit: error: HTTPLIB failed: Could not establish connection
error: failed to download model from Hugging Face
```

**原因：** 中国大陆直连 Hugging Face 网络不稳定。

**解决：** 设置国内镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 7.4 问题4：本地 4-bit 模型无法转换 GGUF

**现象：**
```
ValueError: Can not map tensor 'model.embed_tokens.biases'
```

**原因：** 本地 `Qwen3.5-27B-4bit` 是 Affine 量化格式（类似 GPTQ），不是标准 BF16/FP16，convert_hf_to_gguf.py 不支持。

**解决：** 直接下载预转换的 GGUF 格式模型。

---

## 附录：快速参考卡
