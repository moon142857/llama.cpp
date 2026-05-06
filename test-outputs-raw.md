# Qwen3.6-35B-A3B 测试原始输出

## 1. llama-bench 原始输出

```
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |       4 |           pp512 |        30.19 ± 13.20 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |       4 |           tg128 |          6.17 ± 0.62 |

build: d05fe1d7d (9010)
```

---

## 2. 测试1：中文对话

**请求：**
```json
{
  "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  "messages": [{"role": "user", "content": "你好，请简单自我介绍一下"}],
  "max_tokens": 128,
  "temperature": 0.7
}
```

**响应（提取）：**
```
Content: 你好！我是通义千问（Qwen），是由阿里巴巴集团通义实验室自主研发的大语言模型。
我的目标是成为你聪明、贴心且可靠的AI思维伙伴。我可以协助你完成多种任务，包括但不限于：
*   **解答问题**：无论是专业知识查询还是日常疑问，我都会尽力提供准确、清晰的回答。
*   **内容创作**：帮你撰写邮件、文章、故事、剧本或进行创意头脑风暴。
*   **逻辑与编程**：协助进行复杂推理、数据分析以及编写、理解和调试代码。
*   **多语言支持**：支持全球一百多种语言的流畅

Perf: prompt= 70.03 tok/s, gen= 19.97 tok/s
```

---

## 3. 测试2：代码生成

**请求：**
```json
{
  "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  "messages": [{"role": "user", "content": "Write a Python function to calculate the factorial of a number using recursion."}],
  "max_tokens": 256,
  "temperature": 0.3
}
```

**响应（提取）：**
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

**性能：** prompt= 75.91 tok/s, gen= 22.64 tok/s

---

## 4. 测试3：数学推理

**请求：**
```json
{
  "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  "messages": [{"role": "user", "content": "If a train travels 120 km in 2 hours, how far will it travel in 5 hours at the same speed? Think step by step."}],
  "max_tokens": 256,
  "temperature": 0.3
}
```

**响应（提取）：**
```
To determine how far the train will travel in 5 hours, we need to first find its speed and then apply that speed to the new time duration.

Step 1: Calculate the speed of the train.
The formula for speed is:
$$ Speed = Distance / Time $$

Given:
*   Distance = 120 km
*   Time = 2 hours

$$ Speed = 120 km / 2 hours = 60 km/h $$

Step 2: Calculate the distance traveled in 5 hours.
Now that we know the train travels at a constant speed of 60 km/h, we can calculate the distance for 5 hours using the formula:
$$ Distance = Speed x Time $$

Given:
*   Speed = 60 km/h
*   Time = 5 hours

$$ Distance = 60 km/h x 5 hours = 300 km $$
```

**性能：** prompt= 75.4 tok/s, gen= 22.39 tok/s

---

## 5. 测试4：多轮对话

**请求：**
```json
{
  "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  "messages": [
    {"role": "user", "content": "中国的首都是哪里？"},
    {"role": "assistant", "content": "中国的首都是北京。"},
    {"role": "user", "content": "那里有哪些著名景点？"}
  ],
  "max_tokens": 128,
  "temperature": 0.7
}
```

**响应（提取）：**
```
北京作为中国的古都，拥有众多世界闻名的历史文化遗产和现代地标。以下是一些最著名且不容错过的景点：

### 历史与文化古迹
1. **故宫（紫禁城）**  
   世界上现存规模最大、保存最完整的木质结构古建筑群，明清两代的皇家宫殿，现在是故宫博物院。
2. **长城（八达岭/慕田峪段）**  
   "不到长城非好汉"中的长城，八达岭段开发成熟、交通方便；慕田峪段风景更秀丽、游客相对较少。
3. **天坛公园**  
   明清皇帝祭
```

**性能：** prompt= 74.55 tok/s, gen= 22.52 tok/s
