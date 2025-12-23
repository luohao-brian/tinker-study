# Prompt Distillation

Prompt Distillation（上下文蒸馏）是一种训练方法，让 LLM "将 prompt 内化到参数中"。

## 原理

1. **Teacher 生成数据**: 给定复杂分类 prompt $p$ + 查询 $q$，teacher 模型生成响应 $r$
2. **Student 学习**: Student 通过 SFT 训练，学会仅根据 $q$（不需要 $p$）预测 $r$

训练后，student 模型无需长 prompt 也能正确预测语言标签。

### 示例

训练前需要完整 prompt:
```
Query: Classify the language: "一生、バンドしてくれる？"
Response: ja
```

训练后只需输入文本:
```
Query: 一生、バンドしてくれる？
Response: ja
```

## 使用方法

### 环境配置

```bash
export TINKER_API_KEY="your-api-key"
```

### Step 0: 下载多语言数据

下载 tinker-cookbook 的多语言样本数据:

```bash
curl -o data/multilingual.txt https://raw.githubusercontent.com/thinking-machines-lab/tinker-cookbook/main/tinker_cookbook/example_data/multilingual.txt
```

### Step 1: 生成蒸馏数据

使用 teacher 模型生成语言分类数据:

```bash
uv run python create_data.py --output_file ./output/distillation_data.jsonl
```

参数说明:
- `--model`: Teacher 模型名称 (默认: `Qwen/Qwen3-30B-A3B`)
- `--input_file`: 输入多语言文本文件
- `--output_file`: 输出 JSONL 文件路径
- `--temperature`: 采样温度 (默认: 0.15)

### Step 2: 训练 Student 模型

使用蒸馏数据进行 SFT 训练:

```bash
uv run python sft.py --data_file ./output/distillation_data.jsonl
```

参数说明:
- `--model_name`: 模型名称 (默认: `Qwen/Qwen3-30B-A3B`)
- `--data_file`: 训练数据 JSONL 文件
- `--learning_rate`: 学习率 (默认: 1e-4)
- `--num_epochs`: 训练轮数 (默认: 4)
- `--batch_size`: Batch 大小 (默认: 128)
- `--lora_rank`: LoRA 秩 (默认: 32)
- `--save_every`: 保存 checkpoint 间隔 (默认: 20)

### Step 3: 测试模型

训练完成后，从保存的 checkpoint 加载模型进行测试。

## 数据格式

蒸馏数据为 JSONL 格式，每行一个样本:

```json
{"messages": [{"role": "user", "content": "你好世界"}, {"role": "assistant", "content": "zh"}]}
{"messages": [{"role": "user", "content": "Hello world"}, {"role": "assistant", "content": "en"}]}
```

## 支持的语言标签

| 代码 | 语言 |
|------|------|
| ar | Arabic |
| de | German |
| el | Greek |
| en | English |
| es | Spanish |
| fr | French |
| hi | Hindi |
| ru | Russian |
| tr | Turkish |
| ur | Urdu |
| vi | Vietnamese |
| zh | Chinese |
| ot | Other/Unknown |

