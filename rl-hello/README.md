# Hello RL

基于 Tinker 云端训练 API 的简单强化学习示例。

## 任务说明

训练模型学会对 "Say you are {task}" 的提示回复 "I am {task}"。

## 特性

- 使用 `importance_sampling` 损失函数实现 REINFORCE 算法
- 自动计算优势函数（advantage = reward - mean_batch_reward）
- 支持 LoRA 微调
- 支持从 checkpoint 恢复训练
- 实时打印 prompt 和 completion 用于调试

## 奖励函数

| 情况 | 奖励 |
|------|------|
| 回复包含 "I am {task}" 或 "I'm {task}" | +1.0 |
| 回复包含 "not {task}" | -1.0 |
| 其他 | 0.0 |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 32 | 每批次 prompt 数量 |
| `SAMPLES_PER_PROMPT` | 4 | 每个 prompt 采样次数（group_size） |
| `NUM_TRAIN_TASKS` | 1000 | 训练任务总数 |
| `SAVE_EVERY` | 20 | 每 N 个 batch 保存 checkpoint |
| `LEARNING_RATE` | 1e-5 | 学习率 |

## 使用方法

```bash
# 方式一：使用 .env 文件（项目根目录）
# 在 .env 中添加：TINKER_API_KEY=your_api_key

# 方式二：直接设置环境变量
export TINKER_API_KEY=your_api_key

# 运行训练
cd hello_rl
uv run train.py
```

## 从 Checkpoint 恢复

训练会每 `SAVE_EVERY` 个 batch 保存 checkpoint，日志示例：
```
Saved checkpoint 'batch_000020': {'sampler_path': 'tinker://xxx/sampler_weights/batch_000020'}
```

恢复训练：
```bash
uv run train.py --resume tinker://xxx/sampler_weights/batch_000020 --start-batch 20
```

> **注意**：必须使用 `sampler_weights` 路径（用于采样），不是 `weights` 路径。

## 参考

- [tinker-cookbook/rl](https://github.com/thinking-machines-lab/tinker-cookbook)
- [agent-lightning/examples/tinker/hello.py](https://github.com/microsoft/agent-lightning)
