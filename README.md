# Tinker Study

这是一个用于学习和探索 [Tinker Python SDK](https://github.com/thinking-machines-lab/tinker) 的项目。目前包含一个用于列出 Tinker 平台可用模型的简单脚本。

## 环境要求

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (用于依赖管理)

## 安装与设置

1. **克隆仓库** (如果您尚未克隆):
   ```bash
   git clone <repository-url>
   cd tinker_study
   ```

2. **安装依赖**:
   本项目使用 `uv` 管理依赖。运行以下命令同步环境：
   ```bash
   uv sync
   ```

## 配置

在使用 SDK 之前，您需要配置 API Key。请将其设置为环境变量：

```bash
export TINKER_API_KEY="your_tinker_api_key_here"
```

## 使用方法

### 列出可用模型

运行 `list-models.py` 脚本来查看服务器支持的所有模型：

```bash
uv run list-models.py
```

## 项目结构

- `list-models.py`: 连接 Tinker 服务并打印可用模型列表的脚本。
- `pyproject.toml`: 项目配置文件，包含依赖定义。
- `.python-version`: 指定项目使用的 Python 版本。
