# 📚 Obsidian RAG 系统

一个基于个人Obsidian笔记的增强检索（RAG）系统，帮助你快速查找和学习技术知识点。

## ✨ 特性

- **本地优先**: 所有数据处理在本地完成，保护隐私
- **中文优化**: 使用BAAI/bge-small-zh-v1.5嵌入模型，对中文技术文档优化
- **多模式支持**: 支持本地LM Studio和云API两种LLM模式
- **简单易用**: 命令行界面，类似`hermes agent`的使用体验
- **可扩展**: 支持添加PDF教材和其他文档格式
- **uv管理**: 使用 `uv` 进行依赖管理和虚拟环境

## 🚀 快速开始

### 1. 安装 uv (如果尚未安装)

```bash
# macOS (Homebrew)
brew install uv

# Linux/macOS (curl)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 安装项目依赖

项目使用 `uv` 进行依赖管理，已经配置好 `pyproject.toml` 和 `uv.lock`：

```bash
cd ~/obsidian-rag-system

# 使用 uv 同步依赖（会自动创建/激活虚拟环境）
uv sync

# 或者手动创建虚拟环境
uv venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows
uv pip install -e .
```

### 3. 设置环境

```bash
# 确保配置文件路径正确
# 编辑 config/model_config.yaml 中的 obsidian_vault_path

# 运行设置命令
uv run python main.py init
```

### 4. 使用系统

```bash
# 启动交互式CLI（推荐）
uv run python main.py cli

# 或者直接使用
uv run python -m src.cli --model-type local

# 查看系统配置
uv run python main.py config

# 显示版本信息
uv run python main.py version
```

## 📁 项目结构 (整理后)

```
obsidian-rag-system/
├── main.py              # 主程序入口 (新)
├── pyproject.toml       # uv 项目配置
├── uv.lock              # uv 依赖锁文件
├── src/                 # 🚀 主CLI系统
│   ├── cli.py          # 主CLI程序 (已增强)
│   ├── rag_generator.py # 核心RAG生成器
│   ├── retriever.py    # 检索模块
│   ├── extractors/     # 数据提取器
│   └── ...             # 其他模块
├── config/             # 配置文件
│   └── model_config.yaml
├── data/               # 数据目录
│   ├── raw/           # 原始数据
│   ├── processed/     # 处理后的数据
│   └── embeddings/    # 向量存储
└── README.md           # 本文档
```

## ⚙️ 配置说明

### 数据源配置
```yaml
data_sources:
  obsidian_vault_path: "/Users/fengzhe/Library/Mobile Documents/iCloud~md~obsidian/Documents/知识仓库"
  pdf_paths: []  # PDF教材路径列表
```

### 嵌入模型配置
```yaml
embedding:
  model_name: "BAAI/bge-small-zh-v1.5"  # 中文优化模型
  device: "cpu"  # 或 "cuda" (如果有GPU)
```

### LLM配置
```yaml
llm:
  local:
    enabled: true
    base_url: "http://100.109.51.62:1234"  # 本地LM Studio地址
    model: "deepseek-v3-2-251201"  # 当前使用模型
    timeout: 180  # 长超时设置
```

## 🔧 主要功能

### CLI系统 (src/cli.py)
```bash
# 启动CLI
uv run python -m src.cli --model-type local

# 初始化系统
uv run python -m src.cli init

# 查看配置
uv run python -m src.cli config

# 交互模式
uv run python -m src.cli --model-type local
```

## 🎯 使用场景

### 1. 学习助手
```bash
# 启动CLI
uv run python main.py cli

# 在CLI中查询
> 什么是C++中的虚函数？
> 给我一个C++智能指针的例子
```

### 2. 知识回顾
```bash
# 复习计算机组成原理
> 解释一下补码的概念

# 查看相关笔记
> 列出所有关于指针的笔记
```

## 🔌 集成方式

### 1. 命令行工具
```bash
# 直接使用
uv run python main.py cli

# 作为脚本使用
echo "C++中的模板是什么？" | uv run python -m src.cli --model-type local
```

### 2. LM Studio集成
系统已配置支持本地LM Studio，确保：
1. LM Studio正在运行并提供API服务
2. `config/model_config.yaml` 中的 `base_url` 指向正确的地址
3. 系统会自动设置 `session.trust_env = False` 避免代理问题

## 📊 数据统计

系统已成功提取 83 个Obsidian笔记，保存在 `data/processed/obsidian_notes.json`

运行以下命令查看状态：
```bash
uv run python main.py status
```

显示：
- 系统状态
- 数据统计
- 模型配置
- 向量索引状态

## 🐛 故障排除

### 常见问题

1. **找不到Obsidian仓库**
   ```
   错误: Obsidian仓库不存在
   解决方案: 检查 config/model_config.yaml 中的 obsidian_vault_path
   ```

2. **本地模型连接失败**
   ```
   错误: 无法连接到本地模型
   解决方案: 
   1. 确保LM Studio正在运行
   2. 检查 config/model_config.yaml 中的 base_url
   3. 系统已自动设置 session.trust_env = False 避免代理问题
   ```

3. **uv 命令找不到**
   ```
   错误: command not found: uv
   解决方案: 安装 uv，参考上面的安装步骤
   ```

4. **依赖安装失败**
   ```bash
   # 清理并重新安装
   rm -rf .venv
   uv sync
   ```

### 日志查看
```bash
# 查看系统日志
tail -f data/logs/rag_system.log
```

## 🔄 开发工作流

### 添加新依赖
```bash
# 添加依赖
uv add package-name

# 开发依赖
uv add --dev pytest

# 更新锁文件
uv lock
```

### 运行测试
```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行主程序
python main.py cli

# 退出虚拟环境
deactivate
```

### 更新依赖
```bash
# 更新所有依赖
uv pip compile --upgrade pyproject.toml -o uv.lock
uv sync
```

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📝 许可证

MIT License

## 🙏 致谢

- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) - 中文嵌入模型
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [LM Studio](https://lmstudio.ai/) - 本地LLM运行环境
- [Obsidian](https://obsidian.md/) - 优秀的笔记工具
- [uv](https://github.com/astral-sh/uv) - 快速的Python包管理器和解析器