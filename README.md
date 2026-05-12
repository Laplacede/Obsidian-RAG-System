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

## 🔄 数据流

这个项目的数据流可以分成 5 步：

```text
┌──────────────────────────────┐
│ Obsidian 笔记库 (*.md)        │
└──────────────┬───────────────┘
               │
               v
┌──────────────────────────────────────┐
│ 提取器                                │
│ src/extractors/obsidian_extractor.py │
└───────────┬──────────────────────────┘
            │
            v
┌──────────────────────────────────────────┐
│ data/processed/obsidian_notes.json       │
└───────────┬──────────────────────────────┘
            │
            v
┌──────────────────────────────────┐
│ 分块器                            │
│ src/chunkers/semantic_chunker.py │
└───────────┬──────────────────────┘
            │
            v
┌──────────────────────────────────────────┐
│ data/chunks/*.json                       │
│ small / medium / large                   │
└──────────────┬───────────────────────────┘
               │
     ┌─────────┴───────────────────┐
     │                             │
     v                             v
┌────────────────────┐  ┌──────────────────────┐
│ 向量库构建           │  │ 检索器                │
│ data/vector_store/ │  │ src/retriever.py     │
└─────────┬──────────┘  └──────┬───────────────┘
          │                    │
          └──────────┬─────────┘
                     v
         ┌──────────────────────┐
         │ 重排序器              │
         │ src/reranker.py      │
         └──────────┬───────────┘
                    v
         ┌──────────────────────┐
         │ LLM 生成器            │
         │ src/rag_generator.py │
         └──────────┬───────────┘
                    v
         ┌──────────────────────┐
         │ CLI / main.py 输出回答│
         └──────────────────────┘
```

### 1. 笔记提取
- 读取 Obsidian 仓库中的所有 Markdown 文件。
- 解析 YAML frontmatter、wikilinks 和正文标签。
- 输出结构化笔记到 `data/processed/obsidian_notes.json`。

### 2. 语义分块
- 读取 `data/processed/obsidian_notes.json`。
- 按标题、段落、列表、引用和代码块等边界切分文本。
- 生成三种粒度的分块：`small`、`medium`、`large`。
- 输出到 `data/chunks/small_chunks.json`、`data/chunks/medium_chunks.json`、`data/chunks/large_chunks.json`。

### 3. 向量化与索引
- 分块结果会被写入向量库目录 `data/vector_store/`。
- 检索阶段会同时读取分块 JSON 和向量索引。
- 标签、标题和正文关键词也会进入关键词索引，辅助召回。

### 4. 检索与重排序
- 用户问题先经过查询扩展，再进行向量检索和关键词检索。
- 检索结果会经过可选重排序，优先保留更相关的 chunk。
- 这一层主要由 `src/retriever.py` 和 `src/reranker.py` 负责。

### 5. 生成回答
- 检索到的 chunk 会被拼接成上下文。
- `src/rag_generator.py` 把上下文和问题一起送入 LLM。
- 最终由 `main.py` 或 `src/cli.py` 展示答案、引用和运行状态。

### 相关文件
- [src/extractors/obsidian_extractor.py](src/extractors/obsidian_extractor.py)
- [src/chunkers/semantic_chunker.py](src/chunkers/semantic_chunker.py)
- [src/retriever.py](src/retriever.py)
- [src/reranker.py](src/reranker.py)
- [src/rag_generator.py](src/rag_generator.py)
- [src/cli.py](src/cli.py)
- [main.py](main.py)

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
      base_url: "http://100.109.51.62:1234"  # LM Studio 地址，不带 /api/v1/chat 后缀
      model: "qwen/qwen3.5-9b"  # 需要和 /v1/models 返回的 id 一致
      timeout: 1200  # 大模型建议延长超时
      max_tokens: 1024
```

> 说明：当前本地 LM Studio 已验证可用的接口是 `/api/v1/chat`。生成器会按旧接口格式发送 `input` 和 `system_prompt`，并从 `output -> message -> content` 中读取最终答案。

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

   你也可以直接验证旧接口是否返回答案：
   ```bash
   curl -X POST http://100.109.51.62:1234/api/v1/chat \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen/qwen3.5-9b","input":"请只回复 ok","system_prompt":"你是一个助手。"}'
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