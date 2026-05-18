# 📚 Obsidian RAG 系统

一个基于个人Obsidian笔记的增强检索（RAG）系统，帮助你快速查找和学习技术知识点。

## ✨ 核心特性

- **本地优先**: 所有数据处理在本地完成，保护隐私
- **中文优化**: 使用BAAI/bge-small-zh-v1.5嵌入模型，对中文技术文档优化
- **多模式支持**: 支持本地LM Studio、火山引擎、OpenAI API等多种LLM后端
- **智能检索**: 混合检索（语义+关键词）+ 查询扩展 + 可选重排序
- **RAG模式**: 支持严格模式（仅知识库）和灵活模式（知识库优先+补充）
- **完整CLI**: 交互式命令行界面，支持命令历史、配置管理
- **开发友好**: 使用 `uv` 进行依赖管理和虚拟环境
- **可扩展架构**: 模块化设计，易于添加新数据源和模型

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

### 3. 配置系统

```bash
# 复制配置模板
cd config
cp model_config.yaml.example model_config.yaml

# 编辑配置文件，设置你的Obsidian仓库路径和模型配置
# 详细配置指南见 CONFIG_GUIDE.md

# 运行设置命令
uv run python main.py init
```

### 4. 启动系统

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

## 📊 当前开发状态

### ✅ 已完成功能

1. **数据提取与处理**
   - Obsidian笔记提取（支持YAML frontmatter、wikilinks、标签）
   - 语义分块（small/medium/large三种粒度）
   - 向量嵌入生成（BAAI/bge-small-zh-v1.5）

2. **检索系统**
   - 混合检索：语义向量检索 + 关键词检索
   - 查询扩展：同义词和技术术语扩展
   - 可选重排序：基于交叉编码器的重排序

3. **生成系统**
   - 多后端LLM支持：本地LM Studio、火山引擎、OpenAI API、模拟模式
   - RAG两种模式：严格模式（仅知识库）和灵活模式（知识库优先+补充）
   - 置信度评估：自动计算答案置信度

4. **用户界面**
   - 完整CLI交互系统
   - 命令历史记录
   - 系统状态查看
   - 配置管理

5. **辅助工具**
   - 反馈收集系统
   - 系统评估工具
   - 日志系统
   - 连接诊断工具

### 📈 数据统计

系统已处理：
- 83个Obsidian笔记
- 618个语义分块（small: 354, medium: 155, large: 109）
- 支持中文技术文档的向量嵌入

### 🔄 后续计划

- [ ] PDF教材支持
- [ ] Web界面
- [ ] 知识图谱构建
- [ ] 多语言支持优化
- [ ] 性能优化和缓存

## 📁 项目架构

```
obsidian-rag-system/
├── main.py              # 主程序入口
├── pyproject.toml       # uv 项目配置
├── uv.lock              # uv 依赖锁文件
├── src/                 # 🚀 核心系统
│   ├── cli.py           # 主CLI程序
│   ├── rag_generator.py # 核心RAG生成器（多模式支持）
│   ├── retriever.py     # 检索模块（混合检索+查询扩展）
│   ├── reranker.py      # 重排序模块
│   ├── feedback.py      # 反馈收集系统
│   ├── evaluate.py      # 系统评估工具
│   ├── embedding_generator.py # 向量嵌入生成
│   ├── extractors/      # 数据提取器
│   │   └── obsidian_extractor.py
│   ├── chunkers/        # 分块器
│   │   └── semantic_chunker.py
│   └── generators/      # LLM生成器
│       ├── base.py
│       ├── mock_generator.py
│       ├── local_lm_studio.py
│       ├── volcengine_generator.py
│       └── openai_generator.py
├── config/              # 配置文件
│   ├── model_config.yaml.example  # 配置模板
│   └── model_config.yaml          # 实际配置（.gitignore忽略）
├── data/               # 数据目录（.gitignore忽略）
│   ├── raw/           # 原始数据
│   ├── processed/     # 处理后的数据
│   ├── chunks/        # 分块数据
│   ├── embeddings/    # 向量存储
│   ├── vector_store/  # ChromaDB向量数据库
│   ├── feedback/      # 反馈数据
│   └── evaluation/    # 评估结果
├── docs/              # 文档
│   ├── CONFIG_GUIDE.md  # 配置指南
│   ├── RAG_MODES.md     # RAG模式说明
│   └── USAGE.md         # 使用指南
└── tests/             # 测试文件
```

## 🔄 系统数据流

```text
┌─────────────────┐
│ 数据源          │
│ • Obsidian笔记  │
│ • PDF教材       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 数据提取器      │
│ • 解析markdown  │
│ • 提取元数据    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 语义分块器      │
│ • 三种粒度      │
│ • 语义边界      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 向量化          │
│ • 中文优化模型  │
│ • 向量存储      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 混合检索系统    │
│ • 向量检索      │
│ • 关键词检索    │
│ • 查询扩展      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ RAG生成器       │
│ • 严格/灵活模式 │
│ • 多LLM后端     │
│ • 置信度评估    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 用户界面        │
│ • CLI交互       │
│ • 结果展示      │
└─────────────────┘
```

## ⚙️ 配置说明

### 数据源配置
```yaml
data_sources:
  obsidian_vault_path: "/path/to/your/obsidian/vault"
  pdf_paths: []  # PDF教材路径列表
```

### 嵌入模型配置
```yaml
embedding:
  model_name: "BAAI/bge-small-zh-v1.5"  # 中文优化模型
  device: "cpu"  # 或 "cuda" (如果有GPU)
```

### LLM配置（多后端支持）
```yaml
llm:
  # 本地LM Studio模式
  local:
    enabled: true
    base_url: "http://localhost:1234"
    model: "your-model-name"
  
  # 火山引擎模式
  volcengine:
    enabled: false
    api_key: "your-api-key"
    base_url: "https://ark.cn-beijing.volces.com/api/v3"
    model: "deepseek-v3-2-251201"
  
  # OpenAI API模式
  cloud:
    enabled: false
    provider: "openai"
    api_key: "your-api-key"
    model: "gpt-4"
```

### RAG模式配置
```yaml
rag:
  mode: "flexible"  # 'strict' 或 'flexible'
  confidence:
    high_quality_min_docs: 3
    high_quality_min_score: 0.7
```

## 🎯 使用场景

### 1. 学习助手
```bash
# 启动CLI
uv run python main.py cli

# 在CLI中查询
> 什么是C++中的虚函数？
> 给我一个C++智能指针的例子
> 解释一下补码的概念
```

### 2. 知识回顾
```bash
# 复习特定主题
> 列出所有关于数据结构的笔记
> 计算机网络中的TCP/IP协议是什么？

# 查看相关概念
> 指针和引用的区别是什么？
```

### 3. 技术文档查询
```bash
# API文档查询
> Python的requests库如何使用？
> JavaScript的Promise有哪些方法？

# 代码示例
> 给我一个Python装饰器的例子
```

## 🔌 集成方式

### 1. 命令行工具
```bash
# 直接使用
uv run python main.py cli

# 作为脚本使用
echo "C++中的模板是什么？" | uv run python -m src.cli --model-type local

# 批处理模式
cat questions.txt | while read question; do
  echo "$question" | uv run python -m src.cli --model-type local
done
```

### 2. 程序化调用
```python
from src.rag_generator import RAGGenerator
from src.retriever import Retriever

# 初始化系统
retriever = Retriever(config_path="config/model_config.yaml")
generator = RAGGenerator(config_path="config/model_config.yaml")

# 执行查询
contexts = retriever.retrieve("你的问题")
answer = generator.generate("你的问题", contexts)
```

## 📝 RAG模式详解

### 严格模式 (Strict Mode)
- **特点**：仅基于知识库回答，信息不足时明确说明
- **适用场景**：需要完全符合知识库内容，不希望模型使用先验知识
- **输出示例**：`知识库中没有相关信息`

### 灵活模式 (Flexible Mode) - 推荐
- **特点**：优先使用知识库，不足时补充先验知识
- **适用场景**：希望系统更智能、更有用，知识库是参考而非唯一来源
- **输出示例**：
  ```
  [知识库] 根据我的笔记，Promises用于处理异步操作...
  [补充] 此外，Promises通常与async/await一起使用...
  
  置信度: 0.50 🟠 低
  RAG模式: 灵活模式（知识库优先+补充）
  ```

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
   3. 运行测试脚本：python test_lm_studio_connection.py
   ```

3. **依赖安装失败**
   ```bash
   # 清理并重新安装
   rm -rf .venv
   uv sync
   ```

4. **权限问题**
   ```bash
   # 确保有读取Obsidian笔记的权限
   ls "/path/to/your/obsidian/vault"
   ```

### 诊断工具

系统提供了多个诊断工具：
```bash
# 测试LM Studio连接
uv run python test_lm_studio_connection.py

# 测试生成器
uv run python test_generator_directly.py

# 诊断连接问题
uv run python diagnose_lm_studio.py
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

# 运行单元测试
python -m pytest tests/

# 退出虚拟环境
deactivate
```

### 更新依赖
```bash
# 更新所有依赖
uv pip compile --upgrade pyproject.toml -o uv.lock
uv sync
```

## 📚 相关文档

- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - 详细配置指南
- **[RAG_MODES.md](RAG_MODES.md)** - RAG模式详细说明
- **[USAGE.md](USAGE.md)** - 快速使用指南

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

### 开发规范
- 使用 `uv` 管理依赖
- 遵循项目代码风格
- 添加适当的测试
- 更新相关文档

## 📝 许可证

MIT License

## 🙏 致谢

- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) - 中文嵌入模型
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [LM Studio](https://lmstudio.ai/) - 本地LLM运行环境
- [Obsidian](https://obsidian.md/) - 优秀的笔记工具
- [uv](https://github.com/astral-sh/uv) - 快速的Python包管理器和解析器
- 所有开源依赖和工具

---

**提示**: 系统已自动配置为不读取系统代理环境变量 (`session.trust_env = False`)，避免网络连接问题。