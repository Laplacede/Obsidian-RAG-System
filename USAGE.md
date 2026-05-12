# Obsidian RAG System - 快速使用指南

## 📦 环境设置

```bash
# 1. 确保已安装 uv
uv --version

# 2. 安装项目依赖
uv sync

# 3. 验证安装
uv run python main.py version
```

## 🚀 快速启动

```bash
# 启动交互式CLI（推荐）
uv run python main.py cli

# 或直接使用
uv run python -m src.cli --model-type local
```

## 🔧 常用命令

```bash
# 查看所有命令
uv run python main.py --help

# 初始化系统配置
uv run python main.py init

# 显示系统配置
uv run python main.py config
```

## ⚙️ 配置说明

编辑 `config/model_config.yaml` 文件：

1. **Obsidian仓库路径**：设置你的Obsidian笔记库路径
2. **本地模型配置**：确保 `base_url` 指向正确的 LM Studio 地址（不带 `/api/v1/chat` 后缀）
3. **嵌入模型**：使用BAAI/bge-small-zh-v1.5进行中文优化

当前本地生成器使用的是 LM Studio 的旧接口 `/api/v1/chat`，请求体为 `input + system_prompt`。

## 🐛 故障排除

### 1. 本地模型连接失败
```bash
# 检查LM Studio是否运行
curl http://100.109.51.62:1234/v1/models

# 验证旧接口是否返回答案
curl -X POST http://100.109.51.62:1234/api/v1/chat \
	-H 'Content-Type: application/json' \
	-d '{"model":"qwen/qwen3.5-9b","input":"请只回复 ok","system_prompt":"你是一个助手。"}'
```

### 2. CLI启动错误
```bash
# 如果出现 "TypeError: main() takes 0 positional arguments but 1 was given"
# 确保使用更新后的 main.py
uv run python main.py version
```

### 3. 依赖问题
```bash
# 重新安装依赖
rm -rf .venv
uv sync
```

### 4. 权限问题
```bash
# 确保有读取Obsidian笔记的权限
ls "/Users/fengzhe/Library/Mobile Documents/iCloud~md~obsidian/Documents/知识仓库"
```

## 📊 数据状态

系统已成功提取 **83个** Obsidian笔记，并创建了 **354个** 分块，保存在：
- `data/processed/obsidian_notes.json`
- `data/embeddings/small_embeddings.npy`
- `data/chunks/small_chunks.json`
- `data/vector_store/` (ChromaDB向量存储)

## 🔄 开发工作流

```bash
# 激活虚拟环境
source .venv/bin/activate

# 添加新依赖
uv add package-name

# 运行测试
python -m pytest tests/

# 退出虚拟环境
deactivate
```

## 📝 CLI交互命令

在CLI交互模式中 (`uv run python main.py cli`) 可以使用以下命令：

```
help         显示帮助信息
clear        清屏
config       显示当前配置
config set <选项> <值>  更新配置
history      显示查询历史
query <问题>  执行查询
exit/quit/q  退出CLI
```

---

**提示**: 系统已自动配置为不读取系统代理环境变量 (`session.trust_env = False`)，避免网络连接问题。