# 模型配置指南

## 🔐 隐私管理

所有隐私信息（API密钥、服务器地址等）存储在 `config/model_config.yaml` 中，该文件已被添加到 `.gitignore`，**不会被上传到 Git 仓库**。

## 📋 初始化步骤

### 1. 复制配置模板

```bash
cd config
cp model_config.yaml.example model_config.yaml
```

### 2. 编辑配置文件

打开 `config/model_config.yaml`，根据你的环境修改：

```yaml
# 本地 LM Studio 配置
llm:
  local:
    enabled: true
    base_url: "http://YOUR_SERVER_IP:1234"  # ← 替换为实际的服务器地址
    model: "your-model-name"                 # ← 替换为实际的模型名称
    temperature: 0.1
    timeout: 500
    max_tokens: 1024
```

## 🖥️ LM Studio 配置

### 本地运行 (localhost)

如果 LM Studio 在本地运行：
```yaml
base_url: "http://localhost:1234"
```

### 远程服务器 (通过 Tailscale 或 SSH 隧道)

#### 方式1: 直接 IP 地址
```yaml
base_url: "http://100.109.51.62:1234"  # Tailscale VPN IP
```

#### 方式2: SSH 隧道
```bash
# 在本地建立 SSH 隧道到远程服务器
ssh -L 1234:localhost:1234 user@remote-server

# 配置中使用本地代理
base_url: "http://localhost:1234"
```

## 🧪 测试连接

运行以下命令测试是否能连接到 LM Studio：

```bash
python3 << 'EOF'
import requests
import json

base_url = "http://YOUR_SERVER_IP:1234"  # 替换为你的实际地址

try:
    # 测试连接
    response = requests.get(f"{base_url}/v1/models", timeout=5)
    if response.status_code == 200:
        models = response.json()
        print("✓ 连接成功！可用模型：")
        print(json.dumps(models, indent=2))
    else:
        print(f"✗ 连接失败，状态码: {response.status_code}")
except requests.exceptions.ConnectionError:
    print(f"✗ 无法连接到 {base_url}")
    print("  检查:")
    print("  1) 服务器地址是否正确")
    print("  2) LM Studio 是否在运行")
    print("  3) 网络连接是否正常")
except requests.exceptions.Timeout:
    print("✗ 连接超时，检查网络连接")
except Exception as e:
    print(f"✗ 错误: {e}")
EOF
```

## 🚀 使用示例

### 命令行使用

```bash
# 使用本地模型
python3 src/rag_generator.py --query "你的问题" --model-type local

# 使用模拟模型 (测试)
python3 src/rag_generator.py --query "你的问题" --model-type mock
```

### 交互式 CLI

```bash
python3 src/cli.py
```

系统会自动从 `config/model_config.yaml` 加载模型配置。

## 🔧 架构设计

- **generators/** - 通用的生成器模块，不依赖具体配置
  - `base.py` - 基类和通用方法
  - `mock_generator.py` - 测试用生成器
  - `local_lm_studio.py` - 本地 LM Studio
  - `openai_generator.py` - OpenAI API
  
- **config/model_config.yaml** - 隐私配置 (.gitignore 忽略)
- **config/model_config.yaml.example** - 配置模板 (提交到仓库)

这样设计的好处：
- ✅ 隐私信息不会泄露到仓库
- ✅ generators 模块保持通用，易于扩展
- ✅ 明确的职责分离
- ✅ 多人协作时不会有配置冲突

## 🐛 常见问题

### 问题：连接超时
**原因**：服务器地址不正确或网络不通
**解决**：检查 `base_url` 配置，尝试 `ping` 或 `curl` 测试

### 问题：模型名称错误
**原因**：`model` 配置与 LM Studio 实际加载的模型名称不匹配
**解决**：运行测试脚本获取可用模型列表

### 问题：API 密钥无效
**原因**：OpenAI API 密钥过期或配置错误
**解决**：检查 `config/model_config.yaml` 中的 `api_key`

## 📝 配置安全性

- `config/model_config.yaml` 包含敏感信息，**不要提交到版本控制**
- `.gitignore` 已配置为忽略此文件
- 建议定期备份配置文件
- 如果配置文件泄露，立即更换 API 密钥
