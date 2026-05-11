#!/usr/bin/env python3
"""
快速测试脚本 - 验证 RAG 系统核心功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

print("🚀 RAG 系统快速测试")
print("=" * 50)

# 测试 1: 检查环境
print("\n1. 检查 Python 环境:")
print(f"   Python 路径: {sys.executable}")
print(f"   Python 版本: {sys.version}")

# 测试 2: 检查依赖
print("\n2. 检查核心依赖:")
try:
    import numpy as np
    print("   ✅ numpy")
except ImportError:
    print("   ❌ numpy 未安装")

try:
    import requests
    print("   ✅ requests")
except ImportError:
    print("   ❌ requests 未安装")

try:
    from sentence_transformers import SentenceTransformer
    print("   ✅ sentence-transformers")
except ImportError:
    print("   ❌ sentence-transformers 未安装")

try:
    import yaml
    print("   ✅ PyYAML")
except ImportError:
    print("   ❌ PyYAML 未安装")

# 测试 3: 检查配置文件
print("\n3. 检查配置文件:")
config_path = Path(__file__).parent / "config" / "model_config.yaml"
if config_path.exists():
    print(f"   ✅ 配置文件存在: {config_path}")
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    llm_config = config.get("llm", {}).get("local", {})
    base_url = llm_config.get("base_url", "")
    model = llm_config.get("model", "")
    
    print(f"   📍 LM Studio 地址: {base_url}")
    print(f"   🤖 模型: {model}")
else:
    print(f"   ❌ 配置文件不存在")

# 测试 4: 检查数据文件
print("\n4. 检查数据文件:")
data_dir = Path(__file__).parent / "data" / "embeddings"
if data_dir.exists():
    files = list(data_dir.glob("*"))
    print(f"   ✅ 数据目录存在，包含 {len(files)} 个文件")
    
    for file in files[:3]:  # 显示前3个文件
        print(f"      📄 {file.name} ({file.stat().st_size // 1024} KB)")
else:
    print("   ❌ 数据目录不存在")

# 测试 5: 简单 API 测试
print("\n5. 测试远程 LM Studio 连接:")
try:
    import requests
    session = requests.Session()
    session.trust_env = False
    
    # 简单测试
    test_data = {
        "model": "qwen/qwen3.5-9b",
        "input": "test",
        "system_prompt": "test"
    }
    
    response = session.post(
        "http://100.109.51.62:1234/api/v1/chat",
        json=test_data,
        timeout=10
    )
    
    if response.status_code == 200:
        print("   ✅ 远程 LM Studio 连接成功")
        result = response.json()
        if "output" in result:
            print("   📨 收到有效响应")
    else:
        print(f"   ❌ 连接失败: HTTP {response.status_code}")
        
except Exception as e:
    print(f"   ❌ 连接测试失败: {e}")

print("\n" + "=" * 50)
print("📋 测试总结:")
print("   要启动完整 RAG 系统，请运行:")
print("   python rag_system.py interactive")
print("   或使用远程配置:")
print("   python rag_system.py --lmstudio-url http://100.109.51.62:1234 --lmstudio-model qwen/qwen3.5-9b")
print("\n   使用 uv 环境:")
print("   uv run python rag_system.py interactive")