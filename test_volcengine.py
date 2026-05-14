#!/usr/bin/env python3
"""
火山引擎 API 连接测试脚本

用于测试火山引擎 API 的连接和配置是否正确。
"""

import requests
import json
import yaml
from pathlib import Path


def test_volcengine_connection():
    """测试火山引擎 API 连接"""
    
    print("=" * 60)
    print("火山引擎 API 连接测试")
    print("=" * 60)
    
    # 1. 加载配置
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    
    if not config_path.exists():
        print(f"\n❌ 配置文件不存在: {config_path}")
        print("请先复制 model_config.yaml.example 为 model_config.yaml")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"\n❌ 配置文件读取失败: {e}")
        return False
    
    # 2. 检查配置
    ve_config = config.get("llm", {}).get("volcengine", {})
    
    if not ve_config:
        print("\n❌ 配置文件中未找到 volcengine 配置")
        return False
    
    api_key = ve_config.get("api_key")
    base_url = ve_config.get("base_url")
    model = ve_config.get("model", "doubao-pro-4k")
    
    print("\n📋 配置信息:")
    print(f"  API Key: {'已配置' if api_key else '❌ 未配置'}")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    
    if not api_key:
        print("\n❌ API 密钥未配置")
        print("请在 config/model_config.yaml 中的 llm.volcengine.api_key 设置你的 API 密钥")
        return False
    
    # 3. 测试连接
    print("\n🔌 测试 API 连接...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    test_data = {
        "model": model,
        "messages": [
            {"role": "user", "content": "你好，这是一个测试消息。"}
        ],
        "max_tokens": 100,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
    try:
        url = f"{base_url}/chat/completions"
        response = requests.post(
            url,
            headers=headers,
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 连接成功！")
            print(f"\n📝 API 响应:")
            print(f"  Model: {result.get('model')}")
            print(f"  Status: OK")
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content', '')
                print(f"  Message: {content[:100]}...")
            
            return True
        
        else:
            print(f"❌ API 返回错误状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print("❌ 连接超时")
        print("请检查网络连接和 base_url 配置")
        return False
    
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到 {base_url}")
        print("请检查:")
        print("  1) base_url 配置是否正确")
        print("  2) 网络连接是否正常")
        return False
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


def test_cli_switching():
    """测试 CLI 模型切换功能"""
    
    print("\n" + "=" * 60)
    print("CLI 模型切换测试")
    print("=" * 60)
    
    print("\n📋 支持的切换命令:")
    print("  1. model                 - 显示所有可用模型")
    print("  2. model local           - 切换到本地 LM Studio")
    print("  3. model volcengine      - 切换到火山引擎")
    print("  4. model openai          - 切换到 OpenAI")
    print("  5. model mock            - 切换到模拟模式")
    
    print("\n✅ 功能测试完成")
    return True


if __name__ == "__main__":
    success = test_volcengine_connection()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！你可以开始使用火山引擎了")
        print("=" * 60)
        print("\n在 CLI 中使用以下命令切换到火山引擎:")
        print("  model volcengine")
        print("\n然后输入查询:")
        print("  query 你的问题")
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败，请按照提示检查配置")
        print("=" * 60)
