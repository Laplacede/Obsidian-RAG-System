#!/usr/bin/env python3
"""
LM Studio 连接诊断脚本

用于检查与 LM Studio 服务器的连接、模型列表和可用接口。
这个文件保留为独立测试工具，便于在不改主流程的情况下快速排查问题。
"""

import sys
import os
import yaml
import requests
import json
from pathlib import Path


def build_session():
    """创建不使用系统代理的请求会话。"""
    session = requests.Session()
    session.trust_env = False
    return session


def load_config(config_path=None):
    """加载模型配置"""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "model_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print(f"   请复制配置模板: cp config/model_config.yaml.example config/model_config.yaml")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return None


def test_connection(base_url, timeout=5):
    """测试与 LM Studio 的连接"""
    print(f"\n📡 测试连接到: {base_url}")
    
    try:
        session = build_session()
        response = session.get(
            f"{base_url}/api/v1/models",
            timeout=timeout
        )
        
        if response.status_code == 200:
            print("✅ 连接成功！")
            return response.json()
        else:
            print(f"❌ 服务器返回状态码: {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"❌ 连接超时 ({timeout}秒)")
        print("   可能的原因:")
        print("   1) 服务器地址不正确")
        print("   2) LM Studio 未运行")
        print("   3) 网络延迟过高")
        return None
        
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器")
        print("   可能的原因:")
        print("   1) 服务器地址不正确")
        print("   2) LM Studio 未运行")
        print("   3) 网络不通 (如果是远程服务器)")
        return None
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None


def test_model_list(models_data):
    """显示可用的模型"""
    if not models_data:
        return
    
    print("\n📋 可用的模型:")
    try:
        models = models_data.get("models") or models_data.get("data") or []
        if models:
            for i, model in enumerate(models, 1):
                model_id = model.get("key") or model.get("id") or "unknown"
                display_name = model.get("display_name")
                if display_name:
                    print(f"   {i}. {model_id} ({display_name})")
                else:
                    print(f"   {i}. {model_id}")
        else:
            print(f"   {json.dumps(models_data, indent=3)}")
    except Exception as e:
        print(f"   无法解析模型列表: {e}")


def get_model_items(models_data):
    """从不同接口格式中提取模型列表。"""
    if not models_data:
        return []
    return models_data.get("models") or models_data.get("data") or []


def test_chat_completion(base_url, model_name, timeout=300):
    """测试 chat completion API"""
    print(f"\n💬 测试 Chat Completion API")
    print(f"   模型: {model_name}")
    
    try:
        session = build_session()

        attempts = [
            (
                "/api/v1/chat",
                {
                    "model": model_name,
                    "input": "请只回复：ok",
                    "system_prompt": "你是一个助手。请不要输出思考过程，只输出最终答案。"
                },
                "legacy",
            ),
            (
                "/v1/chat/completions",
                {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say 'Hello, I am working correctly!' and nothing else."}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.1,
                },
                "openai",
            ),
        ]

        for path, payload, mode in attempts:
            print(f"   尝试接口: {base_url}{path}")
            response = session.post(
                f"{base_url}{path}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )

            print(f"   状态码: {response.status_code}")
            try:
                result = response.json()
            except Exception:
                print(f"   非 JSON 响应: {response.text}")
                continue

            if mode == "legacy":
                output = result.get("output", [])
                message = ""
                for item in output:
                    if item.get("type") == "message":
                        message = item.get("content", "").strip()
                        break
                if not message and output:
                    message = output[-1].get("content", "").strip()

                if message:
                    print("✅ /api/v1/chat 可用")
                    print(f"   模型响应: {message}")
                    return True

                print(f"   /api/v1/chat 返回空响应: {result}")
            else:
                message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                reasoning = result.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
                if message:
                    print("✅ /v1/chat/completions 可用")
                    print(f"   模型响应: {message}")
                    return True

                if reasoning:
                    print("⚠️ /v1/chat/completions 只返回 reasoning_content，content 为空")
                    print(f"   reasoning_content: {reasoning[:300]}")
                else:
                    print(f"   /v1/chat/completions 返回空响应: {result}")

        return False
            
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时 ({timeout}秒)")
        print("   模型可能在处理较大的请求，或系统性能有问题")
        return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主诊断流程"""
    print("=" * 60)
    print("LM Studio 连接诊断工具")
    print("=" * 60)
    
    # 1. 加载配置
    print("\n1️⃣  加载配置文件...")
    config = load_config()
    if not config:
        return 1
    
    local_config = config.get("llm", {}).get("local", {})
    
    if not local_config.get("enabled"):
        print("⚠️  本地模型未启用")
        return 1
    
    base_url = local_config.get("base_url", "http://localhost:1234")
    model_name = local_config.get("model", "local-model")
    timeout = local_config.get("timeout", 500)
    
    print(f"✅ 配置加载成功")
    print(f"   服务器: {base_url}")
    print(f"   模型: {model_name}")
    print(f"   超时: {timeout}s")
    
    # 2. 测试连接
    print("\n2️⃣  测试服务器连接...")
    models_data = test_connection(base_url, timeout=min(5, timeout))
    if not models_data:
        print("\n💡 连接诊断建议:")
        print("   1. 检查 config/model_config.yaml 中的 base_url")
        print("   2. 确认 LM Studio 正在运行")
        print("   3. 如果使用远程服务器，检查网络连接和 VPN")
        return 1
    
    # 3. 显示可用模型
    print("\n3️⃣  获取可用模型列表...")
    test_model_list(models_data)
    
    # 4. 验证模型名称
    print(f"\n4️⃣  验证模型 '{model_name}'...")
    available_models = get_model_items(models_data)
    available_keys = [m.get("key") or m.get("id", "") for m in available_models]
    if available_keys:
        if model_name in available_keys:
            print(f"✅ 模型 '{model_name}' 存在")
        else:
            print(f"⚠️  模型 '{model_name}' 未在列表中")
            print(f"   可用模型: {available_keys[0]}")
    else:
        print("⚠️  没有解析到可用模型列表")
    
    # 5. 测试 chat completion
    print(f"\n5️⃣  测试 Chat Completion...")
    if available_models:
        # 优先使用配置中的模型名；如果不在列表中，使用第一个可用模型
        test_model = model_name if model_name in available_keys else available_keys[0]
        success = test_chat_completion(base_url, test_model, timeout=min(300, timeout))

        if success:
            print("\n" + "=" * 60)
            print("✅ 所有诊断都通过了！系统已准备好使用。")
            print("=" * 60)
            return 0
    
    print("\n" + "=" * 60)
    print("⚠️  诊断完成，但存在一些问题。")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
