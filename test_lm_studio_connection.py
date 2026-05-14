#!/usr/bin/env python3
"""测试LM Studio连接"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import requests
from pathlib import Path

def build_session():
    """创建不使用系统代理的请求会话。"""
    session = requests.Session()
    session.trust_env = False
    return session

def load_config():
    """加载模型配置"""
    config_path = Path(__file__).parent / "config" / "model_config.yaml"
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return None

def test_direct_connection():
    """直接测试连接"""
    print("=" * 60)
    print("直接测试LM Studio连接")
    print("=" * 60)
    
    # 加载配置
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
    
    print(f"服务器: {base_url}")
    print(f"模型: {model_name}")
    print(f"超时: {timeout}s")
    
    # 测试连接
    print("\n1. 测试连接到服务器...")
    try:
        session = build_session()
        response = session.get(
            f"{base_url}/api/v1/models",
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ 连接成功！")
            models_data = response.json()
            
            # 显示可用模型
            models = models_data.get("models") or models_data.get("data") or []
            if models:
                print("\n可用模型:")
                for i, model in enumerate(models, 1):
                    model_id = model.get("key") or model.get("id") or "unknown"
                    display_name = model.get("display_name")
                    if display_name:
                        print(f"  {i}. {model_id} ({display_name})")
                    else:
                        print(f"  {i}. {model_id}")
        else:
            print(f"❌ 服务器返回状态码: {response.status_code}")
            return 1
            
    except requests.exceptions.Timeout:
        print(f"❌ 连接超时 (5秒)")
        return 1
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器")
        return 1
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return 1
    
    # 测试聊天完成
    print("\n2. 测试聊天完成...")
    try:
        session = build_session()
        
        # 使用旧接口格式
        data = {
            "model": model_name,
            "input": "请只回复：测试成功",
            "system_prompt": "你是一个助手。请不要输出思考过程，只输出最终答案。"
        }
        
        response = session.post(
            f"{base_url}/api/v1/chat",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=30
        )
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 聊天API调用成功")
            
            # 解析响应
            if "output" in result and len(result["output"]) > 0:
                answer = None
                for output in result["output"]:
                    if output.get("type") == "message":
                        answer = output.get("content", "").strip()
                        break
                
                if not answer:
                    answer = result["output"][-1].get("content", "").strip()
                
                if answer:
                    print(f"模型响应: {answer}")
                    if "测试成功" in answer:
                        print("✅ 模型响应正确！")
                    else:
                        print(f"⚠️  模型响应不正确: {answer}")
                else:
                    print("❌ 模型返回空答案")
            else:
                print(f"❌ 返回格式不支持: {result}")
        else:
            print(f"❌ 请求失败: {response.text}")
            return 1
            
    except Exception as e:
        print(f"❌ 聊天测试失败: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(test_direct_connection())
    except KeyboardInterrupt:
        print("\n\n中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)