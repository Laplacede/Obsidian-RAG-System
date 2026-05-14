#!/usr/bin/env python3
"""测试CLI中的生成器调用（带超时控制）"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generators.local_lm_studio import LocalLMStudioGenerator
import requests

def test_generator_directly():
    """直接测试生成器"""
    print("=" * 60)
    print("直接测试LocalLMStudioGenerator（短超时）")
    print("=" * 60)
    
    # 使用配置文件中的设置
    base_url = "http://100.109.51.62:1234"
    model_name = "qwen/qwen3.5-9b"
    timeout = 30  # 短超时
    
    print(f"创建生成器:")
    print(f"  base_url: {base_url}")
    print(f"  model_name: {model_name}")
    print(f"  timeout: {timeout}")
    
    try:
        generator = LocalLMStudioGenerator(
            base_url=base_url,
            model_name=model_name,
            rag_mode="flexible",
            timeout=timeout
        )
        print("✅ 生成器创建成功")
        
        # 首先测试直接HTTP请求
        print("\n1. 测试直接HTTP请求...")
        try:
            session = requests.Session()
            session.trust_env = False
            
            # 测试连接
            response = session.get(f"{base_url}/api/v1/models", timeout=5)
            print(f"  连接测试状态码: {response.status_code}")
            
            # 测试简单请求
            data = {
                "model": model_name,
                "input": "请只回复：测试成功",
                "system_prompt": "你是一个助手。请不要输出思考过程，只输出最终答案。"
            }
            
            response = session.post(
                f"{base_url}/api/v1/chat",
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=10
            )
            print(f"  聊天测试状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"  响应: {result}")
                
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
                        print(f"  解析的答案: {answer}")
                    else:
                        print("  无法解析答案")
                else:
                    print(f"  响应格式不支持: {result}")
            else:
                print(f"  请求失败: {response.text}")
                
        except requests.exceptions.Timeout:
            print("  ❌ HTTP请求超时")
        except requests.exceptions.ConnectionError:
            print("  ❌ 无法连接到服务器")
        except Exception as e:
            print(f"  ❌ HTTP请求失败: {e}")
        
        # 测试生成器
        print("\n2. 测试生成器调用...")
        context = [
            {
                "text": "这是一个测试文档。",
                "metadata": {"source_document": "test.md"},
                "score": 0.9
            }
        ]
        
        try:
            print("  开始生成...")
            result = generator.generate("测试查询", context, max_tokens=50)
            print("  ✅ 生成器调用成功")
            print(f"  答案: {result.answer}")
            print(f"  置信度: {result.confidence}")
            print(f"  模型: {result.model}")
        except Exception as e:
            print(f"  ❌ 生成器调用失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 生成器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(test_generator_directly())