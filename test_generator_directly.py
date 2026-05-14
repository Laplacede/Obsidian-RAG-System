#!/usr/bin/env python3
"""测试CLI中的生成器调用"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generators.local_lm_studio import LocalLMStudioGenerator

def test_generator_directly():
    """直接测试生成器"""
    print("=" * 60)
    print("直接测试LocalLMStudioGenerator")
    print("=" * 60)
    
    # 使用配置文件中的设置
    base_url = "http://100.109.51.62:1234"
    model_name = "qwen/qwen3.5-9b"
    timeout = 1200
    
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
        
        # 测试简单的查询
        print("\n测试简单查询...")
        context = [
            {
                "text": "这是一个测试文档。",
                "metadata": {"source_document": "test.md"},
                "score": 0.9
            }
        ]
        
        try:
            result = generator.generate("测试查询", context, max_tokens=50)
            print("✅ 生成器调用成功")
            print(f"答案: {result.answer}")
            print(f"置信度: {result.confidence}")
            print(f"模型: {result.model}")
        except Exception as e:
            print(f"❌ 生成器调用失败: {e}")
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