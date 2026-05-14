"""
火山引擎生成器

通过火山引擎API调用豆包等模型。
火山引擎提供兼容OpenAI格式的API接口。

职责说明：
- 输入：接收查询、上下文，以及通过构造函数传入的 API 密钥和模型名称。
- 处理：调用火山引擎API，发送格式化的提示词并获取生成结果。
- 输出：返回 GenerationResult 对象（包含 answer、citations、confidence）。
- 依赖文件：继承自 base.py 的 LLMGenerator；可使用 openai 库或 requests 库。
- 配置管理：隐私信息（API密钥等）应存储在 config/model_config.yaml 中。
"""

from typing import List, Dict, Any
import json
import requests
from .base import LLMGenerator, GenerationResult


class VolcengineGenerator(LLMGenerator):
    """火山引擎生成器
    
    通过火山引擎API调用豆包等模型来生成答案。
    """
    
    def __init__(self, api_key: str, model_name: str = "doubao-pro-4k",
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
                 rag_mode: str = "flexible", timeout: int = 60):
        """
        初始化火山引擎生成器
        
        Args:
            api_key: 火山引擎 API 密钥
            model_name: 模型名称 (默认: doubao-pro-4k)
            base_url: API 基础 URL
            rag_mode: RAG模式 ('strict'|'flexible')
            timeout: 请求超时时间（秒）
        """
        super().__init__(model_name, rag_mode=rag_mode)
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def generate(self, query: str, context: List[Dict[str, Any]], 
                max_tokens: int = 500) -> GenerationResult:
        """
        使用火山引擎生成答案
        
        Args:
            query: 查询文本
            context: 检索到的上下文
            max_tokens: 最大生成token数
            
        Returns:
            生成结果
            
        Raises:
            Exception: 如果请求失败
        """
        try:
            # 格式化上下文
            formatted_context = self.format_context(context)
            
            # 创建提示词
            prompt = self.create_prompt(query, formatted_context)
            
            # 准备请求头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 准备请求数据
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            # 发送请求
            api_url = f"{self.base_url}/chat/completions"
            print(f"[调试] 向火山引擎发送请求到: {api_url}")
            
            session = requests.Session()
            session.trust_env = False
            
            response = session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                
                if not answer:
                    raise Exception("火山引擎返回空答案")
                
                # 从答案中提取引用
                citations = self.extract_citations(answer, context)
                
                # 使用动态置信度
                confidence = self.calculate_confidence(context, has_knowledge_base_answer=True)
                
                return GenerationResult(
                    answer=answer,
                    citations=citations,
                    confidence=confidence,
                    model=self.model_name
                )
            else:
                raise Exception(f"火山引擎API返回错误: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception(f"火山引擎请求超时 (超时设置: {self.timeout}秒)")
        except json.JSONDecodeError:
            raise Exception("火山引擎返回格式不支持")
        except Exception as e:
            raise Exception(f"生成失败: {e}")
