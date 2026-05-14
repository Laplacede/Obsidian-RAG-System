"""
OpenAI生成器

通过OpenAI API调用GPT等模型。
此模块为通用生成器，不依赖具体配置文件，所有参数通过构造函数传入。

职责说明：
- 输入：接收查询、上下文，以及通过构造函数传入的 OpenAI API 密钥和模型名称。
- 处理：调用 OpenAI API，发送格式化的提示词并获取生成结果。
- 输出：返回 GenerationResult 对象（包含 answer、citations、confidence）。
- 依赖文件：继承自 base.py 的 LLMGenerator；需要 openai 库。
- 配置管理：隐私信息（API密钥等）应存储在 config/model_config.yaml 中，
            由 cli.py 或 rag_generator.py 读取后通过参数传入。
"""

from typing import List, Dict, Any, Optional
from .base import LLMGenerator, GenerationResult


class OpenAIGenerator(LLMGenerator):
    """OpenAI生成器
    
    通过OpenAI API调用GPT等模型来生成答案。
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", 
                 rag_mode: str = "flexible", default_max_tokens: Optional[int] = None):
        """
        初始化OpenAI生成器
        
        Args:
            api_key: OpenAI API密钥
            model_name: 模型名称 (默认: gpt-3.5-turbo)
            rag_mode: RAG模式 ('strict'|'flexible')
        """
        super().__init__(model_name, rag_mode=rag_mode, default_max_tokens=default_max_tokens)
        self.api_key = api_key
    
    def generate(self, query: str, context: List[Dict[str, Any]], 
            max_tokens: Optional[int] = None) -> GenerationResult:
        """
        使用OpenAI生成答案
        
        Args:
            query: 查询文本
            context: 检索到的上下文
            max_tokens: 最大生成token数，None 表示使用生成器默认值
            
        Returns:
            生成结果
            
        Raises:
            Exception: 如果API调用失败
        """
        try:
            from openai import OpenAI
            
            # 格式化上下文
            formatted_context = self.format_context(context)
            
            # 创建提示词
            prompt = self.create_prompt(query, formatted_context)
            effective_max_tokens = self.default_max_tokens if max_tokens is None else max_tokens
            
            # 初始化OpenAI客户端
            client = OpenAI(api_key=self.api_key)
            
            # 发送请求
            # 构建请求参数
            create_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            # 只在指定了 max_tokens 时才添加到请求中
            if effective_max_tokens is not None:
                create_params["max_tokens"] = effective_max_tokens
            
            response = self.client.chat.completions.create(**create_params)
            
            answer = response.choices[0].message.content
            
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
                
        except ImportError:
            raise Exception("需要安装openai库: pip install openai")
        except Exception as e:
            raise Exception(f"生成失败: {e}")
