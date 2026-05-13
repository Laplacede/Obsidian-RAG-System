"""
本地LM Studio生成器

通过 HTTP API 调用本地运行的 LM Studio，当前对接的是 /api/v1/chat 旧接口。
此模块不直接读取配置文件，所有连接参数由上层（CLI / factory）传入。

职责说明：
- 输入：接收查询、上下文，以及通过构造函数传入的 LM Studio 连接参数。
- 处理：调用本地 LM Studio API，发送格式化的提示词并获取生成结果。
- 输出：返回 GenerationResult 对象（包含 answer、citations、confidence）。
- 依赖文件：继承自 base.py 的 LLMGenerator；需要 requests 库。
- 配置管理：隐私信息（服务器地址、API密钥等）应存储在 config/model_config.yaml 中，
            由 cli.py 或 rag_generator.py 读取后通过参数传入。
"""

from typing import List, Dict, Any
from .base import LLMGenerator, GenerationResult


class LocalLMStudioGenerator(LLMGenerator):
    """本地LM Studio生成器
    
    通过 HTTP API 调用本地运行的 LM Studio 实例来生成答案。

    当前版本对接 LM Studio 的 /api/v1/chat 端点，响应解析遵循
    output -> message -> content 的格式。
    """
    
    def __init__(self, base_url: str, 
                 model_name: str = "local-model",
                 rag_mode: str = "flexible"):
        """
        初始化本地LM Studio生成器
        
        Args:
            base_url: LM Studio API 地址，不包含 /api/v1/chat 后缀
            model_name: 模型名称
            rag_mode: RAG模式 ('strict'|'flexible')
        """
        super().__init__(model_name, rag_mode=rag_mode)
        if not base_url:
            raise ValueError("base_url 不能为空，必须由上层配置传入")
        self.base_url = base_url
        self.api_key = "not-needed"  # LM Studio不需要API密钥
    
    def generate(self, query: str, context: List[Dict[str, Any]], 
                max_tokens: int = 500) -> GenerationResult:
        """
        使用LM Studio生成答案
        
        Args:
            query: 查询文本
            context: 检索到的上下文
            max_tokens: 最大生成token数
            
        Returns:
            生成结果
            
        Raises:
            Exception: 如果请求失败或返回空答案
        """
        try:
            import requests
            
            # 创建不信任环境代理的session（重要：避免走系统代理）
            session = requests.Session()
            session.trust_env = False
            
            # 格式化上下文
            formatted_context = self.format_context(context)
            
            # 创建提示词
            prompt = self.create_prompt(query, formatted_context)
            
            # 准备 API 请求 - LM Studio 旧接口格式
            headers = {
                "Content-Type": "application/json"
            }
            
            # LM Studio 旧接口请求体
            data = {
                "model": self.model_name,
                "input": prompt,
                "system_prompt": "你是一个基于个人知识库的助手，根据提供的上下文回答问题。请确保答案准确、完整，并引用相关文档。"
            }
            
            # 发送请求到 LM Studio 的旧接口端点
            api_url = f"{self.base_url}/api/v1/chat"
            print(f"[调试] 发送请求到: {api_url}")
            print(f"[调试] 模型: {self.model_name}")
            
            response = session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=120  # 增加超时时间，因为RAG上下文可能较长
            )
            
            print(f"[调试] 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"[调试] 响应内容摘要: {str(result)[:200]}...")
                
                # 解析 LM Studio 旧接口返回格式
                if "output" in result and len(result["output"]) > 0:
                    # 查找message类型的输出
                    answer = None
                    for output in result["output"]:
                        if output.get("type") == "message":
                            answer = output.get("content", "").strip()
                            break

                    # 如果没有message类型，使用最后一个输出
                    if not answer:
                        answer = result["output"][-1].get("content", "").strip()
                else:
                    raise Exception(f"LM Studio返回格式不支持: {result}")
                
                if not answer:
                    raise Exception("LM Studio返回空答案")
                
                # 从答案中提取引用
                citations = self.extract_citations(answer, context)
                
                # 使用动态置信度，基于检索质量计算
                confidence = self.calculate_confidence(context, has_knowledge_base_answer=True)
                
                return GenerationResult(
                    answer=answer,
                    citations=citations,
                    confidence=confidence,
                    model=self.model_name
                )
            else:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
                
        except ImportError:
            raise Exception("需要安装requests库: pip install requests")
        except requests.exceptions.Timeout:
            raise Exception(f"请求超时。检查: 1) 服务器地址是否正确 2) 网络连接是否正常 3) LM Studio是否运行")
        except requests.exceptions.ConnectionError:
            raise Exception(f"无法连接到服务器 ({self.base_url})。检查: 1) 服务器地址是否正确 2) LM Studio是否在运行 3) 网络连接是否正常")
        except Exception as e:
            raise Exception(f"生成失败: {e}")
