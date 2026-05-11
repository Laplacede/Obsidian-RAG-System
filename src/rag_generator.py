#!/usr/bin/env python3
"""
LLM生成器
集成本地或云LLM，实现带引用的答案生成
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class GenerationResult:
    """生成结果"""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    model: str


class LLMGenerator:
    """LLM生成器基类"""
    
    def __init__(self, model_name: str = "default"):
        """
        初始化LLM生成器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.context_window = 4096  # 默认上下文窗口大小
    
    def generate(self, query: str, context: List[Dict[str, Any]], 
                max_tokens: int = 500) -> GenerationResult:
        """
        生成答案
        
        Args:
            query: 查询文本
            context: 检索到的上下文
            max_tokens: 最大生成token数
            
        Returns:
            生成结果
        """
        raise NotImplementedError("子类必须实现generate方法")
    
    def format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        格式化上下文
        
        Args:
            context: 检索到的上下文列表
            
        Returns:
            格式化后的上下文文本
        """
        formatted = []
        
        for i, chunk in enumerate(context, 1):
            # 提取元数据
            metadata = chunk.get('metadata', {})
            title = metadata.get('title', '')
            if not title:
                title = metadata.get('source_document', f"文档{i}")
            
            # 提取内容
            content = chunk.get('text', '')
            
            # 格式化
            formatted.append(f"[文档{i}] {title}\n{content}\n")
        
        return "\n".join(formatted)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        创建提示词
        
        Args:
            query: 查询文本
            context: 格式化后的上下文
            
        Returns:
            提示词
        """
        prompt = f"""基于以下上下文，请回答用户的问题。请确保：
1. 答案准确、完整
2. 引用相关文档作为依据
3. 如果上下文信息不足，请说明
4. 使用中文回答

上下文：
{context}

用户问题：{query}

请根据上下文提供详细的回答，并在回答中引用相关文档（如[文档1]、[文档2]等）："""
        
        return prompt


class LocalLMStudioGenerator(LLMGenerator):
    """本地LM Studio生成器"""
    
    def __init__(self, base_url: str = "http://localhost:1234", 
                 model_name: str = "local-model"):
        """
        初始化本地LM Studio生成器
        
        Args:
            base_url: LM Studio API地址
            model_name: 模型名称
        """
        super().__init__(model_name)
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
        """
        try:
            import requests
            
            # 创建不信任环境代理的session
            session = requests.Session()
            session.trust_env = False  # 禁止读取系统代理环境变量
            
            # 格式化上下文
            formatted_context = self.format_context(context)
            
            # 创建提示词
            prompt = self.create_prompt(query, formatted_context)
            
            # 准备API请求 - LM Studio格式
            headers = {
                "Content-Type": "application/json"
            }
            
            # LM Studio API格式
            data = {
                "model": self.model_name,
                "input": prompt,
                "system_prompt": "你是一个基于个人知识库的助手，根据提供的上下文回答问题。请确保答案准确、完整，并引用相关文档。"
            }
            
            # 发送请求到LM Studio的API端点
            response = session.post(
                f"{self.base_url}/api/v1/chat",
                headers=headers,
                json=data,
                timeout=120  # 增加超时时间，因为RAG上下文可能较长
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 解析LM Studio的响应格式
                if "output" in result and len(result["output"]) > 0:
                    # 查找message类型的输出
                    for output in result["output"]:
                        if output.get("type") == "message":
                            answer = output.get("content", "").strip()
                            break
                    else:
                        # 如果没有message类型，使用最后一个输出
                        answer = result["output"][-1].get("content", "").strip()
                else:
                    # 兼容其他可能的响应格式
                    answer = result.get("content", "").strip()
                
                if not answer:
                    raise Exception("LM Studio返回空答案")
                
                # 从答案中提取引用
                citations = self.extract_citations(answer, context)
                
                return GenerationResult(
                    answer=answer,
                    citations=citations,
                    confidence=0.8,  # 默认置信度
                    model=self.model_name
                )
            else:
                raise Exception(f"API请求失败: {response.status_code} - {response.text}")
                
        except ImportError:
            raise Exception("需要安装requests库: pip install requests")
        except Exception as e:
            raise Exception(f"生成失败: {e}")
    
    def extract_citations(self, answer: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从答案中提取引用
        
        Args:
            answer: 生成的答案
            context: 原始上下文
            
        Returns:
            引用列表
        """
        citations = []
        
        # 查找类似[文档1]、[文档2]的引用
        citation_pattern = r'\[文档(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        
        for match in matches:
            try:
                doc_idx = int(match) - 1  # 转换为0-based索引
                if 0 <= doc_idx < len(context):
                    chunk = context[doc_idx]
                    metadata = chunk.get('metadata', {})
                    
                    citation = {
                        "document_index": doc_idx,
                        "title": metadata.get('title', metadata.get('source_document', f"文档{doc_idx+1}")),
                        "score": chunk.get('score', 0.0),
                        "text_preview": chunk.get('text', '')[:200] + "..."
                    }
                    citations.append(citation)
            except (ValueError, IndexError):
                continue
        
        return citations


class OpenAIGenerator(LLMGenerator):
    """OpenAI生成器"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        初始化OpenAI生成器
        
        Args:
            api_key: OpenAI API密钥
            model_name: 模型名称
        """
        super().__init__(model_name)
        self.api_key = api_key
    
    def generate(self, query: str, context: List[Dict[str, Any]], 
                max_tokens: int = 500) -> GenerationResult:
        """
        使用OpenAI生成答案
        
        Args:
            query: 查询文本
            context: 检索到的上下文
            max_tokens: 最大生成token数
            
        Returns:
            生成结果
        """
        try:
            from openai import OpenAI
            
            # 格式化上下文
            formatted_context = self.format_context(context)
            
            # 创建提示词
            prompt = self.create_prompt(query, formatted_context)
            
            # 初始化OpenAI客户端
            client = OpenAI(api_key=self.api_key)
            
            # 发送请求
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # 从答案中提取引用
            citations = self.extract_citations(answer, context)
            
            return GenerationResult(
                answer=answer,
                citations=citations,
                confidence=0.8,
                model=self.model_name
            )
                
        except ImportError:
            raise Exception("需要安装openai库: pip install openai")
        except Exception as e:
            raise Exception(f"生成失败: {e}")
    
    def extract_citations(self, answer: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从答案中提取引用
        """
        return LocalLMStudioGenerator.extract_citations(self, answer, context)


class MockGenerator(LLMGenerator):
    """模拟生成器（用于测试）"""
    
    def generate(self, query: str, context: List[Dict[str, Any]], 
                max_tokens: int = 500) -> GenerationResult:
        """
        模拟生成答案（用于测试）
        """
        # 格式化上下文
        formatted_context = self.format_context(context)
        
        # 创建模拟回答
        answer = f"这是基于查询'{query}'的模拟回答。\n\n"
        answer += f"我检索到了{len(context)}个相关文档。\n\n"
        
        # 添加一些模拟的引用
        for i, chunk in enumerate(context[:2], 1):
            metadata = chunk.get('metadata', {})
            title = metadata.get('source_document', f"文档{i}")
            answer += f"根据[文档{i}]《{title}》的内容，..."
        
        answer += "\n\n这是一个模拟的回答，实际使用中会连接到真正的LLM。"
        
        # 创建模拟引用
        citations = []
        for i, chunk in enumerate(context[:2], 1):
            metadata = chunk.get('metadata', {})
            citations.append({
                "document_index": i-1,
                "title": metadata.get('source_document', f"文档{i}"),
                "score": chunk.get('score', 0.0),
                "text_preview": chunk.get('text', '')[:100] + "..."
            })
        
        return GenerationResult(
            answer=answer,
            citations=citations,
            confidence=0.7,
            model="mock-model"
        )


class RAGSystem:
    """完整的RAG系统"""
    
    def __init__(self, retriever, generator):
        """
        初始化RAG系统
        
        Args:
            retriever: 检索器实例
            generator: LLM生成器实例
        """
        self.retriever = retriever
        self.generator = generator
    
    def query(self, query: str, n_context: int = 5, 
             use_hybrid: bool = True, use_reranking: bool = False) -> Dict[str, Any]:
        """
        执行完整的RAG查询
        
        Args:
            query: 查询文本
            n_context: 使用的上下文数量
            use_hybrid: 是否使用混合搜索
            use_reranking: 是否使用重排序
            
        Returns:
            查询结果
        """
        # 1. 检索
        print(f"正在检索相关文档...")
        context = self.retriever.retrieve(
            query=query,
            n_results=n_context,
            use_hybrid=use_hybrid,
            use_expansion=True,
            use_reranking=use_reranking
        )
        
        # 转换为字典格式
        context_dict = []
        for result in context:
            context_dict.append({
                'chunk_id': result.chunk_id,
                'text': result.text,
                'metadata': result.metadata,
                'score': result.score
            })
        
        print(f"检索到 {len(context)} 个相关文档")
        
        # 2. 生成
        print(f"正在生成答案...")
        try:
            generation_result = self.generator.generate(query, context_dict)
            
            # 构建完整结果
            result = {
                "query": query,
                "answer": generation_result.answer,
                "citations": generation_result.citations,
                "confidence": generation_result.confidence,
                "model": generation_result.model,
                "context_count": len(context),
                "context_preview": [
                    {
                        "title": chunk.get('metadata', {}).get('source_document', f"文档{i+1}"),
                        "score": chunk.get('score', 0.0),
                        "preview": chunk.get('text', '')[:100] + "..."
                    }
                    for i, chunk in enumerate(context_dict[:3])
                ]
            }
            
            return result
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "context_count": len(context)
            }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG系统")
    parser.add_argument("--query", "-q", required=True, help="查询文本")
    parser.add_argument("--context", "-c", type=int, default=5, help="上下文数量")
    parser.add_argument("--model-type", "-m", default="mock", 
                       choices=["mock", "local", "openai"], help="LLM类型")
    parser.add_argument("--hybrid", action="store_true", help="使用混合搜索")
    parser.add_argument("--rerank", action="store_true", help="启用重排序")
    
    # OpenAI参数
    parser.add_argument("--openai-key", help="OpenAI API密钥")
    parser.add_argument("--openai-model", default="gpt-3.5-turbo", help="OpenAI模型名称")
    
    # LM Studio参数
    parser.add_argument("--lmstudio-url", default="http://localhost:1234", 
                       help="LM Studio API地址")
    parser.add_argument("--lmstudio-model", default="local-model", help="LM Studio模型名称")
    
    args = parser.parse_args()
    
    try:
        # 创建检索器
        print("正在初始化检索器...")
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from retriever import HybridRetriever
        
        retriever = HybridRetriever(
            vector_store_dir="./data/vector_store",
            chunk_data_dir="./data/chunks"
        )
        
        # 加载数据
        retriever.load_vector_store("small")
        retriever.load_chunk_data("small")
        retriever.build_keyword_index("small")
        
        # 创建生成器
        print(f"正在初始化{args.model_type}生成器...")
        if args.model_type == "openai":
            if not args.openai_key:
                print("错误: 使用OpenAI需要提供API密钥")
                return
            generator = OpenAIGenerator(api_key=args.openai_key, model_name=args.openai_model)
        elif args.model_type == "local":
            generator = LocalLMStudioGenerator(
                base_url=args.lmstudio_url,
                model_name=args.lmstudio_model
            )
        else:
            generator = MockGenerator()
        
        # 创建RAG系统
        rag = RAGSystem(retriever, generator)
        
        # 执行查询
        result = rag.query(
            query=args.query,
            n_context=args.context,
            use_hybrid=args.hybrid,
            use_reranking=args.rerank
        )
        
        # 输出结果
        print(f"\n{'='*60}")
        print(f"RAG系统查询结果")
        print(f"{'='*60}\n")
        
        if "error" in result:
            print(f"错误: {result['error']}")
            return
        
        print(f"查询: {result['query']}")
        print(f"模型: {result['model']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"使用上下文: {result['context_count']} 个文档")
        print(f"\n{'='*60}")
        print(f"答案:")
        print(f"{'='*60}\n")
        print(result['answer'])
        
        if result['citations']:
            print(f"\n{'='*60}")
            print(f"引用:")
            print(f"{'='*60}\n")
            for i, citation in enumerate(result['citations'], 1):
                print(f"{i}. {citation['title']} (分数: {citation['score']:.3f})")
                print(f"   预览: {citation['text_preview']}\n")
        
        print(f"\n{'='*60}")
        print(f"上下文预览:")
        print(f"{'='*60}\n")
        for i, ctx in enumerate(result['context_preview'], 1):
            print(f"{i}. {ctx['title']} (分数: {ctx['score']:.3f})")
            print(f"   预览: {ctx['preview']}\n")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()