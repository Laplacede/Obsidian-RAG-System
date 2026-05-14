#!/usr/bin/env python3
"""
RAG系统生成器工厂和协调器

轻量级的生成器工厂和 RAG 系统协调器。具体的生成器实现（MockGenerator、LocalLMStudioGenerator、OpenAIGenerator）
已移至 src/generators/ 目录。

职责说明：
- 输入：接收检索器实例和生成器类型标识符。
- 处理：根据类型创建对应的生成器实例；协调检索和生成过程，完成端到端的RAG流程。
- 输出：返回完整的查询结果，包含生成的答案、引用文档、置信度等。
- 下游传递：结果会被 CLI 或上层业务显示给用户。
- 依赖文件：导入 src/generators/ 下的各个生成器实现；参考 config/model_config.yaml 中的配置。
"""

import os
from typing import List, Dict, Any, Optional

# 导入生成器和结果类型
from generators import (
    GenerationResult,
    LLMGenerator,
    MockGenerator,
    LocalLMStudioGenerator,
    OpenAIGenerator,
    VolcengineGenerator
)


def _load_config():
    """从项目配置文件加载设置。"""
    from pathlib import Path
    import yaml
    
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            pass
    return {}


def create_generator(generator_type: str, **kwargs) -> LLMGenerator:
    """
    生成器工厂函数
    
    Args:
        generator_type: 生成器类型 ('mock', 'local', 'openai', 'volcengine')
        **kwargs: 传递给生成器的参数
            - 对于 'local': base_url, model_name, rag_mode, timeout
            - 对于 'openai': api_key, model_name, rag_mode
            - 对于 'volcengine': api_key, model_name, rag_mode, base_url, timeout
            - rag_mode: 'strict'(仅知识库) | 'flexible'(知识库优先+补充)
            
    Returns:
        LLMGenerator 实例
        
    Raises:
        ValueError: 如果生成器类型不支持
    """
    rag_mode = kwargs.get('rag_mode', 'flexible')  # 默认灵活模式
    config = _load_config()
    llm_config = config.get("llm", {})
    local_config = llm_config.get("local", {})
    cloud_config = llm_config.get("cloud", {})
    volcengine_config = llm_config.get("volcengine", {})
    
    if generator_type == "mock":
        return MockGenerator(rag_mode=rag_mode, default_max_tokens=kwargs.get('max_tokens'))
    
    elif generator_type == "local":
        base_url = kwargs.get('base_url') or local_config.get('base_url')
        if not base_url:
            raise ValueError("本地生成器缺少 base_url，请先在 config/model_config.yaml 中配置")
        
        timeout = kwargs.get('timeout') or local_config.get('timeout', 1200)
        max_tokens = kwargs.get('max_tokens') or local_config.get('max_tokens')
        
        return LocalLMStudioGenerator(
            base_url=base_url,
            model_name=kwargs.get('model_name', local_config.get('model', 'local-model')),
            rag_mode=rag_mode,
            timeout=timeout,
            default_max_tokens=max_tokens
        )
    
    elif generator_type == "openai":
        api_key = kwargs.get('api_key') or cloud_config.get('api_key')
        if not api_key:
            raise ValueError("OpenAI生成器需要提供 api_key 参数")
        
        max_tokens = kwargs.get('max_tokens') or cloud_config.get('max_tokens')
        
        return OpenAIGenerator(
            api_key=api_key,
            model_name=kwargs.get('model_name', cloud_config.get('model', 'gpt-3.5-turbo')),
            rag_mode=rag_mode,
            default_max_tokens=max_tokens
        )
    
    elif generator_type == "volcengine":
        api_key = kwargs.get('api_key') or volcengine_config.get('api_key')
        if not api_key:
            raise ValueError("火山引擎生成器需要提供 api_key，请先在 config/model_config.yaml 中配置")
        
        base_url = kwargs.get('base_url') or volcengine_config.get('base_url', 'https://ark.cn-beijing.volces.com/api/v3')
        timeout = kwargs.get('timeout') or volcengine_config.get('timeout', 60)
        max_tokens = kwargs.get('max_tokens') or volcengine_config.get('max_tokens')
        
        return VolcengineGenerator(
            api_key=api_key,
            model_name=kwargs.get('model_name', volcengine_config.get('model', 'doubao-pro-4k')),
            base_url=base_url,
            rag_mode=rag_mode,
            timeout=timeout,
            default_max_tokens=max_tokens
        )
    
    else:
        raise ValueError(f"不支持的生成器类型: {generator_type}. 支持的类型有: 'mock', 'local', 'openai', 'volcengine'")


class RAGSystem:
    """完整的RAG系统
    
    协调检索器和生成器，实现端到端的RAG流程。
    """
    
    def __init__(self, retriever, generator):
        """
        初始化RAG系统
        
        Args:
            retriever: 检索器实例 (HybridRetriever)
            generator: LLM生成器实例 (LLMGenerator的子类)
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
            查询结果字典，包含:
                - query: 查询文本
                - answer: 生成的答案
                - citations: 引用的文档列表
                - confidence: 置信度
                - model: 使用的模型名称
                - context_count: 使用的上下文数量
                - context_preview: 上下文预览
                
            或在发生错误时:
                - query: 查询文本
                - error: 错误信息
                - context_count: 检索到的文档数量
        """
        try:
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
            generation_result = self.generator.generate(query, context_dict)
            
            # 构建完整结果
            result = {
                "query": query,
                "answer": generation_result.answer,
                "citations": generation_result.citations,
                "confidence": generation_result.confidence,
                "model": generation_result.model,
                "rag_mode": self.generator.rag_mode,  # 显示使用的RAG模式
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
                "context_count": len(context_dict) if 'context_dict' in locals() else 0
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
    
    # LM Studio参数（默认从配置文件读取）
    parser.add_argument("--lmstudio-url", default=None, 
                       help="LM Studio API地址")
    parser.add_argument("--lmstudio-model", default=None, help="LM Studio模型名称")
    parser.add_argument("--lmstudio-timeout", type=int, default=500,
                       help="LM Studio请求超时（秒）")
    # OpenAI参数
    parser.add_argument("--openai-key", help="OpenAI API密钥")
    parser.add_argument("--openai-model", default="gpt-3.5-turbo", help="OpenAI模型名称")
    
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
        
        gen_kwargs = {}
        if args.model_type == "local":
            if args.lmstudio_url:
                gen_kwargs['base_url'] = args.lmstudio_url
            if args.lmstudio_model:
                gen_kwargs['model_name'] = args.lmstudio_model
        elif args.model_type == "openai":
            if args.openai_key:
                gen_kwargs['api_key'] = args.openai_key
            if args.openai_model:
                gen_kwargs['model_name'] = args.openai_model
        
        generator = create_generator(args.model_type, **gen_kwargs)
        
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
