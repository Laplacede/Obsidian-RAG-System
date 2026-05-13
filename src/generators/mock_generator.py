"""
模拟LLM生成器

用于测试和演示，不需要真实的LLM调用。

职责说明：
- 输入：接收查询和上下文。
- 处理：生成模拟的回答，用于系统测试和演示。
- 输出：返回 GenerationResult，包含虚拟的 answer、citations、confidence。
- 依赖文件：继承自 base.py 的 LLMGenerator。
"""

from typing import List, Dict, Any
from .base import LLMGenerator, GenerationResult


class MockGenerator(LLMGenerator):
    """模拟生成器，用于测试和演示
    
    生成虚拟的回答，不实际调用任何LLM。
    """
    
    def __init__(self, model_name: str = "mock-model", rag_mode: str = "flexible"):
        """初始化模拟生成器"""
        super().__init__(model_name, rag_mode=rag_mode)
    
    def generate(self, query: str, context: List[Dict[str, Any]], 
                max_tokens: int = 500) -> GenerationResult:
        """
        生成模拟答案（用于测试）
        
        Args:
            query: 查询文本
            context: 检索到的上下文
            max_tokens: 最大生成token数 (模拟生成器忽略此参数)
            
        Returns:
            模拟的生成结果
        """
        # 格式化上下文
        formatted_context = self.format_context(context)
        
        # 创建模拟回答
        answer = f"这是基于查询'{query}'的模拟回答（{self.rag_mode}模式）。\n\n"
        answer += f"我检索到了{len(context)}个相关文档。\n\n"
        
        # 添加一些模拟的引用（前两个文档）
        for i, chunk in enumerate(context[:2], 1):
            metadata = chunk.get('metadata', {})
            title = metadata.get('source_document', f"文档{i}")
            answer += f"根据[文档{i}]《{title}》的内容，"
            answer += f"我可以说该主题的重点是：{formatted_context[:50]}...\n\n"
        
        answer += "\n这是一个模拟的回答，实际使用中会连接到真正的LLM。"
        
        # 创建模拟引用
        citations = []
        for i, chunk in enumerate(context[:2], 1):
            metadata = chunk.get('metadata', {})
            citations.append({
                "document_index": i - 1,
                "title": metadata.get('source_document', f"文档{i}"),
                "score": chunk.get('score', 0.0),
                "text_preview": chunk.get('text', '')[:100] + "..."
            })
        
        # 使用动态置信度
        confidence = self.calculate_confidence(context, has_knowledge_base_answer=True)
        
        return GenerationResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            model="mock-model"
        )
