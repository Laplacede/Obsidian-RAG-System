"""
LLM生成器基类

定义生成器的通用接口和基本功能。

职责说明：
- 输入：接收检索器返回的上下文块列表和用户查询。
- 处理：定义生成答案的通用接口，提供通用的上下文格式化和提示词创建方法。
- 输出：返回 GenerationResult，包含 answer、citations、confidence 和 model。
- 依赖文件：所有生成器实现都继承这个基类。
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class GenerationResult:
    """生成结果
    
    Attributes:
        answer: 生成的答案文本
        citations: 答案中引用的文档列表
        confidence: 置信度分数 (0-1)
        model: 使用的模型名称
    """
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    model: str


class LLMGenerator:
    """LLM生成器基类
    
    定义所有LLM生成器必须实现的接口，提供通用的工具方法。
    不依赖具体配置，通过构造函数参数接收所有必要信息。
    """
    
    def __init__(self, model_name: str = "default", context_window: int = 4096, 
                 rag_mode: str = "flexible"):
        """
        初始化LLM生成器
        
        Args:
            model_name: 模型名称
            context_window: 上下文窗口大小
            rag_mode: RAG模式 ('strict'|'flexible')
                - strict: 仅基于知识库回答，信息不足时说明
                - flexible: 优先使用知识库，不足时可补充先验知识
        """
        self.model_name = model_name
        self.context_window = context_window
        self.rag_mode = rag_mode
    
    def generate(self, query: str, context: List[Dict[str, Any]], 
                max_tokens: int = 500) -> GenerationResult:
        """
        生成答案
        
        Args:
            query: 查询文本
            context: 检索到的上下文 (SearchResult转换后的字典列表)
            max_tokens: 最大生成token数
            
        Returns:
            生成结果
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现generate方法")
    
    def format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        格式化上下文为提示词可用的格式
        
        Args:
            context: 检索到的上下文列表，每个元素应该包含:
                - 'text': 块文本
                - 'metadata': 元数据字典 (包含 'title' 或 'source_document')
            
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
            
            # 格式化为 [文档i] 标题\n内容 的形式
            formatted.append(f"[文档{i}] {title}\n{content}\n")
        
        return "\n".join(formatted)
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        创建提示词
        
        Args:
            query: 查询文本
            context: 格式化后的上下文
            
        Returns:
            完整的提示词文本
        """
        if self.rag_mode == "strict":
            # 严格模式：仅基于知识库回答
            prompt = f"""你是一个基于知识库的问答助手。请基于以下提供的上下文回答用户的问题。

重要规则：
1. 仅基于提供的文档内容回答
2. 如果上下文中没有相关信息，请明确说"知识库中没有相关信息"
3. 所有答案都应该引用相关的文档
4. 使用中文回答

上下文文档：
{context}

用户问题：{query}

请根据上下文提供详细的回答，并引用相关文档（如[文档1]、[文档2]等）："""
        else:  # flexible mode (default)
            # 灵活模式：优先知识库，可补充先验知识
            prompt = f"""你是一个智能助手，拥有两个知识来源：

【来源1】知识库文档（优先使用）：
{context}

【来源2】你的先验知识：在知识库信息不足时可以补充

回答用户问题时，请遵循以下规则：
1. 首先尽量使用知识库文档回答（这是最权威的来源）
2. 如果知识库文档完整覆盖了问题，只基于文档回答
3. 如果知识库文档信息不足，可以补充你的先验知识来完善答案
4. 请明确标记答案中各部分的来源：
   - [知识库] = 直接来自提供的文档
   - [补充] = 你的先验知识补充
   - [混合] = 综合知识库和先验知识
5. 使用中文回答
6. 力求答案准确、完整、有用

用户问题：{query}

请提供详细的回答。如果使用了文档中的信息，请引用相关文档（如[文档1]、[文档2]等）："""
        
        return prompt
    
    def extract_citations(self, answer: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从答案中提取引用的文档
        
        Args:
            answer: 生成的答案
            context: 原始上下文
            
        Returns:
            引用文档的列表，每个元素包含:
                - 'document_index': 文档在上下文中的索引
                - 'title': 文档标题
                - 'score': 相关性分数
                - 'text_preview': 文本预览
        """
        citations = []
        
        # 查找类似[文档1]、[文档2]的引用模式
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
                    
                    # 避免重复引用
                    if citation not in citations:
                        citations.append(citation)
            except (ValueError, IndexError):
                continue
        
        return citations
    
    def calculate_confidence(self, context: List[Dict[str, Any]], 
                            has_knowledge_base_answer: bool = True) -> float:
        """
        基于检索质量计算置信度
        
        Args:
            context: 检索到的上下文
            has_knowledge_base_answer: 答案是否基于知识库
            
        Returns:
            置信度分数 (0-1)
            
        说明：
            - 5个以上高分文档（分数>0.7）：0.95 (极高信心)
            - 3-5个中等文档（分数>0.5）：0.85 (高信心)
            - 1-3个相关文档：0.7 (中等信心)
            - 没有相关文档但用先验知识：0.5 (低信心)
            - 没有相关文档且无法补充：0.2 (极低信心)
        """
        if not context:
            # 没有检索到任何文档
            return 0.5 if has_knowledge_base_answer else 0.2
        
        # 计算文档数量和平均分数
        doc_count = len(context)
        avg_score = sum(chunk.get('score', 0.0) for chunk in context) / doc_count if doc_count > 0 else 0.0
        high_quality_docs = sum(1 for chunk in context if chunk.get('score', 0.0) > 0.7)
        
        # 基于数量和质量计算置信度
        if doc_count >= 5 and high_quality_docs >= 3 and avg_score > 0.7:
            return 0.95
        elif doc_count >= 3 and avg_score > 0.6:
            return 0.85
        elif doc_count >= 1 and avg_score > 0.5:
            return 0.70
        elif doc_count >= 1:
            return 0.60
        else:
            return 0.50
