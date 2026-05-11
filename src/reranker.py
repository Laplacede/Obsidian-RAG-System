#!/usr/bin/env python3
"""
重排序器
使用交叉编码器或启发式方法优化检索结果
"""

import os
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re


@dataclass
class RerankedResult:
    """重排序结果"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    original_score: float
    reranked_score: float
    source: str


class SimpleReranker:
    """简单重排序器（使用启发式方法）"""
    
    def __init__(self):
        """初始化重排序器"""
        self.weights = {
            'title_match': 0.3,      # 标题匹配
            'tag_match': 0.2,        # 标签匹配
            'keyword_density': 0.25, # 关键词密度
            'recency': 0.1,          # 时效性（如果有时间信息）
            'length_score': 0.15     # 内容长度评分
        }
    
    def rerank(self, query: str, results: List[Dict[str, Any]], n_results: int = 5) -> List[RerankedResult]:
        """
        重排序检索结果
        
        Args:
            query: 查询文本
            results: 原始检索结果
            n_results: 返回结果数量
            
        Returns:
            重排序后的结果
        """
        print(f"正在重排序 {len(results)} 个结果...")
        
        reranked_results = []
        
        for result in results:
            # 计算各项分数
            title_score = self._calculate_title_score(query, result)
            tag_score = self._calculate_tag_score(query, result)
            keyword_score = self._calculate_keyword_density(query, result)
            recency_score = self._calculate_recency_score(result)
            length_score = self._calculate_length_score(result)
            
            # 计算加权总分
            total_score = (
                title_score * self.weights['title_match'] +
                tag_score * self.weights['tag_match'] +
                keyword_score * self.weights['keyword_density'] +
                recency_score * self.weights['recency'] +
                length_score * self.weights['length_score']
            )
            
            # 创建重排序结果
            reranked_result = RerankedResult(
                chunk_id=result.get('chunk_id', ''),
                text=result.get('text', ''),
                metadata=result.get('metadata', {}),
                original_score=result.get('score', 0.0),
                reranked_score=total_score,
                source=result.get('source', 'unknown')
            )
            
            reranked_results.append(reranked_result)
        
        # 按重排序分数排序
        reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)
        
        # 返回前n个结果
        return reranked_results[:n_results]
    
    def _calculate_title_score(self, query: str, result: Dict[str, Any]) -> float:
        """计算标题匹配分数"""
        metadata = result.get('metadata', {})
        
        # 获取标题
        title = metadata.get('title', '')
        if not title:
            title = metadata.get('source_document', '')
        
        if not title:
            return 0.0
        
        # 简单的标题匹配：检查查询关键词是否出现在标题中
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        title_words = set(re.findall(r'\b\w+\b', title.lower()))
        
        # 计算交集
        common_words = query_words.intersection(title_words)
        
        if not query_words:
            return 0.0
        
        # 分数 = 共同词数 / 查询词数
        return len(common_words) / len(query_words)
    
    def _calculate_tag_score(self, query: str, result: Dict[str, Any]) -> float:
        """计算标签匹配分数"""
        metadata = result.get('metadata', {})
        
        # 获取标签
        tags = []
        if 'yaml_frontmatter' in metadata:
            try:
                frontmatter = metadata['yaml_frontmatter']
                if isinstance(frontmatter, dict) and 'tags' in frontmatter:
                    tags = frontmatter['tags']
                    if not isinstance(tags, list):
                        tags = []
            except:
                pass
        
        # 如果没有从yaml_frontmatter获取到，尝试直接从metadata获取
        if not tags and 'tags' in metadata and isinstance(metadata['tags'], list):
            tags = metadata['tags']
        
        if not tags:
            return 0.0
        
        # 提取查询中的关键词
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # 检查是否有标签包含查询关键词
        tag_matches = 0
        for tag in tags:
            if isinstance(tag, str):
                tag_lower = tag.lower()
                for word in query_words:
                    if word in tag_lower:
                        tag_matches += 1
                        break
        
        # 分数 = 匹配的标签数 / 总标签数（如果总标签数>0）
        if tags:
            return tag_matches / len(tags)
        return 0.0
    
    def _calculate_keyword_density(self, query: str, result: Dict[str, Any]) -> float:
        """计算关键词密度分数"""
        text = result.get('text', '')
        
        if not text:
            return 0.0
        
        # 提取查询关键词
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # 统计文本中查询关键词的出现次数
        text_lower = text.lower()
        total_matches = 0
        
        for word in query_words:
            # 简单的词频统计
            matches = len(re.findall(rf'\b{re.escape(word)}\b', text_lower))
            total_matches += matches
        
        # 计算关键词密度（每100词中的关键词数）
        word_count = len(re.findall(r'\b\w+\b', text_lower))
        if word_count > 0:
            density = (total_matches / word_count) * 100
            
            # 归一化到0-1范围（假设密度超过10%为满分）
            return min(density / 10.0, 1.0)
        
        return 0.0
    
    def _calculate_recency_score(self, result: Dict[str, Any]) -> float:
        """计算时效性分数"""
        # 如果没有时间信息，返回基础分
        metadata = result.get('metadata', {})
        
        # 这里可以根据实际的时间信息进行计算
        # 例如：如果有创建时间，可以计算时间衰减
        # 目前返回一个基础分
        return 0.5
    
    def _calculate_length_score(self, result: Dict[str, Any]) -> float:
        """计算内容长度分数"""
        text = result.get('text', '')
        
        if not text:
            return 0.0
        
        # 计算词数
        word_count = len(re.findall(r'\b\w+\b', text))
        
        # 理想长度：100-500词
        if word_count < 50:
            # 太短，分数较低
            return 0.3
        elif word_count < 100:
            return 0.6
        elif word_count < 500:
            # 理想长度
            return 1.0
        elif word_count < 1000:
            # 较长，但还可以接受
            return 0.8
        else:
            # 太长，分数较低
            return 0.5


class CrossEncoderReranker:
    """交叉编码器重排序器（使用sentence-transformers的交叉编码器）"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化交叉编码器
        
        Args:
            model_name: 交叉编码器模型名称
        """
        self.model_name = model_name
        self.model = None
    
    def load_model(self):
        """加载交叉编码器模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            print(f"正在加载交叉编码器: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            print("交叉编码器加载完成")
            
        except ImportError:
            print("错误: 需要安装 sentence-transformers 包")
            raise
    
    def rerank(self, query: str, results: List[Dict[str, Any]], n_results: int = 5) -> List[RerankedResult]:
        """
        使用交叉编码器重排序
        
        Args:
            query: 查询文本
            results: 原始检索结果
            n_results: 返回结果数量
            
        Returns:
            重排序后的结果
        """
        if self.model is None:
            self.load_model()
        
        print(f"使用交叉编码器重排序 {len(results)} 个结果...")
        
        # 准备查询-文档对
        pairs = []
        for result in results:
            text = result.get('text', '')
            if text:
                pairs.append([query, text])
            else:
                pairs.append([query, ""])
        
        # 使用交叉编码器计算分数
        if pairs:
            scores = self.model.predict(pairs)
        else:
            scores = []
        
        # 创建重排序结果
        reranked_results = []
        for i, (result, score) in enumerate(zip(results, scores)):
            reranked_result = RerankedResult(
                chunk_id=result.get('chunk_id', ''),
                text=result.get('text', ''),
                metadata=result.get('metadata', {}),
                original_score=result.get('score', 0.0),
                reranked_score=float(score),
                source=result.get('source', 'unknown')
            )
            reranked_results.append(reranked_result)
        
        # 按交叉编码器分数排序
        reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)
        
        return reranked_results[:n_results]


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="重排序器")
    parser.add_argument("--query", "-q", required=True, help="查询文本")
    parser.add_argument("--results-file", "-f", required=True, help="检索结果文件路径")
    parser.add_argument("--results", "-r", type=int, default=5, help="返回结果数量")
    parser.add_argument("--method", "-m", default="simple", choices=["simple", "cross_encoder"],
                       help="重排序方法")
    parser.add_argument("--model", default="BAAI/bge-reranker-base", help="交叉编码器模型名称")
    
    args = parser.parse_args()
    
    try:
        # 加载检索结果
        print(f"正在加载检索结果: {args.results_file}")
        with open(args.results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # 确保results是列表
        if isinstance(results_data, dict) and 'results' in results_data:
            results = results_data['results']
        elif isinstance(results_data, list):
            results = results_data
        else:
            print("错误: 检索结果文件格式不正确")
            return
        
        print(f"加载了 {len(results)} 个检索结果")
        
        # 创建重排序器
        if args.method == "simple":
            reranker = SimpleReranker()
        else:
            reranker = CrossEncoderReranker(model_name=args.model)
        
        # 执行重排序
        reranked_results = reranker.rerank(args.query, results, args.results)
        
        # 输出结果
        print(f"\n=== 重排序结果 ===\n")
        print(f"查询: '{args.query}'")
        print(f"重排序方法: {args.method}")
        print(f"\n前 {len(reranked_results)} 个结果:\n")
        
        for i, result in enumerate(reranked_results, 1):
            metadata = result.metadata
            title = metadata.get('title', '')
            if not title:
                title = metadata.get('source_document', '未知标题')
            
            print(f"{i}. [{result.reranked_score:.3f}] {title}")
            print(f"   原始分数: {result.original_score:.3f}")
            print(f"   重排序分数: {result.reranked_score:.3f}")
            print(f"   来源: {result.source}")
            print(f"   内容预览: {result.text[:150]}...")
            print()
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()