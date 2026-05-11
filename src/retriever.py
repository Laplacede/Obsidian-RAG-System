#!/usr/bin/env python3
"""
检索器
实现混合搜索（向量+关键词）功能
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class SearchResult:
    """搜索结果"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'vector', 'keyword', 'hybrid'


class HybridRetriever:
    """混合检索器（向量+关键词）"""
    
    def __init__(self, vector_store_dir: str = "./data/vector_store", 
                 chunk_data_dir: str = "./data/chunks"):
        """
        初始化检索器
        
        Args:
            vector_store_dir: 向量存储目录
            chunk_data_dir: 分块数据目录
        """
        self.vector_store_dir = vector_store_dir
        self.chunk_data_dir = chunk_data_dir
        self.vector_store = None
        self.keyword_index = None
        self.chunk_data = {}
        self.embedding_model = None
        
    def load_embedding_model(self):
        """
        加载嵌入模型
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            if self.embedding_model is None:
                print("正在加载BAAI嵌入模型...")
                self.embedding_model = SentenceTransformer(
                    "BAAI/bge-small-zh-v1.5",
                    device="cpu"
                )
                print(f"嵌入模型加载完成: {self.embedding_model.get_sentence_embedding_dimension()} 维")
            
        except ImportError:
            print("错误: 需要安装 sentence-transformers 包")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        生成查询嵌入
        
        Args:
            query: 查询文本
            
        Returns:
            嵌入向量
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # 生成嵌入
        embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
    
    def load_vector_store(self, chunk_type: str = "small"):
        """
        加载向量存储
        
        Args:
            chunk_type: 分块类型
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            print(f"正在加载向量存储: {self.vector_store_dir}")
            
            client = chromadb.PersistentClient(
                path=self.vector_store_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            collection_name = f"obsidian_{chunk_type}_chunks"
            self.vector_store = client.get_collection(collection_name)
            
            print(f"向量存储加载完成: {collection_name} ({self.vector_store.count()} 个文档)")
            
        except ImportError:
            print("错误: 需要安装 chromadb 包")
            raise
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            raise
    
    def load_chunk_data(self, chunk_type: str = "small"):
        """
        加载分块数据
        
        Args:
            chunk_type: 分块类型
        """
        chunk_file = os.path.join(self.chunk_data_dir, f"{chunk_type}_chunks.json")
        
        if not os.path.exists(chunk_file):
            print(f"分块文件不存在: {chunk_file}")
            return
        
        print(f"正在加载分块数据: {chunk_file}")
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            self.chunk_data[chunk_type] = json.load(f)
        
        print(f"分块数据加载完成: {len(self.chunk_data[chunk_type])} 个分块")
    
    def build_keyword_index(self, chunk_type: str = "small"):
        """
        构建关键词索引
        
        Args:
            chunk_type: 分块类型
        """
        if chunk_type not in self.chunk_data:
            self.load_chunk_data(chunk_type)
        
        print("正在构建关键词索引...")
        
        self.keyword_index = {
            'chunk_to_keywords': {},
            'keyword_to_chunks': defaultdict(list)
        }
        
        for chunk in self.chunk_data[chunk_type]:
            chunk_id = chunk.get('chunk_id', '')
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # 提取关键词
            keywords = self.extract_keywords(text, metadata)
            
            # 存储映射关系
            self.keyword_index['chunk_to_keywords'][chunk_id] = keywords
            
            for keyword in keywords:
                self.keyword_index['keyword_to_chunks'][keyword].append(chunk_id)
        
        print(f"关键词索引构建完成: {len(self.keyword_index['keyword_to_chunks'])} 个关键词")
    
    def extract_keywords(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """
        从文本和元数据中提取关键词
        
        Args:
            text: 文本内容
            metadata: 元数据
            
        Returns:
            关键词列表
        """
        keywords = set()
        
        # 从元数据中提取标签
        if 'yaml_frontmatter' in metadata:
            try:
                frontmatter = metadata['yaml_frontmatter']
                if isinstance(frontmatter, dict) and 'tags' in frontmatter:
                    tags = frontmatter['tags']
                    if isinstance(tags, list):
                        for tag in tags:
                            if tag:
                                keywords.add(tag)
                    elif isinstance(tags, str):
                        tag_list = tags.split(',')
                        for tag in tag_list:
                            tag = tag.strip()
                            if tag:
                                keywords.add(tag)
            except:
                pass
        
        # 直接从metadata中提取标签
        if 'tags' in metadata and isinstance(metadata['tags'], list):
            for tag in metadata['tags']:
                if tag:
                    keywords.add(tag)
        
        # 从标题中提取关键词
        if 'title' in metadata:
            title = metadata['title']
            # 提取标题中的技术术语
            tech_terms = re.findall(r'[A-Z][a-z]+|[A-Z]+|[a-z]+[A-Z][a-z]*', title)
            for term in tech_terms:
                if len(term) > 2:  # 过滤短词
                    keywords.add(term.lower())
        
        # 从文本中提取技术术语
        # 简单的技术术语提取（可以扩展为更复杂的NLP方法）
        tech_patterns = [
            r'\b(?:class|struct|interface|enum|template|namespace|typedef|using)\b',
            r'\b(?:function|method|constructor|destructor|operator|parameter|argument)\b',
            r'\b(?:variable|constant|pointer|reference|array|vector|list|map|set)\b',
            r'\b(?:inheritance|polymorphism|encapsulation|abstraction|composition|aggregation)\b',
            r'\b(?:algorithm|data structure|complexity|optimization|implementation)\b',
            r'\b(?:memory|storage|cache|register|processor|instruction|assembly)\b',
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keywords.add(match.lower())
        
        return list(keywords)
    
    def vector_search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """
        向量搜索
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.vector_store:
            raise ValueError("向量存储未加载，请先调用 load_vector_store()")
        
        print(f"执行向量搜索: '{query}'")
        
        # 使用我们自己的BAAI模型生成查询嵌入
        query_embedding = self.generate_query_embedding(query)
        
        # 使用查询嵌入进行搜索
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        search_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                chunk_id = results['ids'][0][i]
                text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                score = results['distances'][0][i] if results.get('distances') else 0.0
                
                # 将距离转换为相似度分数（1 - 距离）
                similarity_score = 1.0 - score if score <= 1.0 else 0.0
                
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    score=similarity_score,
                    source='vector'
                ))
        
        return search_results
    
    def keyword_search(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """
        关键词搜索
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.keyword_index:
            raise ValueError("关键词索引未构建，请先调用 build_keyword_index()")
        
        print(f"执行关键词搜索: '{query}'")
        
        # 从查询中提取关键词
        query_keywords = set()
        words = re.findall(r'\b\w+\b', query.lower())
        query_keywords.update(words)
        
        # 计算每个分块的相关性分数
        chunk_scores = defaultdict(float)
        for keyword in query_keywords:
            if keyword in self.keyword_index['keyword_to_chunks']:
                chunk_ids = self.keyword_index['keyword_to_chunks'][keyword]
                for chunk_id in chunk_ids:
                    # 基础分数：关键词匹配
                    chunk_scores[chunk_id] += 1.0
                    
                    # 如果分块包含多个查询关键词，增加分数
                    chunk_keywords = set(self.keyword_index['chunk_to_keywords'][chunk_id])
                    common_keywords = chunk_keywords.intersection(query_keywords)
                    if len(common_keywords) > 1:
                        chunk_scores[chunk_id] += len(common_keywords) * 0.5
        
        # 按分数排序
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 获取前n个结果
        search_results = []
        for chunk_id, score in sorted_chunks[:n_results]:
            # 查找分块数据
            chunk_data = None
            for chunk_type, chunks in self.chunk_data.items():
                for chunk in chunks:
                    if chunk.get('chunk_id') == chunk_id:
                        chunk_data = chunk
                        break
                if chunk_data:
                    break
            
            if chunk_data:
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=chunk_data.get('text', ''),
                    metadata=chunk_data.get('metadata', {}),
                    score=score,
                    source='keyword'
                ))
        
        return search_results
    
    def hybrid_search(self, query: str, n_results: int = 5, 
                     vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[SearchResult]:
        """
        混合搜索（向量+关键词）
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            vector_weight: 向量搜索权重
            keyword_weight: 关键词搜索权重
            
        Returns:
            搜索结果列表
        """
        print(f"执行混合搜索: '{query}' (向量权重: {vector_weight}, 关键词权重: {keyword_weight})")
        
        # 执行两种搜索
        vector_results = self.vector_search(query, n_results * 2)
        keyword_results = self.keyword_search(query, n_results * 2)
        
        # 合并结果
        all_results = {}
        
        # 添加向量搜索结果
        for result in vector_results:
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = {
                    'result': result,
                    'vector_score': result.score * vector_weight,
                    'keyword_score': 0.0
                }
            else:
                all_results[result.chunk_id]['vector_score'] = result.score * vector_weight
        
        # 添加关键词搜索结果
        for result in keyword_results:
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = {
                    'result': result,
                    'vector_score': 0.0,
                    'keyword_score': result.score * keyword_weight
                }
            else:
                all_results[result.chunk_id]['keyword_score'] = result.score * keyword_weight
        
        # 计算混合分数
        for chunk_id, data in all_results.items():
            total_score = data['vector_score'] + data['keyword_score']
            data['total_score'] = total_score
        
        # 按混合分数排序
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        # 创建最终结果
        final_results = []
        for chunk_id, data in sorted_results[:n_results]:
            result = data['result']
            # 创建新的SearchResult，更新分数和来源
            final_results.append(SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                metadata=result.metadata,
                score=data['total_score'],
                source='hybrid'
            ))
        
        return final_results
    
    def query_expansion(self, query: str) -> str:
        """
        查询扩展
        
        Args:
            query: 原始查询
            
        Returns:
            扩展后的查询
        """
        # 简单的查询扩展：添加技术术语同义词
        expansion_dict = {
            '类': 'class 对象 类型',
            '函数': 'function 方法 method',
            '变量': 'variable 变量名 标识符',
            '指针': 'pointer 引用 reference',
            '内存': 'memory 存储 storage',
            '算法': 'algorithm 算法设计 复杂度',
            '数据结构': 'data structure 数据组织',
            '继承': 'inheritance 派生 子类',
            '多态': 'polymorphism 动态绑定',
            '封装': 'encapsulation 信息隐藏',
        }
        
        expanded_query = query
        for term, synonyms in expansion_dict.items():
            if term in query:
                expanded_query += " " + synonyms
        
        if expanded_query != query:
            print(f"查询扩展: '{query}' -> '{expanded_query}'")
        
        return expanded_query
    
    def retrieve(self, query: str, n_results: int = 5, 
                use_hybrid: bool = True, use_expansion: bool = True,
                use_reranking: bool = False, rerank_method: str = "simple") -> List[SearchResult]:
        """
        检索接口
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            use_hybrid: 是否使用混合搜索
            use_expansion: 是否使用查询扩展
            use_reranking: 是否使用重排序
            rerank_method: 重排序方法（simple或cross_encoder）
            
        Returns:
            搜索结果列表
        """
        # 查询扩展
        final_query = self.query_expansion(query) if use_expansion else query
        
        # 执行搜索
        if use_hybrid:
            results = self.hybrid_search(final_query, n_results * 2)  # 获取更多结果用于重排序
        else:
            results = self.vector_search(final_query, n_results * 2)
        
        # 重排序
        if use_reranking and results:
            results = self.rerank_results(final_query, results, n_results, rerank_method)
        else:
            # 如果没有重排序，只取前n个结果
            results = results[:n_results]
        
        return results
    
    def rerank_results(self, query: str, results: List[SearchResult], n_results: int, 
                      method: str = "simple") -> List[SearchResult]:
        """
        重排序检索结果
        
        Args:
            query: 查询文本
            results: 原始检索结果
            n_results: 返回结果数量
            method: 重排序方法
            
        Returns:
            重排序后的结果
        """
        try:
            # 将SearchResult转换为字典格式
            results_dict = []
            for result in results:
                results_dict.append({
                    'chunk_id': result.chunk_id,
                    'text': result.text,
                    'metadata': result.metadata,
                    'score': result.score,
                    'source': 'vector' if hasattr(result, 'source') and result.source == 'vector' else 'keyword'
                })
            
            # 根据方法选择重排序器
            if method == "simple":
                # 动态加载SimpleReranker
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from reranker import SimpleReranker
                reranker = SimpleReranker()
            else:
                # 动态加载CrossEncoderReranker
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
                reranker.load_model()
            
            # 执行重排序
            reranked_dict = reranker.rerank(query, results_dict, n_results)
            
            # 将重排序结果转换回SearchResult格式
            reranked_results = []
            for reranked in reranked_dict:
                # 找到对应的原始结果
                original_result = next((r for r in results if r.chunk_id == reranked.chunk_id), None)
                if original_result:
                    # 创建新的SearchResult，使用重排序分数
                    reranked_result = SearchResult(
                        chunk_id=reranked.chunk_id,
                        text=reranked.text,
                        metadata=reranked.metadata,
                        score=reranked.reranked_score,  # 使用重排序分数
                        source=original_result.source
                    )
                    reranked_results.append(reranked_result)
            
            return reranked_results
            
        except Exception as e:
            print(f"重排序失败: {e}")
            # 如果重排序失败，返回原始结果的前n个
            return results[:n_results]
    
    def format_results(self, results: List[SearchResult]) -> str:
        """
        格式化搜索结果
        
        Args:
            results: 搜索结果列表
            
        Returns:
            格式化后的字符串
        """
        output = []
        
        for i, result in enumerate(results, 1):
            # 提取元数据信息
            metadata = result.metadata
            # 尝试获取标题，如果没有则使用源文件名
            title = metadata.get('title', '')
            if not title:
                title = metadata.get('source_document', '未知标题')
            source_file = metadata.get('source_file', metadata.get('source_document', '未知文件'))
            
            # 解析YAML frontmatter获取标签
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
            
            output.append(f"{i}. [{result.score:.3f}] {title}")
            output.append(f"   来源: {source_file}")
            if tags:
                output.append(f"   标签: {', '.join([t.strip() for t in tags])}")
            output.append(f"   匹配度: {result.score:.3f} ({result.source})")
            output.append(f"   内容: {result.text[:200]}...")
            output.append("")
        
        return "\n".join(output)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="混合检索器")
    parser.add_argument("--query", "-q", required=True, help="查询文本")
    parser.add_argument("--results", "-r", type=int, default=5, help="返回结果数量")
    parser.add_argument("--chunk-type", "-t", default="small", choices=["small", "medium", "large"],
                       help="分块类型")
    parser.add_argument("--hybrid", action="store_true", help="使用混合搜索")
    parser.add_argument("--no-expansion", action="store_true", help="禁用查询扩展")
    parser.add_argument("--rerank", action="store_true", help="启用重排序")
    parser.add_argument("--rerank-method", default="simple", choices=["simple", "cross_encoder"],
                       help="重排序方法")
    parser.add_argument("--vector-weight", type=float, default=0.7, help="向量搜索权重")
    parser.add_argument("--keyword-weight", type=float, default=0.3, help="关键词搜索权重")
    
    args = parser.parse_args()
    
    try:
        # 创建检索器
        retriever = HybridRetriever(
            vector_store_dir="./data/vector_store",
            chunk_data_dir="./data/chunks"
        )
        
        # 加载数据
        retriever.load_vector_store(args.chunk_type)
        retriever.load_chunk_data(args.chunk_type)
        retriever.build_keyword_index(args.chunk_type)
        
        # 执行检索
        results = retriever.retrieve(
            query=args.query,
            n_results=args.results,
            use_hybrid=args.hybrid,
            use_expansion=not args.no_expansion,
            use_reranking=args.rerank,
            rerank_method=args.rerank_method
        )
        
        # 输出结果
        print(f"\n=== 检索结果 ===\n")
        print(f"查询: '{args.query}'")
        print(f"分块类型: {args.chunk_type}")
        print(f"搜索模式: {'混合搜索' if args.hybrid else '向量搜索'}")
        print(f"查询扩展: {'启用' if not args.no_expansion else '禁用'}")
        print(f"重排序: {'启用' if args.rerank else '禁用'} ({args.rerank_method if args.rerank else '无'})")
        print(f"\n找到 {len(results)} 个相关结果:\n")
        
        if results:
            formatted = retriever.format_results(results)
            print(formatted)
        else:
            print("未找到相关结果")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()