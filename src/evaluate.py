#!/usr/bin/env python3
"""
系统评估脚本
评估RAG系统的检索准确率和性能
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple
import argparse


class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self, retriever, test_cases: List[Dict[str, Any]]):
        """
        初始化评估器
        
        Args:
            retriever: 检索器实例
            test_cases: 测试用例列表
        """
        self.retriever = retriever
        self.test_cases = test_cases
        self.results = []
    
    def evaluate_retrieval(self, use_hybrid: bool = True, use_reranking: bool = False, 
                          n_results: int = 5) -> Dict[str, Any]:
        """
        评估检索性能
        
        Args:
            use_hybrid: 是否使用混合搜索
            use_reranking: 是否使用重排序
            n_results: 返回结果数量
            
        Returns:
            评估结果
        """
        print(f"开始评估检索性能...")
        print(f"测试用例数量: {len(self.test_cases)}")
        print(f"配置: 混合搜索={'启用' if use_hybrid else '禁用'}, "
              f"重排序={'启用' if use_reranking else '禁用'}, 结果数={n_results}")
        print()
        
        total_cases = len(self.test_cases)
        metrics = {
            "precision_at_1": 0.0,
            "precision_at_3": 0.0,
            "precision_at_5": 0.0,
            "recall_at_5": 0.0,
            "avg_response_time": 0.0,
            "success_rate": 0.0
        }
        
        successful_cases = 0
        
        for i, test_case in enumerate(self.test_cases, 1):
            query = test_case["query"]
            expected_docs = test_case.get("expected_documents", [])
            
            print(f"测试 {i}/{total_cases}: '{query}'")
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 执行检索
                results = self.retriever.retrieve(
                    query=query,
                    n_results=n_results,
                    use_hybrid=use_hybrid,
                    use_expansion=True,
                    use_reranking=use_reranking
                )
                
                # 记录结束时间
                end_time = time.time()
                response_time = end_time - start_time
                
                # 提取检索到的文档ID
                retrieved_docs = []
                for result in results:
                    metadata = result.metadata
                    doc_id = metadata.get("source_document", "")
                    if doc_id:
                        retrieved_docs.append(doc_id)
                
                # 计算精确率
                precision_1 = self._calculate_precision(retrieved_docs[:1], expected_docs)
                precision_3 = self._calculate_precision(retrieved_docs[:3], expected_docs)
                precision_5 = self._calculate_precision(retrieved_docs[:5], expected_docs)
                
                # 计算召回率
                recall_5 = self._calculate_recall(retrieved_docs[:5], expected_docs)
                
                # 记录结果
                case_result = {
                    "query": query,
                    "retrieved_docs": retrieved_docs,
                    "expected_docs": expected_docs,
                    "precision_at_1": precision_1,
                    "precision_at_3": precision_3,
                    "precision_at_5": precision_5,
                    "recall_at_5": recall_5,
                    "response_time": response_time,
                    "success": True
                }
                
                self.results.append(case_result)
                
                # 更新指标
                metrics["precision_at_1"] += precision_1
                metrics["precision_at_3"] += precision_3
                metrics["precision_at_5"] += precision_5
                metrics["recall_at_5"] += recall_5
                metrics["avg_response_time"] += response_time
                successful_cases += 1
                
                print(f"  精确率@1: {precision_1:.3f}, @3: {precision_3:.3f}, @5: {precision_5:.3f}")
                print(f"  召回率@5: {recall_5:.3f}, 响应时间: {response_time:.3f}s")
                print()
                
            except Exception as e:
                print(f"  错误: {e}")
                case_result = {
                    "query": query,
                    "error": str(e),
                    "success": False
                }
                self.results.append(case_result)
                print()
        
        # 计算平均指标
        if successful_cases > 0:
            metrics["precision_at_1"] /= successful_cases
            metrics["precision_at_3"] /= successful_cases
            metrics["precision_at_5"] /= successful_cases
            metrics["recall_at_5"] /= successful_cases
            metrics["avg_response_time"] /= successful_cases
            metrics["success_rate"] = successful_cases / total_cases
        
        return metrics
    
    def _calculate_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        计算精确率
        
        Args:
            retrieved: 检索到的文档
            relevant: 相关文档
            
        Returns:
            精确率
        """
        if not retrieved:
            return 0.0
        
        # 计算检索到的相关文档数量
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        
        return relevant_retrieved / len(retrieved)
    
    def _calculate_recall(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        计算召回率
        
        Args:
            retrieved: 检索到的文档
            relevant: 相关文档
            
        Returns:
            召回率
        """
        if not relevant:
            return 0.0
        
        # 计算检索到的相关文档数量
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        
        return relevant_retrieved / len(relevant)
    
    def generate_report(self, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            config: 配置信息
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 60)
        report.append("RAG系统评估报告")
        report.append("=" * 60)
        report.append("")
        
        # 配置信息
        report.append("配置信息:")
        report.append(f"  分块类型: {config.get('chunk_type', 'small')}")
        report.append(f"  混合搜索: {'启用' if config.get('use_hybrid', True) else '禁用'}")
        report.append(f"  重排序: {'启用' if config.get('use_reranking', False) else '禁用'}")
        report.append(f"  结果数量: {config.get('n_results', 5)}")
        report.append("")
        
        # 测试统计
        total_cases = len(self.test_cases)
        successful_cases = sum(1 for r in self.results if r.get('success', False))
        report.append("测试统计:")
        report.append(f"  总测试用例: {total_cases}")
        report.append(f"  成功用例: {successful_cases}")
        report.append(f"  成功率: {metrics.get('success_rate', 0.0):.2%}")
        report.append("")
        
        # 性能指标
        report.append("性能指标:")
        report.append(f"  精确率@1: {metrics.get('precision_at_1', 0.0):.3f}")
        report.append(f"  精确率@3: {metrics.get('precision_at_3', 0.0):.3f}")
        report.append(f"  精确率@5: {metrics.get('precision_at_5', 0.0):.3f}")
        report.append(f"  召回率@5: {metrics.get('recall_at_5', 0.0):.3f}")
        report.append(f"  平均响应时间: {metrics.get('avg_response_time', 0.0):.3f} 秒")
        report.append("")
        
        # 详细结果
        report.append("详细结果:")
        for i, result in enumerate(self.results[:10], 1):  # 只显示前10个
            if result.get('success', False):
                report.append(f"{i}. '{result['query']}'")
                report.append(f"   精确率@1: {result['precision_at_1']:.3f}, @3: {result['precision_at_3']:.3f}")
                report.append(f"   检索到: {', '.join(result['retrieved_docs'][:3])}...")
            else:
                report.append(f"{i}. '{result['query']}' - 失败: {result.get('error', '未知错误')}")
        
        if len(self.results) > 10:
            report.append(f"... 还有 {len(self.results) - 10} 个结果未显示")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, output_path: str):
        """
        保存评估结果
        
        Args:
            output_path: 输出文件路径
        """
        try:
            results_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_cases": self.test_cases,
                "results": self.results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)
            
            print(f"评估结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存结果失败: {e}")


def load_test_cases(test_file: str) -> List[Dict[str, Any]]:
    """
    加载测试用例
    
    Args:
        test_file: 测试文件路径
        
    Returns:
        测试用例列表
    """
    try:
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    
    # 如果没有测试文件，创建一些示例测试用例
    return [
        {
            "query": "C++类与对象",
            "expected_documents": ["类与对象.md", "函数类内定义与类外定义.md"]
        },
        {
            "query": "C++继承", 
            "expected_documents": ["继承.md"]
        },
        {
            "query": "C++内存管理",
            "expected_documents": ["内存管理（new & delete).md", "C++/堆与栈.md"]
        },
        {
            "query": "计算机组成原理",
            "expected_documents": ["寻址演示脚本（manim）.md"]
        },
        {
            "query": "友元函数",
            "expected_documents": ["友元.md"]
        }
    ]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统评估工具")
    parser.add_argument("--test-file", "-t", default="./data/evaluation/test_cases.json",
                       help="测试用例文件路径")
    parser.add_argument("--output", "-o", default="./data/evaluation/results.json",
                       help="评估结果输出路径")
    parser.add_argument("--hybrid", action="store_true", help="使用混合搜索")
    parser.add_argument("--rerank", action="store_true", help="启用重排序")
    parser.add_argument("--results", "-r", type=int, default=5, help="返回结果数量")
    
    args = parser.parse_args()
    
    try:
        # 创建必要的目录
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # 加载测试用例
        print("正在加载测试用例...")
        test_cases = load_test_cases(args.test_file)
        print(f"加载了 {len(test_cases)} 个测试用例")
        
        # 初始化检索器
        print("正在初始化检索器...")
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from retriever import HybridRetriever
        
        retriever = HybridRetriever(
            vector_store_dir="./data/vector_store",
            chunk_data_dir="./data/chunks"
        )
        
        # 加载数据
        retriever.load_vector_store("small")
        retriever.load_chunk_data("small")
        retriever.build_keyword_index("small")
        
        # 创建评估器
        evaluator = RAGEvaluator(retriever, test_cases)
        
        # 执行评估
        config = {
            "chunk_type": "small",
            "use_hybrid": args.hybrid,
            "use_reranking": args.rerank,
            "n_results": args.results
        }
        
        metrics = evaluator.evaluate_retrieval(
            use_hybrid=args.hybrid,
            use_reranking=args.rerank,
            n_results=args.results
        )
        
        # 生成报告
        report = evaluator.generate_report(metrics, config)
        print("\n" + report)
        
        # 保存结果
        evaluator.save_results(args.output)
        
        # 保存报告到文件
        report_file = args.output.replace(".json", "_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"评估报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()