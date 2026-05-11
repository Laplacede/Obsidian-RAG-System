#!/usr/bin/env python3
"""
反馈循环机制
收集用户反馈，持续改进RAG系统
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse


class FeedbackCollector:
    """反馈收集器"""
    
    def __init__(self, feedback_file: str = "./data/feedback/feedback.json"):
        """
        初始化反馈收集器
        
        Args:
            feedback_file: 反馈文件路径
        """
        self.feedback_file = feedback_file
        self.feedback_data = []
        self.load_feedback()
    
    def load_feedback(self):
        """加载反馈数据"""
        try:
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
        except:
            self.feedback_data = []
    
    def save_feedback(self):
        """保存反馈数据"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存反馈失败: {e}")
            return False
    
    def add_feedback(self, query: str, answer: str, 
                    rating: int, comments: str = "",
                    retrieved_docs: List[str] = None,
                    user_correction: str = None) -> bool:
        """
        添加反馈
        
        Args:
            query: 查询文本
            answer: 系统生成的答案
            rating: 评分 (1-5)
            comments: 用户评论
            retrieved_docs: 检索到的文档
            user_correction: 用户修正的答案
            
        Returns:
            是否成功添加
        """
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "rating": max(1, min(5, rating)),  # 确保在1-5范围内
            "comments": comments,
            "retrieved_docs": retrieved_docs or [],
            "user_correction": user_correction,
            "improved": False  # 标记是否已用于改进系统
        }
        
        self.feedback_data.append(feedback)
        return self.save_feedback()
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        获取反馈统计
        
        Returns:
            反馈统计信息
        """
        if not self.feedback_data:
            return {
                "total_feedback": 0,
                "avg_rating": 0.0,
                "recent_feedback": 0
            }
        
        total = len(self.feedback_data)
        
        # 计算平均评分
        ratings = [f.get("rating", 3) for f in self.feedback_data]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        # 计算最近30天的反馈数量
        recent_count = 0
        thirty_days_ago = time.time() - 30 * 24 * 60 * 60
        
        for feedback in self.feedback_data:
            timestamp_str = feedback.get("timestamp", "")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                    if timestamp > thirty_days_ago:
                        recent_count += 1
                except:
                    pass
        
        return {
            "total_feedback": total,
            "avg_rating": round(avg_rating, 2),
            "recent_feedback": recent_count,
            "improved_count": sum(1 for f in self.feedback_data if f.get("improved", False))
        }
    
    def get_low_rated_feedback(self, threshold: int = 3) -> List[Dict[str, Any]]:
        """
        获取低评分反馈
        
        Args:
            threshold: 评分阈值（低于此值的反馈）
            
        Returns:
            低评分反馈列表
        """
        return [
            f for f in self.feedback_data 
            if f.get("rating", 5) < threshold and not f.get("improved", False)
        ]
    
    def mark_as_improved(self, feedback_index: int) -> bool:
        """
        标记反馈为已改进
        
        Args:
            feedback_index: 反馈索引
            
        Returns:
            是否成功标记
        """
        if 0 <= feedback_index < len(self.feedback_data):
            self.feedback_data[feedback_index]["improved"] = True
            return self.save_feedback()
        return False


class SystemImprover:
    """系统改进器"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        """
        初始化系统改进器
        
        Args:
            feedback_collector: 反馈收集器实例
        """
        self.feedback_collector = feedback_collector
        self.improvement_log = []
    
    def analyze_feedback(self) -> Dict[str, Any]:
        """
        分析反馈数据
        
        Returns:
            分析结果
        """
        stats = self.feedback_collector.get_feedback_stats()
        low_rated = self.feedback_collector.get_low_rated_feedback()
        
        # 分析常见问题
        common_issues = self._identify_common_issues(low_rated)
        
        return {
            "stats": stats,
            "low_rated_count": len(low_rated),
            "common_issues": common_issues,
            "recommendations": self._generate_recommendations(stats, common_issues)
        }
    
    def _identify_common_issues(self, low_rated_feedback: List[Dict[str, Any]]) -> List[str]:
        """
        识别常见问题
        
        Args:
            low_rated_feedback: 低评分反馈
            
        Returns:
            常见问题列表
        """
        issues = []
        
        for feedback in low_rated_feedback:
            comments = feedback.get("comments", "").lower()
            query = feedback.get("query", "").lower()
            
            # 基于评论内容识别问题类型
            if any(word in comments for word in ["不相关", "无关", "irrelevant"]):
                issues.append("检索结果不相关")
            elif any(word in comments for word in ["不准确", "错误", "inaccurate", "wrong"]):
                issues.append("答案不准确")
            elif any(word in comments for word in ["不完整", "缺少", "incomplete", "missing"]):
                issues.append("答案不完整")
            elif any(word in comments for word in ["难理解", "混乱", "confusing", "unclear"]):
                issues.append("答案难以理解")
            elif any(word in comments for word in ["太慢", "slow", "延迟"]):
                issues.append("响应速度慢")
            elif "查询扩展" in query or len(query.split()) < 2:
                issues.append("查询过于简短")
        
        # 去重并统计频率
        from collections import Counter
        issue_counts = Counter(issues)
        
        # 返回最常见的问题
        return [issue for issue, count in issue_counts.most_common(5)]
    
    def _generate_recommendations(self, stats: Dict[str, Any], 
                                 common_issues: List[str]) -> List[str]:
        """
        生成改进建议
        
        Args:
            stats: 统计信息
            common_issues: 常见问题
            
        Returns:
            改进建议列表
        """
        recommendations = []
        
        # 基于统计数据的建议
        if stats["total_feedback"] < 10:
            recommendations.append("收集更多用户反馈以获得更准确的分析")
        
        if stats["avg_rating"] < 3.0:
            recommendations.append("系统整体评分较低，需要全面改进")
        
        # 基于常见问题的建议
        for issue in common_issues:
            if issue == "检索结果不相关":
                recommendations.extend([
                    "优化检索算法，提高相关性",
                    "调整查询扩展策略",
                    "改进重排序机制"
                ])
            elif issue == "答案不准确":
                recommendations.extend([
                    "改进LLM提示词工程",
                    "增加上下文验证机制",
                    "提供更多相关上下文"
                ])
            elif issue == "答案不完整":
                recommendations.extend([
                    "增加检索结果数量",
                    "改进答案生成策略",
                    "添加追问机制"
                ])
            elif issue == "响应速度慢":
                recommendations.extend([
                    "优化向量搜索性能",
                    "缓存常用查询结果",
                    "异步处理复杂查询"
                ])
            elif issue == "查询过于简短":
                recommendations.extend([
                    "改进查询扩展算法",
                    "添加查询建议功能",
                    "引导用户提供更详细的查询"
                ])
        
        # 去重
        return list(set(recommendations))
    
    def generate_improvement_plan(self) -> Dict[str, Any]:
        """
        生成改进计划
        
        Returns:
            改进计划
        """
        analysis = self.analyze_feedback()
        
        plan = {
            "generated_at": datetime.now().isoformat(),
            "analysis": analysis,
            "action_items": []
        }
        
        # 基于建议创建具体的行动项
        for recommendation in analysis["recommendations"]:
            if "优化检索算法" in recommendation:
                plan["action_items"].append({
                    "action": "优化检索算法",
                    "priority": "高",
                    "estimated_effort": "中等",
                    "description": "调整混合搜索权重和查询扩展策略"
                })
            elif "改进重排序机制" in recommendation:
                plan["action_items"].append({
                    "action": "改进重排序",
                    "priority": "中",
                    "estimated_effort": "中等",
                    "description": "集成更先进的交叉编码器模型"
                })
            elif "改进LLM提示词" in recommendation:
                plan["action_items"].append({
                    "action": "优化提示词",
                    "priority": "高",
                    "estimated_effort": "低",
                    "description": "基于反馈优化LLM提示词模板"
                })
            elif "优化性能" in recommendation:
                plan["action_items"].append({
                    "action": "性能优化",
                    "priority": "中",
                    "estimated_effort": "高",
                    "description": "实现查询缓存和异步处理"
                })
        
        # 记录改进计划
        self.improvement_log.append(plan)
        
        return plan
    
    def save_improvement_plan(self, plan: Dict[str, Any], 
                             output_path: str = "./data/feedback/improvement_plan.json"):
        """
        保存改进计划
        
        Args:
            plan: 改进计划
            output_path: 输出文件路径
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            
            print(f"改进计划已保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"保存改进计划失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="反馈循环机制")
    parser.add_argument("--action", "-a", default="analyze", 
                       choices=["add", "analyze", "plan", "stats"],
                       help="执行的操作")
    
    # 添加反馈的参数
    parser.add_argument("--query", "-q", help="查询文本")
    parser.add_argument("--answer", help="系统生成的答案")
    parser.add_argument("--rating", "-r", type=int, help="评分 (1-5)")
    parser.add_argument("--comments", "-c", help="用户评论")
    
    # 文件路径参数
    parser.add_argument("--feedback-file", default="./data/feedback/feedback.json",
                       help="反馈文件路径")
    parser.add_argument("--plan-file", default="./data/feedback/improvement_plan.json",
                       help="改进计划文件路径")
    
    args = parser.parse_args()
    
    try:
        # 创建反馈收集器
        collector = FeedbackCollector(args.feedback_file)
        
        if args.action == "add":
            # 添加反馈
            if not all([args.query, args.answer, args.rating]):
                print("错误: 添加反馈需要提供 --query, --answer 和 --rating 参数")
                return
            
            success = collector.add_feedback(
                query=args.query,
                answer=args.answer,
                rating=args.rating,
                comments=args.comments or ""
            )
            
            if success:
                print("反馈已成功添加")
            else:
                print("添加反馈失败")
        
        elif args.action == "stats":
            # 显示统计信息
            stats = collector.get_feedback_stats()
            print("\n反馈统计信息:")
            print("-" * 40)
            print(f"总反馈数量: {stats['total_feedback']}")
            print(f"平均评分: {stats['avg_rating']}/5.0")
            print(f"最近30天反馈: {stats['recent_feedback']}")
            print(f"已改进的反馈: {stats.get('improved_count', 0)}")
            
            # 显示低评分反馈
            low_rated = collector.get_low_rated_feedback()
            if low_rated:
                print(f"\n低评分反馈 ({len(low_rated)} 个):")
                for i, feedback in enumerate(low_rated[:5], 1):
                    print(f"{i}. '{feedback['query']}' - 评分: {feedback['rating']}/5")
                    if feedback.get('comments'):
                        print(f"   评论: {feedback['comments']}")
        
        elif args.action == "analyze":
            # 分析反馈
            improver = SystemImprover(collector)
            analysis = improver.analyze_feedback()
            
            print("\n反馈分析结果:")
            print("-" * 60)
            
            stats = analysis["stats"]
            print(f"统计信息: 总反馈={stats['total_feedback']}, "
                  f"平均评分={stats['avg_rating']}, "
                  f"低评分={analysis['low_rated_count']}")
            
            if analysis["common_issues"]:
                print(f"\n常见问题:")
                for issue in analysis["common_issues"]:
                    print(f"  • {issue}")
            
            if analysis["recommendations"]:
                print(f"\n改进建议:")
                for rec in analysis["recommendations"]:
                    print(f"  • {rec}")
        
        elif args.action == "plan":
            # 生成改进计划
            improver = SystemImprover(collector)
            plan = improver.generate_improvement_plan()
            
            print("\n系统改进计划:")
            print("-" * 60)
            print(f"生成时间: {plan['generated_at']}")
            
            analysis = plan["analysis"]
            print(f"\n分析摘要:")
            print(f"  总反馈: {analysis['stats']['total_feedback']}")
            print(f"  平均评分: {analysis['stats']['avg_rating']}/5.0")
            print(f"  低评分反馈: {analysis['low_rated_count']}")
            
            if plan["action_items"]:
                print(f"\n行动项:")
                for i, item in enumerate(plan["action_items"], 1):
                    print(f"{i}. [{item['priority']}优先级] {item['action']}")
                    print(f"   描述: {item['description']}")
                    print(f"   预计工作量: {item['estimated_effort']}")
            
            # 保存改进计划
            improver.save_improvement_plan(plan, args.plan_file)
        
        else:
            print(f"未知操作: {args.action}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()