#!/usr/bin/env python3
"""
Obsidian RAG系统 - 增强版交互式CLI界面
支持从配置文件加载模型设置，支持本地LM Studio模型

职责说明：
- 输入：读取命令行参数和配置文件，主要依赖 config/model_config.yaml。
- 处理：组织系统初始化、检索、重排序和生成的交互流程，并维护命令历史。
- 输出：向终端展示配置、检索结果、问答结果和系统状态。
- 依赖文件：config/model_config.yaml、用户历史文件 ~/.obsidian_rag_history，以及 src/ 下的检索和生成模块。
- 下游传递：CLI 会把用户问题交给检索器，再把检索结果交给生成器，形成完整 RAG 问答链路。
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
import readline  # 用于命令行历史记录


class ObsidianRAGCLI:
    """Obsidian RAG系统CLI界面"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化CLI界面
        
        Args:
            config_path: 配置文件路径
        """
        self.retriever = None
        self.generator = None
        self.rag_system = None
        self.config = {
            "chunk_type": "small",
            "use_hybrid": True,
            "use_reranking": True,
            "context_count": 5,
            "model_type": "mock",  # mock, local, openai
            "rag_mode": "flexible"  # strict, flexible
        }
        
        # 从配置文件加载模型设置
        self.model_config = self.load_model_config(config_path)
        
        # 命令历史记录
        self.history = []
        self.history_file = os.path.expanduser("~/.obsidian_rag_history")
        
        # 加载历史记录
        self.load_history()
    
    def load_model_config(self, config_path: Optional[str]) -> dict:
        """
        从配置文件加载模型配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            模型配置字典
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"警告: 配置文件不存在: {config_path}")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 提取RAG配置
            rag_config = config.get("rag", {})
            self.config["rag_mode"] = rag_config.get("mode", "flexible")
            
            # 提取模型配置
            model_config = {
                "local": config.get("llm", {}).get("local", {}),
                "cloud": config.get("llm", {}).get("cloud", {}),
                "rag": rag_config
            }
            
            # 如果配置中启用了本地模型，自动设置model_type为local
            if model_config["local"].get("enabled", False):
                self.config["model_type"] = "local"
                print(f"从配置文件加载本地模型设置: {model_config['local'].get('model', 'unknown')}")
            
            # 打印RAG模式信息
            print(f"RAG模式: {self.config['rag_mode']}")
            
            return model_config
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def load_history(self):
        """加载命令历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = [line.strip() for line in f.readlines() if line.strip()]
        except:
            self.history = []
    
    def save_history(self, query: str):
        """保存命令历史记录"""
        self.history.append(query)
        try:
            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(query + '\n')
        except:
            pass
    
    def initialize_system(self):
        """初始化RAG系统"""
        print("正在初始化Obsidian RAG系统...")
        
        try:
            # 添加当前目录到Python路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            
            # 导入必要的模块
            from retriever import HybridRetriever
            from rag_generator import create_generator, RAGSystem
            
            # 创建检索器
            print("正在加载向量数据库和分块数据...")
            self.retriever = HybridRetriever(
                vector_store_dir="./data/vector_store",
                chunk_data_dir="./data/chunks"
            )
            
            # 加载数据
            self.retriever.load_vector_store(self.config["chunk_type"])
            self.retriever.load_chunk_data(self.config["chunk_type"])
            self.retriever.build_keyword_index(self.config["chunk_type"])
            
            # 根据model_type创建生成器
            print(f"正在初始化LLM生成器 ({self.config['model_type']}模式, RAG模式: {self.config['rag_mode']})...")
            
            if self.config["model_type"] == "local":
                # 使用配置文件中的本地模型设置
                local_config = self.model_config.get("local", {})
                base_url = local_config.get("base_url")
                model_name = local_config.get("model", "local-model")

                if not base_url:
                    raise ValueError("本地模型配置缺少 base_url，请在 config/model_config.yaml 中设置")
                
                print(f"使用本地LM Studio: {base_url}, 模型: {model_name}")
                self.generator = create_generator(
                    "local",
                    base_url=base_url,
                    model_name=model_name,
                    rag_mode=self.config["rag_mode"]
                )
            elif self.config["model_type"] == "openai":
                # 使用OpenAI API配置
                openai_config = self.model_config.get("openai", {})
                api_key = openai_config.get("api_key")
                if not api_key:
                    print("警告: OpenAI API密钥未配置，使用模拟模式")
                    self.generator = create_generator("mock", rag_mode=self.config["rag_mode"])
                else:
                    model_name = openai_config.get("model", "gpt-3.5-turbo")
                    self.generator = create_generator(
                        "openai",
                        api_key=api_key,
                        model_name=model_name,
                        rag_mode=self.config["rag_mode"]
                    )
            else:  # mock模式
                self.generator = create_generator("mock", rag_mode=self.config["rag_mode"])
            
            # 创建RAG系统
            self.rag_system = RAGSystem(self.retriever, self.generator)
            
            print("系统初始化完成！")
            print(f"配置: 分块类型={self.config['chunk_type']}, 混合搜索={'启用' if self.config['use_hybrid'] else '禁用'}, "
                  f"重排序={'启用' if self.config['use_reranking'] else '禁用'}, 上下文数量={self.config['context_count']}, "
                  f"模型类型={self.config['model_type']}, RAG模式={self.config['rag_mode']}")
            print("输入 'help' 查看可用命令")
            print()
            
            return True
            
        except Exception as e:
            print(f"系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
可用命令:
  query <问题>           - 查询问题
  search <关键词>        - 仅搜索（不生成答案）
  config                 - 显示当前配置
  config set <选项> <值> - 修改配置
  history                - 显示查询历史
  clear                  - 清屏
  help                   - 显示此帮助信息
  exit/quit              - 退出程序

配置选项:
  chunk_type     - 分块类型: small, medium, large
  use_hybrid     - 混合搜索: true, false
  use_reranking  - 重排序: true, false
  context_count  - 上下文数量: 1-10
  model_type     - 模型类型: mock, local, openai

示例:
  query C++类与对象是什么？
  search 计算机组成原理
  config set context_count 3
  config set use_reranking false
  config set model_type local
"""
        print(help_text)
    
    def show_config(self):
        """显示当前配置"""
        print("\n当前配置:")
        print("-" * 40)
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        # 显示模型配置详情
        if self.config["model_type"] == "local":
            local_config = self.model_config.get("local", {})
            if local_config:
                print(f"\n本地模型配置:")
                print(f"  base_url: {local_config.get('base_url', '未设置')}")
                print(f"  model: {local_config.get('model', '未设置')}")
                print(f"  temperature: {local_config.get('temperature', '未设置')}")
                print(f"  timeout: {local_config.get('timeout', '未设置')}")
                print(f"  max_tokens: {local_config.get('max_tokens', '未设置')}")
        print()
    
    def update_config(self, option: str, value: str):
        """更新配置"""
        option = option.lower()
        
        if option not in self.config:
            print(f"错误: 未知配置选项 '{option}'")
            return False
        
        # 类型转换
        if option in ["use_hybrid", "use_reranking"]:
            if value.lower() in ["true", "yes", "1", "on"]:
                new_value = True
            elif value.lower() in ["false", "no", "0", "off"]:
                new_value = False
            else:
                print(f"错误: 值 '{value}' 无效，应为 true/false")
                return False
        elif option == "context_count":
            try:
                new_value = int(value)
                if not 1 <= new_value <= 10:
                    print(f"错误: 上下文数量应在 1-10 范围内")
                    return False
            except ValueError:
                print(f"错误: 值 '{value}' 无效，应为整数")
                return False
        elif option == "chunk_type":
            if value not in ["small", "medium", "large"]:
                print(f"错误: 分块类型应为 small/medium/large")
                return False
            new_value = value
        elif option == "model_type":
            if value not in ["mock", "local", "openai"]:
                print(f"错误: 模型类型应为 mock/local/openai")
                return False
            new_value = value
        else:
            new_value = value
        
        # 更新配置
        old_value = self.config[option]
        self.config[option] = new_value
        print(f"配置已更新: {option} = {old_value} -> {new_value}")
        
        # 如果更改了分块类型，需要重新加载数据
        if option == "chunk_type" and self.retriever:
            print(f"正在重新加载 {new_value} 分块数据...")
            try:
                self.retriever.load_vector_store(new_value)
                self.retriever.load_chunk_data(new_value)
                self.retriever.build_keyword_index(new_value)
                print("数据重新加载完成")
            except Exception as e:
                print(f"重新加载数据失败: {e}")
                self.config[option] = old_value  # 恢复原值
        
        # 如果更改了模型类型，需要重新初始化系统
        elif option == "model_type":
            print("警告: 更改模型类型需要重新启动系统才能生效")
            print("请退出后重新启动程序")
        
        return True
    
    def execute_query(self, query: str):
        """执行查询"""
        if not self.rag_system:
            print("错误: 系统未初始化")
            return
        
        print(f"\n查询: {query}")
        print("-" * 60)
        
        try:
            # 执行RAG查询
            result = self.rag_system.query(
                query=query,
                n_context=self.config["context_count"],
                use_hybrid=self.config["use_hybrid"],
                use_reranking=self.config["use_reranking"]
            )
            
            if "error" in result:
                print(f"错误: {result['error']}")
                return
            
            # 显示答案
            confidence = result['confidence']
            confidence_label = self._get_confidence_label(confidence)
            rag_mode_label = "严格模式（仅知识库）" if result.get('rag_mode') == 'strict' else "灵活模式（知识库优先+补充）"
            
            print(f"\n答案 (模型: {result['model']}, RAG模式: {rag_mode_label}, 置信度: {confidence:.2f} {confidence_label}):")
            print("-" * 40)
            print(result["answer"])
            
            # 显示引用
            if result["citations"]:
                print(f"\n引用 ({len(result['citations'])} 个):")
                print("-" * 40)
                for i, citation in enumerate(result["citations"], 1):
                    print(f"{i}. {citation['title']} (分数: {citation['score']:.3f})")
                    if i <= 2:  # 只显示前2个的预览
                        print(f"   预览: {citation['text_preview']}")
            
            # 显示上下文统计
            print(f"\n检索统计: {result['context_count']} 个文档")
            
            # 保存到历史记录
            self.save_history(query)
            
        except Exception as e:
            print(f"查询失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_confidence_label(self, confidence: float) -> str:
        """根据置信度返回标签"""
        if confidence >= 0.9:
            return "🟢 极高"
        elif confidence >= 0.75:
            return "🟢 高"
        elif confidence >= 0.6:
            return "🟡 中等"
        elif confidence >= 0.4:
            return "🟠 低"
        else:
            return "🔴 极低"
    
    def execute_search(self, query: str):
        """仅执行搜索（不生成答案）"""
        if not self.retriever:
            print("错误: 检索器未初始化")
            return
        
        print(f"\n搜索: {query}")
        print("-" * 60)
        
        try:
            # 执行检索
            results = self.retriever.retrieve(
                query=query,
                n_results=self.config["context_count"],
                use_hybrid=self.config["use_hybrid"],
                use_expansion=True,
                use_reranking=self.config["use_reranking"]
            )
            
            print(f"找到 {len(results)} 个相关结果:\n")
            
            # 格式化并显示结果
            formatted = self.retriever.format_results(results)
            print(formatted)
            
            # 保存到历史记录
            self.save_history(f"search: {query}")
            
        except Exception as e:
            print(f"搜索失败: {e}")
            import traceback
            traceback.print_exc()
    
    def show_history(self):
        """显示查询历史"""
        if not self.history:
            print("暂无查询历史")
            return
        
        print(f"\n查询历史 ({len(self.history)} 条):")
        print("-" * 60)
        for i, query in enumerate(self.history[-10:], 1):  # 显示最近10条
            print(f"{i}. {query}")
        print()
    
    def clear_screen(self):
        """清屏"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def run(self):
        """运行CLI界面"""
        # 初始化系统
        if not self.initialize_system():
            return
        
        # 主循环
        while True:
            try:
                # 获取用户输入
                user_input = input("\nobsidian-rag> ").strip()
                
                if not user_input:
                    continue
                
                # 解析命令
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # 处理命令
                if command in ["exit", "quit", "q"]:
                    print("感谢使用Obsidian RAG系统，再见！")
                    break
                
                elif command in ["help", "?"]:
                    self.show_help()
                
                elif command == "clear":
                    self.clear_screen()
                
                elif command == "config":
                    if args.startswith("set "):
                        # config set <option> <value>
                        set_parts = args[4:].split(maxsplit=1)
                        if len(set_parts) == 2:
                            option, value = set_parts
                            self.update_config(option, value)
                        else:
                            print("用法: config set <选项> <值>")
                    else:
                        self.show_config()
                
                elif command == "history":
                    self.show_history()
                
                elif command == "query":
                    if not args:
                        print("用法: query <问题>")
                    else:
                        self.execute_query(args)
                
                elif command == "search":
                    if not args:
                        print("用法: search <关键词>")
                    else:
                        self.execute_search(args)
                
                else:
                    # 如果没有匹配的命令，尝试作为查询处理
                    self.execute_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n中断操作，输入 'exit' 退出程序")
                continue
            except EOFError:
                print("\n\n感谢使用Obsidian RAG系统，再见！")
                break
            except Exception as e:
                print(f"错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Obsidian RAG系统增强版CLI界面")
    parser.add_argument("--config", "-c", help="配置文件路径", 
                       default="config/model_config.yaml")
    parser.add_argument("--model-type", choices=["mock", "local", "openai"], 
                       help="模型类型，覆盖配置文件设置")
    parser.add_argument("--rag-mode", choices=["strict", "flexible"],
                       help="RAG模式：strict（仅知识库）或 flexible（知识库优先+补充），覆盖配置文件设置")
    
    args = parser.parse_args()
    
    # 创建并运行CLI
    cli = ObsidianRAGCLI(args.config)
    
    # 如果命令行指定了model-type，覆盖配置
    if args.model_type:
        cli.config["model_type"] = args.model_type
    
    # 如果命令行指定了rag_mode，覆盖配置
    if args.rag_mode:
        cli.config["rag_mode"] = args.rag_mode
    
    cli.run()


if __name__ == "__main__":
    main()