#!/usr/bin/env python3
"""
Obsidian RAG系统 - 主程序入口
基于个人Obsidian笔记的增强检索系统
"""

import os
import sys
import yaml
import click
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.extractors.obsidian_extractor import ObsidianExtractor


class ObsidianRAGSystem:
    """Obsidian RAG系统主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化RAG系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.extractor = None
        self.vector_store = None
        self.llm = None
        
    def load_config(self, config_path: Optional[str]) -> dict:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        if config_path is None:
            config_path = project_root / "config" / "model_config.yaml"
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def setup_environment(self):
        """设置环境"""
        print("正在设置RAG系统环境...")
        
        # 创建必要的目录
        data_dir = project_root / "data"
        for subdir in ["raw", "processed", "embeddings", "logs"]:
            (data_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print("环境设置完成")
    
    def extract_data(self):
        """提取Obsidian数据"""
        print("正在提取Obsidian笔记数据...")
        
        vault_path = self.config['data_sources']['obsidian_vault_path']
        self.extractor = ObsidianExtractor(vault_path)
        
        # 提取并保存数据
        output_dir = project_root / "data" / "processed"
        saved_files = self.extractor.save_extracted_data(str(output_dir))
        
        print("数据提取完成")
        return saved_files
    
    def analyze_data(self):
        """分析数据"""
        if self.extractor is None:
            print("请先提取数据")
            return
        
        print("正在分析数据...")
        structure = self.extractor.analyze_vault_structure()
        
        print(f"\n=== 数据分析报告 ===")
        print(f"📊 总笔记数: {structure['total_notes']}")
        
        print(f"\n📁 目录分布:")
        for dir_name, count in sorted(structure['directories'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dir_name}: {count} 个笔记")
        
        print(f"\n🏷️  热门标签 (前10):")
        for tag, count in list(structure['tag_distribution'].items())[:10]:
            print(f"  #{tag}: {count} 次")
        
        print(f"\n📝 最大的笔记 (按字数，前5):")
        for i, note in enumerate(structure['largest_notes'][:5], 1):
            print(f"  {i}. {note['filename']}: {note['metadata']['word_count']} 字")
        
        return structure
    
    def build_index(self):
        """构建向量索引"""
        print("正在构建向量索引...")
        # TODO: 实现向量索引构建
        print("向量索引构建功能待实现")
    
    def query(self, question: str):
        """查询系统"""
        print(f"查询: {question}")
        # TODO: 实现查询功能
        print("查询功能待实现")
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🔍 Obsidian RAG系统 - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n💭 请输入问题: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if not question:
                    continue
                
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"错误: {e}")


@click.group()
def cli():
    """Obsidian RAG系统 - 基于个人笔记的增强检索系统"""
    pass


@cli.command()
@click.option('--config', '-c', default=None, help='配置文件路径')
def setup(config):
    """设置系统环境"""
    try:
        rag = ObsidianRAGSystem(config)
        rag.setup_environment()
        click.echo("✅ 环境设置完成")
    except Exception as e:
        click.echo(f"❌ 设置失败: {e}", err=True)


@cli.command()
@click.option('--config', '-c', default=None, help='配置文件路径')
def extract(config):
    """提取Obsidian数据"""
    try:
        rag = ObsidianRAGSystem(config)
        rag.extract_data()
        click.echo("✅ 数据提取完成")
    except Exception as e:
        click.echo(f"❌ 数据提取失败: {e}", err=True)


@cli.command()
@click.option('--config', '-c', default=None, help='配置文件路径')
def analyze(config):
    """分析数据"""
    try:
        rag = ObsidianRAGSystem(config)
        rag.analyze_data()
    except Exception as e:
        click.echo(f"❌ 数据分析失败: {e}", err=True)


@cli.command()
@click.option('--config', '-c', default=None, help='配置文件路径')
def index(config):
    """构建向量索引"""
    try:
        rag = ObsidianRAGSystem(config)
        rag.build_index()
        click.echo("✅ 索引构建完成")
    except Exception as e:
        click.echo(f"❌ 索引构建失败: {e}", err=True)


@cli.command()
@click.argument('question')
@click.option('--config', '-c', default=None, help='配置文件路径')
def ask(config, question):
    """查询系统"""
    try:
        rag = ObsidianRAGSystem(config)
        rag.query(question)
    except Exception as e:
        click.echo(f"❌ 查询失败: {e}", err=True)


@cli.command()
@click.option('--config', '-c', default=None, help='配置文件路径')
def chat(config):
    """进入交互模式"""
    try:
        rag = ObsidianRAGSystem(config)
        rag.interactive_mode()
    except Exception as e:
        click.echo(f"❌ 交互模式失败: {e}", err=True)


@cli.command()
def status():
    """显示系统状态"""
    try:
        click.echo("📊 Obsidian RAG系统状态")
        click.echo("-" * 40)
        
        # 检查目录
        data_dir = project_root / "data"
        if data_dir.exists():
            processed_dir = data_dir / "processed"
            if processed_dir.exists():
                notes_file = processed_dir / "obsidian_notes.json"
                if notes_file.exists():
                    import json
                    with open(notes_file, 'r', encoding='utf-8') as f:
                        notes = json.load(f)
                    click.echo(f"✅ 已处理笔记: {len(notes)} 个")
                else:
                    click.echo("❌ 未找到处理后的笔记数据")
            else:
                click.echo("❌ 未找到处理数据目录")
        else:
            click.echo("❌ 未找到数据目录")
            
        click.echo(f"\n📁 项目根目录: {project_root}")
        click.echo(f"📁 数据目录: {data_dir}")
        
    except Exception as e:
        click.echo(f"❌ 状态检查失败: {e}", err=True)


if __name__ == "__main__":
    cli()