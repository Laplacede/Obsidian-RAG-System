#!/usr/bin/env python3
"""
Obsidian RAG System - 主程序入口

这是一个基于个人Obsidian笔记的增强检索系统，支持本地模型和云端API。

主要功能：
1. 从Obsidian笔记库提取知识
2. 生成向量嵌入并建立索引
3. 支持混合检索（语义+关键词）
4. 使用本地或云端LLM生成回答
5. 交互式命令行界面

使用方法（使用 uv 管理环境）：
1. 安装依赖：uv sync
2. 初始化系统：uv run python main.py init
3. 启动CLI：uv run python main.py cli
4. 查看配置：uv run python main.py config
5. 查看帮助：uv run python main.py --help

职责说明：
- 输入：接收命令行参数，作为整个系统的入口。
- 处理：把 init、cli、config、version 等子命令转发给 src/cli.py 中的实际实现。
- 输出：执行初始化、启动交互式问答、显示配置或版本信息。
- 依赖文件：config/model_config.yaml 作为主要配置来源；src/cli.py 提供交互式 CLI 逻辑。
- 下游传递：命令行参数会被转成 CLI 需要的配置，再继续驱动提取、分块、检索和生成流程。
"""

import sys
import click
from pathlib import Path

@click.group()
def main():
    """Obsidian RAG System - 基于个人知识库的增强检索系统"""
    pass

@main.command()
@click.option('--config', '-c', default='config/model_config.yaml', 
              help='配置文件路径')
def init(config):
    """初始化系统配置"""
    from src.cli import ObsidianRAGCLI
    cli = ObsidianRAGCLI(config)
    if cli.initialize_system():
        print("系统初始化成功！")
    else:
        print("系统初始化失败。")

@main.command()
@click.option('--model-type', type=click.Choice(['mock', 'local', 'openai']), 
              default='local', help='模型类型: mock(模拟), local(本地), openai(API)')
@click.option('--config', '-c', default='config/model_config.yaml', 
              help='配置文件路径')
def cli(model_type, config):
    """启动交互式命令行界面"""
    # 通过sys.argv模拟命令行参数
    sys.argv = ['src/cli.py', '--model-type', model_type, '--config', config]
    from src.cli import main as cli_main
    cli_main()

@main.command()
@click.option('--config', '-c', default='config/model_config.yaml', 
              help='配置文件路径')
def config(config):
    """显示系统配置"""
    from src.cli import ObsidianRAGCLI
    cli = ObsidianRAGCLI(config)
    cli.show_config()

@main.command()
def version():
    """显示版本信息"""
    print("Obsidian RAG System v2.0")
    print("增强版CLI系统 - 支持本地模型")
    print("构建时间: 2024-05-11")
    print("使用 uv 管理环境")

if __name__ == '__main__':
    main()