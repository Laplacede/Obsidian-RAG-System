#!/usr/bin/env python3
"""
语义分块器
针对技术文档优化的层次化分块策略。

职责说明：
- 输入：读取提取器输出的结构化笔记数据，通常来自 data/processed/obsidian_notes.json。
- 处理：按标题、段落、列表、引用和代码块等语义边界切分文本，并生成 small/medium/large 三种粒度的分块。
- 输出：将分块结果写入 data/chunks/small_chunks.json、data/chunks/medium_chunks.json、data/chunks/large_chunks.json。
- 统计：同时生成 data/chunks/chunking_statistics.json，记录各类分块数量、词数和平均块大小。
- 下游传递：分块结果会被检索器和向量库构建流程继续使用。
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class Chunk:
    """分块数据类"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    chunk_type: str  # small, medium, large
    word_count: int
    char_count: int


class SemanticChunker:
    """语义分块器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化分块器
        
        Args:
            config: 分块配置
        """
        self.config = config or {
            'chunk_sizes': {
                'small': 256,    # 精确检索
                'medium': 512,   # 上下文理解
                'large': 1024    # 关系提取
            },
            'overlap_ratios': {
                'small': 0.25,   # 25% 重叠
                'medium': 0.2,   # 20% 重叠
                'large': 0.15    # 15% 重叠
            },
            'semantic_boundaries': [
                r'^#+\s+',       # 标题
                r'\n\n+',        # 段落分隔
                r'```\w*',       # 代码块开始
                r'```\s*$',      # 代码块结束
                r'^-\s+',        # 列表项
                r'^\d+\.\s+',    # 数字列表
                r'^>\s+',        # 引用
            ]
        }
        
        # 编译正则表达式
        self.boundary_patterns = [
            re.compile(pattern, re.MULTILINE) 
            for pattern in self.config['semantic_boundaries']
        ]
    
    def split_by_semantic_boundaries(self, text: str) -> List[str]:
        """
        按语义边界分割文本
        
        Args:
            text: 原始文本
            
        Returns:
            语义段落列表
        """
        if not text.strip():
            return []
        
        # 首先按换行符分割
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # 空行表示段落结束
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            elif line.startswith('#') or line.startswith('##') or line.startswith('###'):
                # 标题行，开始新段落
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [line]
            elif line.startswith('```'):
                # 代码块开始/结束
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                paragraphs.append(line)  # 代码块标记单独作为段落
                current_paragraph = []
            elif line.startswith('- ') or line.startswith('* ') or line.startswith('+ '):
                # 列表项
                if current_paragraph and not current_paragraph[-1].startswith(('- ', '* ', '+ ')):
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)
            elif line.startswith('> '):
                # 引用
                if current_paragraph and not current_paragraph[-1].startswith('> '):
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)
            elif re.match(r'^\d+\.\s+', line):
                # 数字列表
                if current_paragraph and not re.match(r'^\d+\.\s+', current_paragraph[-1]):
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [line]
                else:
                    current_paragraph.append(line)
            else:
                # 普通文本行
                current_paragraph.append(line)
        
        # 添加最后一个段落
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # 过滤空段落
        paragraphs = [p for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def create_hierarchical_chunks(self, document: Dict[str, Any]) -> Dict[str, List[Chunk]]:
        """
        创建层次化分块
        
        Args:
            document: 文档数据，包含content和metadata
            
        Returns:
            按类型分组的块字典
        """
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        doc_id = document.get('id', 'unknown')
        
        # 按语义边界分割
        paragraphs = self.split_by_semantic_boundaries(content)
        
        # 创建不同粒度的分块
        chunks = {
            'small': self._create_fixed_size_chunks(
                paragraphs, metadata, doc_id, 'small'
            ),
            'medium': self._create_fixed_size_chunks(
                paragraphs, metadata, doc_id, 'medium'
            ),
            'large': self._create_fixed_size_chunks(
                paragraphs, metadata, doc_id, 'large'
            )
        }
        
        return chunks
    
    def _create_fixed_size_chunks(self, paragraphs: List[str], 
                                 metadata: Dict[str, Any], 
                                 doc_id: str, 
                                 chunk_type: str) -> List[Chunk]:
        """
        创建固定大小的分块
        
        Args:
            paragraphs: 语义段落列表
            metadata: 文档元数据
            doc_id: 文档ID
            chunk_type: 分块类型
            
        Returns:
            分块列表
        """
        chunk_size = self.config['chunk_sizes'][chunk_type]
        overlap_ratio = self.config['overlap_ratios'][chunk_type]
        overlap_words = int(chunk_size * overlap_ratio)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_words = para.split()
            para_word_count = len(para_words)
            
            # 如果单个段落就超过分块大小，需要分割段落
            if para_word_count > chunk_size:
                # 分割大段落
                sub_paragraphs = self._split_large_paragraph(para, chunk_size)
                for sub_para in sub_paragraphs:
                    sub_words = sub_para.split()
                    sub_word_count = len(sub_words)
                    
                    if current_word_count + sub_word_count > chunk_size and current_chunk:
                        # 保存当前分块
                        chunk_content = ' '.join(current_chunk)
                        chunk = self._create_chunk(
                            chunk_content, metadata, doc_id, 
                            chunk_type, chunk_index
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        
                        # 保留重叠部分开始新分块
                        if overlap_words > 0:
                            # 计算重叠部分
                            overlap_text = ' '.join(current_chunk[-overlap_words:])
                            current_chunk = [overlap_text]
                            current_word_count = len(overlap_text.split())
                        else:
                            current_chunk = []
                            current_word_count = 0
                    
                    current_chunk.append(sub_para)
                    current_word_count += sub_word_count
            else:
                # 正常段落处理
                if current_word_count + para_word_count > chunk_size and current_chunk:
                    # 保存当前分块
                    chunk_content = ' '.join(current_chunk)
                    chunk = self._create_chunk(
                        chunk_content, metadata, doc_id, 
                        chunk_type, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # 保留重叠部分开始新分块
                    if overlap_words > 0:
                        overlap_text = ' '.join(current_chunk[-overlap_words:])
                        current_chunk = [overlap_text]
                        current_word_count = len(overlap_text.split())
                    else:
                        current_chunk = []
                        current_word_count = 0
                
                current_chunk.append(para)
                current_word_count += para_word_count
        
        # 处理最后一个分块
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk = self._create_chunk(
                chunk_content, metadata, doc_id, 
                chunk_type, chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_large_paragraph(self, paragraph: str, max_words: int) -> List[str]:
        """
        分割大段落
        
        Args:
            paragraph: 大段落文本
            max_words: 最大单词数
            
        Returns:
            分割后的段落列表
        """
        words = paragraph.split()
        if len(words) <= max_words:
            return [paragraph]
        
        # 寻找自然分割点（句子结束）
        sentences = re.split(r'[.!?。！？]\s+', paragraph)
        if len(sentences) > 1:
            # 按句子分组
            result = []
            current_group = []
            current_word_count = 0
            
            for sentence in sentences:
                sentence_words = sentence.split()
                sentence_word_count = len(sentence_words)
                
                if current_word_count + sentence_word_count > max_words and current_group:
                    result.append(' '.join(current_group))
                    current_group = [sentence]
                    current_word_count = sentence_word_count
                else:
                    current_group.append(sentence)
                    current_word_count += sentence_word_count
            
            if current_group:
                result.append(' '.join(current_group))
            
            return result
        
        # 如果没有句子分隔符，按固定大小分割
        result = []
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i + max_words]
            result.append(' '.join(chunk_words))
        
        return result
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any], 
                     doc_id: str, chunk_type: str, chunk_index: int) -> Chunk:
        """
        创建分块对象
        
        Args:
            content: 分块内容
            metadata: 文档元数据
            doc_id: 文档ID
            chunk_type: 分块类型
            chunk_index: 分块索引
            
        Returns:
            Chunk对象
        """
        # 创建分块特定的元数据
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_type': chunk_type,
            'chunk_index': chunk_index,
            'source_document': doc_id,
            'original_filename': metadata.get('filename', 'unknown'),
            'original_directory': metadata.get('directory', 'unknown'),
            'original_tags': metadata.get('tags', []),
            'original_wikilinks': metadata.get('wikilinks', [])
        })
        
        # 计算统计信息
        words = content.split()
        word_count = len(words)
        char_count = len(content)
        
        # 生成分块ID
        chunk_id = f"{doc_id}_{chunk_type}_{chunk_index}"
        
        return Chunk(
            content=content,
            metadata=chunk_metadata,
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            word_count=word_count,
            char_count=char_count
        )
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        处理多个文档
        
        Args:
            documents: 文档列表
            
        Returns:
            处理后的分块数据
        """
        all_chunks = {
            'small': [],
            'medium': [],
            'large': []
        }
        
        stats = {
            'total_documents': len(documents),
            'total_chunks': 0,
            'chunks_by_type': {'small': 0, 'medium': 0, 'large': 0},
            'words_by_type': {'small': 0, 'medium': 0, 'large': 0},
            'avg_chunk_size': {'small': 0, 'medium': 0, 'large': 0}
        }
        
        for doc in documents:
            chunks = self.create_hierarchical_chunks(doc)
            
            for chunk_type in ['small', 'medium', 'large']:
                for chunk in chunks[chunk_type]:
                    # 转换为字典格式
                    chunk_dict = {
                        'id': chunk.chunk_id,
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'chunk_type': chunk.chunk_type,
                        'word_count': chunk.word_count,
                        'char_count': chunk.char_count
                    }
                    all_chunks[chunk_type].append(chunk_dict)
                    
                    # 更新统计
                    stats['chunks_by_type'][chunk_type] += 1
                    stats['words_by_type'][chunk_type] += chunk.word_count
        
        # 计算总数和平均值
        stats['total_chunks'] = sum(stats['chunks_by_type'].values())
        
        for chunk_type in ['small', 'medium', 'large']:
            count = stats['chunks_by_type'][chunk_type]
            if count > 0:
                stats['avg_chunk_size'][chunk_type] = stats['words_by_type'][chunk_type] / count
            else:
                stats['avg_chunk_size'][chunk_type] = 0
        
        return {
            'chunks': all_chunks,
            'statistics': stats
        }
    
    def save_chunks(self, processed_data: Dict, output_dir: str):
        """
        保存分块数据
        
        Args:
            processed_data: 处理后的数据
            output_dir: 输出目录
        """
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存分块数据
        for chunk_type in ['small', 'medium', 'large']:
            chunks = processed_data['chunks'][chunk_type]
            if chunks:
                file_path = output_path / f"{chunk_type}_chunks.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                print(f"已保存 {len(chunks)} 个 {chunk_type} 分块到 {file_path}")
        
        # 保存统计信息
        stats_file = output_path / "chunking_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data['statistics'], f, ensure_ascii=False, indent=2)
        print(f"统计信息已保存到 {stats_file}")
        
        return {
            'small_chunks_file': str(output_path / "small_chunks.json"),
            'medium_chunks_file': str(output_path / "medium_chunks.json"),
            'large_chunks_file': str(output_path / "large_chunks.json"),
            'statistics_file': str(stats_file)
        }


def main():
    """主函数"""
    import sys
    from pathlib import Path
    
    # 检查参数
    if len(sys.argv) < 2:
        print("用法: python semantic_chunker.py <输入JSON文件> [输出目录]")
        print("示例: python semantic_chunker.py data/processed/obsidian_notes.json data/chunks")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/chunks"
    
    try:
        # 加载文档数据
        print(f"正在加载文档数据: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"加载了 {len(documents)} 个文档")
        
        # 创建分块器
        chunker = SemanticChunker()
        
        # 处理文档
        print("正在处理文档...")
        processed_data = chunker.process_documents(documents)
        
        # 显示统计信息
        stats = processed_data['statistics']
        print(f"\n=== 分块统计 ===")
        print(f"总文档数: {stats['total_documents']}")
        print(f"总分块数: {stats['total_chunks']}")
        
        for chunk_type in ['small', 'medium', 'large']:
            count = stats['chunks_by_type'][chunk_type]
            avg_size = stats['avg_chunk_size'][chunk_type]
            print(f"{chunk_type} 分块: {count} 个，平均 {avg_size:.1f} 词")
        
        # 保存结果
        print(f"\n正在保存分块数据到: {output_dir}")
        saved_files = chunker.save_chunks(processed_data, output_dir)
        
        print("\n=== 分块完成 ===")
        for key, path in saved_files.items():
            print(f"{key}: {path}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()