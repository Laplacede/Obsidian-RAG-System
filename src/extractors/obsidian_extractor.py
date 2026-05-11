#!/usr/bin/env python3
"""
Obsidian笔记数据提取器
从Obsidian仓库提取笔记内容、元数据和链接关系
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

class ObsidianExtractor:
    """Obsidian笔记提取器"""
    
    def __init__(self, vault_path: str):
        """
        初始化提取器
        
        Args:
            vault_path: Obsidian仓库路径
        """
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Obsidian仓库不存在: {vault_path}")
        
        # 正则表达式模式
        self.yaml_pattern = re.compile(r'^---\n(.*?)\n---', re.DOTALL)
        self.wikilink_pattern = re.compile(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]')
        self.tag_pattern = re.compile(r'#([\w\-]+)')
        
    def extract_all_notes(self) -> List[Dict]:
        """
        提取所有笔记
        
        Returns:
            笔记列表，每个笔记包含内容和元数据
        """
        notes = []
        
        # 查找所有Markdown文件
        md_files = list(self.vault_path.rglob("*.md"))
        print(f"找到 {len(md_files)} 个Markdown文件")
        
        for md_file in md_files:
            try:
                note = self.extract_note(md_file)
                if note:
                    notes.append(note)
            except Exception as e:
                print(f"处理文件失败 {md_file}: {e}")
        
        return notes
    
    def extract_note(self, file_path: Path) -> Optional[Dict]:
        """
        提取单个笔记
        
        Args:
            file_path: 笔记文件路径
            
        Returns:
            笔记字典或None（如果文件为空）
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return None
            
            # 提取YAML frontmatter
            metadata = self.extract_yaml_frontmatter(content)
            
            # 提取wikilinks
            wikilinks = self.extract_wikilinks(content)
            
            # 提取标签
            tags = self.extract_tags(content, metadata)
            
            # 清理内容（移除YAML和wikilinks格式）
            cleaned_content = self.clean_content(content)
            
            # 构建笔记数据
            note_data = {
                'id': str(file_path.relative_to(self.vault_path)),
                'filepath': str(file_path.relative_to(self.vault_path)),
                'filename': file_path.stem,
                'content': cleaned_content,
                'metadata': {
                    'source': 'obsidian',
                    'yaml_frontmatter': metadata,
                    'tags': tags,
                    'wikilinks': wikilinks,
                    'word_count': len(cleaned_content.split()),
                    'character_count': len(cleaned_content),
                    'file_size': file_path.stat().st_size
                }
            }
            
            # 添加目录信息
            note_data['metadata']['directory'] = str(file_path.parent.relative_to(self.vault_path))
            
            return note_data
            
        except Exception as e:
            print(f"提取笔记失败 {file_path}: {e}")
            return None
    
    def extract_yaml_frontmatter(self, content: str) -> Dict:
        """
        提取YAML frontmatter
        
        Args:
            content: 笔记内容
            
        Returns:
            YAML frontmatter字典
        """
        metadata = {}
        
        match = self.yaml_pattern.match(content)
        if match:
            yaml_content = match.group(1)
            try:
                metadata = yaml.safe_load(yaml_content) or {}
                # 将非字符串类型转换为字符串，避免JSON序列化问题
                for key, value in metadata.items():
                    if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        metadata[key] = str(value)
            except yaml.YAMLError:
                # 如果YAML解析失败，尝试简单解析
                lines = yaml_content.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
        
        return metadata
    
    def extract_wikilinks(self, content: str) -> List[Dict]:
        """
        提取wikilinks [[link]] 或 [[link|display]]
        
        Args:
            content: 笔记内容
            
        Returns:
            wikilinks列表，每个包含link和display
        """
        wikilinks = []
        
        matches = self.wikilink_pattern.findall(content)
        for match in matches:
            link = match[0].strip()
            display = match[1].strip() if match[1] else link
            wikilinks.append({
                'link': link,
                'display': display
            })
        
        return wikilinks
    
    def extract_tags(self, content: str, metadata: Dict) -> List[str]:
        """
        提取标签
        
        Args:
            content: 笔记内容
            metadata: YAML frontmatter
            
        Returns:
            标签列表
        """
        tags = set()
        
        # 从YAML frontmatter提取标签
        if 'tags' in metadata:
            tags_list = metadata['tags']
            if isinstance(tags_list, list):
                tags.update(tags_list)
            elif isinstance(tags_list, str):
                tags.update([tag.strip() for tag in tags_list.split(',')])
        
        # 从内容中提取#标签
        content_tags = self.tag_pattern.findall(content)
        tags.update(content_tags)
        
        # 从文件名推断标签（如果包含C++等关键词）
        # 这部分在外部处理
        
        return list(tags)
    
    def clean_content(self, content: str) -> str:
        """
        清理笔记内容
        
        Args:
            content: 原始笔记内容
            
        Returns:
            清理后的内容
        """
        # 移除YAML frontmatter
        cleaned = self.yaml_pattern.sub('', content)
        
        # 移除wikilinks格式，保留显示文本
        def replace_wikilink(match):
            groups = match.groups()
            if groups[1]:  # 有显示文本
                return groups[1]
            else:  # 只有链接
                return groups[0]
        
        cleaned = self.wikilink_pattern.sub(replace_wikilink, cleaned)
        
        # 移除标签符号，保留标签文本
        cleaned = self.tag_pattern.sub(r'\1', cleaned)
        
        # 清理多余的空行
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)
        
        return cleaned
    
    def analyze_vault_structure(self) -> Dict:
        """
        分析仓库结构
        
        Returns:
            仓库结构信息
        """
        structure = {
            'total_notes': 0,
            'directories': {},
            'note_types': {},
            'largest_notes': [],
            'tag_distribution': {}
        }
        
        notes = self.extract_all_notes()
        structure['total_notes'] = len(notes)
        
        # 分析目录分布
        for note in notes:
            directory = note['metadata']['directory']
            if directory == '.':
                directory = '根目录'
            
            if directory not in structure['directories']:
                structure['directories'][directory] = 0
            structure['directories'][directory] += 1
        
        # 分析标签分布
        tag_counts = {}
        for note in notes:
            for tag in note['metadata']['tags']:
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                tag_counts[tag] += 1
        
        # 按频率排序
        structure['tag_distribution'] = dict(
            sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        )
        
        # 找出最大的笔记
        structure['largest_notes'] = sorted(
            notes, 
            key=lambda x: x['metadata']['word_count'], 
            reverse=True
        )[:10]
        
        return structure
    
    def save_extracted_data(self, output_dir: str):
        """
        保存提取的数据
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 提取所有笔记
        notes = self.extract_all_notes()
        
        # 保存为JSON
        notes_file = output_path / "obsidian_notes.json"
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 {len(notes)} 个笔记到 {notes_file}")
        
        # 保存结构分析
        structure = self.analyze_vault_structure()
        structure_file = output_path / "vault_structure.json"
        with open(structure_file, 'w', encoding='utf-8') as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)
        
        print(f"仓库结构分析已保存到 {structure_file}")
        
        # 保存统计信息
        stats = {
            'total_notes': len(notes),
            'total_words': sum(n['metadata']['word_count'] for n in notes),
            'total_characters': sum(n['metadata']['character_count'] for n in notes),
            'directories': len(structure['directories']),
            'unique_tags': len(structure['tag_distribution'])
        }
        
        stats_file = output_path / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存到 {stats_file}")
        
        return {
            'notes_file': str(notes_file),
            'structure_file': str(structure_file),
            'stats_file': str(stats_file)
        }


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        vault_path = sys.argv[1]
    else:
        # 使用默认路径
        vault_path = "/Users/fengzhe/Library/Mobile Documents/iCloud~md~obsidian/Documents/知识仓库"
    
    try:
        extractor = ObsidianExtractor(vault_path)
        
        # 分析仓库结构
        print("正在分析Obsidian仓库...")
        structure = extractor.analyze_vault_structure()
        
        print(f"\n=== 仓库分析结果 ===")
        print(f"总笔记数: {structure['total_notes']}")
        print(f"\n目录分布:")
        for dir_name, count in sorted(structure['directories'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {dir_name}: {count} 个笔记")
        
        print(f"\n热门标签 (前10):")
        for tag, count in list(structure['tag_distribution'].items())[:10]:
            print(f"  #{tag}: {count} 次")
        
        print(f"\n最大的笔记 (按字数):")
        for i, note in enumerate(structure['largest_notes'][:5], 1):
            print(f"  {i}. {note['filename']}: {note['metadata']['word_count']} 字")
        
        # 保存提取的数据
        output_dir = "./data/processed"
        print(f"\n正在保存提取的数据到 {output_dir}...")
        saved_files = extractor.save_extracted_data(output_dir)
        
        print("\n=== 数据提取完成 ===")
        for key, path in saved_files.items():
            print(f"{key}: {path}")
            
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()