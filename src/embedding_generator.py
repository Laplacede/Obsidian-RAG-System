#!/usr/bin/env python3
"""
嵌入向量生成器
使用BAAI/bge-small-zh-v1.5模型生成中文技术文档的嵌入向量
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm


class EmbeddingGenerator:
    """嵌入向量生成器"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", 
                 device: str = "cpu", batch_size: int = 32):
        """
        初始化嵌入生成器
        
        Args:
            model_name: 模型名称
            device: 设备 (cpu/cuda)
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"正在加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # 获取模型信息
            model_info = {
                'model_name': self.model_name,
                'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                'max_seq_length': self.model.max_seq_length,
                'device': str(self.device)
            }
            
            print(f"模型信息:")
            print(f"  - 嵌入维度: {model_info['embedding_dimension']}")
            print(f"  - 最大序列长度: {model_info['max_seq_length']}")
            print(f"  - 设备: {model_info['device']}")
            
            return model_info
            
        except ImportError:
            print("错误: 需要安装 sentence-transformers 包")
            print("运行: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], 
                           metadata_list: List[Dict] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        生成嵌入向量
        
        Args:
            texts: 文本列表
            metadata_list: 元数据列表
            
        Returns:
            (嵌入向量数组, 增强的元数据列表)
        """
        if self.model is None:
            self.load_model()
        
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(texts))]
        
        print(f"正在为 {len(texts)} 个文本生成嵌入向量...")
        
        # 分批处理
        all_embeddings = []
        enhanced_metadata = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="生成嵌入"):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]
            
            # 生成嵌入
            batch_embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            all_embeddings.append(batch_embeddings)
            
            # 增强元数据
            for j, metadata in enumerate(batch_metadata):
                enhanced = metadata.copy()
                enhanced.update({
                    'embedding_model': self.model_name,
                    'embedding_timestamp': time.time(),
                    'text_length': len(batch_texts[j]),
                    'word_count': len(batch_texts[j].split()),
                    'batch_index': i // self.batch_size,
                    'item_index_in_batch': j
                })
                enhanced_metadata.append(enhanced)
        
        # 合并所有嵌入
        embeddings = np.vstack(all_embeddings)
        
        print(f"嵌入生成完成: {embeddings.shape}")
        return embeddings, enhanced_metadata
    
    def process_chunks(self, chunks_file: str, chunk_type: str = "small") -> Dict[str, Any]:
        """
        处理分块数据并生成嵌入
        
        Args:
            chunks_file: 分块数据文件路径
            chunk_type: 分块类型
            
        Returns:
            处理结果
        """
        print(f"处理分块文件: {chunks_file}")
        
        # 加载分块数据
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"加载了 {len(chunks)} 个 {chunk_type} 分块")
        
        # 提取文本和元数据
        texts = []
        metadata_list = []
        
        for chunk in chunks:
            texts.append(chunk['content'])
            
            # 准备元数据
            metadata = chunk['metadata'].copy()
            metadata.update({
                'chunk_id': chunk['id'],
                'chunk_type': chunk['chunk_type'],
                'original_word_count': chunk['word_count'],
                'original_char_count': chunk['char_count']
            })
            metadata_list.append(metadata)
        
        # 生成嵌入
        embeddings, enhanced_metadata = self.generate_embeddings(texts, metadata_list)
        
        # 准备结果
        result = {
            'chunk_type': chunk_type,
            'total_chunks': len(chunks),
            'embeddings_shape': embeddings.shape,
            'embeddings': embeddings.tolist(),  # 转换为列表以便JSON序列化
            'metadata': enhanced_metadata,
            'texts': texts  # 可选，用于调试
        }
        
        return result
    
    def save_embeddings(self, result: Dict[str, Any], output_dir: str):
        """
        保存嵌入向量
        
        Args:
            result: 处理结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        chunk_type = result['chunk_type']
        
        # 保存嵌入向量
        embeddings_file = output_path / f"{chunk_type}_embeddings.json"
        
        # 由于嵌入向量可能很大，我们分开保存
        embeddings_data = {
            'chunk_type': chunk_type,
            'total_chunks': result['total_chunks'],
            'embeddings_shape': result['embeddings_shape'],
            'metadata': result['metadata']
        }
        
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
        
        print(f"嵌入元数据已保存到 {embeddings_file}")
        
        # 保存嵌入向量为numpy格式（更高效）
        embeddings_array = np.array(result['embeddings'])
        embeddings_npy_file = output_path / f"{chunk_type}_embeddings.npy"
        np.save(embeddings_npy_file, embeddings_array)
        
        print(f"嵌入向量已保存到 {embeddings_npy_file} (形状: {embeddings_array.shape})")
        
        # 保存统计信息
        stats = self._calculate_statistics(result)
        stats_file = output_path / f"{chunk_type}_embedding_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存到 {stats_file}")
        
        return {
            'embeddings_json': str(embeddings_file),
            'embeddings_npy': str(embeddings_npy_file),
            'statistics': str(stats_file)
        }
    
    def _calculate_statistics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计信息"""
        embeddings = np.array(result['embeddings'])
        
        stats = {
            'chunk_type': result['chunk_type'],
            'total_chunks': result['total_chunks'],
            'embedding_dimension': embeddings.shape[1],
            'embedding_stats': {
                'mean': float(np.mean(embeddings)),
                'std': float(np.std(embeddings)),
                'min': float(np.min(embeddings)),
                'max': float(np.max(embeddings))
            },
            'text_stats': {
                'avg_text_length': np.mean([len(t) for t in result.get('texts', [])]),
                'avg_word_count': np.mean([len(t.split()) for t in result.get('texts', [])])
            },
            'model_info': {
                'model_name': self.model_name,
                'device': self.device
            }
        }
        
        return stats
    
    def create_vector_store(self, result: Dict[str, Any], persist_dir: str):
        """
        创建向量存储（ChromaDB）
        
        Args:
            result: 处理结果
            persist_dir: 持久化目录
        """
        try:
            import chromadb
            from chromadb.config import Settings
            import json
            
            print(f"正在创建向量存储: {persist_dir}")
            
            # 创建Chroma客户端
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            chunk_type = result['chunk_type']
            collection_name = f"obsidian_{chunk_type}_chunks"
            
            # 获取或创建集合
            try:
                collection = client.get_collection(collection_name)
                print(f"使用现有集合: {collection_name}")
            except:
                # 创建自定义嵌入函数，避免使用默认的MiniLM模型
                from chromadb.utils import embedding_functions
                
                # 创建自定义嵌入函数，返回我们预计算的嵌入
                class PrecomputedEmbeddingFunction(embedding_functions.EmbeddingFunction):
                    def __init__(self, embeddings: np.ndarray):
                        self.embeddings = embeddings
                        
                    def __call__(self, input):
                        # 这里我们只返回预计算的嵌入
                        # 在实际查询时，我们需要使用相同的模型
                        return self.embeddings
                
                # 注意：由于我们已经有预计算的嵌入，我们创建一个不实际计算嵌入的函数
                # 在查询时，我们需要使用相同的BAAI模型
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"description": f"Obsidian {chunk_type} chunks"},
                    embedding_function=None  # 禁用默认嵌入函数
                )
                print(f"创建新集合: {collection_name}")
            
            # 准备数据
            embeddings = result['embeddings']
            metadata_list = result['metadata']
            texts = result.get('texts', [])
            
            # 如果没有文本，使用元数据中的内容
            if not texts and metadata_list:
                texts = []
                for metadata in metadata_list:
                    # 尝试从metadata获取内容
                    if 'content' in metadata:
                        texts.append(metadata['content'])
                    else:
                        texts.append("")
            
            # 生成ID
            ids = [m.get('chunk_id', f"chunk_{i}") for i, m in enumerate(metadata_list)]
            
            # 扁平化metadata，确保所有值都是基本类型
            flattened_metadatas = []
            for metadata in metadata_list:
                flattened = {}
                for key, value in metadata.items():
                    if isinstance(value, (list, dict)):
                        # 将列表或字典转换为JSON字符串
                        try:
                            flattened[key] = json.dumps(value, ensure_ascii=False)
                        except:
                            flattened[key] = str(value)
                    elif value is None:
                        # ChromaDB不接受None值，跳过或使用空字符串
                        continue
                    elif isinstance(value, (str, int, float, bool)):
                        flattened[key] = value
                    else:
                        # 其他类型转换为字符串
                        flattened[key] = str(value)
                flattened_metadatas.append(flattened)
            
            # 添加到集合
            print(f"正在添加 {len(ids)} 个文档到向量存储...")
            
            # 分批添加
            batch_size = 100
            for i in tqdm(range(0, len(ids), batch_size), desc="添加到向量存储"):
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadatas = flattened_metadatas[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                
                collection.add(
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts,
                    ids=batch_ids
                )
            
            # 获取集合信息
            collection_info = {
                'name': collection_name,
                'count': collection.count(),
                'metadata': collection.metadata
            }
            
            print(f"向量存储创建完成: {collection_info['count']} 个文档")
            
            return {
                'collection_name': collection_name,
                'document_count': collection_info['count'],
                'persist_directory': persist_dir
            }
            
        except ImportError:
            print("错误: 需要安装 chromadb 包")
            print("运行: pip install chromadb")
            raise
        except Exception as e:
            print(f"创建向量存储失败: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="嵌入向量生成器")
    parser.add_argument("--chunks", "-c", required=True, help="分块数据文件路径")
    parser.add_argument("--type", "-t", default="small", choices=["small", "medium", "large"], 
                       help="分块类型")
    parser.add_argument("--output", "-o", default="./data/embeddings", help="输出目录")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], 
                       help="设备类型")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="批处理大小")
    parser.add_argument("--create-store", action="store_true", help="创建向量存储")
    parser.add_argument("--store-dir", default="./data/vector_store", help="向量存储目录")
    
    args = parser.parse_args()
    
    try:
        # 创建嵌入生成器
        generator = EmbeddingGenerator(
            model_name="BAAI/bge-small-zh-v1.5",
            device=args.device,
            batch_size=args.batch_size
        )
        
        # 加载模型
        model_info = generator.load_model()
        
        # 处理分块数据
        result = generator.process_chunks(args.chunks, args.type)
        
        # 保存嵌入向量
        print(f"\n保存嵌入向量到: {args.output}")
        saved_files = generator.save_embeddings(result, args.output)
        
        # 创建向量存储（如果指定）
        if args.create_store:
            print(f"\n创建向量存储到: {args.store_dir}")
            store_info = generator.create_vector_store(result, args.store_dir)
            print(f"向量存储信息: {store_info}")
        
        print("\n=== 嵌入生成完成 ===")
        print(f"分块类型: {args.type}")
        print(f"处理分块数: {result['total_chunks']}")
        print(f"嵌入维度: {result['embeddings_shape'][1]}")
        
        for key, path in saved_files.items():
            print(f"{key}: {path}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()