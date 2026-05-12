"""
生成器模块

包含多种LLM后端的实现，支持本地推理、云API等。
"""

from .base import LLMGenerator, GenerationResult
from .mock_generator import MockGenerator
from .local_lm_studio import LocalLMStudioGenerator
from .openai_generator import OpenAIGenerator

__all__ = [
    'LLMGenerator',
    'GenerationResult',
    'MockGenerator',
    'LocalLMStudioGenerator',
    'OpenAIGenerator',
]
