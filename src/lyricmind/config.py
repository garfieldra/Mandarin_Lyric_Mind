"""RAG配置文件"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    data_path: str = "data"
    index_save_path: str = "index"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # llm_model: str = "kimi-k2-0711-preview"
    llm_model: str = "deepseek-reasoner"


    top_k: int = 10
    top_cmp_k: int = 3

    temperature: float = 0.7
    max_tokens: int = 2048

    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return{
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

DEFAULT_CONFIG = RAGConfig()







