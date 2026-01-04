import os
import pytest

from lyricmind.config import RAGConfig,DEFAULT_CONFIG


def test_defaut_config_creation():
    assert isinstance(DEFAULT_CONFIG, RAGConfig)

def test_default_config_values():
    """
    测试默认配置项是否符合预期
    """
    cfg = DEFAULT_CONFIG

    assert cfg.data_path == "data"
    assert cfg.index_save_path == "index"
    assert cfg.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert cfg.llm_model == "deepseek-reasoner"
    assert cfg.top_k == 10
    assert cfg.top_cmp_k == 3
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 2048

def test_config_from_dict():
    """
    from_dict 应该能用字典正确构造配置对象
    """
    cfg_dict = {
        "data_path": "test_data",
        "index_save_path": "test_index",
        "embedding_model": "test-embedding",
        "llm_model": "test-llm",
        "top_k": 5,
        "top_cmp_k": 2,
        "temperature": 0.5,
        "max_tokens": 1024,
    }

    cfg = RAGConfig.from_dict(cfg_dict)

    assert cfg.data_path == "test_data"
    assert cfg.index_save_path == "test_index"
    assert cfg.embedding_model == "test-embedding"
    assert cfg.llm_model == "test-llm"
    assert cfg.top_k == 5
    assert cfg.top_cmp_k == 2
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 1024

def test_config_to_dict():
    """
    to_dict 应该返回完整、正确的字典表示
    """
    cfg = RAGConfig(
        data_path="d",
        index_save_path="i",
        embedding_model="e",
        llm_model="l",
        top_k=1,
        top_cmp_k=1,
        temperature=0.1,
        max_tokens=256,
    )

    cfg_dict = cfg.to_dict()

    assert cfg_dict["data_path"] == "d"
    assert cfg_dict["index_save_path"] == "i"
    assert cfg_dict["embedding_model"] == "e"
    assert cfg_dict["llm_model"] == "l"
    assert cfg_dict["top_k"] == 1
    assert cfg_dict["temperature"] == 0.1
    assert cfg_dict["max_tokens"] == 256
