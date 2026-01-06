import os
from unittest.mock import MagicMock

import pytest

from lyricmind.cli import LyricMindRAGSystem

def test_init_skip_env_validation():
    rag = LyricMindRAGSystem(validate_env=False)

def test_init_env_validation_missing_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError):
        LyricMindRAGSystem(validate_env=True)

def test_init_env_validation_skipped():
    system = LyricMindRAGSystem(validate_env=False)
    assert system is not None

def test_initialize_system_with_injected_modules():
    rag = LyricMindRAGSystem(validate_env=False)

    mock_data = MagicMock()
    mock_index = MagicMock()
    mock_gen = MagicMock()

    rag.initialize_system(
        data_module = mock_data,
        index_module = mock_index,
        generation_module = mock_gen
    )
    assert rag.data_module is mock_data

def test_search_by_artist_returns_titles():
    rag = LyricMindRAGSystem(validate_env=False)
    rag.retrieval_module = MagicMock()

    mock_doc = MagicMock()
    mock_doc.metadata = {"title": "测试歌曲"}

    rag.retrieval_module.metadata_filtered_search.return_value = [mock_doc]

    result = rag.search_by_artist("测试歌手")

    assert result == ["测试歌曲"]

def test_ask_question_direct_answer():
    rag = LyricMindRAGSystem(validate_env=False)
    rag.retrieval_module = MagicMock()
    rag.generation_module = MagicMock()

    rag.generation_module.query_router.return_value = "direct"
    rag.generation_module.generate_direct_answer.return_value = "答案"

    result = rag.ask_question("你好", stream=False)

    assert result == "答案"