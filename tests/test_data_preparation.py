import pytest
from pathlib import Path

from lyricmind.ingest.data_preparation import DataPreparationModule


# ---------- fixtures ----------

@pytest.fixture
def sample_markdown(tmp_path: Path) -> Path:
    """
    创建一个最小可用的 markdown 歌词文件
    目录结构用于测试 artist_from_path
    """
    artist_dir = tmp_path / "陈建骐"
    artist_dir.mkdir()

    md_file = artist_dir / "测试歌.md"
    md_file.write_text(
        """## 歌名
测试歌

## 歌手
陈建骐

## 收录专辑
测试专辑

## 发行时间
2023

## 地区
台湾

## 类型
流行

## 歌词
这是第一句歌词
这是第二句歌词
""",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def data_module(sample_markdown: Path) -> DataPreparationModule:
    return DataPreparationModule(data_path=str(sample_markdown), chunk_size_chars=50)


# ---------- tests: load & metadata ----------

def test_load_documents_success(data_module: DataPreparationModule):
    parents = data_module.load_documents()

    assert len(parents) == 1

    doc = parents[0]
    md = doc.metadata

    assert md["artist"] == "陈建骐"
    assert md["title"] == "测试歌"
    assert md["album"] == "测试专辑"
    assert md["year"] == "2023"
    assert md["region"] == "台湾"
    assert "parent_id" in md
    assert md["doc_type"] == "parent"


def test_load_documents_invalid_path():
    module = DataPreparationModule(data_path="not_exist_path")
    with pytest.raises(FileNotFoundError):
        module.load_documents()


# ---------- tests: chunking ----------

def test_chunk_documents_generates_chunks(data_module: DataPreparationModule):
    data_module.load_documents()
    chunks = data_module.chunk_documents()

    assert len(chunks) > 0

    c = chunks[0]
    assert c.metadata["doc_type"] == "child"
    assert "chunk_id" in c.metadata
    assert "parent_id" in c.metadata
    assert c.page_content.strip() != ""


def test_chunk_documents_without_loading():
    module = DataPreparationModule(data_path=".")
    with pytest.raises(ValueError):
        module.chunk_documents()


# ---------- tests: index items ----------

def test_create_index_items(data_module: DataPreparationModule):
    data_module.load_documents()
    data_module.chunk_documents()
    items = data_module.create_index_items()

    assert len(items) > 0
    item = items[0]

    assert "id" in item
    assert "text" in item
    assert "metadata" in item


def test_create_index_items_without_chunks(data_module: DataPreparationModule):
    with pytest.raises(ValueError):
        data_module.create_index_items()


# ---------- tests: filters ----------

def test_filter_documents_by_artist(data_module: DataPreparationModule):
    data_module.load_documents()
    results = data_module.filter_documents_by_artist("陈建骐")

    assert len(results) == 1
    assert results[0].metadata["artist"] == "陈建骐"


def test_filter_chunks_by_keyword(data_module: DataPreparationModule):
    data_module.load_documents()
    data_module.chunk_documents()

    results = data_module.filter_chunks_by_keyword("第一句")

    assert len(results) >= 1
    assert "第一句" in results[0].page_content


# ---------- tests: statistics ----------

def test_get_statistics(data_module: DataPreparationModule):
    data_module.load_documents()
    data_module.chunk_documents()

    stats = data_module.get_statistics()

    assert stats["total_parents"] == 1
    assert stats["total_chunks"] > 0
    assert stats["artists_count"]["陈建骐"] == 1