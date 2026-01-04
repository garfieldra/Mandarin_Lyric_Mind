import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from lyricmind.index.index_construction import IndexConstructionModule


# ---------- fixtures ----------

@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="这是第一段歌词", metadata={"chunk_id": "1"}),
        Document(page_content="这是第二段歌词", metadata={"chunk_id": "2"}),
    ]


@pytest.fixture
def mock_embeddings():
    return MagicMock(name="MockEmbeddings")


@pytest.fixture
def mock_vectorstore():
    vs = MagicMock(name="MockFAISS")
    vs.similarity_search.return_value = [
        Document(page_content="搜索结果", metadata={})
    ]
    return vs


# ---------- tests ----------

@patch("lyricmind.index.index_construction.HuggingFaceEmbeddings")
def test_init_embeddings(mock_hf):
    module = IndexConstructionModule(model_name="fake-model")

    assert module.embeddings is not None
    mock_hf.assert_called_once()


def test_build_vector_index_empty():
    module = IndexConstructionModule()
    with pytest.raises(ValueError):
        module.build_vector_index([])


@patch("lyricmind.index.index_construction.FAISS.from_documents")
@patch("lyricmind.index.index_construction.HuggingFaceEmbeddings")
def test_build_vector_index_success(mock_hf, mock_from_docs, sample_chunks, mock_vectorstore):
    mock_from_docs.return_value = mock_vectorstore

    module = IndexConstructionModule()
    vs = module.build_vector_index(sample_chunks)

    assert vs is mock_vectorstore
    mock_from_docs.assert_called_once()
    assert module.vectorstore is mock_vectorstore


def test_add_documents_without_index(sample_chunks):
    module = IndexConstructionModule()
    with pytest.raises(ValueError):
        module.add_documents(sample_chunks)


def test_add_documents_success(sample_chunks, mock_vectorstore):
    module = IndexConstructionModule()
    module.vectorstore = mock_vectorstore

    module.add_documents(sample_chunks)

    mock_vectorstore.add_documents.assert_called_once_with(sample_chunks)


def test_save_index_without_build():
    module = IndexConstructionModule()
    with pytest.raises(ValueError):
        module.save_index()


def test_save_index_success(tmp_path: Path, mock_vectorstore):
    module = IndexConstructionModule(index_save_path=str(tmp_path))
    module.vectorstore = mock_vectorstore

    module.save_index()

    mock_vectorstore.save_local.assert_called_once_with(str(tmp_path))


@patch("lyricmind.index.index_construction.FAISS.load_local")
def test_load_index_not_exist(mock_load, tmp_path: Path):
    module = IndexConstructionModule(index_save_path=str(tmp_path / "not_exist"))
    result = module.load_index()

    assert result is None
    mock_load.assert_not_called()


@patch("lyricmind.index.index_construction.FAISS.load_local")
def test_load_index_success(mock_load, tmp_path: Path, mock_vectorstore):
    mock_load.return_value = mock_vectorstore
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    module = IndexConstructionModule(index_save_path=str(index_dir))
    vs = module.load_index()

    assert vs is mock_vectorstore
    assert module.vectorstore is mock_vectorstore


def test_similarity_search_without_index():
    module = IndexConstructionModule()
    with pytest.raises(ValueError):
        module.similarity_search("测试")


def test_similarity_search_success(mock_vectorstore):
    module = IndexConstructionModule()
    module.vectorstore = mock_vectorstore

    results = module.similarity_search("测试", k=1)

    assert len(results) == 1
    mock_vectorstore.similarity_search.assert_called_once_with("测试", k=1)