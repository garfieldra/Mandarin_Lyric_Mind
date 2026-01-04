import pytest
from langchain_core.documents import Document
from lyricmind.retrieval.retrieval_optimization import RetrievalOptimizationModule


def make_doc(text, chunk_id):
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id}
    )


def test_rrf_rerank_merges_and_scores_documents():
    module = RetrievalOptimizationModule.__new__(RetrievalOptimizationModule)

    # 向量检索结果
    vector_docs = [
        make_doc("doc A", "1"),
        make_doc("doc B", "2"),
    ]

    # BM25 检索结果（doc A 重叠）
    bm25_docs = [
        make_doc("doc A", "1"),
        make_doc("doc C", "3"),
    ]

    reranked = module._rrf_rerank(vector_docs, bm25_docs, k=60)

    # 一共 3 个文档
    assert len(reranked) == 3

    # doc A 同时出现，RRF 分数最高
    assert reranked[0].page_content == "doc A"

    # 每个文档都应有 rrf_score
    for doc in reranked:
        assert "rrf_score" in doc.metadata
        assert doc.metadata["rrf_score"] > 0