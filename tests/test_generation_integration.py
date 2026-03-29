import pytest
from unittest.mock import MagicMock,patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence

from lyricmind.generation.generation_integration import GenerationIntegrationModule

def test_generate_list_answer_multiple_songs():
    docs = [
        Document(page_content="歌词A", metadata={"title": "歌A"}),
        Document(page_content="歌词B", metadata={"title": "歌B"}),
        Document(page_content="歌词A重复", metadata={"title": "歌A"}),
    ]

    module = GenerationIntegrationModule(llm=MagicMock())
    result = module.generate_list_answer("推荐歌曲", docs)

    assert "歌A" in result
    assert "歌B" in result
    assert result.count("歌A") == 1  # 不重复

def test_generate_list_answer_empty_docs():
    module = GenerationIntegrationModule(llm=MagicMock())
    result = module.generate_list_answer("推荐歌曲", [])

    assert "没有找到相关的歌曲信息" in result

def test_build_context_with_docs():
    docs = [
        Document(
            page_content="这是一段歌词",
            metadata={
                "title": "歌名",
                "artist": "歌手",
                "year": "2015",
            }
        )
    ]

    module = GenerationIntegrationModule(llm=MagicMock())
    context = module._build_context(docs)

    assert "歌名" in context
    assert "歌手" in context
    assert "2015" in context
    assert "这是一段歌词" in context

def test_build_context_empty():
    module = GenerationIntegrationModule(llm=MagicMock())
    context = module._build_context([])

    assert "暂无相关歌曲信息" in context

@pytest.mark.parametrize("llm_output,expected", [
    ("list", "list"),
    ("general", "general"),
    ("compare", "compare"),
    ("direct", "direct"),
    ("nonsense", "general"),  # fallback
])
def test_query_router(llm_output, expected):
    mock_llm = MagicMock()

    module = GenerationIntegrationModule(llm=mock_llm)

    # fake_chain = MagicMock()
    # fake_chain.invoke.return_value = llm_output
    #
    # with patch(
    #     "lyricmind.generation.generation_integration.PromptTemplate.__or__",
    #     return_value=fake_chain
    # ):
    with patch.object(
        RunnableSequence,
        "invoke",
        return_value=llm_output
    ):
        result = module.query_router("随便一个问题")

    assert result == expected

def test_query_rewritten_when_llm_rewrites():
    mock_llm = MagicMock()
    module = GenerationIntegrationModule(llm=mock_llm)

    # fake_chain = MagicMock()
    # fake_chain.invoke.return_value = "经典华语独立音乐歌曲推荐"

    # with patch(
    #     "lyricmind.generation.generation_integration.PromptTemplate.__or__",
    #     return_value=fake_chain
    # ):
    with patch.object(
        RunnableSequence,
        "invoke",
        return_value = "经典华语独立音乐歌曲推荐"
    ):
        result = module.query_rewritten("推荐首歌")

    assert result == "经典华语独立音乐歌曲推荐"

def test_query_rewritten_no_change():
    mock_llm = MagicMock()
    module = GenerationIntegrationModule(llm=mock_llm)

    # fake_chain = MagicMock()
    # fake_chain.invoke.return_value = "张悬的歌词风格是什么"
    #
    # with patch(
    #     "lyricmind.generation.generation_integration.PromptTemplate.__or__",
    #     return_value=fake_chain
    # ):
    with patch.object(
        RunnableSequence,
        "invoke",
        return_value = "张悬的歌词风格是什么"
    ):
        result = module.query_rewritten("张悬的歌词风格是什么")

    assert result == "张悬的歌词风格是什么"

def test_extract_subqueries_multiple_lines():
    mock_llm = MagicMock()
    module = GenerationIntegrationModule(llm=mock_llm)

    # fake_chain = MagicMock()
    # fake_chain.invoke.return_value = (
    #     "张悬 的 作品\n"
    #     "魏如萱 的 作词\n"
    # )
    #
    # with patch(
    #     "lyricmind.generation.generation_integration.PromptTemplate.__or__",
    #     return_value=fake_chain
    # ):
    with patch.object(
        RunnableSequence,
        "invoke",
        return_value="张悬 的 作品\n魏如萱 的 作词\n"
    ):
        subqueries = module.extract_subqueries("对比张悬、魏如萱的歌曲风格")

    assert len(subqueries) == 2
    assert "张悬" in subqueries[0]
    assert "魏如萱" in subqueries[1]





