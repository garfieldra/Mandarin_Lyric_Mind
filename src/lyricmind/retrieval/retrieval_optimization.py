"""
检索优化模块
"""

import logging
import jieba
import json
import numpy as np
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS, Milvus
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from lyricmind.graph.graph_search import LyricGraphSearcher
from lyricmind.generation.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """
    检索优化模块——负责混合检索和过滤
    """

    def __init__ (self, vectorstore: Milvus, chunks: List[Document], graph_searcher: LyricGraphSearcher, llm: Any):
        """
        初始化检索优化模块
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.graph_searcher = graph_searcher
        self.llm = llm
        self.setup_retrievals()

        self.embedder = self.vectorstore.embeddings
        self._init_graph_vocabulary()
        self.generation_module = GenerationIntegrationModule()

    @staticmethod
    def chinese_tokenizer(text: str):
        """
        中文 BM25 分词函数：使用 jieba 将连续汉字切成词。
        使用 cut_for_search 可以兼顾精确度和召回。
        """
        return list(jieba.cut_for_search(text))

    def _build_milvus_expr(self, filters:Dict[str, Any]) -> str:
        """
        将filters字典转换为 Milvus 识别的表达式字符串
        :param filters:
        :return:
        """
        if not filters:
            return ""

        expressions = []
        for key, value in filters.items():
            if isinstance(value, str):
                expressions.append(f'{key} == "{value}"')
            elif isinstance(value, (int, float)):
                expressions.append(f'{key} == {value}')
            elif isinstance(value, list):
                # 处理多选情况
                formatted_list = ",".join([f'"{v}"' for v in value])
                expressions.append(f'{key} in [{formatted_list}]')
        return " and ".join(expressions)

    def _extract_graph_intent(self, query: str):
        """
        使用LLM从查询中提取图谱参数
        :param query:
        :return:
        """
        prompt = f"""
        你是一个歌词知识专家。请从用户的问题中提取neo4j图数据库的检索参数。
        因为是从图数据库中检索，如果查询文本中没有明确的提示，请千万不要自作主张提取，遇到没有的、不确定的、可能的查询条件请一定大胆设置为null。
        用户问题："{query}"
        
        请严格按 JSON 格式返回，没有值则设为 null：
        {{
            "artist": "歌手名",
            "imagery": "意象词",
            "style": "风格词",
            "theme": "主题词",
            "emotion": "情感词"
        }}        
        """
        try:
            response = self.llm.invoke(prompt)
            # 清理 JSON 字符串
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"图谱意图解析失败: {e}")
            return {}

    def _get_graph_docs(self, query: str) -> List[Document]:
        """
        提取意图 -> Nep4j查询歌曲 -> 匹配对应的文档
        :param query:
        :return:
        """
        intent = self._extract_graph_intent(query)
        logger.info(f"图检索意图解析：{intent}")

        if hasattr(self, 'valid_emotions') and self.valid_emotions:
            if intent.get('emotion'):
                intent['emotion'] = self._align_semantic_entity(intent['emotion'], self.valid_emotions, self.emotion_embeddings)
            if intent.get('style'):
                intent['style'] = self._align_semantic_entity(intent['style'], self.valid_styles, self.style_embeddings)
            if intent.get('theme'):
                intent['theme'] = self._align_semantic_entity(intent['theme'], self.valid_themes, self.theme_embeddings)

        song_names = self.graph_searcher.search_songs_by_attributes(
            artist=intent.get('artist'),
            imagery=intent.get('imagery'),
            style=intent.get('style'),
            theme=intent.get('theme'),
            emotion=intent.get('emotion'),
            limit=10
        )

        if not song_names:
            return []

        graph_docs = [
            chunk for chunk in self.chunks
            if chunk.metadata.get('title') in song_names
        ]

        logger.info(f"图检索路径找到 {len(graph_docs)} 个相关片段 (涉及歌曲: {song_names})")
        return graph_docs

    def _init_graph_vocabulary(self):
        """从图数据库中拉取标签，并提前计算好向量缓存在内存中"""
        logger.info("正在初始化图谱语义对齐词表...")
        try:
            self.valid_emotions = self.graph_searcher.get_all_entities_by_label("Emotion")
            self.valid_styles = self.graph_searcher.get_all_entities_by_label("Style")
            self.valid_themes = self.graph_searcher.get_all_entities_by_label("Theme")

            if self.valid_emotions:
                self.emotion_embeddings = self.embedder.embed_documents(self.valid_emotions)
            if self.valid_styles:
                self.style_embeddings = self.embedder.embed_documents(self.valid_styles)
            if self.valid_themes:
                self.theme_embeddings = self.embedder.embed_documents(self.valid_themes)

            logger.info(f"图谱词汇库加载完毕：情感({len(self.valid_emotions)}，风格({len(self.valid_styles)})，主题{len(self.valid_themes)})")
        except Exception as e:
            logger.error(f"初始化图谱词汇库失败：{e}")
            self.valid_emotions = []

    def _cosine_similarity(self, vec1, vec2):
        """余弦相似度计算"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _align_semantic_entity(self, query_term: str, valid_terms: List[str], valid_embeddings: List[List[float]], threshold: float = 0.8):
        if not query_term or not valid_terms:
            return query_term

        try:
            query_vec = self.embedder.embed_query(query_term)
            best_match = None
            highest_score = -1.0

            for i, doc_vec in enumerate(valid_embeddings):
                score = self._cosine_similarity(query_vec, doc_vec)
                if score > highest_score:
                    highest_score = score
                    best_match = valid_terms[i]

            if highest_score > threshold:
                logger.info(f"语义对齐：意图 '{query_term}' -> 转换为图谱标准词 '{best_match}' （相似度：{highest_score:.2f}）")
                return best_match
            else:
                logger.info(f"对齐失败：'{query_term}' 没有找到足够相似的标签（最高相似度：{highest_score:.2f}）")
                return query_term
        except Exception as e:
            logger.error(f"语义对齐计算出错：{e}")
            return query_term

    def setup_retrievals(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        #向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": 10}
        )


        # # BM25 检索器：此时即使用默认 tokenizer（按空格切）也能正确工作
        # self.bm25_retriever = BM25Retriever.from_documents(
        #     bm25_docs,
        #     k=10
        # )

        # BM25 检索器：直接传入原始 chunks，并强制指定分词器
        # 这样 Langchain 就会在【建立索引】和【用户查询】时，统一使用 jieba 进行分词
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            preprocess_func=self.chinese_tokenizer,
            k=10
        )

        # 调试：确认分词效果时可以临时打开
        # logger.debug(f"BM25 分词示例：{self.chinese_tokenizer('张悬的歌曲有哪些')}")

        logger.info("检索器设置完成")

    def hyde_search(self, query: str):
        hypothetical_docs = self.generation_module.generate_hyde_text(query)
        return self.vector_retriever.invoke(hypothetical_docs)

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排
        query:查询文本；top_k:返回结果数量
        """
        # print(">>> vectorstore type =", type(self.vectorstore))
        # 分别获取向量检索和BM25检索结果
        # vector_docs = self.vector_retriever.get_relevant_documents(query)
        # bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        graph_docs = self._get_graph_docs(query)

        hyde_docs = self.hyde_search(query)

        # print("vector retriever type =", type(self.vector_retriever))
        # print("bm25 retriever type =", type(self.bm25_retriever))

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs, graph_docs, hyde_docs)
        return reranked_docs[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        带元数据过滤的检索
        filters:元数据过滤条件
        """
        # 构造Milvus表达式
        expr = self._build_milvus_expr(filters)
        logger.info(f"应用 Milvus 过滤表达式： {expr}")

        # 直接在向量库搜索时传入 expr
        docs = self.vectorstore.similarity_search(
            query=query,
            k=top_k,
            expr=expr
        )
        return docs

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], graph_docs: List[Document] = None, hyde_docs: List[Document] = None, k: int = 60) -> List[Document]:
        """
        使用RRF算法重排文档
        """
        if graph_docs is None:
            graph_docs = []

        if hyde_docs is None:
            hyde_docs = []

        doc_scores = {}
        doc_objects = {}

        def process_source(docs, source_label, weight=1.0):
            for rank, doc in enumerate(docs):
                chunk_id = str(doc.metadata.get("chunk_id", ""))
                doc_id = hash(doc.page_content + chunk_id)

                doc_objects[doc_id] = doc

                score = weight * (1.0 / (k + rank + 1))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

                logger.debug(f"{source_label}检索 - 文档{rank+1}: RRF分数 = {score:.4f}")

        # 处理三路数据
        process_source(vector_docs, "向量")
        process_source(bm25_docs, "BM25")
        process_source(graph_docs, "图谱", weight=1.1)  # 图谱路权重稍微调高

        process_source(hyde_docs, "HyDE")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        #构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档：{doc.page_content[:50]}... 最终RRF分数：{final_score:.4f}")

        logger.info(f"RRF重排完成：向量检索{len(vector_docs)}个文档，BM25检索{len(bm25_docs)}个文档，图谱{len(graph_docs)}个文档，HyDE{len(hyde_docs)}个文档，合并后{len(reranked_docs)}个文档")

        logger.info(">>> Vector检索原始结果:")
        for i, doc in enumerate(vector_docs):
            logger.info(
                f"{i + 1}. 标题={doc.metadata.get('title')} 歌手={doc.metadata.get('artist')} 片段={doc.page_content[:30]}")

        logger.info(">>> BM25检索原始结果:")
        for i, doc in enumerate(bm25_docs):
            logger.info(
                f"{i + 1}. 标题={doc.metadata.get('title')} 歌手={doc.metadata.get('artist')} 片段={doc.page_content[:30]}")

        logger.info(">>> 图谱检索原始结果:")
        for i, doc in enumerate(graph_docs):
            logger.info(
                f"{i + 1}. 标题={doc.metadata.get('title')} 歌手={doc.metadata.get('artist')} 片段={doc.page_content[:30]}")

        logger.info(">>> HyDE检索原始结果：")
        for i, doc in enumerate(hyde_docs):
            logger.info(
                f"{i + 1}. 标题={doc.metadata.get('title')} 歌手={doc.metadata.get('artist')} 片段={doc.page_content[:30]}")


        return reranked_docs












