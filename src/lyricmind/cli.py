"""
主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# sys.path.append(Path(__file__).parent)

from dotenv import load_dotenv
from lyricmind.config import DEFAULT_CONFIG, RAGConfig
from lyricmind.ingest.data_preparation import DataPreparationModule
from lyricmind.index.index_construction import IndexConstructionModule
from lyricmind.retrieval.retrieval_optimization import RetrievalOptimizationModule
from lyricmind.generation.generation_integration import GenerationIntegrationModule

load_dotenv()

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class LyricMindRAGSystem:
    """LyricMind歌词RAG系统主类"""

    def __init__(self, config: RAGConfig = None, validate_env: bool = True):
        """初始化RAG系统"""
        self.config = config or DEFAULT_CONFIG

        # 允许跳过环境检查
        if validate_env:
            if not Path(self.config.data_path).exists():
                raise FileNotFoundError(f"数据路径不存在：{self.config.data_path}")
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("请设置DEEPSEEK_API_KEY环境变量")
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # if not Path(self.config.data_path).exists():
        #     raise FileNotFoundError(f"数据路径不存在：{self.config.data_path}")
        #
        # if not os.getenv("DEEPSEEK_API_KEY"):
        #     raise ValueError("请设置DEEPSEEK_API_KEY环境变量")

    def initialize_system(
            self,
            data_module = None,
            index_module = None,
            generation_module = None,
    ):
        """初始化所有模块"""
        # print("正在初始化RAG系统...")
        logger.info("正在初始化RAG系统...")

        # print("正在初始化数据准备模块...")
        logger.info("正在初始化数据准备模块...")
        self.data_module = data_module or DataPreparationModule(self.config.data_path)

        logger.info("正在初始化索引构建模块...")
        self.index_module = index_module or IndexConstructionModule(
            model_name = self.config.embedding_model,
            index_save_path = self.config.index_save_path
        )

        logger.info("正在初始化生成集成模块")
        self.generation_module = generation_module or GenerationIntegrationModule(
            model_name = self.config.llm_model,
            temperature = self.config.temperature,
            max_tokens = self.config.max_tokens
        )

        print("系统初始化完成")

    def build_knowledge_base(self):
        """构建知识库"""
        logger.info("正在构建知识库...")

        # 尝试加载已经保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            logger.info("成功加载已保存的向量索引")
            logger.info("加载歌曲文档...")
            self.data_module.load_documents()
            logger.info("进行文本分块")
            chunks = self.data_module.chunk_documents()
        else:
            logger.info("未找到已保存的索引，开始构建新索引...")

            logger.info("加载歌曲文档...")
            self.data_module.load_documents()

            logger.info("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            logger.info("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            logger.info("保存向量索引...")
            self.index_module.save_index()

        logger.info("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        stats = self.data_module.get_statistics()
        logger.info("知识库统计｜文档总数：%d|文本块数：%d|歌手分类：%s｜地区分布：%s",
                    stats['total_parents'],
                    stats['total_chunks'],
                    list(stats['artists_count'].keys()),
                    stats['regions_count'])
        # logger.info(f"    文档总数：{stats['total_parents']}")
        # logger.info(f"    文本块数：{stats['total_chunks']}")
        # logger.info(f"    歌手分类：{list(stats['artists_count'].keys())}")
        # logger.info(f"    地区分布：{stats['regions_count']}")

        logger.info("知识库构建完成")

    def ask_question(self, question: str, stream: bool = False):
        """回答用户问题"""
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # print(f"\n 用户问题：{question}")
        logger.info("用户问题：%s", question)

        route_type = self.generation_module.query_router(question)
        # print(f"\n 查询类型：{route_type}")
        logger.info("查询类型：%s", route_type)

        if route_type == 'direct':
            # print("\n 直接回答类型（不使用知识库）")
            logger.info("直接回答类型（不使用知识库）")
            if stream:
                return self.generation_module.generate_direct_answer_stream(question)
            else:
                return self.generation_module.generate_direct_answer(question)


        if route_type == 'list':
            #保持原查询
            rewritten_query = question
            # print(f" 列表查询保持原样: {rewritten_query}")
            logger.info(" 列表查询保持原样: %s", rewritten_query)

        else:
            # print(f" 智能分析查询...")
            logger.info(" 智能分析查询...")
            rewritten_query = self.generation_module.query_rewritten(question)

        if route_type == 'compare':
            subqueries = self.generation_module.extract_subqueries(rewritten_query)
            # print(f"识别到子查询：{subqueries}")
            logger.info("识别到子查询：%s", subqueries)
            all_chunks = []
            for subquery in subqueries:
                print(f"\n 子查询：{subquery}")

                rewritten_subquery = subquery

                filters = self._extract_filters_from_query(subquery)

                if filters:
                    # print(f"\n 应用过滤条件：{filters}")
                    logger.info("应用过滤条件：%s", filters)
                    chunks = self.retrieval_module.metadata_filtered_search(rewritten_subquery, filters, top_k = self.config.top_cmp_k)
                else:
                    chunks = self.retrieval_module.hybrid_search(rewritten_subquery, top_k = self.config.top_cmp_k)

                all_chunks.append(chunks)
            relevant_chunks = all_chunks
            # print("获取完整文档...")
            logger.info("获取完整文档...")
            relevant_docs = []
            for chunk_list in relevant_chunks:
                doc_list = self.data_module.get_parent_documents(chunk_list)
                relevant_docs.append(doc_list)
            # print(" 生成详细回答...")
            logger.info("生成详细回答...")

            if stream:
                return self.generation_module.generate_compare_answer_stream(rewritten_query, relevant_docs)
            else:
                return self.generation_module.generate_compare_answer(rewritten_query, relevant_docs)


        subqueries = self.generation_module.extract_subqueries(rewritten_query)
        # print(f"识别到子查询：{subqueries}")
        logger.info("识别到子查询：%s", subqueries)

        all_chunks = []

        for subquery in subqueries:
            # print(f"\n 子查询：{subquery}")
            logger.info("子查询：%s", subquery)

            if route_type == 'list':
                rewritten_subquery = subquery
            else:
                rewritten_subquery = self.generation_module.query_rewritten(subquery)

            filters = self._extract_filters_from_query(subquery)
            if filters:
                # print(f" 应用过滤条件：{filters}")
                logger.info(" 应用过滤条件：%s", filters)
                chunks = self.retrieval_module.metadata_filtered_search(rewritten_subquery, filters, top_k = self.config.top_k)
            else:
                chunks = self.retrieval_module.hybrid_search(rewritten_subquery, top_k = self.config.top_k)

            all_chunks.extend(chunks)

        relevant_chunks = all_chunks

        # print(" 检索相关文档...")
        # filters = self._extract_filters_from_query(question)
        # if filters:
        #     print(f"应用过滤条件：{filters}")
        #     relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k = self.config.top_k)
        # else:
        #     relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k = self.config.top_k)

        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                title = chunk.metadata.get('title', '未知歌曲')
                artist = chunk.metadata.get('artist', '未知歌手')
                preview = chunk.page_content[:15].replace("\n", " ").strip()

                if preview:
                    chunk_info.append(f"{title}, {artist}(片段：{preview})")
                else:
                    chunk_info.append(f"{title}, {artist}(歌词片段)")

            # print(f"找到{len(relevant_chunks)}个相关文档块：{','.join(chunk_info)}")
            logger.info("找到%d个相关文档块：%s", len(relevant_chunks), ','.join(chunk_info))
        else:
            # print(f"找到{len(relevant_chunks)}个相关文档块")
            logger.info("找到%d个相关文档块", len(relevant_chunks))

        if not relevant_chunks:
            return "抱歉，没有找到相关的歌曲信息。请尝试其他歌曲名称或关键词。"

        if route_type == 'list':
            print(" 生成歌曲列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            doc_names = []
            for doc in relevant_docs:
                title = doc.metadata.get('title', '未知歌曲')
                doc_names.append(title)

            if doc_names:
                # print(f"找到文档：{','. join(doc_names)}")
                logger.info("找到文档：%s", ','.join(doc_names))
            return self.generation_module.generate_list_answer(question, relevant_docs)
        else:
            # 详细查询，获取完整文档生成详细回答。
            # print("获取完整文档...")
            logger.info("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            doc_names = []
            for doc in relevant_docs:
                title = doc.metadata.get('title', '未知歌曲')
                doc_names.append(title)

            if doc_names:
                # print(f"找到文档：{','. join(doc_names)}")
                logger.info("找到文档：%s", ','.join(doc_names))
            else:
                # print(f"对应{len(relevant_docs)}个完整文档")
                logger.info("对应%d个完整文档", len(relevant_docs))

            # print(" 生成详细回答...")
            logger.info(" 生成详细回答...")

            if stream:
                return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
            else:
                return self.generation_module.generate_basic_answer(question, relevant_docs)

    def _extract_filters_from_query(self, query: str) -> dict:
        """从用户问题中提取元数据过滤条件"""
        filters = {}

        artists = {d.metadata.get("artist") for d in self.data_module.parent_documents}
        albums = {d.metadata.get("album") for d in self.data_module.parent_documents}
        regions = {d.metadata.get("region") for d in self.data_module.parent_documents}
        years = {d.metadata.get("year") for d in self.data_module.parent_documents}
        titles = {d.metadata.get("title") for d in self.data_module.parent_documents}

        artists.discard(None); albums.discard(None)
        regions.discard(None); years.discard(None); titles.discard(None)

        for artist in sorted(artists, key = len, reverse = True):
            if artist and artist in query:
                filters["artist"] = artist
                break

        for album in sorted(albums, key = len, reverse = True):
            if album and album in query:
                filters["album"] = album

        for region in sorted(regions, key = len, reverse = True):
            if region and region in query:
                filters["region"] = region

        for year in sorted(years, key = len, reverse = True):
            if year and year in query:
                filters["year"] = year

        for title in sorted(titles, key = len, reverse = True):
            if title and title in query:
                filters["title"] = title

        return filters


    def search_by_artist(self, artist: str, query: str = "") -> List[str]:
        """按歌手名搜索歌曲 query:额外的查询条件"""
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        search_query = query if query else artist
        filters = {"artist": artist}

        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k = 10)

        titles = []
        for doc in docs:
            title = doc.metadata.get('title', '未知歌曲')
            if title not in titles:
                titles.append(title)

        return titles

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print(" LyricMind独立音乐分析系统 - 交互式问答 ")
        print("=" * 60)
        print(" 汇集两岸三地优质独立音乐! ")

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n交互式问答（输入「退出」以退出问答系统）")

        while True:
            try:
                user_input = input("\n您的问题：").strip()
                if user_input.strip().lower() in ['退出', 'quit', 'exit']:
                    break

                stream_choice = input("是否使用流式输出？(y/n，默认y):").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答：")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream = True):
                        print(chunk, end = "", flush = True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream = False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"系统运行时出错：{e}")
                print(f"处理问题时出错：{e}")
        print("\n感谢使用LyricMind！")

def main():
    """主函数"""
    try:
        rag_system = LyricMindRAGSystem()

        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行时出错：{e}")
        print(f"系统运行时出错：{e}")

if __name__ == "__main__":
    main()