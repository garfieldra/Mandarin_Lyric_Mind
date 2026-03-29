"""
索引构建模块
"""
import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class IndexConstructionModule:
    """
    索引构建模块
    """
    # def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", index_save_path: str = "../vector_index"):
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 connection_args: dict = {"uri": "http://localhost:19530"},
                 collection_name:str ="liricmind"):
        self.model_name = model_name
        self.connection_args = connection_args  # 存储数据库地址
        self.collection_name = collection_name  # 存储“表名”
        # """
        # 初始化索引构建模块
        # :param model_name:
        # :param index_save_path:
        # """
        # self.model_name = model_name
        # self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """
        初始化嵌入模型
        :param self:
        :return:
        """
        logger.info(f"正在初始化嵌入模型{self.model_name}")
        self.embeddings  = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {'normalize_embeddings': True}
        )
        logger.info("嵌入模型初始化完成")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建向量索引
        :param chunks: 文档块列表
        :return: FAISS向量索引对象
        """
        logger.info("正在构建FAISS向量索引")
        if not chunks:
            raise ValueError("文档块列表不能为空")

        # self.vectorstore = FAISS.from_documents(
        #     documents=chunks,
        #     embedding=self.embeddings
        # )

        # 构建Milvus向量存储
        self.vectorstore = Milvus.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection_args=self.connection_args,
            drop_old = True #每次重新构建时，必须删掉旧表重建新表
        )

        logger.info(f"向量索引构建完成，包含{len(chunks)}个向量")
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档
        :param new_chunks: 新的文档块列表
        :return:
        """
        if not self.vectorstore:
            self.load_index()
        self.vectorstore.add_documents(new_chunks)
        #     raise ValueError("请先构建向量索引")
        # logger.info(f"正在添加{len(new_chunks)}个新文档到索引")
        # self.vectorstore.add_documents(new_chunks)
        # logger.info("新文档添加完成")

    def save_index(self):
        """
        保存向量到配置的路径
        """
        # if not self.vectorstore:
        #     raise ValueError("请先构建向量索引")
        #
        # Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        #
        # self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已经完成持久化")

    def load_index(self):
        """
        连接现有的 Milvus 集合
        """
        if not self.vectorstore:
            self.setup_embeddings()
        # if not Path(self.index_save_path).exists():
        #     logger.info(f"索引路径{self.index_save_path}不存在，将构建新索引")
        #     return None
        #
        # try:
        #     self.vectorstore = FAISS.load_local(
        #         self.index_save_path,
        #         self.embeddings,
        #         allow_dangerous_deserialization=True
        #     )},将构建新索引")
        #     return None
        try:
            self.vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                auto_id = True
            )
            logger.info(f"成功连接到 Milvus 集合：{self.collection_name}")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载 Milvus 索引失败：{e}")
            return None

    def get_count(self) -> int:
        """获取当前集合中的实体（向量）总数"""
        if self.vectorstore is not None:
            try:
                return self.vectorstore.col.num_entities
            except Exception:
                return 0
        return 0


    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
            logger.info(f"向量索引已经从{self.index_save_path}中加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败：{e
        相似度搜索
        """
        if not self.vectorstore:
            self.load_index()
            if not self.vectorstore:
                raise ValueError("未找到可用的向量存储")

        return self.vectorstore.similarity_search(query, k=k)


















