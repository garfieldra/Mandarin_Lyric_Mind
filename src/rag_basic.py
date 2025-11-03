import os
from dotenv import load_dotenv
#from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
#from langchain.vectorstores import Chroma
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.chains import create_retrieval_chain
# from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from pydantic import SecretStr
#from langchain.docstore.document import Document
from openai import OpenAI

load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
#
# SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
# SILICONFLOW_API_BASE = os.getenv("SILICONFLOW_API_BASE")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_BASE = os.getenv("SILICONFLOW_API_BASE")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")

if not (DEEPSEEK_API_KEY and  DEEPSEEK_API_BASE and SILICONFLOW_API_KEY and SILICONFLOW_API_BASE):
    raise RuntimeError("OPENAI_API_KEY 或 OPENAI_API_BASE 未配置")

os.environ["OPENAI_API_KEY"] = SILICONFLOW_API_KEY
os.environ["OPENAI_API_BASE"] = SILICONFLOW_API_BASE

lyrics_text = """
你眷恋的 都已离去
你问过自己无数次 想放弃的
眼前全在这里
超脱和追求时常是混在一起
你拥抱的并不总是也拥抱你
而我想说的 谁也不可惜
去挥霍和珍惜是同一件事情
我所有的何妨 何必
何其荣幸
在必须发现我们终将一无所有前
至少你可以说
我懂 活着的最寂寞
我拥有的都是侥幸啊
我失去的都是人生
当你不遗忘也不想曾经
我爱你
在必须感觉我们终将一无所有前
你做的让你可以说 是的
我有见过我的梦
我拥有的都是侥幸啊
我失去的都是人生
因为你担心的是你自己
我爱你
我爱你
我爱你
我爱你
我爱你
我爱你
我爱你
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50,
)

chunks = splitter.split_text(lyrics_text)
docs = [Document(page_content = c, metadata = {"source": "demo_lyrics"}) for c in chunks]

print(f"分成了 {len(docs)} 个文本块\n")

for i, d in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(d.page_content)

# import openai
# openai.api_key = OPENAI_API_KEY
# openai.api_base = OPENAI_API_BASE

silicon_client = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_API_BASE
)

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE
)


embedding_model = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_API_BASE
)

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "./chroma_data"
)

print("向量数据库构建完毕，共保存", vectorstore._collection.count(), "个向量\n")

retriever = vectorstore.as_retriever(search_kwargs = {"k":3})

llm = ChatOpenAI(
    model = "deepseek-chat",
    api_key = DEEPSEEK_API_KEY,
    base_url = DEEPSEEK_API_BASE,
    temperature = 1.3
)

prompt = ChatPromptTemplate.from_template(
    """
    你是一个精通华语独立音乐和歌词分析的AI助手。
    请根据以下提供的歌词片段回答用户的问题。
    如果歌词中没有足够信息，请直接说明无法确定。

    歌词片段：
    {context}

    用户问题：
    {question}
    """
)

rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke({"question":"这首歌的写作风格是怎样的，出自哪个歌手？"})

print("\n回答：", result)