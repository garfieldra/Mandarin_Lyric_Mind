"""
生成集成模块
"""
import json
import logging
import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块"""

    def __init__(self, model_name = "deepseek-reasoner", temperature: float = 0.1, max_tokens: int = 2048, llm = None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        if llm is None:
            self.llm = None
            self.setup_llm()
        else:
            self.llm = llm
        # self.llm = None
        # self.setup_llm()

    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        # api_key = os.environ.get("MOONSHOT_API_KEY")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量")

        # self.llm = MoonshotChat(
        #     model = self.model_name,
        #     temperature = self.temperature,
        #     max_tokens = self.max_tokens,
        #     moonshot_api_key = api_key
        # )
        self.llm = ChatOpenAI(
            model = self.model_name,
            temperature = self.temperature,
            max_tokens = self.max_tokens,
            api_key = api_key,
            base_url = "https://api.deepseek.com"
        )

        logger.info("llm初始化完成")

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """生成基础回答"""
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template(
            """
你是一位专业的华语独立音乐歌词助手。请根据以下歌曲信息回答用户提问。

用户问题：{question}

相关歌曲信息：
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答：""")
        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def query_rewritten(self, query: str) -> str:
        """智能查询重写 - 让大模型判断是否需要重写查询"""
        prompt = PromptTemplate(
            template = """
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询：{query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
  - 包含具体歌手或专辑名称：如“张悬的歌词风格是怎样的”、“《战神卡尔迪亚》专辑的作词风格”
  - 明确的歌曲名询问：如“《你是自由的》主要表现了怎样的情感？有什么社会背景”
  
2. **模糊不清的查询** （需要重写）
  - 过于宽泛：如“歌词”、“推荐首歌”
  - 缺乏具体信息：如“乐团音乐”、“台湾音乐”、“2010年的”
  - 口语化表达：如“想听首什么”
  
重写原则
- 保持原意不变
- 增加音乐、歌词相关术语
- 保持简洁性

示例：
- “歌词” → “推荐几首经典的华语独立音乐歌曲”
- “推荐首歌” → “经典华语独立音乐歌曲推荐”
- “乐团音乐” → “经典乐团音乐”
- “台湾音乐” → “经典台湾音乐推荐”
- “2010年的” → “2010年发行的独立音乐推荐”
- “想听首什么” → 推荐一两首经典的华语独立音乐“”

请输出最终查询（如果不需要重写就返回原查询）：""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query)
        if not result:
            return query

        response = result.strip()

        #记录重写结果
        if response != query:
            logger.info(f"查询已重写：'{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写：'{query}'")

        return response

    def query_router(self, query: str) -> str:
        """查询路由 - 根据查询类型选择不同的处理方式"""
        prompt = ChatPromptTemplate.from_template(
            f"""
根据用户的问题，将其分类为以下几种类型之一：

1. 'list' - 用户想获取歌曲列表或推荐，只需要菜名
   例如：推荐几首台湾独立音乐、有什么乐团歌曲、给我3首魏如萱的歌
   
2. 'direct' - 不需要使用歌曲知识库的常识性问题
   例如：介绍一下张悬、为什么歌词中副歌一般会出现好几次
   
3. 'compare' - 比较类问题
   例如：比较一下焦安溥在张悬和安溥时期的作词风格、比较一下魏如萱和艾怡良的作品
   
4. 'general' - 其他需要使用歌曲知识库的一般性问题
   例如：《玫瑰色的你》的写作背景是什么、张悬的歌词写作风格是怎样的
   
请只返回分类结果：list或general或compare或direct

用户问题：{query}

分类结果：""")

# 2. 'detail' - 用户想要具体的歌曲信息
        #    例如：《你是自由的》歌曲的歌词描写了那些社会事件、有什么写作风格、是哪一年发行的

        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip()

        if result in ["list", "general", "compare", "direct"]:
            return result
        else:
            return 'general'

    # 通过llm判断在数据库内部是否需要进行多个查询
    def extract_subqueries(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_template(
            f"""
你是一个查询结构化助手。请将用户的查询拆解为一个或多个“自然语言子查询”。
这些子查询将直接用于向量检索，请保持描述完整、语义连贯。

如果用户的查询只包含一个意图，则只返回一个子查询。
如果用户的查询包含多个歌手、多个年份、多张专辑或者多个地区等条件，请分别拆成多个子查询。

要求：
- 每个子查询必须是自然语言描述（不要关键词）
- 不要解释
- 如果只能拆出一个，就返回一个
- 输出格式：每行一个子查询（不要 JSON）(非常重要)

用户查询：{query}

子查询："""
        )
        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        text = chain.invoke(query)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines


    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """生成列表式回答"""
        if not context_docs:
            return "抱歉，没有找到相关的歌曲信息。"

        # 提取歌曲名称
        song_names = []
        for doc in context_docs:
            song_name = doc.metadata.get('title', '未知歌曲')
            if song_name not in song_names:
                song_names.append(song_name)

        # 构建简洁的列表回答
        if len(song_names) == 1:
            return f"为您推荐：{song_names[0]}"
        elif len(song_names) <= 15:
            return f"为您推荐以下歌曲：\n" + "\n".join([f"{i+1}.{name}" for i, name in enumerate(song_names)])
        else:
            return f"为您推荐以下歌曲：\n" + "\n".join([f"{i+1}.{name}" for i, name in enumerate(song_names[:5])]) + f"\n\n还有其他 {len(song_names) - 5} 首歌曲可供选择"

    def generate_basic_answer_stream(self, query:str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出
        :param query:
        :param context_docs:
        :return:
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template( """
你是一名专业的华语独立音乐歌词鉴赏助手。请根据以下歌曲歌词信息回答用户的问题。
    
用户问题：{question}
    
歌曲歌词信息：
{context}
    
请提供详细、实用的回答。如果信息不足，请如实说明。
    
回答：""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_direct_answer(self, query: str):
        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的华语独立音乐歌词鉴赏助手。请回答用户的问题。
    
用户问题：{question}
    
请提供详细、实用的回答。如果信息不足，请如实说明。
    
回答：""")

        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_direct_answer_stream(self, query: str):
        prompt = ChatPromptTemplate.from_template(
            """
你是一名专业的华语独立音乐歌词鉴赏助手。请回答用户的问题。

用户问题：{question}

请提供详细、实用的回答。如果信息不足，请如实说明。

回答：""")

        chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_compare_answer(self, query: str, docs_list: List[List[Document]]) -> str:
        combined = ""
        for idx, group in enumerate(docs_list):
            combined += f"第{idx}组歌曲信息\n"
            for doc in group:
                title = doc.metadata.get('title')
                artist = doc.metadata.get('artist')
                snippet = (doc.page_content or "")[:400].replace("\n", " ")
                combined += f"- 《{title}》 by {artist}：{snippet}...\n"
            combined += "\n"

        prompt = ChatPromptTemplate.from_template("""
你是一名专业的华语独立音乐歌词鉴赏助手。请根据以下歌曲歌词信息回答用户的问题。

用户问题：{question}

下面是多个检索结果分组，每一组代表一个对比对象。
（例如不同歌手、不同专辑、不同风格，通过文档内容结合用户问题自行判断）
{combined}

请你：
1. 给出一个结构化、清晰、有依据的比较
2. 引用文档中的具体内容支撑观点
""")

        chain = (
            {"question": RunnablePassthrough(), "combined": lambda _: combined}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def generate_compare_answer_stream(self, query: str, docs_list: List[List[Document]]) -> str:
        combined = ""
        for idx, group in enumerate(docs_list):
            combined += f"第{idx}组歌曲信息\n"
            for doc in group:
                title = doc.metadata.get('title')
                artist = doc.metadata.get('artist')
                snippet = (doc.page_content or "")[:400].replace("\n", " ")
                combined += f"- 《{title}》 by {artist}：{snippet}...\n"
            combined += "\n"

        prompt = ChatPromptTemplate.from_template("""
你是一名专业的华语独立音乐歌词鉴赏助手。请根据以下歌曲歌词信息回答用户的问题。

用户问题：{question}

下面是多个检索结果分组，每一组代表一个对比对象。
（例如不同歌手、不同专辑、不同风格，通过文档内容结合用户问题自行判断）
{combined}

请你：
1. 给出一个结构化、清晰、有依据的比较
2. 引用文档中的具体内容支撑观点
""")

        chain = (
                {"question": RunnablePassthrough(), "combined": lambda _: combined}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def extract_metadata_filters(self, query: str) -> dict:
        """使用 LLM 智能提取元数据过滤条件"""
        prompt = ChatPromptTemplate.from_template("""
    你是一个音乐元数据提取专家。请从用户问题中提取用于数据库查询的结构化过滤条件。

    可提取字段及规则：
    1. 'artist': 明确提到的歌手或乐团名称。
    2. 'album': 明确提到的专辑名称。
    3. 'title': 明确提到的歌曲名称。
    4. 'region': 提到的地区（如：台湾、中国大陆、香港）。
    5. 'year': 提到的具体年份。

    **极其重要的约束（防止过度提取）**：
    - 严禁提取描述性词汇。例如“工业城市”、“写作背景”、“作词风格”、“如何”、“意义”等词汇绝对不是歌曲或专辑名。
    - 如果不确定某个词是否为专有名词，请不要提取。
    - 只有当用户明确指代某个作品时才提取。
    - 如果没有任何明确条件，请返回空 JSON：{{}}。

    请仅返回 JSON 格式结果。

    用户问题：{question}

    JSON 输出：""")

        chain = (
                {"question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        try:
            result_text = chain.invoke(query).strip()
            # 清理可能的 Markdown 标记
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            return json.loads(result_text)
        except Exception as e:
            logger.error(f"提取过滤条件出错: {e}")
            return {}


    def generate_hyde_text(self, query: str) -> str:
        prompt = f"请针对以下主题，写一段可能出现在华语独立音乐歌词中的意象描写或相关的乐评片段：{query}"
        return self.llm.invoke(prompt).content

    def decide_retrieval_strategy(self, query: str) -> str:
        prompt = PromptTemplate(
            template="""你是一个音乐搜索意图分析专家。请判断用户查询的检索策略。

    分类准则：
    1. 'filter' (硬性过滤)：查询意图极其明确、简单，直接针对特定的歌手、专辑或歌曲。
       - 示例：“张悬的歌词有怎样的风格”、“搜一下周杰伦的青花瓷”、“陈绮贞有哪些歌”。
    2. 'hybrid' (混合检索)：查询涉及意象分析、情感解读、对比、或是稍微复杂一点的描述性提问。
       - 示例：“对比草东和万青的虚无感”、“分析万青歌词里的工业意象”、“描写失眠的歌词有哪些”。

    注意：除非查询非常简单明确，否则一律判定为 'hybrid'。

    请只返回单词 'filter' 或 'hybrid'。

    问题：{query}
    结果：""",
            input_variables=["query"],
        )
        # 这里可以使用更快的模型，比如 deepseek-chat
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()

        # 简单清洗，确保只返回两个关键词之一
        return 'filter' if 'filter' in result else 'hybrid'

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """构建上下文字符串"""
        if not docs:
            return "暂无相关歌曲信息。"

        context_parts = []
        current_length = 0

        for i,doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"「歌曲{i}」"
            if 'title' in doc.metadata:
                metadata_info += f"{doc.metadata['title']}"
            if 'artist' in doc.metadata:
                metadata_info += f"{doc.metadata['artist']}"
            if 'album' in doc.metadata:
                metadata_info += f"{doc.metadata['album']}"
            if 'year' in doc.metadata:
                metadata_info += f"{doc.metadata['year']}"
            if 'region' in doc.metadata:
                metadata_info += f"{doc.metadata['region']}"
            if 'type' in doc.metadata:
                metadata_info += f"{doc.metadata['type']}"

            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            if current_length + len(doc_text) > max_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n" + "="*50 + "\n".join(context_parts)