import os
import json
import pandas as pd
from tqdm import tqdm
import logging
from lyricmind.cli import LyricMindRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_deepseek_json(content):
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return content

def extract_lyric_graph():
    # 3. 初始化现有的 RAG 系统
    # 它会自动加载你的环境变量、配置和 LLM 模块
    rag = LyricMindRAGSystem()
    rag.initialize_system()

    # 4. 加载歌词文档
    # 我们需要的是父文档（完整歌词），而不是切碎的 chunks
    rag.data_module.load_documents()
    docs = rag.data_module.parent_documents

    output_path = "graph/lyric_triplets.csv"
    os.makedirs("data", exist_ok=True)

    # 3. 断点续传逻辑：获取已处理的歌曲列表
    processed_titles = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            # 假设 'head' 列在 '演唱' 关系中或者三元组中包含了歌曲名
            # 我们通过查找所有关系中的歌曲名来确定哪些歌已经跑过了
            processed_titles = set(existing_df[existing_df['relation'] == '演唱']['tail'].unique())
            logger.info(f"检测到断点：已处理 {len(processed_titles)} 首歌曲，将跳过。")
        except Exception as e:
            logger.warning(f"读取旧数据失败，将重新开始提取: {e}")
    # 5. 定义实体提取 Prompt
    extract_prompt = """
        你是一位严谨的歌词风格分析专家。请根据提供的歌词原文，提取客观存在的关系三元组。

        ### 提取原则（违规将导致任务失败）：
        1. **禁止幻觉**：严禁编造任何歌词或意象。所有提取的词汇必须能在原文中找到完全一致的字符。
        2. **禁止篡位**：Head 必须且只能是当前歌曲名："{title}"。严禁将歌词片段、短句作为 Head。
        3. **禁止发散**：不要根据歌手风格进行“联想式背景描述”。如果没有明确的社会背景，则跳过该项。
        4. **过滤通用词**：严禁提取过于宽泛、通用、常见、不能体现独特性的词汇。
        5. **具象化测试**：如果一个物体无法被准确画出来，或者在日常生活中随处可见且无特定修饰，请过滤掉。
        6. **独特性校验**：问自己：这个实体是否能代表这首歌的独特气质？如果不能，请不要提取这一意象。
        7. **视觉签名**：仅提取能构成该歌曲独特画面的物理名词。严禁提取任何出现在 50% 以上歌词中的通用词。
        4. **宁缺毋滥原则**：如果某项分类下没有【非常独特、非通用】的词汇，请返回空列表 []。
            - 严禁提取任何出现在 80% 流行歌里的词（如：爱、心、风、雨、梦、世界、天）。
            - 哪怕整首歌一个意象都提不出来，也比提取出一个“风”要强。

        ### 提取维度：
        - **包含意象**：提取歌词中出现的重要名词，要能提现独特性，不要提取过于宽泛、通用、常见的词汇。
        - **核心情感**：提取凝练歌词中的具有文学深度的、能够代表整首歌的、较为独特的情绪基调。提取原则：严禁直接搬运歌词里的动词或短句，使用2-4字的专业的学术性或文学性词汇概括整首歌的感情色彩。
        - **风格标签**：基于原文用词特征、遣词造句风格、写作手法等提取相应的作词标签。
        - **歌曲主题**：识别歌曲所属的主题。请从整段歌词中进行分析，分析歌曲的写作主题（例如爱情、亲情、怀旧、社会议题等等，例子仅供参考，一定不要仅限于此，如果不符合所给的所有例子请大胆地自己创造词汇进行描述，但一定要遵循这种简单、笼统的概括理念，便于进行归类），不要太细分、太具体，而是就歌曲主题做一个简单分类，越简单、越通俗越好，比如所有主题跟爱情有关的只写爱情，字数控制在4个字以下。
        

        ### 预期输出格式（严格 JSON）：
        [
          {{"head": "{title}", "relation": "包含意象", "tail": ["词1", "词2"]}},
          {{"head": "{title}", "relation": "核心情感", "tail": ["情感1", "情感2"]}},
          {{"head": "{title}", "relation": "风格标签", "tail": ["标签1"]}}
          {{"head": "{title}", "relation": "歌曲主题", "tail": ["具体主题类别"]}}
        ]

        ### 待分析歌词原文：
        {content}
        """

    print(f"准备从 {len(docs)} 首歌曲中提取图谱三元组...")

    # 6. 循环遍历并调用 LLM
    # 建议先处理前 30 首进行测试
    for doc in tqdm(docs):
        title = doc.metadata.get('title', '未知歌曲')
        artist = doc.metadata.get('artist', '未知歌手')
        # 跳过已处理的歌曲
        if title in processed_titles:
            continue
        # 截取前 1000 字，避免上下文超长
        content = doc.page_content[:1500]

        try:
            # 使用你 generation 模块里的 LLM 实例
            response = rag.generation_module.llm.invoke(
                extract_prompt.format(title=title, content=content)
            )

            # 解析 JSON 结果
            json_str = clean_deepseek_json(response.content)
            triplets = json.loads(json_str)

            # 自动添加“歌手-演唱-歌曲”这一层硬关系
            triplets.append({"head": artist, "relation": "演唱", "tail": title})

            # 6. 实时追加保存到 CSV
            new_data = pd.DataFrame(triplets)
            # 如果文件不存在，写入表头；如果存在，不写表头并追加
            new_data.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path),
                            encoding='utf-8-sig')

            # 更新已处理集合
            processed_titles.add(title)

        except Exception as e:
            logger.error(f"处理歌曲《{title}》时发生错误: {e}")

    # # 7. 将结果保存到根目录下的 data 文件夹
    # output_path = "data/lyric_triplets.csv"
    # os.makedirs("data", exist_ok=True)
    #
    # df = pd.DataFrame(all_triplets)
    # df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n提取完成！文件保存在: {output_path}")


if __name__ == "__main__":
    extract_lyric_graph()

