"""
流程如下：
- 加载faiss索引和metadata
- 使用与构建索引相同的encoder将query编码
- 在FAISS中检索top_k片段
- 把片段作为context拼到prompt中，并且调用DeepSeek生成答案
"""

import os
import pickle
import json
import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from openai import OpenAI

try:
    import faiss
except Exception as e:
    faiss = None
    print("faiss未正确导入")

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")

if not (DEEPSEEK_API_KEY and DEEPSEEK_API_BASE):
    raise RuntimeError("deepseek api未正确配置，请重试")

FAISS_INDEX_PATH = Path("data/indices/faiss_index.pkl")
META_PATH = Path("data/indices/meta.pkl")

ENCODER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K =5
SNIPPET_CHARS = 500
MAX_CONTEXT_TOKENS = 2200
LLM_MODEL = "deepseek-chat"

if faiss is None:
    raise RuntimeError("faiss模块未正确加载，请先安装相关模块或改用chromadb实现")

if not FAISS_INDEX_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError("索引文件不存在，请检查索引文件路径配置")

index = faiss.read_index(str(FAISS_INDEX_PATH))
print("已加载FAISS索引，索引向量总数为：", index.ntotal)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
print("已加载元数据metadata，记录数为：", len(meta))

print(f"加载encoder模型：{ENCODER_MODEL}...")
encoder = SentenceTransformer(ENCODER_MODEL)
print("encoder加载完成。")

def retrieve(query: str, top_k: int = TOP_K):
    """
    输入用户query，返回top_k个检索结果
    :param query:
    :param top_k:
    :return:
    """
    #q_emb = encoder.encode([query], convert_to_tensor=True)[0].astype("float32")
    q_emb = encoder.encode([query], convert_to_tensor=True)[0].cpu().numpy().astype("float32")

    #L2归一化
    q_norm = np.linalg.norm(q_emb)
    if(q_norm == 0):
        q_norm = 1e-12
    q_emb = q_emb / q_norm
    q_emb = q_emb.reshape(1, -1)

    # faiss搜索
    D_scores, I = index.search(q_emb, TOP_K)
    indices = I[0].tolist()
    scores = D_scores[0].tolist()

    results = []
    for idx, score in zip(indices, scores):
        record = meta[idx]
        lyrics = record.get("lyrics", "")
        snippet = lyrics.replace("\n", " ")[:SNIPPET_CHARS]
        results.append({
            "index": idx,
            "score": float(score),
            "title": record.get("title", ""),
            "artist": record.get("artist", ""),
            "snippet": snippet
        })
    return results

def build_messages(query: str, retrieved_results):
    """
    构建chat messages（符合OpenAI chat API 的 messages 列表）
    :param query:
    :param retrieved_results:
    :return:
    """
    system_prompt = (
        "你是一个擅长华语歌词分析的AI助手。"
        "当回答问题时，请尽量使用下面提供的歌词片段（Context），"
        "如果上下文中无法得出结论，请明确说明不能确定。"
    )

    context_blocks = []
    for i,r in enumerate(retrieved_results, start = 1):
        header = f"[片段{i}] {r['title']} - {r['artist']} (相似度={r['score']:.3f})"
        context_blocks.append(header + "\n" + r["snippet"])

    context_text = "\n\n".join(context_blocks)

    #裁剪context_text以防超过限制
    if len(context_text) > (MAX_CONTEXT_TOKENS * 2):
        context_text = context_text[:MAX_CONTEXT_TOKENS * 2]

    user_prompt = (
        f"检索到的歌词片段如下：\n\n{context_text}\n\n"
        f"问题：{query}\n\n"
        "请基于以上歌词片段回答。若片段中没有足够信息，请直接说明“无法从提供的片段中判断”。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE
)

def call_deepseek(messages, max_tokens = 512, temperature = 1.0):
    """
    调用DeepSeek chat completions
    返回：assistant输出的文本内容
    :param messages:
    :param max_tokens:
    :param temperature:
    :return:
    """
    try:
        resp = client.chat.completions.create(
            model = LLM_MODEL,
            messages = messages,
            max_tokens = max_tokens,
            temperature = temperature,
            stream = False
        )
        # DeepSeek/OpenAI 返回的结构通常是 resp.choices[0].message.content
        content = resp.choices[0].message.content
        return content
    except Exception as e:
        print("调用DeepSeek API失败：",e)
        return f"调用模型失败 {e}"

def rag_respond(query):
    #1 检索
    retrieved = retrieve(query, top_k=TOP_K)
    print(f"检索到{len(retrieved)}条片段")

    #2 构造 messages
    messages = build_messages(query, retrieved)

    #3 调用deepseek API并返回结果
    answer = call_deepseek(messages)

    return answer, retrieved

if __name__ == "__main__":
    print("LyricMind服务启动，请输入您的问题（输入q或quit或exit退出）：")
    while True:
        q = input("\n请输入您的问题：").strip() #去除首尾空格
        if q.lower() in ["quit", "q", "exit"]:
            print("退出")
            break
        if not q:
            print("请输入一个问题")
            continue

        answer, retrieved = rag_respond(q)
        print("\n==== DeepSeek 模型回答 ====")
        print(answer)
        print("\n==== 检索到的歌词片段 ====")
        for i,r in enumerate(retrieved, start = 1):
            print(f"{i}. {r['artist']} - {r['title']} (score={r['score']:.3f})")
            print(r["snippet"][:400])
            print("----------")








