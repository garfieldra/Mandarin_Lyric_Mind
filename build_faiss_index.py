from email import encoders

import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception as e:
    raise RuntimeError(
        "导入faiss失败\n错误信息" + str(e)
    )

INPUT_PKL = Path("data/lyrics_filtered/lyrics_with_embeddings.pkl")
INDEX_OUT = Path("data/indices/faiss_index.pkl")   #faiss索引
META_OUT = Path("data/indices/meta.pkl")  #原始记录,保存与向量一一对应的 metadata 列表

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)  #确保输出目录存在

#以pandas DataFrame存储pkl
print("正在加载embeddings数据")
df = pd.read_pickle(INPUT_PKL)
print("总条目数：", len(df))

print("准备向量矩阵")
emb_list = df["embedding"].tolist()
emb_mat = np.array(emb_list, dtype="float32")
N, D = emb_mat.shape
print(f"向量维度：{D}, 向量数量：{N}")

# -----------------------
# 2) 归一化（L2 norm）以使用内积实现 cosine
# -----------------------
# 为什么：FAISS 的 IndexFlatIP 使用内积( dot )检索。
# 要用内积得到 cosine，相当于先对每个向量做 L2 归一化 (norm=1)。
# 说明：FAISS 的 IndexFlatIP 做内积（dot-product）检索。要用内积得到 cosine，相当于
#       先对每个向量做 L2 归一化（每个向量长度 = 1），然后用内积即为 cosine 值。
print("进行 L2 的归一化")

norms = np.linalg.norm(emb_mat, axis=1, keepdims=True) #计算每一行向量的L2范数
norms[norms == 0] = 1e-12   #防止除以0
emb_mat = emb_mat / norms    # 每行除以对应范数 -> 归一化向量

# -----------------------
# 3) 创建 FAISS 索引并添加向量
# -----------------------
# 使用 IndexFlatIP（精确内积索引）——简单且适合本地数据规模（1w 条）
print("创建 FAISS 索引（IndexFlatIP）并添加向量")
index = faiss.IndexFlatL2(D)        # 用内积索引（IndexFlatIP），D 是向量维度
index.add(emb_mat)                  # 添加所有向量到索引（批量加入）
print("索引已包含向量数：", index.ntotal)

# -----------------------
# 4) 将索引和 metadata 持久化
# -----------------------
print("保存索引到磁盘：", INDEX_OUT)
faiss.write_index(index, str(INDEX_OUT))    #将索引序列化写入文件

# metadata 保存：只保存能映射回原始文本的必要字段，节省空间
meta = df[["title", "artist", "lyrics"]].to_dict(orient="records")
with open(META_OUT, "wb") as f:
    pickle.dump(meta, f)
print("保存metadata到:", META_OUT)

# -----------------------
# 5) 示例：如何用查询文本做检索
# -----------------------
print("\n==== 示例查询 ====")
print("(将使用本地的 sentence-transformers 模型将query转成embedding，然后检索top-k)")

# 加载用于查询的 encoder（与构建时模型应尽量一致）
encoder = SentenceTransformer(MODEL_NAME)

# 搜索函数：把 query 文本转 embedding、归一化，然后在 faiss 中检索 top_k
def search(query_text, top_k=5):
    """
    # 将 query 编码为向量，注意 convert_to_numpy=True 可直接得到 numpy 数组
    q_emb = encoder.encode([query_text], convert_to_numpy=True, show_progress_bar=True).astype("float32")

    # 归一化 query 向量（与构建索引时相同的归一化方式）
    q_norm = np.linalg.norm(q_emb, axis=1)
    if q_norm == 0:
        q_norm = 1e-12
    q_emb = q_emb / q_norm

    # 用 faiss 搜索（返回 top_k 索引和相似度分数）（index.search 接受 shape (n_queries, D) 的数组）
    D_scores, I = index.search(np.expand_dims(q_emb, 0), top_k)
    indices = I[0].tolist() # 检索到的索引位置
    scores = D_scores[0].tolist()   # 对应相似度分数（内积，即 cosine）

    results = []
    for idx, score in zip(indices, scores):
        meta_item = meta[idx]   # 用 metadata 列表映射回原始记录
        # 选取歌词片段作为摘要（前 300 字）
        snippet = meta_item["lyrics"][:300].replace("\n", " ")
        results.append({
            "rank": len(results)+1,
            "index": idx,
            "score": float(score),
            "title": meta_item["title"],
            "artist": meta_item["artist"],
            "snippet": snippet
        })
    return results
    """
    """
        更健壮的查询函数：
        - 确保 encoder 输出被正规化为 numpy.float32 的 1D 向量
        - reshape 为 (1, D) 传给 faiss
        - 归一化（L2）与索引构建时一致
    """
    raw = encoder.encode([query_text], convert_to_numpy=True, show_progress_bar=False)

    q_arr = np.asarray(raw, dtype="float32")

    if q_arr.ndim == 2 and q_arr.shape[0] == 1:
        q_vec = q_arr[0]
    elif q_arr.ndim == 1:
        q_vec = q_arr
    else:
        q_vec = q_arr.reshape(-1)
        if q_vec.size != emb_mat.shape[1]:
            raise ValueError(f"编码后向量维度 {q_vec.size} 与索引维度 {emb_mat.shape[1]} 不匹配。"
                             "请检查编码器和构建索引时使用的模型是否一致。")

    # 确保 dtype 为 float32 并为 C 连续
    q_vec = np.asarray(q_vec, dtype="float32", order="C")

    q_vec = q_vec.reshape(1, -1)

    q_norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1e-12
    q_vec = q_vec / q_norms

    D_scores, I = index.search(q_vec, top_k)
    indices = I[0].tolist()
    scores = D_scores[0].tolist()

    results = []
    for idx, score in zip(indices, scores):
        meta_item = meta[idx]  # 用 metadata 列表映射回原始记录
        # 选取歌词片段作为摘要（前 300 字）
        snippet = meta_item["lyrics"][:300].replace("\n", " ")
        results.append({
            "rank": len(results) + 1,
            "index": idx,
            "score": float(score),
            "title": meta_item["title"],
            "artist": meta_item["artist"],
            "snippet": snippet
        })
    return results

example_q = "张悬的歌词"
res = search(example_q, top_k=5)
for r in res:
    print(f"\nRank {r['index']} score={r['score']:.4f} {r['artist']} - {r['title']}")
    print(r["snippet"][:400] + "...")



