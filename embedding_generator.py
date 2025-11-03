import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_FILE = "./data/lyrics_filtered/lyrics.jsonl"
#OUTPUT_FILE = "./data/lyrics_filtered/lyrics_with_embeddings.pkl"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

OUTPUT_FILE = "./data/lyrics_filtered/lyrics_with_embeddings.pkl"

print(f"加载模型 {MODEL_NAME} 中\n")
model = SentenceTransformer(MODEL_NAME)
print("模型加载完成\n")

print(f"读取歌词文件：{INPUT_FILE}\n")
records = []
with open(INPUT_FILE, 'r', encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("lyrics", "").strip():
                records.append(data)
        except Exception as e:
            print(f"跳过格式错误的一行: {e}\n")

print(f"共读取到{len(records)}条歌词")

def combine_fields(r):
    artist = r.get("artist", "").strip()
    title = r.get("title", "").strip()
    lyrics = r.get("lyrics", "").strip()

    return f"歌名:{title}; 歌手:{artist}; 歌词内容:{lyrics}"

combine_texts = [combine_fields(r) for r in records]

# texts = [
#     f"{r.get('artist', '')} - {r.get('title', '')}\n{r.get('lyrics', '')}"
#     for r in records
# ]

print("开始生成向量\n")
embeddings = model.encode(combine_texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

print(f"保存到{OUTPUT_FILE}\n")
df = pd.DataFrame(records)
df["combined_text"] = combine_texts
df["embedding"] = embeddings.tolist()
df.to_pickle(OUTPUT_FILE)

print("向量化完成，输出文件包含每首歌的文本及embedding语义向量")











