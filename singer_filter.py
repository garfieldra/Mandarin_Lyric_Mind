import os
import json
from pathlib import Path

# =========================
# 配置
# =========================
DATA_DIR = "./data/lyrics_raw"  # 五个 json 文件所在目录
OUTPUT_FILE = "./data/lyrics_filtered/lyrics.jsonl"

#INDEPENDENT_ARTISTS = []

with open('./data/indie_artists.txt', 'r') as f:
    # for line in f:
    #     INDEPENDENT_ARTISTS.append(str(line))
    INDEPENDENT_ARTISTS = f.read().splitlines()

# 核心独立音乐歌手名单（可扩展）
# INDEPENDENT_ARTISTS = [
#     "徐佳莹", "郑宜农", "艾怡良", "陈绮贞", "宋冬野",
#     "陈粒", "郭顶", "万能青年旅店", "草东没有派对",
#     "刺猬", "旅行团", "老王乐队", "小河弯乐队",
#     "蛋堡", "陈鸿宇", "郭顶", "张悬", "陈珊妮",
#     "林生祥", "阿肆", "陈奕迅", "李荣浩", "周杰伦",
#     "生祥乐队", "声音玩具", "杨乃文", "安溥", "HUSH",
#     "曹方", "拜金小姐", "丝袜小姐", "Hello Nico", "Vast&Hazy",
#     "My little airport", "林家谦", "Carsick Cars", "伍佰", "伍佰&China Blue",
#     "王若琳", "Tizzy Bac", "陈惠婷", "熊仔", "Karencici",
#     "许钧", "何欣穗", "何韵诗", "黄小桢", "马念先",
#     "许哲佩", "9m88", "伤心欲绝", "好乐团", "持修",
#     "田馥甄", "胡德夫", "杨祖珺", "李建复", "魏如萱",
#     "孔雀眼", "卢广仲", "邵夷贝", "庸俗救星", "猛虎巧克力",
#     "李寿全", "齐豫", "MC Hotdog", "蔡蓝钦", "黄韵玲",
#     "伍佰 & China Blue", "交工乐队", "芒果跑", "苏打绿", "吴青峰",
#     "雷光夏", "旺福", "hush!", "静物乐团", "柯泯薰",
#     "大象体操", "椅子乐团", "茄子蛋", "Crispy脆乐团", "范晓萱",
#     "血肉果汁机", "裘德", "蔡健雅", "洪佩瑜", "拍谢少年",
#     # 可以继续扩充名单
# ]

# =========================
# 合并并筛选
# =========================
all_lyrics = []

# 遍历五个文件
for i in range(1, 6):
    file_path = Path(DATA_DIR) / f"lyrics{i}.json"
    print(f"读取文件: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            # 只保留歌手在独立歌手名单里的歌词
            if "singer" in item and item["singer"] in INDEPENDENT_ARTISTS:
                # 只保留必要字段
                filtered_item = {
                    "title": item.get("name", ""),
                    "artist": item.get("singer", ""),
                    "lyrics": item.get("lyric", "")
                }
                lyrics_text = "\n".join(filtered_item["lyrics"]).strip()
                if lyrics_text:  # 只有非空歌词才保留
                    filtered_item["lyrics"] = lyrics_text
                    all_lyrics.append(filtered_item)
                # 避免空歌词
                # if filtered_item["lyrics"].strip():
                #     all_lyrics.append(filtered_item)

print(f"筛选后的歌词数量: {len(all_lyrics)}")

# =========================
# 输出 JSONL 文件
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in all_lyrics:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"已保存筛选后的歌词到: {OUTPUT_FILE}")