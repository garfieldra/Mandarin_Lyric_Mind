# netease_debug.py
import requests, json, time, os, re
from pathlib import Path
from urllib.parse import quote_plus

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Referer": "https://music.163.com",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)

def save_debug(name, text):
    p = DEBUG_DIR / name
    with open(p, "w", encoding="utf-8") as f:
        f.write(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2))
    print(f"[debug] saved {p}")

def search_artist(name):
    q = quote_plus(name)
    url = f"https://music.163.com/api/search/get?s={q}&type=100"
    print("请求 artist 搜索:", url)
    resp = requests.get(url, headers=HEADERS, timeout=10)
    print("status:", resp.status_code)
    if resp.status_code != 200:
        save_debug("search_artist_status.html", resp.text[:2000])
        return None
    try:
        data = resp.json()
    except Exception as e:
        print("parse json error:", e)
        save_debug("search_artist_raw.html", resp.text[:2000])
        return None
    save_debug("search_artist.json", data)
    artists = data.get("result", {}).get("artists") or data.get("result", {}).get("artist") or []
    if not artists:
        print("search result empty")
        return None
    print("found artist entries:", len(artists))
    return artists[0].get("id")

def get_songs_by_artist(artist_id, limit=100):
    songs = []
    offset = 0
    while True:
        url = f"https://music.163.com/api/artist/songs?id={artist_id}&limit={limit}&offset={offset}"
        print("请求 artist songs:", url)
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print("status:", resp.status_code)
        if resp.status_code != 200:
            save_debug(f"artist_songs_{artist_id}_{offset}.html", resp.text[:2000])
            break
        try:
            data = resp.json()
        except Exception as e:
            print("parse json error:", e)
            save_debug(f"artist_songs_raw_{artist_id}_{offset}.html", resp.text[:2000])
            break

        new_songs = data.get("songs") or data.get("data") or []
        if not new_songs:
            # 回退：尝试从歌手页面解析（HTML），有时 songs 在页面 JS 里
            print("API returned no songs, trying page scrape fallback")
            page_url = f"https://music.163.com/artist?id={artist_id}"
            page_resp = requests.get(page_url, headers=HEADERS, timeout=10)
            save_debug(f"artist_page_{artist_id}.html", page_resp.text[:5000])
            # 尝试从页面中抽取 JSON（页面常把数据放进 <textarea> 或 JS 变量）
            m = re.search(r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;</script>", page_resp.text, re.S)
            if m:
                try:
                    doc = json.loads(m.group(1))
                    # 尝试查找 tracks/albums in doc
                    print("found INITIAL_STATE on artist page")
                    # You may need to adapt this depending on page structure
                except Exception as e:
                    print("failed parse INITIAL_STATE:", e)
            break

        songs.extend(new_songs)
        print("got", len(new_songs), "songs, total:", len(songs))
        offset += limit
        time.sleep(0.6)
    return songs

def get_lyric(song_id):
    url = f"https://music.163.com/api/song/lyric?os=pc&id={song_id}&lv=-1&tv=-1"
    print("请求 lyric:", url)
    resp = requests.get(url, headers=HEADERS, timeout=10)
    print("status:", resp.status_code)
    if resp.status_code != 200:
        save_debug(f"lyric_{song_id}.html", resp.text[:2000])
        return ""
    try:
        data = resp.json()
    except Exception as e:
        print("lyric parse error:", e)
        save_debug(f"lyric_{song_id}_raw.html", resp.text[:2000])
        return ""
    # 官方返回里歌词常在 data["lrc"]["lyric"]
    lyric = data.get("lrc", {}).get("lyric") or data.get("klyric", {}).get("lyric") or ""
    return lyric

def main(artist_name):
    aid = search_artist(artist_name)
    if not aid:
        print("未找到 artist id，退出")
        return
    songs = get_songs_by_artist(aid, limit=50)
    print("总共抓到 songs:", len(songs))
    if not songs:
        print("没有通过 API 抓到 songs，检查 debug/ 目录里的文件")
        return
    # 拉歌词示例（前 10 首）
    out = []
    for s in songs[:10]:
        sid = s.get("id")
        name = s.get("name")
        ly = get_lyric(sid)
        out.append({"id": sid, "name": name, "lyric": ly[:200]})
        time.sleep(0.5)
    save_debug(f"{artist_name}_sample_lyrics.json", out)
    print("done")

if __name__ == "__main__":
    main("徐佳莹")