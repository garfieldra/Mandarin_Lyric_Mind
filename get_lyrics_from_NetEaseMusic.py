import requests
import json
import time

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://music.163.com",
    "Host": "music.163.com"
}

def search_artist(name):
    url = f"https://music.163.com/api/search/get?s={name}&type=100"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    artists = data.get("result", {}).get("artists", [])
    if not artists:
        print("未找到该歌手")
        return None
    return artists[0]["id"]

def get_songs_by_artist(artist_id, limit=100):
    songs = []
    offset = 0
    while True:
        url = f"https://music.163.com/api/artist/songs?id={artist_id}&limit={limit}&offset={offset}"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        new_songs = data.get("songs", [])
        if not new_songs:
            break
        songs.extend(new_songs)
        offset += limit
        time.sleep(0.5)
    return songs

def get_lyric(song_id):
    url = f"https://music.163.com/api/song/lyric?os=pc&id={song_id}&lv=-1&tv=-1"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    return data.get("lrc", {}).get("lyric", "")

def main(artist_name):
    artist_id = search_artist(artist_name)
    if not artist_id:
        return

    songs = get_songs_by_artist(artist_id)
    print(f"{artist_name} 共找到 {len(songs)} 首歌曲")

    lyrics_data = []
    for song in songs:
        sid = song["id"]
        name = song["name"]
        lyric = get_lyric(sid)
        lyrics_data.append({
            "title": name,
            "artist": artist_name,
            "lyric": lyric
        })
        print(f"已抓取: {name}")
        time.sleep(0.5)

    with open(f"./data/lyrics_NetEaseMusic/{artist_name}_lyrics.json", "w", encoding="utf-8") as f:
        json.dump(lyrics_data, f, ensure_ascii=False, indent=2)
    print("全部歌词已保存！")

if __name__ == "__main__":
    main("徐佳莹")