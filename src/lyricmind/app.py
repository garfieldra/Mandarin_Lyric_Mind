"""FastAPI 服务入口（Web API）"""
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from lyricmind.cli import LyricMindRAGSystem

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    query: str
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str

class ArtistSearchRequest(BaseModel):
    artist: str
    query: Optional[str] = ""

class ArtistSearchResponse(BaseModel):
    songs: list[str]

app = FastAPI(
    title="LyricMind API",
    description="独立音乐文本与知识图谱混合检索系统 RESTful API",
    version="1.0"
)

rag_system = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    logger.info("正在启动Fast API服务并初始化LyricMind系统")
    try:
        rag_system = LyricMindRAGSystem()
        rag_system.initialize_system()
        rag_system.build_knowledge_base()
        logger.info("知识库加载完成，API准备就绪")
    except Exception as e:
        logger.error(f"系统初始化失败：{e}")
        raise e

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统初始化中，请稍后再试")

    try:
        answer = rag_system.ask_question(request.query, stream=False)
        return ChatResponse(answer=answer)
    except Exception as e:
        logger.error(f"问答接口处理失败：{e}")
        raise HTTPException(status_code = 500, detail = str(e))


@app.post("/api/artist/search", response_model=ArtistSearchResponse)
async def artist_search_endpoint(request: ArtistSearchRequest):
    """
    查找某个歌手的特定歌曲
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统初始化中...")

    try:
        # 调用 cli.py 里的 search_by_artist 方法
        songs = rag_system.search_by_artist(artist=request.artist, query=request.query)
        return ArtistSearchResponse(songs=songs)
    except Exception as e:
        logger.error(f"歌手搜索接口处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    用于 Kubernetes 存活探针的健康检查接口
    """
    if rag_system:
         return {"status": "ok", "message": "LyricMind RAG Engine is running."}
    else:
         raise HTTPException(status_code=503, detail="Starting up")