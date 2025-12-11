"""
用于数据准备工作
"""
import logging
import hashlib
import re
import json
from chunk import Chunk
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import uuid

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class DataPreparationModule:
    """数据准备模块"""

    CATEGORY_MAPPING = {
        '歌名' : 'title',
        '歌手' : 'artist',
        '收录专辑' : 'album',
        '发行时间' : 'year',
        '地区' : 'region',
        '类型' : 'type',
        '歌词' : 'lyrics'
    }

    def __init__(self, data_path:str, chunk_size_chars:int = 800):
        """
        初始化数据准备模块
        :param data_path:
        """
        self.data_path = Path(data_path)
        # self.documents : List[Document] = []
        # self.chunks : List[Document] = []
        # self.parent_child_map : Dict[str, str] = {}
        self.chunk_size_chars = chunk_size_chars

        self.parent_documents: List[Document] = [] #原始Markdown文档（父）
        self.chunks: List[Document] = []           #切分后的子块（用于索引）
        self.parent_child_map: Dict[str, str] = {} #child_id -> parent_id 映射


    def load_documents(self) -> List[Document]:
        """
        加载文档数据
        :return:
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"data_path {self.data_path} 不存在")
        logger.info(f"正在从{self.data_path}加载文档...")
        parents: List[Document] = []

        for md_file in self.data_path.rglob('*.md'):
            try:
                text = md_file.read_text(encoding = 'utf-8')
            except Exception as e:
                logger.warning(f"无法读取文件{md_file}:{e}")
                continue

            try:
                rel = md_file.resolve().relative_to(self.data_path.resolve()).as_posix()
            except Exception:
                rel = md_file.resolve().as_posix()
            parent_id = hashlib.md5(rel.encode("utf-8")).hexdigest()

            meta = {
                "source": str(md_file),
                "parent_id": parent_id,
                "doc_type": "parent"
            }

            doc = Document(page_content=text, metadata=meta)
            self._enhance_metadata(doc)
            parents.append(doc)

        self.parent_documents = parents
        logger.info(f"加载并增强完成，总父文档数：{len(parents)}")
        return parents

    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据
        从文件路径提取歌手名
        :param doc:
        :return:
        """
        file_path = Path(doc.metadata.get("source", ''))
        path_parts = file_path.parts

        if len(path_parts) >= 2:
            doc.metadata["artist_from_path"] = path_parts[-2]
        else:
            doc.metadata["artist_from_path"] = "未知歌手"

        #提取歌名
        doc.metadata['title_from_filename'] = file_path.stem

        text = doc.page_content or ""

        for cn_key, en_key in self.CATEGORY_MAPPING.items():
            pattern = fr"##\s*{re.escape(cn_key)}\s*\n+(.+?)(?=\n##\s*|\Z)"
            match = re.search(pattern, text, flags = re.M | re.S)
            if match:
                value = match.group(1).strip()
                doc.metadata[en_key] = value

        doc.metadata.setdefault("title", doc.metadata.get("title_from_filename", "未知标题"))
        doc.metadata.setdefault("artist", doc.metadata.get("artist_from_path", "未知歌手"))
        doc.metadata.setdefault("album", doc.metadata.get("album", "未知"))
        doc.metadata.setdefault("year", doc.metadata.get("year", "未知"))
        doc.metadata.setdefault("region", doc.metadata.get("region", "未知"))
        doc.metadata.setdefault("type", doc.metadata.get("year", "未知"))
        doc.metadata.setdefault("lyrics", doc.metadata.get("lyrics", ""))

        return doc.metadata


    #切分chunk逻辑
    def chunk_documents(self, use_header_split: bool= True) -> List[Document]:
        """
        markdown结构感知模块
        返回分块后的文档
        :return:
        """
        if not self.parent_documents:
            raise ValueError("请先调用 load_document() 再调用 chunk_documents()")

        logger.info("正在进行Markdown结构感知分块...")
        child_chunks: List[Document] = []

        for parent in self.parent_documents:
            text = parent.page_content or ""
            parent_meta = dict(parent.metadata)

            # 首先按照标题分块
            if use_header_split:
                blocks = self._split_by_header(text, header_keys = list(self.CATEGORY_MAPPING.keys()))
            else:
                blocks = [text]

            if len(blocks) <= 1:
                lyrics_text = parent.metadata.get("lyrics", "").strip()
                if lyrics_text:
                    blocks = [lyrics_text]
                else:
                    blocks = self._split_by_size(text, self.chunk_size_chars)

            for idx, block in enumerate(blocks):
                child_id = str(uuid.uuid4())
                child_meta = dict(parent.metadata)
                child_meta.update({
                    "chunk_id": child_id,
                    "parent_id": parent_meta.get("parent_id"),
                    "doc_type": "child",
                    "chunk_index": idx
                })

                child_doc = Document(page_content=block.strip(), metadata=child_meta)
                self.parent_child_map[child_id] = parent_meta.get("parent_id")
                child_chunks.append(child_doc)

        self.chunks = child_chunks
        logger.info(f"切分完成：生成 child chunk 数量：{len(child_chunks)}")
        return child_chunks

    def _split_by_header(self, text: str, header_keys: Optional[List[str]] = None) :
        """
        按照markdown的##标题切分文本
        """
        if not text:
            return []

        # 先尝试只对关心的标题进行切分
        if header_keys:
            pattern = r"(##\s*(?:{}))\s*\n(.+?)(?=\n##\s*|\Z)".format("|".join(map(re.escape, header_keys)))
            matches = re.findall(pattern, text, flags = re.M | re.S)
            if matches:
                # return [m.strip() for m in matches]
                return [content.strip() for (_, content) in matches]

        # 通用的按照任何"##"进行切分
        parts = re.split(r"\n(?=##\s+)", text, flags = re.M)
        return [p.strip() for p in parts if p.strip()]

    def _split_by_size(self, text: str, size: int) -> List[str]:
        if not text:
            return []
        text = text.strip()
        if len(text) <= size:
            return [text]
        parts = [text[i:i+size] for i in range(0, len(text), size)]
        return parts

    def _make_summary(self, text: str, max_len:int = 120) -> str:
        s = " ".join(text.splitlines())
        s = s.strip()
        return s[:max_len] + ("..." if len(s) > max_len else "")

    def _clean_chunk_text(self, text: str) -> str:
        """
        生成干净chunk文本
        :param text:
        :return:
        """
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if line.strip().startswith("##"):
                continue
            stripped = line.strip()

            if stripped == "":
                if cleaned and cleaned[-1] == "":
                    continue
                cleaned.append("")
            else:
                cleaned.append(stripped)
        return "\n".join(cleaned).strip()

    def create_index_items(self) -> List[Dict[str, Any]]:
        if not self.chunks:
            raise ValueError("请先调用 chunk_documents 生成 chunks")
        items = []
        for c in self.chunks:
            meta = dict(c.metadata)
            item = {
                "id" : meta.get("chunk_id") or str(uuid.uuid4()),
                "text": c.page_content,
                "metadata": meta
            }
            items.append(item)
        logger.info(f"create_index_items: 生成{len(items)}条索引记录")
        return items

    # 筛选接口统一实现
    def filter_by(self, key:str, value:Any) -> List[Document]:
        return [d for d in self.parent_documents if d.metadata.get(key) == value]

    def filter_documents_by_artist(self, artist: str) -> List[Document]:
        return self.filter_by("artist", artist)

    def filter_documents_by_region(self, region: str) -> List[Document]:
        return self.filter_by("region", region)

    def filter_documents_by_year(self, year: int) -> List[Document]:
        return self.filter_by("year", year)

    def filter_documents_by_title(self, title: str) -> List[Document]:
        return self.filter_by("title", title)

    def filter_documents_by_album(self, album: str) -> List[Document]:
        return self.filter_by("album", album)

    def filter_chunks_by_keyword(self, keyword:str) -> List[Document]:
        return [c for c in self.chunks if keyword in (c.page_content or "")]

    def export_metadata(self, output_path:str):
        """
        导出父文档元数据为JSON列表（便于查看）
        :param output_path:
        :return:
        """
        out = []
        for d in self.parent_documents:
            md = d.metadata
            out.append({
                "title": md.get("title"),
                "artist": md.get("artist"),
                "album": md.get("album"),
                "year": md.get("year"),
                "region": md.get("region"),
                "type": md.get("type"),
                "lyrics_length": len(md.get("lyrics", "")),
                "source": md.get("source")
            })
        Path(output_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"导出父文档元数据到{output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        返回数据集统计信息（父文档数、chunk数、每个歌手计数等等）
        :return:
        """
        total_parents = len(self.parent_documents)
        total_chunks = len(self.chunks)

        artists_count : Dict[str, int] = {}
        regions_count : Dict[str, int] = {}
        years_count : Dict[str, int] = {}
        albums_count : Dict[str, int] = {}

        for d in self.parent_documents:
            md = d.metadata
            a = md.get("artist", "未知")
            artists_count[a] = artists_count.get(a, 0) + 1
            r = md.get("region", "未知")
            regions_count[r] = regions_count.get(r, 0) + 1
            y = md.get("year", "未知")
            years_count[y] = years_count.get(y, 0) + 1
            b = md.get("album", "未知")
            albums_count[b] = albums_count.get(b, 0) + 1

        avg_chunk_size = 0
        if total_parents:
            avg_chunk_size = sum(len(c.page_content or "") for c in self.chunks) / total_chunks

        return {
            "total_parents": total_parents,
            "total_chunks": total_chunks,
            "artists_count": artists_count,
            "regions_count": regions_count,
            "years_count": years_count,
            "albums_count": albums_count,
            "avg_chunk_size_chars": avg_chunk_size,
        }

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据返回的child chunk列表，聚合对应的parent文档并按匹配次数排序（去重）
        :param child_chunks:
        :return:
        """
        parent_scores: Dict[str, int] = {}
        parent_map: Dict[str, Document] = {}
        for c in child_chunks:
            pid = c.metadata.get("parent_id")
            if not pid:
                continue
            parent_scores[pid] = parent_scores.get(pid, 0) + 1
            if pid not in parent_map:
                for p in self.parent_documents:
                    if p.metadata.get("parent_id") == pid:
                        parent_map[pid] = p
                        break

        #按score降序排序并返回父文档去重列表
        sorted_pids = sorted(parent_scores.keys(), key = lambda x: parent_scores[x], reverse= True)
        result = [parent_map[pid] for pid in sorted_pids if pid in parent_map]

        logger.info(f"从 {len(child_chunks)} 个子块聚合 {len(result)} 个父文档")
        return result


























































