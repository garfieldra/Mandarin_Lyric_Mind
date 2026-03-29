from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LyricGraphSearcher:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_song_full_context(self, song_name):
        query = """
                MATCH (s:Song {name: $name})-[r]->(t)
                RETURN type(r) AS relation, t.name AS target
                """
        with self.driver.session() as session:
            result = session.run(query, name=song_name)
            # 格式化成自然语言，方便 LLM 阅读
            knowledges = []
            for record in result:
                rel = record["relation"]
                target = record["target"]

                # 语义化转换
                rel_map = {
                    "HAS_IMAGERY": "包含意象",
                    "HAS_EMOTION": "情绪基调",
                    "HAS_STYLE": "写作风格",
                    "HAS_THEME": "探讨主题"
                }
                knowledges.append(f"《{song_name}》{rel_map.get(rel, '关联')}：{target}")

            return "；".join(knowledges) if knowledges else f"图谱中暂无《{song_name}》的深度信息。"

    def find_related_songs_by_common_nodes(self, song_name, limit=5):
        """多跳查询：寻找与该歌曲有共同意象或母题的其他歌曲"""
        query = """
                MATCH (s1:Song {name: $name})-[r1]->(node)<-[r2]-(s2:Song)
                WHERE s1 <> s2 AND type(r1) = type(r2)
                RETURN s2.name AS related_song, 
                       node.name AS common_element, 
                       type(r1) AS relation_type
                LIMIT $limit
                """
        with self.driver.session() as session:
            result = session.run(query, name=song_name, limit=limit)
            related_info = []
            for record in result:
                related_info.append({
                    "song": record["related_song"],
                    "element": record["common_element"],
                    "type": record["relation_type"]
                })
            return related_info

    def search_as_context(self, song_name):
        """将检索结果封装成 Prompt 可以直接使用的 Context 字符串"""
        # 1. 获取基础背景
        base_info = self.get_song_full_context(song_name)

        # 2. 获取关联推荐
        related = self.find_related_songs_by_common_nodes(song_name, limit=3)
        related_str = ""
        if related:
            related_list = [f"《{r['song']}》（共同点：{r['element']}）" for r in related]
            related_str = "\n关联作品参考：" + "、".join(related_list)

        # 3. 组装
        context = f"【知识图谱增强信息】\n基础背景：{base_info}{related_str}"
        return context

    def search_songs_by_attributes(self, artist=None, imagery=None, style=None, theme=None, emotion=None, limit=10):
        conditions = []
        params = {"limit": limit}

        # 1. 歌手维度的特殊处理 (Artist 节点 -> SINGS 关系 -> Song 节点)
        match_clause = "MATCH (s:Song)"
        if artist:
            # 如果有歌手限制，我们需要从 Artist 节点开始匹配
            match_clause = "MATCH (a:Artist {name: $artist})-[:SINGS]->(s:Song)"
            params["artist"] = artist

        if imagery:
            conditions.append("(s)-[:HAS_IMAGERY]->(:Imagery {name: $imagery})")
            params["imagery"] = imagery
        if style:
            conditions.append("(s)-[:HAS_STYLE]->(:Style {name: $style})")
            params["style"] = style
        if theme:
            conditions.append("(s)-[:HAS_THEME]->(:Theme {name: $theme})")
            params["theme"] = theme
        if emotion:
            conditions.append("(s)-[:HAS_EMOTION]->(:Emotion {name: $emotion})")
            params["emotion"] = emotion
        if not conditions and not artist:
            return []

        # 3. 动态构建完整的查询语句
        where_clause = ""
        if conditions:
            # 确保 AND 前后有空格
            where_clause = " WHERE " + " AND ".join(conditions)

        query = f"""
        {match_clause}
        {where_clause}
        RETURN s.name AS song_name
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [record["song_name"] for record in result]

    def get_all_entities_by_label(self, label: str) -> list:
        """
        获取指定标签下的所有实体名称（用于构建对齐词汇表）
        """
        with self.driver.session() as session:
            # Cypher：查找该标签下的所有节点，并返回去重后的 name
            query = f"MATCH (n:{label}) RETURN DISTINCT n.name AS name"
            result = session.run(query)
            return [record["name"] for record in result if record["name"]]


