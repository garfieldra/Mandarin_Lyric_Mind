import pandas as pd
import ast
import logging
from neo4j import GraphDatabase
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphConstruction:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def _parse_tail(self, tail_str):
        try:
            data = ast.literal_eval(tail_str)
            return data if isinstance(data, list) else [str(data)]
        except Exception:
            return [str(tail_str)]

    def setup_constraints(self):
        """预先创建唯一约束，这是工业级图数据库的标准操作"""
        with self.driver.session() as session:
            labels = ["Artist", "Song", "Imagery", "Emotion", "Style", "Theme", "Entity"]
            for label in labels:
                # 注意：不同版本的 Neo4j 语法略有差异，这里使用较通用的语法
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")
            logger.info("数据库唯一约束已就绪")

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("🗑️ 数据已清空")

    def store_triplets(self, csv_path="graph/lyric_triplets.csv"):
        df = pd.read_csv(csv_path)
        logger.info(f"开始入库，共 {len(df)} 条记录...")

        with self.driver.session() as session:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                head = str(row['head']).strip()
                relation = str(row['relation']).strip()
                tails = self._parse_tail(row['tail'])

                for tail in tails:
                    tail = str(tail).strip()
                    if not tail: continue
                    self._execute_merge(session, head, relation, tail)

        logger.info("Neo4j 数据入库完成")

    def _execute_merge(self, session, head, relation, tail):
        """
        优化后的动态标签逻辑：
        根据关系类型自动决定 Head 和 Tail 的标签
        """
        # 1. 确定关系映射和节点标签
        rel_config = {
            "演唱": ("Artist", "SINGS", "Song"),
            "包含意象": ("Song", "HAS_IMAGERY", "Imagery"),
            "核心情感": ("Song", "HAS_EMOTION", "Emotion"),
            "风格标签": ("Song", "HAS_STYLE", "Style"),
            "歌曲主题": ("Song", "HAS_THEME", "Theme")
        }

        # 如果匹配不到，则使用通用标签
        head_label, cypher_rel, tail_label = rel_config.get(relation, ("Song", "RELATED_TO", "Entity"))

        # 2. 执行参数化 Cypher 语句
        # 使用参数化查询（$head, $tail）而非字符串拼接，防止注入并提升性能
        query = f"""
        MERGE (h:{head_label} {{name: $head}})
        MERGE (t:{tail_label} {{name: $tail}})
        MERGE (h)-[:{cypher_rel}]->(t)
        """
        session.run(query, head=head, tail=tail)


if __name__ == "__main__":
    storer = GraphConstruction(password="password123")

    # 初始化步骤
    storer.clear_database()
    storer.setup_constraints()  # 新增：创建索引和约束

    # 执行入库
    storer.store_triplets()
    storer.close()