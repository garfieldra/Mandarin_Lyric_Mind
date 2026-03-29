import os
import ast
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_openai import ChatOpenAI
from lyricmind.cli import LyricMindRAGSystem
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
from langchain_core.outputs import ChatResult, ChatGeneration

# 屏蔽掉一些无关紧要的连接警告，保持终端整洁
logging.getLogger("langchain_milvus").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def _normalize_verdict(obj):
    """递归把结构里所有 verdict 键的值规范为 0 或 1，避免 Pydantic 因类型（str/bool）解析失败。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "verdict":
                if v in (1, 0):
                    out[k] = int(v)
                elif v in ("1", 1.0, True):
                    out[k] = 1
                elif v in ("0", 0.0, False):
                    out[k] = 0
                else:
                    try:
                        out[k] = 1 if int(v) else 0
                    except (TypeError, ValueError):
                        out[k] = 0
            else:
                out[k] = _normalize_verdict(v)
        return out
    if isinstance(obj, list):
        return [_normalize_verdict(x) for x in obj]
    return obj


def _extract_and_fix_json(text: str) -> str:
    """从 LLM 输出中提取并规范化为合法 JSON，供 RAGAS 的 model_validate_json 解析。

    - 去掉 ```json / ``` 等 Markdown 包裹
    - 提取第一个完整的 {} 或 [] 结构
    - 若为单引号形式的“类 JSON”，尝试转为标准 JSON
    - 递归将 verdict 规范为 0/1，避免 Faithfulness/ContextPrecision 解析失败
    """
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    # 1) 去掉 markdown 代码块
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    # 2) 提取第一个完整 JSON 结构（与 ragas prompt.utils.extract_json 逻辑一致）
    left_bracket = text.find("[")
    left_brace = text.find("{")
    indices = [i for i in (left_bracket, left_brace) if i != -1]
    start = min(indices) if indices else None
    if start is None:
        return text
    open_char = text[start]
    close_char = "]" if open_char == "[" else "}"
    depth = 0
    for i, char in enumerate(text[start:], start=start):
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
        if depth == 0:
            text = text[start : i + 1]
            break
    # 3) 解析并规范化 verdict
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
    if parsed is not None:
        parsed = _normalize_verdict(parsed)
        return json.dumps(parsed, ensure_ascii=False)
    return text


def _fix_generations(generations):
    """统一处理 ChatResult.generations，把每条 message.content 做 JSON 清洗，并同步 .text。"""
    for gen_list in generations:
        for g in gen_list if isinstance(gen_list, list) else [gen_list]:
            if not isinstance(g, ChatGeneration) or not g.message.content:
                continue
            content = _extract_and_fix_json(g.message.content)
            g.message.content = content
            # RAGAS 按 generation.text 取结果，必须同步
            g.text = content


# --- 核心加固：定义 DeepSeek 专用 JSON 修复包装器 ---
class DeepSeekJSONFixLLM(ChatOpenAI):
    """自动剔除 DeepSeek 输出中的 Markdown 代码块并规范化为合法 JSON，避免 RAGAS 解析失败导致指标为 0/nan。"""

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        _fix_generations(result.generations)
        return result

    async def _agenerate(self, *args, **kwargs):
        """RAGAS 使用 agenerate_prompt -> _agenerate，必须在此同样清洗输出。"""
        result = await super()._agenerate(*args, **kwargs)
        _fix_generations(result.generations)
        return result

class RagasEvaluator:
    def __init__(self):
        # 1. 初始化你的系统
        self.rag_system = LyricMindRAGSystem()
        self.rag_system.initialize_system()
        self.embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-zh-v1.5")
        self.rag_system.build_knowledge_base()

        # 2. 配置评委 LLM (使用 DeepSeek)
        self.judge_llm = DeepSeekJSONFixLLM(
            model='deepseek-chat',
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com/v1",
            max_tokens = 8192,  # 显式调大输出上限，给评委足够的说话空间
            temperature = 0
        )

    def collect_data(self, test_cases: list):
        """运行系统并收集评估所需的四元组数据"""
        samples = []

        print(f"开始运行系统，共 {len(test_cases)} 个测试用例...")

        for case in test_cases:
            question = case["question"]
            ground_truth = case.get("ground_truth", "")

            print(f"正在测试: {question}")


            # 获取检索到的上下文 (Contexts)
            # 我们直接复用你重构后的过滤检索逻辑
            filters = self.rag_system._extract_filters_from_query(question)
            if filters:
                chunks = self.rag_system.retrieval_module.metadata_filtered_search(question, filters)
                if not chunks:
                    chunks = self.rag_system.retrieval_module.hybrid_search(question)
            else:
                chunks = self.rag_system.retrieval_module.hybrid_search(question)

            parent_docs = self.rag_system.data_module.get_parent_documents(chunks)

            if parent_docs:
                contexts = [doc.page_content for doc in parent_docs]
            else:
                contexts = [c.page_content for c in chunks]

            # 获取生成的答案 (Answer)
            answer = self.rag_system.ask_question(question)

            samples.append({
                # --- Ragas 新版 (0.2.x+) 标准字段 ---
                "user_input": question,
                "response": str(answer),
                "retrieved_contexts": contexts,
                "reference": ground_truth,  # 新版要求是字符串，不用加 []

                # --- Ragas 老版 (0.1.x) 兼容字段 ---
                "question": question,
                "answer": str(answer),
                "contexts": contexts,
                "ground_truths": [ground_truth] if ground_truth else []
            })

        return Dataset.from_pandas(pd.DataFrame(samples))

    def run(self, test_cases: list):
        """执行 Ragas 评估"""
        dataset = self.collect_data(test_cases)
        print("正在请求 DeepSeek 评委进行打分...")

        # 1. 实例化指标对象
        m_faithfulness = Faithfulness()
        m_relevancy = AnswerRelevancy()
        m_precision = ContextPrecision()
        m_recall = ContextRecall()

        metrics = [m_faithfulness, m_relevancy, m_precision, m_recall]

        for m in metrics:
            # 强制每一个指标使用你配置好的 judge_llm
            m.llm = self.judge_llm
            # 适配 DeepSeek 不支持多回复的特性
            if hasattr(m, 'n_generations'): m.n_generations = 1
            if hasattr(m, 'n'): m.n = 1

        # 使用 RunConfig 控制并发与超时，避免 DeepSeek 限流/超时
        run_config = RunConfig(max_workers=1, timeout=300)

        # 3. 执行评估
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.judge_llm,
            embeddings=self.embeddings,
            # max_workers=1 配合 DeepSeek 的频率限制，timeout 给够 Reasoning 时间
            run_config=run_config,
            raise_exceptions = False
        )

        return result


if __name__ == "__main__":
    # 临时占位符，等下一阶段我们再详细编写
    test_cases = [
        {"question": "分析万能青年旅店歌词中对“工业城市”意象的描写及其背后的情感", "ground_truth": "1. 核心意象：应提到石家庄、制药厂、锅炉、大厦、华北、八角柜台等具象符号。2. 情感基调：描写了计划经济体制瓦解后的失落、时代阵痛中的平庸生活崩溃（如此刻秦皇岛、杀死那个石家庄人）、以及对现代性文明的怀疑与疲惫。"},
        {"question": "对比草东没有派对与万能青年旅店在表达“虚无感”时的异同", "ground_truth": "1. 相同点：都表达了对当下生活状态的不满、幻灭感以及个体在庞大系统下的无力感。2. 不同点：草东的‘虚无’更侧重于个体存在的崩溃、愤怒、自嘲与代际间的绝望（如《大风吹》、《山海》）；万青的‘虚无’则植根于宏大的工业历史背景，侧重于集体记忆的消亡、时代转型期的荒凉与九十年代国企改制下岗大潮下工人阶级幻梦的破灭（如《杀死那个石家庄人》）。"},
        {"question": "《你是自由的》这首歌涉及到了哪些现实发生的社会事件背景", "ground_truth": "应提到发布时间2022年左右的COVID-19新型冠状病毒大流行时期的社会背景、2021年东京奥运会台湾公众人物在社交媒体庆祝中华台北代表队所获金牌被大陆网民出征的相关争议、2018年地平线航空Q400事件、2021年苏伊士运河阻塞事件、2021年机智号直升机在火星起飞等等相关事件。"},
        {"question": "比较张悬和陈珊妮在作词风格方面的异同", "ground_truth": "共同点：两人的词作均以独特、深刻著称，都较多地着眼于人生、社会议题、意味深长。不同点：张悬的词作往往较为诗化、散文感较强，并且并非严格押韵，歌词擅长在疏离中展现出细腻的情感流动，而陈珊妮的词作则更加华丽冷艳，带有一种冷冽的优雅与成熟的华丽感。视角独到、表达对于社会现象的犀利观察和冷嘲热讽。在结构上较为精准、常用直接且尖锐的语句揭露本质。"},
        {"question": "分析徐佳莹《心里学》专辑的作词风格", "ground_truth": "专辑《心里学》中的作词风格趋于内敛、成熟、直击心底且极具共情力，侧重于描写和表达在感情中内心的真实感受。歌词不仅描述情感，更深入剖析内心纠结与释怀，以简洁利落的语言书写深刻的成长体悟。如《言不由衷》借由「言不由衷」来写最真挚的祝福，“愿你永远安康，愿你永远懂得飞翔”。《灰色》这首歌则描述了一个白色的“自我”，然后用“那是爱过你才能成为的灰色自己”来暗示转变，最后话锋一转“与暗黑相依”越过一般的感情层面，上升到对于心中所爱追逐路途的思考。《到此为止》则呼吁消除感情世界里的拖泥带水、互相拉扯。《记得带走》则缅怀逝去的人或是流浪的小动物，侧重于对于生命的表达。《病人》控诉沉沦在感情病态里自作聪明的人。《儿歌》则引发对于人性的探讨，讲人与人之间的相处。《心里学》同名曲则把前面众生百态的心里学揉进了一首歌中，借由感情的立场表达出来。《是日救星》则正面回应网络上的各类负面讨论，使得自身的形象更加血肉丰实。"},
        {"question": "分析周杰伦《青花瓷》中中国风意象的运用", "ground_truth": "抱歉，没有找到相关的歌曲信息。请尝试其他歌曲名称或关键词。"},
        # ... 依此类推
    ]

    evaluator = RagasEvaluator()
    results = evaluator.run(test_cases)

    print("\n" + "=" * 30)
    print("📈 LyricMind RAG 评估报告")
    print("=" * 30)
    # 聚合分数（与 print(results) 一致，便于确认非 0/nan）
    print(results)
    # 若仍出现大量 nan/0，可临时设置 evaluate(..., raise_exceptions=True) 查看具体解析错误
    try:
        df = results.to_pandas()
        df.to_csv("eval_report.csv", index=False)
        print("\n✅ 详细报告已保存至 eval_report.csv")
    except Exception as e:
        logger.warning("导出 CSV 失败: %s", e)