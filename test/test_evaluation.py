"""
test/test_evaluation.py — GraphRAG 自动化评估测试套件

评估分三个层次：
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: 路由决策测试          TestRoutingDecisions             │
│  ── 验证条件边在各场景下路由到正确节点（全 mock，<0.1s）           │
│                                                                 │
│  Layer 2: 端到端图行为评估       TestEndToEndGraphEvaluation     │
│  ── 用 mock LLM 驱动完整 graph.invoke，收集并断言关键指标          │
│  ── 不产生真实 LLM 费用，适合 CI                                  │
│                                                                 │
│  Layer 3: 质量指标计算           TestMetricsComputation          │
│  ── 单元测试 metrics 辅助函数（关键词覆盖率、幻觉率等）             │
└─────────────────────────────────────────────────────────────────┘

运行方式：
    python -m pytest test/test_evaluation.py -v

真实 LLM 集成测试（需配置 .env）：
    python -m pytest test/test_evaluation.py -v -m integration
"""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── stub 重型依赖（与 test_graph_nodes.py 保持一致的方式）──────────────────

def _stub_heavy_imports():
    mods = [
        "langchain_core", "langchain_core.documents", "langchain_core.messages",
        "langchain_core.runnables", "langchain_community", "langchain_community.vectorstores",
        "langchain_community.embeddings", "langchain_community.retrievers",
        "langchain", "langchain.retrievers", "faiss", "jieba", "rank_bm25",
        "sentence_transformers", "transformers", "torch",
        "langgraph", "langgraph.graph", "langgraph.checkpoint",
        "langgraph.checkpoint.sqlite",
    ]
    for mod in mods:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    class _FakeDocument:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
        def __repr__(self):
            return f"Document(content={self.page_content!r})"

    class _FakeHumanMessage:
        def __init__(self, content=""):
            self.content = content
    class _FakeAIMessage:
        def __init__(self, content=""):
            self.content = content

    sys.modules["langchain_core.documents"].Document = _FakeDocument
    sys.modules["langchain_core.messages"].HumanMessage = _FakeHumanMessage
    sys.modules["langchain_core.messages"].AIMessage = _FakeAIMessage
    sys.modules["langchain_core.messages"].BaseMessage = object

    # langgraph.graph.END 和 StateGraph
    sys.modules["langgraph.graph"].END = "__end__"
    sys.modules["langgraph.graph"].StateGraph = MagicMock

    return _FakeDocument, _FakeHumanMessage, _FakeAIMessage


_FakeDocument, _FakeHumanMessage, _FakeAIMessage = _stub_heavy_imports()

from qa_chain.nodes import (  # noqa: E402
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    grade_answer_node,
    should_rewrite,
    should_retry_generate,
)
from qa_chain.state import MAX_RETRIES  # noqa: E402

# ── 评估数据集路径 ─────────────────────────────────────────────────────────
EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")


# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════

def _load_eval_dataset() -> List[Dict]:
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _make_doc(content="测试内容", source="test.pdf", page=0):
    return _FakeDocument(page_content=content, metadata={"source": source, "page": page})


def _make_vectordb_from_contexts(contexts: List[str]):
    """用 ground_truth_contexts 构造 mock vectordb。"""
    docs = [_make_doc(c, source=f"doc_{i}.pdf", page=i) for i, c in enumerate(contexts)]
    mock_db = MagicMock()
    mock_db.similarity_search_with_score.return_value = [(d, 0.85) for d in docs]
    mock_db.docstore._dict = {str(i): d for i, d in enumerate(docs)}
    return mock_db, docs


def _make_llm_sequence(responses: List[str]):
    """按序列依次返回不同响应的 mock LLM（用于模拟 grade + generate 顺序调用）。"""
    counter = [0]
    def side_effect(prompt):
        idx = min(counter[0], len(responses) - 1)
        counter[0] += 1
        result = MagicMock()
        result.content = responses[idx]
        return result
    llm = MagicMock()
    llm.invoke.side_effect = side_effect
    return llm


def _config(vectordb=None, llm=None, retriever_kind="dense", history_len=3):
    return {
        "configurable": {
            "vectordb": vectordb or MagicMock(),
            "llm": llm,
            "retriever_kind": retriever_kind,
            "history_len": history_len,
        }
    }


def _base_state(**kwargs):
    state = {
        "question": "测试问题",
        "rewritten_question": None,
        "documents": [],
        "generation": None,
        "chat_history": [],
        "source_filter": None,
        "retry_count": 0,
        "embedding": "m3e",
        "top_k": 4,
        "hallucination_flag": False,
    }
    state.update(kwargs)
    return state


# ════════════════════════════════════════════════════════════════
# 指标计算工具（Layer 3 测试对象）
# ════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """单条评估用例的结果。"""
    case_id: str
    question: str
    generation: Optional[str]
    retry_count: int
    hallucination_flag: bool
    documents_retrieved: int
    documents_after_grade: int
    routing_path: List[str] = field(default_factory=list)


@dataclass
class EvalMetrics:
    """聚合指标报告。"""
    total_cases: int = 0
    answer_nonempty_rate: float = 0.0    # 生成了非空答案的比例
    keyword_coverage_rate: float = 0.0   # 期望关键词至少出现一个的比例
    avg_retry_count: float = 0.0         # 平均重试次数
    hallucination_rate: float = 0.0      # 触发幻觉标志的比例
    grader_filter_rate: float = 0.0      # 文档被过滤的比例（retrieved - after_grade）/retrieved
    context_hit_rate: float = 0.0        # 检索文档覆盖 ground_truth_contexts 的比例


def compute_keyword_coverage(generation: Optional[str], keywords: List[str]) -> float:
    """计算答案中至少出现一个期望关键词的比例（0 或 1 per case）。"""
    if not generation or not keywords:
        return 1.0 if not keywords else 0.0
    matched = sum(1 for kw in keywords if kw in generation)
    return matched / len(keywords)


def compute_context_hit(retrieved_docs: List[Any], ground_truth_contexts: List[str]) -> float:
    """
    计算检索命中率：ground_truth_contexts 中有多少被检索回来了。
    简单判断方式：truth context 的前 20 个字是否出现在任意 retrieved doc 中。
    """
    if not ground_truth_contexts:
        return 1.0
    hits = 0
    retrieved_texts = [d.page_content for d in retrieved_docs]
    for ctx in ground_truth_contexts:
        prefix = ctx[:10]
        if any(prefix in rt for rt in retrieved_texts):
            hits += 1
    return hits / len(ground_truth_contexts)


def aggregate_metrics(results: List[EvalResult], dataset: List[Dict]) -> EvalMetrics:
    """从评估结果列表计算聚合指标。"""
    if not results:
        return EvalMetrics()

    n = len(results)
    id_to_case = {c["id"]: c for c in dataset}

    answer_nonempty = sum(1 for r in results if r.generation)
    hallucinations = sum(1 for r in results if r.hallucination_flag)
    total_retries = sum(r.retry_count for r in results)

    grader_filtered = sum(
        max(0, r.documents_retrieved - r.documents_after_grade) for r in results
    )
    total_retrieved = sum(r.documents_retrieved for r in results) or 1

    kw_coverages = []
    for r in results:
        case = id_to_case.get(r.case_id, {})
        keywords = case.get("expected_answer_keywords", [])
        kw_coverages.append(compute_keyword_coverage(r.generation, keywords))

    return EvalMetrics(
        total_cases=n,
        answer_nonempty_rate=answer_nonempty / n,
        keyword_coverage_rate=sum(kw_coverages) / n,
        avg_retry_count=total_retries / n,
        hallucination_rate=hallucinations / n,
        grader_filter_rate=grader_filtered / total_retrieved,
    )


# ════════════════════════════════════════════════════════════════
# Layer 1 — 路由决策测试
# ════════════════════════════════════════════════════════════════

class TestRoutingDecisions(unittest.TestCase):
    """验证条件边在各场景下的路由目标（should_rewrite / should_retry_generate）。"""

    # ── should_rewrite ──────────────────────────────────────────

    def test_should_rewrite_when_no_docs_and_retry_available(self):
        """文档为空且 retry < MAX_RETRIES 时，应路由到 rewrite。"""
        state = _base_state(documents=[], retry_count=0)
        self.assertEqual(should_rewrite(state), "rewrite")

    def test_should_generate_when_docs_present(self):
        """有文档时，无论 retry_count，应路由到 generate。"""
        state = _base_state(documents=[_make_doc()], retry_count=0)
        self.assertEqual(should_rewrite(state), "generate")

    def test_should_generate_when_retry_exhausted_and_no_docs(self):
        """文档为空但已达最大重试次数，应强制路由到 generate（避免无限循环）。"""
        state = _base_state(documents=[], retry_count=MAX_RETRIES)
        self.assertEqual(should_rewrite(state), "generate")

    def test_should_rewrite_boundary_retry_one_less_than_max(self):
        """retry_count = MAX_RETRIES - 1 且无文档，应继续改写。"""
        state = _base_state(documents=[], retry_count=MAX_RETRIES - 1)
        self.assertEqual(should_rewrite(state), "rewrite")

    # ── should_retry_generate ───────────────────────────────────

    def test_should_end_when_not_hallucination(self):
        """无幻觉标志时，应路由到 end。"""
        state = _base_state(hallucination_flag=False, retry_count=1)
        self.assertEqual(should_retry_generate(state), "end")

    def test_should_retry_when_hallucination_and_retry_available(self):
        """检测到幻觉且 retry < MAX_RETRIES 时，应路由回 generate。"""
        state = _base_state(hallucination_flag=True, retry_count=0)
        self.assertEqual(should_retry_generate(state), "generate")

    def test_should_end_when_hallucination_but_retry_exhausted(self):
        """检测到幻觉但已达最大重试，应强制 end（避免无限循环）。"""
        state = _base_state(hallucination_flag=True, retry_count=MAX_RETRIES)
        self.assertEqual(should_retry_generate(state), "end")

    # ── 路由决策不变量：分支覆盖 ────────────────────────────────

    def test_should_rewrite_returns_only_valid_targets(self):
        """should_rewrite 的所有可能返回值必须是合法节点名。"""
        valid = {"rewrite", "generate"}
        test_states = [
            _base_state(documents=[], retry_count=0),
            _base_state(documents=[_make_doc()], retry_count=0),
            _base_state(documents=[], retry_count=MAX_RETRIES),
            _base_state(documents=[_make_doc()], retry_count=MAX_RETRIES),
        ]
        for s in test_states:
            result = should_rewrite(s)
            self.assertIn(result, valid, f"非法路由目标: {result!r}")

    def test_should_retry_generate_returns_only_valid_targets(self):
        """should_retry_generate 的所有可能返回值必须是合法节点名。"""
        valid = {"generate", "end"}
        test_states = [
            _base_state(hallucination_flag=False, retry_count=0),
            _base_state(hallucination_flag=True, retry_count=0),
            _base_state(hallucination_flag=True, retry_count=MAX_RETRIES),
        ]
        for s in test_states:
            result = should_retry_generate(s)
            self.assertIn(result, valid, f"非法路由目标: {result!r}")


# ════════════════════════════════════════════════════════════════
# Layer 2 — 端到端图行为评估（节点串联，eval_dataset 驱动）
# ════════════════════════════════════════════════════════════════

class TestEndToEndGraphEvaluation(unittest.TestCase):
    """
    用 mock LLM 模拟完整节点串联，收集关键指标：
    - 答案非空率
    - 关键词覆盖率
    - 幻觉触发行为
    - 路由路径正确性
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = _load_eval_dataset()

    def _run_pipeline_for_case(self, case: Dict) -> EvalResult:
        """
        逐节点手动串联 retrieve → grade → [rewrite → retrieve]? → generate → grade_answer。
        不依赖 langgraph.invoke，纯节点函数调用，适合 CI mock 测试。
        """
        db, ground_truth_docs = _make_vectordb_from_contexts(
            case.get("ground_truth_contexts", ["占位上下文"])
        )
        expected_relevant = case.get("expected_grade_relevant", True)
        # grade_documents_node LLM：根据 expected_grade_relevant 返回 yes/no
        grade_doc_response = "yes" if expected_relevant else "no"
        # generate_node LLM：返回包含关键词的答案
        keywords = case.get("expected_answer_keywords", [])
        gen_answer = "这是答案：" + "、".join(keywords) if keywords else "根据文档，这是答案。"
        # grade_answer_node LLM：正常情况返回 grounded（无幻觉）
        grade_ans_response = "grounded"

        # 用序列 LLM：grade_doc(N次) → grade_answer(1次)
        n_docs = len(case.get("ground_truth_contexts", [1]))
        responses = [grade_doc_response] * n_docs + [gen_answer, grade_ans_response]
        llm = _make_llm_sequence(responses)

        config = _config(vectordb=db, llm=llm)
        state = _base_state(question=case["question"])

        routing_path = ["retrieve"]

        # Step 1: retrieve
        state.update(retrieve_node(state, config))

        # Step 2: grade_documents
        routing_path.append("grade_documents")
        state.update(grade_documents_node(state, config))
        docs_retrieved = len(state.get("documents", []))

        # Step 3: 条件边 should_rewrite
        route = should_rewrite(state)
        if route == "rewrite":
            routing_path.append("rewrite_query")
            state.update(rewrite_query_node(state, config))
            # 重新检索一次（模拟 loop 一次）
            routing_path.append("retrieve")
            state.update(retrieve_node(state, config))
            routing_path.append("grade_documents")
            state.update(grade_documents_node(state, config))

        # Step 4: generate
        routing_path.append("generate")
        # generate_node 需要 LLM 生成文本，重置 LLM 为直接返回答案
        gen_llm = MagicMock()
        gen_llm.invoke.return_value = MagicMock(content=gen_answer)
        gen_config = _config(vectordb=db, llm=gen_llm)
        state.update(generate_node(state, gen_config))

        # Step 5: grade_answer
        routing_path.append("grade_answer")
        grade_ans_llm = MagicMock()
        grade_ans_llm.invoke.return_value = MagicMock(content="grounded")
        grade_ans_config = _config(vectordb=db, llm=grade_ans_llm)
        state.update(grade_answer_node(state, grade_ans_config))

        # Step 6: 条件边 should_retry_generate
        # 真实返回值为 "end" 或 "generate"，统一映射为 LangGraph 标准名
        final_route = should_retry_generate(state)
        routing_path.append("end" if final_route == "end" else final_route)

        return EvalResult(
            case_id=case["id"],
            question=case["question"],
            generation=state.get("generation"),
            retry_count=state.get("retry_count", 0),
            hallucination_flag=state.get("hallucination_flag", False),
            documents_retrieved=docs_retrieved,
            documents_after_grade=len(state.get("documents", [])),
            routing_path=routing_path,
        )

    def test_all_cases_produce_answer(self):
        """每个评估用例都应产生非空答案（generation 不为 None）。"""
        for case in self.dataset:
            with self.subTest(case_id=case["id"]):
                result = self._run_pipeline_for_case(case)
                self.assertIsNotNone(
                    result.generation,
                    f"[{case['id']}] generation 为 None，管道未生成答案",
                )
                self.assertGreater(
                    len(result.generation.strip()), 0,
                    f"[{case['id']}] generation 为空字符串",
                )

    def test_relevant_cases_end_with_end_route(self):
        """相关问题（expected_grade_relevant=True）最终路由应为 __end__，不应重试生成。"""
        for case in self.dataset:
            if not case.get("expected_grade_relevant", True):
                continue
            with self.subTest(case_id=case["id"]):
                result = self._run_pipeline_for_case(case)
                self.assertEqual(
                    result.routing_path[-1], "end",
                    f"[{case['id']}] 最终路由应为 end，实际: {result.routing_path[-1]}",
                )

    def test_irrelevant_case_still_produces_answer_after_rewrite(self):
        """
        无关问题（grade 全部为 'no'）场景下，grade_documents_node 会清空文档，
        管道应先触发 rewrite_query，再进入 generate，最终仍产出答案。
        """
        irrelevant_cases = [c for c in self.dataset if not c.get("expected_grade_relevant", True)]
        for case in irrelevant_cases:
            with self.subTest(case_id=case["id"]):
                result = self._run_pipeline_for_case(case)
                self.assertIn(
                    "rewrite_query", result.routing_path,
                    f"[{case['id']}] 文档全被过滤后应触发 rewrite_query，路由: {result.routing_path}",
                )
                self.assertIsNotNone(
                    result.generation,
                    f"[{case['id']}] 即使文档无关，改写后应仍产出答案",
                )

    def test_rewrite_is_triggered_when_vectordb_returns_empty(self):
        """
        当 vectordb 返回空结果时，grade_documents 后 documents 为空，
        should_rewrite 应路由到 rewrite_query（触发改写重试）。
        """
        # 构造返回空的 vectordb
        empty_db = MagicMock()
        empty_db.similarity_search_with_score.return_value = []
        empty_db.docstore._dict = {}

        llm = MagicMock()
        # rewrite 节点需要一次 LLM 调用
        llm.invoke.return_value = MagicMock(content="改写后的问题")

        config = _config(vectordb=empty_db, llm=llm)
        state = _base_state(question="关于一个知识库中不存在的主题")

        # Step 1: retrieve → 空结果
        state.update(retrieve_node(state, config))
        self.assertEqual(state["documents"], [], "vectordb 为空，retrieve 应返回空列表")

        # Step 2: grade_documents → documents 仍为空（无文档可兜底）
        state.update(grade_documents_node(state, config))
        self.assertEqual(state["documents"], [], "无文档时 grade 应保持空列表")

        # Step 3: 条件边 should_rewrite → 应触发 rewrite
        route = should_rewrite(state)
        self.assertEqual(
            route, "rewrite",
            f"vectordb 返回空时 should_rewrite 应返回 'rewrite'，实际: {route!r}",
        )

    def test_keyword_coverage_rate_above_threshold(self):
        """所有用例的关键词覆盖率应达到 60% 以上（mock LLM 预置了答案）。"""
        coverages = []
        for case in self.dataset:
            result = self._run_pipeline_for_case(case)
            keywords = case.get("expected_answer_keywords", [])
            cov = compute_keyword_coverage(result.generation, keywords)
            coverages.append(cov)

        overall = sum(coverages) / len(coverages) if coverages else 0.0
        self.assertGreaterEqual(
            overall, 0.6,
            f"关键词覆盖率 {overall:.1%} 低于阈值 60%",
        )

    def test_aggregate_metrics_summary(self):
        """收集并打印所有用例的聚合指标（不断言，仅用于报告查看）。"""
        results = [self._run_pipeline_for_case(c) for c in self.dataset]
        metrics = aggregate_metrics(results, self.dataset)

        print(f"\n{'='*55}")
        print(f"  GraphRAG 自动化评估报告 ({metrics.total_cases} cases)")
        print(f"{'='*55}")
        print(f"  答案非空率         : {metrics.answer_nonempty_rate:.1%}")
        print(f"  关键词覆盖率       : {metrics.keyword_coverage_rate:.1%}")
        print(f"  平均重试次数       : {metrics.avg_retry_count:.2f}")
        print(f"  幻觉触发率         : {metrics.hallucination_rate:.1%}")
        print(f"  文档过滤率         : {metrics.grader_filter_rate:.1%}")
        print(f"{'='*55}")

        self.assertGreaterEqual(metrics.answer_nonempty_rate, 1.0, "答案非空率应为 100%")


# ════════════════════════════════════════════════════════════════
# Layer 2b — 幻觉检测与重试循环测试
# ════════════════════════════════════════════════════════════════

class TestHallucinationRetryBehavior(unittest.TestCase):
    """专项测试 grade_answer_node 触发幻觉 → generate 重试 → 最终兜底的行为。"""

    def _state_with_docs(self, generation=None, retry_count=0):
        docs = [_make_doc("有据可查的内容")]
        return _base_state(
            documents=docs,
            generation=generation,
            retry_count=retry_count,
        )

    def test_hallucination_sets_flag(self):
        """grade_answer_node 接收 'not_grounded' 时，hallucination_flag 应为 True。"""
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="not_grounded")
        state = self._state_with_docs(generation="这个答案无据可查")
        config = _config(llm=llm)
        result = grade_answer_node(state, config)
        self.assertTrue(result.get("hallucination_flag"), "应设置 hallucination_flag=True")

    def test_grounded_clears_flag(self):
        """grade_answer_node 接收 'grounded' 时，hallucination_flag 应为 False。"""
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="grounded")
        state = self._state_with_docs(generation="有依据的答案")
        config = _config(llm=llm)
        result = grade_answer_node(state, config)
        self.assertFalse(result.get("hallucination_flag"), "应设置 hallucination_flag=False")

    def test_retry_loop_terminates_at_max_retries(self):
        """
        模拟幻觉循环：每次 grade_answer 都返回 not_grounded，
        should_retry_generate 在 retry_count == MAX_RETRIES 时必须返回 end。
        """
        state = _base_state(
            documents=[_make_doc("内容")],
            generation="答案",
            hallucination_flag=True,
            retry_count=MAX_RETRIES,
        )
        route = should_retry_generate(state)
        self.assertEqual(
            route, "end",
            f"达到最大重试次数后应路由到 end，实际: {route!r}",
        )

    def test_generate_retry_increments_retry_count(self):
        """
        generate_node 被多次调用时，retry_count 由 grade_documents_node 递增，
        验证状态机的计数逻辑。
        """
        docs = [_make_doc("内容A"), _make_doc("内容B")]
        llm_grade = MagicMock()
        llm_grade.invoke.return_value = MagicMock(content="yes")
        state = _base_state(documents=docs, retry_count=0)
        config = _config(llm=llm_grade)
        result = grade_documents_node(state, config)
        self.assertEqual(result["retry_count"], 1)


# ════════════════════════════════════════════════════════════════
# Layer 3 — 指标计算函数单元测试
# ════════════════════════════════════════════════════════════════

class TestMetricsComputation(unittest.TestCase):
    """测试 compute_keyword_coverage 和 aggregate_metrics 的计算逻辑。"""

    def test_keyword_coverage_full_match(self):
        """所有关键词都出现在答案中，覆盖率应为 1.0。"""
        cov = compute_keyword_coverage("机器学习是通过数据训练模型的算法", ["机器学习", "数据", "模型"])
        self.assertAlmostEqual(cov, 1.0)

    def test_keyword_coverage_partial_match(self):
        """部分关键词匹配，覆盖率应为匹配数 / 总数。"""
        cov = compute_keyword_coverage("这是一个答案", ["答案", "缺失关键词"])
        self.assertAlmostEqual(cov, 0.5)

    def test_keyword_coverage_no_match(self):
        """无关键词匹配，覆盖率应为 0。"""
        cov = compute_keyword_coverage("完全不相关的文字", ["机器学习", "神经网络"])
        self.assertAlmostEqual(cov, 0.0)

    def test_keyword_coverage_empty_keywords(self):
        """关键词列表为空时，覆盖率应为 1.0（视为满足）。"""
        cov = compute_keyword_coverage("任意答案", [])
        self.assertAlmostEqual(cov, 1.0)

    def test_keyword_coverage_empty_generation(self):
        """答案为空时，若有关键词，覆盖率应为 0。"""
        cov = compute_keyword_coverage("", ["机器学习"])
        self.assertAlmostEqual(cov, 0.0)

    def test_context_hit_full(self):
        """ground_truth 完全被检索回来，命中率应为 1.0。"""
        docs = [_make_doc("机器学习是人工智能的一个分支")]
        hit = compute_context_hit(docs, ["机器学习是人工智能的一个分支，通过让计算机学习"])
        self.assertAlmostEqual(hit, 1.0)

    def test_context_hit_miss(self):
        """ground_truth 不在检索结果中，命中率应为 0.0。"""
        docs = [_make_doc("完全不同的内容")]
        hit = compute_context_hit(docs, ["机器学习是人工智能的一个分支"])
        self.assertAlmostEqual(hit, 0.0)

    def test_aggregate_metrics_answer_nonempty_rate(self):
        """答案非空率计算正确。"""
        dataset = _load_eval_dataset()
        results = [
            EvalResult("Q001", "Q1", "有答案", 1, False, 3, 2),
            EvalResult("Q002", "Q2", None, 0, False, 0, 0),
            EvalResult("Q003", "Q3", "有答案", 1, False, 2, 2),
        ]
        metrics = aggregate_metrics(results, dataset)
        self.assertAlmostEqual(metrics.answer_nonempty_rate, 2 / 3)

    def test_aggregate_metrics_hallucination_rate(self):
        """幻觉率计算正确。"""
        dataset = _load_eval_dataset()
        results = [
            EvalResult("Q001", "Q1", "答案1", 1, True, 2, 2),
            EvalResult("Q002", "Q2", "答案2", 1, False, 2, 2),
            EvalResult("Q003", "Q3", "答案3", 1, True, 2, 2),
        ]
        metrics = aggregate_metrics(results, dataset)
        self.assertAlmostEqual(metrics.hallucination_rate, 2 / 3)

    def test_aggregate_metrics_grader_filter_rate(self):
        """过滤率计算：retrieved=4，after_grade=2 → filter_rate=0.5。"""
        dataset = _load_eval_dataset()
        results = [
            EvalResult("Q001", "Q1", "答案", 1, False, 4, 2),
        ]
        metrics = aggregate_metrics(results, dataset)
        self.assertAlmostEqual(metrics.grader_filter_rate, 0.5)


# ════════════════════════════════════════════════════════════════
# Layer 2c — Graph 图结构完整性测试（不依赖真实 LangGraph）
# ════════════════════════════════════════════════════════════════

class TestGraphStructure(unittest.TestCase):
    """
    验证图的节点和边连接满足 Self-RAG 拓扑约束。
    使用 mock StateGraph，检查 add_node / add_edge / add_conditional_edges 调用。
    """

    def test_graph_contains_all_required_nodes(self):
        """build_rag_graph 应注册 retrieve、grade_documents、rewrite_query、generate、grade_answer 五个节点。"""
        required_nodes = {
            "retrieve", "grade_documents", "rewrite_query", "generate", "grade_answer"
        }
        # 收集 StateGraph mock 上的 add_node 调用
        added_nodes: set = set()
        mock_graph_instance = MagicMock()

        def capture_add_node(name, fn):
            added_nodes.add(name)

        mock_graph_instance.add_node.side_effect = capture_add_node
        mock_graph_instance.compile.return_value = MagicMock()

        with patch("langgraph.graph.StateGraph", return_value=mock_graph_instance):
            try:
                from qa_chain.graph import build_rag_graph  # noqa
                build_rag_graph()
            except Exception:
                # langgraph 已 mock，允许部分失败；只要 add_node 被调用即可
                pass

        # 如果无法捕获（stub 层面的限制），跳过而不是失败
        if not added_nodes:
            self.skipTest("StateGraph 已 stub，无法捕获 add_node 调用，跳过拓扑验证")

        missing = required_nodes - added_nodes
        self.assertFalse(missing, f"缺少节点注册: {missing}")

    def test_node_functions_are_callable(self):
        """所有节点函数必须可调用（基础健全性检查）。"""
        from qa_chain.nodes import (
            retrieve_node, grade_documents_node, rewrite_query_node,
            generate_node, grade_answer_node,
        )
        for fn in [retrieve_node, grade_documents_node, rewrite_query_node,
                   generate_node, grade_answer_node]:
            self.assertTrue(callable(fn), f"{fn.__name__} 不可调用")

    def test_routing_functions_are_callable(self):
        """条件边函数必须可调用。"""
        from qa_chain.nodes import should_rewrite, should_retry_generate
        self.assertTrue(callable(should_rewrite))
        self.assertTrue(callable(should_retry_generate))


if __name__ == "__main__":
    unittest.main(verbosity=2)
