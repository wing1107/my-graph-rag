"""
test/test_graph_nodes.py — LangGraph 节点单元测试

设计原则：
- 不依赖 langchain / faiss / huggingface —— 全部通过 mock 注入
- 每个测试描述"应该成立的不变量"，而不是"调用了什么函数"
- 运行时间 < 2 秒，可放进 pre-commit hook / CI

运行方式：
    python -m pytest test/test_graph_nodes.py -v
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── 在 import 节点模块前，先 stub 掉重型依赖 ──────────────────────────────

def _stub_heavy_imports():
    """把 langchain / faiss 等全部 stub 为轻量 Mock，避免 ImportError。"""
    mods = [
        "langchain_core", "langchain_core.documents", "langchain_core.messages",
        "langchain_core.runnables", "langchain_community", "langchain_community.vectorstores",
        "langchain_community.embeddings", "langchain_community.retrievers",
        "langchain", "langchain.retrievers", "faiss", "jieba", "rank_bm25",
        "sentence_transformers", "transformers", "torch",
    ]
    for mod in mods:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # langchain_core.documents.Document
    class _FakeDocument:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    sys.modules["langchain_core.documents"].Document = _FakeDocument

    # langchain_core.messages
    class _FakeHumanMessage:
        def __init__(self, content=""):
            self.content = content
    class _FakeAIMessage:
        def __init__(self, content=""):
            self.content = content
    sys.modules["langchain_core.messages"].HumanMessage = _FakeHumanMessage
    sys.modules["langchain_core.messages"].AIMessage = _FakeAIMessage
    sys.modules["langchain_core.messages"].BaseMessage = object

    return _FakeDocument, _FakeHumanMessage, _FakeAIMessage


_FakeDocument, _FakeHumanMessage, _FakeAIMessage = _stub_heavy_imports()

# 现在可以安全 import 节点模块
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


# ════════════════════════════════════════════════════════════════
# 测试工具
# ════════════════════════════════════════════════════════════════

def _make_doc(content="测试内容", source="test.pdf", page=0):
    return _FakeDocument(page_content=content, metadata={"source": source, "page": page})


def _make_vectordb(docs=None):
    """构造一个 mock vectordb，similarity_search_with_score 返回 [(doc, score), ...]。"""
    if docs is None:
        docs = [_make_doc("内容A"), _make_doc("内容B")]
    mock_db = MagicMock()
    mock_db.similarity_search_with_score.return_value = [(d, 0.8) for d in docs]
    # docstore（供 BM25 使用）
    mock_db.docstore._dict = {str(i): d for i, d in enumerate(docs)}
    return mock_db


def _make_llm(response: str = "yes"):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=response)
    return mock_llm


def _config(vectordb=None, llm=None, retriever_kind="dense", history_len=3):
    return {
        "configurable": {
            "vectordb": vectordb or _make_vectordb(),
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
# retrieve_node 测试
# ════════════════════════════════════════════════════════════════

class TestRetrieveNode(unittest.TestCase):

    def test_returns_documents_from_vectordb(self):
        """dense 模式下，retrieve_node 应该返回 vectordb 召回的文档。"""
        docs = [_make_doc("机器学习"), _make_doc("深度学习")]
        db = _make_vectordb(docs)
        state = _base_state(question="什么是机器学习", top_k=2)
        result = retrieve_node(state, _config(vectordb=db))
        self.assertEqual(len(result["documents"]), 2)

    def test_uses_rewritten_question_when_present(self):
        """rewritten_question 不为 None 时，应该用改写后的问题检索。"""
        db = _make_vectordb()
        state = _base_state(
            question="原始问题",
            rewritten_question="改写后的问题",
            top_k=2,
        )
        retrieve_node(state, _config(vectordb=db))
        # 验证 similarity_search_with_score 被调用（不关心 call 参数顺序，只关心调用了）
        db.similarity_search_with_score.assert_called_once()
        call_args = db.similarity_search_with_score.call_args
        self.assertIn("改写后的问题", str(call_args))

    def test_empty_result_when_no_docs(self):
        """向量库为空时，retrieve_node 应返回空文档列表。"""
        db = _make_vectordb(docs=[])
        db.similarity_search_with_score.return_value = []
        state = _base_state(top_k=4)
        result = retrieve_node(state, _config(vectordb=db))
        self.assertEqual(result["documents"], [])


# ════════════════════════════════════════════════════════════════
# grade_documents_node 测试
# ════════════════════════════════════════════════════════════════

class TestGradeDocumentsNode(unittest.TestCase):

    def test_relevant_docs_are_kept(self):
        """LLM 返回 'yes' 时，文档应被保留。"""
        docs = [_make_doc("相关内容")]
        llm = _make_llm("yes")
        state = _base_state(documents=docs, retry_count=0)
        result = grade_documents_node(state, _config(llm=llm))
        self.assertEqual(len(result["documents"]), 1)

    def test_irrelevant_docs_are_filtered(self):
        """LLM 返回 'no' 时，文档应被全部过滤，documents 清空为 []（触发 should_rewrite）。"""
        docs = [_make_doc("无关内容")]
        llm = _make_llm("no")
        state = _base_state(documents=docs, retry_count=0)
        result = grade_documents_node(state, _config(llm=llm))
        # 修复后：全部过滤时清空，不保留兜底 doc，让 should_rewrite 感知并触发改写
        self.assertEqual(len(result["documents"]), 0)

    def test_retry_count_increments(self):
        """grade_documents_node 执行后，retry_count 应加 1。"""
        docs = [_make_doc()]
        llm = _make_llm("yes")
        state = _base_state(documents=docs, retry_count=0)
        result = grade_documents_node(state, _config(llm=llm))
        self.assertEqual(result["retry_count"], 1)

    def test_no_llm_returns_original_docs(self):
        """未注入 LLM 时，应跳过评分，原样返回文档。"""
        docs = [_make_doc("内容")]
        state = _base_state(documents=docs, retry_count=0)
        result = grade_documents_node(state, _config(llm=None))
        self.assertEqual(len(result["documents"]), 1)

    def test_multiple_docs_partial_filter(self):
        """多文档时，only 'yes' 的文档被保留。"""
        docs = [_make_doc("相关"), _make_doc("无关")]
        call_count = [0]

        def side_effect(prompt):
            call_count[0] += 1
            r = MagicMock()
            r.content = "yes" if call_count[0] == 1 else "no"
            return r

        llm = MagicMock()
        llm.invoke.side_effect = side_effect
        state = _base_state(documents=docs, retry_count=0)
        result = grade_documents_node(state, _config(llm=llm))
        self.assertEqual(len(result["documents"]), 1)


# ════════════════════════════════════════════════════════════════
# rewrite_query_node 测试
# ════════════════════════════════════════════════════════════════

class TestRewriteQueryNode(unittest.TestCase):

    def test_rewrites_question(self):
        """LLM 应返回改写后的问题。"""
        llm = _make_llm("更明确的改写问题")
        state = _base_state(question="原始问题")
        result = rewrite_query_node(state, _config(llm=llm))
        self.assertEqual(result["rewritten_question"], "更明确的改写问题")

    def test_falls_back_to_original_on_llm_error(self):
        """LLM 抛异常时，应回退到原始问题。"""
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("API 超时")
        state = _base_state(question="原始问题")
        result = rewrite_query_node(state, _config(llm=llm))
        self.assertEqual(result["rewritten_question"], "原始问题")

    def test_no_llm_returns_original(self):
        """未注入 LLM 时，应回退到原始问题。"""
        state = _base_state(question="原始问题")
        result = rewrite_query_node(state, _config(llm=None))
        self.assertEqual(result["rewritten_question"], "原始问题")


# ════════════════════════════════════════════════════════════════
# generate_node 测试
# ════════════════════════════════════════════════════════════════

class TestGenerateNode(unittest.TestCase):

    def test_generates_answer(self):
        """有文档时，generate_node 应返回非空 generation。"""
        docs = [_make_doc("机器学习是一种 AI 方法")]
        llm = _make_llm("机器学习是一种让计算机自动学习的方法。")
        state = _base_state(question="什么是机器学习", documents=docs)
        result = generate_node(state, _config(llm=llm))
        self.assertIsNotNone(result["generation"])
        self.assertGreater(len(result["generation"]), 0)

    def test_appends_to_chat_history(self):
        """generate_node 执行后，chat_history 应追加 HumanMessage + AIMessage。"""
        docs = [_make_doc("内容")]
        llm = _make_llm("回答")
        state = _base_state(question="问题", documents=docs, chat_history=[])
        result = generate_node(state, _config(llm=llm))
        self.assertEqual(len(result["chat_history"]), 2)

    def test_hallucination_flag_reset_to_false(self):
        """generate_node 应将 hallucination_flag 重置为 False（等待 grade_answer 重新评估）。"""
        docs = [_make_doc()]
        llm = _make_llm("新答案")
        state = _base_state(documents=docs, hallucination_flag=True)
        result = generate_node(state, _config(llm=llm))
        self.assertFalse(result["hallucination_flag"])

    def test_no_llm_returns_error_message(self):
        """未注入 LLM 时，generation 应包含错误提示。"""
        state = _base_state(documents=[_make_doc()])
        result = generate_node(state, _config(llm=None))
        self.assertIn("LLM", result["generation"])


# ════════════════════════════════════════════════════════════════
# grade_answer_node 测试
# ════════════════════════════════════════════════════════════════

class TestGradeAnswerNode(unittest.TestCase):

    def test_grounded_answer_sets_flag_false(self):
        """LLM 返回 'grounded' 时，hallucination_flag 应为 False。"""
        docs = [_make_doc("真实内容")]
        llm = _make_llm("grounded")
        state = _base_state(documents=docs, generation="基于文档的回答")
        result = grade_answer_node(state, _config(llm=llm))
        self.assertFalse(result["hallucination_flag"])

    def test_not_grounded_answer_sets_flag_true(self):
        """LLM 返回 'not_grounded' 时，hallucination_flag 应为 True。"""
        docs = [_make_doc("真实内容")]
        llm = _make_llm("not_grounded")
        state = _base_state(documents=docs, generation="捏造的回答")
        result = grade_answer_node(state, _config(llm=llm))
        self.assertTrue(result["hallucination_flag"])

    def test_no_llm_always_passes(self):
        """未注入 LLM 时，应默认放行（hallucination_flag=False）。"""
        docs = [_make_doc()]
        state = _base_state(documents=docs, generation="回答")
        result = grade_answer_node(state, _config(llm=None))
        self.assertFalse(result["hallucination_flag"])


# ════════════════════════════════════════════════════════════════
# Conditional edge 函数测试
# ════════════════════════════════════════════════════════════════

class TestConditionalEdges(unittest.TestCase):

    def test_should_rewrite_when_no_docs_and_retry_below_max(self):
        """文档为空且 retry_count < MAX_RETRIES 时，应路由到 rewrite。"""
        state = _base_state(documents=[], retry_count=0)
        self.assertEqual(should_rewrite(state), "rewrite")

    def test_should_generate_when_docs_present(self):
        """有文档时，应路由到 generate（不管 retry_count）。"""
        state = _base_state(documents=[_make_doc()], retry_count=0)
        self.assertEqual(should_rewrite(state), "generate")

    def test_should_generate_when_retry_exceeds_max(self):
        """即使文档为空，retry_count >= MAX_RETRIES 时也应路由到 generate（强制结束循环）。"""
        state = _base_state(documents=[], retry_count=MAX_RETRIES)
        self.assertEqual(should_rewrite(state), "generate")

    def test_should_retry_generate_on_hallucination(self):
        """检测到幻觉且 retry_count < MAX_RETRIES 时，应路由到 generate 重试。"""
        state = _base_state(hallucination_flag=True, retry_count=0)
        self.assertEqual(should_retry_generate(state), "generate")

    def test_should_end_when_answer_grounded(self):
        """答案可信（hallucination_flag=False）时，应路由到 end。"""
        state = _base_state(hallucination_flag=False, retry_count=0)
        self.assertEqual(should_retry_generate(state), "end")

    def test_should_end_when_retry_exceeds_max(self):
        """即使有幻觉，retry_count >= MAX_RETRIES 时也应路由到 end（强制终止）。"""
        state = _base_state(hallucination_flag=True, retry_count=MAX_RETRIES)
        self.assertEqual(should_retry_generate(state), "end")


if __name__ == "__main__":
    unittest.main(verbosity=2)
