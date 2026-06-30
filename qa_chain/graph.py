"""
qa_chain/graph.py — LangGraph Self-RAG 图组装工厂

build_rag_graph(checkpointer) → CompiledGraph

节点拓扑：
        router ──[retrieve]──→ retrieve → grade_documents ──[有相关 chunk]──→ generate → grade_answer → END
            │
            └──[direct]──────────────────────────────────────────────→ generate
                     │                                │
                     │[文档为空 + retry<MAX]           │[幻觉 + retry<MAX]
                     ↓                                ↓
              rewrite_query ────────────────→ generate (loop)
                     │
                     └─────────→ retrieve (loop)

SqliteSaver 注入后，graph.invoke 自动按 thread_id 持久化 chat_history，
重启服务后历史对话仍可恢复。
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from qa_chain.state import RAGState
from qa_chain.nodes import (
    router_node,
    generate_node,
    grade_answer_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
    should_rewrite,
    should_retry_generate,
)

logger = logging.getLogger(__name__)

# 节点名称常量（避免魔法字符串散落各处）
NODE_ROUTER = "router"
NODE_RETRIEVE = "retrieve"
NODE_GRADE_DOCS = "grade_documents"
NODE_REWRITE = "rewrite_query"
NODE_GENERATE = "generate"
NODE_GRADE_ANS = "grade_answer"


def build_rag_graph(checkpointer=None):
    """
    构造并编译 Self-RAG 图。

    Parameters
    ----------
    checkpointer : BaseCheckpointSaver, optional
        LangGraph checkpointer（如 SqliteSaver / MemorySaver）；
        不传则无持久化（测试 / 单次调用场景）。

    Returns
    -------
    CompiledGraph
        调用方式：
            result = graph.invoke(
                {
                    "question": "用户问题",
                    "rewritten_question": None,
                    "documents": [],
                    "generation": None,
                    "chat_history": [],          # List[BaseMessage]
                    "source_filter": None,       # 或 {"path1", "path2"}
                    "retry_count": 0,
                    "embedding": "m3e",
                    "top_k": 4,
                    "hallucination_flag": False,
                    "route_decision": None,
                },
                config={
                    "configurable": {
                        "thread_id": "用户会话ID",  # SqliteSaver 按此隔离会话
                        "vectordb": vectordb,       # FAISS 实例
                        "llm": llm,                 # LangChain LLM 对象
                        "retriever_kind": "dense",  # 或 "hybrid"
                        "history_len": 3,
                    }
                },
            )
            answer = result["generation"]
    """
    # ── 构造 StateGraph ──────────────────────────────────────────────────
    graph = StateGraph(RAGState)

    # 注册节点
    graph.add_node(NODE_ROUTER, router_node)
    graph.add_node(NODE_RETRIEVE, retrieve_node)
    graph.add_node(NODE_GRADE_DOCS, grade_documents_node)
    graph.add_node(NODE_REWRITE, rewrite_query_node)
    graph.add_node(NODE_GENERATE, generate_node)
    graph.add_node(NODE_GRADE_ANS, grade_answer_node)

    # 入口：先路由，再决定是否检索
    graph.set_entry_point(NODE_ROUTER)

    graph.add_conditional_edges(
        NODE_ROUTER,
        lambda s: s.get("route_decision", "retrieve"),
        {
            "direct": NODE_GENERATE,
            "retrieve": NODE_RETRIEVE,
        },
    )

    # ── 固定边 ──────────────────────────────────────────────────────────
    # retrieve → grade_documents（每次检索完都要评分）
    graph.add_edge(NODE_RETRIEVE, NODE_GRADE_DOCS)

    # rewrite_query → retrieve（改写后重新检索）
    graph.add_edge(NODE_REWRITE, NODE_RETRIEVE)

    # generate → grade_answer（每次生成完都要验证）
    graph.add_edge(NODE_GENERATE, NODE_GRADE_ANS)

    # ── 条件边 ──────────────────────────────────────────────────────────
    # grade_documents 后：有相关文档 → generate；无文档且可重试 → rewrite
    graph.add_conditional_edges(
        NODE_GRADE_DOCS,
        should_rewrite,
        {
            "rewrite": NODE_REWRITE,
            "generate": NODE_GENERATE,
        },
    )

    # grade_answer 后：幻觉且可重试 → generate；否则 → END
    graph.add_conditional_edges(
        NODE_GRADE_ANS,
        should_retry_generate,
        {
            "generate": NODE_GENERATE,
            "end": END,
        },
    )

    # ── 编译 ─────────────────────────────────────────────────────────────
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info(
        "RAG graph compiled. nodes=%s checkpointer=%s",
        [NODE_ROUTER, NODE_RETRIEVE, NODE_GRADE_DOCS, NODE_REWRITE, NODE_GENERATE, NODE_GRADE_ANS],
        type(checkpointer).__name__ if checkpointer else "None",
    )
    return compiled


def get_empty_state() -> dict:
    """
    返回一个填好默认值的初始 state 字典，供调用方按需覆盖。

    用法：
        state = get_empty_state()
        state["question"] = "用户问题"
        state["embedding"] = "m3e"
        result = graph.invoke(state, config=...)
    """
    return {
        "question": "",
        "rewritten_question": None,
        "documents": [],
        "generation": None,
        "chat_history": [],
        "source_filter": None,
        "retry_count": 0,
        "embedding": "m3e",
        "top_k": 4,
        "hallucination_flag": False,
        "route_decision": None,
    }
