"""
qa_chain/state.py — LangGraph RAG 的全局状态定义

RAGState 是 StateGraph 的"共享黑板"，所有节点函数通过它读写数据。
GraphConfig 承载运行时注入的重型对象（vectordb、llm），避免污染可序列化的 state。
"""

from __future__ import annotations

from typing import Any, List, Optional, Set
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    """Self-RAG 图的共享状态。

    字段说明
    --------
    question : str
        用户原始问题。
    rewritten_question : str | None
        rewrite_query_node 改写后的问题；检索时优先使用，为 None 时回退到 question。
    documents : List[Document]
        retrieve_node 召回并经 grade_documents_node 过滤后的相关文档。
    generation : str | None
        generate_node 生成的回答；grade_answer_node 验证后更新。
    chat_history : List[BaseMessage]
        多轮对话历史（HumanMessage / AIMessage 列表），由 SqliteSaver 持久化。
    source_filter : Set[str] | None
        允许检索的 source 路径集合（来自 Gradio "知识库范围" 下拉框）；
        None 表示不过滤，检索全部知识库。
    retry_count : int
        重试次数累计（rewrite_query 和 generate 共用同一个计数器）；
        防止无限循环，超过 MAX_RETRIES 后强制走 generate/END。
    embedding : str
        当前使用的 Embedding 模型名（如 "m3e"、"multilingual"）；
        用于 retrieve_node 动态解析向量库路径。
    top_k : int
        检索返回的最大文档数。
    hallucination_flag : bool
        grade_answer_node 的判定：True 表示检测到幻觉，触发 generate 重试。
    """

    question: str
    rewritten_question: Optional[str]
    documents: List[Document]
    generation: Optional[str]
    chat_history: List[BaseMessage]
    source_filter: Optional[Set[str]]
    retry_count: int
    embedding: str
    top_k: int
    hallucination_flag: bool


# 重试次数上限：防止 rewrite_query 和 generate 各自无限循环
MAX_RETRIES: int = 2
