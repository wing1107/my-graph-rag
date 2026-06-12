#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI 后端（LangGraph Self-RAG 版本）

变更：
- 移除 RetrievalQA / ConversationalRetrievalChain
- 移除进程内 _SESSION_STORE dict
- 新增 graph 单例（SqliteSaver 持久化）
- POST /answer/ 直接调用 graph.invoke()

运行：
    uvicorn api.fast_api:app --reload --port 8000
"""

import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import faiss
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qa_chain.model_to_llm import model_to_llm          # noqa: E402
from embedding.call_embedding import get_embedding       # noqa: E402
from qa_chain.graph import build_rag_graph, get_empty_state  # noqa: E402
from qa_chain.prompt_templates import DEFAULT_TEMPLATE   # noqa: E402

load_dotenv(find_dotenv())

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_RETRIEVER_KIND = os.getenv("RETRIEVER_KIND", "dense").strip().lower()
if _RETRIEVER_KIND not in ("dense", "hybrid"):
    _RETRIEVER_KIND = "dense"

logger = logging.getLogger("fast_api")
logger.info("retriever kind = %s", _RETRIEVER_KIND)

app = FastAPI(
    title="知识库问答 API（LangGraph Self-RAG）",
    description="Self-RAG 图拓扑：检索 → 文档评分 → 生成 → 答案验证 | SqliteSaver 会话持久化",
)

_VECTOR_DB_ROOT = _PROJECT_ROOT / "data_base" / "vector_db"
_SESSIONS_DB_PATH = _PROJECT_ROOT / "sessions" / "rag_sessions.sqlite"
_EMBEDDING_META_FILE = "embedding_meta.json"
_DEFAULT_EMBEDDING = os.getenv("EMBEDDING_PROVIDER", "m3e")

# ── Graph 单例 ───────────────────────────────────────────────────────────
_graph = None


def _get_graph():
    global _graph
    if _graph is not None:
        return _graph
    os.makedirs(_SESSIONS_DB_PATH.parent, exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string(str(_SESSIONS_DB_PATH))
        logger.info("SqliteSaver 初始化: %s", _SESSIONS_DB_PATH)
    except Exception as e:
        logger.warning("SqliteSaver 失败（%s），使用 MemorySaver", e)
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    _graph = build_rag_graph(checkpointer)
    return _graph


# ── 向量库缓存 ────────────────────────────────────────────────────────────
_vectordb_cache: dict = {}


def _get_persist_path(embedding: str) -> str:
    return str(_VECTOR_DB_ROOT / f"faiss_{embedding}")


def _load_vectordb(persist_path: str, embedding_provider: str = "m3e"):
    """加载 FAISS 向量库（Windows 中文路径兼容 + meta 自动匹配 Embedding）。"""
    if persist_path in _vectordb_cache:
        return _vectordb_cache[persist_path]

    from langchain_community.vectorstores import FAISS

    embedding_obj = None
    meta_path = os.path.join(persist_path, _EMBEDDING_META_FILE)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            resolved_model = meta.get("resolved_model_name")
            if resolved_model:
                embedding_obj = get_embedding(embedding=resolved_model)
        except Exception as exc:
            logger.warning("读取 Embedding 元数据失败: %s", exc)

    if embedding_obj is None:
        embedding_obj = get_embedding(embedding=embedding_provider)

    # Windows 中文路径兼容：复制到 ASCII 目录再加载
    tmp_dir = tempfile.mkdtemp(prefix="faiss_ascii_")
    try:
        shutil.copyfile(os.path.join(persist_path, "index.faiss"), os.path.join(tmp_dir, "index.faiss"))
        shutil.copyfile(os.path.join(persist_path, "index.pkl"), os.path.join(tmp_dir, "index.pkl"))
        index = faiss.read_index(os.path.join(tmp_dir, "index.faiss"))
        with open(os.path.join(tmp_dir, "index.pkl"), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    vectordb = FAISS(embedding_obj, index, docstore, index_to_docstore_id)
    _vectordb_cache[persist_path] = vectordb
    return vectordb


# ── LLM 缓存 ─────────────────────────────────────────────────────────────
_llm_cache: dict = {}


def _get_llm(model: str, temperature: float):
    key = (model, float(temperature))
    if key not in _llm_cache:
        _llm_cache[key] = model_to_llm(model, temperature)
    return _llm_cache[key]


# ── 请求模型 ─────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    prompt: str
    model: str = ""
    temperature: float = 0.1
    session_id: str = ""
    api_key: str = ""
    db_path: str = ""
    embedding: str = _DEFAULT_EMBEDDING
    top_k: int = 5


def _resolve_model(item: QuestionRequest) -> str:
    return item.model or os.getenv("ZHIPUAI_MODEL", "glm-4-flash")


def _resolve_db_path(item: QuestionRequest) -> str:
    if item.db_path:
        return item.db_path
    env_override = os.getenv("VECTOR_DB_DIR")
    if env_override:
        return env_override
    return _get_persist_path(item.embedding)


# ── API 端点 ─────────────────────────────────────────────────────────────

@app.get("/", summary="健康检查")
async def root():
    return {
        "message": "知识库问答 API（LangGraph Self-RAG）已启动",
        "docs": "/docs",
        "retriever_kind": _RETRIEVER_KIND,
    }


@app.post("/answer/", summary="知识库问答（Self-RAG）")
async def get_response(item: QuestionRequest):
    """
    Self-RAG 图问答接口。

    - session_id 不为空时启用 SqliteSaver 会话持久化（多轮对话，重启后保留）
    - session_id 为空时每次生成随机 thread_id（无状态单次查询）
    """
    model_name = _resolve_model(item)
    db_path = _resolve_db_path(item)

    thread_id = item.session_id.strip() or str(uuid.uuid4())

    logger.info(
        "问答请求: model=%s embedding=%s db_path=%s thread_id=%s",
        model_name, item.embedding, db_path, thread_id,
    )

    vectordb = _load_vectordb(db_path, item.embedding)
    llm = _get_llm(model_name, item.temperature)

    graph = _get_graph()
    state = get_empty_state()
    state["question"] = item.prompt
    state["embedding"] = item.embedding
    state["top_k"] = item.top_k

    config = {
        "configurable": {
            "thread_id": thread_id,
            "vectordb": vectordb,
            "llm": llm,
            "retriever_kind": _RETRIEVER_KIND,
            "history_len": 3 if item.session_id.strip() else 0,
        }
    }

    result = graph.invoke(state, config=config)
    answer = result.get("generation") or "（无回答）"

    return {
        "answer": answer,
        "model": model_name,
        "session_id": thread_id if item.session_id.strip() else "",
        "retriever_kind": _RETRIEVER_KIND,
    }


@app.delete("/session/{session_id}", summary="清除历史会话")
async def clear_session(session_id: str):
    """SqliteSaver 会话无法直接删除（需要 checkpointer.delete_checkpoint），
    返回提示信息即可；应用重启或使用新 session_id 等效于新会话。"""
    return {"message": f"请使用新的 session_id 开启新会话（当前 session_id={session_id}）"}
