#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_gradio.py — myGraphRagProject 主入口（LangGraph Self-RAG 版本）

与 myRagProject 的主要变化：
1. 问答核心从 RetrievalQA / get_completion 改为 LangGraph graph.invoke()
2. 会话历史改由 SqliteSaver 持久化（sessions/rag_sessions.sqlite），重启后保留
3. _answer_with_graph() 替换旧版 _answer_with_rag()，节点日志可见 grade 过程
4. langchain 0.2.x import 路径（langchain_community, langchain_text_splitters）
5. 建库/删库/Embedding 切换/Make Markdown 等非问答功能保持不变
"""

import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# 关闭 gradio 启动时的版本检查 / 匿名遥测，避免网络不通时卡在 import gradio 等超时
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# ── 规避 Windows 本机 WMI 卡死 ───────────────────────────────────────────
# pandas（被 gradio 间接 import）在导入时会调用 platform.machine()，进而触发
# platform._wmi_query() 去查 Win32_OperatingSystem。当本机 WMI 服务异常时，
# 该查询会无限阻塞，从而卡住 `import gradio`。这里让 _wmi_query 直接抛 OSError，
# 强制 platform 走 getwindowsversion()+注册表的快速回退路径（不碰 WMI、不联网）。
import platform as _platform


def _wmi_query_disabled(*_args, **_kwargs):
    raise OSError("WMI query disabled to avoid hang on broken WMI service")


_platform._wmi_query = _wmi_query_disabled

# ── 提前配置日志，确保后续 import 阶段的 INFO 日志可见 ────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("run_gradio")
logger.info("[1a] import dotenv ...")
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv())
logger.info("[1b] import gradio ...")
import gradio as gr

# ── langchain 0.2.x imports ──────────────────────────────────────────────
logger.info("[2] import langchain ...")
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 项目内部模块 ──────────────────────────────────────────────────────────
logger.info("[3] import database ...")
from database.create_db import (
    file_loader,
    EMBEDDING_META_FILE,
    LEGACY_FALLBACK_EMBEDDING_MODEL,
    _read_faiss_index_dim,
    _embedding_dim,
    save_faiss_db,
    load_faiss_db,
)
logger.info("[4] import embedding ...")
from embedding.call_embedding import get_embedding, clear_embedding_cache
from llm.call_llm import get_completion
from utils.file_utils import calculate_dest_paths, expand_directory_to_files
from qa_chain.graph import build_rag_graph, get_empty_state
from qa_chain.model_to_llm import model_to_llm

logger.info("所有模块加载完成，初始化应用 ...")

# ── 常量 ─────────────────────────────────────────────────────────────────
LLM_MODEL_DICT = {
    "zhipuai": ["glm-4-flash", "glm-4", "chatglm_pro", "chatglm_std", "chatglm_lite"],
    "qwen": ["qwen3-max", "qwen-flash", "qwen-plus", "qwen-long"],
}
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
INIT_LLM = "glm-4-flash"
EMBEDDING_MODEL_LIST = ["m3e", "multilingual"]
INIT_EMBEDDING_MODEL = "m3e"

DEFAULT_DB_PATH = os.path.join(ROOT, "data_base", "kownledge_db")
VECTOR_DB_ROOT = os.path.join(ROOT, "data_base", "vector_db")
LEGACY_PERSIST_DIR_NAME = "faiss_huggingface"

RETRIEVER_KIND = os.getenv("RETRIEVER_KIND", "dense").strip().lower()
if RETRIEVER_KIND not in ("dense", "hybrid"):
    RETRIEVER_KIND = "dense"

SESSIONS_DB_PATH = os.path.join(ROOT, "sessions", "rag_sessions.sqlite")


def get_persist_path(embedding: str = None) -> str:
    name = embedding or INIT_EMBEDDING_MODEL
    return os.path.join(VECTOR_DB_ROOT, f"faiss_{name}")


DEFAULT_PERSIST_PATH = get_persist_path(INIT_EMBEDDING_MODEL)

load_faiss_vectordb = load_faiss_db  # 别名，供内部函数使用

# ── SqliteSaver + Graph 单例 ─────────────────────────────────────────────
_graph = None


def _get_rag_graph():
    """懒初始化 SqliteSaver 和 graph。"""
    global _graph
    if _graph is not None:
        return _graph
    os.makedirs(os.path.dirname(SESSIONS_DB_PATH), exist_ok=True)
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string(SESSIONS_DB_PATH)
        logger.info("SqliteSaver 初始化: %s", SESSIONS_DB_PATH)
    except Exception as e:
        logger.warning("SqliteSaver 初始化失败（%s），使用 MemorySaver 兜底", e)
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    _graph = build_rag_graph(checkpointer)
    return _graph


# ════════════════════════════════════════════════════════════════
# 知识库建库相关函数（与旧版基本一致，仅 import 路径更新）
# ════════════════════════════════════════════════════════════════

def _inject_lesson_markers(split_docs):
    """为缺少课程编号的切片添加课程标记，为日语关键词添加中文翻译。"""
    import re as _re

    # OCR 会将「第26課」拆成「第 26 か 課」（数字和汉字之间有空格，汉字读音假名夹在中间）。
    # 旧模式 r'第(\d+)\s*' 要求数字紧跟「第」，无法匹配带空格的格式。
    # 新模式：允许「第」后有空格，允许数字与章节字之间有一个假名读音（如「か」「しょう」）。
    lesson_pattern = _re.compile(r'第\s*(\d+)(?:\s*[ぁ-んァ-ン])?\s*([章課课])')
    jp_to_cn_mappings = [
        # 使用 \s* 兼容 OCR 将「会話」拆成「会 話」的情况（页面上方有假名注音导致空格）
        (_re.compile(r'会\s*話'), '【会话/对话】'),
        (_re.compile(r'単\s*語'), '【单词/词汇】'),
        (_re.compile(r'文\s*法'), '【语法】'),
        (_re.compile(r'練\s*習'), '【练习】'),
        (_re.compile(r'例\s*文'), '【例句】'),
        (_re.compile(r'文\s*型'), '【句型】'),
    ]

    page_to_lesson = {}
    for doc in split_docs:
        page = doc.metadata.get('page', -1)
        if page < 0:
            continue
        match = lesson_pattern.search(doc.page_content)
        if match:
            lesson_num = int(match.group(1))
            chapter_char = match.group(2)
            # 只有模式出现在页首（pos<30）才视为真正的课头标题；
            # 正文中的交叉引用（如「参照第14課」）出现位置较靠后，需排除。
            if page not in page_to_lesson and match.start() < 30:
                page_to_lesson[page] = (lesson_num, chapter_char)

    if not page_to_lesson:
        return split_docs

    sorted_pages = sorted(page_to_lesson.keys())

    def get_lesson_for_page(page):
        lesson = None
        for p in sorted_pages:
            if p <= page:
                lesson = page_to_lesson[p]
            else:
                break
        return lesson

    lesson_injected_count = 0
    translation_injected_count = 0

    for doc in split_docs:
        page = doc.metadata.get('page', -1)
        content = doc.page_content
        markers = []
        existing_match = lesson_pattern.search(content)
        if not existing_match:
            entry = get_lesson_for_page(page)
            if entry is not None:
                lesson_num, chapter_char = entry
                markers.append(f"【第{lesson_num}{chapter_char}】")
                # 若继承的课号是日文「課」，同步注入中文形式，兼容 BM25 中文查询
                if chapter_char == '課':
                    markers.append(f"【第{lesson_num}课】")
                lesson_injected_count += 1
        else:
            # 页面已有课程号（如「第26課」日文）；额外注入中文形式「【第26课】」，
            # 让 BM25 能用中文查询字符（课 U+8BFE）命中日文原文（課 U+8AB2）。
            lesson_num = int(existing_match.group(1))
            chapter_char = existing_match.group(2)
            if chapter_char == '課':
                cn_marker = f"【第{lesson_num}课】"
                if cn_marker not in content:
                    markers.append(cn_marker)
                    lesson_injected_count += 1
        for jp_pattern, cn_label in jp_to_cn_mappings:
            if jp_pattern.search(content) and cn_label not in content:
                markers.append(cn_label)
                translation_injected_count += 1
        if markers:
            doc.page_content = " ".join(markers) + "\n" + content

    logger.info(
        "标记注入完成：%d 个切片注入课程标记，%d 个切片注入中文翻译",
        lesson_injected_count, translation_injected_count,
    )
    return split_docs


def _decide_vectorize_mode(has_existing_db, existing_embedding_provider, current_embedding, rebuild):
    if not has_existing_db:
        return "full_rebuild", "首次构建向量库"

    embedding_incompatible = (
        existing_embedding_provider is not None
        and existing_embedding_provider != current_embedding
    )

    if embedding_incompatible:
        if not rebuild:
            return "reject", (
                f"已取消：现有库使用 {existing_embedding_provider} Embedding，"
                f"当前选择 {current_embedding}，不兼容。请切回旧 Embedding 或勾选重建。"
            )
        return "full_rebuild", (
            f"Embedding 不兼容且已勾选重建，清空全量重建（{existing_embedding_provider} → {current_embedding}）"
        )

    if rebuild:
        return "selective_rebuild", "选择性重建（仅替换本次上传文件 source 的向量，保留其他）"
    return "append", "追加新文档到现有知识库"


def build_faiss_db_info(
    files,
    embedding="m3e",
    rebuild=False,
    persist_directory=None,
    progress=gr.Progress(track_tqdm=False),
):
    """构建/更新 FAISS 向量库（生成器，向 Gradio 汇报进度）。"""
    if persist_directory is None:
        persist_directory = get_persist_path(embedding)

    def _status(text):
        logger.info("[向量化] %s", text)
        return text

    try:
        progress(0.0, desc="开始向量化")
        yield "", _status(f"开始向量化，Embedding={embedding}")

        if files is None:
            yield "未上传文件。", _status("失败：未上传文件")
            return
        if not isinstance(files, list):
            files = [files]

        existing_index = os.path.join(persist_directory, "index.faiss")
        existing_pkl = os.path.join(persist_directory, "index.pkl")
        has_existing_db = os.path.exists(existing_index) and os.path.exists(existing_pkl)

        existing_embedding_provider = None
        if has_existing_db:
            meta_path = os.path.join(persist_directory, EMBEDDING_META_FILE)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    existing_embedding_provider = meta.get("embedding_provider")
                except Exception:
                    pass

        mode, mode_msg = _decide_vectorize_mode(
            has_existing_db, existing_embedding_provider, embedding, rebuild
        )

        if mode == "reject":
            yield mode_msg, _status("失败：拒绝向量化以保护旧数据")
            return

        append_mode = mode == "append"
        selective_rebuild_mode = mode == "selective_rebuild"
        yield "", _status(mode_msg)

        # ── 持久化源文件 ──
        progress(0.05, desc="保存源文件")
        yield "", _status("保存源文件到知识库目录")
        saved_files = []
        try:
            raw_paths = []
            for f in files:
                if isinstance(f, tempfile._TemporaryFileWrapper):
                    f = f.name
                if isinstance(f, str) and os.path.exists(f):
                    raw_paths.append(f)
            file_paths = expand_directory_to_files(raw_paths)
            if file_paths:
                os.makedirs(DEFAULT_DB_PATH, exist_ok=True)
                path_mappings = calculate_dest_paths(file_paths, DEFAULT_DB_PATH)
                for src, dest in path_mappings:
                    os.makedirs(os.path.dirname(dest) or DEFAULT_DB_PATH, exist_ok=True)
                    shutil.copy2(src, dest)
                    saved_files.append(dest)
                files = saved_files
        except Exception as e:
            logger.warning("保存源文件失败: %s（将继续向量化）", e)

        # ── 加载文档 ──
        progress(0.1, desc="扫描文件")
        yield "", _status("扫描文件")
        loaders = []
        [file_loader(file, loaders) for file in files]
        yield "", _status(f"发现可解析文件 {len(loaders)} 个")

        progress(0.25, desc="加载文档")
        docs = []
        for idx, loader in enumerate(loaders, 1):
            if loader is not None:
                docs.extend(loader.load())
            if idx == 1 or idx == len(loaders) or idx % 3 == 0:
                yield "", _status(f"加载文档 ({idx}/{len(loaders)})")

        if not docs:
            yield "未解析到可用文档（支持 .txt/.md/.pdf/.docx）。", _status("失败：未解析到文档")
            return

        # ── 切分 ──
        progress(0.6, desc="切分文档")
        yield "", _status("切分文档")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        split_docs = text_splitter.split_documents(docs)
        yield "", _status(f"切分完成：{len(split_docs)} 个切片")

        # ── 注入课程标记 ──
        progress(0.62, desc="注入课程标记")
        split_docs = _inject_lesson_markers(split_docs)

        # ── 加载 Embedding ──
        progress(0.75, desc=f"加载 Embedding（{embedding}）")
        yield "", _status(f"加载 Embedding 模型（{embedding}）")
        embedding_obj = get_embedding(embedding=embedding)

        # ── 维度兼容校验 ──
        if append_mode or selective_rebuild_mode:
            existing_dim = _read_faiss_index_dim(persist_directory)
            current_dim = _embedding_dim(embedding_obj)
            if existing_dim and current_dim and existing_dim != current_dim:
                if append_mode:
                    yield (
                        f"维度不兼容（现有 {existing_dim} ≠ 当前 {current_dim}），已取消追加。",
                        _status("失败：维度不兼容"),
                    )
                    return
                selective_rebuild_mode = False
                yield "", _status(f"维度不兼容，降级为全量重建")

        # ── 构建/更新 FAISS ──
        progress(0.88, desc="构建 FAISS")
        os.makedirs(persist_directory, exist_ok=True)

        def _embed_batched(docs_to_embed, emb_obj, batch_size=8):
            texts = [d.page_content for d in docs_to_embed]
            total = len(texts)
            all_embs = []

            def _embed_one(text):
                clean_text = (text or "").strip() or " "
                if hasattr(emb_obj, "embed_query"):
                    return emb_obj.embed_query(clean_text)
                return emb_obj.embed_documents([clean_text])[0]

            def _log_slow_batch(stop_event: threading.Event, done_cnt: int, total_cnt: int, size: int):
                # 慢批次心跳：避免前端长时间无进展时误判为僵死。
                start_t = time.monotonic()
                while not stop_event.wait(30):
                    elapsed = int(time.monotonic() - start_t)
                    logger.warning(
                        "[向量化] Embedding 批次耗时较长：done=%d/%d, batch=%d, elapsed=%ds",
                        done_cnt,
                        total_cnt,
                        size,
                        elapsed,
                    )

            for start in range(0, total, batch_size):
                batch = [((t or "").strip() or " ") for t in texts[start: start + batch_size]]
                done_before = start

                stop_event = threading.Event()
                heartbeat = threading.Thread(
                    target=_log_slow_batch,
                    args=(stop_event, done_before, total, len(batch)),
                    daemon=True,
                )
                heartbeat.start()

                batch_start_t = time.monotonic()
                try:
                    batch_embs = emb_obj.embed_documents(batch)
                except Exception as batch_err:
                    logger.warning("批量 Embedding 失败，降级逐条处理: %s", batch_err)
                    batch_embs = []
                    for idx, txt in enumerate(batch, 1):
                        try:
                            batch_embs.append(_embed_one(txt))
                        except Exception as item_err:
                            raise RuntimeError(
                                f"第 {start + idx}/{total} 条文本 Embedding 失败: {item_err}"
                            ) from item_err
                finally:
                    stop_event.set()

                elapsed = time.monotonic() - batch_start_t
                if elapsed >= 20:
                    logger.warning(
                        "[向量化] Embedding 慢批次：done=%d/%d, batch=%d, elapsed=%.1fs",
                        min(start + len(batch), total),
                        total,
                        len(batch),
                        elapsed,
                    )

                all_embs.extend(batch_embs)
                done = min(start + len(batch), total)
                yield done, total, all_embs

        def _build_from_embeddings(docs_to_embed, all_embs, emb_obj):
            import numpy as np
            import faiss as faiss_lib
            from langchain_community.docstore.in_memory import InMemoryDocstore

            emb_array = np.array(all_embs, dtype=np.float32)
            dim = emb_array.shape[1]
            index = faiss_lib.IndexFlatL2(dim)
            index.add(emb_array)
            id_map = {i: str(i) for i in range(len(docs_to_embed))}
            docstore_dict = {str(i): docs_to_embed[i] for i in range(len(docs_to_embed))}
            docstore = InMemoryDocstore(docstore_dict)
            return FAISS(emb_obj, index, docstore, id_map)

        if append_mode:
            yield "", _status("追加新文档到现有向量库")
            vectordb = load_faiss_vectordb(persist_directory, embedding)
            all_embs = []
            for done, total, all_embs in _embed_batched(split_docs, embedding_obj):
                progress(0.88 + 0.09 * done / max(total, 1), desc=f"Embedding {done}/{total}")
                yield "", _status(f"Embedding {done}/{total}")
            texts = [d.page_content for d in split_docs]
            metadatas = [d.metadata for d in split_docs]
            vectordb.add_embeddings(list(zip(texts, all_embs)), metadatas=metadatas)
            action_label = "追加"

        elif selective_rebuild_mode:
            yield "", _status("选择性重建：提取保留切片")
            existing_vectordb = load_faiss_vectordb(persist_directory, embedding)
            new_sources = {(d.metadata or {}).get("source", "") for d in split_docs if d.metadata}
            kept_docs, kept_embs = _extract_kept_docs_and_embeddings(existing_vectordb, new_sources)
            yield "", _status(f"保留 {len(kept_docs)} 个旧切片，重新 Embedding 新文档")

            all_embs = []
            for done, total, all_embs in _embed_batched(split_docs, embedding_obj):
                progress(0.88 + 0.09 * done / max(total, 1), desc=f"Embedding {done}/{total}")
                yield "", _status(f"Embedding {done}/{total}")

            combined_docs = kept_docs + list(split_docs)
            combined_embs = kept_embs + [list(e) for e in all_embs]
            vectordb = _build_from_embeddings(combined_docs, combined_embs, embedding_obj)
            action_label = f"选择性重建（保留 {len(kept_docs)}，重建 {len(split_docs)}）"

        else:
            yield "", _status(f"全量重建（{len(split_docs)} 切片）")
            all_embs = []
            for done, total, all_embs in _embed_batched(split_docs, embedding_obj):
                progress(0.88 + 0.09 * done / max(total, 1), desc=f"Embedding {done}/{total}")
                yield "", _status(f"Embedding {done}/{total}")
            vectordb = _build_from_embeddings(split_docs, all_embs, embedding_obj)
            action_label = "全量重建"

        save_faiss_db(vectordb, persist_directory, embedding_obj, embedding)
        model_center.clear_cache()
        progress(1.0, desc="完成")
        yield "向量库完成，请开始提问", _status(f"完成（{action_label}）")

    except Exception as e:
        logger.exception("build_faiss_db_info 失败: %s", e)
        yield f"向量化失败：{e}", _status(f"失败：{e}")


def _extract_kept_docs_and_embeddings(vectordb, sources_to_remove: set):
    docstore_dict = getattr(vectordb.docstore, "_dict", {}) or {}
    index_to_docstore_id = getattr(vectordb, "index_to_docstore_id", {}) or {}
    faiss_index = vectordb.index

    kept_docs, kept_embs = [], []
    for int_idx, docstore_id in index_to_docstore_id.items():
        doc = docstore_dict.get(docstore_id)
        if doc is None:
            continue
        src = (doc.metadata or {}).get("source", "")
        if src in sources_to_remove:
            continue
        try:
            vec = faiss_index.reconstruct(int(int_idx))
            kept_docs.append(doc)
            kept_embs.append([float(x) for x in vec])
        except Exception:
            pass
    return kept_docs, kept_embs


# ════════════════════════════════════════════════════════════════
# 向量库/Source 辅助函数
# ════════════════════════════════════════════════════════════════

ALL_SOURCES_LABEL = "全部知识库"


def _source_to_label(source: str) -> str:
    return os.path.basename(source) if source else "(未知来源)"


def _list_sources_from_pkl(persist_path: str):
    import pickle
    pkl_path = os.path.join(persist_path, "index.pkl")
    if not os.path.exists(pkl_path):
        return []
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple) and len(data) >= 2:
            docstore = data[0]
            docstore_dict = getattr(docstore, "_dict", {}) or {}
        else:
            return []
        seen = {}
        for doc in docstore_dict.values():
            src = (doc.metadata or {}).get("source", "")
            if src and src not in seen:
                seen[src] = _source_to_label(src)
        items = sorted(seen.items(), key=lambda kv: kv[1])
        return [(label, src) for src, label in items]
    except Exception as e:
        logger.warning("_list_sources_from_pkl 失败: %s", e)
        return []


def _list_sources_from_vectordb(vectordb):
    seen = {}
    docstore_dict = getattr(vectordb.docstore, "_dict", {}) or {}
    for doc in docstore_dict.values():
        src = (doc.metadata or {}).get("source", "")
        if src and src not in seen:
            seen[src] = _source_to_label(src)
    items = sorted(seen.items(), key=lambda kv: kv[1])
    return [(label, src) for src, label in items]


def list_sources_in_vectordb(persist_path: str = None, embedding: str = INIT_EMBEDDING_MODEL):
    if persist_path is None:
        persist_path = get_persist_path(embedding)
    if not os.path.exists(os.path.join(persist_path, "index.faiss")):
        return []
    return _list_sources_from_pkl(persist_path)


def get_source_choices(persist_path: str = None, embedding: str = INIT_EMBEDDING_MODEL):
    sources = list_sources_in_vectordb(persist_path, embedding)
    return [ALL_SOURCES_LABEL] + [label for label, _ in sources]


def get_deletable_sources(persist_path: str = None, embedding: str = INIT_EMBEDDING_MODEL):
    sources = list_sources_in_vectordb(persist_path, embedding)
    return [label for label, _ in sources]


def refresh_source_dropdown(embedding: str = INIT_EMBEDDING_MODEL):
    choices = get_source_choices(embedding=embedding)
    return gr.update(choices=choices, value=ALL_SOURCES_LABEL)


def refresh_delete_dropdown(embedding: str = INIT_EMBEDDING_MODEL):
    choices = get_deletable_sources(embedding=embedding)
    return gr.update(choices=choices, value=[])


def on_embedding_change(embedding: str):
    persist_path = get_persist_path(embedding)
    has_db = os.path.exists(os.path.join(persist_path, "index.faiss"))
    source_choices = get_source_choices(embedding=embedding)
    delete_choices = get_deletable_sources(embedding=embedding)
    status = (
        f"已切换到 Embedding={embedding}，共 {len(delete_choices)} 个知识库"
        if has_db else
        f"已切换到 Embedding={embedding}，对应向量库尚未构建，请先向量化"
    )
    return (
        gr.update(choices=source_choices, value=ALL_SOURCES_LABEL),
        gr.update(choices=delete_choices, value=[]),
        status,
    )


def delete_knowledge_base(
    sources_to_delete: list,
    embedding: str = INIT_EMBEDDING_MODEL,
    persist_directory: str = None,
    progress=gr.Progress(track_tqdm=False),
):
    import numpy as np
    import faiss as faiss_lib
    from langchain_community.docstore.in_memory import InMemoryDocstore

    if persist_directory is None:
        persist_directory = get_persist_path(embedding)

    def _status(text):
        logger.info("[删除知识库] %s", text)
        return text

    try:
        progress(0.0, desc="开始删除")
        yield "", _status(f"开始删除（embedding={embedding}）")

        if not sources_to_delete:
            yield "未选择要删除的知识库。", _status("失败：未选择")
            return

        if not os.path.exists(os.path.join(persist_directory, "index.faiss")):
            yield "向量库不存在。", _status("失败：向量库不存在")
            return

        progress(0.2, desc="加载向量库")
        vectordb = load_faiss_vectordb(persist_directory, embedding)
        all_sources = _list_sources_from_vectordb(vectordb)
        label_to_path = {label: path for label, path in all_sources}

        sources_to_remove = set()
        for label in sources_to_delete:
            if label in label_to_path:
                sources_to_remove.add(label_to_path[label])

        if not sources_to_remove:
            yield f"未找到匹配知识库：{sources_to_delete}", _status("失败：未找到")
            return

        progress(0.4, desc="提取保留切片")
        kept_docs, kept_embs = _extract_kept_docs_and_embeddings(vectordb, sources_to_remove)
        deleted_labels = [_source_to_label(s) for s in sources_to_remove]

        progress(0.6, desc="重建向量库")
        if not kept_docs:
            for f in ["index.faiss", "index.pkl", EMBEDDING_META_FILE]:
                fp = os.path.join(persist_directory, f)
                if os.path.exists(fp):
                    os.remove(fp)
            model_center.clear_cache()
            progress(1.0)
            yield f"已删除：{', '.join(deleted_labels)}。向量库已清空。", _status("完成：清空")
            return

        emb_array = np.array(kept_embs, dtype=np.float32)
        dim = emb_array.shape[1]
        index = faiss_lib.IndexFlatL2(dim)
        index.add(emb_array)
        id_map = {i: str(i) for i in range(len(kept_docs))}
        docstore_dict = {str(i): kept_docs[i] for i in range(len(kept_docs))}
        docstore = InMemoryDocstore(docstore_dict)

        embedding_obj = get_embedding(embedding=embedding)
        new_vectordb = FAISS(embedding_obj, index, docstore, id_map)

        progress(0.8, desc="保存")
        save_faiss_db(new_vectordb, persist_directory, embedding_obj, embedding)
        model_center.clear_cache()
        progress(1.0)
        yield (
            f"已删除：{', '.join(deleted_labels)}。剩余 {len(kept_docs)} 个切片。",
            _status(f"完成：删除 {len(deleted_labels)} 个知识库"),
        )

    except Exception as e:
        logger.exception("delete_knowledge_base 失败: %s", e)
        yield f"删除失败：{e}", _status(f"失败：{e}")


# ════════════════════════════════════════════════════════════════
# Gradio 消息格式工具
# ════════════════════════════════════════════════════════════════

def _extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _chat_history_to_pairs(chat_history):
    if not chat_history:
        return []
    first = chat_history[0]
    if isinstance(first, (list, tuple)) and len(first) == 2:
        return [(_extract_text_from_content(u), _extract_text_from_content(a)) for u, a in chat_history]
    if isinstance(first, dict):
        pairs = []
        last_user = None
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                last_user = _extract_text_from_content(content)
            elif role == "assistant" and last_user is not None:
                pairs.append((last_user, _extract_text_from_content(content)))
                last_user = None
        return pairs
    return []


def _pairs_to_chat_messages(pairs):
    messages = []
    for user_message, bot_message in pairs:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})
    return messages


# ════════════════════════════════════════════════════════════════
# Model_center — 核心问答 + 缓存管理
# ════════════════════════════════════════════════════════════════

class Model_center:
    """
    缓存向量库与 LLM 实例，并通过 LangGraph graph.invoke() 执行 Self-RAG 问答。

    会话 thread_id 由 Gradio session state 持有，SqliteSaver 按此持久化 chat_history。
    """

    def __init__(self):
        self.vectordb_cache: dict = {}
        self.llm_cache: dict = {}

    def _normalize_model_and_embedding(self, model: str, embedding: str):
        if model in EMBEDDING_MODEL_LIST and embedding in LLM_MODEL_LIST:
            model, embedding = embedding, model
        if model not in LLM_MODEL_LIST:
            raise ValueError(f"无效 LLM: {model}")
        if embedding not in EMBEDDING_MODEL_LIST:
            raise ValueError(f"无效 Embedding: {embedding}")
        return model, embedding

    def _get_vectordb(self, persist_path: str, embedding: str):
        key = (persist_path, embedding)
        if key not in self.vectordb_cache:
            self.vectordb_cache[key] = load_faiss_vectordb(persist_path, embedding)
        return self.vectordb_cache[key]

    def _get_llm(self, model: str, temperature: float):
        key = (model, float(temperature))
        if key not in self.llm_cache:
            self.llm_cache[key] = model_to_llm(model, temperature)
        return self.llm_cache[key]

    def _resolve_source_filter(self, scope_label: str, vectordb):
        if not scope_label or scope_label == ALL_SOURCES_LABEL:
            return None
        sources = _list_sources_from_vectordb(vectordb)
        matched = {src for label, src in sources if label == scope_label}
        if not matched:
            logger.warning("scope=%s 未找到对应 source，回退全库", scope_label)
            return None
        return matched

    # ── Graph 问答核心 ────────────────────────────────────────────────────

    def _answer_with_graph(
        self,
        question: str,
        model: str,
        embedding: str,
        temperature: float,
        top_k: int,
        scope: str,
        persist_path: str,
        thread_id: str,
        history_len: int = 3,
    ) -> str:
        vectordb = self._get_vectordb(persist_path, embedding)
        llm = self._get_llm(model, temperature)
        grade_llm = llm  # 与主模型保持一致
        allowed_sources = self._resolve_source_filter(scope, vectordb)

        graph = _get_rag_graph()
        state = get_empty_state()
        state["question"] = question
        state["source_filter"] = allowed_sources
        state["embedding"] = embedding
        state["top_k"] = top_k

        config = {
            "configurable": {
                "thread_id": thread_id,
                "vectordb": vectordb,
                "llm": llm,
                "grade_llm": grade_llm,
                "retriever_kind": RETRIEVER_KIND,
                "history_len": history_len,
            }
        }

        logger.info(
            "[graph问答] model=%s embedding=%s top_k=%d scope=%s question=%r thread=%s",
            model, embedding, top_k, scope, question, thread_id,
        )

        try:
            result = graph.invoke(state, config=config)
            answer = result.get("generation") or "（无回答，请检查向量库是否为空）"
        except Exception as e:
            logger.exception("[graph问答] 失败: %s", e)
            answer = f"问答失败：{e}"

        return answer

    # ── 带历史 ────────────────────────────────────────────────────────────

    def _chat_qa_answer_step(
        self,
        question: str,
        chat_history,
        model: str = INIT_LLM,
        embedding: str = INIT_EMBEDDING_MODEL,
        temperature: float = 0.0,
        top_k: int = 4,
        history_len: int = 3,
        scope: str = ALL_SOURCES_LABEL,
        persist_path: str = None,
        thread_id: str = None,
    ):
        """带历史问答：chatbot 中已含用户消息，仅追加助手回答。"""
        if not question:
            return chat_history or []
        try:
            model, embedding = self._normalize_model_and_embedding(model, embedding)
            if persist_path is None:
                persist_path = get_persist_path(embedding)
            if not thread_id:
                thread_id = str(uuid.uuid4())

            answer = self._answer_with_graph(
                question=question,
                model=model,
                embedding=embedding,
                temperature=temperature,
                top_k=top_k,
                scope=scope,
                persist_path=persist_path,
                thread_id=thread_id,
                history_len=history_len,
            )
            history = list(chat_history or [])
            history.append({"role": "assistant", "content": answer})
            return history
        except Exception as e:
            logger.exception("_chat_qa_answer_step 失败: %s", e)
            history = list(chat_history or [])
            history.append({"role": "assistant", "content": str(e)})
            return history

    # ── 无历史 ────────────────────────────────────────────────────────────

    def _qa_answer_step(
        self,
        question: str,
        chat_history,
        model: str = INIT_LLM,
        embedding: str = INIT_EMBEDDING_MODEL,
        temperature: float = 0.0,
        top_k: int = 4,
        scope: str = ALL_SOURCES_LABEL,
        persist_path: str = None,
    ):
        """无历史问答：每次使用新的随机 thread_id（无状态）。"""
        if not question:
            return chat_history or []
        try:
            model, embedding = self._normalize_model_and_embedding(model, embedding)
            if persist_path is None:
                persist_path = get_persist_path(embedding)

            thread_id = str(uuid.uuid4())  # 无历史：每次新会话
            answer = self._answer_with_graph(
                question=question,
                model=model,
                embedding=embedding,
                temperature=temperature,
                top_k=top_k,
                scope=scope,
                persist_path=persist_path,
                thread_id=thread_id,
                history_len=0,
            )
            history = list(chat_history or [])
            history.append({"role": "assistant", "content": answer})
            return history
        except Exception as e:
            logger.exception("_qa_answer_step 失败: %s", e)
            history = list(chat_history or [])
            history.append({"role": "assistant", "content": str(e)})
            return history

    def clear_cache(self):
        self.vectordb_cache.clear()
        self.llm_cache.clear()
        if RETRIEVER_KIND == "hybrid":
            try:
                from qa_chain.hybrid_retriever import clear_bm25_cache
                clear_bm25_cache()
            except ImportError:
                pass
        clear_embedding_cache()
        logger.info("Model_center 缓存已清空")


# ════════════════════════════════════════════════════════════════
# Make Markdown 读书笔记（逻辑不变）
# ════════════════════════════════════════════════════════════════

def make_markdown(
    chat_history,
    note_title: str,
    llm: str,
    temperature: float = 0.1,
    progress=gr.Progress(track_tqdm=False),
):
    import datetime

    _hide_title_error = gr.update(value="", visible=False)

    def _show_title_err(msg):
        return gr.update(
            value=f'<p style="color: #d32f2f; font-weight: bold; margin: 4px 0 0 0;">错误：{msg}</p>',
            visible=True,
        )

    progress(0.0, desc="检查输入")
    if not chat_history:
        yield "", "错误：对话记录为空。", _show_title_err("对话记录为空")
        return
    note_title = (note_title or "").strip()
    if not note_title:
        yield "", "错误：请填写标题。", _show_title_err("请填写标题")
        return

    chat_history = _chat_history_to_pairs(chat_history)
    if not chat_history:
        yield "", "错误：对话记录为空。", _show_title_err("对话记录为空")
        return

    progress(0.2, desc="整理对话")
    yield "", "整理对话记录...", _hide_title_error

    qa_lines = []
    for i, (user_msg, bot_msg) in enumerate(chat_history, 1):
        qa_lines.extend([f"**Q{i}:** {user_msg}\n", f"**A{i}:** {bot_msg}\n", ""])
    qa_section = "\n".join(qa_lines)

    progress(0.4, desc="调用 LLM 生成摘要")
    yield "", "调用 LLM 生成摘要...", _hide_title_error

    summary_section = ""
    summary_error = ""
    try:
        conversation_text = "\n".join(
            f"Q{i}: {u}\nA{i}: {a}" for i, (u, a) in enumerate(chat_history, 1)
        )
        summarize_prompt = (
            "请根据以下对话记录，整理成结构化读书笔记，包含：主要知识点、关键概念、重要结论，"
            "使用 Markdown 格式。\n\n对话记录：\n" + conversation_text
        )
        summary_section = get_completion(summarize_prompt, llm, temperature=temperature)
    except Exception as e:
        summary_error = f"（LLM 摘要生成失败：{e}）"
        logger.warning("make_markdown LLM 失败: %s", e)

    progress(0.8, desc="保存笔记")
    yield "", "保存笔记...", _hide_title_error

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content = f"# {note_title}\n*生成时间: {now_str}*\n\n---\n\n## 对话记录\n\n{qa_section}\n---\n\n## AI 总结笔记\n\n{summary_section}{summary_error}\n"

    notes_dir = os.path.join(ROOT, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    safe_name = re.sub(r'[\\/:*?"<>|\s]+', "_", note_title).strip("_") or "note"
    candidate = os.path.join(notes_dir, f"{safe_name}.md")
    if os.path.exists(candidate):
        suffix = 1
        while os.path.exists(os.path.join(notes_dir, f"{safe_name}_{suffix}.md")):
            suffix += 1
        candidate = os.path.join(notes_dir, f"{safe_name}_{suffix}.md")

    with open(candidate, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    rel_path = os.path.relpath(candidate, ROOT)
    progress(1.0)
    yield markdown_content, f"已保存：{rel_path}", _hide_title_error


def _list_note_files():
    notes_dir = os.path.join(ROOT, "notes")
    if not os.path.exists(notes_dir):
        return []
    return sorted(f for f in os.listdir(notes_dir) if f.endswith(".md"))


def _load_note_content(filename: str) -> str:
    if not filename:
        return ""
    fp = os.path.join(ROOT, "notes", filename)
    if not os.path.isfile(fp):
        return f"（找不到文件：{filename}）"
    with open(fp, "r", encoding="utf-8") as f:
        return f.read()


def _save_note_content(filename: str, content: str) -> str:
    if not filename:
        return "错误：未选择文件。"
    if os.path.basename(filename) != filename or ".." in filename:
        return "错误：非法文件名。"
    notes_dir = os.path.join(ROOT, "notes")
    fp = os.path.join(notes_dir, filename)
    if not os.path.abspath(fp).startswith(os.path.abspath(notes_dir) + os.sep):
        return "错误：非法路径。"
    try:
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"已保存：{filename}"
    except Exception as e:
        return f"保存失败：{e}"


def _open_preview_modal():
    files = _list_note_files()
    first = files[0] if files else None
    content = _load_note_content(first) if first else ""
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(choices=files, value=first),
        content,
    )


# ── 辅助 UI 函数 ─────────────────────────────────────────────────────────

def _disable_vectorize_button():
    return gr.update(interactive=False, value="向量化中...")


def _enable_vectorize_button():
    return gr.update(interactive=True, value="知识库文件向量化")


def _disable_delete_button():
    return gr.update(interactive=False, value="删除中...")


def _enable_delete_button():
    return gr.update(interactive=True, value="删除所选知识库")


def _prefill_user_msg(question: str, chat_history):
    q = (question or "").strip()
    if not q:
        return question, "", chat_history or []
    history = list(chat_history or [])
    history.append({"role": "user", "content": q})
    return "", q, history


model_center = Model_center()

# ════════════════════════════════════════════════════════════════
# Gradio UI
# ════════════════════════════════════════════════════════════════

_MODAL_CSS = """
#note-preview-modal {
    position: fixed !important;
    top: 5vh !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(90vw, 880px) !important;
    max-height: 88vh !important;
    overflow-y: auto !important;
    z-index: 10000 !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 48px rgba(0,0,0,0.55) !important;
    background: var(--background-fill-primary, #fff) !important;
    padding: 24px !important;
    box-sizing: border-box !important;
}
"""

with gr.Blocks(css=_MODAL_CSS) as demo:
    # session-level thread_id：SqliteSaver 按此隔离每个浏览器标签的对话历史
    _thread_id_state = gr.State(str(uuid.uuid4()))

    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            gr.Markdown("""<h1><center>个人知识库小助手（LangGraph Self-RAG）</center></h1>
                <center>Self-RAG 图拓扑：检索 → 文档评分 → 生成 → 答案验证 | 会话持久化</center>
            """)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450)
            msg = gr.Textbox(label="Prompt / 问题")

            with gr.Row():
                scope_dropdown = gr.Dropdown(
                    choices=get_source_choices(),
                    value=ALL_SOURCES_LABEL,
                    label="知识库范围",
                    info="限定检索范围，避免不同知识库串味",
                    interactive=True,
                )

            with gr.Row():
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
            with gr.Row():
                note_title = gr.Textbox(
                    label="读书笔记标题",
                    placeholder="输入读书笔记标题...",
                    scale=4,
                )
                make_markdown_btn = gr.Button("Make Markdown", scale=1)
                preview_note_btn = gr.Button(
                    "预览笔记",
                    scale=1,
                    interactive=bool(_list_note_files()),
                )
            note_title_error = gr.HTML(value="", visible=False)
            with gr.Row():
                clear = gr.ClearButton(components=[chatbot], value="Clear console")

        with gr.Column(scale=2):
            file = gr.File(
                label="请选择知识库目录",
                file_count="directory",
                file_types=[".txt", ".md", ".docx", ".pdf"],
            )
            rebuild_checkbox = gr.Checkbox(
                label="重建本次上传文件（仅替换同名 source 的向量，保留其他知识库）",
                value=False,
            )
            with gr.Row():
                init_db = gr.Button("知识库文件向量化")
            vectorize_status = gr.Textbox(
                label="向量化任务状态",
                value="待执行",
                interactive=False,
                lines=2,
            )

            with gr.Accordion("知识库管理", open=False):
                delete_source_selector = gr.Dropdown(
                    choices=get_deletable_sources(),
                    label="选择要删除的知识库",
                    multiselect=True,
                    interactive=True,
                )
                with gr.Row():
                    refresh_delete_list_btn = gr.Button("刷新列表", size="sm")
                    delete_kb_btn = gr.Button("删除所选知识库", variant="stop", size="sm")
                delete_status = gr.Textbox(label="删除状态", interactive=False, lines=2)

            with gr.Accordion("参数配置", open=False):
                temperature = gr.Slider(0, 1, value=0.01, step=0.01, label="llm temperature")
                top_k = gr.Slider(1, 20, value=6, step=1, label="top k")
                history_len = gr.Slider(0, 5, value=3, step=1, label="history length")

            with gr.Accordion("模型选择"):
                llm = gr.Dropdown(LLM_MODEL_LIST, label="large language model", value=INIT_LLM)
                embeddings = gr.Dropdown(
                    EMBEDDING_MODEL_LIST,
                    label="Embedding model",
                    value=INIT_EMBEDDING_MODEL,
                )

            notes_status = gr.Textbox(label="笔记保存状态", interactive=False, lines=2)

    _modal_backdrop = gr.HTML(
        value='<div style="position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.45);z-index:9998;"></div>',
        visible=False,
    )
    with gr.Column(visible=False, elem_id="note-preview-modal") as preview_modal:
        gr.Markdown("## 笔记预览")
        note_file_selector = gr.Dropdown(label="选择笔记文件", choices=_list_note_files(), interactive=True)
        note_modal_content = gr.Textbox(label="笔记内容（可编辑）", lines=20, interactive=True)
        modal_save_status = gr.Textbox(label="保存状态", interactive=False, lines=1)
        with gr.Row():
            save_note_btn = gr.Button("保存修改", variant="primary")
            close_preview_btn = gr.Button("关闭预览")

    _md_content_state = gr.State("")
    _pending_q = gr.State("")

    # ── 事件绑定 ──────────────────────────────────────────────────────────

    init_db.click(_disable_vectorize_button, outputs=[init_db], queue=False).then(
        build_faiss_db_info,
        inputs=[file, embeddings, rebuild_checkbox],
        outputs=[msg, vectorize_status],
        show_progress="full",
    ).then(refresh_source_dropdown, inputs=[embeddings], outputs=[scope_dropdown], queue=False).then(
        _enable_vectorize_button, outputs=[init_db], queue=False
    )

    db_with_his_btn.click(
        _prefill_user_msg,
        inputs=[msg, chatbot],
        outputs=[msg, _pending_q, chatbot],
        queue=False,
    ).then(
        model_center._chat_qa_answer_step,
        inputs=[_pending_q, chatbot, llm, embeddings, temperature, top_k, history_len,
                scope_dropdown, gr.State(None), _thread_id_state],
        outputs=[chatbot],
    )

    db_wo_his_btn.click(
        _prefill_user_msg,
        inputs=[msg, chatbot],
        outputs=[msg, _pending_q, chatbot],
        queue=False,
    ).then(
        model_center._qa_answer_step,
        inputs=[_pending_q, chatbot, llm, embeddings, temperature, top_k, scope_dropdown],
        outputs=[chatbot],
    )

    make_markdown_btn.click(
        make_markdown,
        inputs=[chatbot, note_title, llm, temperature],
        outputs=[_md_content_state, notes_status, note_title_error],
        show_progress="full",
    ).then(lambda: gr.update(interactive=bool(_list_note_files())), outputs=[preview_note_btn])

    preview_note_btn.click(
        _open_preview_modal,
        outputs=[_modal_backdrop, preview_modal, note_file_selector, note_modal_content],
    )
    note_file_selector.change(_load_note_content, inputs=[note_file_selector], outputs=[note_modal_content])
    save_note_btn.click(_save_note_content, inputs=[note_file_selector, note_modal_content], outputs=[modal_save_status])
    close_preview_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[_modal_backdrop, preview_modal],
    )
    clear.click(lambda: str(uuid.uuid4()), outputs=[_thread_id_state])  # 清空对话时刷新 thread_id

    embeddings.change(lambda: model_center.clear_cache(), queue=False).then(
        on_embedding_change,
        inputs=[embeddings],
        outputs=[scope_dropdown, delete_source_selector, vectorize_status],
        queue=False,
    )

    refresh_delete_list_btn.click(refresh_delete_dropdown, inputs=[embeddings], outputs=[delete_source_selector], queue=False)

    delete_kb_btn.click(_disable_delete_button, outputs=[delete_kb_btn], queue=False).then(
        delete_knowledge_base,
        inputs=[delete_source_selector, embeddings],
        outputs=[msg, delete_status],
        show_progress="full",
    ).then(refresh_source_dropdown, inputs=[embeddings], outputs=[scope_dropdown], queue=False).then(
        refresh_delete_dropdown, inputs=[embeddings], outputs=[delete_source_selector], queue=False
    ).then(_enable_delete_button, outputs=[delete_kb_btn], queue=False)

    gr.Markdown("""
    **使用说明：**
    1. 上传知识库目录 → 「知识库文件向量化」初始化
    2. 「Chat db with history」：Self-RAG 图 + SqliteSaver 历史（重启后保留）
    3. 「Chat db without history」：每次新 thread，无历史
    4. 图节点日志：terminal 可见 grade_documents / generate / grade_answer 过程
    5. 「知识库范围」：限定检索在所选文档内进行
    6. 「Make Markdown」：整理对话为读书笔记
    """)

gr.close_all()
port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
demo.queue(default_concurrency_limit=2).launch(server_name="0.0.0.0")
