"""向量库切片诊断（不调 LLM、不加载 embedding）。

直接读 FAISS 的 index.pkl，避免 1.2GB 模型加载开销，
专门用于回答"某页 / 某关键词 / 某 lesson marker 在向量库里到底是什么样子"。

CLI 用法（通过 cli.py 暴露）：
    python -m evaluation.cli run --dataset pumpkin_book --embedding m3e --diagnose
    # 评估完成后，对每个未命中或 mrr<0.5 的 case，
    # 自动 dump 其 expected_pages 范围的所有切片内容
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class ChunkInfo:
    """单个切片的诊断信息。"""

    doc_id: str
    page: Optional[int]
    source: Optional[str]
    content_len: int
    has_lesson_marker: bool
    has_kaiwa_marker: bool
    has_jp_kaiwa: bool
    preview_200: str


def load_docstore(persist_path: str | Path) -> Dict[str, Any]:
    """从 index.pkl 加载 docstore dict，不加载 embedding 模型。"""
    pkl_path = Path(persist_path) / "index.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"index.pkl 不存在: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, tuple) and len(data) >= 2:
        docstore = data[0]
        return getattr(docstore, "_dict", {}) or {}
    raise ValueError(f"index.pkl 格式不符合预期: {type(data)}")


def _parse_chunk(doc_id: str, doc: Any) -> ChunkInfo:
    meta = getattr(doc, "metadata", None) or {}
    content = getattr(doc, "page_content", "") or ""
    page_raw = meta.get("page")
    try:
        page = int(page_raw) if page_raw is not None else None
    except (TypeError, ValueError):
        page = None
    import re
    lesson_marker = bool(re.search(r"【第\d+[章課课]】", content))
    kaiwa_cn = "【会话/对话】" in content
    jp_kaiwa = "会話" in content
    return ChunkInfo(
        doc_id=doc_id,
        page=page,
        source=meta.get("source"),
        content_len=len(content),
        has_lesson_marker=lesson_marker,
        has_kaiwa_marker=kaiwa_cn,
        has_jp_kaiwa=jp_kaiwa,
        preview_200=content[:200].replace("\n", " "),
    )


def chunks_by_pages(
    docstore: Dict[str, Any],
    pages: Iterable[int],
    source_filter: Optional[str] = None,
) -> List[ChunkInfo]:
    """返回 page 在 pages 集合中的所有切片（按 page 排序）。"""
    target = set(pages)
    results: List[ChunkInfo] = []
    for doc_id, doc in docstore.items():
        info = _parse_chunk(doc_id, doc)
        if info.page not in target:
            continue
        if source_filter:
            src = os.path.basename(str(info.source or ""))
            if source_filter not in src:
                continue
        results.append(info)
    results.sort(key=lambda c: (c.page or -1, c.doc_id))
    return results


def chunks_by_keyword(
    docstore: Dict[str, Any],
    keyword: str,
    source_filter: Optional[str] = None,
) -> List[ChunkInfo]:
    """返回 content 包含 keyword 的所有切片（按 page 排序）。"""
    results: List[ChunkInfo] = []
    for doc_id, doc in docstore.items():
        content = getattr(doc, "page_content", "") or ""
        if keyword not in content:
            continue
        info = _parse_chunk(doc_id, doc)
        if source_filter:
            src = os.path.basename(str(info.source or ""))
            if source_filter not in src:
                continue
        results.append(info)
    results.sort(key=lambda c: (c.page or -1, c.doc_id))
    return results


def summarize_markers(
    docstore: Dict[str, Any],
    source_filter: Optional[str] = None,
) -> str:
    """统计各类 marker 的注入情况，输出诊断摘要。"""
    total = 0
    with_lesson = 0
    with_kaiwa = 0
    pages_seen: set = set()
    for doc_id, doc in docstore.items():
        meta = getattr(doc, "metadata", None) or {}
        src = os.path.basename(str(meta.get("source", "")))
        if source_filter and source_filter not in src:
            continue
        info = _parse_chunk(doc_id, doc)
        total += 1
        if info.has_lesson_marker:
            with_lesson += 1
        if info.has_kaiwa_marker or info.has_jp_kaiwa:
            with_kaiwa += 1
        if info.page is not None:
            pages_seen.add(info.page)

    lines = [
        f"总切片数: {total}",
        f"含课程标记 【第N章/課】: {with_lesson} ({with_lesson/max(total,1)*100:.1f}%)",
        f"含会话关键词: {with_kaiwa}",
        f"页码范围: {min(pages_seen) if pages_seen else '?'} ~ {max(pages_seen) if pages_seen else '?'}",
        f"共 {len(pages_seen)} 页",
    ]
    return "\n".join(lines)


def diagnose_case(
    persist_path: str,
    expected_pages: List[int],
    source_filter: Optional[str] = None,
    neighborhood: int = 1,
) -> str:
    """对一个 case 的 expected_pages 周围切片做诊断，返回可读文本。"""
    try:
        docstore = load_docstore(persist_path)
    except Exception as e:
        return f"(load_docstore 失败: {e})"

    pages_to_check: set = set()
    for p in expected_pages:
        for delta in range(-neighborhood, neighborhood + 1):
            pages_to_check.add(p + delta)

    chunks = chunks_by_pages(docstore, pages_to_check, source_filter=source_filter)
    if not chunks:
        return f"(expected_pages {expected_pages} 附近（±{neighborhood}）无切片)"

    lines = [
        f"expected_pages={expected_pages}，展开 ±{neighborhood} 邻域，共 {len(chunks)} 个切片：",
        "",
    ]
    for c in chunks:
        marker_tag = "[marker✓]" if c.has_lesson_marker else "[no-marker]"
        lines.append(
            f"  page={c.page} {marker_tag} len={c.content_len} | {c.preview_200[:120]}"
        )
    return "\n".join(lines)
