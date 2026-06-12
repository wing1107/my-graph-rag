"""检索层指标。

设计原则：
- 所有函数都是 pure function：输入 retrieved（list of RetrievedDoc 或同构 dict）
  + expected（GoldenCase），输出 float 或简单结构
- 不依赖 langchain / faiss / numpy，方便在 test/ 里无外部依赖测试
- 命中判定（_matches）作为唯一真理来源，三种判定方式 OR 关系：
    a) retrieved.page in expected.expected_pages
    b) retrieved.content_preview 包含 expected.expected_keywords_any 任一项
    c) retrieved.source 文件名包含 expected.expected_source

注意：
- mrr 只计算"第一个命中切片的排名倒数"，未命中时为 0.0
- recall_at_k 在 expected_pages 为空时回退为 hit_at_k（避免除 0），
  这样 cross_lang_kaiwa_keyword 这种"只关心关键词命中"的 case 也能给出合理分数
"""

from __future__ import annotations

from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _as_dict(obj: Any) -> Dict[str, Any]:
    """支持 dataclass / dict / 任意带属性对象的统一访问。"""
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    return {
        k: getattr(obj, k)
        for k in dir(obj)
        if not k.startswith("_") and not callable(getattr(obj, k, None))
    }


def _basename_contains(source: Optional[str], needle: Optional[str]) -> bool:
    if not source or not needle:
        return False
    import os as _os
    return needle in _os.path.basename(str(source))


def _matches(retrieved: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """单个 retrieved 切片是否算"命中" expected。

    三种判定方式 OR 关系；若 expected_source 提供，会作为"必要条件"额外检查：
    即必须先满足 source（如指定了），再满足 page/keyword 任一。
    """
    src = retrieved.get("source")
    expected_src = expected.get("expected_source")
    if expected_src and not _basename_contains(src, expected_src):
        return False

    pages = expected.get("expected_pages") or []
    if pages:
        page = retrieved.get("page")
        if page is not None and page in pages:
            return True

    keywords = expected.get("expected_keywords_any") or []
    if keywords:
        content = (retrieved.get("content_preview") or "") + (retrieved.get("content") or "")
        for kw in keywords:
            if kw and kw in content:
                return True

    if expected_src and not pages and not keywords:
        return True

    return False


def matched_ranks(
    retrieved: Sequence[Any],
    expected: Any,
) -> List[int]:
    """返回所有命中切片的 rank 列表（1-indexed），按出现顺序。"""
    exp = _as_dict(expected)
    ranks: List[int] = []
    for i, doc in enumerate(retrieved, start=1):
        d = _as_dict(doc)
        if _matches(d, exp):
            ranks.append(i)
    return ranks


def hit_at_k(retrieved: Sequence[Any], expected: Any, k: Optional[int] = None) -> float:
    """top-k 内是否至少命中 1 个 expected 切片（0.0 或 1.0）。"""
    limit = k if k is not None else len(retrieved)
    for i, doc in enumerate(retrieved[:limit], start=1):
        if _matches(_as_dict(doc), _as_dict(expected)):
            return 1.0
    return 0.0


def recall_at_k(retrieved: Sequence[Any], expected: Any, k: Optional[int] = None) -> float:
    """top-k 内命中的 expected_pages 占比。

    若 expected_pages 为空，回退到 hit_at_k（对纯关键词 case 给出合理分数）。
    """
    exp = _as_dict(expected)
    expected_pages = exp.get("expected_pages") or []
    if not expected_pages:
        return hit_at_k(retrieved, expected, k)

    limit = k if k is not None else len(retrieved)
    seen_pages = set()
    for doc in retrieved[:limit]:
        d = _as_dict(doc)
        if not _matches(d, exp):
            continue
        page = d.get("page")
        if page in expected_pages:
            seen_pages.add(page)
    return len(seen_pages) / len(expected_pages)


def mrr(retrieved: Sequence[Any], expected: Any, k: Optional[int] = None) -> float:
    """第一个命中切片的排名倒数。"""
    ranks = matched_ranks(retrieved, expected)
    if not ranks:
        return 0.0
    limit = k if k is not None else len(retrieved)
    in_window = [r for r in ranks if r <= limit]
    if not in_window:
        return 0.0
    return 1.0 / in_window[0]


def page_overlap(retrieved: Sequence[Any], expected: Any, k: Optional[int] = None) -> float:
    """top-k 中 page in expected_pages 的切片占比（不要求 source/keyword 同时命中）。

    若 expected_pages 为空，返回 0.0（无对比基准）。
    """
    exp = _as_dict(expected)
    expected_pages = exp.get("expected_pages") or []
    if not expected_pages:
        return 0.0
    limit = k if k is not None else len(retrieved)
    if limit == 0:
        return 0.0
    hits = sum(
        1
        for doc in retrieved[:limit]
        if _as_dict(doc).get("page") in expected_pages
    )
    return hits / limit


def keyword_hit_rate(retrieved: Sequence[Any], expected: Any, k: Optional[int] = None) -> float:
    """top-k 中包含 expected_keywords_any 任一项的切片占比。"""
    exp = _as_dict(expected)
    keywords = exp.get("expected_keywords_any") or []
    if not keywords:
        return 0.0
    limit = k if k is not None else len(retrieved)
    if limit == 0:
        return 0.0
    hits = 0
    for doc in retrieved[:limit]:
        d = _as_dict(doc)
        content = (d.get("content_preview") or "") + (d.get("content") or "")
        if any(kw and kw in content for kw in keywords):
            hits += 1
    return hits / limit


def compute_all(
    retrieved: Sequence[Any],
    expected: Any,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """一次性算完所有检索层指标。"""
    return {
        "hit_at_k": hit_at_k(retrieved, expected, k),
        "recall_at_k": recall_at_k(retrieved, expected, k),
        "mrr": mrr(retrieved, expected, k),
        "page_overlap": page_overlap(retrieved, expected, k),
        "keyword_hit_rate": keyword_hit_rate(retrieved, expected, k),
    }


def aggregate(case_metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """跨 case 平均（macro average）。"""
    case_metrics = list(case_metrics)
    if not case_metrics:
        return {}
    keys = set()
    for cm in case_metrics:
        keys.update(cm.keys())
    out: Dict[str, float] = {}
    for k in sorted(keys):
        values = [cm.get(k, 0.0) for cm in case_metrics]
        out[f"avg_{k}"] = sum(values) / len(values)
    out["n_cases"] = float(len(case_metrics))
    return out
