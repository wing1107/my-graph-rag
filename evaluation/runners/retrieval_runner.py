"""单个黄金集 → 检索 → 指标 → EvalResult。

不调 LLM，只跑 FAISS 检索。性能基线 retrieval_latency_ms 直接在这里采集。

复用：
- evaluation/utils.py::load_vectordb（Windows 中文路径 ASCII fallback 已处理）
- evaluation/metrics/retrieval.py 的纯函数指标

预留扩展点：
- run_dataset() 返回 EvalResult 对象（包含完整 retrieved_docs），
  让未来的 generation_runner 可以链式调用，不需要重跑检索
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.metrics import retrieval as M  # noqa: E402
from evaluation.schemas import (  # noqa: E402
    CaseResult,
    DEFAULT_TOP_K,
    EvalMeta,
    EvalResult,
    GoldenCase,
    GoldenDataset,
    RetrievedDoc,
)


def _resolve_persist_path(embedding: str, persist_path: Optional[str] = None) -> str:
    """优先用显式参数，否则 data_base/vector_db/faiss_{embedding}/。"""
    if persist_path:
        return persist_path
    return str(_PROJECT_ROOT / "data_base" / "vector_db" / f"faiss_{embedding}")


def _build_source_filter(source_filter: Optional[str]) -> Optional[Callable]:
    """把"source 文件名包含 substring"转成 FAISS 用的 metadata filter 闭包。"""
    if not source_filter:
        return None

    def _filter(meta: dict) -> bool:
        src = (meta or {}).get("source", "")
        return source_filter in os.path.basename(str(src))

    return _filter


def _search_dense(vectordb, query: str, top_k: int, source_filter: Optional[Callable]):
    """走 FAISS similarity_search_with_score；带 filter 时扩大 fetch_k。"""
    if source_filter is None:
        return vectordb.similarity_search_with_score(query, k=top_k)
    fetch_k = max(200, top_k * 20)
    return vectordb.similarity_search_with_score(
        query, k=top_k, filter=source_filter, fetch_k=fetch_k
    )


def _search_hybrid(
    query: str,
    top_k: int,
    hybrid_retriever: Any,
) -> List[tuple]:
    """跑一次 EnsembleRetriever（dense+BM25）；返回 [(doc, score), ...]。

    EnsembleRetriever 不暴露融合分数，这里用 1/rank 占位，保持和
    FAISS similarity_search_with_score 的 (doc, score) 契约一致。
    """
    docs = hybrid_retriever.invoke(query)
    docs = list(docs)[:top_k]
    return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]


def _run_case(
    vectordb,
    case: GoldenCase,
    default_top_k: int,
    source_filter: Optional[Callable],
    retriever_kind: str = "dense",
    hybrid_retriever: Any = None,
) -> CaseResult:
    top_k = case.top_k_override or default_top_k

    t0 = time.perf_counter()
    try:
        if retriever_kind == "hybrid":
            if hybrid_retriever is None:
                raise ValueError("retriever_kind=hybrid 需要传入 hybrid_retriever")
            results = _search_hybrid(case.query, top_k, hybrid_retriever)
        else:
            results = _search_dense(vectordb, case.query, top_k, source_filter)
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            query=case.query,
            top_k=top_k,
            retrieved=[],
            metrics={},
            matched_ranks=[],
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    retrieved: List[RetrievedDoc] = [
        RetrievedDoc.from_langchain(rank=i + 1, score=score, doc=doc)
        for i, (doc, score) in enumerate(results)
    ]

    expected_dict = {
        "expected_pages": case.expected_pages,
        "expected_keywords_any": case.expected_keywords_any,
        "expected_source": case.expected_source,
    }
    metrics = M.compute_all(retrieved, expected_dict, k=top_k)
    metrics["retrieval_latency_ms"] = round(elapsed_ms, 2)
    ranks = M.matched_ranks(retrieved, expected_dict)

    return CaseResult(
        case_id=case.id,
        query=case.query,
        top_k=top_k,
        retrieved=retrieved,
        metrics=metrics,
        matched_ranks=ranks,
        error=None,
    )


def run_dataset(
    dataset_path: str | Path,
    embedding: str,
    top_k: int = DEFAULT_TOP_K,
    persist_path: Optional[str] = None,
    notes: Optional[str] = None,
    retriever: str = "dense",
    hybrid_weights: Sequence[float] = (0.5, 0.5),
) -> EvalResult:
    """跑一个黄金集 → 返回 EvalResult。

    不写文件；写报告由 CLI 层负责，便于 matrix_runner / pytest 复用。

    Parameters
    ----------
    retriever : str
        "dense"（默认）= FAISS similarity_search；
        "hybrid" = FAISS + BM25(jieba) 通过 EnsembleRetriever 融合，
        对中文章节号、专有名词类查询通常显著优于纯 dense。
    hybrid_weights : (float, float)
        仅 retriever="hybrid" 生效；(dense_weight, bm25_weight)。
    """
    from evaluation.utils import load_vectordb

    if retriever not in ("dense", "hybrid"):
        raise ValueError(f"retriever 必须是 'dense' 或 'hybrid'，收到: {retriever!r}")

    persist = _resolve_persist_path(embedding, persist_path)
    dataset = GoldenDataset.from_yaml(dataset_path)

    vectordb = load_vectordb(embedding_provider=embedding, db_path=persist)
    source_filter = _build_source_filter(dataset.source_filter)

    hybrid_retriever = None
    if retriever == "hybrid":
        from qa_chain.hybrid_retriever import build_hybrid_retriever

        # 用所有 case 中最大的 top_k 来构建 retriever，确保 top_k_override 生效
        max_top_k = max(
            [top_k] + [c.top_k_override for c in dataset.cases if c.top_k_override]
        )
        hybrid_retriever = build_hybrid_retriever(
            vectordb=vectordb,
            top_k=max_top_k,
            weights=hybrid_weights,
            source_filter=dataset.source_filter,
        )

    cases: List[CaseResult] = []
    for case in dataset.cases:
        cases.append(
            _run_case(
                vectordb,
                case,
                top_k,
                source_filter,
                retriever_kind=retriever,
                hybrid_retriever=hybrid_retriever,
            )
        )

    valid_metrics = [c.metrics for c in cases if c.metrics and not c.error]
    agg = M.aggregate(valid_metrics)
    agg["n_errors"] = float(sum(1 for c in cases if c.error))

    meta = EvalMeta(
        dataset=dataset.dataset,
        embedding=embedding,
        top_k=top_k,
        persist_path=persist,
        retriever=retriever,
        notes=notes,
    )
    return EvalResult(meta=meta, cases=cases, aggregate=agg)


def _render_case_markdown(case: CaseResult, preview_top: int = 3) -> str:
    lines: List[str] = []
    lines.append(f"### Case `{case.case_id}`")
    lines.append("")
    lines.append(f"- query: `{case.query}`")
    lines.append(f"- top_k: {case.top_k}")
    if case.error:
        lines.append(f"- **ERROR**: {case.error}")
        return "\n".join(lines)

    lines.append(f"- matched_ranks: {case.matched_ranks or '(none)'}")
    lines.append("")
    lines.append("**Metrics**:")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in case.metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append(f"**Top {min(preview_top, len(case.retrieved))} retrieved**:")
    lines.append("")
    lines.append("| rank | score | page | source | preview |")
    lines.append("|---|---|---|---|---|")
    for d in case.retrieved[:preview_top]:
        src = os.path.basename(d.source or "") or "?"
        preview = (d.content_preview or "").replace("|", "\\|").replace("\n", " ")[:80]
        lines.append(f"| {d.rank} | {d.score:.4f} | {d.page} | {src} | {preview} |")
    lines.append("")
    return "\n".join(lines)


def render_markdown(result: EvalResult) -> str:
    """把 EvalResult 渲染成人类可读的 markdown 报告。"""
    lines: List[str] = []
    m = result.meta
    lines.append(f"# 评估报告: {m.dataset} @ {m.embedding}")
    lines.append("")
    lines.append(f"- timestamp: {m.timestamp}")
    lines.append(f"- top_k: {m.top_k}")
    lines.append(f"- retriever: {m.retriever}")
    lines.append(f"- persist_path: `{m.persist_path}`")
    if m.notes:
        lines.append(f"- notes: {m.notes}")
    lines.append("")
    lines.append("## 总体指标（macro average）")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in result.aggregate.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## 各 case 明细")
    lines.append("")
    for case in result.cases:
        lines.append(_render_case_markdown(case))
    return "\n".join(lines)
