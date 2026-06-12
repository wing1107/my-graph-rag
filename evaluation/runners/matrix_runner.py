"""参数矩阵评估：datasets × embeddings × top_ks × retrievers 笛卡尔积。

用途：回答"dense vs hybrid 哪个更好？top_k 该设多少？"这类参数选择问题，
避免拍脑袋。

注意：datasets 和 embeddings 必须匹配——两个向量库各自对应一个知识库：
  faiss_m3e         → 南瓜书 (pumpkin_book)
  faiss_multilingual → 大家的日语 初级1 (lesson25)
跨 embedding 对比无意义（向量库里没有对方的数据）。
典型用法：
  --datasets pumpkin_book --embeddings m3e --retrievers dense,hybrid
  --datasets lesson25     --embeddings multilingual --retrievers dense,hybrid
"""

from __future__ import annotations

import os
import sys
from itertools import product
from pathlib import Path
from typing import List, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.runners.retrieval_runner import run_dataset
from evaluation.schemas import EvalResult


def run_matrix(
    dataset_paths: Sequence[str],
    embeddings: Sequence[str],
    top_ks: Sequence[int],
    retrievers: Sequence[str] = ("dense",),
) -> List[EvalResult]:
    """对所有组合跑 retrieval_runner，返回 EvalResult 列表。

    若某 embedding 对应的向量库不存在，会写入一个空 EvalResult（aggregate 含 error 字段），
    不会让整个 matrix 中断。
    """
    from evaluation.schemas import EvalMeta

    results: List[EvalResult] = []
    for dataset_path, embedding, top_k, retriever in product(
        dataset_paths, embeddings, top_ks, retrievers
    ):
        persist = _PROJECT_ROOT / "data_base" / "vector_db" / f"faiss_{embedding}"
        if not (persist / "index.faiss").exists():
            dataset_name = Path(dataset_path).stem
            meta = EvalMeta(
                dataset=dataset_name,
                embedding=embedding,
                top_k=top_k,
                persist_path=str(persist),
                retriever=retriever,
                notes="SKIPPED: 向量库不存在",
            )
            results.append(EvalResult(meta=meta, cases=[], aggregate={"error_skipped": 1.0}))
            continue
        try:
            r = run_dataset(
                dataset_path=dataset_path,
                embedding=embedding,
                top_k=top_k,
                retriever=retriever,
            )
            results.append(r)
        except Exception as exc:
            dataset_name = Path(dataset_path).stem
            meta = EvalMeta(
                dataset=dataset_name,
                embedding=embedding,
                top_k=top_k,
                persist_path=str(persist),
                retriever=retriever,
                notes=f"FAILED: {type(exc).__name__}: {exc}",
            )
            results.append(EvalResult(meta=meta, cases=[], aggregate={"error_failed": 1.0}))
    return results


def render_matrix_markdown(results: List[EvalResult]) -> str:
    """渲染为 markdown 对比表格。"""
    lines: List[str] = []
    lines.append("# 参数矩阵评估")
    lines.append("")
    lines.append("| dataset | embedding | retriever | top_k | avg_hit | avg_mrr | avg_recall | avg_kw_hit | n_cases | n_errors | notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        m = r.meta
        a = r.aggregate
        notes = m.notes or ""
        if "error_skipped" in a or "error_failed" in a:
            lines.append(
                f"| {m.dataset} | {m.embedding} | {m.retriever} | {m.top_k} | - | - | - | - | 0 | - | {notes} |"
            )
            continue
        lines.append(
            f"| {m.dataset} | {m.embedding} | {m.retriever} | {m.top_k} | "
            f"{a.get('avg_hit_at_k', 0):.3f} | "
            f"{a.get('avg_mrr', 0):.4f} | "
            f"{a.get('avg_recall_at_k', 0):.3f} | "
            f"{a.get('avg_keyword_hit_rate', 0):.3f} | "
            f"{int(a.get('n_cases', 0))} | "
            f"{int(a.get('n_errors', 0))} | "
            f"{notes} |"
        )
    lines.append("")
    lines.append(
        "> 主要看 avg_mrr——它对【命中切片排在第几位】最敏感；hit_at_k 是粗筛信号。"
    )
    return "\n".join(lines)
