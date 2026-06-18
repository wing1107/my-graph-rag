"""评估框架的命令行入口。

使用方式：
    python -m evaluation.cli run --dataset pumpkin_book --embedding m3e --retriever hybrid
    python -m evaluation.cli run --dataset lesson25 --embedding multilingual
    python -m evaluation.cli diff --dataset pumpkin_book --embedding m3e --retriever hybrid
    python -m evaluation.cli promote --dataset pumpkin_book --embedding m3e --retriever hybrid
    python -m evaluation.cli matrix --datasets pumpkin_book --embeddings m3e --top-ks 5,10 --retrievers dense,hybrid

惯例：
- --dataset 接 stem（如 pumpkin_book）会自动解析为 evaluation/datasets/pumpkin_book.yaml
- --dataset 接相对/绝对路径也支持
- 报告写到 evaluation/reports/<timestamp>__<dataset>__<embedding>/
- baseline 文件名 evaluation/baselines/baseline_{dataset}_{embedding}.json（dense）
                              或 baseline_{dataset}_{embedding}_{retriever}.json（hybrid）

注意：两个向量库各自对应一个知识库——不要跨 embedding 评估：
  faiss_m3e          → 南瓜书 (pumpkin_book)
  faiss_multilingual → 大家的日语 初级1 (lesson25)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.schemas import DEFAULT_TOP_K, EvalResult  # noqa: E402

_DATASETS_DIR = _PROJECT_ROOT / "evaluation" / "datasets"
_REPORTS_DIR = _PROJECT_ROOT / "evaluation" / "reports"
_BASELINES_DIR = _PROJECT_ROOT / "evaluation" / "baselines"


def _ensure_utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _resolve_dataset_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p.resolve()
    candidate = _DATASETS_DIR / f"{name_or_path}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"未找到 dataset: {name_or_path}（既不是文件路径，也不是 "
        f"{_DATASETS_DIR} 下的 stem）"
    )


def _baseline_path(dataset: str, embedding: str, retriever: str = "dense") -> Path:
    """baseline 文件名编码 retriever 维度，让 dense / hybrid 可以共存对比。

    向后兼容：dense 仍用旧文件名（无 retriever 后缀）。
    """
    if retriever == "dense":
        return _BASELINES_DIR / f"baseline_{dataset}_{embedding}.json"
    return _BASELINES_DIR / f"baseline_{dataset}_{embedding}_{retriever}.json"


def _report_dir(dataset: str, embedding: str, retriever: str = "dense") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "" if retriever == "dense" else f"__{retriever}"
    return _REPORTS_DIR / f"{ts}__{dataset}__{embedding}{suffix}"


def _write_report(result: EvalResult, out_dir: Path) -> None:
    from evaluation.runners.retrieval_runner import render_markdown

    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_json(out_dir / "results.json")
    md = render_markdown(result)
    (out_dir / "results.md").write_text(md, encoding="utf-8")


def _write_diagnose_report(
    result: EvalResult, dataset_path: Path, out_dir: Path, mrr_threshold: float = 0.5
) -> None:
    """对每个失败 case（mrr < threshold 或未命中），dump expected_pages 周围的切片。"""
    from evaluation.diagnostics import diagnose_case
    from evaluation.schemas import GoldenDataset

    dataset = GoldenDataset.from_yaml(dataset_path)
    cases_by_id = {c.id: c for c in dataset.cases}

    lines: List[str] = []
    lines.append(f"# 诊断报告: {result.meta.dataset} @ {result.meta.embedding}")
    lines.append("")
    lines.append(f"- mrr_threshold: {mrr_threshold}")
    lines.append("- 仅展开 mrr < threshold 或 hit_at_k == 0 的 case")
    lines.append("")
    n_diagnosed = 0
    for case in result.cases:
        h = case.metrics.get("hit_at_k", 0.0)
        m = case.metrics.get("mrr", 0.0)
        if h >= 1.0 and m >= mrr_threshold:
            continue
        gc = cases_by_id.get(case.case_id)
        if gc is None:
            continue
        n_diagnosed += 1
        lines.append(f"## case `{case.case_id}` (hit={h:.2f}, mrr={m:.4f})")
        lines.append("")
        lines.append(f"- query: `{case.query}`")
        lines.append(f"- expected_pages: {gc.expected_pages}")
        lines.append(f"- expected_keywords_any: {gc.expected_keywords_any}")
        lines.append("")
        diag = diagnose_case(
            persist_path=result.meta.persist_path,
            expected_pages=gc.expected_pages,
            source_filter=dataset.source_filter,
            neighborhood=1,
        )
        lines.append(diag)
        lines.append("")
    if n_diagnosed == 0:
        lines.append("(所有 case 都达标，无需诊断)")
    (out_dir / "diagnose.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[evaluation] 诊断报告: {out_dir / 'diagnose.md'} (展开 {n_diagnosed} 个 case)")


def _cmd_run(args: argparse.Namespace) -> int:
    from evaluation.runners.retrieval_runner import run_dataset

    dataset_path = _resolve_dataset_path(args.dataset)
    print(f"[evaluation] dataset = {dataset_path}")
    print(
        f"[evaluation] embedding = {args.embedding}, top_k = {args.top_k}, "
        f"retriever = {args.retriever}"
    )

    result = run_dataset(
        dataset_path=dataset_path,
        embedding=args.embedding,
        top_k=args.top_k,
        notes=args.notes,
        retriever=args.retriever,
    )

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = _report_dir(result.meta.dataset, args.embedding, args.retriever)
    _write_report(result, out_dir)

    if args.diagnose:
        _write_diagnose_report(result, dataset_path, out_dir)

    print()
    print(f"[evaluation] 写入报告: {out_dir}")
    print()
    print("=== 总体指标 ===")
    for k, v in result.aggregate.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v:.4f}")
        else:
            print(f"  {k:30s} {v}")
    print()
    print("=== 各 case ===")
    for c in result.cases:
        if c.error:
            print(f"  {c.case_id:30s} ERROR: {c.error}")
            continue
        h = c.metrics.get("hit_at_k", 0.0)
        m = c.metrics.get("mrr", 0.0)
        r = c.metrics.get("recall_at_k", 0.0)
        first_rank = c.matched_ranks[0] if c.matched_ranks else None
        rank_str = f"rank#1={first_rank}" if first_rank else "no hit"
        print(
            f"  {c.case_id:30s} hit={h:.2f} mrr={m:.4f} recall={r:.2f} "
            f"({rank_str}, k={c.top_k})"
        )

    if args.diff_baseline:
        baseline_p = _baseline_path(result.meta.dataset, args.embedding, args.retriever)
        if not baseline_p.exists():
            print(f"\n[evaluation] baseline 不存在: {baseline_p}")
            print("  跑 `python -m evaluation.cli promote ...` 把当前结果固化为基线")
        else:
            from evaluation.runners.baseline_diff import diff_results, render_diff_md

            baseline = EvalResult.from_json(baseline_p)
            diff = diff_results(result, baseline, tolerance=args.tolerance)
            md = render_diff_md(diff)
            (out_dir / "diff_vs_baseline.md").write_text(md, encoding="utf-8")
            print(f"\n[evaluation] 已写入 diff: {out_dir / 'diff_vs_baseline.md'}")
            if diff.regressions:
                print(f"[evaluation] 发现 {len(diff.regressions)} 处回归")
                return 1
            print("[evaluation] 无回归")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    from evaluation.runners.baseline_diff import diff_results, render_diff_md
    from evaluation.runners.retrieval_runner import run_dataset

    baseline_p = _baseline_path(args.dataset, args.embedding, args.retriever)
    if not baseline_p.exists():
        print(f"[evaluation] baseline 不存在: {baseline_p}", file=sys.stderr)
        return 2
    baseline = EvalResult.from_json(baseline_p)

    dataset_path = _resolve_dataset_path(args.dataset)
    current = run_dataset(
        dataset_path=dataset_path,
        embedding=args.embedding,
        top_k=args.top_k,
        retriever=args.retriever,
    )
    out_dir = _report_dir(current.meta.dataset, args.embedding, args.retriever)
    _write_report(current, out_dir)

    diff = diff_results(current, baseline, tolerance=args.tolerance)
    md = render_diff_md(diff)
    (out_dir / "diff_vs_baseline.md").write_text(md, encoding="utf-8")
    print(md)
    return 1 if diff.regressions else 0


def _cmd_promote(args: argparse.Namespace) -> int:
    from evaluation.runners.retrieval_runner import run_dataset

    dataset_path = _resolve_dataset_path(args.dataset)
    result = run_dataset(
        dataset_path=dataset_path,
        embedding=args.embedding,
        top_k=args.top_k,
        retriever=args.retriever,
    )
    out_path = _baseline_path(result.meta.dataset, args.embedding, args.retriever)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(
            f"[evaluation] baseline 已存在: {out_path}\n"
            "  - 用 `--force` 覆盖\n"
            "  - 或先备份当前 baseline 文件"
        )
        return 2

    result.to_json(out_path)
    print(f"[evaluation] 固化基线: {out_path}")
    print(f"  数据集: {result.meta.dataset}")
    print(f"  embedding: {args.embedding}")
    print(f"  retriever: {args.retriever}")
    print(f"  top_k: {args.top_k}")
    print(f"  case 数: {len(result.cases)}")
    return 0


def _cmd_e2e(args: argparse.Namespace) -> int:
    from evaluation.runners.e2e_runner import E2EResult, render_markdown, run_e2e

    dataset_path = _resolve_dataset_path(args.dataset)
    print(f"[e2e] dataset = {dataset_path}")
    print(
        f"[e2e] embedding={args.embedding}  retriever={args.retriever}"
        f"  model={args.model}  top_k={args.top_k}"
    )

    result = run_e2e(
        dataset_path=dataset_path,
        embedding=args.embedding,
        model=args.model,
        top_k=args.top_k,
        retriever=args.retriever,
        notes=args.notes,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "" if args.retriever == "dense" else f"__{args.retriever}"
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_dir = _REPORTS_DIR / f"{ts}__{result.meta.dataset}__{args.embedding}{suffix}__e2e"
    out_dir.mkdir(parents=True, exist_ok=True)

    result.to_json(out_dir / "e2e_results.json")
    md = render_markdown(result)
    (out_dir / "e2e_results.md").write_text(md, encoding="utf-8")

    print()
    print(f"[e2e] 写入报告: {out_dir}")
    print()
    print("=== 总体指标 ===")
    for k, v in result.aggregate.items():
        if isinstance(v, float):
            print(f"  {k:35s} {v:.4f}")
        else:
            print(f"  {k:35s} {v}")
    print()
    print("=== 各 case ===")
    for c in result.cases:
        if c.error:
            print(f"  {c.case_id:35s} ERROR: {c.error}")
            continue
        lat = c.metrics.get("total_latency_ms", 0)
        calls = c.metrics.get("llm_call_count", 0)
        khr = c.metrics.get("keyword_hit_rate", 0)
        nas = c.metrics.get("no_answer_score", 0)
        nodes = " → ".join(c.nodes_visited)
        print(
            f"  {c.case_id:35s} latency={lat:.0f}ms"
            f"  calls={calls:.0f}  kw_hit={khr:.2f}  no_ans={nas:.1f}"
        )
        print(f"  {'':35s} nodes: {nodes}")
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    from evaluation.runners.matrix_runner import (
        render_matrix_markdown,
        run_matrix,
    )

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    embeddings = [e.strip() for e in args.embeddings.split(",") if e.strip()]
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    print(f"[evaluation] matrix:")
    print(f"  datasets   = {datasets}")
    print(f"  embeddings = {embeddings}")
    print(f"  top_ks     = {top_ks}")
    print(f"  retrievers = {retrievers}")

    dataset_paths = [str(_resolve_dataset_path(d)) for d in datasets]
    results = run_matrix(dataset_paths, embeddings, top_ks, retrievers=retrievers)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = _REPORTS_DIR / f"matrix_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_matrix_markdown(results)
    (out_dir / "matrix.md").write_text(md, encoding="utf-8")

    for r in results:
        sub = out_dir / f"{r.meta.dataset}__{r.meta.embedding}__k{r.meta.top_k}__{r.meta.retriever}"
        sub.mkdir(parents=True, exist_ok=True)
        r.to_json(sub / "results.json")

    print()
    print(md)
    print()
    print(f"[evaluation] 矩阵报告: {out_dir / 'matrix.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation.cli",
        description="RAG 检索层自动化评估框架（不调 LLM）",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def _add_retriever_arg(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--retriever",
            choices=("dense", "hybrid"),
            default="dense",
            help="检索方式：dense（默认，仅 FAISS）或 hybrid（FAISS + BM25/jieba）",
        )

    p_run = sub.add_parser("run", help="跑一个黄金集，输出 JSON + Markdown 报告")
    p_run.add_argument("--dataset", required=True, help="dataset stem（如 pumpkin_book）或 YAML 路径")
    p_run.add_argument("--embedding", required=True, help="embedding 名称（如 m3e / multilingual）")
    p_run.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    p_run.add_argument("--notes", default=None)
    p_run.add_argument("--out-dir", default=None, dest="out_dir", help="自定义报告目录")
    p_run.add_argument(
        "--diff-baseline", action="store_true", dest="diff_baseline",
        help="跑完顺便和 baseline 做 diff（baseline 不存在时仅提示）",
    )
    p_run.add_argument("--tolerance", type=float, default=0.0)
    p_run.add_argument(
        "--diagnose", action="store_true",
        help="对失败 case（mrr<0.5 或未命中）dump expected_pages 周围切片到 diagnose.md",
    )
    _add_retriever_arg(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_diff = sub.add_parser("diff", help="跑一次并与 baseline 比对")
    p_diff.add_argument("--dataset", required=True)
    p_diff.add_argument("--embedding", required=True)
    p_diff.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    p_diff.add_argument("--tolerance", type=float, default=0.0)
    _add_retriever_arg(p_diff)
    p_diff.set_defaults(func=_cmd_diff)

    p_prom = sub.add_parser("promote", help="把当前结果固化为新 baseline")
    p_prom.add_argument("--dataset", required=True)
    p_prom.add_argument("--embedding", required=True)
    p_prom.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    p_prom.add_argument("--force", action="store_true")
    _add_retriever_arg(p_prom)
    p_prom.set_defaults(func=_cmd_promote)

    p_e2e = sub.add_parser("e2e", help="端到端评估：跑完整图，输出答案质量 + 效率指标")
    p_e2e.add_argument("--dataset", required=True, help="dataset stem 或 YAML 路径")
    p_e2e.add_argument("--embedding", required=True, help="embedding 名称（如 m3e / multilingual）")
    p_e2e.add_argument(
        "--model", required=True,
        help="LLM 模型名称，如 glm-4-flash / glm-4-air / qwen3-max",
    )
    p_e2e.add_argument("--top-k", type=int, default=4, dest="top_k")
    p_e2e.add_argument("--notes", default=None)
    p_e2e.add_argument("--out-dir", default=None, dest="out_dir", help="自定义报告目录")
    _add_retriever_arg(p_e2e)
    p_e2e.set_defaults(func=_cmd_e2e)

    p_mat = sub.add_parser("matrix", help="参数矩阵：dense vs hybrid / top_k 笛卡尔积对比")
    p_mat.add_argument("--datasets", required=True, help="逗号分隔，如 pumpkin_book")
    p_mat.add_argument("--embeddings", required=True, help="逗号分隔，如 m3e")
    p_mat.add_argument("--top-ks", default="5,10", dest="top_ks", help="逗号分隔，如 5,10")
    p_mat.add_argument(
        "--retrievers",
        default="dense",
        help="逗号分隔，如 'dense,hybrid'；默认 dense",
    )
    p_mat.set_defaults(func=_cmd_matrix)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
