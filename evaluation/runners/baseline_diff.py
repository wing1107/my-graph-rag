"""当次评估结果 vs baseline 的差分报告。

设计要点：
- 关键指标列表 KEY_METRICS 显式声明（不是所有指标都参与回归判定），
  避免 retrieval_latency_ms 等"性能而非质量"指标造成 CI 噪音
- tolerance 允许小幅波动（默认 0.0 = 严格），生产建议 0.01~0.05
- DiffReport 拆分 regressions / improvements / new_cases / removed_cases，
  pytest 断言只看 regressions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from evaluation.schemas import CaseResult, EvalResult


KEY_METRICS = ("hit_at_k", "recall_at_k", "mrr", "page_overlap", "keyword_hit_rate")


@dataclass
class MetricDiff:
    case_id: str
    metric: str
    baseline: float
    current: float
    delta: float

    @property
    def is_regression(self) -> bool:
        return self.delta < 0


@dataclass
class DiffReport:
    dataset: str
    embedding: str
    tolerance: float
    regressions: List[MetricDiff] = field(default_factory=list)
    improvements: List[MetricDiff] = field(default_factory=list)
    new_cases: List[str] = field(default_factory=list)
    removed_cases: List[str] = field(default_factory=list)
    aggregate_baseline: Dict[str, float] = field(default_factory=dict)
    aggregate_current: Dict[str, float] = field(default_factory=dict)

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions)


def _index_by_id(cases: List[CaseResult]) -> Dict[str, CaseResult]:
    return {c.case_id: c for c in cases}


def diff_results(
    current: EvalResult,
    baseline: EvalResult,
    tolerance: float = 0.0,
    metrics: Optional[tuple] = None,
) -> DiffReport:
    """逐 case 比较关键指标。"""
    metrics = metrics or KEY_METRICS
    report = DiffReport(
        dataset=current.meta.dataset,
        embedding=current.meta.embedding,
        tolerance=tolerance,
        aggregate_baseline=dict(baseline.aggregate),
        aggregate_current=dict(current.aggregate),
    )

    cur_by_id = _index_by_id(current.cases)
    base_by_id = _index_by_id(baseline.cases)

    report.new_cases = sorted(set(cur_by_id) - set(base_by_id))
    report.removed_cases = sorted(set(base_by_id) - set(cur_by_id))

    common_ids = sorted(set(cur_by_id) & set(base_by_id))
    for cid in common_ids:
        cur_case = cur_by_id[cid]
        base_case = base_by_id[cid]
        for m in metrics:
            cur_v = float(cur_case.metrics.get(m, 0.0))
            base_v = float(base_case.metrics.get(m, 0.0))
            delta = cur_v - base_v
            md = MetricDiff(case_id=cid, metric=m, baseline=base_v, current=cur_v, delta=delta)
            if delta < -abs(tolerance):
                report.regressions.append(md)
            elif delta > abs(tolerance):
                report.improvements.append(md)
    return report


def render_diff_md(report: DiffReport) -> str:
    lines: List[str] = []
    lines.append(f"# Baseline diff: {report.dataset} @ {report.embedding}")
    lines.append("")
    lines.append(f"- tolerance: {report.tolerance}")
    lines.append(f"- regressions: **{len(report.regressions)}**")
    lines.append(f"- improvements: {len(report.improvements)}")
    if report.new_cases:
        lines.append(f"- new_cases: {report.new_cases}")
    if report.removed_cases:
        lines.append(f"- removed_cases: {report.removed_cases}")
    lines.append("")

    def _table(items: List[MetricDiff], title: str) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| case | metric | baseline | current | delta |")
        lines.append("|---|---|---|---|---|")
        for d in items:
            lines.append(
                f"| {d.case_id} | {d.metric} | {d.baseline:.4f} | "
                f"{d.current:.4f} | {d.delta:+.4f} |"
            )
        lines.append("")

    _table(report.regressions, "Regressions")
    _table(report.improvements, "Improvements")

    lines.append("## 总体指标对比")
    lines.append("")
    keys = sorted(set(report.aggregate_baseline) | set(report.aggregate_current))
    lines.append("| metric | baseline | current | delta |")
    lines.append("|---|---|---|---|")
    for k in keys:
        b = report.aggregate_baseline.get(k, 0.0)
        c = report.aggregate_current.get(k, 0.0)
        if isinstance(b, float) and isinstance(c, float):
            lines.append(f"| {k} | {b:.4f} | {c:.4f} | {c - b:+.4f} |")
        else:
            lines.append(f"| {k} | {b} | {c} | - |")
    return "\n".join(lines)
