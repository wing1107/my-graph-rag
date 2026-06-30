"""端到端评估 runner：跑完整 LangGraph 图，输出答案质量 + 效率指标。

输出指标（每条 case）
--------------------
total_latency_ms      端到端耗时（毫秒，含所有 LLM 调用）
llm_call_count        LLM invoke() 被调用次数（通过计数代理实现）
retrieval_triggered   是否触发了 retrieve 节点（bool → 1.0/0.0）
rewrite_triggered     是否触发了 rewrite_query 节点（bool → 1.0/0.0）
hallucination_flag    最终是否检测到幻觉（来自终态 state）
retry_count           累计重试次数（来自终态 state）
keyword_hit_rate      答案关键词命中率
forbidden_count       禁止短语命中数
no_answer_score       无效回答检测分（0=有效, 1=无效）

使用示例
--------
    python -m evaluation.cli e2e \\
        --dataset pumpkin_book --embedding m3e \\
        --retriever hybrid --model glm-4-flash

节点追踪原理
-----------
使用 ``graph.stream(stream_mode="updates")`` 迭代，每次 yield 一个
``{node_name: partial_state_update}`` dict，收集 node_name 即可得到
访问顺序；同时将 partial_update 合并到 full_state，最终获得完整终态。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.metrics.answer_quality import compute_all as _answer_quality
from evaluation.schemas import GoldenCase, GoldenDataset


# ═══════════════════════════════════════════════════════════════
# 数据契约
# ═══════════════════════════════════════════════════════════════

@dataclass
class E2ECaseResult:
    """一条 case 的端到端评估结果。"""

    case_id: str
    query: str
    final_answer: Optional[str]
    nodes_visited: List[str]           # 按执行顺序排列的节点名列表
    expected_route: Optional[str]
    predicted_route: Optional[str]
    metrics: Dict[str, float]          # 所有数值指标
    error: Optional[str] = None


@dataclass
class E2EMeta:
    """一次 E2E 评估的元信息。"""

    dataset: str
    embedding: str
    retriever: str
    model: str
    top_k: int
    persist_path: str
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    notes: Optional[str] = None


@dataclass
class E2EResult:
    """一次 E2E 评估的总结果（多条 case 聚合）。"""

    meta: E2EMeta
    cases: List[E2ECaseResult]
    aggregate: Dict[str, float] = field(default_factory=dict)

    def to_json(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# LLM 调用计数代理
# ═══════════════════════════════════════════════════════════════

class _CountingLLMProxy:
    """透明代理 LangChain LLM，同时记录 invoke() 调用次数。

    之所以不用继承：LangChain LLM 有复杂的元类和校验逻辑，
    简单组合（getattr 转发）更可靠，且只需拦截 invoke / __call__。
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self.call_count: int = 0

    def reset(self) -> None:
        self.call_count = 0

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self._llm.invoke(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self._llm(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


# ═══════════════════════════════════════════════════════════════
# 单 case 运行
# ═══════════════════════════════════════════════════════════════

def _run_case(
    graph: Any,
    llm_proxy: _CountingLLMProxy,
    vectordb: Any,
    case: GoldenCase,
    retriever_kind: str,
    top_k: int,
) -> E2ECaseResult:
    """运行一条 case，收集端到端指标。"""
    llm_proxy.reset()
    nodes_visited: List[str] = []

    initial_state = {
        "question": case.query,
        "rewritten_question": None,
        "documents": [],
        "generation": None,
        "chat_history": [],
        "source_filter": None,
        "retry_count": 0,
        "embedding": "",      # vectordb 已由 config 注入，此字段不参与检索
        "top_k": top_k,
        "hallucination_flag": False,
        "route_decision": None,
    }
    config = {
        "configurable": {
            # 每条 case 独立 thread，避免 checkpointer 跨 case 混入历史
            "thread_id": f"e2e_eval_{case.id}_{int(time.time() * 1000)}",
            "vectordb": vectordb,
            "llm": llm_proxy,
            "retriever_kind": retriever_kind,
            "history_len": 0,
        }
    }

    # 合并 stream updates 得到完整终态
    full_state: Dict[str, Any] = dict(initial_state)
    error_msg: Optional[str] = None

    t0 = time.perf_counter()
    try:
        for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, updates in chunk.items():
                nodes_visited.append(node_name)
                if isinstance(updates, dict):
                    full_state.update(updates)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    final_answer: Optional[str] = full_state.get("generation")

    # ── 效率指标 ────────────────────────────────────────────────
    retrieval_triggered = 1.0 if "retrieve" in nodes_visited else 0.0
    rewrite_triggered = 1.0 if "rewrite_query" in nodes_visited else 0.0
    hallucination = float(bool(full_state.get("hallucination_flag", False)))
    retry_count = float(full_state.get("retry_count", 0))

    expected_route = (getattr(case, "expected_route", None) or "").strip().lower() or None
    state_route = (full_state.get("route_decision") or "").strip().lower() or None
    predicted_route = state_route
    if predicted_route not in ("retrieve", "direct"):
        predicted_route = "retrieve" if retrieval_triggered > 0 else "direct"

    # ── 答案质量指标 ─────────────────────────────────────────────
    quality = _answer_quality(final_answer, case)

    metrics: Dict[str, float] = {
        "total_latency_ms": round(elapsed_ms, 2),
        "llm_call_count": float(llm_proxy.call_count),
        "retrieval_triggered": retrieval_triggered,
        "rewrite_triggered": rewrite_triggered,
        "hallucination_flag": hallucination,
        "retry_count": retry_count,
        **quality,
    }

    if expected_route in ("retrieve", "direct"):
        metrics["routing_match"] = 1.0 if predicted_route == expected_route else 0.0
        metrics["routing_expected_retrieve"] = 1.0 if expected_route == "retrieve" else 0.0
        metrics["routing_predicted_retrieve"] = 1.0 if predicted_route == "retrieve" else 0.0

    return E2ECaseResult(
        case_id=case.id,
        query=case.query,
        final_answer=final_answer,
        nodes_visited=nodes_visited,
        expected_route=expected_route,
        predicted_route=predicted_route,
        metrics=metrics,
        error=error_msg,
    )


# ═══════════════════════════════════════════════════════════════
# 聚合
# ═══════════════════════════════════════════════════════════════

_AVG_METRICS = [
    "total_latency_ms",
    "llm_call_count",
    "retrieval_triggered",
    "rewrite_triggered",
    "hallucination_flag",
    "retry_count",
    "keyword_hit_rate",
    "no_answer_score",
    "routing_match",
    "routing_expected_retrieve",
    "routing_predicted_retrieve",
]

_SUM_METRICS = ["forbidden_count"]


def _aggregate(cases: List[E2ECaseResult]) -> Dict[str, float]:
    valid = [c for c in cases if not c.error and c.metrics]
    n = len(valid)
    if n == 0:
        return {"n_cases": 0.0, "n_errors": float(len(cases))}

    agg: Dict[str, float] = {"n_cases": float(n), "n_errors": float(len(cases) - n)}
    for key in _AVG_METRICS:
        vals = [c.metrics[key] for c in valid if key in c.metrics]
        agg[f"avg_{key}"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    for key in _SUM_METRICS:
        vals = [c.metrics[key] for c in valid if key in c.metrics]
        agg[f"total_{key}"] = sum(vals)
    return agg


# ═══════════════════════════════════════════════════════════════
# 公开入口
# ═══════════════════════════════════════════════════════════════

def run_e2e(
    dataset_path: "str | Path",
    embedding: str,
    model: str,
    top_k: int = 4,
    retriever: str = "dense",
    persist_path: Optional[str] = None,
    notes: Optional[str] = None,
) -> E2EResult:
    """跑一个黄金集的端到端评估，返回 E2EResult（不写文件）。

    Parameters
    ----------
    dataset_path : str | Path
        黄金集 YAML 文件路径（或 stem，如 "pumpkin_book"）。
    embedding : str
        Embedding 名称（"m3e" / "multilingual"），用于加载向量库。
    model : str
        LLM 模型名称（如 "glm-4-flash"），从 .env 自动读取 API Key。
    top_k : int
        检索返回的最大文档数，默认 4。
    retriever : str
        检索方式："dense" 或 "hybrid"。
    persist_path : str, optional
        向量库路径，不传则按 embedding 自动解析。
    notes : str, optional
        附加说明，写入报告 meta。
    """
    from evaluation.utils import load_vectordb
    from qa_chain.graph import build_rag_graph
    from qa_chain.model_to_llm import model_to_llm

    if retriever not in ("dense", "hybrid"):
        raise ValueError(f"retriever 必须是 'dense' 或 'hybrid'，收到: {retriever!r}")

    dataset_path = Path(dataset_path)
    dataset = GoldenDataset.from_yaml(dataset_path)

    # 解析向量库路径
    if persist_path:
        db_path = persist_path
    else:
        db_path = str(_PROJECT_ROOT / "data_base" / "vector_db" / f"faiss_{embedding}")

    print(f"[e2e] 加载向量库: {db_path}")
    vectordb = load_vectordb(embedding_provider=embedding, db_path=db_path)

    print(f"[e2e] 加载 LLM: {model}")
    base_llm = model_to_llm(model=model, temperature=0.0)
    llm_proxy = _CountingLLMProxy(base_llm)

    print(f"[e2e] 构建图（retriever={retriever}）")
    graph = build_rag_graph(checkpointer=None)

    case_results: List[E2ECaseResult] = []
    for i, case in enumerate(dataset.cases, 1):
        print(f"[e2e] ({i}/{len(dataset.cases)}) {case.id}: {case.query[:50]}...")
        result = _run_case(
            graph=graph,
            llm_proxy=llm_proxy,
            vectordb=vectordb,
            case=case,
            retriever_kind=retriever,
            top_k=case.top_k_override or top_k,
        )
        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            lat = result.metrics.get("total_latency_ms", 0)
            calls = result.metrics.get("llm_call_count", 0)
            khr = result.metrics.get("keyword_hit_rate", 0)
            nas = result.metrics.get("no_answer_score", 0)
            nodes = " → ".join(result.nodes_visited)
            print(f"  latency={lat:.0f}ms  llm_calls={calls:.0f}"
                  f"  kw_hit={khr:.2f}  no_ans={nas:.1f}")
            if result.expected_route:
                print(
                    f"  route: expected={result.expected_route} "
                    f"predicted={result.predicted_route} "
                    f"match={result.metrics.get('routing_match', 0.0):.0f}"
                )
            print(f"  nodes: {nodes}")
        case_results.append(result)

    agg = _aggregate(case_results)
    meta = E2EMeta(
        dataset=dataset.dataset,
        embedding=embedding,
        retriever=retriever,
        model=model,
        top_k=top_k,
        persist_path=db_path,
        notes=notes,
    )
    return E2EResult(meta=meta, cases=case_results, aggregate=agg)


# ═══════════════════════════════════════════════════════════════
# Markdown 报告渲染
# ═══════════════════════════════════════════════════════════════

def render_markdown(result: E2EResult) -> str:
    """将 E2EResult 渲染为 Markdown 字符串。"""
    lines: List[str] = []
    m = result.meta
    lines.append(f"# E2E 评估报告: {m.dataset} @ {m.embedding} / {m.retriever}")
    lines.append("")
    lines.append(f"- **模型**: {m.model}")
    lines.append(f"- **top_k**: {m.top_k}")
    lines.append(f"- **时间**: {m.timestamp}")
    if m.notes:
        lines.append(f"- **备注**: {m.notes}")
    lines.append("")

    # 总体指标表
    lines.append("## 总体指标")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|:---:|")
    for k, v in result.aggregate.items():
        v_str = f"{v:.4f}" if isinstance(v, float) and k not in ("n_cases", "n_errors", "total_forbidden_count") else str(v)
        lines.append(f"| {k} | {v_str} |")
    lines.append("")

    # 各 case 详情
    lines.append("## 各 Case 详情")
    lines.append("")
    for c in result.cases:
        lines.append(f"### `{c.case_id}`")
        lines.append("")
        lines.append(f"- **query**: {c.query}")
        if c.expected_route:
            lines.append(
                f"- **route**: expected={c.expected_route}, predicted={c.predicted_route}"
            )
        lines.append(f"- **nodes**: {' → '.join(c.nodes_visited)}")
        if c.error:
            lines.append(f"- **ERROR**: {c.error}")
            lines.append("")
            continue

        lines.append("")
        lines.append("**指标**")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|:---:|")
        for k, v in c.metrics.items():
            lines.append(f"| {k} | {v:.2f} |")
        lines.append("")

        if c.final_answer:
            preview = c.final_answer[:300].replace("\n", " ")
            if len(c.final_answer) > 300:
                preview += "..."
            lines.append(f"**答案预览**: {preview}")
            lines.append("")

    return "\n".join(lines)
