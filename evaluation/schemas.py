"""评估框架的数据契约（dataclass）。

三层结构：
    GoldenCase     — 黄金集里的一条 case（YAML 反序列化目标）
    RetrievedDoc   — 检索回来的单个切片（同时承载给生成层用的完整内容）
    CaseResult     — 一条 case 的检索结果 + 指标
    EvalResult     — 一次跑的总结果（多条 CaseResult 聚合）

设计原则：
- 所有字段 JSON 友好（基础类型 / list / dict），便于 `dataclasses.asdict()` 直接落盘
- GoldenCase 预留 expected_answer / expected_answer_keywords / forbidden_phrases 字段，
  本次只读不算；未来加生成层评估时不需要改 schema
- CaseResult.retrieved 必须包含 page_content（即使只取前 N 字），让生成层能直接复用，
  不需要再跑一遍检索
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_TOP_K = 10
_PREVIEW_CHARS = 200


@dataclass
class GoldenCase:
    """黄金集里的一条 case。"""

    id: str
    query: str

    expected_pages: List[int] = field(default_factory=list)
    expected_keywords_any: List[str] = field(default_factory=list)
    expected_source: Optional[str] = None

    top_k_override: Optional[int] = None
    expected_route: Optional[str] = None

    # ── 生成层预留字段（本次不实现，YAML 里允许为空） ──
    expected_answer: Optional[str] = None
    expected_answer_keywords: List[str] = field(default_factory=list)
    forbidden_phrases: List[str] = field(default_factory=list)

    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GoldenCase":
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class GoldenDataset:
    """一份黄金集。"""

    dataset: str
    cases: List[GoldenCase]
    source_filter: Optional[str] = None
    embedding_hint: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GoldenDataset":
        """从 YAML 文件加载。"""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_cases = data.pop("cases", [])
        cases = [GoldenCase.from_dict(c) for c in raw_cases]
        return cls(
            dataset=data.get("dataset", Path(path).stem),
            cases=cases,
            source_filter=data.get("source_filter"),
            embedding_hint=data.get("embedding_hint"),
            notes=data.get("notes"),
        )


@dataclass
class RetrievedDoc:
    """单条被检索回来的切片。"""

    rank: int
    score: float
    page: Optional[int]
    source: Optional[str]
    content_preview: str
    content_len: int

    @classmethod
    def from_langchain(
        cls,
        rank: int,
        score: float,
        doc: Any,
        preview_chars: int = _PREVIEW_CHARS,
    ) -> "RetrievedDoc":
        meta = getattr(doc, "metadata", None) or {}
        content = getattr(doc, "page_content", "") or ""
        page_raw = meta.get("page")
        try:
            page = int(page_raw) if page_raw is not None else None
        except (TypeError, ValueError):
            page = None
        return cls(
            rank=rank,
            score=float(score),
            page=page,
            source=meta.get("source"),
            content_preview=content[:preview_chars],
            content_len=len(content),
        )


@dataclass
class CaseResult:
    """一条 case 的检索结果 + 指标。"""

    case_id: str
    query: str
    top_k: int
    retrieved: List[RetrievedDoc]
    metrics: Dict[str, float]
    matched_ranks: List[int] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EvalMeta:
    """一次评估的元信息。

    retriever: 检索方式标识，"dense"（仅 FAISS 密集向量）或 "hybrid"
    （FAISS + BM25 通过 EnsembleRetriever 融合）。默认 dense 以保持与
    旧 baseline 文件兼容（旧 baseline 没写这个字段，反序列化时为缺省值）。
    """

    dataset: str
    embedding: str
    top_k: int
    persist_path: str
    retriever: str = "dense"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    notes: Optional[str] = None


@dataclass
class EvalResult:
    """一次评估的总结果。"""

    meta: EvalMeta
    cases: List[CaseResult]
    aggregate: Dict[str, float] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = EvalMeta(**data["meta"])
        cases = [
            CaseResult(
                case_id=c["case_id"],
                query=c["query"],
                top_k=c["top_k"],
                retrieved=[RetrievedDoc(**rd) for rd in c.get("retrieved", [])],
                metrics=c.get("metrics", {}),
                matched_ranks=c.get("matched_ranks", []),
                error=c.get("error"),
            )
            for c in data.get("cases", [])
        ]
        return cls(meta=meta, cases=cases, aggregate=data.get("aggregate", {}))


DEFAULT_TOP_K = _DEFAULT_TOP_K
