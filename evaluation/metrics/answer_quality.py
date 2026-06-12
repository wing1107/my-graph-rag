"""答案质量指标（轻量版，零外部依赖）。

所有函数均为纯函数，不调用 LLM / 网络，可在任何环境下运行。

函数列表
--------
keyword_hit_rate(answer, keywords) → float
    expected_answer_keywords 命中率（0~1），
    反映答案是否包含期望的关键概念。

forbidden_count(answer, phrases) → int
    forbidden_phrases 命中数（越高越差），
    检测答案中不应出现的表述（如错误说法、过时术语）。

no_answer_score(answer) → float
    无效回答检测分（0=有效回答, 1=完全无效），
    匹配"根据上下文无法回答"等固定模式。

compute_all(answer, golden_case) → Dict[str, float]
    一次计算所有指标，直接写入 E2ECaseResult.metrics。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


# ── 无效回答模式词表 ─────────────────────────────────────────────────────────
# 当前系统（TEMPLATE_V4_COT）要求 LLM 只用上下文回答，
# 检索无关时会生成如下模式；这里做精准匹配（不用正则，避免误判）。
_NO_ANSWER_PATTERNS: List[str] = [
    "根据提供的上下文，无法找到",
    "根据提供的上下文，没有",
    "上下文中没有",
    "上下文中未提到",
    "上下文中未包含",
    "无法从上下文",
    "无法直接从上下文",          # 补充：LLM 常在"无法"后加"直接"
    "没有在提供的上下文中",       # 补充：chapter16_content 触发的模式
    "上下文中没有提供",           # 补充：lesson_example_overview 触发的模式
    "上下文没有涉及",
    "无法回答",
    "我不知道",
    "没有相关信息",
    "无法提供",
    "无相关内容",
    "找不到相关",
    "提供的信息中没有",
    "所给资料中没有",
    "知识库中没有",
]


def keyword_hit_rate(answer: str, keywords: Sequence[str]) -> float:
    """答案关键词命中率。

    计算 `keywords` 中有多少个出现在 `answer` 里，返回比例 [0, 1]。
    若 keywords 为空，认为"无期望关键词"，返回 1.0（不扣分）。

    Parameters
    ----------
    answer : str
        LLM 生成的答案文本。
    keywords : Sequence[str]
        期望出现在答案中的关键词列表（来自 golden_case.expected_answer_keywords）。
    """
    if not keywords:
        return 1.0
    valid = [kw for kw in keywords if kw]
    if not valid:
        return 1.0
    hits = sum(1 for kw in valid if kw in answer)
    return hits / len(valid)


def forbidden_count(answer: str, phrases: Sequence[str]) -> int:
    """forbidden_phrases 命中数。

    返回 `phrases` 中有多少个出现在 `answer` 里（越多越差）。
    """
    if not phrases:
        return 0
    return sum(1 for ph in phrases if ph and ph in answer)


def no_answer_score(answer: str) -> float:
    """无效回答检测分。

    Returns
    -------
    float
        0.0 = 有效回答；1.0 = 无效回答（命中无效模式词表）。
    """
    if not answer or not answer.strip():
        return 1.0
    for pattern in _NO_ANSWER_PATTERNS:
        if pattern in answer:
            return 1.0
    return 0.0


def compute_all(
    answer: Optional[str],
    golden_case: Any,
) -> Dict[str, float]:
    """计算全部答案质量指标。

    Parameters
    ----------
    answer : str | None
        LLM 生成的最终答案。
    golden_case : GoldenCase | dict
        黄金集 case 对象，需含 `expected_answer_keywords` 和 `forbidden_phrases`。

    Returns
    -------
    dict
        键：``keyword_hit_rate``, ``forbidden_count``, ``no_answer_score``
    """
    from dataclasses import is_dataclass, asdict

    if isinstance(golden_case, dict):
        gc = golden_case
    elif is_dataclass(golden_case):
        gc = asdict(golden_case)
    else:
        gc = {
            attr: getattr(golden_case, attr, None)
            for attr in ("expected_answer_keywords", "forbidden_phrases")
        }

    ans = answer or ""
    keywords: List[str] = gc.get("expected_answer_keywords") or []
    forbidden: List[str] = gc.get("forbidden_phrases") or []

    return {
        "keyword_hit_rate": keyword_hit_rate(ans, keywords),
        "forbidden_count": float(forbidden_count(ans, forbidden)),
        "no_answer_score": no_answer_score(ans),
    }
