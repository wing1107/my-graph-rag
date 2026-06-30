"""
qa_chain/nodes.py — LangGraph Self-RAG 图的五个节点函数

节点签名统一为：
    fn(state: RAGState, config: RunnableConfig) -> dict

运行时依赖（vectordb、llm）通过 config["configurable"] 注入，
避免将重型对象放进可序列化的 state。

拓扑：
    retrieve → grade_documents → [不相关 + retry<MAX] → rewrite_query → retrieve(loop)
                                ↓ [有相关 chunk]
                             generate
                                ↓
                          grade_answer → [幻觉 + retry<MAX] → generate(loop)
                                ↓ [可信]
                               END
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from qa_chain.state import MAX_RETRIES, RAGState
from qa_chain.prompt_templates import (
    DEFAULT_TEMPLATE,
    DIRECT_TEMPLATE,
    GRADE_DOC_PROMPT,
    GRADE_ANSWER_PROMPT,
    REWRITE_QUERY_PROMPT,
)

logger = logging.getLogger(__name__)

# ── 轻量 grade 模型（glm-4-flash 够快、够便宜）──────────────────────────────
# 通过 configurable 注入的 llm 是主问答 LLM；grade 节点用同一个实例即可，
# 实际部署时可以在 configurable 里单独注入 "grade_llm"。
_GRADE_MODEL_DEFAULT = "glm-4-flash"

# ── 中文数字→阿拉伯数字章节号映射 ─────────────────────────────────────────
_CN_DIGIT = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_num_to_int(cn: str) -> int:
    """将简单中文数字串（支持百以内）转换为整数，失败返回 0。

    支持形式：一、十、十六、二十、二十五、一百零三 等常见写法。
    南瓜书共 16 章，实际只会碰到 1-20 以内的数字。
    """
    cn = cn.strip()
    if not cn:
        return 0
    result = 0
    # 处理 "百" 位
    if "百" in cn:
        parts = cn.split("百", 1)
        hundreds = _CN_DIGIT.get(parts[0], 0) if parts[0] else 1
        result += hundreds * 100
        cn = parts[1]
    # 处理 "十" 位
    if "十" in cn:
        parts = cn.split("十", 1)
        tens = _CN_DIGIT.get(parts[0], 0) if parts[0] else 1
        units = _CN_DIGIT.get(parts[1], 0) if parts[1] else 0
        result += tens * 10 + units
    else:
        result += _CN_DIGIT.get(cn, 0)
    return result


import re as _re

_CN_CHAPTER_RE = _re.compile(
    r"第\s*([一二三四五六七八九十百零]+)\s*([章課课])"
)


def _normalize_chapter_numbers(text: str) -> str:
    """将查询中的中文章节数字替换为阿拉伯数字，便于 BM25 精确匹配。

    例：第十六章 → 第16章，第二十五課 → 第25課
    对已是阿拉伯数字的查询无影响。
    """
    def _replace(m: "_re.Match") -> str:
        num = _cn_num_to_int(m.group(1))
        if num > 0:
            return f"第{num}{m.group(2)}"
        return m.group(0)  # 无法解析则保留原文

    return _CN_CHAPTER_RE.sub(_replace, text)


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _get_configurable(config: RunnableConfig, key: str, default=None) -> Any:
    return (config or {}).get("configurable", {}).get(key, default)


def _source_to_label(source: str) -> str:
    return os.path.basename(source) if source else "(未知来源)"


def _build_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        meta = d.metadata or {}
        src = _source_to_label(meta.get("source", ""))
        page = meta.get("page", "?")
        parts.append(f"[来源: {src} | 页码: {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _llm_text(llm: Any, prompt: str) -> str:
    """统一调用 LLM，兼容 langchain LLM / ChatModel 两种接口。"""
    result = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    if hasattr(result, "content"):
        return result.content.strip()
    return str(result).strip()


# ── Router 轻量关键词预过滤 ────────────────────────────────────────────
# 命中这些模式的问题大概率需要检索知识库，直接走 retrieve，省一次 LLM 调用
_RETRIEVE_KEYWORDS = _re.compile(
    r'第\s*\d+\s*[课課章节節]|南瓜书|西瓜书|大家的日语|初级[12]|N[1-5]'
    r'|教材|课文|例文|単語|練習|文法|文型|会[话話]',
    _re.IGNORECASE,
)
# 命中这些模式的问题大概率不需要检索，直接走 direct
_DIRECT_KEYWORDS = _re.compile(
    r'^(你好|hello|hi|嗨|早上好|晚上好)'
    r'|伪代码|时间复杂度|空间复杂度|怎么定义.*类|怎么写.*函数'
    r'|Python|Java|C\+\+|JavaScript'
    r'|量子|相对论|化学|物理|数学公式'
    r'|天气|今天|几点了',
    _re.IGNORECASE,
)

_ROUTER_PROMPT = """你是一个路由决策器。判断用户问题是否需要检索教材知识库。

知识库内容：日语教材（大家的日语 初级1/初级2）和机器学习教材（南瓜书/西瓜书）。

规则：
- 问题依赖以上教材内容（章节、课文、语法、单词、公式推导）→ retrieve
- 通用知识、编程问题、闲聊、通识解释 → direct
- 只输出一个词

示例：
问题：第25课的语法要点是什么？ → retrieve
问题：南瓜书第16章讲了什么？ → retrieve
问题：Python怎么定义一个类？ → direct
问题：给我一个快速排序的伪代码 → direct
问题：请解释量子纠缠 → direct
问题：你好 → direct
问题：帮我总结第25课和第26课的动词 → retrieve

用户问题：
{question}
"""


def router_node(state: RAGState, config: RunnableConfig) -> dict:
    """路由节点：判断本轮是否需要先检索知识库。

    三级判定：
    1. 关键词预过滤（零 LLM 调用，<1ms）
    2. LLM 路由分类（一次 LLM 调用）
    3. 异常兜底 → retrieve
    """
    question = state["question"]

    # ── 第一级：关键词预过滤 ─────────────────────────────────────────
    if _RETRIEVE_KEYWORDS.search(question):
        logger.info("[router_node] 关键词命中 retrieve: %r", question)
        return {"route_decision": "retrieve"}
    if _DIRECT_KEYWORDS.search(question):
        logger.info("[router_node] 关键词命中 direct: %r", question)
        return {"route_decision": "direct"}

    # ── 第二级：LLM 路由分类 ─────────────────────────────────────────
    llm = _get_configurable(config, "llm")
    if llm is None:
        logger.warning("[router_node] 未注入 LLM，默认 route=retrieve")
        return {"route_decision": "retrieve"}

    prompt = _ROUTER_PROMPT.format(question=question)
    try:
        raw = _llm_text(llm, prompt).strip().lower()
    except Exception as e:
        logger.warning("[router_node] 路由判定失败: %s，默认 route=retrieve", e)
        return {"route_decision": "retrieve"}

    route = "retrieve" if "retrieve" in raw else "direct"
    logger.info("[router_node] question=%r raw=%r -> route=%s", question, raw, route)
    return {"route_decision": route}


# ═══════════════════════════════════════════════════════════════
# 节点 1 — retrieve_node
# ═══════════════════════════════════════════════════════════════

def retrieve_node(state: RAGState, config: RunnableConfig) -> dict:
    """
    执行向量检索（dense 或 hybrid），将结果写入 state["documents"]。

    检索使用 rewritten_question（若有）或原始 question。
    vectordb 通过 config["configurable"]["vectordb"] 注入。
    source_filter 来自 state，支持"知识库范围"切换。
    """
    vectordb = _get_configurable(config, "vectordb")
    retriever_kind = _get_configurable(config, "retriever_kind", "dense")
    top_k = state.get("top_k", 4)
    question = state.get("rewritten_question") or state["question"]
    # 将中文章节数字（如"第十六章"）规范化为阿拉伯数字（"第16章"），
    # 避免 BM25 因字符不匹配导致章节内容完全检索不到。
    question = _normalize_chapter_numbers(question)

    # 「有哪些 / 所有 / 全部」+ 具体课程 → 该课的语法点分散在多个 chunk，
    # 默认 top_k=4 只能召回少量语法点；自动扩大召回量避免漏答。
    _EXHAUSTIVE_RE = _re.compile(r'有哪些|所有|全部|列举|包括哪些|都有什么|包含哪些')
    _LESSON_RE = _re.compile(r'第\s*\d+\s*[课課章节節]')
    if _EXHAUSTIVE_RE.search(question) and _LESSON_RE.search(question):
        scaled_k = max(top_k * 3, 12)
        logger.info(
            "[retrieve_node] 列举型课程查询，top_k %d → %d", top_k, scaled_k
        )
        top_k = scaled_k

    allowed_sources = state.get("source_filter")

    logger.info(
        "[retrieve_node] question=%r retriever=%s top_k=%d scope=%s",
        question,
        retriever_kind,
        top_k,
        "ALL" if allowed_sources is None else f"{len(allowed_sources)} sources",
    )

    if retriever_kind == "hybrid":
        from qa_chain.hybrid_retriever import hybrid_search_with_filter
        pairs = hybrid_search_with_filter(
            vectordb=vectordb,
            query=question,
            top_k=top_k,
            allowed_sources=allowed_sources,
        )
        docs = [d for d, _ in pairs]
    else:
        # dense — 纯 FAISS
        if allowed_sources is None:
            pairs = vectordb.similarity_search_with_score(question, k=top_k)
        else:
            fetch_k = max(200, top_k * 20)
            pairs = vectordb.similarity_search_with_score(
                question,
                k=top_k,
                filter=lambda meta: meta.get("source") in allowed_sources,
                fetch_k=fetch_k,
            )
        docs = [d for d, _ in pairs]

    logger.info("[retrieve_node] 召回 %d 个切片", len(docs))
    return {"documents": docs}


# ═══════════════════════════════════════════════════════════════
# 节点 2 — grade_documents_node
# ═══════════════════════════════════════════════════════════════

def grade_documents_node(state: RAGState, config: RunnableConfig) -> dict:
    """
    用 LLM 批量过滤不相关的 chunk。

    对每个 Document 调用 GRADE_DOC_PROMPT（返回 yes/no），只保留 yes 的 doc。
    若全部被过滤，清空 documents（不保留兜底），让 should_rewrite 感知
    len(docs)=0 并触发改写重检索；超过 MAX_RETRIES 后 should_rewrite 会放行
    到 generate，此时 generate 依靠 TEMPLATE_V4_COT 的通用知识兜底指令回答。
    retry_count += 1 供 should_rewrite 条件边使用。

    优先使用 configurable 中注入的 grade_llm（轻量评分专用），
    不存在时退回主 llm，保证与主模型切换解耦。
    """
    llm = _get_configurable(config, "grade_llm") or _get_configurable(config, "llm")
    docs: list[Document] = state.get("documents", [])
    question = state.get("rewritten_question") or state["question"]
    retry_count = state.get("retry_count", 0)

    if not docs:
        logger.info("[grade_documents_node] 无文档可评，跳过评分")
        return {"documents": [], "retry_count": retry_count + 1}

    if llm is None:
        # 没有注入 LLM（如测试环境）—— 跳过评分，原样返回
        logger.warning("[grade_documents_node] 未注入 LLM，跳过文档评分")
        return {"documents": docs, "retry_count": retry_count + 1}

    graded: list[Document] = []
    for doc in docs:
        prompt = GRADE_DOC_PROMPT.format(
            question=question,
            document=doc.page_content[:800],  # 截断节省 token
        )
        try:
            verdict = _llm_text(llm, prompt).lower()
        except Exception as e:
            logger.warning("[grade_documents_node] LLM 评分失败: %s，保留该 doc", e)
            verdict = "yes"

        if "yes" in verdict:
            graded.append(doc)
        else:
            meta = doc.metadata or {}
            logger.debug(
                "[grade_documents_node] 过滤: src=%s page=%s",
                _source_to_label(meta.get("source", "")),
                meta.get("page", "?"),
            )

    # 全部过滤时不保留兜底 doc：
    # 旧逻辑保留 docs[0] 会骗过 should_rewrite（len(docs)=1 ≠ 0），
    # 导致 generate 拿着无关 context 生成幻觉/混淆答案。
    # 现在清空 documents，让 should_rewrite 感知 len(docs)=0 并触发改写重检索；
    # 若已超过 MAX_RETRIES，should_rewrite 仍会放行到 generate，
    # 此时 generate 依靠 TEMPLATE_V4_COT 中的通用知识兜底指令回答。
    if not graded:
        logger.info(
            "[grade_documents_node] 全部 %d 个 doc 被过滤，清空文档触发改写重检索",
            len(docs),
        )

    logger.info(
        "[grade_documents_node] 原始 %d 个 → 保留 %d 个，retry_count → %d",
        len(docs),
        len(graded),
        retry_count + 1,
    )
    return {"documents": graded, "retry_count": retry_count + 1}


# ═══════════════════════════════════════════════════════════════
# 节点 3 — rewrite_query_node
# ═══════════════════════════════════════════════════════════════

def rewrite_query_node(state: RAGState, config: RunnableConfig) -> dict:
    """
    LLM 改写问题，结果写入 state["rewritten_question"]。

    在 grade_documents 判定"无相关 doc 且未超重试上限"时被路由到此节点。
    改写后重新执行 retrieve_node（图中的循环边）。
    """
    llm = _get_configurable(config, "llm")
    question = state["question"]

    if llm is None:
        logger.warning("[rewrite_query_node] 未注入 LLM，跳过改写")
        return {"rewritten_question": question}

    prompt = REWRITE_QUERY_PROMPT.format(question=question)
    try:
        rewritten = _llm_text(llm, prompt)
    except Exception as e:
        logger.warning("[rewrite_query_node] 改写失败: %s，使用原始问题", e)
        rewritten = question

    logger.info(
        "[rewrite_query_node] 原始: %r → 改写: %r",
        question,
        rewritten,
    )
    return {"rewritten_question": rewritten}


# ═══════════════════════════════════════════════════════════════
# 节点 4 — generate_node
# ═══════════════════════════════════════════════════════════════

def generate_node(state: RAGState, config: RunnableConfig) -> dict:
    """
    用 DEFAULT_TEMPLATE（V4_COT）+ 过滤后的文档拼 context 调 LLM 生成答案。

    同时将本轮问答追加到 chat_history（HumanMessage + AIMessage），
    供多轮对话场景使用。

    历史对话通过 config["configurable"]["history_len"] 控制使用长度（默认 3 轮）。
    """
    llm = _get_configurable(config, "llm")
    history_len = _get_configurable(config, "history_len", 3)

    docs: list[Document] = state.get("documents", [])
    question = state.get("rewritten_question") or state["question"]
    original_question = state["question"]
    chat_history = state.get("chat_history", [])

    # ── 根据路由决策选择 prompt 模板 ────────────────────────────────
    is_direct = state.get("route_decision") == "direct"

    if is_direct:
        # direct 路径：不拼 context，使用通用知识回答
        context = ""
    else:
        context = _build_context(docs) if docs else "（知识库中未找到相关文档）"

    # 防止 context 过长导致 API 500
    MAX_CONTEXT_CHARS = 9000
    if context and len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[内容已截断]"
        logger.warning("[generate_node] context 超过 %d 字符，已截断", MAX_CONTEXT_CHARS)

    # 拼接历史对话块（仅取最近 history_len 轮）
    history_block = ""
    if chat_history and history_len > 0:
        recent = chat_history[-(history_len * 2):]  # 每轮 2 条（human + ai）
        lines = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                lines.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"助手: {msg.content}")
        if lines:
            history_block = "以下是历史对话，仅供理解上下文：\n" + "\n".join(lines) + "\n\n"

    # 路由分支选择模板
    if is_direct:
        prompt = history_block + DIRECT_TEMPLATE.format(question=question)
    else:
        prompt = history_block + DEFAULT_TEMPLATE.format(
            context=context,
            question=question,
        )

    retry_count = state.get("retry_count", 0)
    logger.info(
        "[generate_node] question=%r docs=%d history_msgs=%d retry_count=%d",
        question,
        len(docs),
        len(chat_history),
        retry_count,
    )

    if llm is None:
        generation = "（LLM 未注入，无法生成答案）"
        logger.error("[generate_node] 未注入 LLM")
    else:
        try:
            generation = _llm_text(llm, prompt)
        except Exception as e:
            logger.exception("[generate_node] LLM 调用失败: %s", e)
            generation = f"生成失败：{e}"

    # 只在第一次生成时追加到 chat_history（retry_count=0 说明是首次进入 generate）；
    # 重试时（retry_count>0）不追加，避免同一问题在历史中重复堆积。
    if retry_count == 0:
        new_history = list(chat_history) + [
            HumanMessage(content=original_question),
            AIMessage(content=generation),
        ]
    else:
        # 重试：更新最后一条 AIMessage 为最新回答
        new_history = list(chat_history)
        if new_history and isinstance(new_history[-1], AIMessage):
            new_history[-1] = AIMessage(content=generation)

    logger.info("[generate_node] 生成完成，长度 %d 字符", len(generation))
    return {"generation": generation, "chat_history": new_history, "hallucination_flag": False}


# ═══════════════════════════════════════════════════════════════
# 节点 5 — grade_answer_node
# ═══════════════════════════════════════════════════════════════

def grade_answer_node(state: RAGState, config: RunnableConfig) -> dict:
    """
    验证生成的答案是否有文档依据（防止幻觉）。

    调用 GRADE_ANSWER_PROMPT（返回 grounded/not_grounded），
    将结果写入 state["hallucination_flag"]：
    - False → 答案可信，图路由到 END
    - True  → 检测到幻觉，路由回 generate_node（最多重试 MAX_RETRIES 次）

    优先使用 configurable 中注入的 grade_llm（轻量评分专用），
    不存在时退回主 llm，保证与主模型切换解耦。
    """
    llm = _get_configurable(config, "grade_llm") or _get_configurable(config, "llm")
    docs: list[Document] = state.get("documents", [])
    generation = state.get("generation", "")

    if llm is None or not docs or not generation:
        # 无法评分时默认放行（保证可用性）
        logger.warning("[grade_answer_node] 跳过答案评分（llm=%s docs=%d gen=%d）",
                       llm is not None, len(docs), len(generation))
        return {"hallucination_flag": False}

    context = _build_context(docs)[:3000]  # 截断节省 token
    prompt = GRADE_ANSWER_PROMPT.format(context=context, generation=generation[:1500])

    try:
        verdict = _llm_text(llm, prompt).lower()
    except Exception as e:
        logger.warning("[grade_answer_node] LLM 评分失败: %s，放行", e)
        return {"hallucination_flag": False}

    is_hallucination = "not_grounded" in verdict
    retry_count = state.get("retry_count", 0)
    logger.info(
        "[grade_answer_node] verdict=%r hallucination=%s retry_count → %d",
        verdict[:50],
        is_hallucination,
        retry_count + 1,
    )
    return {"hallucination_flag": is_hallucination, "retry_count": retry_count + 1}


# ═══════════════════════════════════════════════════════════════
# Conditional edge 决策函数
# ═══════════════════════════════════════════════════════════════

def should_rewrite(state: RAGState) -> str:
    """
    grade_documents 后的路由：
    - 文档为空且 retry_count < MAX_RETRIES → "rewrite" → rewrite_query_node
    - 否则 → "generate" → generate_node
    """
    docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    if not docs and retry_count < MAX_RETRIES:
        logger.info(
            "[should_rewrite] 文档为空且 retry_count=%d < %d，触发改写",
            retry_count,
            MAX_RETRIES,
        )
        return "rewrite"
    return "generate"


def should_retry_generate(state: RAGState) -> str:
    """
    grade_answer 后的路由：
    - 幻觉检测 + retry_count < MAX_RETRIES → "generate" → generate_node 重试
    - 否则 → "end"
    """
    hallucination = state.get("hallucination_flag", False)
    retry_count = state.get("retry_count", 0)

    if hallucination and retry_count < MAX_RETRIES:
        logger.info(
            "[should_retry_generate] 检测到幻觉，retry_count=%d，触发重新生成",
            retry_count,
        )
        return "generate"
    return "end"
