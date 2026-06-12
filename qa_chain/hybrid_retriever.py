"""Hybrid 检索：FAISS 密集向量 + BM25 关键词 通过 EnsembleRetriever 融合。

设计动机
--------
对中文 RAG，纯 dense embedding 在以下场景容易失分：
- 章节号 / 数字编号查询（"第16章讲了什么"）—— PDF 提取常把
  "第16 章" 拆成 "第16\n章"，被向量稀释
- 罕见专有名词、代码、公式 —— dense 模型词表覆盖不足
- 模板化页头页脚水印 —— 稀释每个 chunk 的语义信号

BM25 在以上场景给出强势补强；FAISS 在语义改写、近义词等场景给出强势补强。
两者通过 RRF（Reciprocal Rank Fusion）融合，是中文 RAG 几乎无副作用
的标配增强。

实现要点
--------
- jieba 中文分词：BM25 默认按空格切，对中文几乎不可用；这里用 jieba.cut_for_search
  生成更细粒度的 token，准/召双高
- BM25 的输入 docs 直接从 FAISS 的 docstore 抽（避免重新加载 PDF），
  保持和 dense 检索"看的是同一份语料"
- source_filter 同时作用于 dense 和 BM25：dense 走 FAISS 的 metadata filter，
  BM25 在喂 docs 阶段就先过滤一次
- 返回 langchain Document（不带 score）；score 由调用方按需补 0.0
  （EnsembleRetriever 用 RRF 融合后没有原始 score 概念，只有 rank）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── BM25 retriever 缓存 ──────────────────────────────────────────────────
# 生产环境（run_gradio）每次查询的 source filter 是动态的（用户切换"知识库范围"），
# 不能像评估框架那样在构建时就把 BM25 corpus 缩到 dataset.source_filter 范围内。
# 折中方案：把 BM25 retriever 按 vectordb 实例缓存（id(vectordb) 作 key），
# 整个 vectordb 的全部 docs 作为 corpus 一次性建好，每次查询后再按 allowed_sources
# 后过滤。BM25 倒排建一次是 O(N) jieba 分词，对 1k-10k chunks 量级几秒搞定。
_bm25_cache: dict = {}


def _get_or_build_bm25(vectordb: Any):
    """按 vectordb 实例缓存 BM25Retriever；命中则直接返回，否则全量 corpus 建一次。"""
    key = id(vectordb)
    cached = _bm25_cache.get(key)
    if cached is not None:
        return cached
    docs = _docs_from_faiss(vectordb, source_filter=None)
    if not docs:
        raise ValueError("vectordb 中没有任何文档，无法建 BM25 索引")
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as exc:
        raise ImportError(
            "BM25Retriever 不可用，请确认 langchain_community 已安装"
        ) from exc
    bm25 = BM25Retriever.from_documents(
        documents=docs,
        preprocess_func=_jieba_tokenize,
    )
    _bm25_cache[key] = bm25
    logger.info(
        "BM25 retriever built and cached for vectordb id=%s (n_docs=%d)",
        key,
        len(docs),
    )
    return bm25


def clear_bm25_cache() -> None:
    """清空 BM25 缓存。生产代码在 vectordb 重建后应调用。"""
    _bm25_cache.clear()


# ── BaseRetriever 子类工厂 ────────────────────────────────────────────
# 两个用途：
# 1) _TopKRetrieverWrapper：把 EnsembleRetriever（不做 top_k 截断）截到 top_k，
#    防止 LangChain Chain 拿到超量 context
# 2) HybridFaissBm25Retriever：直接用 hybrid_search_with_filter 的 BaseRetriever，
#    比 EnsembleRetriever 召回更宽（dense+BM25 各 fetch 200 而不是 top_k 个），
#    给 fast_api 这种"无 baseline 约束、要召回稳"的生产场景用
def _make_retriever_classes():
    """延迟创建子类：避免 import langchain_core 阻塞模块加载（CLI / 评估
    入口在没装 hybrid 依赖的极少数场景下也能用 dense 路径）。"""
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict

    class _TopKRetrieverWrapper(BaseRetriever):
        """把内部 retriever 的输出强制截断到 top_k。"""

        inner: Any
        top_k: int

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            docs = list(self.inner.invoke(query))
            return docs[: self.top_k]

        async def _aget_relevant_documents(self, query: str, *, run_manager=None):
            docs = list(await self.inner.ainvoke(query))
            return docs[: self.top_k]

    class HybridFaissBm25Retriever(BaseRetriever):
        """BaseRetriever 适配层，内部调 hybrid_search_with_filter。

        与 build_hybrid_retriever 返回的 EnsembleRetriever 包装相比：
        - dense 召回宽度 = fetch_k（默认 200），不是 top_k
        - 自带 RRF 实现，结果与 hybrid_search_with_filter 完全一致
        - 严格返回 top_k 个 Document
        - 适合 fast_api 这类 "无 baseline 约束" 的生产场景

        给 LangChain RetrievalQA / ConversationalRetrievalChain 直接当 retriever 用。
        """

        vectordb: Any
        top_k: int = 5
        weights: Tuple[float, float] = (0.5, 0.5)
        rrf_k: int = 60
        fetch_k: Optional[int] = None

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            pairs = hybrid_search_with_filter(
                vectordb=self.vectordb,
                query=query,
                top_k=self.top_k,
                weights=self.weights,
                rrf_k=self.rrf_k,
                fetch_k=self.fetch_k,
            )
            return [d for d, _ in pairs]

    return _TopKRetrieverWrapper, HybridFaissBm25Retriever


_retriever_classes_cache = None


def _get_topk_wrapper_class():
    global _retriever_classes_cache
    if _retriever_classes_cache is None:
        _retriever_classes_cache = _make_retriever_classes()
    return _retriever_classes_cache[0]


def get_hybrid_retriever_class():
    """返回 HybridFaissBm25Retriever 类（延迟 import langchain_core）。"""
    global _retriever_classes_cache
    if _retriever_classes_cache is None:
        _retriever_classes_cache = _make_retriever_classes()
    return _retriever_classes_cache[1]


def _jieba_tokenize(text: str) -> List[str]:
    """jieba 分词，给 BM25 用。

    用 cut_for_search 的"搜索引擎模式"——它会把长词再切成短词，
    比如 "强化学习" 会输出 ["强化", "学习", "强化学习"]，召回更稳。
    """
    import jieba

    if not text:
        return []
    return [tok for tok in jieba.cut_for_search(text) if tok and not tok.isspace()]


def _docs_from_faiss(
    vectordb: Any,
    source_filter: Optional[str] = None,
) -> List[Any]:
    """从 FAISS 的内部 docstore 抽出 langchain Document 列表。

    BM25Retriever 需要全量 docs 来构建倒排索引；从 docstore 直接拿避免
    重新加载 PDF / 重新分块。
    """
    docstore = getattr(vectordb, "docstore", None)
    if docstore is None:
        raise ValueError("vectordb 没有 docstore 属性，无法抽取 BM25 语料")
    raw_dict = getattr(docstore, "_dict", None)
    if raw_dict is None:
        raise ValueError("vectordb.docstore 没有 _dict 属性（langchain 接口变了？）")

    docs: List[Any] = []
    for doc in raw_dict.values():
        if source_filter:
            meta = getattr(doc, "metadata", None) or {}
            src = str(meta.get("source", ""))
            if source_filter not in os.path.basename(src):
                continue
        docs.append(doc)
    return docs


def build_hybrid_retriever(
    vectordb: Any,
    top_k: int = 10,
    weights: Sequence[float] = (0.5, 0.5),
    source_filter: Optional[str] = None,
    bm25_fetch_k: Optional[int] = None,
    dense_fetch_k: Optional[int] = None,
):
    """构造 FAISS + BM25 EnsembleRetriever。

    Parameters
    ----------
    vectordb : FAISS
        已加载的 FAISS vectorstore（来自 validation.utils.load_vectordb）。
    top_k : int
        最终返回 top_k 个文档（融合后）。
    weights : (float, float)
        (dense_weight, bm25_weight)，对 RRF 融合分数加权；和不必为 1。
    source_filter : str, optional
        文件名 substring；同时过滤 dense（metadata filter）和 BM25（喂语料前过滤）。
    bm25_fetch_k : int, optional
        BM25 内部召回数；默认 max(top_k * 4, 40)，给后续融合留余量。
    dense_fetch_k : int, optional
        FAISS 内部 fetch_k；默认 max(top_k * 20, 200)。带 metadata filter 时
        必须显著大于 top_k，否则过滤后可能不足 top_k 个候选。

    Returns
    -------
    retriever-like object
        调用 .invoke(query) 返回融合后的 Document 列表（无 score）。
        优先使用 LangChain 的 EnsembleRetriever；若当前环境导入路径不可用，
        自动降级为本地 RRF fallback 实现（行为保持一致）。
    """
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError as exc:
        raise ImportError(
            "BM25Retriever 不可用，请确认 langchain_community 已安装"
        ) from exc
    EnsembleRetriever = None
    try:
        # 常见路径（langchain 0.2+）
        from langchain.retrievers import EnsembleRetriever  # type: ignore
    except Exception:
        try:
            # 部分版本需要从子模块导入
            from langchain.retrievers.ensemble import EnsembleRetriever  # type: ignore
        except Exception:
            try:
                # 兼容部分旧版本导入路径
                from langchain_community.retrievers import EnsembleRetriever  # type: ignore
            except Exception:
                EnsembleRetriever = None

    # 评估框架已用此默认值固化了 hybrid baseline（pumpkin mrr 0.75→0.83），
    # 不要在不通知用户的情况下偷偷改这两个参数；如果生产侧需要更宽召回，
    # 在调用 build_hybrid_retriever 时显式传 bm25_fetch_k / dense_fetch_k。
    bm25_fetch_k = bm25_fetch_k or max(top_k * 4, 40)
    dense_fetch_k = dense_fetch_k or max(top_k * 20, 200)

    # ── BM25 retriever ──
    bm25_docs = _docs_from_faiss(vectordb, source_filter=source_filter)
    if not bm25_docs:
        raise ValueError(
            f"过滤 source_filter={source_filter!r} 后没有 BM25 候选文档；"
            "请检查向量库是否为空或 filter 是否过严"
        )
    bm25 = BM25Retriever.from_documents(
        documents=bm25_docs,
        preprocess_func=_jieba_tokenize,
    )
    bm25.k = bm25_fetch_k

    # ── FAISS dense retriever ──
    # langchain 的 .as_retriever 接受 search_kwargs；带 metadata filter 时
    # 用闭包函数（langchain_community.vectorstores.FAISS 支持 callable filter）。
    search_kwargs: dict = {"k": top_k, "fetch_k": dense_fetch_k}
    if source_filter:
        sf = source_filter

        def _filter(meta: dict) -> bool:
            return sf in os.path.basename(str((meta or {}).get("source", "")))

        search_kwargs["filter"] = _filter
    dense = vectordb.as_retriever(search_kwargs=search_kwargs)

    if EnsembleRetriever is not None:
        inner = EnsembleRetriever(
            retrievers=[dense, bm25],
            weights=list(weights),
        )

        # ── 关键：EnsembleRetriever 不做 top_k 截断，会返回 dense+BM25 的去重
        # 并集（RRF 排序），喂给 LangChain Chain 时会塞超量 context。
        # 这里包一层 BaseRetriever 子类强制截断到 top_k。
        Wrapper = _get_topk_wrapper_class()
        ensemble = Wrapper(inner=inner, top_k=top_k)
    else:
        logger.warning(
            "EnsembleRetriever 不可用，降级为本地 RRF fallback 实现"
        )

        allowed_sources = None
        if source_filter:
            allowed_sources = {
                str((getattr(d, "metadata", None) or {}).get("source", ""))
                for d in bm25_docs
            }

        class _FallbackHybridRetriever:
            def invoke(self, query: str):
                pairs = hybrid_search_with_filter(
                    vectordb=vectordb,
                    query=query,
                    top_k=top_k,
                    allowed_sources=allowed_sources,
                    weights=(float(weights[0]), float(weights[1])),
                    fetch_k=max(bm25_fetch_k, dense_fetch_k),
                    bm25_retriever=bm25,
                )
                return [doc for doc, _ in pairs]

        ensemble = _FallbackHybridRetriever()

    logger.info(
        "hybrid retriever ready: bm25_docs=%d, top_k=%d, weights=%s, "
        "bm25_fetch_k=%d, dense_fetch_k=%d, source_filter=%r",
        len(bm25_docs),
        top_k,
        list(weights),
        bm25_fetch_k,
        dense_fetch_k,
        source_filter,
    )
    return ensemble


def hybrid_search(
    vectordb: Any,
    query: str,
    top_k: int = 10,
    weights: Sequence[float] = (0.5, 0.5),
    source_filter: Optional[str] = None,
    retriever: Optional[Any] = None,
) -> List[tuple]:
    """便捷封装：执行一次 hybrid 检索，返回 [(doc, score), ...]。

    score 在 EnsembleRetriever 下是 RRF 融合分数，但 langchain 不直接暴露；
    这里用 1/rank 作为 score 占位（保持和 FAISS similarity_search_with_score
    相同的 (doc, score) 契约，便于 retrieval_runner 复用）。

    Parameters
    ----------
    retriever : EnsembleRetriever, optional
        预构建的 retriever；不传则按其他参数现场构造。
        重复调用同一 vectordb 时强烈建议外部缓存 retriever 以避免每次
        重新构建 BM25 倒排（构建是 O(N) 的）。
    """
    if retriever is None:
        retriever = build_hybrid_retriever(
            vectordb=vectordb,
            top_k=top_k,
            weights=weights,
            source_filter=source_filter,
        )
    docs = retriever.invoke(query)
    docs = list(docs)[:top_k]
    return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]


# ══════════════════════════════════════════════════════════════════════════
# 生产代码用：hybrid_search_with_filter
# ══════════════════════════════════════════════════════════════════════════
#
# 与评估框架的 build_hybrid_retriever() 区别：
# - source filter 是 **每次查询动态传**（不是 corpus 级），支持 run_gradio 的
#   "知识库范围"切换场景
# - 自己实现 RRF（Reciprocal Rank Fusion），返回 [(doc, fused_score), ...]，
#   score 可以用于现有日志输出（_retrieve_with_scope 里会打 score）
# - 不依赖 EnsembleRetriever（它不暴露融合分数、也不支持 per-call filter）

def _doc_identity_key(doc: Any) -> Tuple[Any, ...]:
    """跨 retriever 的稳定 doc 标识。

    FAISS 和 BM25 都从同一个 docstore 取 Document 实例，理论上 id(doc) 也能用，
    但用 (source, page, content[:64]) 更稳，避免 langchain 内部偶尔做 doc copy
    时 id 漂移。
    """
    meta = getattr(doc, "metadata", None) or {}
    src = str(meta.get("source", ""))
    page = meta.get("page", -1)
    content_head = (getattr(doc, "page_content", "") or "")[:64]
    return (src, page, content_head)


def hybrid_search_with_filter(
    vectordb: Any,
    query: str,
    top_k: int = 5,
    allowed_sources: Optional[Iterable[str]] = None,
    weights: Tuple[float, float] = (0.5, 0.5),
    rrf_k: int = 60,
    fetch_k: Optional[int] = None,
    bm25_retriever: Optional[Any] = None,
) -> List[Tuple[Any, float]]:
    """按"动态 source 集合"做 Hybrid 检索，返回 [(doc, fused_score), ...]。

    用于生产代码（如 run_gradio.py）：每次查询的 allowed_sources 由用户当前
    选择的"知识库范围"决定，不是预先固定。

    Parameters
    ----------
    vectordb : FAISS
        已加载的 langchain FAISS vectorstore。
    query : str
        用户查询。
    top_k : int
        最终返回的结果数。
    allowed_sources : Iterable[str], optional
        允许的 source 集合（完整路径，与 vectordb chunk metadata.source 直接相等比较）；
        None / 空集合表示不过滤。
    weights : (float, float)
        (dense_weight, bm25_weight)；权重不必为 1。
    rrf_k : int
        RRF 平滑常数，typical 值 60。
    fetch_k : int, optional
        dense 和 BM25 各自的内部召回数；默认 max(200, top_k * 20)。
        必须显著大于 top_k 才能给 source 过滤留够余量。
    bm25_retriever : optional
        预构建的 BM25Retriever；不传时按 vectordb 自动 cache 一份。

    Returns
    -------
    List[(Document, fused_score)]
        按 fused_score 降序排序，长度 ≤ top_k。score 是 RRF 融合分数（不是相似度）。
    """
    if fetch_k is None:
        fetch_k = max(200, top_k * 20)

    allowed_set: Optional[set] = None
    if allowed_sources is not None:
        allowed_set = {s for s in allowed_sources if s}
        if not allowed_set:
            allowed_set = None

    # ── Dense 召回（FAISS）──
    if allowed_set is None:
        dense_pairs = vectordb.similarity_search_with_score(query, k=fetch_k)
    else:
        dense_pairs = vectordb.similarity_search_with_score(
            query,
            k=fetch_k,
            filter=lambda meta: (meta or {}).get("source") in allowed_set,
            fetch_k=fetch_k,
        )
    dense_docs = [d for d, _ in dense_pairs]

    # ── BM25 召回（全 corpus，结果 post-filter）──
    if bm25_retriever is None:
        bm25_retriever = _get_or_build_bm25(vectordb)
    bm25_retriever.k = fetch_k
    bm25_docs_raw = list(bm25_retriever.invoke(query))
    if allowed_set is not None:
        bm25_docs = [
            d for d in bm25_docs_raw
            if (getattr(d, "metadata", None) or {}).get("source") in allowed_set
        ]
    else:
        bm25_docs = bm25_docs_raw

    # ── RRF 融合 ──
    w_dense, w_bm25 = weights
    fused: dict = {}
    doc_by_key: dict = {}
    for rank, doc in enumerate(dense_docs, start=1):
        k = _doc_identity_key(doc)
        fused[k] = fused.get(k, 0.0) + w_dense / (rrf_k + rank)
        doc_by_key.setdefault(k, doc)
    for rank, doc in enumerate(bm25_docs, start=1):
        k = _doc_identity_key(doc)
        fused[k] = fused.get(k, 0.0) + w_bm25 / (rrf_k + rank)
        doc_by_key.setdefault(k, doc)

    if not fused:
        logger.info(
            "hybrid_search_with_filter: 0 results (allowed_sources=%s)",
            "ALL" if allowed_set is None else f"{len(allowed_set)} sources",
        )
        return []

    ranked_keys = sorted(fused, key=fused.get, reverse=True)[:top_k]
    out = [(doc_by_key[k], fused[k]) for k in ranked_keys]
    logger.debug(
        "hybrid_search_with_filter: dense=%d bm25=%d fused=%d top_k=%d",
        len(dense_docs),
        len(bm25_docs),
        len(fused),
        len(out),
    )
    return out
