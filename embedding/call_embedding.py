import logging
import os

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

_embedding_cache: dict = {}


def _get_local_multilingual_e5_path() -> str | None:
    _ = load_dotenv(find_dotenv())
    return os.environ.get("MULTILINGUAL_E5_PATH")


def _is_valid_model_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    if size < 1024:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        if b"version https://git-lfs" in header or b"oid sha256:" in header:
            return False
    except Exception:
        return False
    return True


def _find_local_model(candidates: list[str]) -> str | None:
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        weight_files = [
            f for f in os.listdir(candidate)
            if f.endswith((".bin", ".safetensors", ".pt"))
        ]
        if any(_is_valid_model_file(os.path.join(candidate, f)) for f in weight_files):
            return candidate
    return None


def get_embedding(embedding: str, embedding_key: str = None, env_file: str = None):
    """
    按名称选择并返回 Embedding 对象。

    推荐路径（LangChain 新版）：
        langchain_huggingface.HuggingFaceEmbeddings
    OpenAI 路径：
        langchain_community.embeddings.OpenAIEmbeddings
    本地 HuggingFace 模型（m3e、multilingual）启用模块级缓存，避免重复加载。
    """
    # ── 本地 HuggingFace 模型：命中缓存直接返回 ──
    if embedding in ("m3e", "multilingual") and embedding in _embedding_cache:
        logger.info("Embedding 缓存命中: %s", embedding)
        return _embedding_cache[embedding]

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # 兼容旧环境：未安装 langchain-huggingface 时回退旧导入路径。
        from langchain_community.embeddings import HuggingFaceEmbeddings

    if embedding == "openai":
        from langchain_community.embeddings import OpenAIEmbeddings
        if embedding_key is None:
            from llm.call_llm import parse_llm_api_key
            embedding_key = parse_llm_api_key("openai", env_file)
        return OpenAIEmbeddings(openai_api_key=embedding_key)

    elif embedding == "zhipuai":
        from embedding.zhipuai_embedding import ZhipuAIEmbeddings
        if embedding_key is None:
            from llm.call_llm import parse_llm_api_key
            embedding_key = parse_llm_api_key("zhipuai", env_file)
        return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)

    elif embedding == "m3e":
        _local_m3e = os.environ.get("M3E_LOCAL_PATH", "")
        primary = _local_m3e if (_local_m3e and _find_local_model([_local_m3e])) else "moka-ai/m3e-base"
        _local_minilm = os.environ.get("MINILM_LOCAL_PATH", "")
        fallback = _local_minilm if (_local_minilm and _find_local_model([_local_minilm])) else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        try:
            emb = HuggingFaceEmbeddings(model_name=primary)
            _embedding_cache["m3e"] = emb
            logger.info("m3e embedding 加载成功: %s", primary)
            return emb
        except OSError:
            logger.warning("m3e 模型不可用，回退到 %s", fallback)
            emb = HuggingFaceEmbeddings(model_name=fallback)
            _embedding_cache["m3e"] = emb
            return emb

    elif embedding == "multilingual":
        env_path = _get_local_multilingual_e5_path()
        _local_minilm = os.environ.get("MINILM_LOCAL_PATH", "")
        fallback_small = _local_minilm if (_local_minilm and _find_local_model([_local_minilm])) else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        primary_hf = "intfloat/multilingual-e5-large"

        candidates = [p for p in [env_path] if p]
        local_path = _find_local_model(candidates)

        if local_path:
            try:
                emb = HuggingFaceEmbeddings(model_name=local_path)
                _embedding_cache["multilingual"] = emb
                logger.info("multilingual-e5 本地模型加载成功: %s", local_path)
                return emb
            except MemoryError:
                logger.warning("内存不足，回退到小模型")
            except Exception as e:
                logger.warning("本地模型加载失败（%s），尝试 HuggingFace 在线", e)

        try:
            emb = HuggingFaceEmbeddings(model_name=primary_hf)
            _embedding_cache["multilingual"] = emb
            logger.info("multilingual-e5-large 在线模型加载成功")
            return emb
        except MemoryError:
            logger.warning("内存不足，回退到小模型")
        except Exception as e:
            logger.warning("multilingual-e5-large 加载失败（%s），回退到小模型", e)

        emb = HuggingFaceEmbeddings(model_name=fallback_small)
        _embedding_cache["multilingual"] = emb
        return emb

    else:
        # 兜底：当成 HuggingFace model_name 直接传入
        return HuggingFaceEmbeddings(model_name=embedding)


def clear_embedding_cache() -> None:
    """清空 Embedding 缓存，供需要释放内存时调用。"""
    _embedding_cache.clear()
    logger.info("Embedding 缓存已清空")
