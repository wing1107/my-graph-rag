"""evaluation/utils.py — 向量库加载工具（适配层）

与 myRagProject/validation/utils.py 提供相同的 load_vectordb() 接口，
内部使用 database.create_db.load_faiss_db，并处理 Windows 中文路径问题。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from embedding.call_embedding import get_embedding  # noqa: E402


def _is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _load_faiss_with_ascii_fallback(db_path: Path, embedding: Any):
    """
    优先用原路径加载 FAISS；若 Windows 下中文路径触发 RuntimeError，
    则将索引文件复制到 ASCII 临时目录后重试。
    """
    from langchain_community.vectorstores import FAISS

    try:
        return FAISS.load_local(
            folder_path=str(db_path),
            embeddings=embedding,
            allow_dangerous_deserialization=True,
        )
    except RuntimeError as exc:
        err_msg = str(exc)
        is_windows_unicode_issue = (
            os.name == "nt"
            and not _is_ascii_path(db_path)
            and ("could not open" in err_msg or "index.faiss" in err_msg)
        )
        if not is_windows_unicode_issue:
            raise

        ascii_dir = Path(tempfile.gettempdir()) / "faiss_ascii_cache"
        ascii_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path / "index.faiss", ascii_dir / "index.faiss")
        if (db_path / "index.pkl").exists():
            shutil.copy2(db_path / "index.pkl", ascii_dir / "index.pkl")

        print(
            "[INFO] 检测到 Windows 下 FAISS Unicode 路径兼容问题，"
            f"已自动切换到 ASCII 缓存目录: {ascii_dir}"
        )
        return FAISS.load_local(
            folder_path=str(ascii_dir),
            embeddings=embedding,
            allow_dangerous_deserialization=True,
        )


def load_vectordb(embedding_provider: str = None, db_path: str = None) -> Any:
    """
    加载项目 FAISS 向量库。

    参数
    ----
    embedding_provider : str, optional
        Embedding 提供商，支持 "m3e"（默认）或 "multilingual"。
        也可通过环境变量 EMBEDDING_PROVIDER 指定。
    db_path : str, optional
        FAISS 向量库目录。默认按 embedding_provider 自动解析为
        data_base/vector_db/faiss_{provider}/（如 faiss_m3e、faiss_multilingual），
        也可通过环境变量 VECTOR_DB_DIR 显式指定。

    返回
    ----
    FAISS vectorstore 对象
    """
    provider = embedding_provider or os.getenv("EMBEDDING_PROVIDER", "m3e")
    embedding = get_embedding(embedding=provider)

    if db_path is None:
        db_path = os.getenv(
            "VECTOR_DB_DIR",
            str(_PROJECT_ROOT / "data_base" / "vector_db" / f"faiss_{provider}"),
        )

    resolved = Path(db_path).resolve()

    if not (resolved / "index.faiss").exists():
        raise FileNotFoundError(
            f"向量库目录无效: {resolved}\n"
            "请确认存在 index.faiss/index.pkl，或通过环境变量 VECTOR_DB_DIR 指定正确路径。"
        )

    print(f"[INFO] 向量库路径: {resolved}")
    vectordb = _load_faiss_with_ascii_fallback(resolved, embedding)
    return vectordb
