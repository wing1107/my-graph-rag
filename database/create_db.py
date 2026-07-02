import base64
import glob
import hashlib
import json
import logging
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_OCR_CACHE_DIR = os.path.join(_ROOT, "data_base", "ocr_cache")

# OCR 引擎优先级：qwen-vl > easyocr > paddle
OCR_ENGINE_PRIORITY = os.environ.get("OCR_ENGINE", "qwen-vl")


def _ocr_cache_key(file_path: str) -> str:
    """根据文件内容生成稳定缓存键（MD5）。"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _ocr_cache_path(file_path: str) -> str:
    """根据文件内容（MD5）生成缓存文件路径（与文件名无关）。"""
    key_hash = _ocr_cache_key(file_path)
    os.makedirs(_OCR_CACHE_DIR, exist_ok=True)
    return os.path.join(_OCR_CACHE_DIR, f"{key_hash}.json")


def _load_ocr_cache(file_path: str):
    os.makedirs(_OCR_CACHE_DIR, exist_ok=True)
    key_hash = _ocr_cache_key(file_path)
    canonical = os.path.join(_OCR_CACHE_DIR, f"{key_hash}.json")
    legacy_candidates = glob.glob(os.path.join(_OCR_CACHE_DIR, f"*_{key_hash}.json"))
    candidates = [canonical] + legacy_candidates

    for cp in candidates:
        if not os.path.exists(cp):
            continue
        try:
            with open(cp, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容旧命名缓存：命中后回填为新命名，后续可直接命中。
            if cp != canonical:
                try:
                    with open(canonical, "w", encoding="utf-8") as wf:
                        json.dump(data, wf, ensure_ascii=False)
                except Exception:
                    pass
            return data
        except Exception:
            continue
    return None


def _save_ocr_cache(file_path: str, docs_data: list):
    cp = _ocr_cache_path(file_path)
    try:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False)
    except Exception as e:
        logging.getLogger(__name__).warning("写入OCR缓存失败: %s", e)


# ── LangChain 0.2.x import 路径 ──────────────────────────────────────────
from dotenv import load_dotenv, find_dotenv
from embedding.call_embedding import get_embedding

# langchain_community 0.2.x
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

# langchain_text_splitters 独立包（0.2.x 拆出）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma 保留（兼容 create_db legacy 函数）
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

_OCR_THRESHOLD_CHARS_PER_PAGE = 50


class SmartPDFLoader:
    """
    智能 PDF 加载器。
    优先用 PyMuPDF 提取嵌入文本；若每页平均字符数低于阈值则自动改用 OCR。
    """

    def __init__(self, file_path: str, ocr_langs: list = None):
        self.file_path = file_path
        self.ocr_langs = ocr_langs or ["ja", "en"]

    def load(self):
        import fitz

        doc = fitz.open(self.file_path)
        total_chars = sum(len(page.get_text()) for page in doc)
        num_pages = len(doc)
        doc.close()

        avg_chars = total_chars / max(num_pages, 1)
        logger.info(
            "PDF检测 [%s]: 共%d页，平均%.1f字符/页",
            os.path.basename(self.file_path),
            num_pages,
            avg_chars,
        )

        if avg_chars >= _OCR_THRESHOLD_CHARS_PER_PAGE:
            return PyMuPDFLoader(self.file_path).load()

        logger.warning(
            "检测到图片型PDF（平均%.1f字符/页 < %d），将使用OCR识别。",
            avg_chars,
            _OCR_THRESHOLD_CHARS_PER_PAGE,
        )
        return self._load_with_ocr(num_pages)

    def _load_with_ocr(self, num_pages: int):
        import fitz
        from langchain_core.documents import Document

        cached = _load_ocr_cache(self.file_path)
        if cached is not None:
            logger.info(
                "OCR缓存命中 [%s]：直接加载 %d 个文档",
                os.path.basename(self.file_path),
                len(cached),
            )
            return [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in cached
            ]

        reader_info = self._init_ocr_reader()
        if reader_info is None:
            logger.error("所有 OCR 引擎均不可用。")
            return []

        backend, reader = reader_info
        doc = fitz.open(self.file_path)
        result_docs = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            text = self._ocr_image(backend, reader, img_bytes)
            if text:
                result_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": self.file_path, "page": page_num},
                    )
                )
            if (page_num + 1) % 10 == 0 or page_num == len(doc) - 1:
                logger.info("OCR进度: %d/%d 页", page_num + 1, num_pages)

        doc.close()
        _save_ocr_cache(
            self.file_path,
            [{"page_content": d.page_content, "metadata": d.metadata} for d in result_docs],
        )
        return result_docs

    def _init_ocr_reader(self):
        engine_priority = OCR_ENGINE_PRIORITY.lower()

        if engine_priority in ("qwen-vl", "qwen", "dashscope"):
            try:
                api_key = os.environ.get("DASHSCOPE_API_KEY")
                if api_key:
                    from dashscope import MultiModalConversation  # noqa: F401
                    logger.info("Qwen-VL OCR 初始化成功")
                    return ("qwen-vl", api_key)
                else:
                    logger.warning("未设置 DASHSCOPE_API_KEY，跳过 Qwen-VL OCR")
            except ImportError:
                logger.warning("未安装 dashscope，尝试本地 OCR...")
            except Exception as e:
                logger.warning("Qwen-VL 初始化失败（%s）", e)

        try:
            import easyocr
            reader = easyocr.Reader(self.ocr_langs, gpu=False)
            logger.info("EasyOCR 初始化成功")
            return ("easyocr", reader)
        except ImportError:
            logger.warning("未安装 easyocr，尝试 PaddleOCR...")
        except Exception as e:
            logger.warning("EasyOCR 初始化失败（%s）", e)

        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang="japan")
            logger.info("PaddleOCR 初始化成功")
            return ("paddle", ocr)
        except ImportError:
            logger.error("未安装 paddleocr：pip install paddlepaddle paddleocr")
        except Exception as e:
            logger.error("PaddleOCR 初始化失败：%s", e)

        return None

    @staticmethod
    def _ocr_image(backend: str, reader, img_bytes: bytes) -> str:
        if backend == "qwen-vl":
            return SmartPDFLoader._ocr_with_qwen_vl(reader, img_bytes)
        if backend == "easyocr":
            texts = reader.readtext(img_bytes, detail=0)
            return "\n".join(t for t in texts if t.strip())
        if backend == "paddle":
            import io
            import numpy as np
            from PIL import Image
            img_array = np.array(Image.open(io.BytesIO(img_bytes)))
            try:
                results = list(reader.predict(img_array))
                texts = []
                for res in results:
                    if hasattr(res, "rec_texts"):
                        texts.extend(t for t in res.rec_texts if t and t.strip())
                    elif isinstance(res, dict) and "rec_texts" in res:
                        texts.extend(t for t in res["rec_texts"] if t and t.strip())
                return "\n".join(texts)
            except Exception:
                result = reader.ocr(img_array)
                if result and result[0]:
                    return "\n".join(
                        line[1][0] for line in result[0]
                        if line and line[1] and line[1][0].strip()
                    )
                return ""
        return ""

    @staticmethod
    def _ocr_with_qwen_vl(api_key: str, img_bytes: bytes) -> str:
        from dashscope import MultiModalConversation
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{img_base64}"},
                    {
                        "text": (
                            "请精确识别并提取这张图片中的所有文字内容。"
                            "这是一本日语教材的页面，包含日语（平假名、片假名、汉字）和中文。"
                            "请保持原文的格式和换行，不要翻译或解释，只输出识别到的原文文字。"
                            "如果有课程编号（如「第1課」「第25課」等），请务必准确识别。"
                        )
                    },
                ],
            }
        ]
        try:
            response = MultiModalConversation.call(
                model="qwen3.6-plus",
                messages=messages,
                api_key=api_key,
            )
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)
                    return "\n".join(texts)
                return str(content) if content else ""
            else:
                logger.warning("Qwen-VL OCR 调用失败: status=%s", response.status_code)
                return ""
        except Exception as e:
            logger.warning("Qwen-VL OCR 异常: %s", e)
            return ""


# ── 路径常量 ──────────────────────────────────────────────────────────────
DEFAULT_DB_PATH = os.path.join(_ROOT, "data_base", "kownledge_db")
DEFAULT_PERSIST_PATH = os.path.join(_ROOT, "data_base", "vector_db", "chroma")
EMBEDDING_META_FILE = "embedding_meta.json"
LEGACY_FALLBACK_EMBEDDING_MODEL = "m3e"


# ── FAISS 工具函数 ─────────────────────────────────────────────────────────

def _read_faiss_index_dim(persist_directory: str):
    index_path = os.path.join(persist_directory, "index.faiss")
    if not os.path.exists(index_path):
        return None
    try:
        import faiss
        index = faiss.read_index(index_path)
        return index.d
    except Exception as e:
        logger.warning("读取 FAISS 索引维度失败: %s", e)
        return None


def _embedding_dim(embedding_obj) -> int:
    try:
        test_emb = embedding_obj.embed_query("test")
        return len(test_emb)
    except Exception as e:
        logger.warning("获取 Embedding 维度失败: %s", e)
        return 0


def save_faiss_db(vectordb, persist_directory: str, embedding_obj, embedding_provider: str):
    os.makedirs(persist_directory, exist_ok=True)
    vectordb.save_local(persist_directory)
    meta = {
        "embedding_provider": embedding_provider,
        "dimension": _embedding_dim(embedding_obj),
    }
    meta_path = os.path.join(persist_directory, EMBEDDING_META_FILE)
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("已保存 FAISS 元数据: %s", meta_path)
    except Exception as e:
        logger.warning("保存 Embedding 元数据失败: %s", e)


# ── 文件加载工具 ───────────────────────────────────────────────────────────

def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_list.append(os.path.join(filepath, filename))
    return file_list


def file_loader(file, loaders):
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in os.listdir(file)]
        return
    file_type = file.split(".")[-1].lower()
    if file_type == "pdf":
        loaders.append(SmartPDFLoader(file))
    elif file_type == "md":
        loaders.append(UnstructuredMarkdownLoader(file))
    elif file_type == "txt":
        loaders.append(UnstructuredFileLoader(file))
    elif file_type == "docx":
        loaders.append(UnstructuredWordDocumentLoader(file))


# ── 向量库加载 ─────────────────────────────────────────────────────────────

def load_knowledge_db(path, embeddings):
    """加载 Chroma 向量库（legacy 兼容接口）。"""
    vectordb = Chroma(persist_directory=path, embedding_function=embeddings)
    return vectordb


def load_faiss_db(path, embeddings):
    """加载 FAISS 向量库。"""
    from langchain_community.vectorstores import FAISS

    if isinstance(embeddings, str):
        embeddings = get_embedding(embedding=embeddings)

    vectordb = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectordb


# ── Legacy create_db（Gradio 建库流程不使用此函数，保留向后兼容）──────────

def create_db(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="openai"):
    if files is None:
        return "can't load empty file"
    if not isinstance(files, list):
        files = [files]
    loaders = []
    [file_loader(file, loaders) for file in files]
    docs = []
    for loader in loaders:
        if loader is not None:
            docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    if isinstance(embeddings, str):
        embeddings = get_embedding(embedding=embeddings)
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectordb.persist()
    return vectordb


if __name__ == "__main__":
    create_db(embeddings="openai")
