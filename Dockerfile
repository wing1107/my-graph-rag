# ============================================================
# Stage 1: 安装 Python 依赖
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装系统依赖（unstructured / easyocr 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libgl1 poppler-utils libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. 先装 torch CPU 版
# 2. 装其余依赖（排除 langchain-huggingface，它的依赖声明与 langchain-core 1.x 冲突）
# 3. 单独用 --no-deps 装 langchain-huggingface（运行时实际兼容）
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && grep -v 'langchain-huggingface' requirements.txt > /tmp/req_no_hf.txt \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/req_no_hf.txt \
    && pip install --no-cache-dir --no-deps langchain-huggingface==0.3.0

# ============================================================
# Stage 2: 运行镜像（只保留运行时必需内容）
# ============================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# 运行时系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libgl1 poppler-utils libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目代码（不含 .venv / data_base，由 volume 挂载）
COPY . .

# 创建数据目录挂载点
RUN mkdir -p data_base/kownledge_db data_base/vector_db data_base/ocr_cache sessions

# 预下载 Embedding 模型到固定本地路径（绕过 HF hub 缓存查找，离线可靠）
ENV M3E_LOCAL_PATH=/app/models/m3e-base
ENV MINILM_LOCAL_PATH=/app/models/paraphrase-multilingual-minilm-l12-v2
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('moka-ai/m3e-base', local_dir='/app/models/m3e-base'); \
snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', local_dir='/app/models/paraphrase-multilingual-minilm-l12-v2')"

# 离线模式：模型已内置，禁止运行时联网
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

VOLUME ["/app/data_base/vector_db", "/app/data_base/kownledge_db", "/app/sessions"]

EXPOSE 7860

# 默认启动 Gradio Web UI；可用 CMD 覆盖改为 FastAPI
CMD ["python", "run_gradio.py"]
