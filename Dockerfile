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

# 一次性安装：先固定 torch CPU 版，再装其余依赖
# --extra-index-url 让 pip 同时查 PyPI 和 torch CPU 源，避免 ResolutionImpossible
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

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

# 模型缓存目录（接收方可挂载自己的 HuggingFace 缓存避免重复下载）
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface", "data_base/vector_db", "data_base/kownledge_db", "sessions"]

# 离线模式（镜像已含模型时开启）；若模型通过 volume 挂载，注释掉这两行
# ENV TRANSFORMERS_OFFLINE=1
# ENV HF_HUB_OFFLINE=1

EXPOSE 7860

# 默认启动 Gradio Web UI；可用 CMD 覆盖改为 FastAPI
CMD ["python", "run_gradio.py"]
