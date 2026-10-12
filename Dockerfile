# semantic-ocr: FastAPI + FastEmbed(ONNX/CPU) + RapidOCR(ONNX/CPU). No torch, no GPU.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    EMBED_MODEL="BAAI/bge-base-en-v1.5" \
    FASTEMBED_CACHE_PATH=/srv/.cache/fastembed

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

# Non-root runtime; own the model cache.
RUN useradd -m -u 10001 app \
    && mkdir -p /srv/.cache/fastembed \
    && chown -R app:app /srv
USER app

# Pre-download the configured embedding weights so the first request is fast
# and runtime needs no HuggingFace (HF_HUB_OFFLINE=1 makes a cache miss fatal,
# so this must load the exact EMBED_MODEL — single source of truth via env).
RUN HF_HUB_OFFLINE=0 python -c "from fastembed import TextEmbedding; import os; TextEmbedding(model_name=os.environ['EMBED_MODEL'])" \
    || echo "WARN: embed pre-download failed, runtime load will fail offline"
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" \
    || echo "WARN: ocr pre-download failed, will lazy-load at runtime"

COPY --chown=app:app app.py .

EXPOSE 8000
# Single worker: onnx + rapidocr are internally threaded; extra workers
# would duplicate model RAM (~1GB each).
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
