"""semantic-ocr — one container, two inference routes, one API key.

Routes:
  GET  /health  open, instant, never touches models (platform probe)
  GET  /ready   open, read-only status (loaded flags, versions)
  GET  /        open, route index
  POST /embed   Bearer key — semantic embeddings (FastEmbed/ONNX, CPU-only)
  POST /ocr     Bearer key — OCR via RapidOCR (detection + recognition, CPU-only)

Fail-closed: container refuses to start without API_TOKEN.
"""

from __future__ import annotations

import asyncio
import hmac
import io
import logging
import os
import sys
import tempfile
import time

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

log = logging.getLogger("semantic-ocr")

API_TOKEN = os.environ.get("API_TOKEN", "")
if not API_TOKEN:
    print("FATAL: API_TOKEN env var is required (fail-closed).", file=sys.stderr)
    sys.exit(1)

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
MAX_EMBED_BATCH = int(os.environ.get("MAX_EMBED_BATCH", "64"))
MAX_EMBED_CHARS = int(os.environ.get("MAX_EMBED_CHARS", "4000"))
MAX_OCR_BYTES = int(os.environ.get("MAX_OCR_BYTES", str(12 * 1024 * 1024)))
OCR_TIMEOUT_S = int(os.environ.get("OCR_TIMEOUT_S", "180"))
BUILD = os.environ.get("BUILD_SHA", "dev")
STARTED_AT = time.time()

app = FastAPI(title="semantic-ocr", version="1.0.0")

_embedder = None
_embed_error: str | None = None


def require_key(request: Request, authorization: str = Header(default="")) -> bool:
    # Bearer primary. ?api_key= fallback for clients that can't set headers
    # (single-user box; note the key then lands in proxy access logs).
    scheme, _, token = authorization.partition(" ")
    candidate = (
        token
        if scheme.lower() == "bearer" and token
        else request.query_params.get("api_key", "")
    )
    if not candidate:
        raise HTTPException(status_code=401, detail="missing API key")
    if not hmac.compare_digest(candidate, API_TOKEN):
        raise HTTPException(status_code=401, detail="invalid API key")
    return True


def get_embedder():
    """Lazy-load on first /embed call so /health stays instant and
    a failed model download never blocks startup."""
    global _embedder, _embed_error
    if _embedder is None and _embed_error is None:
        try:
            from fastembed import TextEmbedding

            _embedder = TextEmbedding(model_name=EMBED_MODEL)
            log.info("loaded embed model %s", EMBED_MODEL)
        except Exception as e:  # noqa: BLE001 — report, don't crash
            _embed_error = f"{type(e).__name__}: {e}"
            log.exception("embedding model failed to load")
    if _embedder is None:
        raise HTTPException(
            status_code=503, detail=f"embed model unavailable: {_embed_error}"
        )
    return _embedder


def _vectors(texts: list[str], normalize: bool) -> tuple[list[list[float]], int]:
    import numpy as np

    model = get_embedder()
    data: list[list[float]] = []
    dims = 0
    for vec in model.embed(texts):
        v = np.asarray(vec, dtype=np.float32)
        if normalize:
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
        dims = v.shape[0]
        data.append(v.tolist())
    return data, dims


class EmbedRequest(BaseModel):
    inputs: list[str] = Field(min_length=1)
    normalize: bool = True


@app.get("/")
def index():
    return {
        "service": "semantic-ocr",
        "build": BUILD,
        "routes": ["/health", "/ready", "/embed", "/api/embeddings", "/api/embed", "/v1/embeddings", "/ocr"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {
        "embed_model": EMBED_MODEL,
        "embed_loaded": _embedder is not None,
        "ocr_engine": "rapidocr",
        "ocr_loaded": _ocr_engine is not None,
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "build": BUILD,
    }


@app.post("/embed")
def embed(req: EmbedRequest, _ok: bool = Depends(require_key)):
    if len(req.inputs) > MAX_EMBED_BATCH:
        raise HTTPException(
            status_code=413, detail=f"max {MAX_EMBED_BATCH} inputs per request"
        )
    for text in req.inputs:
        if not text.strip():
            raise HTTPException(status_code=422, detail="empty input at index")
        if len(text) > MAX_EMBED_CHARS:
            raise HTTPException(
                status_code=413, detail=f"input exceeds {MAX_EMBED_CHARS} chars"
            )
    data, dims = _vectors(req.inputs, req.normalize)
    return {
        "model": EMBED_MODEL,
        "dimensions": dims,
        "data": [{"index": i, "embedding": e} for i, e in enumerate(data)],
    }


_ocr_engine = None
_ocr_slots = asyncio.Semaphore(int(os.environ.get("OCR_MAX_CONCURRENT", "3")))
_ocr_error: str | None = None


def get_ocr_engine():
    """Lazy-load RapidOCR (detection + recognition models) on first /ocr call."""
    global _ocr_engine, _ocr_error
    if _ocr_engine is None and _ocr_error is None:
        try:
            from rapidocr_onnxruntime import RapidOCR

            # NOTE: rapidocr 1.2.3 rejects engine kwargs (KeyError on unknown
            # keys) — no per-engine thread control on this version. Plain
            # constructor; do NOT pass kwargs here.
            _ocr_engine = RapidOCR()
            log.info("loaded rapidocr engine")
        except Exception as e:  # noqa: BLE001 — report, don't crash
            _ocr_error = f"{type(e).__name__}: {e}"
            log.exception("ocr engine failed to load")
    if _ocr_engine is None:
        raise HTTPException(status_code=503, detail=f"ocr unavailable: {_ocr_error}")
    return _ocr_engine


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    _ok: bool = Depends(require_key),
):
    # Slot covers decode + tmpfile + inference only (~1GB transient each).
    # The upload body is read BEFORE the slot: holding the slot through slow
    # client uploads serialized multi-MB screenshots behind the inference
    # gate (prod 2026-10-13: ~79s slot occupancy per real screenshot, waits
    # 16->170s with slots mostly busy at only ~150% CPU). Body read costs
    # ~12MB RAM per in-flight request; N concurrent uploads fit the 6GiB
    # limit. Extras wait at the gate, never crash. Tune via
    # OCR_MAX_CONCURRENT.
    if not (file.content_type or "").startswith("image/") or (file.content_type or "").endswith("svg+xml"):
        raise HTTPException(status_code=415, detail="upload must be a raster image")
    body = await file.read()
    if len(body) > MAX_OCR_BYTES:
        raise HTTPException(status_code=413, detail=f"image exceeds {MAX_OCR_BYTES} bytes")

    t_enter = time.perf_counter()
    async with _ocr_slots:
        queue_wait_s = time.perf_counter() - t_enter
        if queue_wait_s > 1.0:
            log.warning("ocr queue_wait_s=%.3f (slot held elsewhere)", queue_wait_s)
        result = await _ocr_guarded_body(body)
    timings = result.setdefault("timings", {})
    timings["queue_wait_s"] = round(queue_wait_s, 3)
    timings["server_time_s"] = round(
        time.perf_counter() - t_enter, 3
    )
    return result


async def _ocr_guarded_body(body: bytes):
    # Per-stage timings (diagnostic, additive): splits decode / tmpfile /
    # inference so prod slowdowns can be attributed instead of guessed.
    t_all = time.perf_counter()

    from PIL import Image

    t0 = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(body))
        img.load()
    except Exception:
        raise HTTPException(status_code=422, detail="unreadable image")
    # Native 4K/Retina screenshots multiply decode/tmpfile/inference work per
    # pixel with no OCR-quality gain at OCR-relevant sizes; cap the long edge
    # (0 disables).
    max_side = int(os.environ.get("OCR_MAX_SIDE", "2400"))
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    t_decode = time.perf_counter() - t0

    engine = get_ocr_engine()
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "PNG")
        tmp_path = tmp.name
    t_tmpfile = time.perf_counter() - t0
    try:
        from starlette.concurrency import run_in_threadpool

        try:
            t0 = time.perf_counter()
            result, _elapse = await asyncio.wait_for(
                run_in_threadpool(engine, tmp_path), timeout=OCR_TIMEOUT_S
            )
            t_infer = time.perf_counter() - t0
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="ocr timed out")
    finally:
        os.unlink(tmp_path)

    lines: list[str] = []
    boxes: list = []
    confs: list[float] = []
    for box, txt, conf in result or []:
        txt = (txt or "").strip()
        if not txt:
            continue
        lines.append(txt)
        boxes.append([[round(float(x), 1), round(float(y), 1)] for x, y in box])
        confs.append(float(conf))
    timings = {
        "decode_s": round(t_decode, 3),
        "tmpfile_s": round(t_tmpfile, 3),
        "infer_s": round(t_infer, 3),
        "total_s": round(time.perf_counter() - t_all, 3),
    }
    # WARNING, not INFO: uvicorn's default config suppresses app-logger INFO,
    # which hid real per-stage costs during the 2026-10-13 investigation.
    # Drop to INFO once stage costs are tuned.
    log.warning("ocr timings=%s lines=%d", timings, len(lines))
    return {
        "text": "\n".join(lines),
        "confidence": round(sum(confs) / len(confs), 3) if confs else None,
        "lines": len(lines),
        "boxes": boxes,
        "timings": timings,
    }


class OllamaEmbedRequest(BaseModel):
    model: str = ""
    prompt: str = ""
    options: dict | None = None


class OpenAIEmbedRequest(BaseModel):
    input: str | list[str]
    model: str = ""


@app.post("/api/embeddings")
def ollama_embeddings(req: OllamaEmbedRequest, _ok: bool = Depends(require_key)):
    """Legacy Ollama schema: {"model","prompt"} -> {"model","embedding":[...]}.

    Also returns 2D "embeddings" so strict (Serde) and lenient parsers agree.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="empty prompt")
    if len(req.prompt) > MAX_EMBED_CHARS:
        raise HTTPException(
            status_code=413, detail=f"prompt exceeds {MAX_EMBED_CHARS} chars"
        )
    data, _ = _vectors([req.prompt], True)
    return {
        "model": req.model or EMBED_MODEL,
        "embedding": data[0],
        "embeddings": data,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": len(req.prompt),
    }


@app.post("/api/embed")
def ollama_embed(req: OpenAIEmbedRequest, _ok: bool = Depends(require_key)):
    """Current Ollama schema: {"model","input"} -> {"model","embeddings":[...]}."""
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs or not all(t.strip() for t in inputs):
        raise HTTPException(status_code=422, detail="empty input")
    if len(inputs) > MAX_EMBED_BATCH:
        raise HTTPException(
            status_code=413, detail=f"max {MAX_EMBED_BATCH} inputs per request"
        )
    if any(len(t) > MAX_EMBED_CHARS for t in inputs):
        raise HTTPException(
            status_code=413, detail=f"input exceeds {MAX_EMBED_CHARS} chars"
        )
    data, _ = _vectors(inputs, True)
    return {
        "model": req.model or EMBED_MODEL,
        "embeddings": data,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": sum(len(t) for t in inputs),
    }


@app.post("/v1/embeddings")
def openai_embeddings(req: OpenAIEmbedRequest, _ok: bool = Depends(require_key)):
    """OpenAI-compatible: {"input","model"} -> {"object":"list","data":[...]}."""
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        raise HTTPException(status_code=422, detail="empty input")
    if len(inputs) > MAX_EMBED_BATCH:
        raise HTTPException(
            status_code=413, detail=f"max {MAX_EMBED_BATCH} inputs per request"
        )
    for text in inputs:
        if not text.strip():
            raise HTTPException(status_code=422, detail="empty input at index")
        if len(text) > MAX_EMBED_CHARS:
            raise HTTPException(
                status_code=413, detail=f"input exceeds {MAX_EMBED_CHARS} chars"
            )
    data, _ = _vectors(inputs, True)
    total_chars = sum(len(t) for t in inputs)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": e}
            for i, e in enumerate(data)
        ],
        "model": req.model or EMBED_MODEL,
        "usage": {"prompt_tokens": total_chars // 4, "total_tokens": total_chars // 4},
    }
