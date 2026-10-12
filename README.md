# semantic-ocr

One container, two inference routes, one API key. Built for a single user
(RewindOS) — tuned for per-request speed and capability, not multi-tenant
throughput.

| Route | Auth | Purpose |
|---|---|---|
| `GET /health` | open | Instant liveness, never touches models (platform probe) |
| `GET /ready` | open | Read-only status: model name, loaded flags, uptime |
| `GET /` | open | Route index |
| `POST /embed` | key | Native embeddings: `{"inputs":[...],"normalize":true}` → 768-dim vectors |
| `POST /api/embeddings` | key | Legacy Ollama: `{"model","prompt"}` → `{"model","embedding":[...], "embeddings":[[...]]}` (both shapes for strict parsers) |
| `POST /api/embed` | key | Current Ollama: `{"model","input"}` → `{"model","embeddings":[...]}` |
| `POST /v1/embeddings` | key | OpenAI schema: `{"input","model"}` → `{"object":"list","data":[...]}` |
| `POST /ocr` | key | Multipart image → `{"text","confidence","lines","boxes"}` (word boxes included) |

Auth: `Authorization: Bearer <API_TOKEN>`, or `?api_key=<token>` fallback for
clients that can't set headers (note: the key then lands in proxy access logs).
`/health` and `/ready` stay open so Dokploy/Traefik probes work without a key.
The container refuses to start without `API_TOKEN` (fail-closed).

## Models (capability-per-CPU picks, verified 2026-10-12)

- **Embeddings: `BAAI/bge-base-en-v1.5`** (768-dim, ONNX via FastEmbed, CPU-only).
  Clearly stronger retrieval than small/MiniLM at ~2–3× the CPU cost; still
  milliseconds per text on 2 CPUs. No torch (~2 GB image bloat avoided).
  Context is 512 tokens — chunk long docs client-side; inputs over 4000 chars
  are rejected (tune via `MAX_EMBED_CHARS`).
- **OCR: RapidOCR (PaddleOCR ONNX, CPU-only).** Measured on a 4-line synthetic
  screenshot: all lines perfect at ~0.9 confidence in ~1.3 s wall (~0.5 s
  inference); tesseract on the same image garbled 3 of 4 lines. Returns
  per-line bounding boxes for grounding. No torch, no GPU.

## RewindOS wiring

- Base URL: `https://ai.shrijit.tech` (after deploy below)
- Embeddings endpoint: `POST https://ai.shrijit.tech/api/embeddings`
  with `{"model": "<anything>", "prompt": "<text>"}` — the `model` name is
  echoed back; vectors are always 768-dim `bge-base-en-v1.5`.
- Header: `Authorization: Bearer <API_TOKEN>` (or `?api_key=` fallback).
- OCR: `POST https://ai.shrijit.tech/ocr` as multipart `file` (png/jpg/webp/
  tiff/bmp, ≤12 MB).

## Deploy (Dokploy)

No Dokploy MCP is mounted in the agent runtime, so this is a dashboard deploy:

1. `openssl rand -base64 32` → store as `SEMANTIC_OCR_API_TOKEN` in
   self-hosted Infisical (`secrets.shrijit.tech`). Never commit it.
2. Push this folder to `github.com/shrijit37/semantic-ocr` (`main`).
3. Dokploy (`manage.shrijit.tech`) → **AI Gateway** project → Create →
   **Application** → source `shrijit37/semantic-ocr`, build type Dockerfile.
4. Domain: `ai.shrijit.tech` → container port `8000`, https, letsencrypt.
5. Environment: `API_TOKEN=<token from step 1>`. Defaults cover the rest
   (`EMBED_MODEL`, caps, `OMP_NUM_THREADS=2`).
6. Deploy. First boot downloads ~450 MB of weights at build time
   (pre-downloaded in Docker build); runtime needs no HuggingFace access
   (`HF_HUB_OFFLINE=1`).

Alternative: paste `dokploy-compose.yaml` as a raw Compose in the same project
(requires the repo from step 2 for its build context).

Verify:

```bash
curl https://ai.shrijit.tech/health
curl -X POST https://ai.shrijit.tech/api/embeddings \
  -H "Authorization: Bearer $SEMANTIC_OCR_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text","prompt":"rewind workflow state"}'
curl -X POST https://ai.shrijit.tech/ocr \
  -H "Authorization: Bearer $SEMANTIC_OCR_API_TOKEN" \
  -F "file=@scan.png;type=image/png"
```

## Limits (single-user burst budget)

Container capped at **2 CPU / 4 GB** (reserve 0.25 CPU / 1 GB). Per-request
caps: 64 texts or 4000 chars per embed call, 12 MB per image, 180 s OCR
timeout. Single uvicorn worker — ONNX and RapidOCR thread internally; extra
workers would only duplicate ~1 GB model RAM each.

## After deploy

Record in `agent.md`: service row under AI Gateway (compose/app ID, domain
`ai.shrijit.tech`), tag `tier-2, active, stateless, public`. Possible
follow-ups (not now): multilingual OCR packs, `bge-m3` if Hindi/multilingual
retrieval is needed, query-prefix tuning per corpus.
