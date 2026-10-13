"""Offline OCR backfill — runs RapidOCR LOCALLY on a Google Colab VM.

Compute runs on the VM, not against the live service: this exists to keep
bulk/backlog OCR off the prod node entirely. Nothing here POSTs to
ai.shrijit.tech, and no API token is needed or used.

Output is one JSON line per image in a single JSONL file, shaped like the
live /ocr response (text/confidence/lines/boxes) so downstream consumers
treat both sources uniformly:

    {"file": "shot_001.png", "text": "...", "confidence": 0.93,
     "lines": 4, "boxes": [[[x, y], ...]], "timings": {...}}

Failures get {"file": ..., "error": "..."} lines. The run is resumable:
stems already present in the JSONL are skipped.

Engine is pinned to rapidocr-onnxruntime 1.2.3 — the same version prod
runs (1.4.x measured slower; see repo history). 1.2.3 accepts no engine
kwargs, so it uses plain CPU onnxruntime; a GPU VM does not help and a
free CPU VM (2 vCPU) sustains roughly the full 5s-cadence capture rate on
its own.

Usage (on the VM):
    python colab_backfill.py --in input --out output/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
ENGINE_VERSION = "1.2.3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local RapidOCR batch backfill")
    p.add_argument("--in", dest="in_dir", default=os.environ.get("BACKFILL_IN", "input"))
    p.add_argument(
        "--out",
        dest="out_jsonl",
        default=os.environ.get("BACKFILL_OUT", "output/results.jsonl"),
    )
    p.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    return p.parse_args()


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["file"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def ocr_file(engine, path: Path) -> dict:
    """Run one image through the engine; service-compatible record or error line."""
    t_all = time.perf_counter()
    try:
        result, elapse = engine(str(path))
    except Exception as e:  # noqa: BLE001 — one bad image must not kill the batch
        return {"file": path.name, "error": f"{type(e).__name__}: {e}"}

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

    try:  # engine returns per-stage seconds; tolerate unexpected shapes
        infer_s = round(sum(elapse), 3) if elapse else None
    except TypeError:
        infer_s = None

    return {
        "file": path.name,
        "text": "\n".join(lines),
        "confidence": round(sum(confs) / len(confs), 3) if confs else None,
        "lines": len(lines),
        "boxes": boxes,
        "timings": {
            "engine": f"rapidocr-onnxruntime {ENGINE_VERSION} (local)",
            "infer_s": infer_s,
            "total_s": round(time.perf_counter() - t_all, 3),
        },
    }


def main() -> int:
    args = parse_args()
    in_dir, out_path = Path(args.in_dir), Path(args.out_jsonl)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no images in {in_dir}")
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_path)
    todo = [p for p in files if p.name not in done]
    already = len(files) - len(todo)
    print(f"{len(files)} images, {already} already done, {len(todo)} to run")
    if not todo:
        return 0

    from rapidocr_onnxruntime import RapidOCR  # noqa: import cost paid once

    engine = RapidOCR()
    print("engine loaded", flush=True)

    t0 = time.perf_counter()
    ok = err = 0
    with out_path.open("a") as out:
        for n, path in enumerate(todo, 1):
            rec = ocr_file(engine, path)
            if "error" in rec:
                err += 1
            else:
                ok += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()  # crash-safe: finished lines survive a dead session
            if n % 10 == 0 or n == len(todo):
                rate = n / max(time.perf_counter() - t0, 0.001)
                print(f"[{n}/{len(todo)}] ok={ok} err={err}  {rate:.2f} files/s  last={path.name}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s: ok={ok} err={err} -> {out_path}")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
