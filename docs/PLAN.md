# detecto — Build Plan & Working Log

> Single source of truth for resuming this project from scratch.
> Read this first. Update it as work progresses.

---

## 1. What we are building

A real-time person detection and counting system:

- **Backend** — FastAPI serving a pretrained YOLOv8n model, exposed as
  `POST /detect`, `GET /history`, `DELETE /reset`.
- **Frontend** — React/Vite dashboard with a **Detection View** (upload +
  bounding-box overlay) and a **History View** (table + chart + filters).
- **Storage** — SQLite logging timestamp, person count, average confidence,
  and inference time per detection.
- **Target** — CPU-only. Graded metrics in §7.

Role-play context: an automation/safety team monitoring entrances, classrooms,
or warehouses. The output must be accurate, responsive, and operator-friendly.

---

## 2. Locked-in decisions (do not relitigate without a reason)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **ONNX Runtime, not PyTorch/ultralytics** | Target machine has no sudo and limited disk. Torch needs ~2.5 GB; ONNX stack is ~400 MB. Same YOLOv8n model, same results. |
| D2 | **Model = YOLOv8n**, exported to ONNX at imgsz=640 | Best CPU speed/accuracy tradeoff; 6 MB weights; COCO class 0 = person. |
| D3 | **Backend owns the coordinate transform** | It knows the letterbox parameters; the frontend only scales original-pixel boxes to display size. Single source of truth — this is the #1 bug source. |
| D4 | **SQLite + WAL mode** | Zero-config, supports concurrent reads, and filtering lives in SQL. |
| D5 | **Filtering in SQL, not the client** | Hundreds of rows, not millions. Smaller payloads, simpler frontend. |
| D6 | **Bounding boxes rendered on a `<canvas>` overlay** | Sharp at any display size, efficient for ~50 boxes, uses React refs. |
| D7 | **Recharts** for the count-over-time chart | React-native, lightweight, good API. |
| D8 | **NMS implemented by us** (numpy / `cv2.dnn.NMSBoxes`) | ONNX export excludes NMS, so we control IoU threshold and ordering. |
| D9 | **CORS from env, never `*`** | Brief requirement; real origins only. |

---

## 3. Environment constraints (the target machine)

Measured on the ThinkPad X1 Carbon 5th gen:

| Resource | Value | Implication |
|----------|-------|-------------|
| CPU | Intel i5-7300U, 2 cores / 4 threads @ 2.6 GHz | Inference may be slow; `IMG_SIZE` must be tunable (640→416→320). |
| RAM | 7.5 GB total, ~4.4 GB available | ONNX YOLOv8n uses a few hundred MB — fine. |
| Disk | 233 GB, ~5.7 GB free after cache cleanup | Fits the ONNX stack. Torch would not. |
| Privileges | **No sudo** | Python venv + `pip install` only. No system packages. |
| Python | 3.12.3 (`/usr/bin/python3`) | venv works; pip 24.0. |
| Node | v22.15.0 / npm 11.19.0 | Vite frontend is fine. |
| venv | `.venv` at repo root, 402 MB | Created and verified. |

Disk cleanup performed (no sudo): cleared `~/.cache` and npm cache → freed 4.6 GB.

---

## 4. Progress tracker

### Phase 0 — Setup & configuration
- [x] `requirements.txt` — pinned, clean install verified in fresh venv
- [x] `.env.example` — all tunables documented
- [x] `.gitignore` — verified `.venv` and `.env` are ignored
- [ ] `docs/PLAN.md` ← this file (you are reading it)
- [ ] `docs/decisions.md` — expand D1–D9 into full write-ups

### Phase 1 — Backend core
- [x] `backend/models/record.py` — Pydantic schemas + SQLite repository (smoke-tested)
- [ ] `backend/utils/preprocessing.py` — decode, letterbox, normalize, **inverse transform**, NMS
- [ ] `backend/routes/detect.py` — `POST /detect`
- [ ] `backend/routes/history.py` — `GET /history`, `DELETE /reset`
- [ ] `backend/main.py` — lifespan model load, CORS, error handlers

### Phase 2 — Backend hardening
- [ ] Validation: 415 / 400 / 413 with structured error bodies
- [ ] Structured timing logs (inference vs. total)
- [ ] `scripts/smoke.sh` — curl tests for every endpoint + 3 failures

### Phase 3 — Frontend: Detection View
- [ ] Vite scaffold (`package.json`, `vite.config.js`, `main.jsx`, `App.jsx`)
- [ ] `ImageUpload.jsx` — drag-drop + picker, client-side validation
- [ ] `CanvasOverlay.jsx` — refs + draw loop
- [ ] `DetectionPanel.jsx` — count / avg confidence / timing / errors
- [ ] `DetectionPage.jsx` — orchestration

### Phase 4 — Frontend: History View
- [ ] `HistoryFilters.jsx`
- [ ] `HistoryTable.jsx`
- [ ] `CountChart.jsx`
- [ ] `HistoryPage.jsx`

### Phase 5 — Testing & benchmarking
- [ ] Assemble 10+ sample images (easy, occlusion, partial, distant, dark, zero-person)
- [ ] `tests/ground_truth.csv` — manual hand count
- [ ] `scripts/benchmark.py` — per-image metrics + markdown table
- [ ] Run benchmark, record **real** numbers

### Phase 6 — Documentation
- [ ] `README.md` — purpose, architecture diagram, setup, methodology, results, screenshots, retrospective

### Phase 7 — Bonus (only after 0–6)
- [ ] Webcam feed, region alerts, CSV export, hourly stats

---

## 5. Build order (exact next actions)

One file per step. Complete, commented, runnable. Stop and verify after each.

| Step | File | Status |
|------|------|--------|
| 1 | `requirements.txt` | DONE |
| 2 | `.env.example` | DONE |
| 3 | `.gitignore` | DONE |
| 4 | `backend/models/record.py` | DONE |
| 5 | `backend/utils/preprocessing.py` | **NEXT** |
| 6 | `backend/routes/detect.py` | pending |
| 7 | `backend/routes/history.py` | pending |
| 8 | `backend/main.py` | pending |
| 9 | `scripts/smoke.sh` | pending |
| 10 | frontend scaffold | pending |
| 11 | `ImageUpload.jsx` | pending |
| 12 | `CanvasOverlay.jsx` | pending |
| 13 | `DetectionPanel.jsx` | pending |
| 14 | `DetectionPage.jsx` | pending |
| 15 | `HistoryFilters.jsx` | pending |
| 16 | `HistoryTable.jsx` | pending |
| 17 | `CountChart.jsx` | pending |
| 18 | `HistoryPage.jsx` | pending |
| 19 | `scripts/benchmark.py` | pending |
| 20 | `README.md` | pending |

---

## 6. Blocker: obtaining `yolov8n.onnx`

The ONNX stack does **not** include torch, so we cannot export the model
locally. Resolve this before Step 6 (`routes/detect.py`) can be tested.

**Chosen path (B1): export once on Google Colab.**
1. In a Colab cell: `pip install ultralytics`
2. `yolo export model=yolov8n.pt format=onnx opset=12 imgsz=640`
3. Download the resulting `yolov8n.onnx` (~12 MB).
4. Place it at `backend/weights/yolov8n.onnx` (gitignored).

Alternatives (documented, not chosen):
- **B2** — download a trusted pre-exported ONNX (faster, but third-party weights).
- **B3** — MobileNet-SSD via OpenCV DNN (no export, lower accuracy).

**Model I/O contract (record this so preprocessing matches):**
- Input: float32 tensor `[1, 3, IMG_SIZE, IMG_SIZE]`, RGB, values 0–1, NCHW.
- Output: float32 `[1, 84, 8400]` — 4 box coords (cx, cy, w, h in input pixels)
  + 80 COCO class scores. No NMS included.
- Class 0 = person.

---

## 7. Graded metrics (targets to measure honestly)

| Metric | Definition | Target |
|--------|-----------|--------|
| Detection accuracy | correct detections ÷ total visible persons × 100 | ≥ 85 % |
| False positives | non-person boxes | ≤ 10 % |
| Avg inference time | mean per-image processing time (same hardware) | ≤ 1.5 s |
| Avg confidence | mean confidence of valid person detections | ≥ 0.7 |
| System reliability | all test images processed without crashing | 100 % |

**Never fabricate numbers.** If a target is missed, document why.

---

## 8. Command reference

```bash
# Activate the backend environment
source .venv/bin/activate          # or: .venv/bin/python ...

# Verify dependencies are installed
.venv/bin/python -c "import fastapi, cv2, onnxruntime; print('ok')"

# Run the backend (once main.py exists)
.venv/bin/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run the frontend (once scaffolded)
cd frontend && npm install && npm run dev

# Run smoke tests (once written)
bash scripts/smoke.sh

# Run the benchmark (once written)
.venv/bin/python scripts/benchmark.py
```

---

## 9. Open questions / to-decide

- [ ] Confirm `yolov8n.onnx` obtained (B1) and placed in `backend/weights/`.
- [ ] Decide exact error-response envelope shape (proposed:
      `{"error": {"code": str, "message": str, "detail": str | null}}`).
- [ ] Decide whether annotated image is returned by default or opt-in
      (proposed: opt-in via `?annotated=true` to save CPU).
- [ ] Confirm `IMG_SIZE=640` meets the ≤1.5 s target on this CPU; if not,
      drop to 416 and record the accuracy tradeoff.
- [ ] Sample image sources + licences to record in README.
