"""
Benchmark detecto against hand-counted ground truth.

WHAT IT DOES
    Sends every image in backend/samples/ through the running API, joins the
    results against tests/ground_truth.csv, and computes:
        * detection accuracy  = sum(min(detected, truth)) / sum(truth)
        * false-positive rate = sum(max(0, detected - truth)) / sum(detected)
        * mean inference time = mean(result.inference_time_ms)
        * mean confidence     = mean of ALL returned box confidences
        * reliability         = images processed without error / total
    It prints a per-image markdown table plus a summary, and writes
    tests/benchmark_results.csv for the record.

WHY IT CALLS THE API
    The reported inference_time_ms is measured inside the request path and the
    boxes come back through the real coordinate transform, so this exercises
    the actual system rather than a parallel code path.

REQUIREMENTS
    * Backend running:  uvicorn backend.main:app --port 8000
    * Model loaded:     backend/weights/yolov8n.onnx  (see docs/PLAN.md section 6)
    * Ground truth:     fill the manual_count column in tests/ground_truth.csv

    DO NOT fabricate manual_count. Open each image and count visible people.

Usage:
    .venv/bin/python scripts/benchmark.py
    .venv/bin/python scripts/benchmark.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "backend" / "samples"
GROUND_TRUTH = REPO_ROOT / "tests" / "ground_truth.csv"
RESULTS_CSV = REPO_ROOT / "tests" / "benchmark_results.csv"

BOUNDARY = "----detectoBenchmarkBoundary"


def post_detect(base_url: str, image_path: Path, timeout: int = 60) -> dict:
    """POST one image as multipart/form-data and return the parsed JSON."""
    body = bytearray()
    body += f"--{BOUNDARY}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
    ).encode()
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += image_path.read_bytes()
    body += f"\r\n--{BOUNDARY}--\r\n".encode()

    request = urllib.request.Request(
        f"{base_url}/detect",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_ground_truth() -> dict[str, dict]:
    if not GROUND_TRUTH.exists():
        sys.exit(f"Ground truth file not found: {GROUND_TRUTH}")
    with GROUND_TRUTH.open(newline="") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def check_health(base_url: str) -> None:
    """Fail fast with a clear message if the server or model is not ready."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
            health = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        sys.exit(f"Cannot reach backend at {base_url}: {exc}\nStart it with: uvicorn backend.main:app --port 8000")
    if not health.get("model_loaded"):
        sys.exit(
            "Backend is running but the model is NOT loaded.\n"
            "Place yolov8n.onnx at backend/weights/yolov8n.onnx (see docs/PLAN.md section 6)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark detecto")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    ground_truth = load_ground_truth()

    images = sorted(SAMPLES_DIR.glob("*.jpg")) + sorted(SAMPLES_DIR.glob("*.png"))
    if not images:
        sys.exit(f"No sample images found in {SAMPLES_DIR}")

    # Refuse to run on unverified ground truth.
    missing = [
        img.name
        for img in images
        if img.name not in ground_truth
        or not ground_truth[img.name]["manual_count"].strip()
    ]
    if missing:
        sys.exit(
            "manual_count is empty for these images:\n  "
            + "\n  ".join(missing)
            + "\n\nOpen each image and count visible people, then fill the "
            "manual_count column in tests/ground_truth.csv."
        )

    check_health(args.base_url)

    rows = []
    all_confidences = []
    errors = 0

    for image_path in images:
        truth_row = ground_truth[image_path.name]
        truth = int(truth_row["manual_count"])
        category = truth_row.get("category", "")

        try:
            result = post_detect(args.base_url, image_path)
        except urllib.error.HTTPError as exc:
            errors += 1
            detail = exc.read().decode("utf-8", "replace")
            print(f"ERROR {image_path.name}: HTTP {exc.code} {detail}")
            rows.append(
                {
                    "filename": image_path.name,
                    "category": category,
                    "detected": "",
                    "truth": truth,
                    "avg_conf": "",
                    "inference_ms": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "status": "ERROR",
                }
            )
            continue

        detected = int(result["person_count"])
        confidences = [box["confidence"] for box in result.get("boxes", [])]
        all_confidences.extend(confidences)

        tp = min(detected, truth)
        fp = max(0, detected - truth)
        fn = max(0, truth - detected)

        rows.append(
            {
                "filename": image_path.name,
                "category": category,
                "detected": detected,
                "truth": truth,
                "avg_conf": round(result["avg_confidence"], 3)
                if result["avg_confidence"] is not None
                else "",
                "inference_ms": round(result["inference_time_ms"], 1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "status": "ok",
            }
        )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    total_truth = sum(r["truth"] for r in ok_rows)
    total_detected = sum(r["detected"] for r in ok_rows)
    total_tp = sum(r["tp"] for r in ok_rows)
    total_fp = sum(r["fp"] for r in ok_rows)

    accuracy = (total_tp / total_truth * 100) if total_truth else 0.0
    fpr = (total_fp / total_detected * 100) if total_detected else 0.0
    mean_inference = mean(r["inference_ms"] for r in ok_rows) if ok_rows else 0.0
    mean_confidence = mean(all_confidences) if all_confidences else None
    reliability = (len(ok_rows) / len(images) * 100) if images else 0.0

    # --- per-image markdown table ---
    print("\n### Per-image results\n")
    print("| Image | Category | Detected | Truth | Avg conf | Inference (ms) | TP | FP | FN |")
    print("|-------|----------|---------:|------:|---------:|---------------:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['filename']} | {r['category']} | {r['detected']} | {r['truth']} | "
            f"{r['avg_conf']} | {r['inference_ms']} | {r['tp']} | {r['fp']} | {r['fn']} |"
        )

    # --- summary markdown table ---
    print("\n### Summary\n")
    print("| Metric | Value | Target |")
    print("|--------|-------|--------|")
    print(f"| Detection accuracy | {accuracy:.1f} % | >= 85 % |")
    print(f"| False-positive rate | {fpr:.1f} % | <= 10 % |")
    print(f"| Mean inference time | {mean_inference:.0f} ms | <= 1500 ms |")
    conf_str = f"{mean_confidence:.3f}" if mean_confidence is not None else "n/a"
    print(f"| Mean confidence | {conf_str} | >= 0.700 |")
    print(f"| Reliability | {reliability:.0f} % | 100 % |")
    print(f"\n({len(ok_rows)}/{len(images)} images processed; {errors} error(s))")

    with RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename", "category", "detected", "truth", "avg_conf",
                "inference_ms", "tp", "fp", "fn", "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-image results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
