"""
Export YOLOv8n to ONNX for detecto.

WHY THIS IS A SEPARATE SCRIPT
    The runtime stack deliberately has NO torch (see docs/PLAN.md D1): it is
    ~2.5 GB and does not fit the target machine. But *exporting* the model
    requires ultralytics + torch. So the export is a one-time build step,
    performed on a machine that can afford torch, and the resulting ~12 MB
    yolov8n.onnx is copied into backend/weights/.

RECOMMENDED: GOOGLE COLAB (free, no local install)
    1. Open https://colab.research.google.com and create a new notebook.
    2. Cell 1:
           !pip install -q ultralytics
           !python export_model.py
       ...or, if you did not upload this file, just run:
           from ultralytics import YOLO
           YOLO("yolov8n.pt").export(format="onnx", opset=12, imgsz=640)
    3. Download the generated yolov8n.onnx from the Colab file browser.
    4. Place it at:  backend/weights/yolov8n.onnx

LOCAL ALTERNATIVE (only if you have disk + torch):
    pip install ultralytics
    python scripts/export_model.py --imgsz 640

MODEL I/O CONTRACT (must match backend/utils/preprocessing.py)
    input : float32 [1, 3, IMG_SIZE, IMG_SIZE], RGB, 0-1, NCHW
    output: float32 [1, 84, 8400] = 4 box (cx, cy, w, h in input px)
            + 80 COCO class scores. NMS is NOT included (we do our own).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX for detecto")
    parser.add_argument("--model", default="yolov8n.pt", help="Source weights (auto-downloaded)")
    parser.add_argument("--imgsz", type=int, default=640, help="Export input size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--dest",
        default="backend/weights/yolov8n.onnx",
        help="Where to copy the exported model",
    )
    args = parser.parse_args()

    # Imported lazily so this file can be read/edited without torch installed.
    from ultralytics import YOLO

    model = YOLO(args.model)

    # nms=False keeps the raw [1, 84, 8400] output so we control NMS ourselves
    # (needed to own the IoU threshold and the coordinate transform).
    exported = model.export(format="onnx", opset=args.opset, imgsz=args.imgsz, nms=False)

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, dest)
    print(f"Exported model copied to: {dest.resolve()}")
    print("Place this file at backend/weights/yolov8n.onnx and restart the backend.")


if __name__ == "__main__":
    main()
