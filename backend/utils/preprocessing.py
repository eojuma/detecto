"""
detecto.utils.preprocessing
===========================

The image geometry + tensor pipeline, and the single source of truth for
coordinate mapping in the whole project.

Pipeline
--------
    raw bytes
      -> decode_image()          # bytes  -> BGR ndarray (or raise)
      -> letterbox()             # BGR    -> square BGR + transform metadata
      -> to_tensor()             # BGR    -> float32 NCHW RGB in [0,1]
      -> [model inference happens elsewhere]
      -> postprocess()           # raw model output -> boxes in ORIGINAL pixels
      -> draw_boxes() / encode_png_base64()   # optional annotated output

The coordinate-transform problem
--------------------------------
The model only ever sees a letterboxed ``IMG_SIZE x IMG_SIZE`` square. A box
returned in that square's coordinates is meaningless to the frontend, which
displays the ORIGINAL image. So we invert the letterbox here, before the
response leaves the API. The frontend then only has to scale original-image
pixels to its display size -- a pure ratio with no padding involved.

    forward:  x_letterboxed = x_original * scale_x + pad_w
    inverse:  x_original    = (x_letterboxed - pad_w) / scale_x

We store the *actual* per-axis scale (new_w / orig_w) rather than the
pre-rounding ratio, so integer rounding of the resized dimensions cannot
accumulate into a visible drift.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

# YOLO's canonical letterbox fill. Mid-grey is neutral, so the padding does not
# look like a strong edge to the network.
_LETTERBOX_FILL = 114

__all__ = [
    "ImageDecodeError",
    "decode_image",
    "letterbox",
    "apply_clahe",
    "to_tensor",
    "preprocess_image",
    "non_max_suppression",
    "postprocess",
    "draw_boxes",
    "encode_png_base64",
]


class ImageDecodeError(ValueError):
    """Raised when bytes cannot be decoded as an image.

    A ValueError subclass so callers can treat it as a client error (4xx),
    never a server error (5xx).
    """


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR uint8 ndarray.

    Uses OpenCV, which handles JPEG and PNG. Raises ImageDecodeError on
    empty, truncated, or non-image input so the route can answer 400 rather
    than letting a None propagate into a 500.

    Note: cv2.IMREAD_COLOR drops any alpha channel and converts grayscale to
    BGR, so the returned array is always HxWx3.
    """
    if not image_bytes:
        raise ImageDecodeError("Empty image payload")

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageDecodeError("Bytes could not be decoded as a JPEG or PNG image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageDecodeError(f"Unexpected image shape: {image.shape}")

    return image


# ---------------------------------------------------------------------------
# Letterbox (forward transform)
# ---------------------------------------------------------------------------


def letterbox(image: np.ndarray, input_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Resize preserving aspect ratio, then pad to ``input_size x input_size``.

    Returns the square image and a metadata dict containing everything needed
    to invert the transform later. That dict is the contract between the
    forward and inverse steps -- do not compute padding anywhere else.
    """
    orig_h, orig_w = image.shape[:2]
    if orig_w == 0 or orig_h == 0:
        raise ImageDecodeError("Image has zero width or height")

    # Uniform scale that fits the longest side inside the square.
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Clamp: rounding can, in edge cases, push a dimension one pixel over.
    new_w = max(1, min(new_w, input_size))
    new_h = max(1, min(new_h, input_size))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Integer padding. Left/top and right/bottom may differ by one pixel when
    # the gap is odd; the inverse transform uses these exact integers, so the
    # asymmetry is harmless.
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2

    canvas = np.full((input_size, input_size, 3), _LETTERBOX_FILL, dtype=np.uint8)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    meta: dict[str, Any] = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "input_size": input_size,
        "new_w": new_w,
        "new_h": new_h,
        "pad_w": pad_w,
        "pad_h": pad_h,
        # Actual post-rounding scale. This is what makes the inverse exact.
        "scale_x": new_w / orig_w,
        "scale_y": new_h / orig_h,
    }
    return canvas, meta


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Optional contrast-limited adaptive histogram equalisation.

    Off by default. Applied to the L channel of LAB so colour is untouched.
    Documented honestly: this helps low-contrast / poor-lighting frames but
    can hurt already well-exposed ones, so it is a tunable, not a default.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    merged = cv2.merge((l_eq, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------


def to_tensor(letterboxed_bgr: np.ndarray) -> np.ndarray:
    """Convert a square BGR uint8 image to a float32 NCHW RGB tensor in [0,1].

    The model's ONNX graph expects exactly this layout: batch=1, channels=3,
    RGB order, values 0-1. Getting the channel order wrong (BGR vs RGB) does
    not crash -- it just quietly degrades accuracy -- so it is called out here.
    """
    rgb = cv2.cvtColor(letterboxed_bgr, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
    tensor = np.expand_dims(tensor, axis=0)   # CHW -> NCHW
    return np.ascontiguousarray(tensor)


def preprocess_image(
    image_bytes: bytes, input_size: int, enhance_contrast: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full forward pipeline: bytes -> (NCHW tensor, transform metadata)."""
    image = decode_image(image_bytes)
    if enhance_contrast:
        image = apply_clahe(image)
    square, meta = letterbox(image, input_size)
    return to_tensor(square), meta


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Greedy NMS over xyxy boxes. Returns kept indices.

    Why NMS exists: a detector emits many overlapping boxes for the same
    person (one per nearby anchor). Without suppression we would count the
    same person several times. We keep the highest-scoring box and discard
    any box whose IoU with it exceeds the threshold, repeating until done.

    Implemented in numpy (rather than cv2.dnn.NMSBoxes) so the logic is
    explicit and testable.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]  # highest score first
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        # Intersection rectangle of box i against all remaining boxes.
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-6)

        # Surviving boxes are those with low overlap; keep their positions.
        survivors = np.where(iou <= iou_threshold)[0]
        order = rest[survivors]

    return np.asarray(keep, dtype=int)


# ---------------------------------------------------------------------------
# Postprocess (inverse transform)
# ---------------------------------------------------------------------------


def postprocess(
    model_output: np.ndarray,
    meta: dict[str, Any],
    *,
    conf_threshold: float,
    iou_threshold: float,
    person_class_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn raw YOLOv8 output into boxes in ORIGINAL image pixel coordinates.

    Expected output shape: (1, 4 + num_classes, num_anchors), i.e. (1, 84, 8400)
    for COCO at 640. The first 4 rows are cx, cy, w, h in *letterboxed* pixels;
    the remaining rows are per-class probabilities.

    Returns (boxes_xyxy, scores) both in original-image pixels, de-duplicated
    by NMS and clamped to the image bounds.
    """
    pred = np.squeeze(model_output, axis=0)  # (4 + C, A)

    # Some exports emit (A, 4 + C) instead; normalise to (A, 4 + C).
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    if person_class_id >= class_scores.shape[1]:
        raise ValueError(
            f"person_class_id {person_class_id} out of range "
            f"for {class_scores.shape[1]} classes"
        )

    person_scores = class_scores[:, person_class_id]

    # Confidence gate before NMS: cheaper and avoids suppressing a real box
    # with a spurious low-score one.
    mask = person_scores >= conf_threshold
    boxes_xywh = boxes_xywh[mask]
    person_scores = person_scores[mask]

    if boxes_xywh.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # cx, cy, w, h -> x1, y1, x2, y2 (still letterboxed coordinates).
    cx, cy, bw, bh = (
        boxes_xywh[:, 0],
        boxes_xywh[:, 1],
        boxes_xywh[:, 2],
        boxes_xywh[:, 3],
    )
    boxes_xyxy = np.stack(
        [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=1
    )

    keep = non_max_suppression(boxes_xyxy, person_scores, iou_threshold)
    boxes_xyxy = boxes_xyxy[keep]
    person_scores = person_scores[keep]

    # --- The inverse transform: letterboxed pixels -> original pixels ---
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - meta["pad_w"]) / meta["scale_x"]
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - meta["pad_h"]) / meta["scale_y"]

    # Clamp to the original image so no box can point outside it.
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, meta["orig_w"])
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, meta["orig_h"])

    return boxes_xyxy.astype(np.float32), person_scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Annotation output
# ---------------------------------------------------------------------------


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    color: tuple[int, int, int] = (0, 200, 0),
) -> np.ndarray:
    """Draw boxes + confidence labels on a copy of the ORIGINAL image.

    ``boxes`` must already be in original-image pixels (postprocess output).
    Line thickness and font scale adapt to image size so labels stay legible
    on both small and very large images. Each label sits on a filled plate so
    it remains readable over busy backgrounds.
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]

    thickness = max(1, int(round(min(width, height) / 400)))
    font_scale = max(0.4, min(width, height) / 900.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(annotated, p1, p2, color, thickness)

        label = f"{score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Prefer the label above the box; fall back to inside if it would clip.
        plate_top = p1[1] - text_h - baseline - 4
        if plate_top < 0:
            plate_top = p1[1] + 2
        plate_bottom = plate_top + text_h + baseline + 4

        cv2.rectangle(
            annotated,
            (p1[0], plate_top),
            (p1[0] + text_w + 6, plate_bottom),
            color,
            thickness=-1,  # filled
        )
        cv2.putText(
            annotated,
            label,
            (p1[0] + 3, plate_bottom - baseline - 1),
            font,
            font_scale,
            (0, 0, 0),  # black text on the coloured plate
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def encode_png_base64(image: np.ndarray) -> str:
    """Encode a BGR image as a base64 PNG string for the JSON response."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode annotated image as PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
