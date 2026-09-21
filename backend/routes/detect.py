"""
detecto.routes.detect
=====================

POST /detect -- accepts an image as multipart/form-data OR as a base64 JSON
body, runs the ONNX YOLOv8n model, and returns person detections in
ORIGINAL image pixel coordinates.

Why manual request parsing?
    FastAPI cannot bind a multipart UploadFile and a JSON body to the same
    route; the two content types are mutually exclusive. The brief requires a
    single endpoint that accepts either, so we dispatch on Content-Type here.
    This also lets us return precise 4xx errors instead of framework 422s.

Error contract:
    Every rejection returns the shared envelope
    {"error": {"code", "message", "detail"}}, produced by the handler in
    main.py. This module raises HTTPException whose ``detail`` is that inner
    dict. Inference failures are genuine 5xx; bad input is always 4xx.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, ValidationError

from backend.models.record import (
    BoundingBox,
    DetectionResponse,
    insert_detection,
)
from backend.utils.preprocessing import (
    ImageDecodeError,
    draw_boxes,
    encode_png_base64,
    postprocess,
    preprocess_image,
)

logger = logging.getLogger("detecto.detect")

router = APIRouter(tags=["detection"])

# Authoritative format signatures. We trust these over client-supplied MIME.
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class DetectBase64Request(BaseModel):
    """JSON body alternative to multipart upload."""

    image_base64: str = Field(..., description="Base64-encoded JPEG or PNG bytes")
    filename: Optional[str] = Field(default=None, description="Optional name")
    mime_type: Optional[str] = Field(default=None, description="Optional client hint")


def _error(status_code: int, code: str, message: str, detail: str | None = None) -> HTTPException:
    """Build an HTTPException carrying our structured error envelope."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "detail": detail},
    )


def _sniff_mime(data: bytes) -> str | None:
    """Return the real MIME type from magic bytes, or None if unrecognised."""
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    return None


async def _read_upload(request: Request, settings: Any) -> bytes:
    """Extract raw image bytes from either a multipart or JSON request.

    Enforces, in order: known content type -> non-empty -> size cap ->
    allowed image format. Each failure maps to a specific 4xx.
    """
    content_type = request.headers.get("content-type", "")
    max_bytes = int(settings.max_upload_mb) * 1024 * 1024

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file") or form.get("image")
        if upload is None:
            raise _error(400, "missing_file", "No 'file' field found in the form data")
        # UploadFile.read() is async; the object also has .filename/.content_type.
        data = await upload.read()
        source = getattr(upload, "filename", None) or "upload"
    elif content_type.startswith("application/json"):
        try:
            body = await request.json()
        except Exception as exc:  # malformed JSON
            raise _error(400, "invalid_json", "Request body is not valid JSON") from exc
        try:
            parsed = DetectBase64Request(**body)
        except ValidationError as exc:
            raise _error(
                400, "invalid_body", "Missing or invalid 'image_base64' field", str(exc)
            ) from exc
        try:
            data = base64.b64decode(parsed.image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise _error(400, "invalid_base64", "image_base64 is not valid base64") from exc
        source = parsed.filename or "base64-upload"
    else:
        raise _error(
            415,
            "unsupported_content_type",
            "Content-Type must be multipart/form-data or application/json",
            f"Received: {content_type or '(none)'}",
        )

    # --- validation, cheapest checks first ---
    if not data:
        raise _error(400, "empty_file", "The uploaded file is empty")

    if len(data) > max_bytes:
        raise _error(
            413,
            "file_too_large",
            f"File exceeds the {settings.max_upload_mb} MB limit",
            f"Received {len(data)} bytes; limit {max_bytes}",
        )

    sniffed = _sniff_mime(data)
    if sniffed is None:
        raise _error(
            415,
            "unsupported_format",
            "File is not a valid JPEG or PNG image",
            f"source={source}",
        )
    if sniffed not in settings.allowed_mime_types:
        raise _error(
            415,
            "unsupported_format",
            f"{sniffed} is not in the allowed types",
            f"allowed={sorted(settings.allowed_mime_types)}",
        )

    return data


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request,
    annotated: bool = Query(
        default=False,
        description="If true, include a base64 PNG of the image with boxes drawn.",
    ),
) -> DetectionResponse:
    """Detect people in an uploaded image and record the result."""
    state = request.app.state
    session = getattr(state, "session", None)
    if session is None:
        raise _error(503, "model_unavailable", "Model is not loaded yet")

    settings = state.settings
    input_name = state.model_input_name

    image_bytes = await _read_upload(request, settings)

    # --- preprocess (decode + letterbox + tensor) ---
    total_start = time.perf_counter()
    try:
        tensor, meta = preprocess_image(
            image_bytes,
            input_size=settings.img_size,
            enhance_contrast=settings.enhance_contrast,
        )
    except ImageDecodeError as exc:
        # Valid magic bytes but corrupt body -> still a client error, not 500.
        raise _error(400, "corrupt_image", "Image could not be decoded", str(exc)) from exc

    # --- inference (timed on its own, per the brief) ---
    inference_start = time.perf_counter()
    try:
        outputs = session.run(None, {input_name: tensor})
    except Exception as exc:  # genuine server-side failure
        logger.exception("Inference failed")
        raise _error(500, "inference_failed", "Model inference failed", str(exc)) from exc
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    # --- postprocess: filter class 0, NMS, inverse-transform to original px ---
    boxes, scores = postprocess(
        outputs[0],
        meta,
        conf_threshold=settings.conf_threshold,
        iou_threshold=settings.nms_iou_threshold,
        person_class_id=settings.person_class_id,
    )

    person_count = int(boxes.shape[0])
    avg_confidence = float(np.mean(scores)) if person_count > 0 else None

    bbox_models = [
        BoundingBox(
            x1=float(b[0]),
            y1=float(b[1]),
            x2=float(b[2]),
            y2=float(b[3]),
            confidence=float(s),
        )
        for b, s in zip(boxes, scores)
    ]

    # --- optional annotation ---
    annotated_b64: str | None = None
    if annotated and person_count > 0:
        from backend.utils.preprocessing import decode_image

        original = decode_image(image_bytes)
        annotated_b64 = encode_png_base64(draw_boxes(original, boxes, scores))

    total_ms = (time.perf_counter() - total_start) * 1000.0

    # --- persist only after a fully successful detection ---
    insert_detection(
        settings.db_path,
        person_count=person_count,
        avg_confidence=avg_confidence,
        inference_time_ms=inference_ms,
    )

    # Structured, greppable log line for the benchmark run.
    logger.info(
        "detect inference_ms=%.1f total_ms=%.1f image=%dx%d persons=%d avg_conf=%s",
        inference_ms,
        total_ms,
        meta["orig_w"],
        meta["orig_h"],
        person_count,
        f"{avg_confidence:.3f}" if avg_confidence is not None else "n/a",
    )

    return DetectionResponse(
        person_count=person_count,
        boxes=bbox_models,
        avg_confidence=avg_confidence,
        inference_time_ms=round(inference_ms, 3),
        total_time_ms=round(total_ms, 3),
        image_width=int(meta["orig_w"]),
        image_height=int(meta["orig_h"]),
        annotated_image_base64=annotated_b64,
    )
