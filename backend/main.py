"""
detecto.main
============

Application composition root.

Responsibilities:
    * Load configuration from the environment (and a repo-root .env).
    * Load the ONNX model ONCE at startup via a lifespan handler.
    * Initialise the SQLite schema.
    * Mount the detect and history routers.
    * Configure CORS from an explicit allow-list (never "*").
    * Normalise every error into a single {"error": {...}} envelope.
    * Expose a cheap /health probe that does not run inference.

Run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.models.record import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    init_db,
)
from backend.routes import detect as detect_route
from backend.routes import history as history_route

# Paths are anchored to this file so the app behaves the same regardless of the
# working directory uvicorn is launched from.
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent

logger = logging.getLogger("detecto")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve(path_value: str) -> str:
    """Resolve a possibly-relative path against the backend directory."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((BACKEND_DIR / path).resolve())


@dataclass
class Settings:
    """Runtime configuration. Every field has a safe default so the app can
    boot with no .env present."""

    model_path: str = "weights/yolov8n.onnx"
    img_size: int = 640
    conf_threshold: float = 0.5
    nms_iou_threshold: float = 0.45
    person_class_id: int = 0
    max_upload_mb: int = 10
    allowed_mime_types: set[str] = field(default_factory=lambda: {"image/jpeg", "image/png"})
    db_path: str = "data/detecto.db"
    history_default_limit: int = 100
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    ort_num_threads: int = 4
    enhance_contrast: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, resolving file paths."""
        return cls(
            model_path=_resolve(os.getenv("MODEL_PATH", "weights/yolov8n.onnx")),
            img_size=int(os.getenv("IMG_SIZE", "640")),
            conf_threshold=float(os.getenv("CONF_THRESHOLD", "0.5")),
            nms_iou_threshold=float(os.getenv("NMS_IOU_THRESHOLD", "0.45")),
            person_class_id=int(os.getenv("PERSON_CLASS_ID", "0")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "10")),
            allowed_mime_types=set(
                _env_list("ALLOWED_MIME_TYPES", ["image/jpeg", "image/png"])
            ),
            db_path=_resolve(os.getenv("DB_PATH", "data/detecto.db")),
            history_default_limit=int(os.getenv("HISTORY_DEFAULT_LIMIT", "100")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            cors_origins=_env_list("CORS_ORIGINS", ["http://localhost:5173"]),
            ort_num_threads=int(os.getenv("ORT_NUM_THREADS", "4")),
            enhance_contrast=_env_bool("ENHANCE_CONTRAST", False),
        )


# Settings are built at import time, not inside the lifespan, because the CORS
# middleware is configured when the app object is created -- before lifespan
# runs. Building them here guarantees CORS_ORIGINS from .env actually applies.
load_dotenv(REPO_ROOT / ".env")
settings = Settings.from_env()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(settings: Settings) -> tuple[Any, str | None]:
    """Create the ONNX InferenceSession. Returns (session, input_name).

    On a missing/corrupt model we do NOT crash: we return (None, None) so the
    app still serves /health, /history, and /reset. /detect then answers 503
    with a clear message instead of taking the whole server down.
    """
    model_path = Path(settings.model_path)
    if not model_path.exists():
        logger.error(
            "Model file not found at %s -- /detect will return 503. "
            "See docs/PLAN.md section 6 for how to obtain yolov8n.onnx.",
            model_path,
        )
        return None, None

    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.ort_num_threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    except Exception:
        logger.exception("Failed to create ONNX InferenceSession from %s", model_path)
        return None, None

    input_name = session.get_inputs()[0].name
    return session, input_name


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterator[None]:
    """Startup/shutdown. The model is loaded exactly once, here."""
    app.state.settings = settings

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logger.info("Starting detecto")
    logger.info("CORS origins: %s", settings.cors_origins)
    logger.info("Database: %s", settings.db_path)
    init_db(settings.db_path)

    load_start = time.perf_counter()
    session, input_name = _load_model(settings)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    app.state.session = session
    app.state.model_input_name = input_name
    if session is not None:
        logger.info(
            "Model loaded in %.1f ms (input=%s, threads=%d, imgsz=%d)",
            load_ms,
            input_name,
            settings.ort_num_threads,
            settings.img_size,
        )
    else:
        logger.warning("Running WITHOUT a model -- /detect will return 503")

    yield  # application serves requests here

    app.state.session = None
    logger.info("Shutting down detecto")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="detecto",
    version="0.1.0",
    description="Real-time person detection and counting (YOLOv8n via ONNX Runtime).",
    lifespan=lifespan,
)

# CORS: explicit allow-list from env. allow_credentials with a specific origin
# list is valid; "*" is deliberately never used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(detect_route.router)
app.include_router(history_route.router)


# ---------------------------------------------------------------------------
# Error handling -- one envelope for every failure
# ---------------------------------------------------------------------------


def _envelope(status_code: int, error: ErrorDetail) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorResponse(error=error).model_dump())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Our routes raise HTTPException with a structured dict detail; normalise
    both those and plain framework HTTPExceptions into the envelope."""
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        error = ErrorDetail(
            code=str(detail.get("code")),
            message=str(detail.get("message", "")),
            detail=detail.get("detail"),
        )
    else:
        error = ErrorDetail(code="http_error", message=str(detail))
    return _envelope(exc.status_code, error)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Wrap FastAPI's 422 body in the same envelope."""
    error = ErrorDetail(
        code="validation_error",
        message="Request validation failed",
        detail=str(exc.errors()),
    )
    return _envelope(422, error)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler so unexpected failures still return the envelope."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    error = ErrorDetail(code="internal_error", message="An unexpected error occurred")
    return _envelope(500, error)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    """Liveness probe. Does not run inference."""
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.session is not None,
        model_path=settings.model_path,
    )
