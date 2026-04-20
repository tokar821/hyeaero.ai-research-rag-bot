"""Structured timing logs for Ask Consultant (per-request stages and data sources)."""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger("rag.consultant_progress")


def consultant_progress_enabled() -> bool:
    v = (os.getenv("CONSULTANT_PROGRESS_LOG") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def consultant_progress_verbose() -> bool:
    return (os.getenv("CONSULTANT_PROGRESS_VERBOSE") or "").strip().lower() in ("1", "true", "yes")


def _default_max_len() -> int:
    return 360 if consultant_progress_verbose() else 112


def _fmt_val(v: Any, max_len: int | None = None) -> str:
    if v is None:
        return "null"
    cap = max_len if max_len is not None else _default_max_len()
    s = str(v)
    if len(s) > cap:
        return s[: cap - 1] + "…"
    return s


def log_detail(request_id: str, title: str, body: str) -> None:
    """Multi-line blob (full RAG query list, etc.) when verbose is on."""
    if not consultant_progress_enabled() or not consultant_progress_verbose():
        return
    for line in (body or "").splitlines():
        logger.info("[consultant %s] detail | %s | %s", request_id, title, line[:2000])


def new_progress_logger(request_id: Optional[str] = None) -> Optional["ConsultantProgressLogger"]:
    if not consultant_progress_enabled():
        return None
    return ConsultantProgressLogger(request_id=request_id)


class ConsultantProgressLogger:
    """Wall-clock stages: cumulative ms from request start and Δms from previous step."""

    def __init__(self, request_id: Optional[str] = None) -> None:
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self._t0 = time.perf_counter()
        self._last = self._t0

    def detail(self, title: str, body: str) -> None:
        log_detail(self.request_id, title, body)

    def step(self, stage: str, **kwargs: Any) -> None:
        now = time.perf_counter()
        rel_ms = int((now - self._t0) * 1000)
        delta_ms = int((now - self._last) * 1000)
        self._last = now
        if kwargs:
            parts = [f"{k}={_fmt_val(v)}" for k, v in sorted(kwargs.items())]
            suffix = " | " + " ".join(parts)
        else:
            suffix = ""
        logger.info(
            "[consultant %s] +%dms d%dms | %s%s",
            self.request_id,
            rel_ms,
            delta_ms,
            stage,
            suffix,
        )
