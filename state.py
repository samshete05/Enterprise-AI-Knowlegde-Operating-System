from __future__ import annotations

from typing import Any


ACTIVE_PIPELINE: dict[str, Any] | None = None


def get_active_pipeline() -> dict[str, Any] | None:
    return ACTIVE_PIPELINE


def set_active_pipeline(pipeline: dict[str, Any]) -> None:
    global ACTIVE_PIPELINE
    ACTIVE_PIPELINE = pipeline