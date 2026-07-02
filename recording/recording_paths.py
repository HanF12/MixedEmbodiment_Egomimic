"""Shared output root for a recording session (set via RECORDING_DATA_ROOT)."""

from __future__ import annotations

import os


def recording_root() -> str:
    return os.environ.get("RECORDING_DATA_ROOT", "").strip()


def under_recording(*parts: str) -> str:
    root = recording_root()
    if root:
        return os.path.join(root, *parts)
    return os.path.join(*parts)
