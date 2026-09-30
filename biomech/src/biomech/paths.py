"""Filesystem locations, resolved relative to this file.

Owns: where the package's own assets live.
Does NOT own: anything the user passes in, which arrives as an explicit Path.

Derived from `__file__` rather than the working directory, so the application
behaves the same whether it is run from the repository root, from an installed
package, or from anywhere else.
"""

from __future__ import annotations

from pathlib import Path

# src/biomech/paths.py -> src/biomech -> src -> biomech
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PACKAGE_ROOT / "models"

MODEL_BASE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
)


def model_path(variant: str) -> Path:
    """Where a pose model file is expected to be."""
    return MODELS_DIR / f"pose_landmarker_{variant}.task"


def model_download_url(variant: str) -> str:
    """Where to obtain it, for the error message when it is missing."""
    return (
        f"{MODEL_BASE_URL}/pose_landmarker_{variant}/float16/latest/"
        f"pose_landmarker_{variant}.task"
    )
