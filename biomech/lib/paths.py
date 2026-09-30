"""Project paths, resolved relative to this file.

Absolute paths like "D:/zeuron/..." were embedded in 28 scripts. They break on any
other machine, which matters when the work is going to be handed to someone else.
"""
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PKG_ROOT / "fixtures"
MODELS = PKG_ROOT / "models"
DOCS = PKG_ROOT / "docs"

_HOME = {
    "session.jsonl": "sessions",
    "exposure_session.jsonl": "sessions",
    "accuracy_session.jsonl": "accuracy",
    "accuracy_multi.jsonl": "accuracy",
    "anchored.jsonl": "accuracy",
    "robustness_session.jsonl": "robustness",
    "framing.json": "robustness",
    "frame_raw.png": "raw",
    "protractor_A4.pdf": "reference",
    "protractor_A4.png": "reference",
}


def fixture(name):
    """Absolute path to a fixture, creating its directory if needed."""
    sub = _HOME.get(name)
    if sub is None:
        sub = "sessions" if (name.startswith(("sess_", "exp_")) or name.endswith(".mp4")) else "misc"
    p = FIXTURES / sub / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def model_path(variant="full"):
    """Absolute path to a pose_landmarker task file."""
    return str(MODELS / f"pose_landmarker_{variant}.task")
