"""Shared repository paths for CLI scripts under scripts/."""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
ASSETS_DIR = REPO_ROOT / "assets"
BODY_MODELS_DIR = ASSETS_DIR / "body_models"
RETARGETING_DATA_DIR = REPO_ROOT / "retargeting_data"


def ensure_repo_on_path() -> Path:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT
