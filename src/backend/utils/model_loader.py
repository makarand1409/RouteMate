"""
RouteMATE Backend - Utility to load trained ML models.
"""
import os
from pathlib import Path
from typing import Optional

# Resolve project root (routemate/)
_BACKEND_DIR = Path(__file__).resolve().parent.parent  # src/
_PROJECT_ROOT = _BACKEND_DIR.parent  # routemate/
MODELS_DIR = _PROJECT_ROOT / "outputs" / "models"


def get_available_models():
    """Return a list of available trained model names (without .zip)."""
    if not MODELS_DIR.exists():
        return []
    return sorted(
        p.stem for p in MODELS_DIR.glob("*.zip")
    )


def get_model_path(model_name: str = "ppo_routemate_final") -> Optional[Path]:
    """Return the full path to a model file, or None if not found."""
    path = MODELS_DIR / f"{model_name}.zip"
    if not path.exists():
        # Also try without .zip in case caller already appended it
        path = MODELS_DIR / model_name
        if not path.exists():
            return None
    return path


def load_ppo_model(model_name: str = "ppo_routemate_final"):
    """Load and return a stable-baselines3 PPO model."""
    from stable_baselines3 import PPO

    path = get_model_path(model_name)
    if path is None:
        raise FileNotFoundError(
            f"Model '{model_name}' not found in {MODELS_DIR}. "
            f"Available: {get_available_models()}"
        )
    # SB3 expects path without .zip extension
    load_path = str(path).replace(".zip", "") if str(path).endswith(".zip") else str(path)
    return PPO.load(load_path)
