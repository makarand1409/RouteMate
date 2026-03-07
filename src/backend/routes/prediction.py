"""
Prediction endpoints - get ML model predictions for a given observation.
"""
import sys
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException

_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from backend.utils.model_loader import load_ppo_model
from backend.models import PredictionRequest, PredictionResult

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResult)
async def predict(req: PredictionRequest):
    """
    Predict the best vehicle assignment for a given observation vector.

    The observation should have length = 4 + (num_vehicles * 5):
    - [0:4] = request pickup_x, pickup_y, dropoff_x, dropoff_y
    - Per vehicle: location_x, location_y, occupancy, queue_length, distance_to_pickup
    """
    try:
        model = load_ppo_model(req.model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    obs = np.array(req.observation, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    action_int = int(action)

    return PredictionResult(
        action=action_int,
        vehicle_id=action_int,
        model_name=req.model_name,
    )
