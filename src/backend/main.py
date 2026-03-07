"""
RouteMATE FastAPI Backend — Phase 4

Serves the trained PPO model, exposes simulation endpoints, and provides
performance metrics via a REST API.

Run with:
    cd routemate
    uvicorn src.backend.main:app --reload
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the project src/ directory is importable
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from backend.models import HealthResponse
from backend.routes.simulation import router as simulation_router
from backend.routes.prediction import router as prediction_router
from backend.routes.metrics import router as metrics_router

app = FastAPI(
    title="RouteMATE API",
    description=(
        "ML-based dynamic ride-sharing optimisation backend. "
        "Supports greedy, random, and PPO-trained policies."
    ),
    version="0.4.0",
)

# ── CORS — allow the React frontend on localhost:3000 ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register route modules ────────────────────────────────────────
app.include_router(simulation_router)
app.include_router(prediction_router)
app.include_router(metrics_router)


@app.get("/", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health-check / root endpoint."""
    return HealthResponse(status="ok", version="0.4.0")
