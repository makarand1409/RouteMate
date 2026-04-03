"""
Pydantic models for RouteMATE API request/response schemas.
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Request Bodies ──────────────────────────────────────────────

class SimulationRequest(BaseModel):
    policy: Literal["greedy", "random", "ml"] = Field(
        default="greedy",
        description="Matching policy to use: greedy (nearest vehicle), random, or ml (PPO agent).",
    )
    num_steps: int = Field(
        default=100, ge=1, le=1000,
        description="Number of simulation time steps to run.",
    )
    num_vehicles: int = Field(
        default=5, ge=1, le=20,
        description="Number of vehicles in the fleet.",
    )
    vehicle_capacity: int = Field(
        default=4, ge=1, le=8,
        description="Max passengers per vehicle.",
    )
    request_rate: float = Field(
        default=2.0, gt=0, le=10.0,
        description="Avg ride requests per time step (Poisson λ).",
    )
    model_name: Optional[str] = Field(
        default="ppo_routemate_final",
        description="Name of the trained model to use when policy='ml'.",
    )


class PredictionRequest(BaseModel):
    observation: List[float] = Field(
        ...,
        description="Observation vector (length = 4 + num_vehicles * 5).",
    )
    model_name: str = Field(
        default="ppo_routemate_final",
        description="Name of the trained model to use for prediction.",
    )


# ── Response Bodies ─────────────────────────────────────────────

class VehicleState(BaseModel):
    vehicle_id: int
    location: List[int]
    occupancy: int
    queue_length: int
    total_distance: float
    total_served: int


class SimulationResult(BaseModel):
    policy: str
    num_steps: int
    total_requests: int
    completed_requests: int
    completion_rate: float
    avg_wait_time: float
    avg_trip_time: float
    total_distance_traveled: float
    vehicles: List[VehicleState]


class VehiclesLiveResponse(BaseModel):
    """Live vehicle positions for frontend map updates."""
    vehicles: List[VehicleState]
    policy: str
    timestamp: float


class PredictionResult(BaseModel):
    action: int
    vehicle_id: int
    model_name: str


class ModelInfo(BaseModel):
    name: str
    path: str


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    count: int


class MetricsResponse(BaseModel):
    policies: Dict[str, Dict[str, float]]
    comparison_summary: str


class HealthResponse(BaseModel):
    status: str
    version: str
