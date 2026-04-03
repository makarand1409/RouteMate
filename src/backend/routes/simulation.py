"""
Simulation endpoints - run the RouteMATE simulator via REST API.
"""
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Ensure project src is on the path
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from simulator import (
    SimulationEngine,
    NearestVehiclePolicy,
    RandomPolicy,
    GridCity,
)
from environment.gym_environment import RideSharingEnv
from backend.utils.model_loader import load_ppo_model
from backend.models import SimulationRequest, SimulationResult, VehicleState, VehiclesLiveResponse

router = APIRouter(prefix="/api", tags=["simulation"])

# Module-level state to store the latest simulation for live tracking
_latest_vehicles = []
_latest_policy = "greedy"


def _run_baseline_simulation(req: SimulationRequest) -> dict:
    """Run a simulation using a baseline (greedy or random) policy."""
    city = GridCity(size=10)
    if req.policy == "greedy":
        policy = NearestVehiclePolicy(city)
    else:
        policy = RandomPolicy(city)

    engine = SimulationEngine(
        city_size=10,
        num_vehicles=req.num_vehicles,
        vehicle_capacity=req.vehicle_capacity,
        request_rate=req.request_rate,
        matching_policy=policy,
    )
    results = engine.run(max_steps=req.num_steps, verbose=False)

    vehicles = [
        VehicleState(
            vehicle_id=v.vehicle_id,
            location=list(v.current_location),
            occupancy=v.get_occupancy(),
            queue_length=len(v.destination_queue),
            total_distance=v.total_distance_traveled,
            total_served=v.total_customers_served,
        )
        for v in engine.vehicles
    ]

    return SimulationResult(
        policy=req.policy,
        num_steps=req.num_steps,
        total_requests=results["total_requests"],
        completed_requests=results["completed_requests"],
        completion_rate=round(results["completion_rate"], 4),
        avg_wait_time=round(results["avg_wait_time"], 2),
        avg_trip_time=round(results["avg_trip_time"], 2),
        total_distance_traveled=results["total_distance_traveled"],
        vehicles=vehicles,
    )


def _run_ml_simulation(req: SimulationRequest) -> dict:
    """Run a simulation using the trained PPO agent."""
    model = load_ppo_model(req.model_name)

    env = RideSharingEnv(
        city_size=10,
        num_vehicles=req.num_vehicles,
        vehicle_capacity=req.vehicle_capacity,
        request_rate=req.request_rate,
        max_steps=req.num_steps,
    )

    obs, _info = env.reset()
    total_reward = 0.0

    for _ in range(req.num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break

    # Build vehicle states from the environment's internal vehicles list
    vehicles = [
        VehicleState(
            vehicle_id=v.vehicle_id,
            location=list(v.current_location),
            occupancy=v.get_occupancy(),
            queue_length=len(v.destination_queue),
            total_distance=v.total_distance_traveled,
            total_served=v.total_customers_served,
        )
        for v in env.vehicles
    ]

    total_requests = info.get("total_requests", env.episode_metrics["total_requests"])
    completed = info.get("completed_requests", env.episode_metrics["completed_requests"])

    return SimulationResult(
        policy="ml",
        num_steps=req.num_steps,
        total_requests=total_requests,
        completed_requests=completed,
        completion_rate=round(completed / total_requests, 4) if total_requests > 0 else 0.0,
        avg_wait_time=round(
            env.episode_metrics["total_wait_time"] / completed if completed > 0 else 0.0, 2
        ),
        avg_trip_time=0.0,  # Not tracked in gym env
        total_distance_traveled=sum(v.total_distance_traveled for v in env.vehicles),
        vehicles=vehicles,
    )


@router.post("/simulate", response_model=SimulationResult)
async def simulate(req: SimulationRequest):
    """
    Run a simulation with the specified policy and parameters.

    - **greedy**: Nearest-vehicle heuristic
    - **random**: Random vehicle assignment
    - **ml**: Trained PPO reinforcement learning agent
    """
    global _latest_vehicles, _latest_policy
    try:
        if req.policy in ("greedy", "random"):
            result = _run_baseline_simulation(req)
        else:
            result = _run_ml_simulation(req)
        
        # Store the vehicles for live tracking
        _latest_vehicles = result.vehicles
        _latest_policy = result.policy
        
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {e}")


@router.get("/vehicles/live", response_model=VehiclesLiveResponse)
async def get_live_vehicles():
    """
    Get current vehicle positions from the latest simulation.
    
    Frontend polls this endpoint every 500ms to update vehicle positions on the map.
    """
    global _latest_vehicles, _latest_policy
    return VehiclesLiveResponse(
        vehicles=_latest_vehicles or [],
        policy=_latest_policy,
        timestamp=time.time()
    )
