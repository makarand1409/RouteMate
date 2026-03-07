"""
Metrics & model listing endpoints.
"""
import sys
from pathlib import Path

from fastapi import APIRouter

_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from simulator import (
    SimulationEngine,
    NearestVehiclePolicy,
    RandomPolicy,
    GridCity,
)
from backend.utils.model_loader import get_available_models, MODELS_DIR
from backend.models import MetricsResponse, ModelsListResponse, ModelInfo

router = APIRouter(prefix="/api", tags=["metrics"])

# Number of evaluation episodes for quick benchmark
_EVAL_STEPS = 100
_EVAL_RUNS = 3


def _evaluate_policy(policy_name: str, num_runs: int = _EVAL_RUNS) -> dict:
    """Run several simulation trials and return averaged metrics."""
    rewards, completions, waits = [], [], []
    for _ in range(num_runs):
        city = GridCity(size=10)
        policy = (
            NearestVehiclePolicy(city) if policy_name == "greedy" else RandomPolicy(city)
        )
        engine = SimulationEngine(
            city_size=10,
            num_vehicles=5,
            vehicle_capacity=4,
            request_rate=2.0,
            matching_policy=policy,
        )
        results = engine.run(max_steps=_EVAL_STEPS, verbose=False)
        completions.append(results["completion_rate"])
        waits.append(results["avg_wait_time"])
        rewards.append(results["total_distance_traveled"])

    avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0
    return {
        "completion_rate": avg(completions),
        "avg_wait_time": avg(waits),
        "total_distance": avg(rewards),
        "num_runs": num_runs,
    }


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Compare greedy vs random baseline policies over multiple simulation runs.
    """
    greedy_metrics = _evaluate_policy("greedy")
    random_metrics = _evaluate_policy("random")

    better = (
        "greedy" if greedy_metrics["completion_rate"] >= random_metrics["completion_rate"] else "random"
    )
    summary = (
        f"Greedy completion rate: {greedy_metrics['completion_rate']:.2%}, "
        f"Random completion rate: {random_metrics['completion_rate']:.2%}. "
        f"'{better}' policy performs better on average."
    )

    return MetricsResponse(
        policies={"greedy": greedy_metrics, "random": random_metrics},
        comparison_summary=summary,
    )


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List all available trained models."""
    names = get_available_models()
    models = [
        ModelInfo(name=n, path=str(MODELS_DIR / f"{n}.zip"))
        for n in names
    ]
    return ModelsListResponse(models=models, count=len(models))
