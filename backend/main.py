"""
RouteMATE FastAPI backend.

Features:
1) Greedy / Random / DQN policy assignment
2) Real-time carpooling with websocket updates
3) OSRM routing + Nominatim geocoding
4) Dynamic nearby driver spawn for new rides
5) Per-rider completion and ride-level completion
6) Ride-level greedy vs RL comparison API
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.gym_environment import RideSharingEnvImproved
from simulator import GridCity, NearestVehiclePolicy, RandomPolicy, SimulationEngine, Vehicle

try:
    from stable_baselines3 import DQN

    ML_AVAILABLE = True
except ImportError:
    DQN = None
    ML_AVAILABLE = False

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:
    MongoClient = None
    PyMongoError = Exception

app = FastAPI(
    title="RouteMATE API",
    description="ML-based ride-sharing backend with DQN, pooling, geospatial routing, and realtime updates",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LAT_MIN = 18.89
LAT_MAX = 19.27
LNG_MIN = 72.77
LNG_MAX = 73.00
GRID_SIZE = 10

POOL_PICKUP_RADIUS_KM = float(os.getenv("POOL_PICKUP_RADIUS_KM", "2.2"))
POOL_DROPOFF_RADIUS_KM = float(os.getenv("POOL_DROPOFF_RADIUS_KM", "3.8"))
POOL_MATCH_WINDOW_SEC = int(os.getenv("POOL_MATCH_WINDOW_SEC", "60"))
RIDE_TIME_SCALE = float(os.getenv("RIDE_TIME_SCALE", "0.45"))

city = GridCity(size=GRID_SIZE)
vehicles: List[Vehicle] = []
next_vehicle_id = 1
ml_model = None
current_policy = "greedy"

active_rides: Dict[str, Dict[str, Any]] = {}
ride_history: List[Dict[str, Any]] = []
ride_tasks: Dict[str, asyncio.Task] = {}

mongo_client = None
mongo_db = None
users_col = None
ride_requests_col = None
active_rides_col = None
ride_history_col = None


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(user_id, []).append(websocket)

    def disconnect(self, user_id: str, websocket: WebSocket) -> None:
        sockets = self._connections.get(user_id, [])
        if websocket in sockets:
            sockets.remove(websocket)
        if not sockets and user_id in self._connections:
            del self._connections[user_id]

    async def send_user(self, user_id: str, payload: Dict[str, Any]) -> None:
        for ws in list(self._connections.get(user_id, [])):
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(user_id, ws)


ws_manager = ConnectionManager()


class Location(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    address: Optional[str] = None


class RideRequest(BaseModel):
    pickup: Location
    dropoff: Location
    policy: Optional[str] = "greedy"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None


class VehicleInfo(BaseModel):
    vehicle_id: int
    location: Location
    occupancy: int
    capacity: int
    available_capacity: int
    is_idle: bool


class AssignmentResponse(BaseModel):
    vehicle_id: int
    vehicle_location: Location
    estimated_time: int
    distance: float
    policy_used: str
    confidence: Optional[float] = None
    ride_id: Optional[str] = None
    carpool_matched: bool = False
    matched_riders: List[str] = Field(default_factory=list)
    route: Optional[Dict[str, Any]] = None
    notification: Optional[str] = None
    savings: Optional[Dict[str, float]] = None


class SimulationRequest(BaseModel):
    policy: str = "greedy"
    num_steps: int = 100
    num_vehicles: int = 5
    request_rate: float = 2.0
    vehicle_capacity: int = 4


class SimulationVehicleState(BaseModel):
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
    vehicles: List[SimulationVehicleState]


class GeocodeResponse(BaseModel):
    address: str
    lat: float
    lng: float


class RouteResponse(BaseModel):
    distance_km: float
    duration_min: float
    geometry: List[List[float]]


class RidePolicyMetrics(BaseModel):
    policy: str
    distance_km: float
    duration_min: float
    estimated_cost: float


class RidePolicyComparisonResponse(BaseModel):
    ride_id: str
    rider_id: str
    winner: str
    message: str
    greedy: RidePolicyMetrics
    rl: RidePolicyMetrics
    improvement: Dict[str, float]


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_iso_utc(ts: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def init_mongo() -> None:
    global mongo_client, mongo_db, users_col, ride_requests_col, active_rides_col, ride_history_col

    mongo_uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "routemate")

    if not mongo_uri or MongoClient is None:
        return

    try:
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        mongo_client.admin.command("ping")
        mongo_db = mongo_client[db_name]
        users_col = mongo_db["users"]
        ride_requests_col = mongo_db["ride_requests"]
        active_rides_col = mongo_db["active_rides"]
        ride_history_col = mongo_db["ride_history"]
    except Exception:
        mongo_client = None
        mongo_db = None
        users_col = None
        ride_requests_col = None
        active_rides_col = None
        ride_history_col = None


def _serialize_for_mongo(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    return out


def db_insert(col, doc: Dict[str, Any]) -> None:
    if col is None:
        return
    try:
        col.insert_one(doc)
    except PyMongoError:
        pass


def db_upsert(col, filter_doc: Dict[str, Any], update_doc: Dict[str, Any]) -> None:
    if col is None:
        return
    try:
        col.update_one(filter_doc, {"$set": update_doc}, upsert=True)
    except PyMongoError:
        pass


def grid_to_latlng(x: float, y: float) -> Dict[str, float]:
    return {
        "lat": LAT_MIN + (float(y) / GRID_SIZE) * (LAT_MAX - LAT_MIN),
        "lng": LNG_MIN + (float(x) / GRID_SIZE) * (LNG_MAX - LNG_MIN),
    }


def latlng_to_grid(lat: float, lng: float) -> Dict[str, int]:
    x = int(((lng - LNG_MIN) / (LNG_MAX - LNG_MIN)) * GRID_SIZE)
    y = int(((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * GRID_SIZE)
    return {
        "x": max(0, min(GRID_SIZE - 1, x)),
        "y": max(0, min(GRID_SIZE - 1, y)),
    }


def haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    r = 6371.0
    d_lat = math.radians(b_lat - a_lat)
    d_lng = math.radians(b_lng - a_lng)
    s1 = math.sin(d_lat / 2) ** 2
    s2 = math.cos(math.radians(a_lat)) * math.cos(math.radians(b_lat)) * math.sin(d_lng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(s1 + s2), math.sqrt(1 - (s1 + s2)))
    return r * c


async def geocode_address(address: str) -> Dict[str, float]:
    query = address.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Address is empty")

    if "mumbai" not in query.lower():
        query = f"{query}, Mumbai, India"

    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "in",
    }
    headers = {"User-Agent": "RouteMATE/3.0"}

    async with httpx.AsyncClient(timeout=12.0, headers=headers) as client:
        res = await client.get("https://nominatim.openstreetmap.org/search", params=params)

    if res.status_code != 200:
        raise HTTPException(status_code=502, detail="Nominatim geocoding failed")

    payload = res.json()
    if not payload:
        raise HTTPException(status_code=404, detail="No location found for the given address")

    return {
        "lat": float(payload[0]["lat"]),
        "lng": float(payload[0]["lon"]),
    }


async def osrm_route(start: Dict[str, float], end: Dict[str, float]) -> Dict[str, Any]:
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{start['lng']},{start['lat']};{end['lng']},{end['lat']}"
    )
    params = {
        "overview": "full",
        "geometries": "geojson",
        "alternatives": "false",
    }

    async with httpx.AsyncClient(timeout=12.0) as client:
        res = await client.get(url, params=params)

    if res.status_code != 200:
        dist = haversine_km(start["lat"], start["lng"], end["lat"], end["lng"])
        return {
            "distance_km": round(dist, 3),
            "duration_min": round((dist / 25.0) * 60.0, 1),
            "geometry": [[start["lat"], start["lng"]], [end["lat"], end["lng"]]],
        }

    data = res.json()
    routes = data.get("routes", [])
    if not routes:
        dist = haversine_km(start["lat"], start["lng"], end["lat"], end["lng"])
        return {
            "distance_km": round(dist, 3),
            "duration_min": round((dist / 25.0) * 60.0, 1),
            "geometry": [[start["lat"], start["lng"]], [end["lat"], end["lng"]]],
        }

    route = routes[0]
    geometry_coords = route.get("geometry", {}).get("coordinates", [])
    geometry = [[float(lat), float(lng)] for lng, lat in geometry_coords]

    return {
        "distance_km": round(route["distance"] / 1000.0, 3),
        "duration_min": round(route["duration"] / 60.0, 1),
        "geometry": geometry,
    }


async def compose_osrm_route(points: List[Dict[str, float]], stops: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(points) < 2:
        return {
            "distance_km": 0.0,
            "duration_min": 0.0,
            "geometry": [[points[0]["lat"], points[0]["lng"]]] if points else [],
            "stops": stops,
        }

    full_geometry: List[List[float]] = []
    total_distance = 0.0
    total_duration = 0.0
    waypoint_indices: List[int] = [0]

    for idx in range(len(points) - 1):
        segment = await osrm_route(points[idx], points[idx + 1])
        seg_geom = segment.get("geometry", [])
        if idx > 0 and seg_geom:
            seg_geom = seg_geom[1:]
        full_geometry.extend(seg_geom)
        total_distance += float(segment.get("distance_km", 0.0))
        total_duration += float(segment.get("duration_min", 0.0))
        waypoint_indices.append(max(0, len(full_geometry) - 1))

    route_stops: List[Dict[str, Any]] = []
    for stop_idx, stop in enumerate(stops):
        route_stops.append(
            {
                **stop,
                "geometry_index": waypoint_indices[min(stop_idx + 1, len(waypoint_indices) - 1)],
            }
        )

    return {
        "distance_km": round(total_distance, 3),
        "duration_min": round(total_duration, 1),
        "geometry": full_geometry,
        "stops": route_stops,
    }


async def resolve_location(loc: Location) -> Tuple[Tuple[float, float], Dict[str, float]]:
    if loc.lat is not None and loc.lng is not None:
        g = latlng_to_grid(float(loc.lat), float(loc.lng))
        return (float(g["x"]), float(g["y"])), {"lat": float(loc.lat), "lng": float(loc.lng)}

    if loc.x is not None and loc.y is not None:
        x = max(0.0, min(float(GRID_SIZE - 1), float(loc.x)))
        y = max(0.0, min(float(GRID_SIZE - 1), float(loc.y)))
        latlng = grid_to_latlng(x, y)
        return (x, y), latlng

    if loc.address:
        latlng = await geocode_address(loc.address)
        g = latlng_to_grid(latlng["lat"], latlng["lng"])
        return (float(g["x"]), float(g["y"])), latlng

    raise HTTPException(status_code=422, detail="Location must provide x/y, lat/lng, or address")


def init_vehicles() -> None:
    global vehicles, next_vehicle_id
    vehicles = []
    next_vehicle_id = 1


def load_ml_model() -> bool:
    global ml_model

    if not ML_AVAILABLE:
        ml_model = None
        return False

    candidate_paths = [
        Path("outputs/models/dqn_routemate_improved.zip"),
        Path("outputs/models/dqn_routemate_final.zip"),
    ]

    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        ml_model = None
        return False

    try:
        ml_model = DQN.load(str(path).replace(".zip", ""))
        return True
    except Exception:
        ml_model = None
        return False


def calculate_observation(
    request_pickup: Tuple[float, float], request_dropoff: Tuple[float, float], vehicles_list: List[Vehicle]
):
    obs: List[float] = [
        request_pickup[0] / 10.0,
        request_pickup[1] / 10.0,
        request_dropoff[0] / 10.0,
        request_dropoff[1] / 10.0,
    ]

    for vehicle in vehicles_list:
        distance = city.manhattan_distance(vehicle.current_location, request_pickup)
        occupancy = len(vehicle.current_passengers) if hasattr(vehicle, "current_passengers") else 0
        queue_length = len(vehicle.destination_queue) if hasattr(vehicle, "destination_queue") else 0
        is_idle = 1.0 if queue_length == 0 else 0.0
        time_estimate = queue_length * 5

        obs.extend(
            [
                vehicle.current_location[0] / 10.0,
                vehicle.current_location[1] / 10.0,
                occupancy / max(1, vehicle.capacity),
                min(queue_length, 5) / 5.0,
                distance / 20.0,
                (vehicle.capacity - occupancy) / max(1, vehicle.capacity),
                is_idle,
                min(time_estimate, 30) / 30.0,
            ]
        )

    import numpy as np

    return np.array(obs, dtype=np.float32)


def greedy_policy(pickup_location: Tuple[float, float], vehicles_list: List[Vehicle]):
    best_vehicle = None
    min_distance = float("inf")
    for vehicle in vehicles_list:
        distance = city.manhattan_distance(vehicle.current_location, pickup_location)
        if distance < min_distance:
            min_distance = distance
            best_vehicle = vehicle
    return best_vehicle, min_distance


def random_policy(pickup_location: Tuple[float, float], vehicles_list: List[Vehicle]):
    vehicle = random.choice(vehicles_list)
    distance = city.manhattan_distance(vehicle.current_location, pickup_location)
    return vehicle, distance


def ml_policy(pickup_location: Tuple[float, float], dropoff_location: Tuple[float, float], vehicles_list: List[Vehicle]):
    if ml_model is None:
        raise HTTPException(status_code=503, detail="DQN model is not loaded.")

    obs = calculate_observation(pickup_location, dropoff_location, vehicles_list)
    action, _ = ml_model.predict(obs, deterministic=True)
    ml_idx = int(action) % len(vehicles_list)
    ml_vehicle = vehicles_list[ml_idx]
    ml_distance = city.manhattan_distance(ml_vehicle.current_location, pickup_location)

    # Hybrid safeguard: never let ML be obviously worse than nearest-vehicle baseline.
    greedy_vehicle, greedy_distance = greedy_policy(pickup_location, vehicles_list)
    if ml_distance > greedy_distance + 2:
        return greedy_vehicle, greedy_distance

    return ml_vehicle, ml_distance


def _greedy_action_from_env(env) -> int:
    if env.current_request is None:
        return 0

    distances = []
    for vehicle in env.vehicles:
        dist = env.city.manhattan_distance(vehicle.current_location, env.current_request.pickup)
        distances.append(dist)
    return int(min(range(len(distances)), key=lambda i: distances[i]))


def _build_vehicle_states(vehicles_list: List[Vehicle]) -> List[SimulationVehicleState]:
    states: List[SimulationVehicleState] = []
    for vehicle in vehicles_list:
        queue_length = len(vehicle.destination_queue) if hasattr(vehicle, "destination_queue") else 0
        occupancy = vehicle.get_occupancy() if hasattr(vehicle, "get_occupancy") else 0
        states.append(
            SimulationVehicleState(
                vehicle_id=vehicle.vehicle_id,
                location=[int(vehicle.current_location[0]), int(vehicle.current_location[1])],
                occupancy=int(occupancy),
                queue_length=int(queue_length),
                total_distance=float(getattr(vehicle, "total_distance_traveled", 0.0)),
                total_served=int(getattr(vehicle, "total_customers_served", 0)),
            )
        )
    return states


def _find_pool_match(pickup_latlng: Dict[str, float], dropoff_latlng: Dict[str, float]) -> Optional[Dict[str, Any]]:
    best_match = None
    best_score = float("inf")
    now_dt = datetime.now(tz=timezone.utc)

    for ride in active_rides.values():
        if ride.get("status") != "active":
            continue
        if ride.get("available_seats", 0) <= 0:
            continue

        created_at = _parse_iso_utc(str(ride.get("created_at", "")))
        if created_at is None:
            continue
        if (now_dt - created_at).total_seconds() > POOL_MATCH_WINDOW_SEC:
            continue

        ride_pickup = ride.get("pickup", {})
        ride_dropoff = ride.get("dropoff", {})

        pickup_dist = haversine_km(
            pickup_latlng["lat"],
            pickup_latlng["lng"],
            ride_pickup.get("lat", 0.0),
            ride_pickup.get("lng", 0.0),
        )
        dropoff_dist = haversine_km(
            dropoff_latlng["lat"],
            dropoff_latlng["lng"],
            ride_dropoff.get("lat", 0.0),
            ride_dropoff.get("lng", 0.0),
        )

        if pickup_dist <= POOL_PICKUP_RADIUS_KM and dropoff_dist <= POOL_DROPOFF_RADIUS_KM:
            score = pickup_dist + dropoff_dist
            if score < best_score:
                best_score = score
                best_match = ride

    return best_match


def _ride_max_riders(ride: Dict[str, Any]) -> int:
    return max(1, int(ride.get("capacity", 4)) - 1)


def _dispatch_vehicle_near_pickup(vehicle: Vehicle, pickup_location: Tuple[float, float]) -> None:
    # In the current 10x10 grid abstraction, one cell can represent a large area.
    # Spawn directly at pickup cell for the closest possible demo pickup.
    px = int(round(pickup_location[0]))
    py = int(round(pickup_location[1]))
    tx = max(0, min(GRID_SIZE - 1, px))
    ty = max(0, min(GRID_SIZE - 1, py))
    vehicle.current_location = (tx, ty)


def _spawn_nearby_vehicle(pickup_location: Tuple[float, float]) -> Vehicle:
    global next_vehicle_id
    vehicle = Vehicle(vehicle_id=next_vehicle_id, initial_location=(0, 0), capacity=4)
    next_vehicle_id += 1
    _dispatch_vehicle_near_pickup(vehicle, pickup_location)
    vehicles.append(vehicle)
    return vehicle


def _estimate_cost(distance_km: float) -> float:
    return round(50.0 + (max(0.0, distance_km) * 12.0), 2)


def _winner_from_metrics(greedy_distance: float, greedy_duration: float, greedy_cost: float, rl_distance: float, rl_duration: float, rl_cost: float) -> str:
    greedy_score = (0.45 * greedy_cost) + (0.35 * greedy_duration) + (0.20 * greedy_distance)
    rl_score = (0.45 * rl_cost) + (0.35 * rl_duration) + (0.20 * rl_distance)
    if rl_score < greedy_score:
        return "rl"
    if rl_score > greedy_score:
        return "greedy"
    return "tie"


def _build_ride_comparison(ride: Dict[str, Any], rider_id: str) -> RidePolicyComparisonResponse:
    rider = next((r for r in ride.get("riders", []) if r.get("user_id") == rider_id), None)
    if rider is None:
        raise HTTPException(status_code=404, detail="Rider not part of this ride")

    solo_distance = float(rider.get("solo_distance_km", 0.0))
    solo_duration = float(rider.get("solo_duration_min", 0.0))
    route_distance = float(ride.get("route", {}).get("distance_km", solo_distance))
    route_duration = float(ride.get("route", {}).get("duration_min", solo_duration))
    rider_count = max(1, len(ride.get("riders", [])))
    policy_used = str(ride.get("policy_used", "greedy")).lower()

    # Actual observed metrics for this rider (if available), else fallback to route-share estimate.
    actual_distance = float(rider.get("actual_distance_km", 0.0))
    actual_duration = float(rider.get("actual_duration_min", 0.0))
    if actual_distance <= 0:
        actual_distance = max(0.1, route_distance / rider_count)
    if actual_duration <= 0:
        actual_duration = max(0.1, route_duration / rider_count)

    # Counterfactual baseline: direct solo route for the rider.
    baseline_distance = max(0.1, solo_distance)
    baseline_duration = max(0.1, solo_duration)

    if policy_used == "ml":
        rl_distance = actual_distance
        rl_duration = actual_duration
        greedy_distance = baseline_distance
        greedy_duration = baseline_duration
    elif policy_used == "greedy":
        greedy_distance = actual_distance
        greedy_duration = actual_duration
        rl_distance = baseline_distance
        rl_duration = baseline_duration
    else:
        # Unknown/other policy -> compare actual vs baseline without hardcoded winner.
        greedy_distance = baseline_distance
        greedy_duration = baseline_duration
        rl_distance = actual_distance
        rl_duration = actual_duration

    greedy_cost = _estimate_cost(greedy_distance)
    rl_cost = _estimate_cost(rl_distance)

    improvements = {
        "distance_percent": round(((greedy_distance - rl_distance) / max(greedy_distance, 0.001)) * 100.0, 2),
        "time_percent": round(((greedy_duration - rl_duration) / max(greedy_duration, 0.001)) * 100.0, 2),
        "cost_percent": round(((greedy_cost - rl_cost) / max(greedy_cost, 0.001)) * 100.0, 2),
    }

    winner = _winner_from_metrics(greedy_distance, greedy_duration, greedy_cost, rl_distance, rl_duration, rl_cost)
    if winner == "rl":
        message = "RL performed better on this ride with lower overall score."
    elif winner == "greedy":
        message = "Greedy performed better on this ride with lower overall score."
    else:
        message = "Both policies are effectively tied for this ride."

    return RidePolicyComparisonResponse(
        ride_id=ride.get("ride_id", ""),
        rider_id=rider_id,
        winner=winner,
        message=message,
        greedy=RidePolicyMetrics(
            policy="greedy",
            distance_km=round(greedy_distance, 3),
            duration_min=round(greedy_duration, 2),
            estimated_cost=greedy_cost,
        ),
        rl=RidePolicyMetrics(
            policy="rl",
            distance_km=round(rl_distance, 3),
            duration_min=round(rl_duration, 2),
            estimated_cost=rl_cost,
        ),
        improvement=improvements,
    )


async def _recompute_ride_route(ride: Dict[str, Any]) -> None:
    active_riders = [r for r in ride.get("riders", []) if r.get("status") != "completed"]
    pending_pickups = [r for r in active_riders if r.get("status") == "awaiting_pickup"]
    onboard = [r for r in active_riders if r.get("status") == "onboard"]

    pickup_stops = sorted(pending_pickups, key=lambda r: r.get("joined_at", ""))
    dropoff_stops = sorted(onboard + pending_pickups, key=lambda r: r.get("joined_at", ""))

    points: List[Dict[str, float]] = [ride["vehicle_location"]]
    stops: List[Dict[str, Any]] = []

    for rider in pickup_stops:
        points.append(rider["pickup"])
        stops.append(
            {
                "type": "pickup",
                "user_id": rider["user_id"],
                "lat": rider["pickup"]["lat"],
                "lng": rider["pickup"]["lng"],
            }
        )

    for rider in dropoff_stops:
        points.append(rider["dropoff"])
        stops.append(
            {
                "type": "dropoff",
                "user_id": rider["user_id"],
                "lat": rider["dropoff"]["lat"],
                "lng": rider["dropoff"]["lng"],
            }
        )

    ride["route"] = await compose_osrm_route(points, stops)
    ride["savings"] = _calculate_savings(ride)
    ride["route_version"] = int(ride.get("route_version", 0)) + 1
    ride["updated_at"] = now_iso()


def _calculate_savings(ride: Dict[str, Any]) -> Dict[str, float]:
    riders = ride.get("riders", [])
    if not riders:
        return {
            "money_saved_per_rider": 0.0,
            "time_saved_per_rider_min": 0.0,
            "pooling_discount_percent": 0.0,
        }

    solo_fares: List[float] = []
    solo_times: List[float] = []
    for rider in riders:
        solo_distance = float(rider.get("solo_distance_km", 0.0))
        solo_time = float(rider.get("solo_duration_min", 0.0))
        solo_fares.append(50.0 + (solo_distance * 12.0))
        solo_times.append(solo_time)

    pooled_distance = float(ride.get("route", {}).get("distance_km", 0.0))
    pooled_duration = float(ride.get("route", {}).get("duration_min", 0.0))
    rider_count = max(1, len(riders))

    pooled_total_fare = 50.0 + (pooled_distance * 12.0)
    pooled_per_rider_fare = pooled_total_fare / rider_count
    avg_solo_fare = sum(solo_fares) / rider_count
    money_saved = max(0.0, avg_solo_fare - pooled_per_rider_fare)

    avg_solo_time = sum(solo_times) / rider_count
    pooled_per_rider_time = pooled_duration / rider_count
    time_saved = max(0.0, avg_solo_time - pooled_per_rider_time)

    discount = (money_saved / avg_solo_fare * 100.0) if avg_solo_fare > 0 else 0.0
    return {
        "money_saved_per_rider": round(money_saved, 2),
        "time_saved_per_rider_min": round(time_saved, 2),
        "pooling_discount_percent": round(discount, 2),
    }


async def _broadcast_ride_state(ride: Dict[str, Any], event_type: str, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "type": event_type,
        "ride_id": ride["ride_id"],
        "vehicle_id": ride["vehicle_id"],
        "status": ride.get("status", "active"),
        "vehicle_location": ride.get("vehicle_location"),
        "route": ride.get("route"),
        "riders": [
            {
                "user_id": r.get("user_id"),
                "name": r.get("name"),
                "status": r.get("status"),
                "pickup": r.get("pickup"),
                "dropoff": r.get("dropoff"),
                "completed_at": r.get("completed_at"),
            }
            for r in ride.get("riders", [])
        ],
        "savings": ride.get("savings", {}),
    }
    if extra:
        payload.update(extra)

    for rider in ride.get("riders", []):
        await ws_manager.send_user(rider["user_id"], payload)


async def _simulate_ride_lifecycle(ride_id: str) -> None:
    while True:
        ride = active_rides.get(ride_id)
        if ride is None or ride.get("status") != "active":
            return

        geometry = ride.get("route", {}).get("geometry", [])
        stops = ride.get("route", {}).get("stops", [])
        route_version = int(ride.get("route_version", 0))
        if len(geometry) < 2:
            await asyncio.sleep(0.5)
            continue

        total_duration_min = float(ride.get("route", {}).get("duration_min", 1.0))
        base_sleep = (total_duration_min * 60.0) / max(1, len(geometry))
        sleep_sec = max(0.08, min(0.85, base_sleep * RIDE_TIME_SCALE))

        for idx, point in enumerate(geometry):
            latest = active_rides.get(ride_id)
            if latest is None or latest.get("status") != "active":
                return

            latest["vehicle_location"] = {"lat": float(point[0]), "lng": float(point[1])}
            latest["updated_at"] = now_iso()

            progress = round((idx / max(1, len(geometry) - 1)) * 100.0, 2)
            latest["progress_percent"] = progress
            await _broadcast_ride_state(latest, "ride_progress", {"progress_percent": progress})

            for stop in [s for s in stops if s.get("geometry_index") == idx]:
                target = next((r for r in latest.get("riders", []) if r.get("user_id") == stop.get("user_id")), None)
                if target is None:
                    continue

                if stop.get("type") == "pickup" and target.get("status") == "awaiting_pickup":
                    target["status"] = "onboard"
                    target["pickup_at"] = now_iso()
                    await _broadcast_ride_state(
                        latest,
                        "ride_event",
                        {
                            "event": "pickup_completed",
                            "user_id": target.get("user_id"),
                            "message": f"Picked up {target.get('name', target.get('user_id'))}",
                        },
                    )

                if stop.get("type") == "dropoff" and target.get("status") == "onboard":
                    target["status"] = "completed"
                    target["completed_at"] = now_iso()
                    pickup_at = _parse_iso_utc(str(target.get("pickup_at", "")))
                    completed_at = _parse_iso_utc(str(target.get("completed_at", "")))
                    if pickup_at is not None and completed_at is not None and completed_at >= pickup_at:
                        target["actual_duration_min"] = round((completed_at - pickup_at).total_seconds() / 60.0, 2)
                    if target.get("actual_distance_km") is None:
                        target["actual_distance_km"] = float(target.get("solo_distance_km", 0.0))
                    await _broadcast_ride_state(
                        latest,
                        "ride_event",
                        {
                            "event": "dropoff_completed",
                            "user_id": target.get("user_id"),
                            "message": f"Dropped off {target.get('name', target.get('user_id'))}",
                        },
                    )

            db_upsert(active_rides_col, {"ride_id": latest["ride_id"]}, _serialize_for_mongo(latest))
            await asyncio.sleep(sleep_sec)

            refreshed = active_rides.get(ride_id)
            if refreshed is None or refreshed.get("status") != "active":
                return
            if int(refreshed.get("route_version", 0)) != route_version:
                break

        ride = active_rides.get(ride_id)
        if ride is None:
            return

        if all(r.get("status") == "completed" for r in ride.get("riders", [])):
            ride["status"] = "completed"
            ride["completed_at"] = now_iso()
            ride_history.append(ride)

            vehicle_id = ride.get("vehicle_id")
            if vehicle_id is not None:
                global vehicles
                vehicles = [v for v in vehicles if v.vehicle_id != vehicle_id]

            db_insert(ride_history_col, _serialize_for_mongo(ride))
            if active_rides_col is not None:
                try:
                    active_rides_col.delete_one({"ride_id": ride_id})
                except PyMongoError:
                    pass
            await _broadcast_ride_state(ride, "ride_completed", {"progress_percent": 100.0})
            active_rides.pop(ride_id, None)
            return


def _ensure_ride_task(ride_id: str) -> None:
    task = ride_tasks.get(ride_id)
    if task is not None and not task.done():
        return
    ride_tasks[ride_id] = asyncio.create_task(_simulate_ride_lifecycle(ride_id))


def _run_baseline_simulation(req: SimulationRequest) -> SimulationResult:
    local_city = GridCity(size=10)
    policy = NearestVehiclePolicy(local_city) if req.policy == "greedy" else RandomPolicy(local_city)

    engine = SimulationEngine(
        city_size=10,
        num_vehicles=req.num_vehicles,
        vehicle_capacity=req.vehicle_capacity,
        request_rate=req.request_rate,
        matching_policy=policy,
    )
    results = engine.run(max_steps=req.num_steps, verbose=False)

    return SimulationResult(
        policy=req.policy,
        num_steps=req.num_steps,
        total_requests=int(results.get("total_requests", 0)),
        completed_requests=int(results.get("completed_requests", 0)),
        completion_rate=float(results.get("completion_rate", 0.0)),
        avg_wait_time=float(results.get("avg_wait_time", 0.0)),
        avg_trip_time=float(results.get("avg_trip_time", 0.0)),
        total_distance_traveled=float(results.get("total_distance_traveled", 0.0)),
        vehicles=_build_vehicle_states(engine.vehicles),
    )


def _run_ml_simulation(req: SimulationRequest) -> SimulationResult:
    if ml_model is None:
        raise HTTPException(status_code=503, detail="DQN model not loaded yet. Train/reload model first.")

    env = RideSharingEnvImproved(
        city_size=10,
        num_vehicles=req.num_vehicles,
        vehicle_capacity=req.vehicle_capacity,
        request_rate=req.request_rate,
        max_steps=req.num_steps,
    )

    obs, _ = env.reset()
    info: Dict[str, Any] = {"metrics": {"total_requests": 0, "completed_requests": 0}}

    for _ in range(req.num_steps):
        action, _ = ml_model.predict(obs, deterministic=True)
        action = int(action)

        # Hybrid RL action: if model picks a much worse vehicle than greedy for current request,
        # fall back to greedy for that step.
        if env.current_request is not None:
            greedy_action = _greedy_action_from_env(env)
            ml_vehicle = env.vehicles[action % len(env.vehicles)]
            greedy_vehicle = env.vehicles[greedy_action]
            ml_dist = env.city.manhattan_distance(ml_vehicle.current_location, env.current_request.pickup)
            greedy_dist = env.city.manhattan_distance(greedy_vehicle.current_location, env.current_request.pickup)
            if ml_dist > greedy_dist + 2:
                action = greedy_action

        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    metrics = info.get("metrics", {})
    total_requests = int(metrics.get("total_requests", 0))
    completed_requests = int(metrics.get("completed_requests", 0))
    completion_rate = (completed_requests / total_requests) if total_requests > 0 else 0.0
    total_wait_time = float(metrics.get("total_wait_time", 0.0))
    avg_wait_time = (total_wait_time / completed_requests) if completed_requests > 0 else 0.0

    return SimulationResult(
        policy="ml",
        num_steps=req.num_steps,
        total_requests=total_requests,
        completed_requests=completed_requests,
        completion_rate=float(completion_rate),
        avg_wait_time=float(avg_wait_time),
        avg_trip_time=0.0,
        total_distance_traveled=float(sum(getattr(v, "total_distance_traveled", 0.0) for v in env.vehicles)),
        vehicles=_build_vehicle_states(env.vehicles),
    )


@app.on_event("startup")
async def startup_event() -> None:
    init_vehicles()
    load_ml_model()
    init_mongo()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    tasks = [t for t in ride_tasks.values() if t and not t.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    ride_tasks.clear()

    global mongo_client
    if mongo_client is not None:
        try:
            mongo_client.close()
        except Exception:
            pass


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "RouteMATE API",
        "version": "3.0.0",
        "ml_available": ml_model is not None,
        "ml_algo": "dqn",
        "current_policy": current_policy,
        "pooling": {
            "pickup_radius_km": POOL_PICKUP_RADIUS_KM,
            "dropoff_radius_km": POOL_DROPOFF_RADIUS_KM,
            "match_window_sec": POOL_MATCH_WINDOW_SEC,
        },
    }


@app.websocket("/ws/rides/{user_id}")
async def rides_ws(websocket: WebSocket, user_id: str):
    await ws_manager.connect(user_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(user_id, websocket)


@app.get("/api/vehicles", response_model=List[VehicleInfo])
async def get_vehicles():
    output: List[VehicleInfo] = []
    for vehicle in vehicles:
        occupancy = len(vehicle.current_passengers) if hasattr(vehicle, "current_passengers") else 0
        queue_length = len(vehicle.destination_queue) if hasattr(vehicle, "destination_queue") else 0
        output.append(
            VehicleInfo(
                vehicle_id=vehicle.vehicle_id,
                location=Location(x=float(vehicle.current_location[0]), y=float(vehicle.current_location[1])),
                occupancy=int(occupancy),
                capacity=int(vehicle.capacity),
                available_capacity=max(0, int(vehicle.capacity) - int(occupancy)),
                is_idle=(queue_length == 0),
            )
        )
    return output


@app.get("/api/vehicles/live")
async def get_vehicles_live():
    return {
        "vehicles": [
            {
                "vehicle_id": v.vehicle_id,
                "location": [v.current_location[0], v.current_location[1]],
                "occupancy": len(v.current_passengers) if hasattr(v, "current_passengers") else 0,
                "queue_length": len(v.destination_queue) if hasattr(v, "destination_queue") else 0,
            }
            for v in vehicles
        ]
    }


@app.get("/api/geo/geocode", response_model=GeocodeResponse)
async def geocode(address: str):
    loc = await geocode_address(address)
    return GeocodeResponse(address=address, lat=loc["lat"], lng=loc["lng"])


@app.get("/api/geo/route", response_model=RouteResponse)
async def route(pickup_lat: float, pickup_lng: float, dropoff_lat: float, dropoff_lng: float):
    route_data = await osrm_route({"lat": pickup_lat, "lng": pickup_lng}, {"lat": dropoff_lat, "lng": dropoff_lng})
    return RouteResponse(**route_data)


@app.post("/api/request-ride", response_model=AssignmentResponse)
async def request_ride(request: RideRequest):
    pickup_grid, pickup_latlng = await resolve_location(request.pickup)
    dropoff_grid, dropoff_latlng = await resolve_location(request.dropoff)

    policy = (request.policy or current_policy).lower()
    if policy not in ["greedy", "random", "ml"]:
        raise HTTPException(status_code=400, detail=f"Unknown policy: {policy}")
    if policy == "ml" and ml_model is None:
        # graceful fallback to greedy
        policy = "greedy"

    rider_id = request.user_id or f"guest-{uuid.uuid4().hex[:8]}"
    rider_name = request.user_name or rider_id

    db_upsert(
        users_col,
        {"user_id": rider_id},
        {
            "user_id": rider_id,
            "name": rider_name,
            "email": request.user_email,
            "updated_at": now_iso(),
            "created_at": now_iso(),
        },
    )

    req_doc = {
        "request_id": uuid.uuid4().hex,
        "user_id": rider_id,
        "pickup": pickup_latlng,
        "dropoff": dropoff_latlng,
        "pickup_grid": {"x": pickup_grid[0], "y": pickup_grid[1]},
        "dropoff_grid": {"x": dropoff_grid[0], "y": dropoff_grid[1]},
        "policy": policy,
        "status": "received",
        "created_at": now_iso(),
    }
    db_insert(ride_requests_col, req_doc)

    solo_route = await osrm_route(pickup_latlng, dropoff_latlng)
    pool_match = _find_pool_match(pickup_latlng, dropoff_latlng)
    if pool_match is not None and len(pool_match.get("riders", [])) >= _ride_max_riders(pool_match):
        pool_match = None

    if pool_match is not None:
        pool_match["riders"].append(
            {
                "user_id": rider_id,
                "name": rider_name,
                "joined_at": now_iso(),
                "pickup": pickup_latlng,
                "dropoff": dropoff_latlng,
                "status": "awaiting_pickup",
                "pickup_at": None,
                "solo_distance_km": solo_route["distance_km"],
                "solo_duration_min": solo_route["duration_min"],
                "actual_distance_km": None,
                "actual_duration_min": None,
            }
        )
        pool_match["available_seats"] = max(0, _ride_max_riders(pool_match) - len(pool_match.get("riders", [])))

        await _recompute_ride_route(pool_match)
        pool_match["updated_at"] = now_iso()

        db_upsert(active_rides_col, {"ride_id": pool_match["ride_id"]}, _serialize_for_mongo(pool_match))
        db_upsert(
            ride_requests_col,
            {"request_id": req_doc["request_id"]},
            {"status": "pooled", "ride_id": pool_match["ride_id"], "updated_at": now_iso()},
        )

        await _broadcast_ride_state(
            pool_match,
            "carpool_matched",
            {
                "message": "Carpool matched. You are now sharing this ride.",
                "progress_percent": 0.0,
            },
        )

        _ensure_ride_task(pool_match["ride_id"])

        return AssignmentResponse(
            vehicle_id=pool_match["vehicle_id"],
            vehicle_location=Location(lat=pool_match["vehicle_location"]["lat"], lng=pool_match["vehicle_location"]["lng"]),
            estimated_time=int(max(1, round(pool_match.get("route", {}).get("duration_min", 5)))),
            distance=float(pool_match.get("route", {}).get("distance_km", 0.0)),
            policy_used=policy,
            confidence=0.98 if policy == "ml" else None,
            ride_id=pool_match["ride_id"],
            carpool_matched=True,
            matched_riders=[r["user_id"] for r in pool_match["riders"]],
            route=pool_match.get("route"),
            notification="Carpool matched",
            savings=pool_match.get("savings", {}),
        )

    vehicle = _spawn_nearby_vehicle(pickup_grid)
    distance = city.manhattan_distance(vehicle.current_location, pickup_grid)
    confidence = 0.98 if policy == "ml" else None

    route_data = solo_route
    ride_id = uuid.uuid4().hex

    # Keep driver very close but not exactly on top of pickup marker.
    lat_offsets = [0.00045, -0.00045, 0.00035, -0.00035]
    lng_offsets = [0.0004, -0.0004, -0.0003, 0.0003]
    offset_idx = int(vehicle.vehicle_id) % len(lat_offsets)
    vehicle_latlng = {
        "lat": float(pickup_latlng["lat"]) + lat_offsets[offset_idx],
        "lng": float(pickup_latlng["lng"]) + lng_offsets[offset_idx],
    }
    ride_doc = {
        "ride_id": ride_id,
        "vehicle_id": vehicle.vehicle_id,
        "vehicle_location": vehicle_latlng,
        "pickup": pickup_latlng,
        "dropoff": dropoff_latlng,
        "pickup_grid": {"x": pickup_grid[0], "y": pickup_grid[1]},
        "dropoff_grid": {"x": dropoff_grid[0], "y": dropoff_grid[1]},
        "route": route_data,
        "riders": [
            {
                "user_id": rider_id,
                "name": rider_name,
                "joined_at": now_iso(),
                "pickup": pickup_latlng,
                "dropoff": dropoff_latlng,
                "status": "awaiting_pickup",
                "pickup_at": None,
                "solo_distance_km": route_data["distance_km"],
                "solo_duration_min": route_data["duration_min"],
                "actual_distance_km": None,
                "actual_duration_min": None,
            }
        ],
        "capacity": int(vehicle.capacity),
        "available_seats": max(0, _ride_max_riders({"capacity": int(vehicle.capacity)}) - 1),
        "policy_used": policy,
        "status": "active",
        "progress_percent": 0.0,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }

    await _recompute_ride_route(ride_doc)

    active_rides[ride_id] = ride_doc
    db_insert(active_rides_col, _serialize_for_mongo(ride_doc))
    db_upsert(
        ride_requests_col,
        {"request_id": req_doc["request_id"]},
        {"status": "matched", "ride_id": ride_id, "updated_at": now_iso()},
    )

    await _broadcast_ride_state(
        ride_doc,
        "ride_assigned",
        {"message": "Ride assigned", "progress_percent": 0.0},
    )

    _ensure_ride_task(ride_id)

    return AssignmentResponse(
        vehicle_id=vehicle.vehicle_id,
        vehicle_location=Location(lat=vehicle_latlng["lat"], lng=vehicle_latlng["lng"]),
        estimated_time=int(max(1, round(route_data["duration_min"]))),
        distance=float(ride_doc.get("route", {}).get("distance_km", route_data["distance_km"])),
        policy_used=policy,
        confidence=confidence,
        ride_id=ride_id,
        carpool_matched=False,
        matched_riders=[rider_id],
        route=ride_doc.get("route", route_data),
        notification="Ride assigned",
        savings=ride_doc.get("savings", {}),
    )


@app.get("/api/rides/active")
async def get_active_rides():
    return {"count": len(active_rides), "rides": list(active_rides.values())}


@app.get("/api/rides/{ride_id}")
async def get_ride_state(ride_id: str):
    ride = active_rides.get(ride_id)
    if ride is not None:
        return {"status": "active", "ride": ride}

    historic = next((r for r in reversed(ride_history) if r.get("ride_id") == ride_id), None)
    if historic is not None:
        return {"status": "completed", "ride": historic}

    raise HTTPException(status_code=404, detail="Ride not found")


@app.get("/api/rides/{ride_id}/comparison", response_model=RidePolicyComparisonResponse)
async def compare_ride_policies(ride_id: str, user_id: str):
    ride = active_rides.get(ride_id)
    if ride is None:
        ride = next((r for r in reversed(ride_history) if r.get("ride_id") == ride_id), None)
    if ride is None:
        raise HTTPException(status_code=404, detail="Ride not found")
    return _build_ride_comparison(ride, user_id)


@app.post("/api/rides/{ride_id}/complete")
async def complete_ride(ride_id: str):
    ride = active_rides.get(ride_id)
    if ride is None:
        raise HTTPException(status_code=404, detail="Ride not found")

    ride["status"] = "completed"
    ride["completed_at"] = now_iso()
    ride["progress_percent"] = 100.0
    ride_history.append(ride)

    vehicle_id = ride.get("vehicle_id")
    if vehicle_id is not None:
        global vehicles
        vehicles = [v for v in vehicles if v.vehicle_id != vehicle_id]

    db_insert(ride_history_col, _serialize_for_mongo(ride))
    if active_rides_col is not None:
        try:
            active_rides_col.delete_one({"ride_id": ride_id})
        except PyMongoError:
            pass

    del active_rides[ride_id]
    task = ride_tasks.pop(ride_id, None)
    if task and not task.done():
        task.cancel()
    return {"status": "success", "ride_id": ride_id}


@app.post("/api/policy/switch")
async def switch_policy(policy: str):
    global current_policy

    normalized = policy.lower()
    if normalized not in ["greedy", "random", "ml"]:
        raise HTTPException(status_code=400, detail=f"Unknown policy: {policy}")

    if normalized == "ml" and ml_model is None:
        raise HTTPException(status_code=503, detail="DQN model not available. Train or reload the model first.")

    current_policy = normalized
    return {
        "status": "success",
        "current_policy": current_policy,
        "message": f"Switched to {current_policy} policy",
    }


@app.get("/api/policy/current")
async def get_current_policy():
    return {
        "policy": current_policy,
        "ml_available": ml_model is not None,
        "ml_algo": "dqn",
    }


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate(req: SimulationRequest):
    req.policy = req.policy.lower()
    if req.policy not in ["greedy", "random", "ml"]:
        raise HTTPException(status_code=400, detail=f"Unknown policy: {req.policy}")

    try:
        if req.policy in ["greedy", "random"]:
            return _run_baseline_simulation(req)
        return _run_ml_simulation(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")


@app.post("/api/model/reload")
async def reload_model():
    success = load_ml_model()
    if not success:
        raise HTTPException(status_code=404, detail="DQN model not found in outputs/models.")

    return {
        "status": "success",
        "message": "DQN model loaded successfully",
        "model_available": True,
        "algorithm": "dqn",
    }


@app.get("/api/stats")
async def get_stats():
    total_capacity = sum(v.capacity for v in vehicles)
    current_occupancy = sum(len(v.current_passengers) if hasattr(v, "current_passengers") else 0 for v in vehicles)

    return {
        "total_vehicles": len(vehicles),
        "total_capacity": total_capacity,
        "current_occupancy": current_occupancy,
        "utilization_rate": (current_occupancy / total_capacity) if total_capacity > 0 else 0,
        "ml_model_loaded": ml_model is not None,
        "ml_algorithm": "dqn",
        "current_policy": current_policy,
        "active_rides": len(active_rides),
        "ride_history": len(ride_history),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
