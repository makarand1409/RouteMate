"""
backend/main.py - FastAPI Backend for RouteMATE

This backend serves:
1. Greedy policy (nearest vehicle)
2. Random policy (random selection)
3. ML policy (trained PPO model) - auto-loads when available

Frontend connects to these endpoints to get vehicle assignments.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import simulator components
from simulator import GridCity, Vehicle
from environment.gym_environment import RideSharingEnvImproved

# Try to import ML model (will work after training completes)
try:
    from stable_baselines3 import PPO
    ML_AVAILABLE = True
    print("✓ Stable-Baselines3 imported - ML model can be loaded")
except ImportError:
    ML_AVAILABLE = False
    print("⚠ Stable-Baselines3 not available - ML predictions disabled")

# Create FastAPI app
app = FastAPI(
    title="RouteMATE API",
    description="ML-Based Ride-Sharing Backend",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
city = GridCity(size=10)
vehicles = []
ml_model = None
current_policy = "greedy"  # Default policy

# Initialize vehicles
def init_vehicles():
    global vehicles
    vehicles = []
    for i in range(5):
        vehicle = Vehicle(
            vehicle_id=i,
            initial_location=city.get_random_location(),
            capacity=4
        )
        vehicles.append(vehicle)
    print(f"✓ Initialized {len(vehicles)} vehicles")

# Load ML model if available
def load_ml_model():
    global ml_model
    model_path = "outputs/models/ppo_routemate_improved.zip"
    
    if not ML_AVAILABLE:
        print("✗ Cannot load ML model - Stable-Baselines3 not installed")
        return False
    
    if not Path(model_path).exists():
        print(f"✗ ML model not found at {model_path}")
        print("  Model will be loaded automatically after training completes")
        return False
    
    try:
        ml_model = PPO.load(model_path)
        print(f"✓ ML model loaded from {model_path}")
        return True
    except Exception as e:
        print(f"✗ Error loading ML model: {e}")
        return False

# Pydantic models for API
class Location(BaseModel):
    x: float
    y: float

class RideRequest(BaseModel):
    pickup: Location
    dropoff: Location
    policy: Optional[str] = "greedy"  # greedy, random, or ml

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

# Helper functions
def calculate_observation(request_pickup, request_dropoff, vehicles_list):
    """Create 44-feature observation for ML model"""
    obs = []
    
    # Request features (normalized)
    obs.extend([
        request_pickup[0] / 10.0,
        request_pickup[1] / 10.0,
        request_dropoff[0] / 10.0,
        request_dropoff[1] / 10.0
    ])
    
    # Vehicle features
    for vehicle in vehicles_list:
        distance = city.manhattan_distance(vehicle.current_location, request_pickup)
        current_occupancy = len(vehicle.current_passengers) if hasattr(vehicle, 'current_passengers') else 0
        available_capacity = vehicle.capacity - current_occupancy
        
        # Get queue length safely
        if hasattr(vehicle, 'request_queue'):
            queue_length = len(vehicle.request_queue)
        elif hasattr(vehicle, 'assigned_requests'):
            queue_length = sum(1 for req in vehicle.assigned_requests if not req.is_picked_up())
        else:
            queue_length = 0
        
        is_idle = 1.0 if queue_length == 0 else 0.0
        time_estimate = queue_length * 5
        
        obs.extend([
            vehicle.current_location[0] / 10.0,
            vehicle.current_location[1] / 10.0,
            current_occupancy / vehicle.capacity,
            min(queue_length, 5) / 5.0,
            distance / 20.0,
            available_capacity / vehicle.capacity,
            is_idle,
            min(time_estimate, 30) / 30.0
        ])
    
    return np.array(obs, dtype=np.float32)

def greedy_policy(pickup_location, vehicles_list):
    """Nearest vehicle policy"""
    best_vehicle = None
    min_distance = float('inf')
    
    for vehicle in vehicles_list:
        distance = city.manhattan_distance(vehicle.current_location, pickup_location)
        if distance < min_distance:
            min_distance = distance
            best_vehicle = vehicle
    
    return best_vehicle, min_distance

def random_policy(vehicles_list):
    """Random vehicle selection"""
    import random
    vehicle = random.choice(vehicles_list)
    distance = city.manhattan_distance(vehicle.current_location, (0, 0))  # Placeholder
    return vehicle, distance

def ml_policy(pickup_location, dropoff_location, vehicles_list):
    """ML model prediction"""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded yet. Try again after training completes.")
    
    # Create observation
    obs = calculate_observation(
        (pickup_location.x, pickup_location.y),
        (dropoff_location.x, dropoff_location.y),
        vehicles_list
    )
    
    # Get prediction
    action, _states = ml_model.predict(obs, deterministic=True)
    vehicle = vehicles_list[int(action)]
    distance = city.manhattan_distance(vehicle.current_location, (pickup_location.x, pickup_location.y))
    
    return vehicle, distance

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 70)
    print("RouteMATE Backend Starting...")
    print("=" * 70)
    init_vehicles()
    load_ml_model()
    print("=" * 70)
    print("✓ Backend ready!")
    print("✓ API docs available at: http://localhost:8000/docs")
    print("=" * 70)

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "service": "RouteMATE API",
        "version": "1.0.0",
        "ml_available": ml_model is not None,
        "current_policy": current_policy
    }

@app.get("/api/vehicles", response_model=List[VehicleInfo])
async def get_vehicles():
    """Get all vehicle locations and status"""
    vehicle_info = []
    
    for vehicle in vehicles:
        current_occupancy = len(vehicle.current_passengers) if hasattr(vehicle, 'current_passengers') else 0
        available_capacity = vehicle.capacity - current_occupancy
        
        # Check if idle
        queue_length = 0
        if hasattr(vehicle, 'request_queue'):
            queue_length = len(vehicle.request_queue)
        
        vehicle_info.append(VehicleInfo(
            vehicle_id=vehicle.vehicle_id,
            location=Location(x=vehicle.current_location[0], y=vehicle.current_location[1]),
            occupancy=current_occupancy,
            capacity=vehicle.capacity,
            available_capacity=available_capacity,
            is_idle=queue_length == 0
        ))
    
    return vehicle_info

@app.post("/api/request-ride", response_model=AssignmentResponse)
async def request_ride(request: RideRequest):
    """Request a ride and get vehicle assignment"""
    
    pickup = (request.pickup.x, request.pickup.y)
    dropoff = (request.dropoff.x, request.dropoff.y)
    policy = request.policy.lower()
    
    # Select policy
    try:
        if policy == "greedy":
            vehicle, distance = greedy_policy(pickup, vehicles)
            confidence = None
        elif policy == "random":
            vehicle, distance = random_policy(vehicles)
            confidence = None
        elif policy == "ml":
            vehicle, distance = ml_policy(request.pickup, request.dropoff, vehicles)
            confidence = 0.95  # Placeholder - could extract from model if needed
        else:
            raise HTTPException(status_code=400, detail=f"Unknown policy: {policy}")
        
        # Calculate estimated time (distance + current queue)
        queue_length = 0
        if hasattr(vehicle, 'request_queue'):
            queue_length = len(vehicle.request_queue)
        
        estimated_time = int(distance) + (queue_length * 5)
        
        return AssignmentResponse(
            vehicle_id=vehicle.vehicle_id,
            vehicle_location=Location(x=vehicle.current_location[0], y=vehicle.current_location[1]),
            estimated_time=estimated_time,
            distance=distance,
            policy_used=policy,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/policy/switch")
async def switch_policy(policy: str):
    """Switch active policy"""
    global current_policy
    
    if policy.lower() not in ["greedy", "random", "ml"]:
        raise HTTPException(status_code=400, detail=f"Unknown policy: {policy}")
    
    if policy.lower() == "ml" and ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not available. Please wait for training to complete."
        )
    
    current_policy = policy.lower()
    return {
        "status": "success",
        "current_policy": current_policy,
        "message": f"Switched to {current_policy} policy"
    }

@app.get("/api/policy/current")
async def get_current_policy():
    """Get current active policy"""
    return {
        "policy": current_policy,
        "ml_available": ml_model is not None
    }

@app.post("/api/model/reload")
async def reload_model():
    """Reload ML model (use after training completes)"""
    success = load_ml_model()
    
    if success:
        return {
            "status": "success",
            "message": "ML model loaded successfully",
            "model_available": True
        }
    else:
        raise HTTPException(
            status_code=404,
            detail="ML model not found. Check if training has completed."
        )

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    total_capacity = sum(v.capacity for v in vehicles)
    current_occupancy = sum(
        len(v.current_passengers) if hasattr(v, 'current_passengers') else 0 
        for v in vehicles
    )
    
    return {
        "total_vehicles": len(vehicles),
        "total_capacity": total_capacity,
        "current_occupancy": current_occupancy,
        "utilization_rate": current_occupancy / total_capacity if total_capacity > 0 else 0,
        "ml_model_loaded": ml_model is not None,
        "current_policy": current_policy
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("Starting RouteMATE Backend Server")
    print("=" * 70)
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
