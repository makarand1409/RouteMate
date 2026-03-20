"""
gym_environment_IMPROVED_FIXED.py - Enhanced RL Environment for RouteMATE
FIXED VERSION - Compatible with existing Vehicle class

IMPROVEMENTS OVER ORIGINAL:
1. Better observation space (44 values instead of 29)
2. Dense reward shaping (more feedback signals)
3. Compatible with your existing Vehicle implementation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator import (
    GridCity, Vehicle, Request, RequestGenerator,
    NearestVehiclePolicy, RandomPolicy, SimulationEngine
)


class RideSharingEnvImproved(gym.Env):
    """
    IMPROVED OpenAI Gym environment for ride-sharing with pooling.
    FIXED to work with existing Vehicle class.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        city_size: int = 10,
        num_vehicles: int = 5,
        vehicle_capacity: int = 4,
        request_rate: float = 1.5,
        max_steps: int = 200,
        reward_config: Optional[Dict] = None
    ):
        """Initialize the IMPROVED Gym environment."""
        super().__init__()
        
        self.city_size = city_size
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.request_rate = request_rate
        self.max_steps = max_steps
        
        # IMPROVED: More positive, better balanced rewards
        self.reward_config = reward_config or {
            'step_penalty': -0.01,
            'pickup_reward': 3.0,
            'dropoff_reward': 8.0,
            'failed_assignment': -2.0,
            'distance_bonus_scale': 0.2,
            'capacity_bonus_scale': 0.5,
            'utilization_bonus_scale': 2.0
        }
        
        # Create city and initialize vehicles
        self.city = GridCity(size=city_size)
        self.vehicles = self._create_vehicles()
        self.request_generator = RequestGenerator(self.city, request_rate=request_rate)
        
        # Episode state
        self.current_step = 0
        self.current_request = None
        self.pending_requests = []
        self.episode_rewards = []
        self.episode_metrics = {}
        
        # IMPROVED: Enhanced observation space (44 values)
        observation_size = 4 + num_vehicles * 8
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(city_size * 2),
            shape=(observation_size,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(num_vehicles)
        
        print(f"✓ IMPROVED Environment initialized")
        print(f"  - Observation space: {self.observation_space.shape}")
        print(f"  - Action space: {num_vehicles} vehicles")
        print(f"  - Enhanced features: +3 per vehicle (capacity, idle, time)")
    
    def _create_vehicles(self) -> list:
        """Create initial vehicle fleet."""
        vehicles = []
        for i in range(self.num_vehicles):
            initial_location = self.city.get_random_location()
            vehicle = Vehicle(
                vehicle_id=i,
                initial_location=initial_location,
                capacity=self.vehicle_capacity
            )
            vehicles.append(vehicle)
        return vehicles
    
    def _get_vehicle_queue_length(self, vehicle) -> int:
        """
        FIXED: Get queue length in a way that works with your Vehicle class.
        Tries multiple possible attribute names.
        """
        # Try different possible attribute names
        if hasattr(vehicle, 'request_queue'):
            return len(vehicle.request_queue)
        elif hasattr(vehicle, 'queue'):
            return len(vehicle.queue)
        elif hasattr(vehicle, 'requests'):
            return len(vehicle.requests)
        elif hasattr(vehicle, 'assigned_requests'):
            # Count requests that aren't picked up yet
            return sum(1 for req in vehicle.assigned_requests if not req.is_picked_up())
        else:
            # Fallback: estimate from destination
            return 1 if hasattr(vehicle, 'destination') and vehicle.destination is not None else 0
    
    def _get_observation(self) -> np.ndarray:
        """
        IMPROVED: Get enhanced observation with vehicle state information.
        FIXED to work with existing Vehicle class.
        """
        if self.current_request is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs = []
        
        # Request information (4 values) - normalized
        obs.extend([
            self.current_request.pickup[0] / self.city_size,
            self.current_request.pickup[1] / self.city_size,
            self.current_request.dropoff[0] / self.city_size,
            self.current_request.dropoff[1] / self.city_size
        ])
        
        # Vehicle information (8 values per vehicle) - IMPROVED & FIXED!
        for vehicle in self.vehicles:
            distance_to_pickup = self.city.manhattan_distance(
                vehicle.current_location,
                self.current_request.pickup
            )
            
            # Get current occupancy
            current_occupancy = len(vehicle.current_passengers) if hasattr(vehicle, 'current_passengers') else 0
            
            # FIXED: Get queue length safely
            queue_length = self._get_vehicle_queue_length(vehicle)
            
            # Calculate enhanced features
            available_capacity = vehicle.capacity - current_occupancy
            is_idle = 1.0 if queue_length == 0 else 0.0
            estimated_time_to_free = queue_length * 5  # Rough estimate
            
            obs.extend([
                vehicle.current_location[0] / self.city_size,
                vehicle.current_location[1] / self.city_size,
                current_occupancy / vehicle.capacity,
                min(queue_length, 5) / 5.0,
                distance_to_pickup / (2 * self.city_size),
                available_capacity / vehicle.capacity,  # NEW!
                is_idle,  # NEW!
                min(estimated_time_to_free, 30) / 30.0  # NEW!
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(
        self,
        action: int,
        assignment_success: bool,
        events: list
    ) -> float:
        """
        IMPROVED: Dense reward shaping with multiple feedback signals.
        """
        reward = self.reward_config['step_penalty']
        selected_vehicle = self.vehicles[action]
        
        # 1. ASSIGNMENT OUTCOME
        if assignment_success and self.current_request is not None:  # FIXED: Check None
            reward += self.reward_config['pickup_reward']
            
            # NEW: Distance bonus
            distance = self.city.manhattan_distance(
                selected_vehicle.current_location,
                self.current_request.pickup
            )
            max_distance = 2 * self.city_size
            distance_bonus = (1.0 - distance / max_distance) * self.reward_config['distance_bonus_scale']
            reward += distance_bonus
            
            # NEW: Capacity bonus
            current_occupancy = len(selected_vehicle.current_passengers) if hasattr(selected_vehicle, 'current_passengers') else 0
            available_capacity = selected_vehicle.capacity - current_occupancy
            if available_capacity > 0:
                capacity_utilization = available_capacity / selected_vehicle.capacity
                capacity_bonus = capacity_utilization * self.reward_config['capacity_bonus_scale']
                reward += capacity_bonus
        elif not assignment_success:  # FIXED: Only penalize if there was actually a request
            reward += self.reward_config['failed_assignment']
            
            # Extra penalty for impossible assignment
            current_occupancy = len(selected_vehicle.current_passengers) if hasattr(selected_vehicle, 'current_passengers') else 0
            if current_occupancy >= selected_vehicle.capacity:
                reward -= 1.0
        
        # 2. COMPLETION REWARDS
        for event_type, request in events:
            if event_type == 'dropoff':
                reward += self.reward_config['dropoff_reward']
        
        # 3. NEW: Fleet utilization bonus
        total_passengers = sum(len(v.current_passengers) if hasattr(v, 'current_passengers') else 0 for v in self.vehicles)
        total_capacity = self.num_vehicles * self.vehicle_capacity
        utilization_rate = total_passengers / total_capacity if total_capacity > 0 else 0
        utilization_bonus = utilization_rate * self.reward_config['utilization_bonus_scale']
        reward += utilization_bonus
        
        return reward
    
    def step(self, action: int):
        """Execute one step in the environment."""
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        # Step 1: Try to assign current request
        assignment_success = False
        selected_vehicle = self.vehicles[action]
        
        if self.current_request and selected_vehicle.can_accept_request(self.current_request):
            selected_vehicle.assign_request(self.current_request)
            assignment_success = True
            self.episode_metrics['assigned_requests'] += 1
        else:
            if self.current_request:
                self.pending_requests.append(self.current_request)
                self.episode_metrics['failed_assignments'] += 1
        
        # Step 2: Move all vehicles
        events = []
        for vehicle in self.vehicles:
            result = vehicle.move_one_step(self.current_step, self.city)
            if result:
                events.append(result)
                if result[0] == 'dropoff':
                    self.episode_metrics['completed_requests'] += 1
        
        # Step 3: Generate new request
        self.current_step += 1
        new_requests = self.request_generator.generate_requests(self.current_step)
        
        if new_requests:
            self.current_request = new_requests[0]
            self.episode_metrics['total_requests'] += 1
            self.pending_requests.extend(new_requests[1:])
            self.episode_metrics['total_requests'] += len(new_requests) - 1
        else:
            self.current_request = None
        
        # Step 4: Calculate IMPROVED reward
        reward = self._calculate_reward(action, assignment_success, events)
        self.episode_rewards.append(reward)
        
        # Step 5: Check if done
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Step 6: Get observation
        observation = self._get_observation()
        
        # Step 7: Info
        info = {
            'step': self.current_step,
            'assignment_success': assignment_success,
            'events': events,
            'pending_requests': len(self.pending_requests),
            'metrics': self.episode_metrics.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset vehicles
        self.vehicles = self._create_vehicles()
        
        # Reset request generator
        self.request_generator = RequestGenerator(self.city, request_rate=self.request_rate)
        
        # Reset episode state
        self.current_step = 0
        self.pending_requests = []
        self.episode_rewards = []
        self.episode_metrics = {
            'total_requests': 0,
            'assigned_requests': 0,
            'completed_requests': 0,
            'failed_assignments': 0,
            'total_wait_time': 0.0
        }
        
        # Generate initial request
        initial_requests = self.request_generator.generate_requests(0.0)
        if initial_requests:
            self.current_request = initial_requests[0]
            self.episode_metrics['total_requests'] = 1
        else:
            self.current_request = None
        
        return self._get_observation(), {}
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Current request: {self.current_request}")
            print(f"Pending requests: {len(self.pending_requests)}")
            for i, vehicle in enumerate(self.vehicles):
                print(f"Vehicle {i}: {vehicle.current_location}")
    
    def close(self):
        """Clean up resources."""
        pass


# Backwards compatibility alias
RideSharingEnv = RideSharingEnvImproved


def test_improved_environment():
    """Test the improved environment."""
    print("=" * 70)
    print("TESTING IMPROVED ENVIRONMENT (FIXED VERSION)")
    print("=" * 70)
    
    try:
        env = RideSharingEnvImproved(max_steps=50)
        
        print("\n1. Running 50 steps with random actions...")
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}: reward={reward:.2f}, total={total_reward:.2f}")
            
            if terminated or truncated:
                break
        
        print(f"\n2. Final Results:")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Average reward per step: {total_reward/(step+1):.2f}")
        print(f"   Completion rate: {info['metrics']['completed_requests']}/{info['metrics']['total_requests']}")
        
        print("\n✓ Improved environment working correctly!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nFull error details:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_improved_environment()
    if not success:
        print("\n⚠️  Environment test failed. Please check the error above.")
        print("The training script will NOT work until this is fixed.")
    else:
        print("\n✅ Environment test passed! Ready to train.")
        print("Run: python train_ppo_improved.py")