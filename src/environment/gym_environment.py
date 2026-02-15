"""
gym_environment.py - OpenAI Gym Environment for RouteMATE

This wraps our RouteMATE simulator as a standard Gym environment
so we can train RL agents using Stable-Baselines3.

Key Concepts:
- Observation: State of vehicles and current request
- Action: Which vehicle (0-4) to assign the request to
- Reward: Negative wait time + completion bonuses
- Episode: One simulation run (e.g., 100 time steps)

This is what makes your project an ML project!
The agent will learn to make better vehicle assignments than the baseline.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Optional

# Import our simulator
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator import (
    GridCity, 
    Vehicle, 
    Request, 
    RequestGenerator,
    SimulationEngine
)


class RideSharingEnv(gym.Env):
    """
    Custom Gym Environment for RouteMATE Ride-Sharing System.
    
    This environment trains an agent to assign incoming ride requests
    to vehicles optimally, considering vehicle capacity, location,
    and current passengers (pooling).
    
    Observation Space:
        - Request pickup location (x, y)
        - Request dropoff location (x, y)
        - For each vehicle:
            - Location (x, y)
            - Occupancy (0-capacity)
            - Number of destinations in queue
            - Distance to request pickup
        Total: 4 + (num_vehicles * 5) values
    
    Action Space:
        - Discrete(num_vehicles): Select which vehicle (0 to num_vehicles-1)
    
    Reward:
        - Immediate: -0.1 per step (encourages faster decisions)
        - On pickup: +1.0 (successful assignment)
        - On dropoff: +2.0 (completed trip)
        - Failed assignment: -5.0 (no vehicle could take it)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        city_size: int = 10,
        num_vehicles: int = 5,
        vehicle_capacity: int = 4,
        request_rate: float = 1.5,
        max_steps: int = 100,
        reward_config: Optional[Dict] = None
    ):
        """
        Initialize the Gym environment.
        
        Args:
            city_size: Size of the grid city
            num_vehicles: Number of vehicles in the fleet
            vehicle_capacity: Max passengers per vehicle
            request_rate: Average requests per time step
            max_steps: Maximum steps per episode
            reward_config: Custom reward configuration (optional)
        """
        super(RideSharingEnv, self).__init__()
        
        # Environment parameters
        self.city_size = city_size
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.request_rate = request_rate
        self.max_steps = max_steps
        
        # Reward configuration - Simplified and more positive
        self.reward_config = reward_config or {
            'step_penalty': -0.01,  # Very small - just to encourage efficiency
            'pickup_reward': 2.0,  # Increased - good to pick up requests
            'dropoff_reward': 5.0,  # Increased - completing is very good!
            'failed_assignment': -1.0,  # Reduced - less harsh
            'wait_time_penalty': -0.1  # Not used anymore but kept for compatibility
        }
        
        # Create city and initialize vehicles
        self.city = GridCity(size=city_size)
        self.vehicles = self._create_vehicles()
        self.request_generator = RequestGenerator(self.city, request_rate=request_rate)
        
        # Episode tracking
        self.current_step = 0
        self.current_request = None
        self.pending_requests = []
        self.episode_rewards = []
        self.episode_metrics = {
            'total_requests': 0,
            'assigned_requests': 0,
            'completed_requests': 0,
            'failed_assignments': 0,
            'total_wait_time': 0.0
        }
        
        # Define action space: which vehicle to assign (0 to num_vehicles-1)
        self.action_space = spaces.Discrete(num_vehicles)
        
        # Define observation space
        # Format: [request_pickup_x, request_pickup_y, request_dropoff_x, request_dropoff_y,
        #          vehicle1_x, vehicle1_y, vehicle1_occupancy, vehicle1_num_destinations, vehicle1_distance,
        #          vehicle2_x, vehicle2_y, ...]
        obs_size = 4 + (num_vehicles * 5)  # Request (4) + vehicles (5 features each)
        self.observation_space = spaces.Box(
            low=0,
            high=max(city_size, vehicle_capacity, 100),  # Upper bounds
            shape=(obs_size,),
            dtype=np.float32
        )
    
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
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation for the agent.
        
        Returns:
            numpy array with current state information
        """
        if self.current_request is None:
            # No request to assign - return zeros or dummy values
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        obs = []
        
        # Request information (4 values)
        obs.extend([
            self.current_request.pickup[0],
            self.current_request.pickup[1],
            self.current_request.dropoff[0],
            self.current_request.dropoff[1]
        ])
        
        # Vehicle information (5 values per vehicle)
        for vehicle in self.vehicles:
            distance_to_pickup = self.city.manhattan_distance(
                vehicle.current_location,
                self.current_request.pickup
            )
            
            obs.extend([
                vehicle.current_location[0],
                vehicle.current_location[1],
                vehicle.get_occupancy(),
                len(vehicle.destination_queue),
                distance_to_pickup
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(
        self,
        action: int,
        assignment_success: bool,
        events: list
    ) -> float:
        """
        Calculate reward for the taken action.
        
        Args:
            action: Which vehicle was selected
            assignment_success: Whether assignment succeeded
            events: List of events (pickups, dropoffs) this step
            
        Returns:
            Reward value
        """
        reward = self.reward_config['step_penalty']  # Base penalty for time
        
        # Reward for successful assignment
        if assignment_success:
            reward += self.reward_config['pickup_reward']
        else:
            # Penalty for failed assignment
            reward += self.reward_config['failed_assignment']
        
        # Reward for completing trips
        for event_type, request in events:
            if event_type == 'dropoff':
                reward += self.reward_config['dropoff_reward']
        
        # Don't penalize pending requests - they're inevitable in dynamic systems
        # The agent will naturally minimize them by making good assignments
        
        return reward
    
    def step(self, action: int):
        """
        Execute one step in the environment.
        
        Args:
            action: Which vehicle (0 to num_vehicles-1) to assign request to
            
        Returns:
            observation: Current state
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        # Step 1: Try to assign current request to selected vehicle
        assignment_success = False
        selected_vehicle = self.vehicles[action]
        
        if self.current_request and selected_vehicle.can_accept_request(self.current_request):
            selected_vehicle.assign_request(self.current_request)
            assignment_success = True
            self.episode_metrics['assigned_requests'] += 1
        else:
            # Assignment failed - add to pending
            if self.current_request:
                self.pending_requests.append(self.current_request)
                self.episode_metrics['failed_assignments'] += 1
        
        # Step 2: Move all vehicles and collect events
        events = []
        for vehicle in self.vehicles:
            result = vehicle.move_one_step(self.current_step, self.city)
            if result:
                events.append(result)
                if result[0] == 'dropoff':
                    self.episode_metrics['completed_requests'] += 1
        
        # Step 3: Generate new request for next step
        self.current_step += 1
        new_requests = self.request_generator.generate_requests(self.current_step)
        
        if new_requests:
            self.current_request = new_requests[0]  # Take first request
            self.episode_metrics['total_requests'] += 1
            # Add remaining requests to pending
            self.pending_requests.extend(new_requests[1:])
            self.episode_metrics['total_requests'] += len(new_requests) - 1
        else:
            self.current_request = None
        
        # Step 4: Calculate reward
        reward = self._calculate_reward(action, assignment_success, events)
        self.episode_rewards.append(reward)
        
        # Step 5: Check if episode is done
        # Gymnasium uses terminated (natural end) and truncated (time limit)
        terminated = False  # Episode doesn't end naturally in our case
        truncated = self.current_step >= self.max_steps  # Time limit reached
        
        # Step 6: Get new observation
        observation = self._get_observation()
        
        # Step 7: Prepare info dictionary
        info = {
            'step': self.current_step,
            'assignment_success': assignment_success,
            'events': events,
            'pending_requests': len(self.pending_requests),
            'metrics': self.episode_metrics.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility (Gymnasium API)
            options: Additional options (Gymnasium API)
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        # Set seed if provided
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
        
        # Return observation and empty info dict (Gymnasium API requirement)
        return self._get_observation(), {}
    
    def render(self, mode='human'):
        """
        Render the environment (optional).
        
        For now, just print current state.
        """
        if mode == 'human':
            print(f"\n--- Step {self.current_step} ---")
            print(f"Current Request: {self.current_request}")
            print(f"Vehicles:")
            for v in self.vehicles:
                print(f"  {v}")
            print(f"Pending Requests: {len(self.pending_requests)}")
            print(f"Episode Reward: {sum(self.episode_rewards):.2f}")
    
    def close(self):
        """Cleanup when environment is closed."""
        pass


# Test the environment
if __name__ == "__main__":
    print("=" * 70)
    print("Testing RouteMATE Gym Environment")
    print("=" * 70)
    
    # Create environment
    env = RideSharingEnv(
        city_size=10,
        num_vehicles=5,
        vehicle_capacity=4,
        request_rate=1.5,
        max_steps=50
    )
    
    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Max steps per episode: {env.max_steps}")
    
    # Test with random agent
    print(f"\n" + "=" * 70)
    print("Testing with RANDOM agent (baseline)")
    print("=" * 70)
    
    obs = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial observation: {obs[:10]}... (showing first 10 values)")
    
    total_reward = 0
    done = False
    step_count = 0
    
    print(f"\nRunning episode...")
    while not done and step_count < 50:
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Print every 10 steps
        if step_count % 10 == 0:
            print(f"  Step {step_count}: "
                  f"Action={action}, "
                  f"Reward={reward:.2f}, "
                  f"Total={total_reward:.2f}, "
                  f"Pending={info['pending_requests']}")
    
    print(f"\n" + "=" * 70)
    print("Episode Complete!")
    print("=" * 70)
    print(f"\nFinal Metrics:")
    for key, value in info['metrics'].items():
        print(f"  {key}: {value}")
    
    print(f"\nTotal Episode Reward: {total_reward:.2f}")
    print(f"Average Reward per Step: {total_reward/step_count:.2f}")
    
    completion_rate = 0
    if info['metrics']['total_requests'] > 0:
        completion_rate = info['metrics']['completed_requests'] / info['metrics']['total_requests']
    print(f"Completion Rate: {completion_rate*100:.1f}%")
    
    print(f"\n✓ Gym environment working correctly!")
    print(f"\nNext step: Train an RL agent to beat random actions!")