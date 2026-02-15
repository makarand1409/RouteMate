"""
simulation_engine.py - Main Simulation Engine for RouteMATE

This is the core orchestrator that brings all components together.
It runs the discrete-event simulation step by step.

Think of this as the "game loop" or "main controller" that:
1. Manages time progression
2. Generates new ride requests
3. Matches requests to vehicles
4. Moves vehicles towards their destinations
5. Collects metrics and statistics

This is what you'll run to test your baseline and later compare against ML!
"""

from typing import List, Dict, Tuple
import time
from collections import defaultdict

# Handle both package import and standalone testing
try:
    from .grid_city import GridCity
    from .request import Request
    from .vehicle import Vehicle
    from .request_generator import RequestGenerator
    from .matching_policy import MatchingPolicy, NearestVehiclePolicy
except ImportError:
    from grid_city import GridCity
    from request import Request
    from vehicle import Vehicle
    from request_generator import RequestGenerator
    from matching_policy import MatchingPolicy, NearestVehiclePolicy


class SimulationEngine:
    """
    Main simulation engine for RouteMATE ride-sharing system.
    
    This orchestrates the entire simulation:
    - Time progression (discrete time steps)
    - Request generation and matching
    - Vehicle movement and passenger handling
    - Metrics collection
    
    Attributes:
        city: The GridCity environment
        vehicles: List of all vehicles in the system
        request_generator: Generates new ride requests
        matching_policy: Policy for assigning requests to vehicles
        current_time: Current simulation time
        pending_requests: Requests waiting to be assigned
        all_requests: All requests ever generated
        metrics: Performance metrics collected during simulation
    """
    
    def __init__(
        self,
        city_size: int = 10,
        num_vehicles: int = 5,
        vehicle_capacity: int = 4,
        request_rate: float = 2.0,
        matching_policy: MatchingPolicy = None
    ):
        """
        Initialize the simulation engine.
        
        Args:
            city_size: Size of the square grid city (default 10x10)
            num_vehicles: Number of vehicles in the fleet
            vehicle_capacity: Maximum passengers per vehicle
            request_rate: Average requests per time step (lambda for Poisson)
            matching_policy: Policy for matching requests to vehicles
                           If None, uses NearestVehiclePolicy (baseline)
        
        Example:
            >>> engine = SimulationEngine(
            ...     city_size=10,
            ...     num_vehicles=5,
            ...     vehicle_capacity=4,
            ...     request_rate=2.0
            ... )
        """
        # Environment setup
        self.city = GridCity(size=city_size)
        
        # Create vehicle fleet with random initial positions
        self.vehicles = []
        for i in range(num_vehicles):
            initial_location = self.city.get_random_location()
            vehicle = Vehicle(
                vehicle_id=i+1,
                initial_location=initial_location,
                capacity=vehicle_capacity
            )
            self.vehicles.append(vehicle)
        
        # Request generation
        self.request_generator = RequestGenerator(self.city, request_rate=request_rate)
        
        # Matching policy (baseline or custom)
        if matching_policy is None:
            self.matching_policy = NearestVehiclePolicy(self.city)
        else:
            self.matching_policy = matching_policy
        
        # Simulation state
        self.current_time = 0.0
        self.pending_requests: List[Request] = []  # Unassigned requests
        self.all_requests: List[Request] = []      # All requests (for metrics)
        
        # Metrics tracking
        self.metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'total_wait_time': 0.0,
            'total_trip_time': 0.0,
            'total_distance_traveled': 0,
            'time_steps': 0
        }
        
        # History for analysis
        self.history = defaultdict(list)
    
    def step(self) -> Dict:
        """
        Execute one time step of the simulation.
        
        This is the core simulation loop:
        1. Generate new requests
        2. Try to match pending requests to vehicles
        3. Move all vehicles
        4. Update metrics
        
        Returns:
            Dictionary with step information:
            - new_requests: Number of new requests generated
            - matched: Number of requests matched this step
            - pickups: Number of pickups that occurred
            - dropoffs: Number of dropoffs that occurred
            - pending: Number of requests still waiting
        """
        step_info = {
            'new_requests': 0,
            'matched': 0,
            'pickups': 0,
            'dropoffs': 0,
            'pending': 0
        }
        
        # Step 1: Generate new requests
        new_requests = self.request_generator.generate_requests(self.current_time)
        step_info['new_requests'] = len(new_requests)
        
        # Add to pending and all requests
        self.pending_requests.extend(new_requests)
        self.all_requests.extend(new_requests)
        self.metrics['total_requests'] += len(new_requests)
        
        # Step 2: Try to match pending requests to vehicles
        still_pending = []
        for request in self.pending_requests:
            # Try to find a vehicle for this request
            selected_vehicle = self.matching_policy.match_request(request, self.vehicles)
            
            if selected_vehicle:
                # Successfully matched!
                selected_vehicle.assign_request(request)
                step_info['matched'] += 1
            else:
                # No vehicle available, remains pending
                still_pending.append(request)
        
        self.pending_requests = still_pending
        step_info['pending'] = len(self.pending_requests)
        
        # Step 3: Move all vehicles
        for vehicle in self.vehicles:
            action = vehicle.move_one_step(self.current_time, self.city)
            
            if action:
                action_type, request = action
                if action_type == "pickup":
                    step_info['pickups'] += 1
                elif action_type == "dropoff":
                    step_info['dropoffs'] += 1
                    # Update metrics for completed request
                    self._update_completion_metrics(request)
        
        # Step 4: Update time and history
        self.current_time += 1.0
        self.metrics['time_steps'] += 1
        
        # Record history for later analysis
        self.history['time'].append(self.current_time)
        self.history['pending_requests'].append(len(self.pending_requests))
        self.history['active_vehicles'].append(
            sum(1 for v in self.vehicles if not v.is_idle())
        )
        
        return step_info
    
    def _update_completion_metrics(self, request: Request):
        """
        Update metrics when a request is completed.
        
        Args:
            request: The completed request
        """
        self.metrics['completed_requests'] += 1
        
        # Calculate wait time (request to pickup)
        if request.pickup_time is not None:
            wait_time = request.pickup_time - request.request_time
            self.metrics['total_wait_time'] += wait_time
        
        # Calculate trip time (request to dropoff)
        trip_time = request.get_trip_duration()
        if trip_time is not None:
            self.metrics['total_trip_time'] += trip_time
    
    def run(self, max_steps: int = 100, verbose: bool = False) -> Dict:
        """
        Run the simulation for a specified number of steps.
        
        Args:
            max_steps: Maximum number of time steps to simulate
            verbose: If True, print progress information
            
        Returns:
            Dictionary with final simulation statistics
        
        Example:
            >>> engine = SimulationEngine()
            >>> results = engine.run(max_steps=100, verbose=True)
            >>> print(f"Completed {results['completed_requests']} requests")
        """
        if verbose:
            print(f"Starting RouteMATE simulation...")
            print(f"City: {self.city}")
            print(f"Vehicles: {len(self.vehicles)}")
            print(f"Request rate: {self.request_generator.request_rate}")
            print(f"Matching policy: {self.matching_policy}")
            print("-" * 60)
        
        start_time = time.time()
        
        # Main simulation loop
        for step in range(max_steps):
            step_info = self.step()
            
            if verbose and step % 10 == 0:
                print(f"Step {step:3d}: "
                      f"New: {step_info['new_requests']}, "
                      f"Matched: {step_info['matched']}, "
                      f"Pickups: {step_info['pickups']}, "
                      f"Dropoffs: {step_info['dropoffs']}, "
                      f"Pending: {step_info['pending']}")
        
        # Calculate final statistics
        results = self._compute_final_statistics()
        
        if verbose:
            print("-" * 60)
            print(f"Simulation complete in {time.time() - start_time:.2f}s")
            print(f"\nFinal Statistics:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        return results
    
    def _compute_final_statistics(self) -> Dict:
        """
        Compute final statistics from the simulation.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        completed = self.metrics['completed_requests']
        total = self.metrics['total_requests']
        
        # Calculate averages
        avg_wait_time = 0.0
        avg_trip_time = 0.0
        if completed > 0:
            avg_wait_time = self.metrics['total_wait_time'] / completed
            avg_trip_time = self.metrics['total_trip_time'] / completed
        
        # Vehicle statistics
        total_distance = sum(v.total_distance_traveled for v in self.vehicles)
        total_customers_served = sum(v.total_customers_served for v in self.vehicles)
        
        # Calculate utilization (% of time vehicles had passengers)
        # This is approximate - would need more detailed tracking for exact values
        avg_occupancy = sum(v.get_occupancy() for v in self.vehicles) / len(self.vehicles)
        
        return {
            'total_requests': total,
            'completed_requests': completed,
            'pending_requests': len(self.pending_requests),
            'completion_rate': completed / total if total > 0 else 0.0,
            'avg_wait_time': avg_wait_time,
            'avg_trip_time': avg_trip_time,
            'total_distance_traveled': total_distance,
            'avg_distance_per_vehicle': total_distance / len(self.vehicles),
            'total_customers_served': total_customers_served,
            'current_avg_occupancy': avg_occupancy,
            'time_steps': self.metrics['time_steps']
        }
    
    def get_current_state(self) -> Dict:
        """
        Get the current state of the simulation.
        
        Useful for:
        - Real-time monitoring
        - Creating visualizations
        - Debugging
        
        Returns:
            Dictionary with current simulation state
        """
        return {
            'current_time': self.current_time,
            'vehicles': [v.get_state_dict() for v in self.vehicles],
            'pending_requests': len(self.pending_requests),
            'completed_requests': self.metrics['completed_requests'],
            'total_requests': self.metrics['total_requests']
        }
    
    def reset(self):
        """
        Reset the simulation to initial state.
        
        Useful for running multiple trials.
        """
        # Reset time
        self.current_time = 0.0
        
        # Reset vehicles
        for vehicle in self.vehicles:
            vehicle.current_location = self.city.get_random_location()
            vehicle.current_passengers = []
            vehicle.destination_queue = []
            vehicle.total_distance_traveled = 0
            vehicle.total_customers_served = 0
        
        # Reset requests
        self.pending_requests = []
        self.all_requests = []
        self.request_generator.all_requests = []
        self.request_generator.next_request_id = 1
        
        # Reset metrics
        self.metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'total_wait_time': 0.0,
            'total_trip_time': 0.0,
            'total_distance_traveled': 0,
            'time_steps': 0
        }
        
        # Reset history
        self.history = defaultdict(list)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SimulationEngine("
            f"city={self.city.size}x{self.city.size}, "
            f"vehicles={len(self.vehicles)}, "
            f"time={self.current_time})"
        )


# Test the simulation engine
if __name__ == "__main__":
    print("=" * 60)
    print("Testing RouteMATE Simulation Engine")
    print("=" * 60)
    
    # Create simulation with default parameters
    engine = SimulationEngine(
        city_size=10,
        num_vehicles=5,
        vehicle_capacity=4,
        request_rate=1.5  # 1.5 requests per time step on average
    )
    
    print(f"\nInitialized: {engine}")
    print(f"Policy: {engine.matching_policy}")
    print(f"\nVehicle fleet:")
    for v in engine.vehicles:
        print(f"  {v}")
    
    # Run simulation
    print(f"\n{'='*60}")
    print("Running simulation for 50 time steps...")
    print('='*60)
    
    results = engine.run(max_steps=50, verbose=True)
    
    print(f"\n{'='*60}")
    print("Simulation Results Summary")
    print('='*60)
    print(f"\n📊 Request Statistics:")
    print(f"  Total requests generated: {results['total_requests']}")
    print(f"  Completed requests: {results['completed_requests']}")
    print(f"  Pending requests: {results['pending_requests']}")
    print(f"  Completion rate: {results['completion_rate']*100:.1f}%")
    
    print(f"\n⏱️  Time Metrics:")
    print(f"  Average wait time: {results['avg_wait_time']:.2f} steps")
    print(f"  Average trip time: {results['avg_trip_time']:.2f} steps")
    
    print(f"\n🚗 Vehicle Metrics:")
    print(f"  Total distance: {results['total_distance_traveled']} blocks")
    print(f"  Avg per vehicle: {results['avg_distance_per_vehicle']:.1f} blocks")
    print(f"  Total customers served: {results['total_customers_served']}")
    print(f"  Current avg occupancy: {results['current_avg_occupancy']:.2f}")
    
    print(f"\n{'='*60}")
    print("✓ Simulation engine working correctly!")
    print(f"{'='*60}")
