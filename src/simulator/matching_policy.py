"""
matching_policy.py - Vehicle-Request Matching Strategies

This module decides WHICH vehicle should serve WHICH request.
This is the core decision-making component that ML will eventually replace!

Matching Strategies:
1. Nearest Available Vehicle (Baseline) - What we implement here
2. RL Agent (Future) - What we'll train in Phase 3

Key Insight:
The baseline heuristic is simple and explainable, but not optimal.
The RL agent will learn to make better decisions by considering:
- Vehicle capacity and current passengers
- Future request patterns
- System-wide efficiency

For exams: "We compare our learned RL policy against this baseline heuristic"
"""

from typing import List, Optional, Tuple

# Handle both package import and standalone testing
try:
    from .request import Request
    from .vehicle import Vehicle
    from .grid_city import GridCity
except ImportError:
    from request import Request
    from vehicle import Vehicle
    from grid_city import GridCity


class MatchingPolicy:
    """
    Base class for matching policies.
    
    A matching policy decides which vehicle should serve a new request.
    This is an abstract class that can have different implementations:
    - NearestVehiclePolicy (baseline heuristic)
    - RLAgentPolicy (learned policy - Phase 3)
    """
    
    def __init__(self, city: GridCity):
        """
        Initialize the matching policy.
        
        Args:
            city: GridCity for distance calculations
        """
        self.city = city
        self.total_assignments = 0
        self.failed_assignments = 0  # Requests that couldn't be assigned
    
    def match_request(
        self, 
        request: Request, 
        vehicles: List[Vehicle]
    ) -> Optional[Vehicle]:
        """
        Match a request to a vehicle.
        
        Args:
            request: The ride request to assign
            vehicles: List of all vehicles
            
        Returns:
            The selected vehicle, or None if no vehicle available
        """
        raise NotImplementedError("Subclasses must implement match_request")
    
    def get_statistics(self) -> dict:
        """Get statistics about matching performance."""
        success_rate = 0.0
        if self.total_assignments > 0:
            success_rate = (self.total_assignments - self.failed_assignments) / self.total_assignments
        
        return {
            'total_assignments': self.total_assignments,
            'failed_assignments': self.failed_assignments,
            'success_rate': success_rate
        }


class NearestVehiclePolicy(MatchingPolicy):
    """
    Baseline Policy: Assign request to nearest available vehicle.
    
    Strategy:
    1. Find all vehicles that can accept the request (have capacity)
    2. Among those, pick the one closest to the pickup location
    3. If no vehicle can accept, request fails (waits for next time step)
    
    Pros:
    - Simple and explainable
    - Fast computation (O(n) where n = number of vehicles)
    - Minimizes immediate pickup time
    
    Cons:
    - Doesn't consider future requests
    - Doesn't optimize for system-wide efficiency
    - Might send a vehicle far from its current route
    
    This is what ML will try to beat!
    """
    
    def match_request(
        self, 
        request: Request, 
        vehicles: List[Vehicle]
    ) -> Optional[Vehicle]:
        """
        Assign request to the nearest available vehicle.
        
        Args:
            request: The ride request to assign
            vehicles: List of all vehicles in the system
            
        Returns:
            The nearest available vehicle, or None if none available
        """
        self.total_assignments += 1
        
        # Step 1: Filter vehicles that can accept this request
        available_vehicles = [
            v for v in vehicles 
            if v.can_accept_request(request)
        ]
        
        # Step 2: If no vehicles available, assignment fails
        if not available_vehicles:
            self.failed_assignments += 1
            return None
        
        # Step 3: Find the nearest vehicle to the pickup location
        nearest_vehicle = min(
            available_vehicles,
            key=lambda v: self.city.manhattan_distance(v.current_location, request.pickup)
        )
        
        return nearest_vehicle
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return "NearestVehiclePolicy(baseline)"


class RandomPolicy(MatchingPolicy):
    """
    Random Policy: Assign to a random available vehicle.
    
    This is even simpler than nearest vehicle, useful for:
    - Testing the simulation framework
    - Worst-case baseline comparison
    - Verifying ML agent actually learns something useful
    
    If your ML agent can't beat random assignment, something is wrong!
    """
    
    def match_request(
        self, 
        request: Request, 
        vehicles: List[Vehicle]
    ) -> Optional[Vehicle]:
        """
        Assign request to a random available vehicle.
        
        Args:
            request: The ride request to assign
            vehicles: List of all vehicles
            
        Returns:
            Random available vehicle, or None if none available
        """
        import random
        
        self.total_assignments += 1
        
        # Filter available vehicles
        available_vehicles = [
            v for v in vehicles 
            if v.can_accept_request(request)
        ]
        
        # If no vehicles available, fail
        if not available_vehicles:
            self.failed_assignments += 1
            return None
        
        # Pick random vehicle
        return random.choice(available_vehicles)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return "RandomPolicy(random_baseline)"


# Test the matching policies
if __name__ == "__main__":
    print("Testing Matching Policies...")
    print("=" * 60)
    
    # Setup: Create city, vehicles, and requests
    city = GridCity(size=10)
    print(f"\n1. Created city: {city}")
    
    # Create 3 vehicles at different locations
    vehicles = [
        Vehicle(vehicle_id=1, initial_location=(0, 0), capacity=4),
        Vehicle(vehicle_id=2, initial_location=(5, 5), capacity=4),
        Vehicle(vehicle_id=3, initial_location=(9, 9), capacity=4),
    ]
    print(f"\n2. Created {len(vehicles)} vehicles:")
    for v in vehicles:
        print(f"   {v}")
    
    # Create a request at (2, 2)
    request = Request(
        request_id=1,
        pickup=(2, 2),
        dropoff=(7, 7),
        request_time=0.0
    )
    print(f"\n3. New request: {request}")
    print(f"   Pickup at: {request.pickup}")
    
    # Test Nearest Vehicle Policy
    print(f"\n4. Testing NearestVehiclePolicy:")
    print("   " + "-" * 50)
    nearest_policy = NearestVehiclePolicy(city)
    
    # Calculate distances for demonstration
    print("   Distance from each vehicle to pickup:")
    for v in vehicles:
        dist = city.manhattan_distance(v.current_location, request.pickup)
        print(f"   - Vehicle {v.vehicle_id} at {v.current_location}: {dist} blocks")
    
    # Match the request
    selected = nearest_policy.match_request(request, vehicles)
    print(f"\n   Selected vehicle: {selected.vehicle_id if selected else 'None'}")
    print(f"   Reason: Closest to pickup location")
    
    # Test Random Policy
    print(f"\n5. Testing RandomPolicy:")
    print("   " + "-" * 50)
    random_policy = RandomPolicy(city)
    
    # Try matching 5 times to see randomness
    print("   Running 5 random assignments:")
    for i in range(5):
        # Create a new request for each test
        test_request = Request(
            request_id=100+i,
            pickup=(2, 2),
            dropoff=(7, 7),
            request_time=0.0
        )
        selected = random_policy.match_request(test_request, vehicles)
        print(f"   - Attempt {i+1}: Vehicle {selected.vehicle_id if selected else 'None'}")
    
    # Test with no available vehicles
    print(f"\n6. Testing edge case: No available vehicles")
    print("   " + "-" * 50)
    
    # Make all vehicles unavailable by filling them up
    for v in vehicles:
        v.current_passengers = [None] * v.capacity  # Hack to fill capacity
    
    unavailable_request = Request(
        request_id=200,
        pickup=(5, 5),
        dropoff=(8, 8),
        request_time=0.0
    )
    
    selected = nearest_policy.match_request(unavailable_request, vehicles)
    print(f"   Selected vehicle: {selected}")
    print(f"   Expected: None (all vehicles full)")
    
    # Statistics
    print(f"\n7. Policy Statistics:")
    print("   " + "-" * 50)
    nearest_stats = nearest_policy.get_statistics()
    print(f"   NearestVehiclePolicy:")
    for key, value in nearest_stats.items():
        print(f"   - {key}: {value}")
    
    random_stats = random_policy.get_statistics()
    print(f"\n   RandomPolicy:")
    for key, value in random_stats.items():
        print(f"   - {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ Matching policies working correctly!")
    print("\nKey Insights:")
    print("1. NearestVehicle is deterministic (always same result)")
    print("2. Random is stochastic (different results each time)")
    print("3. Both handle edge cases (no available vehicles)")
    print("4. These are baselines - ML will try to beat them!")
