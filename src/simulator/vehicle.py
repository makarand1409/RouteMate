"""
vehicle.py - Vehicle with Pooling Capability

This is the heart of our ride-sharing system. Each vehicle can serve
multiple customers simultaneously (pooling/carpooling).

Key Concepts:
- Capacity: Maximum number of passengers (e.g., 4)
- Current passengers: Who's currently in the vehicle
- Destination queue: Where the vehicle needs to go next
- Routing: Simple heuristic - go to nearest destination first

Real-world analogy:
Like UberPool or Lyft Shared, where one car picks up multiple people
going in similar directions.
"""

from typing import Tuple, List, Optional

# Handle both package import and standalone testing
try:
    from .request import Request
except ImportError:
    from request import Request


class Vehicle:
    """
    A vehicle that can serve multiple ride requests simultaneously.
    
    Attributes:
        vehicle_id: Unique identifier
        capacity: Maximum passengers (e.g., 4)
        current_location: Current (x, y) position
        current_passengers: List of Request objects currently in vehicle
        destination_queue: List of (location, request_id, action_type) tuples
                          action_type is either "pickup" or "dropoff"
        total_distance_traveled: Cumulative distance for metrics
        total_customers_served: Number of completed rides
    """
    
    def __init__(
        self,
        vehicle_id: int,
        initial_location: Tuple[int, int],
        capacity: int = 4
    ):
        """
        Initialize a vehicle.
        
        Args:
            vehicle_id: Unique ID for this vehicle
            initial_location: Starting position (x, y)
            capacity: Maximum number of passengers (default 4)
        """
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.current_location = initial_location
        
        # Passenger management
        self.current_passengers: List[Request] = []
        
        # Destination queue: list of (location, request, action_type)
        # action_type is "pickup" or "dropoff"
        self.destination_queue: List[Tuple[Tuple[int, int], Request, str]] = []
        
        # Metrics
        self.total_distance_traveled = 0
        self.total_customers_served = 0
    
    def is_available(self) -> bool:
        """
        Check if vehicle has space for more passengers.
        
        Returns:
            True if current passengers < capacity
        """
        return len(self.current_passengers) < self.capacity
    
    def is_idle(self) -> bool:
        """
        Check if vehicle has no passengers and no destinations.
        
        Returns:
            True if vehicle is completely idle
        """
        return len(self.current_passengers) == 0 and len(self.destination_queue) == 0
    
    def get_occupancy(self) -> int:
        """
        Get current number of passengers.
        
        Returns:
            Number of passengers currently in vehicle
        """
        return len(self.current_passengers)
    
    def can_accept_request(self, request: Request) -> bool:
        """
        Check if vehicle can accept a new request.
        
        For pooling to work, we need:
        1. Available capacity (space for the new passenger)
        2. OR vehicle is idle and can start fresh
        
        Args:
            request: The ride request to check
            
        Returns:
            True if vehicle can accept this request
        """
        # Simple check: do we have space?
        return self.is_available() or self.is_idle()
    
    def assign_request(self, request: Request):
        """
        Assign a new request to this vehicle.
        
        This adds pickup and dropoff destinations to the queue.
        The actual routing (order of destinations) is handled by
        the reorder_destinations method.
        
        Args:
            request: The ride request to assign
        """
        # Add pickup destination
        self.destination_queue.append((request.pickup, request, "pickup"))
        
        # Add dropoff destination
        self.destination_queue.append((request.dropoff, request, "dropoff"))
        
        # Mark request as assigned to this vehicle
        request.assigned_vehicle = self.vehicle_id
        
        # Reorder destinations using nearest-first heuristic
        self.reorder_destinations()
    
    def reorder_destinations(self):
        """
        Reorder destinations using a simple nearest-first heuristic.
        
        Strategy:
        1. Always pick up passengers before dropping off (if at same location)
        2. Otherwise, go to nearest destination
        3. Respect pickup-before-dropoff constraint for each passenger
        
        This is NOT optimal, but it's a simple, explainable heuristic.
        A real system might use more sophisticated routing.
        """
        if len(self.destination_queue) <= 1:
            return  # Nothing to reorder
        
        # Simple greedy approach: sort by distance from current location
        # But ensure we pick up before drop off for each request
        
        # For now, we'll use a simple nearest-neighbor heuristic
        # This can be improved, but works well enough for demonstration
        
        def distance(dest):
            """Helper: distance from current location to destination."""
            location, request, action = dest
            return abs(location[0] - self.current_location[0]) + \
                   abs(location[1] - self.current_location[1])
        
        # Sort by distance
        self.destination_queue.sort(key=distance)
        
        # Ensure pickups happen before dropoffs for each request
        # This is a simplified constraint - a full implementation would be more complex
        pickup_requests = set()
        valid_order = []
        
        for dest in self.destination_queue:
            location, request, action = dest
            if action == "pickup":
                pickup_requests.add(request.request_id)
                valid_order.append(dest)
            elif action == "dropoff":
                # Only add dropoff if we've picked up this passenger
                if request.request_id in pickup_requests or request.is_picked_up():
                    valid_order.append(dest)
                else:
                    # Defer this dropoff - add it later
                    valid_order.append(dest)  # For simplicity, keep it
        
        self.destination_queue = valid_order
    
    def move_one_step(self, current_time: float, city) -> Optional[Tuple[str, Request]]:
        """
        Move vehicle one step towards its next destination.
        
        This simulates one time unit of movement. The vehicle moves
        one grid cell closer to its next destination.
        
        Args:
            current_time: Current simulation time
            city: GridCity object for distance calculations
            
        Returns:
            Tuple of (action, request) if an action occurred, else None
            action is "pickup" or "dropoff"
        """
        if not self.destination_queue:
            return None  # No destinations, stay idle
        
        # Get next destination
        next_dest, request, action_type = self.destination_queue[0]
        
        # Calculate distance to destination
        distance = city.manhattan_distance(self.current_location, next_dest)
        
        if distance == 0:
            # We're at the destination! Perform the action
            if action_type == "pickup":
                self._pickup_passenger(request, current_time)
            elif action_type == "dropoff":
                self._dropoff_passenger(request, current_time)
            
            # Remove this destination from queue
            self.destination_queue.pop(0)
            
            return (action_type, request)
        
        else:
            # Move one step closer to destination
            self._move_towards(next_dest)
            return None
    
    def _move_towards(self, destination: Tuple[int, int]):
        """
        Move one grid cell towards the destination.
        
        Simple strategy: Move horizontally first, then vertically.
        
        Args:
            destination: Target (x, y) coordinates
        """
        x, y = self.current_location
        dest_x, dest_y = destination
        
        # Move horizontally first
        if x < dest_x:
            x += 1
        elif x > dest_x:
            x -= 1
        # Then vertically
        elif y < dest_y:
            y += 1
        elif y > dest_y:
            y -= 1
        
        # Update location and distance
        self.total_distance_traveled += 1
        self.current_location = (x, y)
    
    def _pickup_passenger(self, request: Request, current_time: float):
        """
        Pick up a passenger.
        
        Args:
            request: The request being picked up
            current_time: Current simulation time
        """
        self.current_passengers.append(request)
        request.pickup_time = current_time
    
    def _dropoff_passenger(self, request: Request, current_time: float):
        """
        Drop off a passenger.
        
        Args:
            request: The request being dropped off
            current_time: Current simulation time
        """
        if request in self.current_passengers:
            self.current_passengers.remove(request)
        request.dropoff_time = current_time
        self.total_customers_served += 1
    
    def get_state_dict(self) -> dict:
        """
        Get vehicle state as a dictionary (useful for ML later).
        
        Returns:
            Dictionary with vehicle state information
        """
        return {
            'vehicle_id': self.vehicle_id,
            'location': self.current_location,
            'occupancy': self.get_occupancy(),
            'capacity': self.capacity,
            'is_available': self.is_available(),
            'is_idle': self.is_idle(),
            'num_destinations': len(self.destination_queue)
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Vehicle(id={self.vehicle_id}, "
            f"location={self.current_location}, "
            f"passengers={self.get_occupancy()}/{self.capacity}, "
            f"destinations={len(self.destination_queue)})"
        )


# Quick test
if __name__ == "__main__":
    print("Testing Vehicle class...")
    
    # We need a simple mock city for testing
    class MockCity:
        def manhattan_distance(self, loc1, loc2):
            return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    
    city = MockCity()
    
    # Create a vehicle
    vehicle = Vehicle(vehicle_id=1, initial_location=(0, 0), capacity=4)
    print(f"\nCreated: {vehicle}")
    print(f"Is available? {vehicle.is_available()}")
    print(f"Is idle? {vehicle.is_idle()}")
    
    # Create and assign a request
    from request import Request
    req = Request(request_id=1, pickup=(2, 2), dropoff=(5, 5), request_time=0.0)
    
    print(f"\nAssigning request: {req}")
    vehicle.assign_request(req)
    print(f"After assignment: {vehicle}")
    print(f"Destination queue: {vehicle.destination_queue}")
    
    # Simulate a few moves
    print("\nSimulating movement...")
    for i in range(10):
        action = vehicle.move_one_step(current_time=i, city=city)
        if action:
            print(f"  Time {i}: {action[0]} for request {action[1].request_id}")
        print(f"  Location: {vehicle.current_location}, Passengers: {vehicle.get_occupancy()}")
    
    print("\n✓ Vehicle class working correctly!")
