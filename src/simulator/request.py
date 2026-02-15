"""
request.py - Customer Ride Request

This represents a single customer's ride request in our system.

Real-world analogy:
When you open Uber and request a ride, you provide:
- Where you are (pickup location)
- Where you want to go (dropoff location)
- When you requested it (timestamp)

Our Request class captures this information.
"""

from typing import Tuple
import time


class Request:
    """
    Represents a single ride request from a customer.
    
    Attributes:
        request_id: Unique identifier for this request
        pickup: Pickup location as (x, y) coordinates
        dropoff: Destination location as (x, y) coordinates
        request_time: When the request was made (simulation time)
        assigned_vehicle: Which vehicle is serving this (None if unassigned)
        pickup_time: When customer was picked up (None if not yet)
        dropoff_time: When customer was dropped off (None if not yet)
    """
    
    def __init__(
        self,
        request_id: int,
        pickup: Tuple[int, int],
        dropoff: Tuple[int, int],
        request_time: float
    ):
        """
        Create a new ride request.
        
        Args:
            request_id: Unique ID for this request
            pickup: Pickup coordinates (x, y)
            dropoff: Dropoff coordinates (x, y)
            request_time: Simulation time when request was made
        """
        self.request_id = request_id
        self.pickup = pickup
        self.dropoff = dropoff
        self.request_time = request_time
        
        # These are set later during the simulation
        self.assigned_vehicle = None  # Will be set when vehicle is assigned
        self.pickup_time = None        # Will be set when customer is picked up
        self.dropoff_time = None       # Will be set when customer is dropped off
    
    def is_assigned(self) -> bool:
        """Check if this request has been assigned to a vehicle."""
        return self.assigned_vehicle is not None
    
    def is_picked_up(self) -> bool:
        """Check if the customer has been picked up."""
        return self.pickup_time is not None
    
    def is_completed(self) -> bool:
        """Check if the ride is completed (customer dropped off)."""
        return self.dropoff_time is not None
    
    def get_wait_time(self, current_time: float) -> float:
        """
        Calculate how long the customer has been waiting.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Wait time in simulation time units
            
        Note:
            If customer is already picked up, returns 0 (not waiting anymore)
        """
        if self.is_picked_up():
            return 0.0
        return current_time - self.request_time
    
    def get_trip_duration(self) -> float:
        """
        Calculate total trip duration (from request to dropoff).
        
        Returns:
            Total duration, or None if trip not completed
        """
        if not self.is_completed():
            return None
        return self.dropoff_time - self.request_time
    
    def get_status(self) -> str:
        """
        Get human-readable status of this request.
        
        Returns:
            One of: "pending", "assigned", "picked_up", "completed"
        """
        if self.is_completed():
            return "completed"
        elif self.is_picked_up():
            return "picked_up"
        elif self.is_assigned():
            return "assigned"
        else:
            return "pending"
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Request(id={self.request_id}, "
            f"pickup={self.pickup}, "
            f"dropoff={self.dropoff}, "
            f"status={self.get_status()})"
        )


# Quick test
if __name__ == "__main__":
    print("Testing Request class...")
    
    # Create a sample request
    req = Request(
        request_id=1,
        pickup=(2, 3),
        dropoff=(7, 8),
        request_time=0.0
    )
    
    print(f"\nCreated: {req}")
    print(f"Status: {req.get_status()}")
    print(f"Is assigned? {req.is_assigned()}")
    
    # Simulate assignment
    req.assigned_vehicle = 5
    print(f"\nAfter assignment:")
    print(f"Status: {req.get_status()}")
    print(f"Wait time at t=10: {req.get_wait_time(10.0)}")
    
    # Simulate pickup
    req.pickup_time = 15.0
    print(f"\nAfter pickup:")
    print(f"Status: {req.get_status()}")
    print(f"Wait time: {req.get_wait_time(20.0)}")  # Should be 0
    
    # Simulate dropoff
    req.dropoff_time = 25.0
    print(f"\nAfter dropoff:")
    print(f"Status: {req.get_status()}")
    print(f"Trip duration: {req.get_trip_duration()}")
    
    print("\n✓ Request class working correctly!")
