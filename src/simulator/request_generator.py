"""
request_generator.py - Generates Ride Requests Over Time

This module simulates customers requesting rides throughout the simulation.

Key Concepts:
- Poisson Process: Models random arrival of requests (like real customers)
- Lambda (λ): Average request rate (e.g., 2 requests per minute)
- Random locations: Pickup and dropoff chosen randomly from the city grid

Real-world analogy:
Think of this as the "backend" of the Uber app that receives new ride requests
from customers throughout the day.
"""

import numpy as np
from typing import Tuple, List

# Handle both package import and standalone testing
try:
    from .request import Request
    from .grid_city import GridCity
except ImportError:
    from request import Request
    from grid_city import GridCity


class RequestGenerator:
    """
    Generates ride requests dynamically during simulation.
    
    This simulates the arrival of customers requesting rides over time.
    Uses a Poisson process to model realistic random arrivals.
    
    Attributes:
        city: The GridCity where requests are generated
        request_rate: Average number of requests per time unit (lambda)
        next_request_id: Counter for assigning unique IDs to requests
        all_requests: List of all requests generated (for tracking)
    """
    
    def __init__(self, city: GridCity, request_rate: float = 2.0):
        """
        Initialize the request generator.
        
        Args:
            city: The GridCity object (for valid locations)
            request_rate: Average requests per time unit (lambda)
                         Default 2.0 means ~2 requests per time step on average
        
        Example:
            >>> city = GridCity(size=10)
            >>> generator = RequestGenerator(city, request_rate=3.0)
            # This will generate ~3 requests per time step on average
        """
        self.city = city
        self.request_rate = request_rate
        self.next_request_id = 1
        self.all_requests: List[Request] = []
    
    def generate_requests(self, current_time: float) -> List[Request]:
        """
        Generate new requests for the current time step.
        
        This uses a Poisson distribution to determine how many requests
        arrive at this time step. Sometimes 0, sometimes 1, sometimes many!
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of newly generated Request objects
            
        Example:
            >>> requests = generator.generate_requests(current_time=10.0)
            >>> print(f"Generated {len(requests)} requests at time 10")
            Generated 3 requests at time 10
        """
        # Step 1: Determine how many requests arrive this time step
        # Poisson distribution: some steps have 0, some have 1, some have more
        num_requests = np.random.poisson(self.request_rate)
        
        # Step 2: Create each request
        new_requests = []
        for _ in range(num_requests):
            request = self._create_random_request(current_time)
            new_requests.append(request)
            self.all_requests.append(request)
        
        return new_requests
    
    def _create_random_request(self, request_time: float) -> Request:
        """
        Create a single request with random pickup and dropoff locations.
        
        Important: Ensures pickup ≠ dropoff (no one wants a ride to nowhere!)
        
        Args:
            request_time: When this request was made
            
        Returns:
            New Request object with random locations
        """
        # Get random pickup location
        pickup = self.city.get_random_location()
        
        # Get random dropoff location (must be different from pickup)
        dropoff = self.city.get_random_location()
        
        # Keep trying until we get a different location
        while dropoff == pickup:
            dropoff = self.city.get_random_location()
        
        # Create the request
        request = Request(
            request_id=self.next_request_id,
            pickup=pickup,
            dropoff=dropoff,
            request_time=request_time
        )
        
        # Increment ID counter for next request
        self.next_request_id += 1
        
        return request
    
    def create_batch_requests(self, num_requests: int, current_time: float) -> List[Request]:
        """
        Create a specific number of requests at once.
        
        Useful for:
        - Testing with a known number of requests
        - Simulating a rush hour surge
        - Initial population of requests
        
        Args:
            num_requests: Exact number of requests to create
            current_time: When these requests were made
            
        Returns:
            List of Request objects
            
        Example:
            >>> # Simulate 10 customers all requesting rides at time 0
            >>> rush_hour_requests = generator.create_batch_requests(10, 0.0)
        """
        requests = []
        for _ in range(num_requests):
            request = self._create_random_request(current_time)
            requests.append(request)
            self.all_requests.append(request)
        
        return requests
    
    def get_statistics(self) -> dict:
        """
        Get statistics about generated requests.
        
        Returns:
            Dictionary with request statistics
        """
        completed = sum(1 for r in self.all_requests if r.is_completed())
        pending = sum(1 for r in self.all_requests if not r.is_completed())
        
        return {
            'total_generated': len(self.all_requests),
            'completed': completed,
            'pending': pending,
            'request_rate': self.request_rate
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"RequestGenerator("
            f"rate={self.request_rate}, "
            f"generated={len(self.all_requests)})"
        )


# Test the request generator
if __name__ == "__main__":
    print("Testing RequestGenerator...")
    print("=" * 60)
    
    # Create a city
    city = GridCity(size=10)
    print(f"\n1. Created city: {city}")
    
    # Create request generator with rate of 2 requests per time unit
    generator = RequestGenerator(city, request_rate=2.0)
    print(f"\n2. Created generator: {generator}")
    print(f"   Average rate: {generator.request_rate} requests/time unit")
    
    # Test batch creation (deterministic)
    print("\n3. Testing batch creation:")
    batch = generator.create_batch_requests(num_requests=5, current_time=0.0)
    print(f"   Created {len(batch)} requests:")
    for req in batch:
        dist = city.manhattan_distance(req.pickup, req.dropoff)
        print(f"   - {req.request_id}: {req.pickup} → {req.dropoff} (distance: {dist})")
    
    # Test Poisson generation (random)
    print("\n4. Testing Poisson generation over 10 time steps:")
    print("   Time | Requests Generated")
    print("   " + "-" * 30)
    
    total_generated = 0
    for t in range(10):
        requests = generator.generate_requests(current_time=float(t))
        total_generated += len(requests)
        print(f"   {t:>4} | {len(requests)} requests")
    
    print(f"\n   Total: {total_generated} requests over 10 steps")
    print(f"   Average: {total_generated / 10:.1f} requests/step")
    print(f"   Expected: ~{generator.request_rate} requests/step")
    
    # Show statistics
    print("\n5. Generator statistics:")
    stats = generator.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ RequestGenerator working correctly!")
    print("\nKey takeaways:")
    print("- Poisson process creates realistic random arrivals")
    print("- Sometimes you get 0 requests, sometimes 3+")
    print("- Over time, it averages to the request_rate")
    print("- Each request has valid, different pickup and dropoff")
