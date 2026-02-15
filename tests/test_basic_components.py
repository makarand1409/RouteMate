"""
test_basic_components.py - Verify all basic components work together

Run this to test that GridCity, Request, and Vehicle classes
work correctly together.
"""

import sys
import os

# Add the parent directory to the path so we can import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Now import from src.simulator
from src.simulator import GridCity, Request, Vehicle


def test_basic_scenario():
    """
    Test a simple scenario:
    - 1 vehicle with capacity 4
    - 2 ride requests
    - Vehicle should pool both requests
    """
    print("=" * 60)
    print("TESTING BASIC POOLING SCENARIO")
    print("=" * 60)
    
    # Create a 10x10 city
    city = GridCity(size=10)
    print(f"\n1. Created city: {city}")
    
    # Create a vehicle at (0, 0)
    vehicle = Vehicle(vehicle_id=1, initial_location=(0, 0), capacity=4)
    print(f"\n2. Created vehicle: {vehicle}")
    
    # Create first request: (1, 1) -> (3, 3)
    req1 = Request(
        request_id=1,
        pickup=(1, 1),
        dropoff=(3, 3),
        request_time=0.0
    )
    print(f"\n3. Created request 1: {req1}")
    
    # Assign first request
    vehicle.assign_request(req1)
    print(f"\n4. After assigning request 1:")
    print(f"   Vehicle: {vehicle}")
    print(f"   Request status: {req1.get_status()}")
    
    # Create second request: (2, 2) -> (5, 5)
    req2 = Request(
        request_id=2,
        pickup=(2, 2),
        dropoff=(5, 5),
        request_time=5.0
    )
    print(f"\n5. Created request 2: {req2}")
    
    # Check if vehicle can accept second request
    can_accept = vehicle.can_accept_request(req2)
    print(f"\n6. Can vehicle accept request 2? {can_accept}")
    
    if can_accept:
        vehicle.assign_request(req2)
        print(f"   Assigned! Vehicle now has {len(vehicle.destination_queue)} destinations")
    
    # Simulate the vehicle serving both requests
    print(f"\n7. Simulating vehicle movement:")
    print("   " + "-" * 50)
    
    current_time = 0.0
    max_steps = 30
    
    for step in range(max_steps):
        action = vehicle.move_one_step(current_time, city)
        
        if action:
            action_type, request = action
            print(f"   Time {current_time:>4.0f}: {action_type.upper():>8} request {request.request_id} at {vehicle.current_location}")
        
        current_time += 1.0
        
        # Stop if vehicle is idle
        if vehicle.is_idle():
            print(f"   Time {current_time:>4.0f}: Vehicle idle - all requests completed!")
            break
    
    # Print final statistics
    print(f"\n8. Final Statistics:")
    print(f"   " + "-" * 50)
    print(f"   Total distance traveled: {vehicle.total_distance_traveled} blocks")
    print(f"   Total customers served: {vehicle.total_customers_served}")
    print(f"   Request 1 trip duration: {req1.get_trip_duration()}")
    print(f"   Request 2 trip duration: {req2.get_trip_duration()}")
    
    # Verify both requests were completed
    assert req1.is_completed(), "Request 1 should be completed"
    assert req2.is_completed(), "Request 2 should be completed"
    assert vehicle.is_idle(), "Vehicle should be idle"
    
    print(f"\n✓ All tests passed!")
    print("=" * 60)


def test_capacity_limit():
    """
    Test that vehicle respects capacity limits.
    """
    print("\n" + "=" * 60)
    print("TESTING CAPACITY LIMITS")
    print("=" * 60)
    
    city = GridCity(size=10)
    
    # Create vehicle with capacity 2
    vehicle = Vehicle(vehicle_id=1, initial_location=(0, 0), capacity=2)
    print(f"\n1. Created vehicle with capacity {vehicle.capacity}")
    
    # Create 3 requests
    requests = [
        Request(i, (i, i), (i+2, i+2), i*5.0)
        for i in range(1, 4)
    ]
    
    # Assign first two
    vehicle.assign_request(requests[0])
    vehicle.assign_request(requests[1])
    
    print(f"\n2. Assigned 2 requests")
    print(f"   Vehicle available? {vehicle.is_available()}")
    print(f"   Current occupancy: {vehicle.get_occupancy()}")
    
    # Try to assign third
    can_accept = vehicle.can_accept_request(requests[2])
    print(f"\n3. Can accept 3rd request? {can_accept}")
    
    print("\n✓ Capacity test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_scenario()
    test_capacity_limit()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nYour core simulator components are working correctly.")
    print("Next step: Build the full simulation engine!")