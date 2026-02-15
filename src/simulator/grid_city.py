"""
grid_city.py - The Grid-Based City Environment

This module defines our abstract city as a grid. Think of it like a chessboard
where vehicles and customers exist at specific coordinates.

Key Concepts:
- Grid: 2D coordinate system (like (row, col) or (x, y))
- Manhattan Distance: How we calculate distances in a grid city
  Example: From (0,0) to (3,4) = |3-0| + |4-0| = 7 blocks
"""

import numpy as np
from typing import Tuple, List


class GridCity:
    """
    A simple grid-based city for our ride-sharing simulation.
    
    Think of this as a simplified version of a real city where:
    - Streets form a perfect grid
    - Movement is only horizontal or vertical (no diagonals)
    - Each location is identified by (x, y) coordinates
    """
    
    def __init__(self, size: int = 10):
        """
        Initialize the city grid.
        
        Args:
            size: The dimension of the square grid (default 10x10)
                 A 10x10 grid has 100 possible locations
        """
        self.size = size
        self.grid = np.zeros((size, size))  # Not used yet, but useful for visualization
        
    def is_valid_location(self, location: Tuple[int, int]) -> bool:
        """
        Check if a location is within the city boundaries.
        
        Args:
            location: A tuple (x, y) representing coordinates
            
        Returns:
            True if location is valid, False otherwise
            
        Example:
            >>> city = GridCity(size=10)
            >>> city.is_valid_location((5, 7))
            True
            >>> city.is_valid_location((12, 3))
            False
        """
        x, y = location
        return 0 <= x < self.size and 0 <= y < self.size
    
    def manhattan_distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two locations.
        
        Manhattan distance = |x1 - x2| + |y1 - y2|
        This is the actual distance a vehicle would travel in a grid city.
        
        Args:
            loc1: First location (x1, y1)
            loc2: Second location (x2, y2)
            
        Returns:
            Integer distance in grid blocks
            
        Example:
            >>> city = GridCity()
            >>> city.manhattan_distance((0, 0), (3, 4))
            7
        """
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    
    def get_random_location(self) -> Tuple[int, int]:
        """
        Generate a random valid location in the city.
        
        Useful for:
        - Spawning new ride requests
        - Initializing vehicle positions
        
        Returns:
            Random (x, y) coordinates within grid bounds
        """
        x = np.random.randint(0, self.size)
        y = np.random.randint(0, self.size)
        return (x, y)
    
    def get_all_locations(self) -> List[Tuple[int, int]]:
        """
        Get a list of all possible locations in the city.
        
        Returns:
            List of all (x, y) coordinates
        """
        locations = []
        for x in range(self.size):
            for y in range(self.size):
                locations.append((x, y))
        return locations
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"GridCity(size={self.size}, total_locations={self.size**2})"


# Quick test to verify our city works
if __name__ == "__main__":
    print("Testing GridCity...")
    
    # Create a 10x10 city
    city = GridCity(size=10)
    print(f"Created: {city}")
    
    # Test location validity
    print(f"\nIs (5, 7) valid? {city.is_valid_location((5, 7))}")
    print(f"Is (12, 3) valid? {city.is_valid_location((12, 3))}")
    
    # Test distance calculation
    dist = city.manhattan_distance((0, 0), (3, 4))
    print(f"\nDistance from (0,0) to (3,4): {dist} blocks")
    
    # Test random location generation
    random_loc = city.get_random_location()
    print(f"Random location: {random_loc}")
    
    print("\n✓ GridCity working correctly!")
