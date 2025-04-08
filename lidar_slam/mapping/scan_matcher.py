import math
import numpy as np
import sys
import os

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lidar_slam.lidar_processing import scan_converter 

class ScanMatcher:
    """Scan matching algorithm for pose correction"""
    
    def __init__(self, grid_resolution=0.05, search_radius=0.5, search_angle=0.2):
        """
        Initialize the scan matcher with parameters
        
        Args:
            grid_resolution (float): Grid resolution in meters
            search_radius (float): Search radius for position corrections in meters
            search_angle (float): Search angle for orientation corrections in radians
        """
        self.search_radius = search_radius
        self.search_angle = search_angle
        self.position_step = grid_resolution
        self.angle_step = min(0.05, search_angle / 4)  # Reasonable step size
        
        # For scan conversion
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        self.config = {
            'flip_x': False,
            'flip_y': False,
            'reverse_scan': True,
            'flip_theta': False
        }
    
    def match_scan(self, scan_data, occupancy_grid_map, initial_pose):
        """
        Match a scan to an existing occupancy grid map
        
        Args:
            scan_data (dict): Parsed LiDAR data
            occupancy_grid_map: OccupancyGrid object with the current map
            initial_pose (dict): Initial pose estimate with 'x', 'y', 'theta' keys
            
        Returns:
            tuple: (best_pose, score) with corrected pose and match score
        """
        best_score = 0.0
        best_pose = initial_pose.copy()
        
        # Grid search over possible pose corrections
        for dx in np.arange(-self.search_radius, self.search_radius + self.position_step, self.position_step):
            for dy in np.arange(-self.search_radius, self.search_radius + self.position_step, self.position_step):
                for dtheta in np.arange(-self.search_angle, self.search_angle + self.angle_step, self.angle_step):
                    # Create test pose
                    test_pose = {
                        'x': initial_pose['x'] + dx,
                        'y': initial_pose['y'] + dy,
                        'theta': initial_pose['theta'] + dtheta
                    }
                    
                    # Compute the match score for this pose
                    score = self.compute_match_score(scan_data, occupancy_grid_map, test_pose)
                    
                    # Update best match if score is better
                    if score > best_score:
                        best_score = score
                        best_pose = test_pose.copy()
        
        return best_pose, best_score
    
    def compute_match_score(self, scan_data, occupancy_grid_map, test_pose):
        """
        Compute a match score between a scan and a map
        
        Args:
            scan_data (dict): Parsed LiDAR data
            occupancy_grid_map: OccupancyGrid object with the current map
            test_pose (dict): Test pose to evaluate
            
        Returns:
            float: Match score between 0 and 1
        """
        # Convert scan to Cartesian coordinates using test pose
        scan_x, scan_y = scan_converter.convert_scans_to_cartesian(
            scan_data['scan_ranges'],
            self.angle_min,
            self.angle_max,
            test_pose,
            **self.config
        )
        
        # Get the occupancy grid data
        grid_data = occupancy_grid_map.get_grid()
        
        # Count matches
        match_count = 0
        total_points = len(scan_x)
        
        for x, y in zip(scan_x, scan_y):
            # Convert to grid coordinates
            grid_x, grid_y = occupancy_grid_map.world_to_grid(x, y)
            
            # Check if within grid bounds
            if (0 <= grid_x < occupancy_grid_map.grid_width and 
                0 <= grid_y < occupancy_grid_map.grid_height):
                
                # Check if cell is occupied (value > 0.55)
                if grid_data[grid_y, grid_x] > 0.55:
                    match_count += 1
                # Penalize matching free space (more confident free spaces)
                elif grid_data[grid_y, grid_x] < 0.3:
                    match_count -= 0.1
        
        # Calculate score as percentage of matches
        score = max(0, match_count / total_points if total_points > 0 else 0)
        
        return score