#!/usr/bin/env python3
"""
Scan Matcher Example

This example demonstrates how to use the OccupancyGrid module
within a scan matching algorithm implementation.
"""

import os
import sys
import numpy as np
import math
import time

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lidar_slam.lidar_processing import data_parser, scan_converter
from lidar_slam.mapping import occupancy_grid
from lidar_slam.utils import file_utils

class ScanMatcher:
    """Simple scan matching algorithm using the OccupancyGrid module"""
    
    def __init__(self, resolution=0.05, width=20, height=20):
        """
        Initialize the scan matcher with an occupancy grid
        
        Args:
            resolution (float): Grid resolution in meters
            width (float): Grid width in meters
            height (float): Grid height in meters
        """
        # Initialize the occupancy grid for map building
        self.grid = occupancy_grid.OccupancyGrid(
            resolution=resolution,
            width=width, 
            height=height
        )
        
        # Parameters for scan matching
        self.search_radius = 0.5  # meters
        self.search_angle = 0.2   # radians
        self.score_threshold = 0.65  # minimum score to accept a match
        
        # Discretization step sizes
        self.position_step = resolution
        self.angle_step = 0.05  # radians
        
        # Track the robot's pose history
        self.pose_history = []
        
        # For scan conversion
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        self.config = {
            'flip_x': False,
            'flip_y': False,
            'reverse_scan': True,
            'flip_theta': False
        }
    
    def match_scan(self, scan_data, initial_pose=None):
        """
        Match a new scan to the existing map, updating the pose estimate
        
        Args:
            scan_data (dict): Parsed LiDAR data
            initial_pose (dict): Initial pose estimate (if None, use scan_data['pose'])
            
        Returns:
            dict: Corrected pose estimate
        """
        # If no initial pose provided, use the one from scan data
        if initial_pose is None:
            initial_pose = scan_data['pose']
        
        # Convert scan to world coordinates using initial pose
        scan_x, scan_y = scan_converter.convert_scans_to_cartesian(
            scan_data['scan_ranges'],
            self.angle_min,
            self.angle_max,
            initial_pose,
            **self.config
        )
        
        # If this is the first scan, just add it to the map and return
        if not self.pose_history:
            self.grid.update_grid(
                initial_pose['x'], 
                initial_pose['y'], 
                scan_x, 
                scan_y
            )
            self.pose_history.append(initial_pose)
            return initial_pose
        
        # Otherwise, perform scan matching to refine the pose
        best_score = 0
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
                    
                    # Convert scan to world coordinates using test pose
                    test_x, test_y = scan_converter.convert_scans_to_cartesian(
                        scan_data['scan_ranges'],
                        self.angle_min,
                        self.angle_max,
                        test_pose,
                        **self.config
                    )
                    
                    # Compute match score
                    score = self._compute_match_score(test_pose, test_x, test_y)
                    
                    # Update best match if score is better
                    if score > best_score:
                        best_score = score
                        best_pose = test_pose.copy()
        
        # If match score is good enough, use the corrected pose
        if best_score > self.score_threshold:
            # Convert scan to world coordinates using best pose
            best_x, best_y = scan_converter.convert_scans_to_cartesian(
                scan_data['scan_ranges'],
                self.angle_min,
                self.angle_max,
                best_pose,
                **self.config
            )
            
            # Update the map with the new scan
            self.grid.update_grid(
                best_pose['x'], 
                best_pose['y'], 
                best_x, 
                best_y
            )
            
            # Add to pose history
            self.pose_history.append(best_pose)
            
            return best_pose
        else:
            # If match score is too low, use the original pose
            self.grid.update_grid(
                initial_pose['x'], 
                initial_pose['y'], 
                scan_x, 
                scan_y
            )
            
            # Add to pose history
            self.pose_history.append(initial_pose)
            
            return initial_pose
    
    def _compute_match_score(self, pose, scan_x, scan_y):
        """
        Compute a match score between a scan and the existing map
        
        Args:
            pose (dict): Test pose
            scan_x (list): X coordinates of scan points
            scan_y (list): Y coordinates of scan points
            
        Returns:
            float: Match score between 0 and 1
        """
        # This is a simple implementation - check what percentage of scan points
        # align with occupied cells in the grid
        
        # Get the occupancy grid
        grid_data = self.grid.get_grid()
        
        # Count matches
        match_count = 0
        total_points = len(scan_x)
        
        for x, y in zip(scan_x, scan_y):
            # Convert to grid coordinates
            grid_x, grid_y = self.grid.world_to_grid(x, y)
            
            # Check if within grid bounds
            if (0 <= grid_x < self.grid.grid_width and 
                0 <= grid_y < self.grid.grid_height):
                
                # Check if occupied (value > 0.55)
                if grid_data[grid_y, grid_x] > 0.55:
                    match_count += 1
        
        # Calculate score as percentage of matches
        score = match_count / total_points if total_points > 0 else 0
        
        return score
    
    def process_data(self, data_list):
        """
        Process a list of scan data, performing scan matching on each
        
        Args:
            data_list (list): List of parsed LiDAR data
            
        Returns:
            list: Corrected pose history
        """
        # Reset state
        self.pose_history = []
        
        # Initialize the grid to unknown state
        self.grid = occupancy_grid.OccupancyGrid(
            resolution=self.grid.resolution,
            width=self.grid.width,
            height=self.grid.height
        )
        
        # Process each scan
        for i, scan_data in enumerate(data_list):
            if i % 10 == 0:
                print(f"Processing scan {i}/{len(data_list)}")
            
            # Get previous pose as initial estimate, if available
            if self.pose_history:
                prev_pose = self.pose_history[-1]
                
                # Use odometry to estimate new pose
                dx = scan_data['pose']['x'] - data_list[i-1]['pose']['x']
                dy = scan_data['pose']['y'] - data_list[i-1]['pose']['y']
                dtheta = scan_data['pose']['theta'] - data_list[i-1]['pose']['theta']
                
                # Apply odometry change to previous corrected pose
                initial_pose = {
                    'x': prev_pose['x'] + dx,
                    'y': prev_pose['y'] + dy,
                    'theta': prev_pose['theta'] + dtheta
                }
            else:
                # First scan, use raw pose
                initial_pose = None
            
            # Perform scan matching
            corrected_pose = self.match_scan(scan_data, initial_pose)
        
        return self.pose_history
    
    def save_map(self, filename=None, format='png'):
        """
        Save the generated map
        
        Args:
            filename (str): Base filename without extension
            format (str): Format to save in ('png', 'npy', 'csv', or 'all')
            
        Returns:
            list: List of saved filenames
        """
        if filename is None:
            # Create a default filename
            maps_dir = "maps"
            file_utils.ensure_directory_exists(maps_dir)
            timestamp = math.floor(time.time())
            filename = os.path.join(maps_dir, f"scan_matcher_map_{timestamp}")
        
        # Extract path from pose history
        if self.pose_history:
            path_x = [pose['x'] for pose in self.pose_history]
            path_y = [pose['y'] for pose in self.pose_history]
            path = list(zip(path_x, path_y))
            start_pos = (path_x[0], path_y[0])
            end_pos = (path_x[-1], path_y[-1])
        else:
            path = None
            start_pos = None
            end_pos = None
        
        # Save as image if requested
        if format == 'png' or format == 'all':
            file_utils.save_grid_as_image(
                self.grid.get_grid(),
                filename,
                resolution=self.grid.resolution,
                width=self.grid.width,
                height=self.grid.height,
                robot_path=path,
                start_position=start_pos,
                current_position=end_pos
            )
        
        # Always save the raw grid data
        return self.grid.save_to_file(filename, format=format, include_metadata=True)


def main():
    """Main function to demonstrate scan matching"""
    import argparse
    import time
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Scan Matcher Example')
    parser.add_argument('--file', '-f', type=str, 
                       default="../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max-entries', '-m', type=int, default=500,
                       help='Maximum number of entries to read from the file')
    parser.add_argument('--resolution', '-r', type=float, default=0.05,
                       help='Grid resolution in meters')
    args = parser.parse_args()
    
    # Read data from file
    print(f"Reading LiDAR data from: {args.file}")
    data_list = data_parser.read_lidar_data_from_file(args.file, args.max_entries)
    
    if not data_list:
        print("No data was read from the file.")
        return
    
    # Create scan matcher
    print("Initializing scan matcher...")
    matcher = ScanMatcher(resolution=args.resolution)
    
    # Process data
    print("Processing data with scan matching...")
    start_time = time.time()
    corrected_poses = matcher.process_data(data_list)
    end_time = time.time()
    
    # Print stats
    print(f"Scan matching completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(data_list)} scans")
    
    # Save the map
    print("Saving map...")
    saved_files = matcher.save_map(format='all')
    print(f"Saved map to: {', '.join(saved_files)}")
    
    print("Done!")

if __name__ == "__main__":
    main()