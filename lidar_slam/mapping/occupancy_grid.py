"""
Occupancy Grid Mapping Module

Core implementation of the occupancy grid mapping algorithm, which can be used
by other modules like Scan Matcher or FastSLAM. This module focuses on the mapping
functionality without visualization dependencies.
"""

import math
import numpy as np
import os
import time
import json
from scipy.ndimage import gaussian_filter, binary_dilation

class OccupancyGrid:
    """Enhanced class to handle occupancy grid mapping from LiDAR data"""
    
    def __init__(self, resolution=0.05, width=20, height=20, init_position=None, 
                 expansion_factor=1.5, sensor_noise_variance=0.01):
        """
        Initialize an occupancy grid
        
        Args:
            resolution (float): Grid cell size in meters
            width (float): Width of the grid in meters
            height (float): Height of the grid in meters
            init_position (dict): Optional initial position with 'x', 'y', 'theta' keys
            expansion_factor (float): Factor to expand grid when needed (1.5 = 50% expansion)
            sensor_noise_variance (float): Variance parameter for sensor noise model
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        self.expansion_factor = expansion_factor
        self.sensor_noise_variance = sensor_noise_variance
        
        # Calculate grid dimensions
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Initialize grid with unknown values (0.5 represents unknown)
        # Values closer to 1 will represent occupied
        # Values closer to 0 will represent free
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5
        
        # Origin of the grid (center of the grid)
        self.origin_x = width / 2
        self.origin_y = height / 2
        
        # Store world boundaries
        self.x_min = -self.origin_x
        self.x_max = self.origin_x
        self.y_min = -self.origin_y
        self.y_max = self.origin_y
        
        # Log odds version of the grid (for Bayesian updates)
        # Initialize with log odds of 0.5 probability -> log(0.5/0.5) = 0
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))
        
        # Parameters for occupancy update
        self.log_odds_occupied = math.log(0.7/0.3)  # Probability of cell being occupied given a hit
        self.log_odds_free = math.log(0.3/0.7)      # Probability of cell being occupied given a miss
        
        # Maximum and minimum log odds to prevent overflow
        self.log_odds_min = np.log(0.01/0.99)  # ~-4.6
        self.log_odds_max = np.log(0.99/0.01)  # ~4.6
        
        # Additional parameters for scan matching
        self.lidar_max_range = 10.0  # Default LiDAR range in meters
        self.lidar_fov = math.pi     # Default LiDAR field of view in radians
        self.num_samples_per_rev = 180  # Default number of samples per revolution
        self.angular_step = math.pi / 180  # Default angular step (1 degree)
        
        # Keep track of visits for each cell (for scan matching)
        self.occupancy_grid_visited = np.zeros((self.grid_height, self.grid_width))
        self.occupancy_grid_total = np.ones((self.grid_height, self.grid_width))
        
        # Create coordinate matrices for the grid (useful for scan matching)
        self.OccupancyGridX, self.OccupancyGridY = np.meshgrid(
            np.linspace(-width/2, width/2, self.grid_width),
            np.linspace(-height/2, height/2, self.grid_height)
        )
        
        # Optional: Store metadata about the mapping
        self.metadata = {
            'resolution': resolution,
            'width': width,
            'height': height,
            'updates': 0,
            'creation_time': time.time()
        }
        
        # Store initial position if provided
        if init_position:
            self.metadata['init_x'] = init_position['x']
            self.metadata['init_y'] = init_position['y']
            self.metadata['init_theta'] = init_position['theta']
            
        # Performance tracking and statistics
        self.update_count = 0
        self.last_resize_count = 0
        self.resize_frequency = 10  # Check for resize every N updates
        self.stats = {
            'free_cell_count': 0,
            'occupied_cell_count': 0,
            'unknown_cell_count': self.grid_width * self.grid_height,
            'updates': 0,
            'resizes': 0
        }
        
        # Parameters for enhanced mapping features
        self.near_obstacle_dilation = 2  # How many cells to dilate obstacles by
        self.max_range_variance = 0.1  # Higher variance for cells near max range
        
        # For turn handling
        self.angle_of_incidence_factor = 0.1  # Factor to adjust updates based on angle of incidence
        self.motion_compensation = True       # Whether to compensate for robot motion during scans
        self.scan_history_length = 20         # Number of previous scans to keep for motion analysis
        self.previous_poses = []              # Store previous robot poses to detect turns
        self.turn_detection_threshold = 0.001 # Threshold for detecting turns (radians)
        self.max_angle_of_incidence = 80      # Maximum angle (degrees) for reliable measurements
        self.outlier_rejection_threshold = 3  # Standard deviations for outlier rejection
        self.dynamic_object_threshold = 0.3   # Threshold for detecting potentially dynamic objects
        
        # For temporal filtering
        self.temporal_consistency_grid = np.zeros((self.grid_height, self.grid_width))
        self.consistency_threshold = 3        # Number of consistent observations needed
    
    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices
        
        Args:
            x (float or list or array): X coordinate(s) in world frame
            y (float or list or array): Y coordinate(s) in world frame
            
        Returns:
            tuple: (grid_x, grid_y) indices
        """
        # Handle array-like inputs
        if isinstance(x, (list, np.ndarray)) and isinstance(y, (list, np.ndarray)):
            # Convert inputs to numpy arrays if they aren't already
            x_array = np.asarray(x)
            y_array = np.asarray(y)
            
            # Calculate grid indices
            grid_x = np.floor((x_array + self.origin_x) / self.resolution).astype(int)
            grid_y = np.floor((y_array + self.origin_y) / self.resolution).astype(int)
            
            return grid_x, grid_y
        else:
            # Handle single value case
            grid_x = int((x + self.origin_x) / self.resolution)
            grid_y = int((y + self.origin_y) / self.resolution)
            
            # Ensure we're within grid bounds
            grid_x = max(0, min(grid_x, self.grid_width - 1))
            grid_y = max(0, min(grid_y, self.grid_height - 1))
            
            return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices to world coordinates
        
        Args:
            grid_x (int or list or array): X index in grid
            grid_y (int or list or array): Y index in grid
            
        Returns:
            tuple: (x, y) coordinates in world frame
        """
        # Handle array-like inputs
        if isinstance(grid_x, (list, np.ndarray)) and isinstance(grid_y, (list, np.ndarray)):
            # Convert inputs to numpy arrays if they aren't already
            grid_x_array = np.asarray(grid_x)
            grid_y_array = np.asarray(grid_y)
            
            # Calculate world coordinates (center of cell)
            x = grid_x_array * self.resolution + self.resolution/2 - self.origin_x
            y = grid_y_array * self.resolution + self.resolution/2 - self.origin_y
            
            return x, y
        else:
            # Single value case
            x = grid_x * self.resolution - self.origin_x
            y = grid_y * self.resolution - self.origin_y
            return x, y
    
    def update_grid(self, robot_x, robot_y, scan_x, scan_y):
        """
        Update the occupancy grid with a laser scan (compatibility method)
        
        Args:
            robot_x (float): Robot's x position in world coordinates
            robot_y (float): Robot's y position in world coordinates
            scan_x (list): List of scan x points in world coordinates
            scan_y (list): List of scan y points in world coordinates
        """
        # Call the vectorized version for better performance
        self.update_grid_vectorized(robot_x, robot_y, scan_x, scan_y)
    
    def update_grid_vectorized(self, robot_x, robot_y, scan_x, scan_y):
        """
        Update the occupancy grid with a laser scan using vectorized operations
        
        Args:
            robot_x (float): Robot's x position in world coordinates
            robot_y (float): Robot's y position in world coordinates
            scan_x (list): List of scan x points in world coordinates
            scan_y (list): List of scan y points in world coordinates
        """
        # First check if we need to expand the grid
        self.check_and_expand_grid(scan_x, scan_y)
        
        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        # Convert scan points to grid coordinates
        scan_grid_x, scan_grid_y = self.world_to_grid(scan_x, scan_y)
        
        # Create a temporary grid to store updates
        temp_update_grid = np.zeros_like(self.log_odds_grid)
        
        # Process each scan point in batches to maintain memory efficiency
        batch_size = 50  # Process 50 scan points at a time
        num_points = len(scan_grid_x)
        
        for batch_start in range(0, num_points, batch_size):
            batch_end = min(batch_start + batch_size, num_points)
            batch_grid_x = scan_grid_x[batch_start:batch_end]
            batch_grid_y = scan_grid_y[batch_start:batch_end]
            
            # Process this batch of scan points
            for i in range(batch_end - batch_start):
                endpoint_x, endpoint_y = batch_grid_x[i], batch_grid_y[i]
                
                # Skip if endpoint is outside grid
                if (endpoint_x < 0 or endpoint_x >= self.grid_width or 
                    endpoint_y < 0 or endpoint_y >= self.grid_height):
                    continue
                
                # Mark the endpoint as occupied
                # Using sensor model with distance-based uncertainty
                scan_distance = np.sqrt((robot_x - scan_x[batch_start + i])**2 + 
                                     (robot_y - scan_y[batch_start + i])**2)
                
                # Adjust occupancy update based on distance (more uncertain at long ranges)
                distance_factor = 1.0 + scan_distance * self.sensor_noise_variance
                occupied_update = self.log_odds_occupied / distance_factor
                
                # Get cells along ray using Bresenham line algorithm
                cells_x, cells_y = self._bresenham_line(robot_grid_x, robot_grid_y, endpoint_x, endpoint_y)
                
                # Mark all cells along ray as free except the endpoint
                for j in range(len(cells_x) - 1):  # Exclude the last point (the endpoint)
                    cell_x, cell_y = cells_x[j], cells_y[j]
                    if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                        # Adjust free update based on distance from robot
                        cell_distance = np.sqrt((cell_x - robot_grid_x)**2 + (cell_y - robot_grid_y)**2) * self.resolution
                        free_update = self.log_odds_free * (1.0 - cell_distance * self.sensor_noise_variance/2)
                        # Ensure free update doesn't exceed original log odds free value
                        free_update = max(free_update, self.log_odds_free * 0.5)
                        
                        # Apply the update to the temporary grid
                        temp_update_grid[cell_y, cell_x] += free_update
                        
                        # Update tracking for scan matching
                        self.occupancy_grid_total[cell_y, cell_x] += 1
                
                # Mark the endpoint as occupied
                if (endpoint_x >= 0 and endpoint_x < self.grid_width and
                    endpoint_y >= 0 and endpoint_y < self.grid_height):
                    temp_update_grid[endpoint_y, endpoint_x] += occupied_update
                    
                    # Update tracking for scan matching
                    self.occupancy_grid_visited[endpoint_y, endpoint_x] += 1
                    self.occupancy_grid_total[endpoint_y, endpoint_x] += 1
        
        # Apply all updates at once to the log odds grid
        self.log_odds_grid += temp_update_grid
        
        # Clamp log odds values to prevent numerical issues
        self.log_odds_grid = np.clip(self.log_odds_grid, self.log_odds_min, self.log_odds_max)
        
        # Convert log odds to probabilities
        self.grid = 1 - 1 / (1 + np.exp(self.log_odds_grid))
        
        # Update statistics and metadata
        self.metadata['updates'] += 1
        self.metadata['last_update_time'] = time.time()
        
        self.stats['updates'] += 1
        self.stats['free_cell_count'] = np.sum(self.grid < 0.4)
        self.stats['occupied_cell_count'] = np.sum(self.grid > 0.6)
        self.stats['unknown_cell_count'] = self.grid_width * self.grid_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count']
        )
    
    def update_cells_along_ray(self, x0, y0, x1, y1):
        """
        Mark cells along a ray from (x0,y0) to (x1,y1) as free using Bresenham's algorithm
        
        Args:
            x0 (int): Starting x grid coordinate
            y0 (int): Starting y grid coordinate
            x1 (int): Ending x grid coordinate
            y1 (int): Ending y grid coordinate
        """
        # Use the Bresenham line implementation
        cells_x, cells_y = self._bresenham_line(x0, y0, x1, y1)
        
        # Mark cells as free (skip the last cell which is the endpoint)
        for j in range(len(cells_x) - 1):
            cell_x, cell_y = cells_x[j], cells_y[j]
            if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                self.log_odds_grid[cell_y, cell_x] += self.log_odds_free
                self.occupancy_grid_total[cell_y, cell_x] += 1
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Compute cells on a line from (x0, y0) to (x1, y1) using Bresenham's algorithm
        
        Args:
            x0, y0: Starting point coordinates
            x1, y1: Ending point coordinates
            
        Returns:
            lists: (x_cells, y_cells) with all cell coordinates on the line
        """
        # Initialize lists to store cell coordinates
        x_cells = []
        y_cells = []
        
        # Make sure all inputs are integers
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
        # Calculate deltas
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        # Calculate step directions
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        # Initialize error
        err = dx - dy
        
        # Bresenham's line algorithm
        x, y = x0, y0
        while True:
            # Add current cell to the lists
            x_cells.append(x)
            y_cells.append(y)
            
            # Check if we've reached the end
            if x == x1 and y == y1:
                break
                
            # Calculate error for next step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return x_cells, y_cells
    
    def get_grid(self):
        """
        Get a copy of the grid
        
        Returns:
            numpy.ndarray: Copy of the occupancy grid
        """
        return self.grid.copy()
    
    def get_grid_for_display(self, enhance=False, show_dynamic=False):
        """
        Get a copy of the grid suitable for display with optional enhancements
        
        Args:
            enhance (bool): Whether to enhance obstacles for display
            show_dynamic (bool): Whether to highlight dynamic/inconsistent areas
            
        Returns:
            array: Grid values between 0 and 1
        """
        # Make a copy to avoid modifying the original
        temp_grid = self.grid.copy()
        
        if enhance:
            # Create a binary grid of obstacles and dilate
            binary_obstacles = (temp_grid > 0.6).astype(np.uint8)
            dilated_obstacles = binary_dilation(
                binary_obstacles,
                iterations=1  # Just for display
            )
            
            # Mark dilated obstacles with a special value (0.8) for display
            dilate_mask = dilated_obstacles & ~binary_obstacles
            temp_grid[dilate_mask] = 0.8
        
        if show_dynamic:
            # Highlight potentially dynamic objects with value 0.9
            dynamic_mask = self.identify_dynamic_objects()
            temp_grid[dynamic_mask] = 0.9
            
        return temp_grid
    
    def identify_dynamic_objects(self):
        """
        Identify potentially dynamic objects in the map based on inconsistent updates
        
        Returns:
            2D array: Boolean mask of potentially dynamic objects
        """
        # Areas with intermediate probability that have been updated multiple times
        # may represent dynamic objects
        if hasattr(self, 'temporal_consistency_grid'):
            dynamic_mask = ((self.grid > 0.4) & (self.grid < 0.6) & 
                          (self.temporal_consistency_grid > 0) &
                          (self.temporal_consistency_grid < self.consistency_threshold))
        else:
            # Fallback for compatibility with older versions
            dynamic_mask = (self.grid > 0.4) & (self.grid < 0.6)
        
        return dynamic_mask
    
    def get_metadata(self):
        """
        Get grid metadata
        
        Returns:
            dict: Dictionary with grid metadata
        """
        return self.metadata.copy()
    
    def set_lidar_params(self, max_range, field_of_view, num_samples):
        """
        Set LiDAR parameters for scan matching
        
        Args:
            max_range (float): Maximum range of the LiDAR in meters
            field_of_view (float): Field of view in radians
            num_samples (int): Number of samples per revolution
        """
        self.lidar_max_range = max_range
        self.lidar_fov = field_of_view
        self.num_samples_per_rev = num_samples
        self.angular_step = field_of_view / num_samples
    
    def check_and_expand_grid(self, world_x, world_y):
        """
        Check if points are outside current grid bounds and expand if necessary
        
        Args:
            world_x: X coordinates in world frame to check
            world_y: Y coordinates in world frame to check
            
        Returns:
            bool: True if grid was expanded, False otherwise
        """
        # Increment update counter
        self.update_count += 1
        
        # Only check for resize periodically to improve performance
        if self.update_count - self.last_resize_count < self.resize_frequency:
            return False
        
        self.last_resize_count = self.update_count
        
        # Convert to numpy arrays if not already
        x_array = np.asarray(world_x)
        y_array = np.asarray(world_y)
        
        # Find extremes
        min_x, max_x = np.min(x_array), np.max(x_array)
        min_y, max_y = np.min(y_array), np.max(y_array)
        
        # Add padding for safety (1 meter)
        padding = 1.0
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        # Check if we need to expand in any direction
        expanded = False
        
        # Check if we need to expand left
        if min_x < self.x_min:
            self._expand_grid('left')
            expanded = True
        
        # Check if we need to expand right
        if max_x > self.x_max:
            self._expand_grid('right')
            expanded = True
        
        # Check if we need to expand down
        if min_y < self.y_min:
            self._expand_grid('down')
            expanded = True
        
        # Check if we need to expand up
        if max_y > self.y_max:
            self._expand_grid('up')
            expanded = True
        
        return expanded
    
    def _expand_grid(self, direction):
        """
        Expand the occupancy grid in a specified direction
        
        Args:
            direction: String indicating expansion direction ('left', 'right', 'up', 'down')
        """
        # Track this resize operation
        self.stats['resizes'] += 1
        
        # Save old grid shape
        old_height, old_width = self.grid.shape
        
        # Calculate new dimensions for the full grid
        if direction in ['left', 'right']:
            # Expand width by expansion_factor
            new_width = int(old_width * self.expansion_factor)
            new_height = old_height
            # Calculate additional cells needed
            additional_width = new_width - old_width
        else:  # up or down
            # Expand height by expansion_factor
            new_height = int(old_height * self.expansion_factor)
            new_width = old_width
            # Calculate additional cells needed
            additional_height = new_height - old_height
        
        # Create new grids
        new_grid = np.ones((new_height, new_width)) * 0.5
        new_log_odds = np.zeros((new_height, new_width))
        new_visited = np.zeros((new_height, new_width))
        new_total = np.ones((new_height, new_width))
        
        # Create new temporal consistency grid if it exists
        if hasattr(self, 'temporal_consistency_grid'):
            new_consistency = np.zeros((new_height, new_width))
        
        # Calculate where to place the old grid in the new one
        if direction == 'left':
            # Add cells to the left
            x_offset = additional_width
            y_offset = 0
            # Update world boundaries
            self.x_min -= additional_width * self.resolution
        elif direction == 'right':
            # Add cells to the right
            x_offset = 0
            y_offset = 0
            # Update world boundaries
            self.x_max += additional_width * self.resolution
        elif direction == 'down':
            # Add cells to the bottom
            x_offset = 0
            y_offset = additional_height
            # Update world boundaries
            self.y_min -= additional_height * self.resolution
        else:  # 'up'
            # Add cells to the top
            x_offset = 0
            y_offset = 0
            # Update world boundaries
            self.y_max += additional_height * self.resolution
        
        # Copy old data to new grids
        new_grid[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.grid
        new_log_odds[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.log_odds_grid
        new_visited[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.occupancy_grid_visited
        new_total[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.occupancy_grid_total
        
        # Copy temporal consistency data if it exists
        if hasattr(self, 'temporal_consistency_grid'):
            new_consistency[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.temporal_consistency_grid
        
        # Update origin if needed
        if direction == 'left':
            self.origin_x += additional_width * self.resolution
        elif direction == 'down':
            self.origin_y += additional_height * self.resolution
        
        # Update grid dimensions
        self.grid_width = new_width
        self.grid_height = new_height
        self.width = new_width * self.resolution
        self.height = new_height * self.resolution
        
        # Update grids
        self.grid = new_grid
        self.log_odds_grid = new_log_odds
        self.occupancy_grid_visited = new_visited
        self.occupancy_grid_total = new_total
        
        # Update temporal consistency grid if it exists
        if hasattr(self, 'temporal_consistency_grid'):
            self.temporal_consistency_grid = new_consistency
        
        # Regenerate coordinate matrices
        self.OccupancyGridX, self.OccupancyGridY = np.meshgrid(
            np.linspace(-self.width/2, self.width/2, self.grid_width),
            np.linspace(-self.height/2, self.height/2, self.grid_height)
        )
        
        # Update metadata
        self.metadata['width'] = self.width
        self.metadata['height'] = self.height
        self.metadata['grid_width'] = self.grid_width
        self.metadata['grid_height'] = self.grid_height
        
        # Update statistics
        self.stats['unknown_cell_count'] = new_width * new_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count']
        )
        
        print(f"Grid expanded {direction}: New dimensions {new_width}x{new_height} cells, "
              f"{self.width:.1f}x{self.height:.1f} meters")
    
    def detect_robot_turning(self, current_theta):
        """
        Detect if the robot is currently in a turning motion
        
        Args:
            current_theta: Current robot orientation
            
        Returns:
            bool: True if robot is turning, False otherwise
            float: Turn rate (radians per update)
        """
        # Need at least 3 poses to detect turning
        if len(self.previous_poses) < 3:
            self.previous_poses.append(current_theta)
            return False, 0.0
            
        # Calculate changes in orientation
        theta_changes = []
        for i in range(1, len(self.previous_poses)):
            change = self.previous_poses[i] - self.previous_poses[i-1]
            # Handle angle wrap-around
            if change > np.pi:
                change -= 2 * np.pi
            elif change < -np.pi:
                change += 2 * np.pi
            theta_changes.append(abs(change))
            
        # Current change
        current_change = current_theta - self.previous_poses[-1]
        # Handle angle wrap-around
        if current_change > np.pi:
            current_change -= 2 * np.pi
        elif current_change < -np.pi:
            current_change += 2 * np.pi
            
        # Update pose history (keep limited history)
        self.previous_poses.append(current_theta)
        if len(self.previous_poses) > self.scan_history_length:
            self.previous_poses.pop(0)
            
        # Calculate average change rate
        avg_change = np.mean(theta_changes + [abs(current_change)])
        
        # Detect if turning
        is_turning = avg_change > self.turn_detection_threshold
        return is_turning, current_change
        
    def calculate_angle_of_incidence(self, robot_x, robot_y, scan_x, scan_y, robot_theta):
        """
        Calculate angle of incidence for each scan point
        
        Args:
            robot_x, robot_y: Robot position
            scan_x, scan_y: Scan point coordinates
            robot_theta: Robot orientation
            
        Returns:
            array: Angles of incidence in degrees (0 = perpendicular to surface)
        """
        # Convert to numpy arrays if needed
        scan_x_array = np.asarray(scan_x)
        scan_y_array = np.asarray(scan_y)
        
        # Calculate vectors from robot to scan points
        dx = scan_x_array - robot_x
        dy = scan_y_array - robot_y
        
        # Calculate distances and angles to each point
        distances = np.sqrt(dx**2 + dy**2)
        angles_to_points = np.arctan2(dy, dx)
        
        # Calculate angle differences between adjacent points (for surface normal estimation)
        # We'll use a simple approach - calculate angle between adjacent points
        # This works best when scan points are ordered
        normals = np.zeros_like(angles_to_points)
        
        for i in range(1, len(angles_to_points) - 1):
            # Estimate surface normal as perpendicular to the line between adjacent points
            dx_surface = scan_x_array[i+1] - scan_x_array[i-1]
            dy_surface = scan_y_array[i+1] - scan_y_array[i-1]
            
            if dx_surface == 0 and dy_surface == 0:
                # Handle the case when adjacent points are identical
                normals[i] = angles_to_points[i] + np.pi/2
            else:
                # Normal is perpendicular to surface
                surface_angle = np.arctan2(dy_surface, dx_surface)
                normals[i] = surface_angle + np.pi/2
        
        # Handle endpoints
        normals[0] = normals[1] if len(normals) > 1 else angles_to_points[0] + np.pi/2
        normals[-1] = normals[-2] if len(normals) > 1 else angles_to_points[-1] + np.pi/2
        
        # Calculate angle of incidence (angle between ray and normal)
        incidence_angles = np.abs(angles_to_points - normals)
        
        # Normalize to [0, 90] degrees
        incidence_angles = np.minimum(incidence_angles, np.pi - incidence_angles)
        incidence_angles = np.rad2deg(incidence_angles)
        
        return incidence_angles
    
    def reject_outliers(self, scan_distances, max_deviation=3):
        """
        Reject outlier measurements based on statistical properties
        
        Args:
            scan_distances: Array of distance measurements
            max_deviation: Maximum allowed standard deviations from mean
            
        Returns:
            array: Boolean mask where True indicates valid measurements
        """
        distances = np.asarray(scan_distances)
        if len(distances) < 3:
            # Not enough data for statistical analysis
            return np.ones(len(distances), dtype=bool)
            
        # Calculate median and standard deviation
        median = np.median(distances)
        std = np.std(distances)
        
        if std == 0:
            # All values are the same
            return np.ones(len(distances), dtype=bool)
            
        # Find points that are within the allowed deviation
        deviation = np.abs(distances - median)
        valid_mask = deviation <= (max_deviation * std)
        
        return valid_mask
    
    def update_grid_with_turn_handling(self, robot_x, robot_y, robot_theta, scan_x, scan_y, scan_ranges=None):
        """
        Update occupancy grid with enhanced handling for turns
        
        Args:
            robot_x, robot_y: Robot position
            robot_theta: Robot orientation
            scan_x, scan_y: Scan point coordinates (already filtered for max range)
            scan_ranges: Original scan distances (if available)
        """
        # Detect if the robot is turning
        is_turning, turn_rate = self.detect_robot_turning(robot_theta)
        
        # Initialize valid mask based on points available
        valid_mask = np.ones(len(scan_x), dtype=bool)
        
        # Calculate angles of incidence if we have enough points
        if len(scan_x) > 3:
            incidence_angles = self.calculate_angle_of_incidence(
                robot_x, robot_y, scan_x, scan_y, robot_theta)
            
            # Filter out measurements with high angle of incidence
            valid_mask = incidence_angles < self.max_angle_of_incidence
        
        # Additional outlier rejection for the filtered points if needed
        if len(scan_x) > 3:
            # Calculate distances for the already filtered points
            distances = np.sqrt((np.array(scan_x) - robot_x)**2 + (np.array(scan_y) - robot_y)**2)
            outlier_mask = self.reject_outliers(distances, self.outlier_rejection_threshold)
            # Now both masks have the same length
            valid_mask = valid_mask & outlier_mask
        
        # Apply motion compensation during turns
        if is_turning and self.motion_compensation:
            # During turns, we give less weight to all measurements
            # And we're especially cautious about measurements with high incidence angles
            turn_factor = min(1.0, 1.0 - abs(turn_rate) * 2)
            
            # Create adjusted update weights
            incidence_factors = np.ones(len(scan_x))
            if len(scan_x) > 3:
                # Reduce weight based on angle of incidence
                incidence_factors = 1.0 - (incidence_angles / 90.0) * self.angle_of_incidence_factor
                
            # Apply combined factor
            update_factors = incidence_factors * turn_factor
            
            # Now use these factors to scale the occupancy updates
            # First check if we need to expand the grid
            self.check_and_expand_grid(scan_x, scan_y)
            
            # Convert robot position to grid coordinates
            robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
            
            # Convert scan points to grid coordinates
            filtered_scan_x = np.array(scan_x)[valid_mask]
            filtered_scan_y = np.array(scan_y)[valid_mask]
            scan_grid_x, scan_grid_y = self.world_to_grid(filtered_scan_x, filtered_scan_y)
            valid_update_factors = update_factors[valid_mask]
            
            # Create a temporary grid to store updates
            temp_update_grid = np.zeros_like(self.log_odds_grid)
            
            # Process each scan point in batches
            num_points = len(scan_grid_x)
            batch_size = 50
            
            for batch_start in range(0, num_points, batch_size):
                batch_end = min(batch_start + batch_size, num_points)
                batch_grid_x = scan_grid_x[batch_start:batch_end]
                batch_grid_y = scan_grid_y[batch_start:batch_end]
                batch_factors = valid_update_factors[batch_start:batch_end]
                
                # Process scan points
                for i in range(batch_end - batch_start):
                    endpoint_x, endpoint_y = batch_grid_x[i], batch_grid_y[i]
                    update_factor = batch_factors[i]
                    
                    # Skip if endpoint is outside grid
                    if (endpoint_x < 0 or endpoint_x >= self.grid_width or 
                        endpoint_y < 0 or endpoint_y >= self.grid_height):
                        continue
                    
                    # Calculate scan distance for uncertainty model
                    orig_idx = np.where(valid_mask)[0][batch_start + i]
                    scan_distance = np.sqrt(((robot_x - scan_x[orig_idx])**2 + 
                                          (robot_y - scan_y[orig_idx])**2))
                    
                    # Adjust occupancy update based on distance and update factor
                    distance_factor = 1.0 + scan_distance * self.sensor_noise_variance
                    occupied_update = self.log_odds_occupied * update_factor / distance_factor
                    
                    # Get cells along ray
                    cells_x, cells_y = self._bresenham_line(robot_grid_x, robot_grid_y, endpoint_x, endpoint_y)
                    
                    # Mark cells along ray as free except the endpoint
                    for j in range(len(cells_x) - 1):
                        cell_x, cell_y = cells_x[j], cells_y[j]
                        if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                            # Adjust free update based on distance
                            cell_distance = np.sqrt((cell_x - robot_grid_x)**2 + (cell_y - robot_grid_y)**2) * self.resolution
                            free_update = self.log_odds_free * update_factor * (1.0 - cell_distance * self.sensor_noise_variance/2)
                            # Ensure free update doesn't exceed original log odds free value
                            free_update = max(free_update, self.log_odds_free * 0.5)
                            
                            # Apply the update to the temporary grid
                            temp_update_grid[cell_y, cell_x] += free_update
                            
                            # Update tracking for scan matching
                            self.occupancy_grid_total[cell_y, cell_x] += 1
                    
                    # Mark the endpoint as occupied
                    if (endpoint_x >= 0 and endpoint_x < self.grid_width and
                        endpoint_y >= 0 and endpoint_y < self.grid_height):
                        # During turning, increase temporal consistency requirement
                        if is_turning:
                            # Increment consistency counter
                            self.temporal_consistency_grid[endpoint_y, endpoint_x] += 1
                            
                            # Only apply update if we've seen this obstacle multiple times
                            if self.temporal_consistency_grid[endpoint_y, endpoint_x] >= self.consistency_threshold:
                                temp_update_grid[endpoint_y, endpoint_x] += occupied_update
                        else:
                            # Standard update for non-turning case
                            temp_update_grid[endpoint_y, endpoint_x] += occupied_update
                            # Still track consistency
                            self.temporal_consistency_grid[endpoint_y, endpoint_x] += 1
                            
                        # Update tracking for scan matching
                        self.occupancy_grid_visited[endpoint_y, endpoint_x] += 1
                        self.occupancy_grid_total[endpoint_y, endpoint_x] += 1
            
            # Apply all updates at once to the log odds grid
            self.log_odds_grid += temp_update_grid
            
            # Decay temporal consistency grid (slowly forget old observations)
            self.temporal_consistency_grid *= 0.95
            
        else:
            # Use the standard update for non-turning case, but still apply the valid mask
            filtered_scan_x = np.array(scan_x)[valid_mask]
            filtered_scan_y = np.array(scan_y)[valid_mask]
            self.update_grid_vectorized(robot_x, robot_y, filtered_scan_x, filtered_scan_y)
            
            # Update temporal consistency grid if it exists
            if hasattr(self, 'temporal_consistency_grid'):
                # Convert scan points to grid coordinates
                scan_grid_x, scan_grid_y = self.world_to_grid(filtered_scan_x, filtered_scan_y)
                
                # Increment consistency for occupied cells
                for x, y in zip(scan_grid_x, scan_grid_y):
                    if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                        self.temporal_consistency_grid[y, x] += 1
                
                # Decay temporal consistency grid
                self.temporal_consistency_grid *= 0.95
        
        # Clamp log odds values
        self.log_odds_grid = np.clip(self.log_odds_grid, self.log_odds_min, self.log_odds_max)
        
        # Convert log odds to probabilities
        self.grid = 1 - 1 / (1 + np.exp(self.log_odds_grid))
        
        # Update statistics and metadata
        self.metadata['updates'] += 1
        self.metadata['last_update_time'] = time.time()
        
        self.stats['updates'] += 1
        self.stats['free_cell_count'] = np.sum(self.grid < 0.4)
        self.stats['occupied_cell_count'] = np.sum(self.grid > 0.6)
        self.stats['unknown_cell_count'] = self.grid_width * self.grid_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count']
        )
    
    def enhance_obstacles(self, iterations=1):
        """
        Enhance obstacles in the grid to increase safety margins
        Uses dilation to expand occupied areas
        
        Args:
            iterations (int): Number of dilation iterations
            
        Returns:
            array: Dilated obstacle mask
        """
        # Create a binary grid of obstacles (1 = occupied, 0 = free or unknown)
        binary_obstacles = (self.grid > 0.6).astype(np.uint8)
        
        # Create a structuring element for dilation
        # This will expand obstacles by the specified number of cells
        dilated_obstacles = binary_dilation(
            binary_obstacles, 
            iterations=iterations
        )
        
        # Update the grid - any dilated areas become more likely to be occupied
        # but don't overwrite strong free evidence
        dilation_mask = dilated_obstacles & (self.grid < 0.7)
        self.grid[dilation_mask] = 0.65  # Mark as slightly occupied
        
        # Update log odds grid to match
        self.log_odds_grid = np.log(self.grid / (1 - self.grid + 1e-10))
        
        return dilated_obstacles
    
    def save_to_file(self, filename, format='npy', include_metadata=True, dpi=300, 
                    robot_path=None, start_position=None, current_position=None,
                    enhance_for_display=False, show_dynamic=False):
        """
        Save the occupancy grid to a file with enhanced visualization options
        
        Args:
            filename (str): Base filename without extension
            format (str): File format ('png', 'npy', 'csv', or 'all')
            include_metadata (bool): Whether to save metadata
            dpi (int): DPI for image output (PNG only)
            robot_path (list): Optional list of (x,y) coordinates of robot path
            start_position (tuple): Optional (x,y) coordinates of start position
            current_position (tuple): Optional (x,y) coordinates of current position
            enhance_for_display (bool): Whether to enhance obstacles for display (PNG only)
            show_dynamic (bool): Whether to highlight dynamic objects (PNG only)
            
        Returns:
            list: List of saved filenames
        """
        saved_files = []
        
        # Create a timestamped filename if none provided
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"occupancy_grid_{timestamp}"
        
        # Make sure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                return saved_files
        
        # Save as NumPy array (NPY)
        if format.lower() == 'npy' or format.lower() == 'all':
            npy_filename = f"{filename}.npy"
            try:
                # Save the grid data
                np.save(npy_filename, self.grid)
                
                # Save log odds grid as well
                log_odds_filename = f"{filename}_log_odds.npy"
                np.save(log_odds_filename, self.log_odds_grid)
                saved_files.append(log_odds_filename)
                
                # Save visited and total counts
                visited_filename = f"{filename}_visited.npy"
                total_filename = f"{filename}_total.npy"
                np.save(visited_filename, self.occupancy_grid_visited)
                np.save(total_filename, self.occupancy_grid_total)
                saved_files.append(visited_filename)
                saved_files.append(total_filename)
                
                # Save temporal consistency grid if it exists
                if hasattr(self, 'temporal_consistency_grid'):
                    consistency_filename = f"{filename}_consistency.npy"
                    np.save(consistency_filename, self.temporal_consistency_grid)
                    saved_files.append(consistency_filename)
                
                # If metadata requested, save it as a separate JSON file
                if include_metadata:
                    metadata_filename = f"{filename}_metadata.json"
                    
                    # Create a copy of metadata with added stats
                    extended_metadata = self.metadata.copy()
                    extended_metadata.update(self.stats)
                    
                    # Add dimensions
                    extended_metadata['width'] = self.width
                    extended_metadata['height'] = self.height
                    extended_metadata['grid_width'] = self.grid_width
                    extended_metadata['grid_height'] = self.grid_height
                    extended_metadata['origin_x'] = self.origin_x
                    extended_metadata['origin_y'] = self.origin_y
                    extended_metadata['x_min'] = self.x_min
                    extended_metadata['x_max'] = self.x_max
                    extended_metadata['y_min'] = self.y_min
                    extended_metadata['y_max'] = self.y_max
                    
                    # Add path and positions to metadata if provided
                    if robot_path is not None:
                        extended_metadata['robot_path'] = robot_path
                    if start_position is not None:
                        extended_metadata['start_position'] = start_position
                    if current_position is not None:
                        extended_metadata['current_position'] = current_position
                    
                    with open(metadata_filename, 'w') as f:
                        json.dump(extended_metadata, f, indent=4)
                    
                    print(f"Saved grid metadata: {metadata_filename}")
                    saved_files.append(metadata_filename)
                
                print(f"Saved grid as NumPy array: {npy_filename}")
                saved_files.append(npy_filename)
            except Exception as e:
                print(f"Error saving grid as NumPy array: {e}")
        
        # Save as CSV
        if format.lower() == 'csv' or format.lower() == 'all':
            csv_filename = f"{filename}.csv"
            try:
                # Save the grid data as CSV
                np.savetxt(csv_filename, self.grid, delimiter=',')
                
                # Save log odds grid as well
                log_odds_csv = f"{filename}_log_odds.csv"
                np.savetxt(log_odds_csv, self.log_odds_grid, delimiter=',')
                saved_files.append(log_odds_csv)
                
                # If metadata requested, save it as a separate CSV file
                if include_metadata:
                    metadata_csv_filename = f"{filename}_metadata.csv"
                    with open(metadata_csv_filename, 'w') as f:
                        for key, value in self.metadata.items():
                            f.write(f"{key},{value}\n")
                        
                        # Add statistics
                        for key, value in self.stats.items():
                            f.write(f"{key},{value}\n")
                        
                        # Add dimensions
                        f.write(f"width,{self.width}\n")
                        f.write(f"height,{self.height}\n")
                        f.write(f"grid_width,{self.grid_width}\n")
                        f.write(f"grid_height,{self.grid_height}\n")
                        f.write(f"origin_x,{self.origin_x}\n")
                        f.write(f"origin_y,{self.origin_y}\n")
                        f.write(f"x_min,{self.x_min}\n")
                        f.write(f"x_max,{self.x_max}\n")
                        f.write(f"y_min,{self.y_min}\n")
                        f.write(f"y_max,{self.y_max}\n")
                        
                        # Add path and positions to metadata if provided
                        if robot_path is not None:
                            f.write("robot_path_x,")
                            f.write(",".join([str(x) for x, y in robot_path]))
                            f.write("\n")
                            f.write("robot_path_y,")
                            f.write(",".join([str(y) for x, y in robot_path]))
                            f.write("\n")
                        
                        if start_position is not None:
                            f.write(f"start_position_x,{start_position[0]}\n")
                            f.write(f"start_position_y,{start_position[1]}\n")
                        
                        if current_position is not None:
                            f.write(f"current_position_x,{current_position[0]}\n")
                            f.write(f"current_position_y,{current_position[1]}\n")
                    
                    print(f"Saved grid metadata as CSV: {metadata_csv_filename}")
                    saved_files.append(metadata_csv_filename)
                
                print(f"Saved grid as CSV: {csv_filename}")
                saved_files.append(csv_filename)
            except Exception as e:
                print(f"Error saving grid as CSV: {e}")
        
        # For PNG format, we'll import visualization-specific dependencies only if needed
        if format.lower() == 'png' or format.lower() == 'all':
            try:
                # Import matplotlib locally to avoid dependency if not needed
                import matplotlib.pyplot as plt
                import matplotlib.colors as colors
                
                img_filename = f"{filename}.png"
                
                # Get grid data for display, with optional enhancements
                if enhance_for_display or show_dynamic:
                    display_grid = self.get_grid_for_display(enhance=enhance_for_display, show_dynamic=show_dynamic)
                else:
                    display_grid = self.grid
                
                # Create a figure for the image
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Custom colormap with enhanced visualization options
                if enhance_for_display or show_dynamic:
                    # Advanced colormap: white (unknown), light gray (free), black (occupied), blue (enhanced), red (dynamic)
                    cmap = colors.ListedColormap(['white', 'lightgray', 'black', 'darkblue', 'red'])
                    bounds = [0, 0.4, 0.6, 0.8, 0.9, 1]
                    norm = colors.BoundaryNorm(bounds, cmap.N)
                else:
                    # Basic colormap: white (unknown), light gray (free), black (occupied)
                    cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
                    bounds = [0, 0.4, 0.6, 1]
                    norm = colors.BoundaryNorm(bounds, cmap.N)
                
                # Calculate extent for proper display
                extent = [-self.origin_x, self.width - self.origin_x, 
                         -self.origin_y, self.height - self.origin_y]
                
                # Plot the grid
                img = ax.imshow(display_grid, cmap=cmap, norm=norm, origin='lower', extent=extent)
                
                # Add grid lines
                ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                
                # Plot robot path if provided
                if robot_path is not None and len(robot_path) > 0:
                    path_x, path_y = zip(*robot_path)
                    ax.plot(path_x, path_y, 'r-', linewidth=2, label='Robot Path')
                
                # Plot start position if provided
                if start_position is not None:
                    ax.scatter(start_position[0], start_position[1], c='green', s=100, 
                              marker='*', label='Start Position')
                
                # Plot current position if provided
                if current_position is not None:
                    ax.scatter(current_position[0], current_position[1], c='blue', s=100, 
                              marker='*', linewidth=1, label='Final Position')
                
                # Add title and labels
                ax.set_title('Occupancy Grid Map')
                ax.set_xlabel('X (meters)')
                ax.set_ylabel('Y (meters)')
                
                # Add legend if any path or positions were added
                if robot_path is not None or start_position is not None or current_position is not None:
                    ax.legend(loc='best')
                
                # Add metadata as text if requested
                if include_metadata:
                    # Create info for display
                    free_cells = np.sum(self.grid < 0.4)
                    occupied_cells = np.sum(self.grid > 0.6)
                    unknown_cells = self.grid_width * self.grid_height - (free_cells + occupied_cells)
                    
                    metadata_text = (
                        f"Resolution: {self.resolution:.3f}m/cell\n"
                        f"Dimensions: {self.width:.1f}m × {self.height:.1f}m\n"
                        f"Grid Size: {self.grid_width}×{self.grid_height} cells\n"
                        f"Free: {free_cells} cells, Occupied: {occupied_cells} cells\n"
                        f"Updates: {self.metadata['updates']}"
                    )
                    plt.figtext(0.02, 0.02, metadata_text, wrap=True, fontsize=8,
                               bbox=dict(facecolor='white', alpha=0.7))
                
                # Add color legend for grid types if enhanced visualization is used
                if enhance_for_display or show_dynamic:
                    # Create legend patches
                    legend_elements = []
                    
                    # Always include basic elements
                    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='white', alpha=0.7, label='Unknown'))
                    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='lightgray', alpha=0.7, label='Free'))
                    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='black', alpha=0.7, label='Occupied'))
                    
                    # Add enhanced elements if used
                    if enhance_for_display:
                        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='darkblue', alpha=0.7, label='Enhanced'))
                    
                    # Add dynamic elements if used
                    if show_dynamic:
                        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='Dynamic'))
                    
                    # Add the legend
                    ax.legend(handles=legend_elements, loc='upper right', title='Map Legend')
                
                # Save the figure
                plt.savefig(img_filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved grid as image: {img_filename}")
                saved_files.append(img_filename)
            except ImportError:
                print("Warning: Visualization dependencies not available. Cannot save as PNG.")
            except Exception as e:
                print(f"Error saving grid as PNG: {e}")
        
        return saved_files

    def load_from_file(self, filename, format='npy', load_metadata=True):
        """
        Load the occupancy grid from a file
        
        Args:
            filename (str): Filename without extension if format is specified
            format (str): File format ('npy' or 'csv')
            load_metadata (bool): Whether to load metadata if available
            
        Returns:
            bool: Success or failure
        """
        try:
            # Load grid data
            if format.lower() == 'npy':
                npy_filename = f"{filename}.npy"
                self.grid = np.load(npy_filename)
                
                # Try to load log odds grid
                try:
                    log_odds_filename = f"{filename}_log_odds.npy"
                    self.log_odds_grid = np.load(log_odds_filename)
                    
                    # Try to load visited and total counts
                    try:
                        visited_filename = f"{filename}_visited.npy"
                        total_filename = f"{filename}_total.npy"
                        self.occupancy_grid_visited = np.load(visited_filename)
                        self.occupancy_grid_total = np.load(total_filename)
                    except:
                        print("Warning: Could not load visit counts, initializing to defaults")
                        self.occupancy_grid_visited = np.zeros_like(self.grid)
                        self.occupancy_grid_total = np.ones_like(self.grid)
                        
                    # Try to load temporal consistency grid
                    try:
                        consistency_filename = f"{filename}_consistency.npy"
                        self.temporal_consistency_grid = np.load(consistency_filename)
                    except:
                        print("Warning: Could not load consistency grid, initializing to zeros")
                        self.temporal_consistency_grid = np.zeros_like(self.grid)
                        
                except:
                    print("Warning: Could not load log odds grid, reconstructing from probabilities")
                    # Reconstruct log odds grid from probability grid
                    epsilon = 1e-10  # Small constant to avoid log(0) or log(1)
                    prob_grid = np.clip(self.grid, epsilon, 1 - epsilon)
                    self.log_odds_grid = np.log(prob_grid / (1 - prob_grid))
            elif format.lower() == 'csv':
                csv_filename = f"{filename}.csv"
                self.grid = np.loadtxt(csv_filename, delimiter=',')
                
                # Try to load log odds grid
                try:
                    log_odds_csv = f"{filename}_log_odds.csv"
                    self.log_odds_grid = np.loadtxt(log_odds_csv, delimiter=',')
                except:
                    print("Warning: Could not load log odds grid, reconstructing from probabilities")
                    # Reconstruct log odds grid from probability grid
                    epsilon = 1e-10  # Small constant to avoid log(0) or log(1)
                    prob_grid = np.clip(self.grid, epsilon, 1 - epsilon)
                    self.log_odds_grid = np.log(prob_grid / (1 - prob_grid))
                    
                # Initialize visited and total counts
                self.occupancy_grid_visited = np.zeros_like(self.grid)
                self.occupancy_grid_total = np.ones_like(self.grid)
                
                # Initialize temporal consistency grid
                self.temporal_consistency_grid = np.zeros_like(self.grid)
            else:
                print(f"Unsupported format: {format}")
                return False
            
            # Update grid dimensions from loaded data
            self.grid_height, self.grid_width = self.grid.shape
            
            # Load metadata if requested
            if load_metadata:
                metadata_loaded = False
                
                # Try JSON format first
                metadata_json = f"{filename}_metadata.json"
                if os.path.exists(metadata_json):
                    try:
                        with open(metadata_json, 'r') as f:
                            loaded_metadata = json.load(f)
                        self.metadata.update(loaded_metadata)
                        
                        # Update grid properties from metadata
                        if 'resolution' in loaded_metadata:
                            self.resolution = loaded_metadata['resolution']
                        if 'width' in loaded_metadata:
                            self.width = loaded_metadata['width']
                        if 'height' in loaded_metadata:
                            self.height = loaded_metadata['height']
                        if 'x_min' in loaded_metadata:
                            self.x_min = loaded_metadata['x_min']
                            self.x_max = loaded_metadata['x_max']
                            self.y_min = loaded_metadata['y_min']
                            self.y_max = loaded_metadata['y_max']
                        
                        # Update stats if available
                        for stat_key in ['free_cell_count', 'occupied_cell_count', 'unknown_cell_count', 'updates', 'resizes']:
                            if stat_key in loaded_metadata:
                                self.stats[stat_key] = loaded_metadata[stat_key]
                        
                        # Recalculate derived properties
                        self.origin_x = self.width / 2
                        self.origin_y = self.height / 2
                        
                        # Recreate coordinate matrices
                        self.OccupancyGridX, self.OccupancyGridY = np.meshgrid(
                            np.linspace(-self.width/2, self.width/2, self.grid_width),
                            np.linspace(-self.height/2, self.height/2, self.grid_height)
                        )
                        
                        metadata_loaded = True
                    except Exception as e:
                        print(f"Error loading metadata from JSON: {e}")
                
                # Try CSV format if JSON failed
                if not metadata_loaded:
                    metadata_csv = f"{filename}_metadata.csv"
                    if os.path.exists(metadata_csv):
                        try:
                            with open(metadata_csv, 'r') as f:
                                for line in f:
                                    key, value = line.strip().split(',', 1)
                                    try:
                                        # Convert numeric values
                                        self.metadata[key] = float(value)
                                    except ValueError:
                                        self.metadata[key] = value
                            
                            # Update grid properties from metadata
                            if 'resolution' in self.metadata:
                                self.resolution = self.metadata['resolution']
                            if 'width' in self.metadata:
                                self.width = self.metadata['width']
                            if 'height' in self.metadata:
                                self.height = self.metadata['height']
                            
                            # Recalculate derived properties
                            self.origin_x = self.width / 2
                            self.origin_y = self.height / 2
                            
                            # Recreate coordinate matrices
                            self.OccupancyGridX, self.OccupancyGridY = np.meshgrid(
                                np.linspace(-self.width/2, self.width/2, self.grid_width),
                                np.linspace(-self.height/2, self.height/2, self.grid_height)
                            )
                        except Exception as e:
                            print(f"Error loading metadata from CSV: {e}")
            
            print(f"Successfully loaded grid from {filename}.{format}")
            return True
        
        except Exception as e:
            print(f"Error loading grid from file: {e}")
            return False
    
    def get_scan_match_info(self):
        """
        Get information needed for scan matching
        
        Returns:
            dict: Dictionary with scan matching related information
        """
        return {
            'resolution': self.resolution,
            'width': self.width,
            'height': self.height,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'lidar_max_range': self.lidar_max_range,
            'lidar_fov': self.lidar_fov,
            'num_samples_per_rev': self.num_samples_per_rev,
            'angular_step': self.angular_step
        }
    
    def get_occupancy_probability(self, x, y):
        """
        Get the occupancy probability at a specific world coordinate
        
        Args:
            x (float): X coordinate in world frame
            y (float): Y coordinate in world frame
            
        Returns:
            float: Occupancy probability (0 to 1)
        """
        grid_x, grid_y = self.world_to_grid(x, y)
        return self.grid[grid_y, grid_x]
    
    def is_occupied(self, x, y, threshold=0.55):
        """
        Check if a cell is occupied
        
        Args:
            x (float): X coordinate in world frame
            y (float): Y coordinate in world frame
            threshold (float): Occupancy threshold (default 0.55)
            
        Returns:
            bool: True if occupied, False otherwise
        """
        return self.get_occupancy_probability(x, y) > threshold
    
    def is_free(self, x, y, threshold=0.45):
        """
        Check if a cell is free
        
        Args:
            x (float): X coordinate in world frame
            y (float): Y coordinate in world frame
            threshold (float): Free space threshold (default 0.45)
            
        Returns:
            bool: True if free, False otherwise
        """
        return self.get_occupancy_probability(x, y) < threshold

    def get_binary_grid(self, free_threshold=0.3, occupied_threshold=0.7):
        """
        Get a binary representation of the grid
        
        Args:
            free_threshold: Cells with probability below this are free
            occupied_threshold: Cells with probability above this are occupied
            
        Returns:
            array: Binary grid (0 = free, 1 = occupied, 0.5 = unknown)
        """
        binary_grid = np.ones_like(self.grid) * 0.5  # Start with all unknown
        binary_grid[self.grid < free_threshold] = 0.0  # Free
        binary_grid[self.grid > occupied_threshold] = 1.0  # Occupied
        return binary_grid