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

class OccupancyGrid:
    """Class to handle occupancy grid mapping from LiDAR data"""
    
    def __init__(self, resolution=0.05, width=20, height=20, init_position=None):
        """
        Initialize an occupancy grid
        
        Args:
            resolution (float): Grid cell size in meters
            width (float): Width of the grid in meters
            height (float): Height of the grid in meters
            init_position (dict): Optional initial position with 'x', 'y', 'theta' keys
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        
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
        
        # Log odds version of the grid (for Bayesian updates)
        # Initialize with log odds of 0.5 probability -> log(0.5/0.5) = 0
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))
        
        # Parameters for occupancy update
        self.log_odds_occupied = math.log(0.7/0.3)  # Probability of cell being occupied given a hit
        self.log_odds_free = math.log(0.3/0.7)      # Probability of cell being occupied given a miss
        
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
    
    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices
        
        Args:
            x (float): X coordinate in world frame
            y (float): Y coordinate in world frame
            
        Returns:
            tuple: (grid_x, grid_y) indices
        """
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
            grid_x (int): X index in grid
            grid_y (int): Y index in grid
            
        Returns:
            tuple: (x, y) coordinates in world frame
        """
        x = grid_x * self.resolution - self.origin_x
        y = grid_y * self.resolution - self.origin_y
        return x, y
    
    def update_grid(self, robot_x, robot_y, scan_x, scan_y):
        """
        Update the occupancy grid with a laser scan
        
        Args:
            robot_x (float): Robot's x position in world coordinates
            robot_y (float): Robot's y position in world coordinates
            scan_x (list): List of scan x points in world coordinates
            scan_y (list): List of scan y points in world coordinates
        """
        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        # Mark cells with scan points as occupied
        for x, y in zip(scan_x, scan_y):
            # Check if the point is within the grid boundaries
            if (abs(x) < self.width/2 and abs(y) < self.height/2):
                # Convert scan point to grid coordinates
                grid_x, grid_y = self.world_to_grid(x, y)
                
                # Mark the cell as occupied (update log odds)
                self.log_odds_grid[grid_y, grid_x] += self.log_odds_occupied
                
                # Also update visit counts for scan matching
                self.occupancy_grid_visited[grid_y, grid_x] += 1
                self.occupancy_grid_total[grid_y, grid_x] += 1
                
                # Use bresenham's line algorithm to identify free cells along the ray
                self.update_cells_along_ray(robot_grid_x, robot_grid_y, grid_x, grid_y)
        
        # Convert log odds back to probabilities
        self.grid = 1 - (1 / (1 + np.exp(self.log_odds_grid)))
        
        # Update metadata
        self.metadata['updates'] += 1
        self.metadata['last_update_time'] = time.time()
    
    def update_cells_along_ray(self, x0, y0, x1, y1):
        """
        Mark cells along a ray from (x0,y0) to (x1,y1) as free using Bresenham's algorithm
        
        Args:
            x0 (int): Starting x grid coordinate
            y0 (int): Starting y grid coordinate
            x1 (int): Ending x grid coordinate
            y1 (int): Ending y grid coordinate
        """
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            # Mark the current cell as free (except the endpoint)
            if x0 != x1 or y0 != y1:  # Don't update the endpoint
                # Ensure we're within grid bounds
                if 0 <= x0 < self.grid_width and 0 <= y0 < self.grid_height:
                    self.log_odds_grid[y0, x0] += self.log_odds_free
                    
                    # Also update visit counts for scan matching
                    self.occupancy_grid_total[y0, x0] += 1
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            
            # Stop if we reach the endpoint or one cell before it
            if (x0 == x1 and y0 == y1) or \
               (x0 + sx == x1 and y0 == y1) or \
               (x0 == x1 and y0 + sy == y1):
                break
    
    def get_grid(self):
        """
        Get a copy of the grid
        
        Returns:
            numpy.ndarray: Copy of the occupancy grid
        """
        return self.grid.copy()
    
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
    
    def checkAndExapndOG(self, x_range, y_range):
        """
        Check if the specified range is covered by the occupancy grid and expand if needed
        
        Args:
            x_range (list): [min_x, max_x] in world coordinates
            y_range (list): [min_y, max_y] in world coordinates
        """
        # Check if we need to expand the grid
        need_expansion = False
        new_width = self.width
        new_height = self.height
        
        # Check if x_range is outside the current grid
        if x_range[0] < -self.width/2:
            expand_left = abs(x_range[0]) - self.width/2
            new_width += expand_left
            need_expansion = True
            
        if x_range[1] > self.width/2:
            expand_right = x_range[1] - self.width/2
            new_width += expand_right
            need_expansion = True
            
        # Check if y_range is outside the current grid
        if y_range[0] < -self.height/2:
            expand_bottom = abs(y_range[0]) - self.height/2
            new_height += expand_bottom
            need_expansion = True
            
        if y_range[1] > self.height/2:
            expand_top = y_range[1] - self.height/2
            new_height += expand_top
            need_expansion = True
        
        # Expand grid if needed
        if need_expansion:
            self.resize_grid(new_width, new_height)
    
    def convertRealXYToMapIdx(self, x_range, y_range):
        """
        Convert world coordinate ranges to grid indices
        
        Args:
            x_range (list): [min_x, max_x] in world coordinates
            y_range (list): [min_y, max_y] in world coordinates
            
        Returns:
            tuple: (x_indices, y_indices) corresponding to the x and y ranges
        """
        x_idx = [
            max(0, min(self.grid_width - 1, int((x + self.origin_x) / self.resolution)))
            for x in x_range
        ]
        
        y_idx = [
            max(0, min(self.grid_height - 1, int((y + self.origin_y) / self.resolution)))
            for y in y_range
        ]
        
        return x_idx, y_idx
    
    def resize_grid(self, new_width, new_height):
        """
        Resize the grid to new dimensions, preserving existing data
        
        Args:
            new_width (float): New width in meters
            new_height (float): New height in meters
        """
        # Calculate new grid dimensions
        new_grid_width = int(new_width / self.resolution)
        new_grid_height = int(new_height / self.resolution)
        
        # Create new grids with unknown values
        new_grid = np.ones((new_grid_height, new_grid_width)) * 0.5
        new_log_odds_grid = np.zeros((new_grid_height, new_grid_width))
        new_visited = np.zeros((new_grid_height, new_grid_width))
        new_total = np.ones((new_grid_height, new_grid_width))
        
        # Calculate offsets to center the old grid in the new one
        old_center_x = self.grid_width // 2
        old_center_y = self.grid_height // 2
        new_center_x = new_grid_width // 2
        new_center_y = new_grid_height // 2
        
        offset_x = new_center_x - old_center_x
        offset_y = new_center_y - old_center_y
        
        # Copy old grid data to new grid
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                new_x = x + offset_x
                new_y = y + offset_y
                
                if 0 <= new_x < new_grid_width and 0 <= new_y < new_grid_height:
                    new_grid[new_y, new_x] = self.grid[y, x]
                    new_log_odds_grid[new_y, new_x] = self.log_odds_grid[y, x]
                    new_visited[new_y, new_x] = self.occupancy_grid_visited[y, x]
                    new_total[new_y, new_x] = self.occupancy_grid_total[y, x]
        
        # Update grid properties
        self.width = new_width
        self.height = new_height
        self.grid_width = new_grid_width
        self.grid_height = new_grid_height
        self.grid = new_grid
        self.log_odds_grid = new_log_odds_grid
        self.occupancy_grid_visited = new_visited
        self.occupancy_grid_total = new_total
        self.origin_x = new_width / 2
        self.origin_y = new_height / 2
        
        # Regenerate coordinate matrices
        self.OccupancyGridX, self.OccupancyGridY = np.meshgrid(
            np.linspace(-new_width/2, new_width/2, new_grid_width),
            np.linspace(-new_height/2, new_height/2, new_grid_height)
        )
        
        # Update metadata
        self.metadata['width'] = new_width
        self.metadata['height'] = new_height
        self.metadata['grid_width'] = new_grid_width
        self.metadata['grid_height'] = new_grid_height
    
    def save_to_file(self, filename, format='npy', include_metadata=True):
        """
        Save the occupancy grid to a file (without visualization dependencies)
        
        Args:
            filename (str): Base filename without extension
            format (str): File format ('npy', 'csv', 'all', or 'png')
            include_metadata (bool): Whether to save metadata
            
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
                
                # If metadata requested, save it as a separate JSON file
                if include_metadata:
                    metadata_filename = f"{filename}_metadata.json"
                    
                    with open(metadata_filename, 'w') as f:
                        json.dump(self.metadata, f, indent=4)
                    
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
                    
                    print(f"Saved grid metadata as CSV: {metadata_csv_filename}")
                    saved_files.append(metadata_csv_filename)
                
                print(f"Saved grid as CSV: {csv_filename}")
                saved_files.append(csv_filename)
            except Exception as e:
                print(f"Error saving grid as CSV: {e}")
        
        # For PNG format, we'll import visualization-specific dependencies only if needed
        if format.lower() == 'png' or format.lower() == 'all':
            try:
                # Import here to avoid circular import and dependency on visualization
                from lidar_slam.utils import file_utils
                
                # If robot path is in metadata, extract it
                robot_path = None
                start_pos = None
                current_pos = None
                
                if 'robot_path' in self.metadata:
                    robot_path = self.metadata['robot_path']
                    if robot_path:
                        start_pos = robot_path[0]
                        current_pos = robot_path[-1]
                
                # Save as image using the utility function
                img_file = file_utils.save_grid_as_image(
                    self.grid,
                    filename,
                    resolution=self.resolution,
                    width=self.width,
                    height=self.height,
                    robot_path=robot_path,
                    start_position=start_pos,
                    current_position=current_pos
                )
                
                if img_file:
                    saved_files.append(img_file)
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