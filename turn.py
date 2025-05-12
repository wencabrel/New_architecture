import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import math
import os
import time
from matplotlib.widgets import Button
from scipy.ndimage import binary_dilation

def parse_lidar_data(data_string):
    """Parse the LiDAR data string and extract all relevant information"""
    parts = data_string.strip().split()
    
    # Extract header information
    header = parts[0]  # 'LiDAR_E300'
    
    # Extract number of scan points
    num_points = int(parts[1])  # 180
    
    # Extract scan ranges (LiDAR measurements)
    scan_ranges = [float(parts[i]) for i in range(2, 2 + num_points)]
    
    # Extract pose information (x, y, theta)
    pose_idx = 2 + num_points
    x = float(parts[pose_idx])        # -0.001035
    y = float(parts[pose_idx + 1])    # -0.000620
    theta = float(parts[pose_idx + 2]) # 0.383864
    
    # Extract timestamp and robot ID
    timestamp = float(parts[pose_idx + 6])  # 1742988203.467810
    robot_id = parts[pose_idx + 7]          # zjnu-R1
    
    # Last value (possibly speed or other data)
    last_value = float(parts[pose_idx + 8])  # 0.000246
    
    return {
        'header': header,
        'num_points': num_points,
        'scan_ranges': scan_ranges,
        'pose': {
            'x': x,
            'y': y,
            'theta': theta
        },
        'timestamp': timestamp,
        'robot_id': robot_id,
        'last_value': last_value
    }

def convert_scans_to_cartesian(scan_ranges, angle_min, angle_max, pose, 
                              flip_x=False, flip_y=False, reverse_scan=False, flip_theta=False):
    """
    Convert scan ranges to Cartesian coordinates based on robot's pose
    
    Args:
        scan_ranges: List of LiDAR distance measurements
        angle_min: Starting angle of the scan (radians)
        angle_max: Ending angle of the scan (radians)
        pose: Dictionary with 'x', 'y', 'theta' keys for robot's pose
        flip_x: Whether to flip the x-axis
        flip_y: Whether to flip the y-axis
        reverse_scan: Whether to reverse the scan direction
        flip_theta: Whether to negate the orientation angle
    """
    num_points = len(scan_ranges)
    
    # Generate angle array for each measurement
    if reverse_scan:
        angles = np.linspace(angle_max, angle_min, num_points)
    else:
        angles = np.linspace(angle_min, angle_max, num_points)
    
    # Filter out the max range values (11.9 in this case)
    max_range = 11.9
    valid_indices = [i for i, r in enumerate(scan_ranges) if r < max_range]
    valid_ranges = [scan_ranges[i] for i in valid_indices]
    valid_angles = [angles[i] for i in valid_indices]
    
    # Convert from polar to Cartesian coordinates (in robot's local frame)
    x_local = [r * math.cos(angle) for r, angle in zip(valid_ranges, valid_angles)]
    y_local = [r * math.sin(angle) for r, angle in zip(valid_ranges, valid_angles)]
    
    # Apply any coordinate flips to local coordinates
    if flip_x:
        x_local = [-x for x in x_local]
    if flip_y:
        y_local = [-y for y in y_local]
    
    # Get orientation angle
    theta = pose['theta']
    if flip_theta:
        theta = -theta
    
    # Transform to world coordinates based on robot pose
    x_world = [pose['x'] + x_l * math.cos(theta) - y_l * math.sin(theta) for x_l, y_l in zip(x_local, y_local)]
    y_world = [pose['y'] + x_l * math.sin(theta) + y_l * math.cos(theta) for x_l, y_l in zip(x_local, y_local)]
    
    return x_world, y_world

def read_lidar_data_from_file(file_path, max_entries=200):
    """Read LiDAR data from a file, up to max_entries"""
    parsed_data_list = []
    
    try:
        with open(file_path, 'r') as file:
            count = 0
            for line in file:
                if count >= max_entries:
                    break
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    parsed_data = parse_lidar_data(line)
                    parsed_data_list.append(parsed_data)
                    count += 1
                except Exception as e:
                    print(f"Error parsing line {count + 1}: {e}")
                    continue
            
            print(f"Successfully read {len(parsed_data_list)} entries from {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return parsed_data_list

class OccupancyGrid:
    """Enhanced class to handle occupancy grid mapping from LiDAR data with improved functionality"""
    
    def __init__(self, resolution=0.05, initial_width=10, initial_height=10, 
                 expansion_factor=1.5, sensor_noise_variance=0.01):
        """
        Initialize an improved occupancy grid with dynamic resizing capabilities
        
        Args:
            resolution: Grid cell size in meters
            initial_width: Initial width of the grid in meters
            initial_height: Initial height of the grid in meters
            expansion_factor: Factor by which to expand the grid when needed (e.g., 1.5 = 50% expansion)
            sensor_noise_variance: Variance parameter for sensor noise model
        """
        self.resolution = resolution
        self.expansion_factor = expansion_factor
        self.sensor_noise_variance = sensor_noise_variance
        
        # Calculate initial grid dimensions
        self.grid_width = int(initial_width / resolution)
        self.grid_height = int(initial_height / resolution)
        
        # Initialize grid with unknown values (0.5 represents unknown)
        # Values closer to 1 will represent occupied
        # Values closer to 0 will represent free
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5
        
        # Track actual world dimensions (initialized from the center)
        self.width = initial_width
        self.height = initial_height
        
        # Origin of the grid (center of the grid)
        self.origin_x = initial_width / 2
        self.origin_y = initial_height / 2
        
        # World boundaries
        self.x_min = -self.origin_x
        self.x_max = self.origin_x
        self.y_min = -self.origin_y
        self.y_max = self.origin_y
        
        # Log odds version of the grid (for Bayesian updates)
        # Initialize with log odds of 0.5 probability -> log(0.5/0.5) = 0
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))
        
        # Parameters for occupancy update
        self.log_odds_occupied = np.log(0.7/0.3)  # Probability of cell being occupied given a hit
        self.log_odds_free = np.log(0.3/0.7)      # Probability of cell being occupied given a miss
        
        # Maximum and minimum log odds to prevent overflow
        self.log_odds_min = np.log(0.01/0.99)  # ~-4.6
        self.log_odds_max = np.log(0.99/0.01)  # ~4.6
        
        # Additional parameters for sensor model
        self.max_range_variance = 0.1  # Higher variance for cells near max range
        self.near_obstacle_dilation = 2  # How many cells to dilate obstacles by
        
        # Performance tracking
        self.update_count = 0
        self.last_resize_count = 0
        self.resize_frequency = 10  # Check for resize every N updates
        
        # Statistics for map
        self.stats = {
            'free_cell_count': 0,
            'occupied_cell_count': 0,
            'unknown_cell_count': self.grid_width * self.grid_height,
            'updates': 0,
            'resizes': 0
        }
        
        # Add parameters for turn handling
        self.angle_of_incidence_factor = 0.1  # Factor to adjust updates based on angle of incidence
        self.motion_compensation = True       # Whether to compensate for robot motion during scans
        self.scan_history_length = 150          # Number of previous scans to keep for motion analysis
        self.previous_poses = []              # Store previous robot poses to detect turns
        self.turn_detection_threshold = 0.001   # Threshold for detecting turns (radians)
        self.max_angle_of_incidence = 80      # Maximum angle (degrees) for reliable measurements
        self.outlier_rejection_threshold = 3  # Standard deviations for outlier rejection
        self.dynamic_object_threshold = 0.3   # Threshold for detecting potentially dynamic objects
        
        # For temporal filtering
        self.temporal_consistency_grid = np.zeros((self.grid_height, self.grid_width))
        self.consistency_threshold = 3       # Number of consistent observations needed
        
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
        self.temporal_consistency_grid = new_consistency
        
        # Update statistics
        self.stats['unknown_cell_count'] = new_width * new_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count']
        )
        
        print(f"Grid expanded {direction}: New dimensions {new_width}x{new_height} cells, "
              f"{self.width:.1f}x{self.height:.1f} meters")
        
    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices
        
        Args:
            x (float or list or array): X coordinate(s) in world frame
            y (float or list or array): Y coordinate(s) in world frame
            
        Returns:
            tuple: (grid_x, grid_y) Grid indices
        """
        # Convert inputs to numpy arrays if they aren't already
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        
        # Calculate grid indices
        grid_x = np.floor((x_array + self.origin_x) / self.resolution).astype(int)
        grid_y = np.floor((y_array + self.origin_y) / self.resolution).astype(int)
        
        # If input was a single value, return a single value
        if np.isscalar(x) and np.isscalar(y):
            return int(grid_x.item()), int(grid_y.item())
        else:
            return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices to world coordinates (center of cell)
        
        Args:
            grid_x (int or list or array): Grid x index/indices
            grid_y (int or list or array): Grid y index/indices
            
        Returns:
            tuple: (x, y) World coordinates of cell center
        """
        # Convert inputs to numpy arrays if they aren't already
        grid_x_array = np.asarray(grid_x)
        grid_y_array = np.asarray(grid_y)
        
        # Calculate world coordinates
        x = grid_x_array * self.resolution + self.resolution/2 - self.origin_x
        y = grid_y_array * self.resolution + self.resolution/2 - self.origin_y
        
        # If input was a single value, return a single value
        if np.isscalar(grid_x) and np.isscalar(grid_y):
            return x.item(), y.item()
        else:
            return x, y
    
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
    
    def update_grid_vectorized(self, robot_x, robot_y, scan_x, scan_y):
        """
        Update the occupancy grid with a laser scan using vectorized operations
        
        Args:
            robot_x: Robot's x position in world frame
            robot_y: Robot's y position in world frame
            scan_x: List of scan x points in world frame
            scan_y: List of scan y points in world frame
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
                
                # Get cells along ray using Bresenham line algorithm (vectorized version)
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
                
                # Mark the endpoint as occupied
                if (endpoint_x >= 0 and endpoint_x < self.grid_width and
                    endpoint_y >= 0 and endpoint_y < self.grid_height):
                    temp_update_grid[endpoint_y, endpoint_x] += occupied_update
        
        # Apply all updates at once to the log odds grid
        self.log_odds_grid += temp_update_grid
        
        # Clamp log odds values to prevent numerical issues
        self.log_odds_grid = np.clip(self.log_odds_grid, self.log_odds_min, self.log_odds_max)
        
        # Convert log odds to probabilities
        self.grid = 1 - 1 / (1 + np.exp(self.log_odds_grid))
        
        # Update statistics
        self.stats['updates'] += 1
        self.stats['free_cell_count'] = np.sum(self.grid < 0.4)
        self.stats['occupied_cell_count'] = np.sum(self.grid > 0.6)
        self.stats['unknown_cell_count'] = self.grid_width * self.grid_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count']
        )
    
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
    
    def enhance_obstacles(self):
        """
        Enhance obstacles in the grid to increase safety margins
        Uses dilation to expand occupied areas
        """
        # Create a binary grid of obstacles (1 = occupied, 0 = free or unknown)
        binary_obstacles = (self.grid > 0.6).astype(np.uint8)
        
        # Create a structuring element for dilation
        # This will expand obstacles by the specified number of cells
        dilated_obstacles = binary_dilation(
            binary_obstacles, 
            iterations=self.near_obstacle_dilation
        )
        
        # Update the grid - any dilated areas become more likely to be occupied
        # but don't overwrite strong free evidence
        dilation_mask = dilated_obstacles & (self.grid < 0.7)
        self.grid[dilation_mask] = 0.65  # Mark as slightly occupied
        
        # Update log odds grid to match
        self.log_odds_grid = np.log(self.grid / (1 - self.grid + 1e-10))
        
        return dilated_obstacles
    
    def get_grid_for_display(self, enhance=False, show_dynamic=False):
        """
        Get a copy of the grid suitable for display
        
        Args:
            enhance: Whether to enhance obstacles for display
            show_dynamic: Whether to highlight dynamic/inconsistent areas
            
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
            
            # Update the temporary grid with dilated obstacles
            temp_grid[dilated_obstacles] = 0.8
        
        if show_dynamic:
            # Highlight potentially dynamic objects
            dynamic_mask = self.identify_dynamic_objects()
            temp_grid[dynamic_mask] = 0.9  # Special value for dynamic objects
            
        return temp_grid
    
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
    
    def save_to_file(self, filename, format='png', include_metadata=True, dpi=300, 
                    robot_path=None, start_position=None, current_position=None,
                    enhance_for_display=True, show_dynamic=True):
        """
        Save the occupancy grid to a file with improvements
        
        Args:
            filename: Base filename without extension
            format: File format ('png', 'npy', 'csv', or 'all')
            include_metadata: Whether to save metadata
            dpi: DPI for image output
            robot_path: List of (x,y) coordinates of robot path
            start_position: (x,y) coordinates of robot start position
            current_position: (x,y) coordinates of robot current position
            enhance_for_display: Whether to enhance obstacles for display
            show_dynamic: Whether to highlight dynamic objects
        
        Returns:
            list: Paths to saved files
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
        
        # Get the grid for display, potentially enhanced
        display_grid = self.get_grid_for_display(enhance=enhance_for_display, show_dynamic=show_dynamic)
        
        # Save as image (PNG)
        if format.lower() == 'png' or format.lower() == 'all':
            img_filename = f"{filename}.png"
            try:
                # Create a figure for the image
                fig, ax = plt.subplots(figsize=(12, 12))
                
                # Custom colormap: white (unknown), light gray (free), black (occupied), 
                # dark blue (enhanced), red (dynamic)
                cmap = colors.ListedColormap(['white', 'lightgray', 'black', 'darkblue', 'red'])
                bounds = [0, 0.4, 0.6, 0.8, 0.9, 1]
                norm = colors.BoundaryNorm(bounds, cmap.N)
                
                # Calculate world extent based on grid size
                extent = [
                    -self.origin_x, 
                    -self.origin_x + self.width,
                    -self.origin_y, 
                    -self.origin_y + self.height
                ]
                
                # Plot the grid
                img = ax.imshow(display_grid, cmap=cmap, norm=norm, origin='lower',
                               extent=extent)
                
                # Add grid lines (less frequent for better readability)
                grid_interval = max(1, int(5 / self.resolution))  # 5 meter intervals
                ax.set_xticks(np.arange(extent[0], extent[1], grid_interval * self.resolution))
                ax.set_yticks(np.arange(extent[2], extent[3], grid_interval * self.resolution))
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
                    metadata_text = (
                        f"Resolution: {self.resolution:.3f}m/cell\n"
                        f"Dimensions: {self.width:.1f}m × {self.height:.1f}m\n"
                        f"Grid Size: {self.grid_width}×{self.grid_height} cells\n"
                        f"Occupied: {self.stats['occupied_cell_count']} cells\n"
                        f"Free: {self.stats['free_cell_count']} cells\n"
                        f"Unknown: {self.stats['unknown_cell_count']} cells\n"
                        f"Updates: {self.stats['updates']}, Resizes: {self.stats['resizes']}"
                    )
                    plt.figtext(0.02, 0.02, metadata_text, wrap=True, fontsize=8, 
                                bbox=dict(facecolor='white', alpha=0.7))
                
                # Add color legend for dynamic objects if showing them
                if show_dynamic:
                    dynamic_legend = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7)
                    enhanced_legend = plt.Rectangle((0, 0), 1, 1, fc="darkblue", alpha=0.7)
                    occupied_legend = plt.Rectangle((0, 0), 1, 1, fc="black", alpha=0.7)
                    free_legend = plt.Rectangle((0, 0), 1, 1, fc="lightgray", alpha=0.7)
                    unknown_legend = plt.Rectangle((0, 0), 1, 1, fc="white", alpha=0.7)
                    
                    ax.legend([unknown_legend, free_legend, occupied_legend, enhanced_legend, dynamic_legend],
                             ['Unknown', 'Free', 'Occupied', 'Enhanced', 'Dynamic/Inconsistent'],
                             loc='upper right')
                
                # Save the figure
                plt.savefig(img_filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved grid as image: {img_filename}")
                saved_files.append(img_filename)
            except Exception as e:
                print(f"Error saving grid as image: {e}")
        
        # Save as NumPy array (NPY)
        if format.lower() == 'npy' or format.lower() == 'all':
            # Save both probability grid and log odds grid
            prob_npy_filename = f"{filename}_prob.npy"
            log_odds_npy_filename = f"{filename}_log_odds.npy"
            binary_npy_filename = f"{filename}_binary.npy"
            consistency_npy_filename = f"{filename}_consistency.npy"
            
            try:
                # Save the probability grid
                np.save(prob_npy_filename, self.grid)
                saved_files.append(prob_npy_filename)
                
                # Save the log odds grid
                np.save(log_odds_npy_filename, self.log_odds_grid)
                saved_files.append(log_odds_npy_filename)
                
                # Save a binary version of the grid
                binary_grid = self.get_binary_grid()
                np.save(binary_npy_filename, binary_grid)
                saved_files.append(binary_npy_filename)
                
                # Save the consistency grid
                np.save(consistency_npy_filename, self.temporal_consistency_grid)
                saved_files.append(consistency_npy_filename)
                
                # If metadata requested, save it as a separate JSON file
                if include_metadata:
                    metadata_filename = f"{filename}_metadata.json"
                    metadata = {
                        'resolution': self.resolution,
                        'width': self.width,
                        'height': self.height,
                        'grid_width': self.grid_width,
                        'grid_height': self.grid_height,
                        'origin_x': self.origin_x,
                        'origin_y': self.origin_y,
                        'x_min': self.x_min,
                        'x_max': self.x_max,
                        'y_min': self.y_min,
                        'y_max': self.y_max,
                        'stats': self.stats,
                        'turn_detection_threshold': self.turn_detection_threshold,
                        'consistency_threshold': self.consistency_threshold
                    }
                    
                    # Add path and positions to metadata if provided
                    if robot_path is not None:
                        metadata['robot_path'] = robot_path
                    if start_position is not None:
                        metadata['start_position'] = start_position
                    if current_position is not None:
                        metadata['current_position'] = current_position
                    
                    import json
                    with open(metadata_filename, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    
                    print(f"Saved grid metadata: {metadata_filename}")
                    saved_files.append(metadata_filename)
                
                print(f"Saved grid as NumPy arrays: {prob_npy_filename}, {log_odds_npy_filename}, {binary_npy_filename}, {consistency_npy_filename}")
            except Exception as e:
                print(f"Error saving grid as NumPy array: {e}")
        
        # Save as CSV
        if format.lower() == 'csv' or format.lower() == 'all':
            csv_filename = f"{filename}.csv"
            try:
                # Save the grid data as CSV
                np.savetxt(csv_filename, self.grid, delimiter=',')
                
                # If metadata requested, save it as a separate CSV file
                if include_metadata:
                    metadata_csv_filename = f"{filename}_metadata.csv"
                    with open(metadata_csv_filename, 'w') as f:
                        f.write(f"resolution,{self.resolution}\n")
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
                        
                        # Add statistics
                        f.write(f"occupied_cells,{self.stats['occupied_cell_count']}\n")
                        f.write(f"free_cells,{self.stats['free_cell_count']}\n")
                        f.write(f"unknown_cells,{self.stats['unknown_cell_count']}\n")
                        f.write(f"updates,{self.stats['updates']}\n")
                        f.write(f"resizes,{self.stats['resizes']}\n")
                        
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
        
        return saved_files

    # Compatibility method for backward compatibility with original code
    def update_grid(self, robot_x, robot_y, scan_x, scan_y):
        """
        Compatibility method that calls the vectorized update
        """
        return self.update_grid_vectorized(robot_x, robot_y, scan_x, scan_y)
        
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
        
# Replace the update_grid_with_turn_handling method with this fixed version

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
        # Note: We're NOT using the original scan_ranges here, since they have a different length
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
            
            # Apply all updates at once to the log odds grid
            self.log_odds_grid += temp_update_grid
            
            # Decay temporal consistency grid (slowly forget old observations)
            self.temporal_consistency_grid *= 0.95
            
        else:
            # Use the standard update for non-turning case, but still apply the valid mask
            filtered_scan_x = np.array(scan_x)[valid_mask]
            filtered_scan_y = np.array(scan_y)[valid_mask]
            self.update_grid_vectorized(robot_x, robot_y, filtered_scan_x, filtered_scan_y)
            
            # Update temporal consistency grid
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
        
        # Update statistics
        self.stats['updates'] += 1
        self.stats['free_cell_count'] = np.sum(self.grid < 0.4)
        self.stats['occupied_cell_count'] = np.sum(self.grid > 0.6)
        self.stats['unknown_cell_count'] = self.grid_width * self.grid_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count']
        )
    def identify_dynamic_objects(self):
        """
        Identify potentially dynamic objects in the map based on inconsistent updates
        
        Returns:
            2D array: Mask of potentially dynamic objects
        """
        # Areas with high update frequency but intermediate probability
        # may represent dynamic objects
        dynamic_mask = ((self.grid > 0.4) & (self.grid < 0.6) & 
                      (self.temporal_consistency_grid < self.consistency_threshold) &
                      (self.temporal_consistency_grid > 0))
                      
        return dynamic_mask

def animate_lidar_data(parsed_data_list, flip_x=False, flip_y=True, reverse_scan=True, flip_theta=False, 
                      show_occupancy_grid=True, grid_resolution=0.05, save_grid=False,
                      save_format='png', save_path='maps/', use_turn_handling=True):
    """
    Animate LiDAR scans showing robot movement based on pose with interactive zooming
    
    Args:
        parsed_data_list: List of parsed LiDAR data dictionaries
        flip_x: Whether to flip the x-axis
        flip_y: Whether to flip the y-axis
        reverse_scan: Whether to reverse the scan direction
        flip_theta: Whether to negate the orientation angle
        show_occupancy_grid: Whether to show the occupancy grid
        grid_resolution: Resolution of the occupancy grid in meters
        save_grid: Whether to save the final occupancy grid
        save_format: Format to save the grid ('png', 'npy', 'csv', or 'all')
        save_path: Directory to save the grid
        use_turn_handling: Whether to use improved turn handling
    """
    if not parsed_data_list:
        print("No data to animate.")
        return None
    
    # Assuming the LiDAR scan covers 180 degrees (π radians)
    angle_min = -math.pi/2
    angle_max = math.pi/2
    
    # Find max range for consistent scaling by converting all data points
    all_x_points = []
    all_y_points = []
    for parsed_data in parsed_data_list:
        x_points, y_points = convert_scans_to_cartesian(
            parsed_data['scan_ranges'], angle_min, angle_max, parsed_data['pose'],
            flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
        )
        all_x_points.extend(x_points)
        all_y_points.extend(y_points)
    
    # Calculate proper axis limits for visualization
    x_min, x_max = min(all_x_points), max(all_x_points)
    y_min, y_max = min(all_y_points), max(all_y_points)
    
    # Add some padding (20%)
    x_padding = max(1.0, (x_max - x_min) * 0.2)
    y_padding = max(1.0, (y_max - y_min) * 0.2)
    
    # Set limits with padding
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Calculate grid dimensions based on data range
    grid_width = max(20, int(math.ceil((x_max - x_min) * 1.5)))  # Make grid at least 20m wide
    grid_height = max(20, int(math.ceil((y_max - y_min) * 1.5)))  # Make grid at least 20m tall
    
    # Track turn statistics for visualization
    turn_stats = {
        'is_turning': [],
        'turn_rates': [],
        'timestamps': []
    }
    
    # Initialize occupancy grid with the improved version
    if show_occupancy_grid:
        occupancy_grid = OccupancyGrid(
            resolution=grid_resolution, 
            initial_width=grid_width,
            initial_height=grid_height,
            expansion_factor=1.5,
            sensor_noise_variance=0.01
        )
    
    # Store robot path (applying the same transformations)
    robot_path_x = []
    robot_path_y = []
    for data in parsed_data_list:
        x, y = data['pose']['x'], data['pose']['y']
        if flip_x:
            x = -x
        if flip_y:
            y = -y
        robot_path_x.append(x)
        robot_path_y.append(y)
    
    # Calculate time difference between timestamps
    timestamps = [data['timestamp'] for data in parsed_data_list]
    start_time = timestamps[0]
    time_diffs = [t - start_time for t in timestamps]
    
    # Track the current frame index for saving the displayed state
    current_frame_index = [0]  # Using a list to make it mutable inside nested functions
    
    # Create a figure with two subplots side by side if showing occupancy grid
    if show_occupancy_grid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # Set up the occupancy grid image
        # Custom colormap: white (unknown), light gray (free), black (occupied), blue (enhanced), red (dynamic)
        cmap = colors.ListedColormap(['white', 'lightgray', 'black', 'darkblue', 'red'])
        bounds = [0, 0.4, 0.6, 0.8, 0.9, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Initialize the occupancy grid display
        grid_img = ax2.imshow(occupancy_grid.get_grid_for_display(enhance=True), 
                             cmap=cmap, norm=norm, 
                             origin='lower', 
                             extent=[-occupancy_grid.width/2, occupancy_grid.width/2, 
                                     -occupancy_grid.height/2, occupancy_grid.height/2])
        
        # Add reference grid lines
        ax2.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Create a line for robot path on the occupancy grid
        grid_path_line, = ax2.plot([], [], 'r-', linewidth=2, label='Robot Path')
        
        # Also plot the starting position on the grid
        grid_start_point = ax2.scatter([], [], c='green', s=100, marker='*', label='Start')
        
        # Add a star marker for the current robot position
        grid_current_pos = ax2.scatter([], [], c='blue', s=100, marker='*', label='Current Position')
        
        # Add text elements for status information on the grid
        grid_timestamp_text = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, va='top', ha='left', color='blue')
        grid_robot_id_text = ax2.text(0.02, 0.94, "", transform=ax2.transAxes, va='top', ha='left', color='blue')
        grid_pose_text = ax2.text(0.02, 0.90, "", transform=ax2.transAxes, va='top', ha='left', color='blue')
        grid_settings_text = ax2.text(0.02, 0.86, "", transform=ax2.transAxes, va='top', ha='left', color='blue')
        grid_turn_text = ax2.text(0.02, 0.82, "", transform=ax2.transAxes, va='top', ha='left', color='red', 
                                fontweight='bold')
        
        # Add zoom information text
        zoom_info_text = ax2.text(0.5, 0.02, "Left-click: Zoom in | Right-click: Zoom out | Middle-click: Reset zoom", 
                                 transform=ax2.transAxes, va='bottom', ha='center', 
                                 fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add color legend for grid
        # Create custom legend for grid
        unknown_patch = plt.Rectangle((0, 0), 1, 1, fc='white', alpha=0.7)
        free_patch = plt.Rectangle((0, 0), 1, 1, fc='lightgray', alpha=0.7)
        occupied_patch = plt.Rectangle((0, 0), 1, 1, fc='black', alpha=0.7)
        enhanced_patch = plt.Rectangle((0, 0), 1, 1, fc='darkblue', alpha=0.7)
        dynamic_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7)
        
        grid_legend = ax2.legend(
            [unknown_patch, free_patch, occupied_patch, enhanced_patch, dynamic_patch],
            ['Unknown', 'Free', 'Occupied', 'Enhanced', 'Dynamic/Inconsistent'],
            loc='lower right', title='Map Legend'
        )
        
        # Store original axis limits for reset
        original_xlim = ax2.get_xlim()
        original_ylim = ax2.get_ylim()
        
        # Flag to track if animation is running
        is_running = [True]
        
        # Zoom factor for mouse wheel zoom
        zoom_factor = 0.5  # How much to zoom in/out (0.5 = 50% zoom)
        
        # Store the current robot position for centering when following
        current_robot_pos = [0, 0]
        
        # Flag to determine if we're following the robot
        follow_robot = [True]
        
        # Flag to show dynamic objects
        show_dynamic = [False]
        
        # Add a Follow Robot button
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons
        follow_button_ax = plt.axes([0.85, 0.05, 0.1, 0.04])
        follow_button = Button(follow_button_ax, 'Follow Robot', color='lightgoldenrodyellow', hovercolor='0.975')
        
        # Add a Save Map button
        save_button_ax = plt.axes([0.70, 0.05, 0.1, 0.04])
        save_button = Button(save_button_ax, 'Save Map', color='lightblue', hovercolor='0.8')
        
        # Add a Toggle Dynamic Objects button
        dynamic_button_ax = plt.axes([0.55, 0.05, 0.1, 0.04])
        dynamic_button = Button(dynamic_button_ax, 'Show Dynamic', color='lightcoral', hovercolor='coral')
        
        def toggle_follow(event):
            follow_robot[0] = not follow_robot[0]
            follow_button.label.set_text('Following' if follow_robot[0] else 'Not Following')
        
        def toggle_dynamic(event):
            show_dynamic[0] = not show_dynamic[0]
            dynamic_button.label.set_text('Hide Dynamic' if show_dynamic[0] else 'Show Dynamic')
            # Update grid display with current frame's data
            frame = current_frame_index[0]
            update(frame)
            
        def save_current_map(event):
            if not show_occupancy_grid:
                print("Cannot save map - occupancy grid is disabled.")
                return
                
            # Create the save directory if it doesn't exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Generate a timestamp-based filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.join(save_path, f"occupancy_grid_{timestamp}")
            
            # Get the current path based on the frame we're displaying
            current_frame = current_frame_index[0]
            displayed_path_x = robot_path_x[:current_frame+1]
            displayed_path_y = robot_path_y[:current_frame+1]
            
            # Create path coordinates for saving
            displayed_path_coords = list(zip(displayed_path_x, displayed_path_y))
            
            # Get start and current positions from displayed path
            start_pos = (displayed_path_x[0], displayed_path_y[0]) if len(displayed_path_x) > 0 else None
            current_pos = (displayed_path_x[-1], displayed_path_y[-1]) if len(displayed_path_x) > 0 else None
            
            # Save the grid with currently displayed robot path and positions
            occupancy_grid.save_to_file(
                base_filename, 
                format=save_format, 
                include_metadata=True,
                robot_path=displayed_path_coords,
                start_position=start_pos,
                current_position=current_pos,
                enhance_for_display=True,
                show_dynamic=show_dynamic[0]
            )
            
            print(f"\nOccupancy grid map saved to {base_filename}.{save_format} with current robot path and positions")
            
        follow_button.on_clicked(toggle_follow)
        save_button.on_clicked(save_current_map)
        dynamic_button.on_clicked(toggle_dynamic)
        
        # Define click event handler for zooming
        def on_click(event):
            # Only process clicks in the map axis
            if event.inaxes != ax2:
                return
                
            # Get click coordinates
            x, y = event.xdata, event.ydata
            
            # Current axis limits
            xmin, xmax = ax2.get_xlim()
            ymin, ymax = ax2.get_ylim()
            width = xmax - xmin
            height = ymax - ymin
            
            # Left-click: Zoom in
            if event.button == 1:  # Left click
                # Zoom in by 50% around the clicked point
                new_width = width * zoom_factor
                new_height = height * zoom_factor
                ax2.set_xlim(x - new_width/2, x + new_width/2)
                ax2.set_ylim(y - new_height/2, y + new_height/2)
                follow_robot[0] = False  # Turn off follow mode when manually zooming
                follow_button.label.set_text('Not Following')
                
            # Right-click: Zoom out
            elif event.button == 3:  # Right click
                # Zoom out by 200%
                new_width = width / zoom_factor
                new_height = height / zoom_factor
                # Center on the clicked point
                ax2.set_xlim(x - new_width/2, x + new_width/2)
                ax2.set_ylim(y - new_height/2, y + new_height/2)
                
            # Middle-click: Reset zoom
            elif event.button == 2:  # Middle click
                ax2.set_xlim(original_xlim)
                ax2.set_ylim(original_ylim)
                
            # Redraw the figure
            fig.canvas.draw_idle()
            
        # Connect the click event handler
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # Set occupancy grid plot properties
        ax2.set_title('Occupancy Grid Map')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.set_aspect('equal')
        
        # Add legend to grid map
        ax2.legend(loc='upper right')
        
        # Set the LiDAR scan plot in the first subplot
        ax = ax1
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a scatter plot for LiDAR points
    scatter = ax.scatter([], [], c='blue', s=3, label='LiDAR Points')
    
    # Create a scatter plot for robot position
    robot_pos = ax.scatter([], [], c='red', s=100, marker='*', label='Robot Position')
    
    # Create a line for robot path
    path_line, = ax.plot([], [], 'g-', linewidth=2, label='Robot Path')
    
    # Initialize text objects for information display
    timestamp_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left')
    robot_id_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, va='top', ha='left')
    pose_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, va='top', ha='left')
    settings_text = ax.text(0.02, 0.86, "", transform=ax.transAxes, va='top', ha='left')
    
    # Initialize arrow for robot orientation
    arrow = None
    
    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('2D LiDAR Scan Visualization')
        ax.legend(loc='upper right')
        
        # Show orientation settings
        settings_str = f"Settings: flip_x={flip_x}, flip_y={flip_y}, reverse_scan={reverse_scan}, flip_theta={flip_theta}"
        settings_text.set_text(settings_str)
        
        # Set robot ID text
        robot_id = parsed_data_list[0]['robot_id'] if parsed_data_list else "Unknown"
        robot_id_str = f"Robot ID: {robot_id}"
        robot_id_text.set_text(robot_id_str)
        
        if show_occupancy_grid:
            # Initialize the grid path with the starting point
            if len(robot_path_x) > 0:
                grid_start_point.set_offsets([[robot_path_x[0], robot_path_y[0]]])
                grid_current_pos.set_offsets([[robot_path_x[0], robot_path_y[0]]])
            
            # Initialize text on grid
            grid_settings_text.set_text(settings_str)
            grid_robot_id_text.set_text(robot_id_str)
            grid_turn_text.set_text("Turn Detection: Inactive")
            
            return (scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, settings_text, 
                   grid_img, grid_path_line, grid_start_point, grid_current_pos, grid_timestamp_text, grid_robot_id_text, 
                   grid_pose_text, grid_settings_text, grid_turn_text)
        else:
            return scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, settings_text
    
    def update(frame):
        nonlocal arrow
        
        # Update the current frame index
        current_frame_index[0] = frame
        
        parsed_data = parsed_data_list[frame]
        
        # Convert current scan to Cartesian coordinates with configured orientation
        x_points, y_points = convert_scans_to_cartesian(
            parsed_data['scan_ranges'], angle_min, angle_max, parsed_data['pose'],
            flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
        )
        
        # Update LiDAR points
        scatter.set_offsets(np.column_stack((x_points, y_points)))
        
        # Get transformed robot pose for visualization
        robot_x, robot_y = parsed_data['pose']['x'], parsed_data['pose']['y']
        if flip_x:
            robot_x = -robot_x
        if flip_y:
            robot_y = -robot_y
        
        # Get robot orientation
        theta = parsed_data['pose']['theta']
        if flip_theta:
            theta = -theta
        
        # Store current robot position for zoom centering
        if show_occupancy_grid:
            current_robot_pos[0] = robot_x
            current_robot_pos[1] = robot_y
            
            # If following robot is enabled, center the view on the robot
            if follow_robot[0]:
                # Get current zoom level (width and height)
                xmin, xmax = ax2.get_xlim()
                ymin, ymax = ax2.get_ylim()
                width = xmax - xmin
                height = ymax - ymin
                
                # Center on robot position while maintaining zoom level
                ax2.set_xlim(robot_x - width/2, robot_x + width/2)
                ax2.set_ylim(robot_y - height/2, robot_y + height/2)
        
        # Update robot position
        robot_pos.set_offsets([[robot_x, robot_y]])
        
        # Update robot path
        path_line.set_data(robot_path_x[:frame+1], robot_path_y[:frame+1])
        
        # Update text information
        elapsed_time = time_diffs[frame]
        timestamp_str = f"Time: {elapsed_time:.3f}s"
        timestamp_text.set_text(timestamp_str)
        
        # Show robot ID
        robot_id_str = f"Robot ID: {parsed_data['robot_id']}"
        robot_id_text.set_text(robot_id_str)
        
        # Show original pose values
        pose_str = f"Pose: x={parsed_data['pose']['x']:.3f}, y={parsed_data['pose']['y']:.3f}, θ={parsed_data['pose']['theta']:.3f}"
        pose_text.set_text(pose_str)
        
        # Update robot orientation arrow
        if arrow:
            arrow.remove()
        
        # Apply orientation transformation
        arrow_length = 0.5
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)
        
        if flip_x:
            dx = -dx
        if flip_y:
            dy = -dy
            
        arrow = ax.arrow(robot_x, robot_y, dx, dy, 
                        head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # Update occupancy grid if enabled
        if show_occupancy_grid:
            if use_turn_handling:
                # Use the turn-aware update method
                is_turning, turn_rate = occupancy_grid.detect_robot_turning(theta)
                
                # Record turn statistics
                turn_stats['is_turning'].append(is_turning)
                turn_stats['turn_rates'].append(turn_rate)
                turn_stats['timestamps'].append(elapsed_time)
                
                # Update grid with turn handling
                occupancy_grid.update_grid_with_turn_handling(
                    robot_x, robot_y, theta, x_points, y_points, parsed_data['scan_ranges']
                )
                
                # Update turn status display
                turn_status = "TURNING" if is_turning else "STRAIGHT"
                turn_color = "red" if is_turning else "green"
                grid_turn_text.set_text(f"Motion: {turn_status} (rate: {abs(turn_rate):.3f} rad/update)")
                grid_turn_text.set_color(turn_color)
                
                # If dynamic object display is enabled, show them
                if show_dynamic[0]:
                    # Get grid with dynamic objects highlighted
                    display_grid = occupancy_grid.get_grid_for_display(enhance=True, show_dynamic=True)
                else:
                    # Standard display
                    display_grid = occupancy_grid.get_grid_for_display(enhance=True, show_dynamic=False)
                    
                grid_img.set_data(display_grid)
            else:
                # Standard update method
                occupancy_grid.update_grid_vectorized(robot_x, robot_y, x_points, y_points)
                grid_img.set_data(occupancy_grid.get_grid_for_display(enhance=True))
                grid_turn_text.set_text("Turn Detection: Disabled")
            
            # Update the robot path on the grid map
            grid_path_line.set_data(robot_path_x[:frame+1], robot_path_y[:frame+1])
            
            # Update the current position marker
            grid_current_pos.set_offsets([[robot_x, robot_y]])
            
            # Update text information on grid
            grid_timestamp_text.set_text(timestamp_str)
            grid_robot_id_text.set_text(robot_id_str)
            grid_pose_text.set_text(pose_str)
            
            # Display grid statistics
            stats_str = (f"Free: {occupancy_grid.stats['free_cell_count']} cells\n"
                         f"Occupied: {occupancy_grid.stats['occupied_cell_count']} cells\n"
                         f"Unknown: {occupancy_grid.stats['unknown_cell_count']} cells")
            grid_settings_text.set_text(stats_str)
            
            return (scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, 
                   settings_text, arrow, grid_img, grid_path_line, grid_current_pos, grid_timestamp_text, 
                   grid_robot_id_text, grid_pose_text, grid_settings_text, grid_turn_text)
        else:
            return scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, settings_text, arrow
    
    # Create animation with faster frame rate for smoother visualization
    animation = FuncAnimation(fig, update, frames=len(parsed_data_list), 
                             init_func=init, interval=10, blit=False)
    
    # plt.tight_layout()

    if show_occupancy_grid:
        # For dual-plot layout with buttons, use a more flexible approach
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, wspace=0.1)
    else:
        # For single plot, tight_layout works fine
        plt.tight_layout()

    plt.show()
    
    # After the animation completes, create turn analysis plot if we used turn handling
    if save_grid and show_occupancy_grid and use_turn_handling and len(turn_stats['timestamps']) > 0:
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Generate a timestamp-based filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save final map
        base_filename = os.path.join(save_path, f"final_occupancy_grid_{timestamp}")
        
        # Create final path coordinates for saving
        path_coords = list(zip(robot_path_x, robot_path_y))
        
        # Save the final grid state
        occupancy_grid.save_to_file(
            base_filename, 
            format=save_format, 
            include_metadata=True,
            robot_path=path_coords,
            start_position=(robot_path_x[0], robot_path_y[0]) if len(robot_path_x) > 0 else None,
            current_position=(robot_path_x[-1], robot_path_y[-1]) if len(robot_path_x) > 0 else None,
            enhance_for_display=True,
            show_dynamic=True
        )
        print(f"\nFinal occupancy grid map saved to {base_filename}.{save_format}")
        
        # Additionally, save turn analysis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot turn rates
        ax.plot(turn_stats['timestamps'], [abs(r) for r in turn_stats['turn_rates']], 'b-', label='Turn Rate')
        ax.axhline(y=occupancy_grid.turn_detection_threshold, color='r', linestyle='--', 
                  label=f'Turn Threshold ({occupancy_grid.turn_detection_threshold:.2f})')
        
        # Mark turning periods with yellow background
        for i, is_turning in enumerate(turn_stats['is_turning']):
            if is_turning:
                ax.axvspan(turn_stats['timestamps'][i] - 0.1, turn_stats['timestamps'][i] + 0.1, 
                          alpha=0.3, color='yellow')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Turn Rate (rad/update)')
        ax.set_title('Robot Turn Analysis')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save the turn analysis
        turn_analysis_file = os.path.join(save_path, f"turn_analysis_{timestamp}.png")
        plt.savefig(turn_analysis_file, dpi=300)
        plt.close(fig)
        print(f"Turn analysis saved to {turn_analysis_file}")
    
    return animation

def visualize_lidar_data_realtime(file_path, max_entries=200, show_occupancy_grid=True, 
                             grid_resolution=0.05, save_grid=True, save_format='all',
                             use_turn_handling=True):
    """
    Main function to visualize LiDAR data in real-time with occupancy grid mapping
    
    Args:
        file_path: Path to the LiDAR data file
        max_entries: Maximum number of entries to read from the file
        show_occupancy_grid: Whether to show the occupancy grid visualization
        grid_resolution: Resolution of the occupancy grid in meters (smaller = more detail but slower)
        save_grid: Whether to save the final occupancy grid map to a file
        save_format: Format to save the grid ('png', 'npy', 'csv', or 'all')
        use_turn_handling: Whether to use improved turn handling
    """
    print(f"Reading LiDAR data from: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    # Read the data from file
    parsed_data_list = read_lidar_data_from_file(file_path, max_entries)
    
    if not parsed_data_list:
        print("No data was read from the file.")
        return
    
    # Display data summary
    first_timestamp = parsed_data_list[0]['timestamp']
    last_timestamp = parsed_data_list[-1]['timestamp']
    duration = last_timestamp - first_timestamp
    
    print(f"\nData Summary:")
    print(f"  Number of entries: {len(parsed_data_list)}")
    print(f"  Robot ID: {parsed_data_list[0]['robot_id']}")
    print(f"  Data duration: {duration:.2f} seconds")
    
    if show_occupancy_grid:
        print(f"  Starting visualization with occupancy grid mapping (resolution: {grid_resolution}m)...")
        if use_turn_handling:
            print(f"  Turn handling enabled for improved corridor and circular environment mapping")
        if save_grid:
            print(f"  The final occupancy grid will be saved in '{save_format}' format")
    else:
        print(f"  Starting visualization with orientation correction...")
    
    # Create output directory for maps
    maps_dir = "maps"
    if save_grid and not os.path.exists(maps_dir):
        try:
            os.makedirs(maps_dir)
            print(f"  Created directory for maps: {maps_dir}/")
        except Exception as e:
            print(f"  Error creating maps directory: {e}")
    
    # Start the animation with corrected orientation (flip y-axis and reverse scan)
    # These settings should fix the issue with the corner appearing on the wrong side
    animate_lidar_data(
        parsed_data_list,
        flip_x=False,          # Whether to flip the x-axis
        flip_y=False,          # Whether to flip the y-axis (common fix)
        reverse_scan=True,     # Whether to reverse the scan direction (common fix) 
        flip_theta=False,      # Whether to negate the orientation angle
        show_occupancy_grid=show_occupancy_grid,  # Whether to show occupancy grid
        grid_resolution=grid_resolution,          # Resolution of the grid in meters
        save_grid=save_grid,                      # Whether to save the final grid
        save_format=save_format,                  # Format to save the grid
        save_path=maps_dir,                       # Directory to save the grid
        use_turn_handling=use_turn_handling       # Whether to use turn handling improvements
    )

# Main execution
if __name__ == "__main__":
    # File path to read LiDAR data from
    file_path = "./lidar_slam/dataset/raw_data/laser_data_synchronized_data_u_turn_processed_reduced180.clf"
    
    # Run the visualization with improved occupancy grid mapping
    visualize_lidar_data_realtime(
        file_path, 
        max_entries=8000,          # Process up to 800 LiDAR scans
        show_occupancy_grid=True, # Enable occupancy grid mapping
        grid_resolution=0.05,     # Grid resolution in meters (5cm per cell)
        save_grid=True,           # Save the final occupancy grid map
        save_format='png',        # Save in all available formats (png, npy, csv)
        use_turn_handling=True    # Enable improved turn handling for corridors and circular paths
    )