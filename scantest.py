import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import math
import os
import time
from matplotlib.widgets import Button

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
    """Class to handle occupancy grid mapping from LiDAR data"""
    
    def __init__(self, resolution=0.05, width=20, height=20):
        """
        Initialize an occupancy grid
        
        Args:
            resolution: Grid cell size in meters
            width: Width of the grid in meters
            height: Height of the grid in meters
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
        
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int((x + self.origin_x) / self.resolution)
        grid_y = int((y + self.origin_y) / self.resolution)
        
        # Ensure we're within grid bounds
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        x = grid_x * self.resolution - self.origin_x
        y = grid_y * self.resolution - self.origin_y
        return x, y
    
    # Fix for the index error in update_grid method
    # Replace the OccupancyGrid.update_grid method with this improved version

    def update_grid(self, robot_x, robot_y, scan_x, scan_y):
        """
        Update the occupancy grid with a laser scan - with improved bounds checking
        
        Args:
            robot_x: Robot's x position in world coordinates
            robot_y: Robot's y position in world coordinates
            scan_x: List of scan x points in world coordinates
            scan_y: List of scan y points in world coordinates
        """
        # Check if we need to expand the grid
        need_expansion = False
        
        # Calculate the maximum extent of the scan
        min_x = min(scan_x) if scan_x else 0
        max_x = max(scan_x) if scan_x else 0
        min_y = min(scan_y) if scan_y else 0
        max_y = max(scan_y) if scan_y else 0
        
        # Check if scan points are near the grid boundary
        if (abs(min_x) > 0.8 * self.width/2 or abs(max_x) > 0.8 * self.width/2 or
            abs(min_y) > 0.8 * self.height/2 or abs(max_y) > 0.8 * self.height/2):
            need_expansion = True
        
        # Expand the grid if needed - this must be done before processing the scan
        if need_expansion:
            self.expand_grid()
        
        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        # Skip if robot is outside the grid
        if robot_grid_x < 0 or robot_grid_x >= self.grid_width or robot_grid_y < 0 or robot_grid_y >= self.grid_height:
            print(f"Warning: Robot position ({robot_x}, {robot_y}) is outside the grid. Skipping update.")
            return
        
        # Mark cells with scan points as occupied
        for x, y in zip(scan_x, scan_y):
            # Check if the point is within the grid boundaries with a buffer
            buffer = 1.0  # 1 meter buffer
            if (abs(x) >= self.width/2 - buffer or abs(y) >= self.height/2 - buffer):
                continue
                
            # Convert scan point to grid coordinates
            grid_x, grid_y = self.world_to_grid(x, y)
            
            # Double-check the grid bounds to be absolutely sure
            if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
                continue
            
            # Mark the cell as occupied (update log odds)
            self.log_odds_grid[grid_y, grid_x] += self.log_odds_occupied
            
            # Use bresenham's line algorithm to identify free cells along the ray
            self.update_cells_along_ray(robot_grid_x, robot_grid_y, grid_x, grid_y)
        
        # Convert log odds back to probabilities
        self.grid = 1 - (1 / (1 + np.exp(self.log_odds_grid)))

    def update_cells_along_ray(self, x0, y0, x1, y1):
        """Mark cells along a ray from (x0,y0) to (x1,y1) as free using Bresenham's algorithm"""
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            # Mark the current cell as free (except the endpoint)
            if x0 != x1 or y0 != y1:  # Don't update the endpoint
                # Ensure we're within grid bounds - CRITICAL CHECK
                if 0 <= x0 < self.grid_width and 0 <= y0 < self.grid_height:
                    self.log_odds_grid[y0, x0] += self.log_odds_free
            
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

    def expand_grid(self):
        """Expand the grid when needed to accommodate robot movement"""
        # Save current map data
        old_grid = self.grid.copy()
        old_log_odds_grid = self.log_odds_grid.copy()
        old_width = self.width
        old_height = self.height
        old_grid_width = self.grid_width
        old_grid_height = self.grid_height
        
        # Calculate new dimensions (50% larger)
        expansion_factor = 1.5
        new_width = old_width * expansion_factor
        new_height = old_height * expansion_factor
        
        # Create new grids with expanded dimensions
        new_grid_width = int(new_width / self.resolution)
        new_grid_height = int(new_height / self.resolution)
        
        # Initialize new grids
        new_grid = np.ones((new_grid_height, new_grid_width)) * 0.5
        new_log_odds_grid = np.zeros((new_grid_height, new_grid_width))
        
        # Calculate offsets to center old grid in new grid
        offset_x = (new_grid_width - old_grid_width) // 2
        offset_y = (new_grid_height - old_grid_height) // 2
        
        # Copy old grids to center of new grids
        new_grid[offset_y:offset_y+old_grid_height, offset_x:offset_x+old_grid_width] = old_grid
        new_log_odds_grid[offset_y:offset_y+old_grid_height, offset_x:offset_x+old_grid_width] = old_log_odds_grid
        
        # Update map attributes
        self.width = new_width
        self.height = new_height
        self.grid_width = new_grid_width
        self.grid_height = new_grid_height
        self.grid = new_grid
        self.log_odds_grid = new_log_odds_grid
        
        # Update origin to keep it centered
        self.origin_x = new_width / 2
        self.origin_y = new_height / 2
        
        print(f"Grid expanded from {old_width:.1f}x{old_height:.1f}m to {new_width:.1f}x{new_height:.1f}m ({new_grid_width}x{new_grid_height} cells)")
    
    def get_grid_for_display(self):
        """Get a copy of the grid suitable for display"""
        return self.grid.copy()
    
    def save_to_file(self, filename, format='png', include_metadata=True, dpi=300, 
                    robot_path=None, start_position=None, current_position=None):
        """
        Save the occupancy grid to a file
        
        Args:
            filename: Base filename without extension
            format: File format ('png', 'npy', or 'csv')
            include_metadata: Whether to save metadata (resolution, dimensions, etc.)
            dpi: DPI for image output (for PNG format)
            robot_path: List of (x,y) coordinates of robot path
            start_position: (x,y) coordinates of robot start position
            current_position: (x,y) coordinates of robot current position
        
        Returns:
            List of saved filenames
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
        
        # Save as image (PNG)
        if format.lower() == 'png':
            img_filename = f"{filename}.png"
            try:
                # Create a figure for the image
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Custom colormap: white (unknown), black (occupied), light gray (free)
                cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
                bounds = [0, 0.4, 0.6, 1]
                norm = colors.BoundaryNorm(bounds, cmap.N)
                
                # Plot the grid
                img = ax.imshow(self.grid, cmap=cmap, norm=norm, origin='lower',
                               extent=[-self.width/2, self.width/2, -self.height/2, self.height/2])
                
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
                    metadata_text = (
                        f"Resolution: {self.resolution:.3f}m/cell\n"
                        f"Dimensions: {self.width:.1f}m × {self.height:.1f}m\n"
                        f"Grid Size: {self.grid_width}×{self.grid_height} cells"
                    )
                    plt.figtext(0.02, 0.02, metadata_text, wrap=True, fontsize=8)
                
                # Save the figure
                plt.savefig(img_filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved grid as image: {img_filename}")
                saved_files.append(img_filename)
            except Exception as e:
                print(f"Error saving grid as image: {e}")
        
        # Save as NumPy array (NPY)
        if format.lower() == 'npy' or format.lower() == 'all':
            npy_filename = f"{filename}.npy"
            try:
                # Save the grid data
                np.save(npy_filename, self.grid)
                
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
                        'origin_y': self.origin_y
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

#########################################################################
# NEW SCAN MATCHING ALGORITHM CLASSES AND FUNCTIONS (FLOWCHART INTEGRATION)
#########################################################################

class PoseEstimate:
    """Class to represent a robot pose estimate with uncertainty"""
    
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """Initialize a pose estimate"""
        self.x = x
        self.y = y
        self.theta = theta
        
        # Covariance matrix for x, y, theta
        self.covariance = np.eye(3) * 0.01  # Default small uncertainty
        
    def to_dict(self):
        """Convert pose to dictionary format compatible with existing code"""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta
        }
    
    def from_dict(self, pose_dict):
        """Set pose from dictionary format"""
        self.x = pose_dict['x']
        self.y = pose_dict['y']
        self.theta = pose_dict['theta']
        return self
    
    def copy(self):
        """Create a copy of this pose estimate"""
        new_pose = PoseEstimate(self.x, self.y, self.theta)
        new_pose.covariance = self.covariance.copy()
        return new_pose

class ImprovedScanMatchingLocalization:
    """
    Improved implementation of scan matching localization using ICP algorithm
    with adaptive parameters, alignment reset, and aggressive resampling to prevent unmapped regions.
    """
    
    def __init__(self, occupancy_grid=None, debug_level=1):
        """
        Initialize the improved scan matching system
        
        Args:
            occupancy_grid: OccupancyGrid object representing the map
            debug_level: 0=none, 1=basic info, 2=detailed, 3=verbose
        """
        self.map = occupancy_grid
        self.debug_level = debug_level
        
        # Default ICP parameters (will be adjusted adaptively)
        self.max_iterations = 15
        self.convergence_threshold = 0.001
        self.max_correspondence_distance = 1.5  # meters
        
        # Parameters for adaptive adjustment
        self.default_correspondence_distance = 1.5  # Starting value
        self.max_possible_correspondence_distance = 3.0  # Maximum allowed value
        self.min_correspondence_distance = 1.0  # Minimum allowed value
        
        # For aggressive resampling (when no correspondences found)
        self.aggressive_max_correspondence_distance = 5.0  # Much larger search radius
        self.aggressive_max_resampling_attempts = 5  # How many times to try resampling
        
        # Occupancy threshold (will be adjusted adaptively)
        self.occupancy_threshold = 0.55
        self.default_occupancy_threshold = 0.55  # Starting value
        self.min_occupancy_threshold = 0.45  # Minimum allowed value
        
        # For aggressive resampling
        self.aggressive_min_occupancy_threshold = 0.35  # Much lower threshold
        
        # Motion validation parameters (will be adjusted adaptively)
        self.max_translation_per_frame = 0.5  # meters
        self.default_max_translation = 0.5  # Starting value
        self.max_possible_translation = 1.0  # Maximum allowed value
        
        self.max_rotation_per_frame = 0.5  # radians (~28 degrees)
        self.default_max_rotation = 0.5  # Starting value
        self.max_possible_rotation = 0.8  # Maximum allowed value
        
        # Current estimated trajectory
        self.trajectory = []  # List of PoseEstimate objects
        
        # Last matched pose (used as reference for next match)
        self.last_matched_pose = None
        
        # Keep track of odometry poses for comparison
        self.odometry_trajectory = []
        
        # Flag to determine if we've loaded a map or are building it
        self.mapping_mode = True if occupancy_grid is None else False
        
        # For visualization purposes
        self.current_visualization_data = None
        
        # Keep track of match count for logging
        self.match_count = 0
        
        # Flag to track if we've built the map enough
        self.map_built = False
        
        # Track the building progress for the map
        self.map_build_progress = 0
        
        # Alignment reset tracking
        self.frames_since_last_reset = 0
        self.reset_interval = 50  # Check alignment every 50 frames
        self.drift_threshold = 0.7  # Trigger reset if drift exceeds 0.7m
        
        # Match quality tracking
        self.match_qualities = []  # Track recent match qualities
        self.quality_history_size = 5  # How many recent matches to consider
        
        # Health monitoring
        self.consecutive_poor_matches = 0
        self.match_quality_threshold = 0.4  # Threshold for a "good" match
        
        # Recovery mode
        self.in_recovery_mode = False
        self.recovery_counter = 0
        self.recovery_frames = 5  # How many frames to stay in recovery mode
        
        # Map expansion tracking
        self.should_expand_map = False
        self.force_map_expansion = False  # Used for emergency expansion
        
        # Resampling stats
        self.resampling_attempts = 0  # Track how many times we've had to resample
        self.frames_with_resampling = 0  # Track how many frames needed resampling
        
        if self.debug_level > 0:
            print("[ScanMatcher] Initialized with adaptive parameters and aggressive resampling")
            print(f"[ScanMatcher]   - Base correspondence distance: {self.default_correspondence_distance}m (can increase to {self.max_possible_correspondence_distance}m)")
            print(f"[ScanMatcher]   - Base occupancy threshold: {self.occupancy_threshold} (can decrease to {self.min_occupancy_threshold})")
            print(f"[ScanMatcher]   - Base max translation: {self.max_translation_per_frame}m (can increase to {self.max_possible_translation}m)")
            print(f"[ScanMatcher]   - Alignment reset interval: {self.reset_interval} frames (drift threshold: {self.drift_threshold}m)")
            print(f"[ScanMatcher]   - Aggressive resampling enabled (max search radius: {self.aggressive_max_correspondence_distance}m, min threshold: {self.aggressive_min_occupancy_threshold})")
            print("\n" + "="*80)
            print("        POSE INFORMATION FOR EACH SCAN MATCH WILL BE PRINTED BELOW")
            print("="*80 + "\n")
    
    def processSensorData(self, lidar_data, initial_pose=None, angle_min=-math.pi/2, angle_max=math.pi/2, 
                         flip_x=False, flip_y=False, reverse_scan=False, flip_theta=False):
        """
        Process a sequence of LiDAR scans to localize the robot
        
        Args:
            lidar_data: List of parsed LiDAR data dictionaries
            initial_pose: Initial pose estimate (PoseEstimate object or None)
            angle_min: Starting angle of the scan (radians)
            angle_max: Ending angle of the scan (radians)
            flip_x: Whether to flip the x-axis
            flip_y: Whether to flip the y-axis
            reverse_scan: Whether to reverse the scan direction
            flip_theta: Whether to negate the orientation angle
            
        Returns:
            List of updated pose estimates (trajectory)
        """
        if self.debug_level > 0:
            print("[ScanMatcher] Processing sensor data with ICP scan matching...")
        
        # Initialize trajectory with initial pose if provided
        if initial_pose:
            self.last_matched_pose = initial_pose
            self.trajectory = [initial_pose.copy()]
            self.odometry_trajectory = [initial_pose.copy()]
        else:
            # Use the first scan's pose as initial estimate
            first_pose_dict = lidar_data[0]['pose']
            initial_pose = PoseEstimate(
                first_pose_dict['x'], 
                first_pose_dict['y'], 
                first_pose_dict['theta']
            )
            self.last_matched_pose = initial_pose
            self.trajectory = [initial_pose.copy()]
            self.odometry_trajectory = [initial_pose.copy()]
        
        # Set the minimum number of frames to build the map before matching
        map_build_frames = 20  # Frames dedicated to building the initial map
        
        # Process each scan
        for i, scan_data in enumerate(lidar_data):
            if i == 0 and initial_pose:
                # Skip first scan if we already set the initial pose
                continue
            
            if self.debug_level > 0:
                print(f"\r[ScanMatcher] Processing scan {i+1}/{len(lidar_data)}", end="")
            
            # First, store the odometry pose
            odometry_pose = PoseEstimate().from_dict(scan_data['pose'])
            self.odometry_trajectory.append(odometry_pose.copy())
            
            # Convert from polar to Cartesian coordinates
            scan_x, scan_y = convert_scans_to_cartesian(
                scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
                flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
            )
            
            # Update the map with current scan
            if self.map:
                # Check if we need to expand the map
                if (self.should_expand_map or self.force_map_expansion) and hasattr(self.map, 'expand_grid'):
                    self.map.expand_grid()
                    self.should_expand_map = False
                    self.force_map_expansion = False
                
                # Update the map with the current scan
                try:
                    self.map.update_grid(
                        odometry_pose.x,  # Use odometry pose for mapping
                        odometry_pose.y,
                        scan_x,
                        scan_y
                    )
                except Exception as e:
                    print(f"\n[ScanMatcher] Warning: Error updating map: {e}")
                    # If we get grid bounds errors, force map expansion next frame
                    self.force_map_expansion = True
                
                # Track the building progress
                self.map_build_progress = (i * 100) // map_build_frames if i <= map_build_frames else 100
                
                # DEBUG: Print map statistics every 10 frames
                if i % 10 == 0 and self.debug_level > 1:
                    try:
                        occupied_cells = np.sum(self.map.grid > self.occupancy_threshold)
                        total_cells = self.map.grid_width * self.map.grid_height
                        print(f"\n[ScanMatcher] Map stats: Min={np.min(self.map.grid):.3f}, "
                              f"Max={np.max(self.map.grid):.3f}, "
                              f"Mean={np.mean(self.map.grid):.3f}, "
                              f"Occupied={occupied_cells}/{total_cells} cells "
                              f"({occupied_cells/total_cells*100:.2f}%)")
                    except Exception as e:
                        print(f"\n[ScanMatcher] Error calculating map stats: {e}")
            
            # Check if we've built enough of the map
            if i >= map_build_frames and not self.map_built:
                self.map_built = True
                print(f"\n[ScanMatcher] Map building complete after {i+1} frames. Starting scan matching.")
            
            # Only perform scan matching if we have a map and have built it enough
            if self.map_built:
                # Check if we need to reset alignment between odometry and scan matcher
                alignment_reset = self.checkAndResetAlignment()
                if alignment_reset:
                    print(f"[ScanMatcher] Alignment reset performed at frame {i}")
                    continue  # Skip this frame's scan matching
                
                # Now adapt parameters based on matching history
                self.adaptParametersBasedOnMatchQuality()
                
                # Get the relative odometry movement since last frame
                relative_dx = odometry_pose.x - self.odometry_trajectory[-2].x
                relative_dy = odometry_pose.y - self.odometry_trajectory[-2].y
                relative_dtheta = odometry_pose.theta - self.odometry_trajectory[-2].theta
                
                # Apply this relative movement to our last matched pose as the initial guess
                initial_guess = self.last_matched_pose.copy()
                initial_guess.x += relative_dx 
                initial_guess.y += relative_dy
                initial_guess.theta += relative_dtheta
                
                # Print poses before matching
                self.print_pose_comparison(
                    match_num=self.match_count+1,
                    previous_pose=self.last_matched_pose,
                    odometry_pose=odometry_pose,
                    initial_guess=initial_guess,
                    stage="BEFORE MATCHING"
                )
                
                # Check for emergency alignment reset if drift is extreme
                odom_guess_diff_x = odometry_pose.x - initial_guess.x
                odom_guess_diff_y = odometry_pose.y - initial_guess.y
                odom_guess_diff_dist = np.sqrt(odom_guess_diff_x**2 + odom_guess_diff_y**2)
                
                if odom_guess_diff_dist > 2.0:  # More than 2 meters difference is extreme
                    print(f"\n[ScanMatcher] EMERGENCY: Extreme drift detected ({odom_guess_diff_dist:.2f}m). "
                          f"Forcing immediate alignment reset.")
                    
                    # Create reset pose using odometry position but keeping matched orientation
                    reset_pose = PoseEstimate(
                        odometry_pose.x,
                        odometry_pose.y,
                        self.last_matched_pose.theta
                    )
                    
                    # Update trajectory
                    self.trajectory.append(reset_pose.copy())
                    self.last_matched_pose = reset_pose
                    
                    # Reset counters and flags
                    self.frames_since_last_reset = 0
                    self.consecutive_poor_matches = 0
                    
                    # Skip to next frame
                    self.match_count += 1
                    continue
                
                # Check if we're in recovery mode
                if self.in_recovery_mode:
                    # In recovery mode, use odometry directly for a few frames
                    self.recovery_counter += 1
                    
                    # Print recovery mode status
                    print(f"\n[ScanMatcher] In recovery mode (frame {self.recovery_counter}/{self.recovery_frames})")
                    
                    # Create a pose that's a blend between odometry and last matched
                    recovery_pose = self.createRecoveryPose(odometry_pose, self.last_matched_pose)
                    match_info = {
                        'iterations': 0,
                        'final_score': 0.5,  # Arbitrary middle score
                        'error': 0.0,
                        'correspondences': 0,
                        'resampling_attempts': 0
                    }
                    
                    # Print recovery pose
                    self.print_pose_comparison(
                        match_num=self.match_count+1,
                        previous_pose=self.last_matched_pose,
                        odometry_pose=odometry_pose,
                        initial_guess=initial_guess,
                        estimated_pose=recovery_pose,
                        match_info=match_info,
                        stage="RECOVERY MODE"
                    )
                    
                    # Update trajectory with recovery pose
                    self.trajectory.append(recovery_pose.copy())
                    self.last_matched_pose = recovery_pose
                    
                    # Exit recovery mode after enough frames
                    if self.recovery_counter >= self.recovery_frames:
                        self.in_recovery_mode = False
                        self.recovery_counter = 0
                        self.consecutive_poor_matches = 0
                        print(f"\n[ScanMatcher] Exiting recovery mode")
                else:
                    # Normal mode - match current scan against the map using ICP
                    matched_pose, match_info = self.matchScan(scan_x, scan_y, initial_guess)
                    
                    # Check if we have the special case of resampling
                    if match_info['resampling_attempts'] > 0:
                        resampling_str = f"[ScanMatcher] Used aggressive resampling - {match_info['resampling_attempts']} attempts needed"
                        if match_info['correspondences'] > 0:
                            resampling_str += f", found {match_info['correspondences']} correspondences"
                        print(f"\n{resampling_str}")
                        
                        self.frames_with_resampling += 1
                        self.resampling_attempts += match_info['resampling_attempts']
                        
                    # Check for boundary issues - if many points are out of bounds, flag for map expansion
                    if self.checkForMapBoundaryIssues(scan_x, scan_y, matched_pose):
                        self.should_expand_map = True
                        
                    # Validate the match - check if the movement is reasonable
                    is_valid = self.validateMatch(matched_pose, self.last_matched_pose, match_info)
                    
                    # Update parameters based on match quality for next frame
                    self.adaptParametersBasedOnMatchQuality(match_info)
                    
                    if is_valid:
                        # Reset the consecutive failures counter
                        self.consecutive_poor_matches = 0
                        
                        # Print final matched pose
                        self.print_pose_comparison(
                            match_num=self.match_count+1,
                            previous_pose=self.last_matched_pose,
                            odometry_pose=odometry_pose,
                            initial_guess=initial_guess,
                            estimated_pose=matched_pose,
                            match_info=match_info,
                            stage="AFTER MATCHING (VALID)"
                        )
                        
                        # Update trajectory with the matched pose
                        self.trajectory.append(matched_pose.copy())
                        self.last_matched_pose = matched_pose
                        
                        if self.debug_level > 1:
                            print(f"\n[ScanMatcher] Valid match found. Score: {match_info['final_score']:.4f}")
                    else:
                        # Increment the consecutive failures counter
                        self.consecutive_poor_matches += 1
                        
                        # If match is invalid, use the odometry pose with small correction
                        corrected_pose = self.applySmallCorrection(odometry_pose, self.last_matched_pose)
                        
                        # Print corrected pose 
                        self.print_pose_comparison(
                            match_num=self.match_count+1,
                            previous_pose=self.last_matched_pose,
                            odometry_pose=odometry_pose,
                            initial_guess=initial_guess,
                            estimated_pose=matched_pose,
                            corrected_pose=corrected_pose,
                            match_info=match_info,
                            stage="AFTER MATCHING (INVALID - USING CORRECTION)"
                        )
                        
                        self.trajectory.append(corrected_pose.copy())
                        self.last_matched_pose = corrected_pose
                        
                        # Check if we need to enter recovery mode
                        if self.consecutive_poor_matches >= 3:
                            print(f"\n[ScanMatcher] ⚠️ {self.consecutive_poor_matches} consecutive match failures! Entering recovery mode.")
                            self.in_recovery_mode = True
                            self.recovery_counter = 0
                        else:
                            if self.debug_level > 0:
                                print(f"\n[ScanMatcher] ⚠️ Invalid match rejected! Using odometry with correction.")
                
                # Increment match count
                self.match_count += 1
            else:
                # In mapping mode or early frames, use odometry for trajectory
                current_pose = PoseEstimate().from_dict(scan_data['pose'])
                self.trajectory.append(current_pose.copy())
                self.last_matched_pose = current_pose
                
                # Print the building progress
                if i % 5 == 0:
                    print(f"\n[ScanMatcher] Building map... {self.map_build_progress}% complete")
        
        if self.debug_level > 0:
            print(f"\n[ScanMatcher] Processed {len(lidar_data)} scans. Trajectory contains {len(self.trajectory)} poses.")
            if self.frames_with_resampling > 0:
                print(f"[ScanMatcher] Aggressive resampling was used in {self.frames_with_resampling} frames " 
                      f"({self.frames_with_resampling/self.match_count*100:.1f}% of matches).")
                print(f"[ScanMatcher] Average of {self.resampling_attempts/self.frames_with_resampling:.1f} " 
                      f"resampling attempts per frame when needed.")
        return self.trajectory
    
    def adaptParametersBasedOnMatchQuality(self, match_info=None):
        """
        Adaptively adjust parameters based on recent match quality
        
        Args:
            match_info: Information from the last match attempt
        """
        # If we have match info, add it to our history
        if match_info is not None:
            self.match_qualities.append({
                'score': match_info['final_score'],
                'correspondences': match_info['correspondences'],
                'error': match_info['error']
            })
            
            # Keep only the most recent N matches
            if len(self.match_qualities) > self.quality_history_size:
                self.match_qualities.pop(0)
        
        # If we don't have enough history yet, use default settings
        if len(self.match_qualities) < 2:
            return
        
        # Calculate the average match quality
        avg_score = sum(q['score'] for q in self.match_qualities) / len(self.match_qualities)
        avg_correspondences = sum(q['correspondences'] for q in self.match_qualities) / len(self.match_qualities)
        
        # Check if we're having matching problems
        poor_match = avg_score < self.match_quality_threshold or avg_correspondences < 10
        
        # Get the most recent match result
        last_match = self.match_qualities[-1]
        
        # Calculate adaptive parameter adjustments
        if poor_match:
            self.consecutive_poor_matches += 1
            
            # Adaptively increase search parameters based on consecutive poor matches
            adjustment_factor = min(1.0, 0.2 * self.consecutive_poor_matches)  # Up to 100% adjustment
            
            # Increase search radius
            self.max_correspondence_distance = min(
                self.max_possible_correspondence_distance,
                self.default_correspondence_distance * (1.0 + adjustment_factor)
            )
            
            # Lower occupancy threshold
            self.occupancy_threshold = max(
                self.min_occupancy_threshold,
                self.default_occupancy_threshold * (1.0 - adjustment_factor * 0.3)
            )
            
            # Increase motion limits
            self.max_translation_per_frame = min(
                self.max_possible_translation,
                self.default_max_translation * (1.0 + adjustment_factor)
            )
            
            self.max_rotation_per_frame = min(
                self.max_possible_rotation,
                self.default_max_rotation * (1.0 + adjustment_factor * 0.5)
            )
            
            if self.debug_level > 1 and self.consecutive_poor_matches > 0:
                print(f"\n[ScanMatcher] Low match quality detected ({self.consecutive_poor_matches} consecutive). Adapting parameters:")
                print(f"  - Correspondence distance: {self.max_correspondence_distance:.2f}m")
                print(f"  - Occupancy threshold: {self.occupancy_threshold:.2f}")
                print(f"  - Max translation: {self.max_translation_per_frame:.2f}m")
        else:
            # Good match, gradually return to default values
            self.consecutive_poor_matches = 0
            
            # Gradually move back towards defaults (10% step)
            self.max_correspondence_distance = self.max_correspondence_distance * 0.9 + self.default_correspondence_distance * 0.1
            self.occupancy_threshold = self.occupancy_threshold * 0.9 + self.default_occupancy_threshold * 0.1
            self.max_translation_per_frame = self.max_translation_per_frame * 0.9 + self.default_max_translation * 0.1
            self.max_rotation_per_frame = self.max_rotation_per_frame * 0.9 + self.default_max_rotation * 0.1
    
    def checkAndResetAlignment(self):
        """
        Check alignment between odometry and scan matcher, and reset if necessary
        
        Returns:
            True if alignment was reset, False otherwise
        """
        # Make sure we have enough data
        if len(self.odometry_trajectory) < 2 or len(self.trajectory) < 1:
            return False
        
        # Increment counter for frames since last reset
        self.frames_since_last_reset += 1
        
        # Only check at specified interval
        if self.frames_since_last_reset < self.reset_interval:
            return False
        
        # Get the most recent odometry pose
        current_odom = self.odometry_trajectory[-1]
        
        # Get the current scan-matched pose
        current_matched = self.trajectory[-1]
        
        # Calculate drift between odometry and scan matcher
        drift_x = current_odom.x - current_matched.x
        drift_y = current_odom.y - current_matched.y
        drift_dist = math.sqrt(drift_x**2 + drift_y**2)
        
        if self.debug_level > 0:
            print(f"\n[ScanMatcher] Alignment check - Current drift: {drift_dist:.2f}m between odometry and scan matcher")
        
        # Reset alignment if drift exceeds threshold
        if drift_dist > self.drift_threshold:
            if self.debug_level > 0:
                print(f"[ScanMatcher] Excessive drift detected! Odometry: ({current_odom.x:.2f}, {current_odom.y:.2f}), "
                      f"Matched: ({current_matched.x:.2f}, {current_matched.y:.2f})")
            
            # Create a new pose that uses the odometry position but keeps the scan matcher's orientation
            reset_pose = PoseEstimate(
                current_odom.x, 
                current_odom.y,
                current_matched.theta  # Keep the scan matcher's orientation estimate
            )
            
            # Update the last matched pose
            self.last_matched_pose = reset_pose
            
            # Add to trajectory
            self.trajectory.append(reset_pose.copy())
            
            # Reset the counter
            self.frames_since_last_reset = 0
            
            # Reset parameters to defaults when realigning
            self.max_correspondence_distance = self.default_correspondence_distance
            self.occupancy_threshold = self.default_occupancy_threshold
            self.max_translation_per_frame = self.default_max_translation
            self.max_rotation_per_frame = self.default_max_rotation
            
            if self.debug_level > 0:
                print(f"[ScanMatcher] ALIGNMENT RESET to odometry position: ({reset_pose.x:.2f}, {reset_pose.y:.2f})")
                print(f"[ScanMatcher] Parameters reset to defaults")
            
            return True
        
        # If we performed a check but didn't reset, still reset the counter
        self.frames_since_last_reset = 0
        return False
    
    def createRecoveryPose(self, odometry_pose, last_matched_pose):
        """
        Create a recovery pose by blending odometry and last matched pose
        
        Args:
            odometry_pose: Current odometry pose
            last_matched_pose: Last matched pose
            
        Returns:
            Recovery pose
        """
        # Calculate relative movement from odometry
        if len(self.odometry_trajectory) < 2:
            return odometry_pose.copy()
            
        last_odometry_pose = self.odometry_trajectory[-2]
        relative_dx = odometry_pose.x - last_odometry_pose.x
        relative_dy = odometry_pose.y - last_odometry_pose.y
        relative_dtheta = odometry_pose.theta - last_odometry_pose.theta
        
        # Create recovery pose by using odometry movement from last matched pose
        recovery_pose = last_matched_pose.copy()
        recovery_pose.x += relative_dx
        recovery_pose.y += relative_dy
        recovery_pose.theta += relative_dtheta
        
        return recovery_pose
    
    def checkForMapBoundaryIssues(self, scan_x, scan_y, pose):
        """
        Check if the current scan is near map boundaries
        
        Args:
            scan_x, scan_y: Scan points
            pose: Current pose
            
        Returns:
            True if map expansion is needed
        """
        if self.map is None:
            return False
            
        # Create points array from scan
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Transform points to world frame
        world_points = self.transformPointsToWorld(scan_points, pose)
        
        # Count how many points are near the boundary
        buffer = 2.0  # 2 meter buffer
        boundary_points = 0
        
        map_width = self.map.width
        map_height = self.map.height
        
        for point in world_points:
            # Check if point is near the boundary
            if (abs(point[0]) >= map_width/2 - buffer or 
                abs(point[1]) >= map_height/2 - buffer):
                boundary_points += 1
        
        # If more than 20% of points are near boundary, suggest expansion
        if boundary_points > 0.2 * len(world_points):
            if self.debug_level > 0:
                print(f"\n[ScanMatcher] Warning: {boundary_points} scan points ({boundary_points/len(world_points)*100:.1f}%) "
                     f"are near map boundaries. Map expansion recommended.")
            return True
            
        return False
    
    def print_pose_comparison(self, match_num, previous_pose, odometry_pose, initial_guess, 
                            estimated_pose=None, corrected_pose=None, match_info=None, stage=""):
        """
        Print a detailed comparison of poses for debugging
        
        Args:
            match_num: The match number (for tracking)
            previous_pose: The previous matched pose
            odometry_pose: The current odometry pose
            initial_guess: The initial guess for ICP 
            estimated_pose: The estimated pose from ICP (if available)
            corrected_pose: The corrected pose (if applicable)
            match_info: Match information dictionary
            stage: Description of the matching stage
        """
        # Calculate deltas from previous pose
        odom_delta_x = odometry_pose.x - previous_pose.x
        odom_delta_y = odometry_pose.y - previous_pose.y
        odom_delta_theta = self.normalize_angle(odometry_pose.theta - previous_pose.theta)
        
        guess_delta_x = initial_guess.x - previous_pose.x
        guess_delta_y = initial_guess.y - previous_pose.y
        guess_delta_theta = self.normalize_angle(initial_guess.theta - previous_pose.theta)
        
        # Print header
        print(f"\n{'='*100}")
        print(f"MATCH #{match_num}: {stage}")
        print(f"{'-'*100}")
        
        # Print previous pose
        print(f"PREVIOUS POSE:    x={previous_pose.x:.4f}, y={previous_pose.y:.4f}, θ={previous_pose.theta:.4f}")
        
        # Print odometry pose and delta
        print(f"ODOMETRY POSE:    x={odometry_pose.x:.4f}, y={odometry_pose.y:.4f}, θ={odometry_pose.theta:.4f}")
        print(f"ODOMETRY DELTA:   Δx={odom_delta_x:.4f}, Δy={odom_delta_y:.4f}, Δθ={odom_delta_theta:.4f}")
        
        # Print initial guess
        print(f"INITIAL GUESS:    x={initial_guess.x:.4f}, y={initial_guess.y:.4f}, θ={initial_guess.theta:.4f}")
        print(f"GUESS DELTA:      Δx={guess_delta_x:.4f}, Δy={guess_delta_y:.4f}, Δθ={guess_delta_theta:.4f}")
        
        # Calculate the difference between odometry and initial guess
        odom_guess_diff_x = odometry_pose.x - initial_guess.x
        odom_guess_diff_y = odometry_pose.y - initial_guess.y
        odom_guess_diff_dist = np.sqrt(odom_guess_diff_x**2 + odom_guess_diff_y**2)
        
        # Print the difference
        print(f"ODOM-GUESS DIFF:  Δx={odom_guess_diff_x:.4f}, Δy={odom_guess_diff_y:.4f}, dist={odom_guess_diff_dist:.4f}")
        
        # Print estimated pose if available
        if estimated_pose:
            est_delta_x = estimated_pose.x - previous_pose.x
            est_delta_y = estimated_pose.y - previous_pose.y
            est_delta_theta = self.normalize_angle(estimated_pose.theta - previous_pose.theta)
            
            print(f"ESTIMATED POSE:   x={estimated_pose.x:.4f}, y={estimated_pose.y:.4f}, θ={estimated_pose.theta:.4f}")
            print(f"ESTIMATED DELTA:  Δx={est_delta_x:.4f}, Δy={est_delta_y:.4f}, Δθ={est_delta_theta:.4f}")
            
            # Print match info if available
            if match_info:
                # Include resampling info
                if match_info.get('resampling_attempts', 0) > 0:
                    print(f"MATCH INFO:       Score={match_info['final_score']:.4f}, Iterations={match_info['iterations']}, "
                          f"Error={match_info['error']:.6f}, Correspondences={match_info['correspondences']}, "
                          f"Resampling Attempts={match_info['resampling_attempts']}")
                else:
                    print(f"MATCH INFO:       Score={match_info['final_score']:.4f}, Iterations={match_info['iterations']}, "
                          f"Error={match_info['error']:.6f}, Correspondences={match_info['correspondences']}")
        
        # Print corrected pose if available
        if corrected_pose:
            corr_delta_x = corrected_pose.x - previous_pose.x
            corr_delta_y = corrected_pose.y - previous_pose.y
            corr_delta_theta = self.normalize_angle(corrected_pose.theta - previous_pose.theta)
            
            print(f"CORRECTED POSE:   x={corrected_pose.x:.4f}, y={corrected_pose.y:.4f}, θ={corrected_pose.theta:.4f}")
            print(f"CORRECTED DELTA:  Δx={corr_delta_x:.4f}, Δy={corr_delta_y:.4f}, Δθ={corr_delta_theta:.4f}")
        
        print(f"{'='*100}\n")
    
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        return ((angle + math.pi) % (2 * math.pi)) - math.pi
    
    def matchScan(self, scan_x, scan_y, initial_pose):
        """
        Match the current scan against the map using ICP algorithm with aggressive resampling
        
        Args:
            scan_x: List of scan x coordinates
            scan_y: List of scan y coordinates
            initial_pose: Initial pose estimate (PoseEstimate object)
            
        Returns:
            Updated pose estimate (PoseEstimate object) and match info dictionary
        """
        # Create points array from scan
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Transform scan points to world frame using initial pose
        transformed_points = self.transformPointsToWorld(scan_points, initial_pose)
        
        # Try with normal parameters first
        current_pose = initial_pose.copy()
        correspondences, mean_error = self.findCorrespondences(transformed_points)
        
        # If we don't have enough correspondences, use aggressive resampling
        resampling_attempts = 0
        original_max_correspondence_distance = self.max_correspondence_distance
        original_occupancy_threshold = self.occupancy_threshold
        
        if len(correspondences) < 5:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Only {len(correspondences)} correspondences found initially. Starting aggressive resampling.")
            
            # Gradually increase search parameters until we find enough correspondences
            for attempt in range(1, self.aggressive_max_resampling_attempts + 1):
                resampling_attempts += 1
                
                # Calculate more aggressive parameters based on attempt number
                progress = attempt / self.aggressive_max_resampling_attempts
                
                # Increase search radius dramatically
                search_radius = self.max_correspondence_distance + progress * (self.aggressive_max_correspondence_distance - self.max_correspondence_distance)
                
                # Lower occupancy threshold dramatically
                occupancy_threshold = self.occupancy_threshold - progress * (self.occupancy_threshold - self.aggressive_min_occupancy_threshold)
                
                if self.debug_level > 1:
                    print(f"[ScanMatcher] Resampling attempt {attempt}: search radius={search_radius:.2f}m, "
                          f"occupancy threshold={occupancy_threshold:.2f}")
                
                # Temporarily set the new parameters
                self.max_correspondence_distance = search_radius
                self.occupancy_threshold = occupancy_threshold
                
                # Try to find correspondences with these more aggressive parameters
                correspondences, mean_error = self.findCorrespondences(transformed_points)
                
                if len(correspondences) >= 5:
                    if self.debug_level > 1:
                        print(f"[ScanMatcher] Found {len(correspondences)} correspondences after {attempt} resampling attempts.")
                    break
            
            # Restore original parameters
            self.max_correspondence_distance = original_max_correspondence_distance
            self.occupancy_threshold = original_occupancy_threshold
        
        # If we still don't have enough correspondences, use the initial pose
        if len(correspondences) < 5:
            if self.debug_level > 0:
                print(f"\n[ScanMatcher] Warning: Still only found {len(correspondences)} correspondences "
                      f"after {resampling_attempts} resampling attempts.")
            
            # Use initial pose but flag it as a poor match
            return initial_pose.copy(), {
                'iterations': 0,
                'final_score': 0.3,  # Low score to indicate it's not a good match
                'error': float('inf'),
                'correspondences': len(correspondences),
                'resampling_attempts': resampling_attempts
            }
        
        # Now proceed with ICP using the found correspondences
        iterations_data = []
        prev_error = mean_error
        
        # Main ICP loop
        for iteration in range(self.max_iterations):
            # Store the current pose before updating
            prev_pose = current_pose.copy()
            
            # Estimate new transformation that minimizes the distance between corresponding points
            updated_pose = self.estimateTransformation(scan_points, correspondences, current_pose)
            
            # Store iteration data for visualization
            iterations_data.append({
                'iteration': iteration,
                'pose': updated_pose.copy(),
                'error': mean_error,
                'correspondences': len(correspondences),
                'transformed_points': transformed_points.copy()
            })
            
            # Update current pose
            current_pose = updated_pose
            
            # Transform scan points to world frame using the updated pose
            transformed_points = self.transformPointsToWorld(scan_points, current_pose)
            
            # Find new correspondences
            correspondences, mean_error = self.findCorrespondences(transformed_points)
            
            # If we lost too many correspondences, stop
            if len(correspondences) < 5:
                if self.debug_level > 1:
                    print(f"\n[ScanMatcher] Lost correspondences during ICP (down to {len(correspondences)}). Stopping.")
                break
            
            # Check for convergence
            if abs(prev_error - mean_error) < self.convergence_threshold:
                if self.debug_level > 2:
                    print(f"\n[ScanMatcher] ICP converged after {iteration+1} iterations. Error: {mean_error:.6f}")
                break
                
            prev_error = mean_error
        
        # Score the final match
        final_score = self.scoreFinalMatch(scan_points, current_pose)
        
        # Store visualization data
        self.current_visualization_data = {
            'iterations': iterations_data,
            'final_pose': current_pose,
            'initial_pose': initial_pose,
            'scan_points': scan_points,
            'final_score': final_score,
            'resampling_attempts': resampling_attempts
        }
        
        # Return the matched pose and match information
        match_info = {
            'iterations': len(iterations_data),
            'final_score': final_score,
            'error': prev_error,
            'correspondences': len(correspondences),
            'resampling_attempts': resampling_attempts
        }
        
        return current_pose, match_info
    
    def transformPointsToWorld(self, points, pose):
        """
        Transform points from robot frame to world frame
        
        Args:
            points: Array of [x, y] points in robot's local frame
            pose: Robot pose (PoseEstimate object)
            
        Returns:
            Array of transformed points in world frame
        """
        # Extract pose components
        x, y, theta = pose.x, pose.y, pose.theta
        
        # Create rotation matrix
        c = math.cos(theta)
        s = math.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        # Apply rotation
        rotated_points = np.dot(points, rotation_matrix.T)
        
        # Apply translation
        transformed_points = rotated_points + np.array([x, y])
        
        return transformed_points
    
    def findCorrespondences(self, transformed_points):
        """
        Find corresponding points between the transformed scan and the map
        
        Args:
            transformed_points: Array of scan points in world frame
            
        Returns:
            List of correspondences and mean error
        """
        if self.map is None:
            return [], float('inf')
        
        correspondences = []
        total_error = 0.0
        
        for point in transformed_points:
            # Skip points outside the map with a buffer
            buffer = 1.0  # 1 meter buffer
            if (abs(point[0]) >= self.map.width/2 - buffer or 
                abs(point[1]) >= self.map.height/2 - buffer):
                continue
                
            # Convert to grid coordinates
            grid_x, grid_y = self.map.world_to_grid(point[0], point[1])
            
            # Make sure the grid coordinates are valid
            if not (0 <= grid_x < self.map.grid_width and 0 <= grid_y < self.map.grid_height):
                continue
                
            # Find closest occupied cell within search radius
            closest_cell, distance = self.findClosestOccupiedCell(grid_x, grid_y)
            
            if closest_cell is not None and distance < self.max_correspondence_distance / self.map.resolution:
                # Convert back to world coordinates
                world_x, world_y = self.map.grid_to_world(closest_cell[0], closest_cell[1])
                
                # Add correspondence
                correspondences.append({
                    'scan_point': point,
                    'map_point': np.array([world_x, world_y]),
                    'distance': distance * self.map.resolution  # Convert to meters
                })
                
                total_error += distance * self.map.resolution
        
        mean_error = total_error / len(correspondences) if correspondences else float('inf')
        
        return correspondences, mean_error
    
    def findClosestOccupiedCell(self, grid_x, grid_y):
        """
        Find the closest occupied cell to the given grid coordinates
        
        Args:
            grid_x, grid_y: Grid coordinates to search from
            
        Returns:
            Closest occupied cell coordinates and distance
        """
        # Define search radius (in grid cells)
        search_radius = int(self.max_correspondence_distance / self.map.resolution)
        
        min_distance = float('inf')
        closest_cell = None
        
        # Simple grid search in a square area
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                
                # Check if within grid bounds
                if (0 <= nx < self.map.grid_width and 0 <= ny < self.map.grid_height):
                    # Check if this cell is occupied - USING CURRENT THRESHOLD
                    if self.map.grid[ny, nx] > self.occupancy_threshold:
                        # Calculate Euclidean distance
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_cell = (nx, ny)
        
        return closest_cell, min_distance
    
    def estimateTransformation(self, scan_points, correspondences, current_pose):
        """
        Estimate the transformation that minimizes the distance between corresponding points
        
        Args:
            scan_points: Original scan points in robot frame
            correspondences: List of correspondences between scan and map
            current_pose: Current pose estimate
            
        Returns:
            Updated pose estimate
        """
        if not correspondences:
            return current_pose.copy()
        
        try:
            # Extract corresponding points
            scan_points_array = np.array([corr['scan_point'] for corr in correspondences])
            map_points_array = np.array([corr['map_point'] for corr in correspondences])
            
            # Calculate centroids
            scan_centroid = np.mean(scan_points_array, axis=0)
            map_centroid = np.mean(map_points_array, axis=0)
            
            # Center the points
            centered_scan = scan_points_array - scan_centroid
            centered_map = map_points_array - map_centroid
            
            # Compute the covariance matrix
            H = np.dot(centered_scan.T, centered_map)
            
            # Singular Value Decomposition
            U, S, Vt = np.linalg.svd(H)
            
            # Calculate rotation matrix
            R = np.dot(Vt.T, U.T)
            
            # Ensure proper rotation matrix (det=1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            # Calculate translation
            t = map_centroid - np.dot(scan_centroid, R.T)
            
            # Extract rotation angle from rotation matrix
            theta = math.atan2(R[1, 0], R[0, 0])
            
            # Create updated pose
            updated_pose = current_pose.copy()
            updated_pose.x = t[0]
            updated_pose.y = t[1]
            updated_pose.theta = theta
            
            return updated_pose
            
        except Exception as e:
            # If there's any error in the estimation, return the original pose
            if self.debug_level > 0:
                print(f"\n[ScanMatcher] Error in pose estimation: {e}")
            return current_pose.copy()
    
    def calculatePoseChange(self, pose1, pose2):
        """
        Calculate the change between two poses
        
        Args:
            pose1, pose2: PoseEstimate objects
            
        Returns:
            Dictionary with dx, dy, dtheta
        """
        dx = pose2.x - pose1.x
        dy = pose2.y - pose1.y
        dtheta = (pose2.theta - pose1.theta + math.pi) % (2 * math.pi) - math.pi  # Normalize angle difference
        
        return {
            'dx': dx,
            'dy': dy,
            'dtheta': dtheta,
            'distance': math.sqrt(dx**2 + dy**2)
        }
    
    def scoreFinalMatch(self, scan_points, pose):
        """
        Score the final match quality
        
        Args:
            scan_points: Original scan points in robot frame
            pose: Final pose estimate
            
        Returns:
            Match quality score (higher is better)
        """
        if self.map is None:
            return 0.0
        
        # Transform points to world frame
        world_points = self.transformPointsToWorld(scan_points, pose)
        
        total_score = 0.0
        valid_points = 0
        
        for point in world_points:
            # Skip points outside the map
            if (abs(point[0]) >= self.map.width/2 or abs(point[1]) >= self.map.height/2):
                continue
                
            # Convert to grid coordinates
            grid_x, grid_y = self.map.world_to_grid(point[0], point[1])
            
            # Ensure grid coordinates are valid
            if not (0 <= grid_x < self.map.grid_width and 0 <= grid_y < self.map.grid_height):
                continue
                
            # Get the occupancy value at this point
            occupancy = self.map.grid[grid_y, grid_x]
            
            # Score higher for points that land on occupied cells
            # and lower for points that land on free space
            if occupancy > self.occupancy_threshold:  # Occupied
                total_score += 1.0
            elif occupancy < 0.3:  # Free
                total_score -= 0.5
            
            valid_points += 1
        
        # Normalize score between 0 and 1 - FIXED to handle zero valid points
        if valid_points > 0:
            normalized_score = (total_score / valid_points + 0.5) / 1.5
            return max(0.0, min(1.0, normalized_score))
        else:
            # If no valid points, return a very low score
            return 0.1
    
    def validateMatch(self, matched_pose, previous_pose, match_info):
        """
        Validate if the match is reasonable
        
        Args:
            matched_pose: New matched pose
            previous_pose: Previous pose
            match_info: Information about the match
            
        Returns:
            Boolean indicating if the match is valid
        """
        # If there were resampling attempts but still few correspondences, be stricter
        min_required_correspondences = 5
        if match_info['resampling_attempts'] > 0:
            min_required_correspondences = 3 + match_info['resampling_attempts']
            
        # If no correspondences were found, match is invalid
        if match_info['correspondences'] < min_required_correspondences:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Too few correspondences ({match_info['correspondences']} < {min_required_correspondences})")
            return False
        
        # Calculate pose change
        pose_change = self.calculatePoseChange(previous_pose, matched_pose)
        
        # Check if the translation is within limits
        if pose_change['distance'] > self.max_translation_per_frame:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Translation too large ({pose_change['distance']:.3f}m > {self.max_translation_per_frame}m)")
            return False
        
        # Check if the rotation is within limits
        if abs(pose_change['dtheta']) > self.max_rotation_per_frame:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Rotation too large ({abs(pose_change['dtheta']):.3f}rad > {self.max_rotation_per_frame}rad)")
            return False
        
        # Check if the match score is reasonable - be more lenient if we had to resample
        score_threshold = 0.3
        if match_info['resampling_attempts'] > 0:
            score_threshold = max(0.2, 0.3 - 0.02 * match_info['resampling_attempts'])
            
        if match_info['final_score'] < score_threshold:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Score too low ({match_info['final_score']:.3f} < {score_threshold})")
            return False
        
        # All checks passed
        return True
    
    def applySmallCorrection(self, odometry_pose, previous_matched_pose):
        """
        Apply a small correction to the odometry pose based on the previous matched pose
        
        Args:
            odometry_pose: Current odometry pose
            previous_matched_pose: Previous matched pose
            
        Returns:
            Corrected pose
        """
        # Calculate the odometry change from the previous frame
        last_odometry_pose = self.odometry_trajectory[-2]
        odom_change = self.calculatePoseChange(last_odometry_pose, odometry_pose)
        
        # Apply the same change to the previous matched pose, with a slight correction factor
        # (this helps prevent drift by applying a small correction towards the matched trajectory)
        correction_factor = 0.9  # Apply 90% of the odometry change
        
        corrected_pose = previous_matched_pose.copy()
        corrected_pose.x += odom_change['dx'] * correction_factor
        corrected_pose.y += odom_change['dy'] * correction_factor
        corrected_pose.theta += odom_change['dtheta'] * correction_factor
        
        return corrected_pose
    
    def plotMatchOverlay(self, scan_x, scan_y, pose, ax=None, show_iterations=False):
        """
        Plot the scan overlaid on the map to visualize the match quality
        
        Args:
            scan_x: List of scan x coordinates
            scan_y: List of scan y coordinates
            pose: Current pose estimate (PoseEstimate object)
            ax: Matplotlib axis to plot on (or None to create new figure)
            show_iterations: Whether to show the ICP iterations
            
        Returns:
            Matplotlib axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create scan points array
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Transform scan points using the pose
        transformed_points = self.transformPointsToWorld(scan_points, pose)
        
        # Plot the map if we have one
        if self.map:
            # Custom colormap: white (unknown), black (occupied), light gray (free)
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                self.map.get_grid_for_display(),
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-self.map.width/2, self.map.width/2, -self.map.height/2, self.map.height/2]
            )
            
            # Draw map boundaries
            ax.axhline(y=-self.map.height/2, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=self.map.height/2, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=-self.map.width/2, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.map.width/2, color='red', linestyle='--', alpha=0.5)
        
        # Plot the transformed scan points
        ax.scatter(transformed_points[:, 0], transformed_points[:, 1], c='red', s=3, label='Scan Points')
        
        # Plot the robot position
        ax.scatter(pose.x, pose.y, c='blue', s=100, marker='*', label='Robot Position')
        
        # Plot orientation arrow
        arrow_length = 0.5
        dx = arrow_length * math.cos(pose.theta)
        dy = arrow_length * math.sin(pose.theta)
        
        ax.arrow(
            pose.x, pose.y, dx, dy,
            head_width=0.1, head_length=0.1, fc='blue', ec='blue'
        )
        
        # Plot search radius circle to visualize correspondence distance
        search_circle = plt.Circle((pose.x, pose.y), 
                                  self.max_correspondence_distance,
                                  color='blue', fill=False, alpha=0.3)
        ax.add_patch(search_circle)
        
        # If we have visualization data and want to show iterations
        if show_iterations and self.current_visualization_data:
            data = self.current_visualization_data
            
            # Plot initial pose
            ax.scatter(
                data['initial_pose'].x, 
                data['initial_pose'].y, 
                c='orange', s=100, marker='o', 
                label='Initial Pose'
            )
            
            # Plot iteration poses with color gradient
            iterations = data['iterations']
            if iterations:
                colors_iter = plt.cm.viridis(np.linspace(0, 1, len(iterations)))
                
                for i, iter_data in enumerate(iterations):
                    iter_pose = iter_data['pose']
                    ax.scatter(
                        iter_pose.x, iter_pose.y, 
                        c=[colors_iter[i]], s=50, alpha=0.7,
                        marker='x'
                    )
                
                # Add a custom legend entry for iterations
                ax.scatter([], [], c='green', marker='x', s=50, label='ICP Iterations')
            
            # Add match score and resampling info to the plot
            info_text = f"Match Score: {data['final_score']:.3f}"
            if data.get('resampling_attempts', 0) > 0:
                info_text += f"\nResampling Attempts: {data['resampling_attempts']}"
                
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    va='top', ha='left', color='blue', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # If aggressive resampling was used, also show the aggressive search radius
            if data.get('resampling_attempts', 0) > 0:
                aggressive_circle = plt.Circle((pose.x, pose.y), 
                                            self.aggressive_max_correspondence_distance,
                                            color='red', fill=False, alpha=0.2, linestyle='--')
                ax.add_patch(aggressive_circle)
                ax.scatter([], [], c='red', marker='o', s=0, label=f'Aggressive Search ({self.aggressive_max_correspondence_distance}m)', 
                          linestyle='--', alpha=0.2)
        
        # Add grid and labels
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Scan-Map Match Overlay')
        ax.legend(loc='upper right')
        
        return ax
    
    def visualizeIcpProcess(self):
        """
        Create a comprehensive visualization of the ICP process
        
        Returns:
            Matplotlib figure with the visualization
        """
        if not self.current_visualization_data:
            print("[ScanMatcher] No visualization data available.")
            return None
        
        data = self.current_visualization_data
        iterations = data['iterations']
        
        if not iterations:
            print("[ScanMatcher] No iteration data available.")
            return None
        
        # Create figure with multiple subplots
        n_iterations = min(4, len(iterations))  # Show at most 4 iterations
        fig, axes = plt.subplots(1, n_iterations + 1, figsize=(5 * (n_iterations + 1), 5))
        
        # Handle the case where n_iterations is 0 (single plot)
        if n_iterations == 0:
            axes = [axes]
        
        # Plot the map in all subplots
        if self.map:
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            for ax in axes:
                ax.imshow(
                    self.map.get_grid_for_display(),
                    cmap=cmap, norm=norm,
                    origin='lower',
                    extent=[-self.map.width/2, self.map.width/2, -self.map.height/2, self.map.height/2]
                )
                ax.set_aspect('equal')
                ax.grid(True)
        
        # Plot initial state
        axes[0].scatter(
            data['initial_pose'].x, 
            data['initial_pose'].y, 
            c='orange', s=100, marker='o', 
            label='Initial Pose'
        )
        
        # Transform points using initial pose
        initial_transformed = self.transformPointsToWorld(data['scan_points'], data['initial_pose'])
        axes[0].scatter(
            initial_transformed[:, 0], 
            initial_transformed[:, 1], 
            c='orange', s=3, alpha=0.7,
            label='Initial Points'
        )
        
        # Show search radius
        search_circle = plt.Circle(
            (data['initial_pose'].x, data['initial_pose'].y), 
            self.max_correspondence_distance,
            color='blue', fill=False, alpha=0.3
        )
        axes[0].add_patch(search_circle)
        
        # If resampling was used, show that info
        if data.get('resampling_attempts', 0) > 0:
            info_text = f"Initial State\nResampling: {data['resampling_attempts']} attempts"
            
            # Also show aggressive search radius
            aggressive_circle = plt.Circle(
                (data['initial_pose'].x, data['initial_pose'].y), 
                self.aggressive_max_correspondence_distance,
                color='red', fill=False, alpha=0.2, linestyle='--'
            )
            axes[0].add_patch(aggressive_circle)
        else:
            info_text = "Initial State"
            
        axes[0].set_title(info_text)
        axes[0].legend()
        
        # Plot iteration states
        if len(iterations) > 0:
            selected_indices = np.linspace(0, len(iterations) - 1, n_iterations, dtype=int)
            
            for i, idx in enumerate(selected_indices):
                if i >= len(axes) - 1:  # Skip if we don't have enough axes
                    break
                    
                iter_data = iterations[idx]
                ax = axes[i + 1]
                
                # Plot the transformed points
                if 'transformed_points' in iter_data:
                    ax.scatter(
                        iter_data['transformed_points'][:, 0], 
                        iter_data['transformed_points'][:, 1], 
                        c='red', s=3, alpha=0.7,
                        label='Scan Points'
                    )
                
                # Plot the pose
                ax.scatter(
                    iter_data['pose'].x, 
                    iter_data['pose'].y, 
                    c='blue', s=100, marker='*', 
                    label='Robot Pose'
                )
                
                # Plot orientation arrow
                arrow_length = 0.5
                dx = arrow_length * math.cos(iter_data['pose'].theta)
                dy = arrow_length * math.sin(iter_data['pose'].theta)
                
                ax.arrow(
                    iter_data['pose'].x, iter_data['pose'].y, dx, dy,
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue'
                )
                
                # Add iteration info
                info_text = (
                    f"Iteration {iter_data['iteration'] + 1}\n"
                    f"Error: {iter_data['error']:.4f}\n"
                    f"Correspondences: {iter_data['correspondences']}"
                )
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                        va='top', ha='left', color='blue', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7))
                
                ax.set_title(f"Iteration {iter_data['iteration'] + 1}")
                ax.legend()
        
        plt.tight_layout()
        return fig

    def visualize_map_and_scan(self, scan_x, scan_y, pose):
        """Create a visualization of the map and current scan for debugging"""
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create points array from scan
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Transform points to world frame
        world_points = self.transformPointsToWorld(scan_points, pose)
        
        # Plot the map
        if self.map:
            # Custom colormap
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                self.map.get_grid_for_display(),
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-self.map.width/2, self.map.width/2, -self.map.height/2, self.map.height/2]
            )
            
            # Count the number of occupied cells
            try:
                occupied_cells = np.sum(self.map.grid > self.occupancy_threshold)
                total_cells = self.map.grid_width * self.map.grid_height
                
                ax.set_title(f"Map Visualization - {occupied_cells} occupied cells ({occupied_cells/total_cells*100:.2f}%)")
            except:
                ax.set_title("Map Visualization")
        
        # Plot odometry trajectory
        odom_x = [pose.x for pose in self.odometry_trajectory]
        odom_y = [pose.y for pose in self.odometry_trajectory]
        ax.plot(odom_x, odom_y, 'r-', linewidth=1, alpha=0.5, label='Odometry')
        
        # Plot matched trajectory
        matched_x = [pose.x for pose in self.trajectory]
        matched_y = [pose.y for pose in self.trajectory]
        ax.plot(matched_x, matched_y, 'g-', linewidth=1, label='Matched')
        
        # Plot scan points
        ax.scatter(world_points[:, 0], world_points[:, 1], c='blue', s=3, alpha=0.5, label='Current Scan')
        
        # Plot the current position from both odometry and matched pose
        if len(self.odometry_trajectory) > 0:
            ax.scatter(self.odometry_trajectory[-1].x, self.odometry_trajectory[-1].y, 
                      c='red', s=100, marker='*', label='Odometry Position')
        
        if len(self.trajectory) > 0:
            ax.scatter(self.trajectory[-1].x, self.trajectory[-1].y, 
                      c='green', s=100, marker='*', label='Matched Position')
        
        # Draw map boundaries
        ax.axhline(y=-self.map.height/2, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=self.map.height/2, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=-self.map.width/2, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=self.map.width/2, color='red', linestyle='--', alpha=0.5)
        
        # Add search radius visualization around current matched position
        if len(self.trajectory) > 0:
            current_pos = self.trajectory[-1]
            search_circle = plt.Circle((current_pos.x, current_pos.y), 
                                      self.max_correspondence_distance,
                                      color='blue', fill=False, alpha=0.3)
            ax.add_patch(search_circle)
            
            # Also show the aggressive search radius
            aggressive_circle = plt.Circle((current_pos.x, current_pos.y), 
                                        self.aggressive_max_correspondence_distance,
                                        color='red', fill=False, alpha=0.2, linestyle='--')
            ax.add_patch(aggressive_circle)
        
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save the figure to a file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"map_scan_debug_{timestamp}.png", dpi=150)
        
        print(f"\n[ScanMatcher] Map visualization saved to map_scan_debug_{timestamp}.png")
        
        # Close the figure to free memory
        plt.close(fig)
    
def animate_lidar_data(parsed_data_list, flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False, 
                      show_occupancy_grid=True, grid_resolution=0.05, save_grid=False,
                      save_format='png', save_path='maps/', enable_scan_matching=False):
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
        enable_scan_matching: Whether to use scan matching localization
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
    
    # Initialize occupancy grid
    if show_occupancy_grid:
        occupancy_grid = OccupancyGrid(resolution=grid_resolution, 
                                      width=grid_width, 
                                      height=grid_height)
    else:
        occupancy_grid = None
    
    # Initialize scan matching localization if enabled
    if enable_scan_matching:
        localizer = ImprovedScanMatchingLocalization(occupancy_grid)
        
        # Process sensor data to build trajectory with scan matching
        # For the first pass, we'll use odometry for the trajectory while building the map
        localizer.processSensorData(
            parsed_data_list,
            angle_min=angle_min,
            angle_max=angle_max,
            flip_x=flip_x,
            flip_y=flip_y,
            reverse_scan=reverse_scan,
            flip_theta=flip_theta
        )
        
        # If we want to improve localization, we can now run scan matching
        # against the built map (not done in this basic implementation)
        
        # Extract trajectory for visualization
        trajectory = localizer.trajectory
        robot_path_x = [pose.x for pose in trajectory]
        robot_path_y = [pose.y for pose in trajectory]
    else:
        # Use odometry-based trajectory without scan matching
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
        # Custom colormap: white (unknown), black (occupied), light gray (free)
        cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
        bounds = [0, 0.4, 0.6, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Initialize the occupancy grid display
        grid_img = ax2.imshow(occupancy_grid.get_grid_for_display(), 
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
        
        # Add scan matching status text if enabled
        if enable_scan_matching:
            grid_scan_match_text = ax2.text(0.02, 0.82, "Scan Matching: Enabled", 
                                           transform=ax2.transAxes, va='top', ha='left', color='green')
        
        # Add zoom information text
        zoom_info_text = ax2.text(0.5, 0.02, "Left-click: Zoom in | Right-click: Zoom out | Middle-click: Reset zoom", 
                                 transform=ax2.transAxes, va='bottom', ha='center', 
                                 fontsize=10, color='blue', bbox=dict(facecolor='white', alpha=0.7))
        
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
        
        # Add a Follow Robot button
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons
        follow_button_ax = plt.axes([0.85, 0.05, 0.1, 0.04])
        follow_button = Button(follow_button_ax, 'Follow Robot', color='lightgoldenrodyellow', hovercolor='0.975')
        
        # Add a Save Map button
        save_button_ax = plt.axes([0.70, 0.05, 0.1, 0.04])
        save_button = Button(save_button_ax, 'Save Map', color='lightblue', hovercolor='0.8')
        
        # Add a Scan Match Overlay button if scan matching is enabled
        if enable_scan_matching:
            match_overlay_button_ax = plt.axes([0.55, 0.05, 0.1, 0.04])
            match_overlay_button = Button(match_overlay_button_ax, 'Show Match', color='lightgreen', hovercolor='0.8')
            show_match_overlay = [False]  # Flag to track if match overlay should be shown
        
        def toggle_follow(event):
            follow_robot[0] = not follow_robot[0]
            follow_button.label.set_text('Following' if follow_robot[0] else 'Not Following')
            
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
                current_position=current_pos
            )
            
            print(f"\nOccupancy grid map saved to {base_filename}.{save_format} with current robot path and positions")
        
        def toggle_match_overlay(event):
            if not enable_scan_matching:
                return
                
            show_match_overlay[0] = not show_match_overlay[0]
            match_overlay_button.label.set_text('Hide Match' if show_match_overlay[0] else 'Show Match')
            
            # If showing the overlay, create a new figure
            if show_match_overlay[0]:
                # Get current frame
                current_frame = current_frame_index[0]
                
                # Get current scan data
                scan_data = parsed_data_list[current_frame]
                
                # Convert scan to Cartesian coordinates
                scan_x, scan_y = convert_scans_to_cartesian(
                    scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
                    flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
                )
                
                # Get current pose from trajectory
                current_pose = localizer.trajectory[current_frame]
                
                # Create a new figure for the overlay
                overlay_fig = plt.figure(figsize=(10, 10))
                overlay_ax = overlay_fig.add_subplot(111)
                
                # Plot the overlay using the localizer's function
                localizer.plotMatchOverlay(scan_x, scan_y, current_pose, ax=overlay_ax)
                
                plt.tight_layout()
                plt.show()
            
        follow_button.on_clicked(toggle_follow)
        save_button.on_clicked(save_current_map)
        
        if enable_scan_matching:
            match_overlay_button.on_clicked(toggle_match_overlay)
        
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
    
    # Add scan matching status if enabled
    if enable_scan_matching:
        scan_match_text = ax.text(0.02, 0.82, "Scan Matching: Enabled", 
                                 transform=ax.transAxes, va='top', ha='left', color='green')
    
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
            
            return_values = [scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, settings_text, 
                           grid_img, grid_path_line, grid_start_point, grid_current_pos, grid_timestamp_text, 
                           grid_robot_id_text, grid_pose_text, grid_settings_text]
            
            # Add scan matching text to return values if enabled
            if enable_scan_matching:
                return_values.append(scan_match_text)
                return_values.append(grid_scan_match_text)
            
            return tuple(return_values)
        else:
            return_values = [scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, settings_text]
            
            # Add scan matching text to return values if enabled
            if enable_scan_matching:
                return_values.append(scan_match_text)
            
            return tuple(return_values)
    
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
        
        # Get transformed robot pose
        if enable_scan_matching and frame < len(localizer.trajectory):
            # Use the scan-matched pose from trajectory
            robot_x = localizer.trajectory[frame].x
            robot_y = localizer.trajectory[frame].y
            robot_theta = localizer.trajectory[frame].theta
        else:
            # Use odometry-based pose
            robot_x, robot_y = parsed_data['pose']['x'], parsed_data['pose']['y']
            if flip_x:
                robot_x = -robot_x
            if flip_y:
                robot_y = -robot_y
            robot_theta = parsed_data['pose']['theta']
            if flip_theta:
                robot_theta = -robot_theta
        
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
        
        # Show pose values
        if enable_scan_matching:
            # Show both odometry and scan-matched pose
            odom_pose_str = f"Odometry Pose: x={parsed_data['pose']['x']:.3f}, y={parsed_data['pose']['y']:.3f}, θ={parsed_data['pose']['theta']:.3f}"
            matched_pose_str = f"Matched Pose: x={robot_x:.3f}, y={robot_y:.3f}, θ={robot_theta:.3f}"
            pose_text.set_text(f"{odom_pose_str}\n{matched_pose_str}")
        else:
            # Show only odometry pose
            pose_str = f"Pose: x={parsed_data['pose']['x']:.3f}, y={parsed_data['pose']['y']:.3f}, θ={parsed_data['pose']['theta']:.3f}"
            pose_text.set_text(pose_str)
        
        # Update robot orientation arrow
        if arrow:
            arrow.remove()
        
        # Use the appropriate orientation
        arrow_length = 0.5
        dx = arrow_length * math.cos(robot_theta)
        dy = arrow_length * math.sin(robot_theta)
            
        arrow = ax.arrow(robot_x, robot_y, dx, dy, 
                        head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # Update occupancy grid if enabled
        if show_occupancy_grid:
            # Update the grid with current scan if not using pre-built map
            if not enable_scan_matching or frame == 0:
                occupancy_grid.update_grid(robot_x, robot_y, x_points, y_points)
            
            # Update the grid image
            grid_img.set_data(occupancy_grid.get_grid_for_display())
            
            # Update the robot path on the grid map
            grid_path_line.set_data(robot_path_x[:frame+1], robot_path_y[:frame+1])
            
            # Update the current position marker
            grid_current_pos.set_offsets([[robot_x, robot_y]])
            
            # Update text information on grid
            grid_timestamp_text.set_text(timestamp_str)
            grid_robot_id_text.set_text(robot_id_str)
            
            if enable_scan_matching:
                grid_pose_text.set_text(matched_pose_str)
            else:
                grid_pose_text.set_text(pose_str)
            
            return_values = [scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, 
                           settings_text, arrow, grid_img, grid_path_line, grid_current_pos, grid_timestamp_text, 
                           grid_robot_id_text, grid_pose_text, grid_settings_text]
            
            # Add scan matching text to return values if enabled
            if enable_scan_matching:
                return_values.append(scan_match_text)
                return_values.append(grid_scan_match_text)
            
            return tuple(return_values)
        else:
            return_values = [scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, 
                            settings_text, arrow]
            
            # Add scan matching text to return values if enabled
            if enable_scan_matching:
                return_values.append(scan_match_text)
            
            return tuple(return_values)
    
    # Create animation with faster frame rate for smoother visualization
    animation = FuncAnimation(fig, update, frames=len(parsed_data_list), 
                             init_func=init, interval=10, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    # Note: The save functionality is now handled by the Save Map button
    # If you still want to automatically save at the end, you can use:
    # if save_grid and show_occupancy_grid:
    #     save_current_map(None)  # Call the save function without an event
    
    return animation

def visualize_lidar_data_realtime(file_path, max_entries=200, show_occupancy_grid=True, 
                             grid_resolution=0.05, save_grid=True, save_format='all',
                             enable_scan_matching=True):
    """
    Main function to visualize LiDAR data in real-time with occupancy grid mapping and scan matching
    
    Args:
        file_path: Path to the LiDAR data file
        max_entries: Maximum number of entries to read from the file
        show_occupancy_grid: Whether to show the occupancy grid visualization
        grid_resolution: Resolution of the occupancy grid in meters (smaller = more detail but slower)
        save_grid: Whether to save the final occupancy grid map to a file
        save_format: Format to save the grid ('png', 'npy', 'csv', or 'all')
        enable_scan_matching: Whether to use scan matching localization algorithm
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
        if save_grid:
            print(f"  The final occupancy grid will be saved in '{save_format}' format")
    else:
        print(f"  Starting visualization with orientation correction...")
    
    # Assuming the LiDAR scan covers 180 degrees (π radians)
    angle_min = -math.pi/2
    angle_max = math.pi/2
    
    # Find max range for consistent scaling by converting all data points
    all_x_points = []
    all_y_points = []
    for parsed_data in parsed_data_list:
        x_points, y_points = convert_scans_to_cartesian(
            parsed_data['scan_ranges'], angle_min, angle_max, parsed_data['pose'],
            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
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
    
    # Initialize occupancy grid
    if show_occupancy_grid:
        occupancy_grid = OccupancyGrid(resolution=grid_resolution, 
                                      width=grid_width, 
                                      height=grid_height)
    else:
        occupancy_grid = None
    
    # Initialize scan matching localization if enabled
    if enable_scan_matching:
        print(f"  Using improved ICP scan matching algorithm with motion validation")
        localizer = ImprovedScanMatchingLocalization(occupancy_grid, debug_level=1)
        
        # Process sensor data to build trajectory with scan matching
        localizer.processSensorData(
            parsed_data_list,
            angle_min=angle_min,
            angle_max=angle_max,
            flip_x=False,
            flip_y=False,
            reverse_scan=True,
            flip_theta=False
        )
        
        # Extract trajectory for visualization
        trajectory = localizer.trajectory
        robot_path_x = [pose.x for pose in trajectory]
        robot_path_y = [pose.y for pose in trajectory]
        
        # Also keep track of odometry trajectory for comparison
        odometry_trajectory = localizer.odometry_trajectory
        odometry_path_x = [pose.x for pose in odometry_trajectory]
        odometry_path_y = [pose.y for pose in odometry_trajectory]
    else:
        # Use odometry-based trajectory without scan matching
        print(f"  Scan matching is DISABLED - using raw odometry")
        robot_path_x = []
        robot_path_y = []
        for data in parsed_data_list:
            x, y = data['pose']['x'], data['pose']['y']
            robot_path_x.append(x)
            robot_path_y.append(y)
        odometry_path_x = robot_path_x
        odometry_path_y = robot_path_y
    
    # Create output directory for maps
    maps_dir = "maps"
    if save_grid and not os.path.exists(maps_dir):
        try:
            os.makedirs(maps_dir)
            print(f"  Created directory for maps: {maps_dir}/")
        except Exception as e:
            print(f"  Error creating maps directory: {e}")
    
    # Create a figure with two subplots side by side if showing occupancy grid
    if show_occupancy_grid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # Set up the occupancy grid image
        # Custom colormap: white (unknown), black (occupied), light gray (free)
        cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
        bounds = [0, 0.4, 0.6, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Initialize the occupancy grid display
        grid_img = ax2.imshow(occupancy_grid.get_grid_for_display(), 
                             cmap=cmap, norm=norm, 
                             origin='lower', 
                             extent=[-occupancy_grid.width/2, occupancy_grid.width/2, 
                                     -occupancy_grid.height/2, occupancy_grid.height/2])
        
        # Add reference grid lines
        ax2.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Create a line for robot path on the occupancy grid
        grid_path_line, = ax2.plot(robot_path_x, robot_path_y, 'b-', linewidth=2, label='Matched Path')
        
        # Create a line for odometry path if scan matching is enabled
        if enable_scan_matching:
            grid_odom_line, = ax2.plot(odometry_path_x, odometry_path_y, 'r--', linewidth=1, alpha=0.6, label='Odometry Path')
        
        # Also plot the starting position on the grid
        if len(robot_path_x) > 0:
            grid_start_point = ax2.scatter(robot_path_x[0], robot_path_y[0], c='green', s=100, marker='*', label='Start')
            grid_current_pos = ax2.scatter(robot_path_x[-1], robot_path_y[-1], c='blue', s=100, marker='*', label='End')
        
        # Add a button for visualizing the ICP process if scan matching is enabled
        if enable_scan_matching:
            plt.subplots_adjust(bottom=0.15)  # Make room for buttons
            
            # Add a Visualize ICP Process button
            icp_viz_button_ax = plt.axes([0.55, 0.05, 0.15, 0.04])
            icp_viz_button = Button(icp_viz_button_ax, 'Visualize ICP Process', color='lightgreen', hovercolor='0.8')
            
            def visualize_icp_process(event):
                if not enable_scan_matching:
                    print("ICP visualization is only available when scan matching is enabled.")
                    return
                
                # Create ICP process visualization
                fig = localizer.visualizeIcpProcess()
                if fig:
                    plt.figure(fig.number)
                    plt.show()
                else:
                    print("No ICP visualization data available.")
            
            icp_viz_button.on_clicked(visualize_icp_process)
        
        # Add a Save Map button
        save_button_ax = plt.axes([0.75, 0.05, 0.1, 0.04])
        save_button = Button(save_button_ax, 'Save Map', color='lightblue', hovercolor='0.8')
        
        def save_map(event):
            if not show_occupancy_grid:
                print("Cannot save map - occupancy grid is disabled.")
                return
                
            # Create the save directory if it doesn't exist
            if not os.path.exists(maps_dir):
                os.makedirs(maps_dir)
            
            # Generate a timestamp-based filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.join(maps_dir, f"occupancy_grid_{timestamp}")
            
            # Create path coordinates for saving
            robot_path_coords = list(zip(robot_path_x, robot_path_y))
            
            # Get start and current positions from path
            start_pos = (robot_path_x[0], robot_path_y[0]) if len(robot_path_x) > 0 else None
            current_pos = (robot_path_x[-1], robot_path_y[-1]) if len(robot_path_x) > 0 else None
            
            # Save the grid with robot path and positions
            occupancy_grid.save_to_file(
                base_filename, 
                format=save_format, 
                include_metadata=True,
                robot_path=robot_path_coords,
                start_position=start_pos,
                current_position=current_pos
            )
            
            print(f"\nOccupancy grid map saved to {base_filename}.{save_format} with robot path and positions")
        
        save_button.on_clicked(save_map)
        
        # Add a Compare Paths button if scan matching is enabled
        if enable_scan_matching:
            compare_button_ax = plt.axes([0.90, 0.05, 0.08, 0.04])
            compare_button = Button(compare_button_ax, 'Compare Paths', color='lightcoral', hovercolor='0.8')
            
            def compare_paths(event):
                # Create a figure to compare odometry and matched paths
                compare_fig, compare_ax = plt.subplots(figsize=(10, 10))
                
                # Plot the map
                if show_occupancy_grid:
                    compare_ax.imshow(
                        occupancy_grid.get_grid_for_display(),
                        cmap=cmap, norm=norm,
                        origin='lower',
                        extent=[-occupancy_grid.width/2, occupancy_grid.width/2, 
                               -occupancy_grid.height/2, occupancy_grid.height/2]
                    )
                
                # Plot both paths
                compare_ax.plot(odometry_path_x, odometry_path_y, 'r-', linewidth=2, label='Odometry Path')
                compare_ax.plot(robot_path_x, robot_path_y, 'b-', linewidth=2, label='Matched Path')
                
                # Plot start and end points
                compare_ax.scatter(odometry_path_x[0], odometry_path_y[0], c='green', s=100, marker='*', label='Start')
                compare_ax.scatter(odometry_path_x[-1], odometry_path_y[-1], c='red', s=100, marker='*', label='Odometry End')
                compare_ax.scatter(robot_path_x[-1], robot_path_y[-1], c='blue', s=100, marker='*', label='Matched End')
                
                # Add grid, labels, and legend
                compare_ax.grid(True)
                compare_ax.set_aspect('equal')
                compare_ax.set_xlabel('X (meters)')
                compare_ax.set_ylabel('Y (meters)')
                compare_ax.set_title('Odometry vs. Scan-Matched Path Comparison')
                compare_ax.legend(loc='upper right')
                
                plt.tight_layout()
                plt.show()
            
            compare_button.on_clicked(compare_paths)
        
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
    scan_x, scan_y = convert_scans_to_cartesian(
        parsed_data_list[-1]['scan_ranges'], angle_min, angle_max, parsed_data_list[-1]['pose'],
        flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
    )
    scatter = ax.scatter(scan_x, scan_y, c='blue', s=3, label='LiDAR Points')
    
    # Create a scatter plot for robot position
    last_x = robot_path_x[-1] if robot_path_x else 0
    last_y = robot_path_y[-1] if robot_path_y else 0
    robot_pos = ax.scatter(last_x, last_y, c='red', s=100, marker='*', label='Final Position')
    
    # Create a line for robot path
    path_line, = ax.plot(robot_path_x, robot_path_y, 'g-', linewidth=2, label='Robot Path')
    
    # Create a line for odometry path if scan matching is enabled
    if enable_scan_matching:
        odom_line, = ax.plot(odometry_path_x, odometry_path_y, 'r--', linewidth=1, alpha=0.6, label='Odometry Path')
    
    # Add scan matching status if enabled
    if enable_scan_matching:
        match_text = ax.text(0.02, 0.98, "Using Improved ICP Scan Matching", 
                            transform=ax.transAxes, va='top', ha='left', 
                            color='green', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.7))
    
    # Add grid and labels
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('2D LiDAR Scan Visualization')
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # If save_grid is enabled, save the final map
    if save_grid and show_occupancy_grid:
        # Generate a timestamp-based filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(maps_dir, f"final_map_{timestamp}")
        
        # Create path coordinates for saving
        robot_path_coords = list(zip(robot_path_x, robot_path_y))
        
        # Get start and current positions from path
        start_pos = (robot_path_x[0], robot_path_y[0]) if len(robot_path_x) > 0 else None
        current_pos = (robot_path_x[-1], robot_path_y[-1]) if len(robot_path_x) > 0 else None
        
        # Save the grid with robot path and positions
        saved_files = occupancy_grid.save_to_file(
            base_filename, 
            format=save_format, 
            include_metadata=True,
            robot_path=robot_path_coords,
            start_position=start_pos,
            current_position=current_pos
        )
        
        print(f"\nFinal occupancy grid map saved to:")
        for file in saved_files:
            print(f"  - {file}")

def main():
    """
    Main function to parse arguments and run the visualization
    """
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='LiDAR Visualization and Localization')
    
    # Add arguments
    parser.add_argument('--file', type=str, default="./lidar_slam/dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max_entries', type=int, default=3162,
                       help='Maximum number of entries to read from the file')
    parser.add_argument('--grid', action='store_true', default=True,
                       help='Enable occupancy grid mapping')
    parser.add_argument('--resolution', type=float, default=0.05,
                       help='Resolution of the occupancy grid in meters')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save the final occupancy grid map')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'npy', 'csv', 'all'],
                       help='Format to save the grid')
    parser.add_argument('--scan_matching', action='store_true', default=True,
                       help='Enable scan matching localization algorithm')
    parser.add_argument('--debug', type=int, default=1, choices=[0, 1, 2, 3],
                       help='Debug level (0=none, 1=basic, 2=detailed, 3=verbose)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the visualization
    visualize_lidar_data_realtime(
        file_path=args.file,
        max_entries=args.max_entries,
        show_occupancy_grid=args.grid,
        grid_resolution=args.resolution,
        save_grid=args.save,
        save_format=args.format,
        enable_scan_matching=args.scan_matching
    )

# Main execution
if __name__ == "__main__":
    main()