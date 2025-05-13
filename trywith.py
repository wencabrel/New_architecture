import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import math
import os
import time
from matplotlib.widgets import Button
from scipy.spatial import KDTree
import scipy.optimize as optimize
from sklearn.neighbors import NearestNeighbors
from collections import deque

#########################################################################
# UTILITY FUNCTIONS
#########################################################################

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

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return ((angle + math.pi) % (2 * math.pi)) - math.pi

#########################################################################
# OCCUPANCY GRID CLASS
#########################################################################

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
# POSE ESTIMATE CLASS
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
    
    def get_transform_matrix(self):
        """Get the 3x3 homogeneous transformation matrix for this pose"""
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        # Create 3x3 transform matrix for 2D pose
        transform = np.array([
            [cos_theta, -sin_theta, self.x],
            [sin_theta, cos_theta, self.y],
            [0, 0, 1]
        ])
        
        return transform
    
    def apply_transform(self, transform):
        """Apply a transformation matrix to this pose"""
        # Extract the rotation part (2x2)
        rotation = transform[:2, :2]
        
        # Extract translation part (2x1)
        translation = transform[:2, 2]
        
        # Calculate new rotation angle
        new_theta = math.atan2(rotation[1, 0], rotation[0, 0])
        
        # Apply the transformation
        self.theta = normalize_angle(self.theta + new_theta)
        
        # Apply the translation (rotated by current theta)
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        delta_pos = np.dot(rot_matrix, translation)
        self.x += delta_pos[0]
        self.y += delta_pos[1]
        
        return self
    
    def interpolate(self, other_pose, factor):
        """
        Interpolate between this pose and another by the given factor
        
        Args:
            other_pose: The other pose to interpolate with
            factor: Interpolation factor (0.0 = this pose, 1.0 = other pose)
            
        Returns:
            A new interpolated pose
        """
        # Linear interpolation for position
        x = self.x + factor * (other_pose.x - self.x)
        y = self.y + factor * (other_pose.y - self.y)
        
        # Spherical linear interpolation for orientation
        delta_theta = normalize_angle(other_pose.theta - self.theta)
        theta = normalize_angle(self.theta + factor * delta_theta)
        
        return PoseEstimate(x, y, theta)
    
    def distance_to(self, other_pose):
        """Calculate Euclidean distance to another pose"""
        dx = self.x - other_pose.x
        dy = self.y - other_pose.y
        return math.sqrt(dx*dx + dy*dy)
    
    def angle_difference(self, other_pose):
        """Calculate the absolute angular difference to another pose"""
        return abs(normalize_angle(self.theta - other_pose.theta))

#########################################################################
# ROBUST ROTATION HANDLER CLASS
#########################################################################

class RobustRotationHandler:
    """Class to handle large rotations in scan matching more robustly"""
    
    def __init__(self, max_rotation_per_increment=0.15, num_increments=10, 
                min_correspondences_per_increment=3, max_correspondence_distance=1.5,
                debug_level=1):
        """
        Initialize with configurable parameters for rotation handling
        
        Args:
            max_rotation_per_increment: Maximum rotation per increment (radians)
            num_increments: Maximum number of increments to break rotation into
            min_correspondences_per_increment: Minimum correspondences required in each step
            max_correspondence_distance: Maximum distance for point correspondences
            debug_level: Verbosity level for debug messages
        """
        self.max_rotation_per_increment = max_rotation_per_increment
        self.num_increments = num_increments
        self.min_correspondences_per_increment = min_correspondences_per_increment
        self.max_correspondence_distance = max_correspondence_distance
        self.debug_level = debug_level
        self.max_total_rotation = max_rotation_per_increment * num_increments
        self.rotation_hypotheses = []
        
    def handle_large_rotation(self, scan_points, occupancy_grid, initial_pose, 
                             estimated_rotation, occupancy_threshold=0.55):
        """
        Handle large rotation by breaking it into smaller increments
        
        Args:
            scan_points: Array of scan points in robot local frame
            occupancy_grid: OccupancyGrid object
            initial_pose: Initial pose estimate
            estimated_rotation: Estimated rotation change
            occupancy_threshold: Threshold for considering cells occupied
            
        Returns:
            Best pose after handled rotation, or None if unsuccessful
        """
        # Clear previous hypotheses
        self.rotation_hypotheses = []
        
        # If rotation is within normal range, no special handling needed
        if abs(estimated_rotation) <= self.max_rotation_per_increment:
            return None
        
        if self.debug_level > 0:
            print(f"[RotationHandler] Handling large rotation: {estimated_rotation:.4f} rad")
        
        # Calculate number of increments needed for this rotation
        num_steps = min(self.num_increments, 
                       math.ceil(abs(estimated_rotation) / self.max_rotation_per_increment))
        
        # Calculate rotation increment
        increment = estimated_rotation / num_steps
        
        if self.debug_level > 1:
            print(f"[RotationHandler] Breaking into {num_steps} increments of {increment:.4f} rad each")
        
        # Process rotation incrementally
        current_pose = initial_pose.copy()
        best_pose = initial_pose.copy()
        best_score = 0.0
        
        # Create array for transformed points
        transformed_points = None
        
        # Iterate through increments
        for step in range(1, num_steps + 1):
            # Apply rotation increment
            test_pose = current_pose.copy()
            test_pose.theta += increment
            
            # Transform scan points to world frame
            transformed_points = self.transform_points_to_world(scan_points, test_pose)
            
            # Evaluate this pose hypothesis
            score, correspondences = self.evaluate_hypothesis(transformed_points, occupancy_grid, 
                                                           occupancy_threshold)
            
            # Store hypothesis
            self.rotation_hypotheses.append({
                'pose': test_pose.copy(),
                'score': score,
                'correspondences': len(correspondences),
                'increment': step
            })
            
            if self.debug_level > 1:
                print(f"[RotationHandler] Increment {step}/{num_steps}: score={score:.4f}, "
                      f"correspondences={len(correspondences)}")
            
            # Update if sufficient correspondences found
            if len(correspondences) >= self.min_correspondences_per_increment:
                current_pose = test_pose.copy()
                
                # Track best pose so far
                if score > best_score:
                    best_score = score
                    best_pose = test_pose.copy()
            else:
                # If we lose correspondences, try a smaller increment
                if self.debug_level > 0:
                    print(f"[RotationHandler] Warning: Lost correspondences at increment {step}. "
                          f"Only found {len(correspondences)}")
                
                # Try a half increment
                half_test_pose = current_pose.copy()
                half_test_pose.theta += increment / 2
                
                # Transform scan points
                half_transformed_points = self.transform_points_to_world(scan_points, half_test_pose)
                
                # Evaluate half increment
                half_score, half_correspondences = self.evaluate_hypothesis(
                    half_transformed_points, occupancy_grid, occupancy_threshold)
                
                if len(half_correspondences) >= self.min_correspondences_per_increment:
                    if self.debug_level > 0:
                        print(f"[RotationHandler] Half increment successful: score={half_score:.4f}, "
                              f"correspondences={len(half_correspondences)}")
                    
                    current_pose = half_test_pose.copy()
                    
                    if half_score > best_score:
                        best_score = half_score
                        best_pose = half_test_pose.copy()
                else:
                    # If even half increment fails, try a quarter increment
                    quarter_test_pose = current_pose.copy()
                    quarter_test_pose.theta += increment / 4
                    
                    # Evaluate quarter increment
                    quarter_transformed_points = self.transform_points_to_world(scan_points, quarter_test_pose)
                    quarter_score, quarter_correspondences = self.evaluate_hypothesis(
                        quarter_transformed_points, occupancy_grid, occupancy_threshold)
                    
                    if len(quarter_correspondences) >= self.min_correspondences_per_increment:
                        if self.debug_level > 0:
                            print(f"[RotationHandler] Quarter increment successful: score={quarter_score:.4f}, "
                                  f"correspondences={len(quarter_correspondences)}")
                        
                        current_pose = quarter_test_pose.copy()
                        
                        if quarter_score > best_score:
                            best_score = quarter_score
                            best_pose = quarter_test_pose.copy()
                    else:
                        # If we can't find a small enough increment that works, break
                        if self.debug_level > 0:
                            print(f"[RotationHandler] Failed to maintain correspondences. "
                                  f"Stopping at increment {step}")
                        
                        break
        
        # Check if we improved over the initial pose
        if best_score > 0.3:  # Threshold for accepting rotation results
            if self.debug_level > 0:
                print(f"[RotationHandler] Rotation handling successful. "
                      f"Best score: {best_score:.4f}")
            
            return best_pose
        else:
            if self.debug_level > 0:
                print(f"[RotationHandler] Rotation handling unsuccessful. "
                      f"Best score too low: {best_score:.4f}")
            
            return None
    
    def transform_points_to_world(self, points, pose):
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
    
    def evaluate_hypothesis(self, transformed_points, occupancy_grid, occupancy_threshold):
        """
        Evaluate how well a pose hypothesis matches the map
        
        Args:
            transformed_points: Array of scan points in world frame
            occupancy_grid: OccupancyGrid object
            occupancy_threshold: Threshold for considering cells occupied
            
        Returns:
            Score and list of correspondences
        """
        correspondences = []
        total_score = 0.0
        
        for point in transformed_points:
            # Skip points outside the map with a buffer
            buffer = 1.0  # 1 meter buffer
            if (abs(point[0]) >= occupancy_grid.width/2 - buffer or 
                abs(point[1]) >= occupancy_grid.height/2 - buffer):
                continue
                
            # Convert to grid coordinates
            grid_x, grid_y = occupancy_grid.world_to_grid(point[0], point[1])
            
            # Make sure the grid coordinates are valid
            if not (0 <= grid_x < occupancy_grid.grid_width and 0 <= grid_y < occupancy_grid.grid_height):
                continue
                
            # Find closest occupied cell within search radius
            closest_cell, distance = self.find_closest_occupied_cell(
                grid_x, grid_y, occupancy_grid, occupancy_threshold)
            
            if closest_cell is not None and distance < self.max_correspondence_distance / occupancy_grid.resolution:
                # Convert back to world coordinates
                world_x, world_y = occupancy_grid.grid_to_world(closest_cell[0], closest_cell[1])
                
                # Add correspondence
                correspondences.append({
                    'scan_point': point,
                    'map_point': np.array([world_x, world_y]),
                    'distance': distance * occupancy_grid.resolution  # Convert to meters
                })
                
                # Calculate score based on distance (closer is better)
                point_score = 1.0 - (distance * occupancy_grid.resolution) / self.max_correspondence_distance
                total_score += point_score
        
        # Normalize score by number of correspondences
        average_score = total_score / len(correspondences) if correspondences else 0.0
        
        return average_score, correspondences
    
    def find_closest_occupied_cell(self, grid_x, grid_y, occupancy_grid, occupancy_threshold):
        """
        Find the closest occupied cell to the given grid coordinates
        
        Args:
            grid_x, grid_y: Grid coordinates to search from
            occupancy_grid: OccupancyGrid object
            occupancy_threshold: Threshold for considering cells occupied
            
        Returns:
            Closest occupied cell coordinates and distance
        """
        # Define search radius (in grid cells)
        search_radius = int(self.max_correspondence_distance / occupancy_grid.resolution)
        
        min_distance = float('inf')
        closest_cell = None
        
        # Simple grid search in a square area
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                
                # Check if within grid bounds
                if (0 <= nx < occupancy_grid.grid_width and 0 <= ny < occupancy_grid.grid_height):
                    # Check if this cell is occupied - USING CURRENT THRESHOLD
                    if occupancy_grid.grid[ny, nx] > occupancy_threshold:
                        # Calculate Euclidean distance
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_cell = (nx, ny)
        
        return closest_cell, min_distance
    
    def visualize_rotation_handling(self, scan_points, occupancy_grid):
        """
        Visualize the rotation handling process with all hypotheses
        
        Args:
            scan_points: Original scan points
            occupancy_grid: OccupancyGrid object
            
        Returns:
            Matplotlib figure
        """
        if not self.rotation_hypotheses:
            print("[RotationHandler] No rotation hypotheses to visualize")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the map
        if occupancy_grid:
            # Custom colormap
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                occupancy_grid.get_grid_for_display(),
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-occupancy_grid.width/2, occupancy_grid.width/2, 
                       -occupancy_grid.height/2, occupancy_grid.height/2]
            )
        
        # Create color gradient for hypotheses
        colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(self.rotation_hypotheses)))
        
        # Plot each hypothesis with its transformed points
        for i, hypothesis in enumerate(self.rotation_hypotheses):
            pose = hypothesis['pose']
            
            # Transform scan points
            transformed_points = self.transform_points_to_world(scan_points, pose)
            
            # Plot the transformed points
            ax.scatter(
                transformed_points[:, 0], 
                transformed_points[:, 1],
                c=[colors_gradient[i]],
                s=3,
                alpha=0.5,
                label=f"Increment {hypothesis['increment']}"
            )
            
            # Plot the pose
            ax.scatter(
                pose.x, 
                pose.y, 
                c=[colors_gradient[i]],
                s=50,
                marker='*'
            )
            
            # Add an arrow for orientation
            arrow_length = 0.5
            dx = arrow_length * math.cos(pose.theta)
            dy = arrow_length * math.sin(pose.theta)
            
            ax.arrow(
                pose.x, pose.y, dx, dy,
                head_width=0.1, 
                head_length=0.1, 
                fc=colors_gradient[i], 
                ec=colors_gradient[i],
                alpha=0.7
            )
        
        # Find the best hypothesis
        best_hypothesis = max(self.rotation_hypotheses, key=lambda h: h['score'])
        
        # Highlight the best hypothesis
        ax.scatter(
            best_hypothesis['pose'].x, 
            best_hypothesis['pose'].y, 
            c='red',
            s=100,
            marker='*',
            label=f"Best (Score: {best_hypothesis['score']:.3f})"
        )
        
        # Add grid and labels
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Rotation Handling Visualization')
        
        # Add legend (showing only a few to avoid clutter)
        num_to_show = min(5, len(self.rotation_hypotheses))
        step = len(self.rotation_hypotheses) // num_to_show
        handles, labels = ax.get_legend_handles_labels()
        
        # Only show a subset of the hypotheses in the legend
        indices = list(range(0, len(handles)-1, step)) + [len(handles)-1]
        ax.legend([handles[i] for i in indices], [labels[i] for i in indices], loc='best')
        
        plt.tight_layout()
        return fig

#########################################################################
# LOOP CLOSURE DETECTOR CLASS
#########################################################################

class LoopClosureDetector:
    """Class to detect and handle loop closures for SLAM optimization"""
    
    def __init__(self, min_loop_length=50, distance_threshold=2.0, scan_similarity_threshold=0.7,
                min_matches=10, keyframe_interval=5, correction_ratio=0.5, debug_level=1):
        """
        Initialize with parameters to control loop closure detection sensitivity
        
        Args:
            min_loop_length: Minimum number of poses between potential loop closures
            distance_threshold: Maximum distance (meters) for potential loop closures
            scan_similarity_threshold: Minimum similarity score for scans to be considered similar
            min_matches: Minimum number of matched points required for a valid loop closure
            keyframe_interval: Interval for keyframe creation
            correction_ratio: How much of the detected error to correct (0.0-1.0)
            debug_level: Verbosity level for debug output
        """
        self.min_loop_length = min_loop_length
        self.distance_threshold = distance_threshold
        self.scan_similarity_threshold = scan_similarity_threshold
        self.min_matches = min_matches
        self.keyframe_interval = keyframe_interval
        self.correction_ratio = correction_ratio
        self.debug_level = debug_level
        
        # Storage for keyframes and descriptors
        self.keyframes = []
        self.loop_closures = []
        self.current_detection_data = None
        
        # Feature extraction parameters
        self.feature_grid_size = 0.5  # meters
        self.min_feature_points = 3   # minimum points to consider a cell a feature
        
        # For pose graph optimization
        self.pose_constraints = []
        
        if self.debug_level > 0:
            print(f"[LoopClosureDetector] Initialized with: min_loop_length={min_loop_length}, "
                 f"distance_threshold={distance_threshold}m, "
                 f"similarity_threshold={scan_similarity_threshold}, "
                 f"min_matches={min_matches}")
    
    def add_keyframe(self, pose, scan_points, trajectory_index, trajectory_length, timestamp=None):
        """
        Store important frames with distinctive features for later comparison
        
        Args:
            pose: The pose estimate for this keyframe
            scan_points: Scan points in robot local frame
            trajectory_index: Current index in the trajectory
            trajectory_length: Total number of poses in trajectory
            timestamp: Optional timestamp
            
        Returns:
            Boolean indicating if keyframe was added
        """
        # Only add keyframes periodically to save computation
        if len(self.keyframes) > 0 and trajectory_index % self.keyframe_interval != 0:
            return False
            
        # Skip if we don't have enough scan points
        if len(scan_points) < 10:
            return False
        
        # Calculate features for this scan
        features = self.extract_features(scan_points, pose)
        
        # Generate scan descriptor
        scan_descriptor = self.generate_scan_descriptor(scan_points)
        
        # Add as a new keyframe
        keyframe_id = len(self.keyframes)
        self.keyframes.append({
            'id': keyframe_id,
            'pose': pose.copy(),
            'scan_points': scan_points.copy() if isinstance(scan_points, np.ndarray) else np.array(scan_points),
            'timestamp': timestamp,
            'features': features,
            'descriptor': scan_descriptor,
            'trajectory_index': trajectory_index,
            'trajectory_progress': trajectory_index / max(1, trajectory_length - 1)
        })
        
        if self.debug_level > 1:
            print(f"[LoopClosureDetector] Added keyframe #{keyframe_id} at trajectory index {trajectory_index} "
                 f"({features['num_features']} features)")
        
        return True
    
    def detect_loop_closures(self, current_pose, scan_points, trajectory, current_index):
        """
        Find potential loop closures by comparing current scan to keyframes
        
        Args:
            current_pose: Current robot pose
            scan_points: Current scan points in robot local frame
            trajectory: Full trajectory so far
            current_index: Current index in trajectory
            
        Returns:
            List of detected loop closures
        """
        # Only try loop closure with enough history
        if len(self.keyframes) < 5 or current_index < self.min_loop_length:
            return []
            
        # Calculate current scan descriptor
        current_descriptor = self.generate_scan_descriptor(scan_points)
        
        # Find keyframes that could be loop closures
        candidates = []
        
        for keyframe in self.keyframes:
            # Skip recent frames to ensure minimum loop length
            keyframe_idx = keyframe['trajectory_index']
            if current_index - keyframe_idx < self.min_loop_length:
                continue
                
            keyframe_pose = keyframe['pose']
            
            # Calculate distance between poses
            dx = current_pose.x - keyframe_pose.x
            dy = current_pose.y - keyframe_pose.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If within distance threshold, compare descriptors
            if distance < self.distance_threshold:
                descriptor_similarity = self.calculate_descriptor_similarity(
                    current_descriptor, keyframe['descriptor'])
                
                if descriptor_similarity > self.scan_similarity_threshold:
                    candidates.append({
                        'keyframe_id': keyframe['id'],
                        'keyframe_idx': keyframe_idx,
                        'current_idx': current_index,
                        'distance': distance,
                        'similarity': descriptor_similarity
                    })
                    
                    if self.debug_level > 1:
                        print(f"[LoopClosureDetector] Potential loop closure: keyframe #{keyframe['id']} "
                             f"(trajectory idx {keyframe_idx}) -> current (idx {current_index}), "
                             f"distance={distance:.2f}m, similarity={descriptor_similarity:.3f}")
        
        # Sort candidates by similarity score
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take best candidates and verify with ICP
        verified_closures = []
        max_candidates = 3  # Only verify top candidates to save computation
        
        for i, candidate in enumerate(candidates[:max_candidates]):
            keyframe = self.keyframes[candidate['keyframe_id']]
            
            # Verify with ICP
            transform, inliers, score = self.verify_loop_closure_with_icp(
                scan_points, keyframe['scan_points'], current_pose, keyframe['pose'])
            
            if inliers >= self.min_matches and score > 0.5:
                # This is a valid loop closure
                verified = {
                    'keyframe_id': candidate['keyframe_id'],
                    'current_idx': current_index,
                    'keyframe_idx': keyframe['trajectory_index'],
                    'transform': transform,
                    'inliers': inliers,
                    'score': score,
                    'error_x': transform[0, 2],
                    'error_y': transform[1, 2],
                    'error_theta': math.atan2(transform[1, 0], transform[0, 0]) if transform is not None else 0.0
                }
                
                # Calculate error magnitude
                error_magnitude = math.sqrt(verified['error_x']**2 + verified['error_y']**2)
                verified['error_magnitude'] = error_magnitude
                
                # Only accept if error isn't too large
                max_acceptable_error = 3.0  # meters
                if error_magnitude < max_acceptable_error:
                    verified_closures.append(verified)
                    
                    if self.debug_level > 0:
                        print(f"[LoopClosureDetector] VERIFIED loop closure: keyframe #{candidate['keyframe_id']} "
                             f"(idx {keyframe['trajectory_index']}) -> current (idx {current_index}), "
                             f"error: {error_magnitude:.3f}m, {verified['error_theta']:.3f}rad, "
                             f"inliers: {inliers}, score: {score:.3f}")
                else:
                    if self.debug_level > 0:
                        print(f"[LoopClosureDetector] Loop closure REJECTED due to large error: {error_magnitude:.3f}m")
        
        # Store detected closures
        if verified_closures:
            self.loop_closures.extend(verified_closures)
        
        # Apply loop closure correction if we have verified closures
        if verified_closures:
            corrected_trajectory = self.apply_loop_closure_correction(
                trajectory, verified_closures[0], current_index)
            
            return verified_closures, corrected_trajectory
        
        return [], None
    
    def extract_features(self, scan_points, pose):
        """
        Extract distinctive features from scan points
        
        Args:
            scan_points: Scan points in robot local frame
            pose: Robot pose
            
        Returns:
            Dictionary of features
        """
        # Transform points to world frame
        if isinstance(scan_points, list):
            scan_points = np.array(scan_points)
            
        # Create rotation matrix
        c = math.cos(pose.theta)
        s = math.sin(pose.theta)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        # Apply rotation
        rotated_points = np.dot(scan_points, rotation_matrix.T)
        
        # Apply translation
        world_points = rotated_points + np.array([pose.x, pose.y])
        
        # Create a grid-based feature representation
        # Divide the space into grid cells and count points in each cell
        grid_size = self.feature_grid_size
        
        # Find min and max coordinates
        min_x, max_x = np.min(world_points[:, 0]), np.max(world_points[:, 0])
        min_y, max_y = np.min(world_points[:, 1]), np.max(world_points[:, 1])
        
        # Calculate grid dimensions
        grid_cols = max(3, int(np.ceil((max_x - min_x) / grid_size)))
        grid_rows = max(3, int(np.ceil((max_y - min_y) / grid_size)))
        
        # Create grid
        point_grid = np.zeros((grid_rows, grid_cols), dtype=int)
        
        # Count points in each grid cell
        for point in world_points:
            col = min(grid_cols - 1, max(0, int((point[0] - min_x) / grid_size)))
            row = min(grid_rows - 1, max(0, int((point[1] - min_y) / grid_size)))
            point_grid[row, col] += 1
        
        # Identify feature cells (cells with more than min_feature_points)
        feature_cells = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if point_grid[row, col] >= self.min_feature_points:
                    # Calculate world coordinates of cell center
                    cell_x = min_x + (col + 0.5) * grid_size
                    cell_y = min_y + (row + 0.5) * grid_size
                    
                    # Add as a feature
                    feature_cells.append({
                        'x': cell_x,
                        'y': cell_y,
                        'row': row,
                        'col': col,
                        'count': point_grid[row, col]
                    })
        
        return {
            'grid': point_grid,
            'grid_origin': (min_x, min_y),
            'grid_size': grid_size,
            'grid_dims': (grid_rows, grid_cols),
            'features': feature_cells,
            'num_features': len(feature_cells)
        }
    
    def generate_scan_descriptor(self, scan_points):
        """
        Generate a compact descriptor for scan matching
        
        Args:
            scan_points: Scan points in robot local frame
            
        Returns:
            Descriptor dictionary
        """
        if len(scan_points) < 3:
            return {
                'valid': False,
                'num_points': len(scan_points)
            }
            
        if isinstance(scan_points, list):
            scan_points = np.array(scan_points)
        
        # Calculate statistics about the scan
        distances = np.sqrt(np.sum(scan_points**2, axis=1))
        angles = np.arctan2(scan_points[:, 1], scan_points[:, 0])
        
        # Sort angles and corresponding distances
        sorted_indices = np.argsort(angles)
        sorted_angles = angles[sorted_indices]
        sorted_distances = distances[sorted_indices]
        
        # Create histogram of distances at different angles
        num_bins = 36  # 10-degree bins
        angle_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        
        # Compute histogram
        hist, _ = np.histogram(sorted_angles, bins=angle_bins)
        
        # Compute mean distance in each bin
        mean_distances = np.zeros(num_bins)
        std_distances = np.zeros(num_bins)
        
        for i in range(num_bins):
            bin_mask = (sorted_angles >= angle_bins[i]) & (sorted_angles < angle_bins[i+1])
            bin_distances = sorted_distances[bin_mask]
            
            if len(bin_distances) > 0:
                mean_distances[i] = np.mean(bin_distances)
                std_distances[i] = np.std(bin_distances)
        
        # Compute shape indicators
        compactness = np.var(distances) / (np.mean(distances) + 1e-6)
        
        # Calculate the bounding box of the scan
        min_x, max_x = np.min(scan_points[:, 0]), np.max(scan_points[:, 0])
        min_y, max_y = np.min(scan_points[:, 1]), np.max(scan_points[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        aspect_ratio = width / (height + 1e-6)
        
        return {
            'valid': True,
            'num_points': len(scan_points),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances),
            'compactness': compactness,
            'aspect_ratio': aspect_ratio,
            'angle_histogram': hist,
            'distance_profile': mean_distances,
            'distance_std': std_distances,
            'bounding_box': (min_x, min_y, max_x, max_y)
        }
    
    def calculate_descriptor_similarity(self, desc1, desc2):
        """
        Calculate similarity between two scan descriptors
        
        Args:
            desc1, desc2: Scan descriptors
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Check if descriptors are valid
        if not desc1.get('valid', False) or not desc2.get('valid', False):
            return 0.0
        
        # Calculate similarity based on multiple metrics
        
        # 1. Compare distance profiles (normalized correlation)
        profile1 = desc1['distance_profile']
        profile2 = desc2['distance_profile']
        
        # Handle potential rotation differences by computing correlation with shifts
        max_correlation = 0.0
        num_bins = len(profile1)
        
        for shift in range(num_bins):
            # Roll the second profile to simulate rotation
            shifted_profile = np.roll(profile2, shift)
            
            # Calculate correlation
            correlation = np.corrcoef(profile1, shifted_profile)[0, 1]
            
            if correlation > max_correlation:
                max_correlation = correlation
        
        # 2. Compare shape metrics
        ratio_similarity = 1.0 - min(1.0, abs(desc1['aspect_ratio'] - desc2['aspect_ratio']) / max(desc1['aspect_ratio'], desc2['aspect_ratio']))
        size_similarity = 1.0 - min(1.0, abs(desc1['mean_distance'] - desc2['mean_distance']) / max(desc1['mean_distance'], desc2['mean_distance']))
        compactness_similarity = 1.0 - min(1.0, abs(desc1['compactness'] - desc2['compactness']) / max(desc1['compactness'], desc2['compactness']))
        
        # 3. Consider points count - similar number of visible points
        point_ratio = min(desc1['num_points'], desc2['num_points']) / max(desc1['num_points'], desc2['num_points'])
        
        # Combine metrics with weights
        similarity = (
            0.5 * max(0, max_correlation) +  # Profile correlation
            0.15 * ratio_similarity +        # Shape aspect ratio
            0.15 * size_similarity +         # Overall size
            0.1 * compactness_similarity +   # Compactness
            0.1 * point_ratio                # Points count
        )
        
        return min(1.0, max(0.0, similarity))
    
    def verify_loop_closure_with_icp(self, current_scan, keyframe_scan, current_pose, keyframe_pose):
        """
        Verify potential loop closure using ICP to align scans
        
        Args:
            current_scan: Current scan points in robot local frame
            keyframe_scan: Keyframe scan points in its local frame
            current_pose: Current pose estimate
            keyframe_pose: Keyframe pose estimate
            
        Returns:
            Transformation matrix, number of inliers, and score
        """
        # Convert to numpy arrays if needed
        if isinstance(current_scan, list):
            current_scan = np.array(current_scan)
        if isinstance(keyframe_scan, list):
            keyframe_scan = np.array(keyframe_scan)
        
        # Transform scans to world frame
        current_world = self.transform_scan_to_world(current_scan, current_pose)
        keyframe_world = self.transform_scan_to_world(keyframe_scan, keyframe_pose)
        
        # Use ICP to align the scans
        try:
            # Initialize transformation
            init_transform = np.eye(3)
            
            # Run ICP
            final_transform, distances, iterations = self.icp_align(
                current_world, keyframe_world, init_transform, max_iterations=20)
            
            if final_transform is None:
                return None, 0, 0.0
            
            # Count inliers (points with distance below threshold)
            inlier_threshold = 0.3  # meters
            inliers = np.sum(distances < inlier_threshold)
            
            # Calculate score based on inlier ratio and mean distance
            if len(distances) > 0:
                inlier_ratio = inliers / len(distances)
                mean_error = np.mean(distances)
                score = inlier_ratio * (1.0 - min(1.0, mean_error / inlier_threshold))
            else:
                score = 0.0
            
            return final_transform, inliers, score
            
        except Exception as e:
            if self.debug_level > 0:
                print(f"[LoopClosureDetector] Error in ICP: {e}")
            
            return None, 0, 0.0
    
    def transform_scan_to_world(self, scan, pose):
        """
        Transform scan from robot local frame to world frame
        
        Args:
            scan: Array of [x, y] points in robot's local frame
            pose: Robot pose 
            
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
        rotated_points = np.dot(scan, rotation_matrix.T)
        
        # Apply translation
        transformed_points = rotated_points + np.array([x, y])
        
        return transformed_points
    
    def icp_align(self, points1, points2, initial_transform=None, max_iterations=20, 
                 distance_threshold=1.0, convergence_threshold=1e-5):
        """
        Align two point clouds using the Iterative Closest Point algorithm
        
        Args:
            points1: First point cloud (Nx2 array)
            points2: Second point cloud (Nx2 array)
            initial_transform: Initial transformation matrix (3x3)
            max_iterations: Maximum number of iterations
            distance_threshold: Maximum distance for point correspondences
            convergence_threshold: Threshold for convergence
            
        Returns:
            Transformation matrix, distances, and number of iterations
        """
        if initial_transform is None:
            initial_transform = np.eye(3)
            
        # Convert to homogeneous coordinates
        points1_h = np.hstack((points1, np.ones((points1.shape[0], 1))))
        current_transform = initial_transform.copy()
        
        # Create nearest neighbor structure for points2
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points2)
        
        # Store info for each iteration
        prev_error = float('inf')
        
        # Main ICP loop
        for iteration in range(max_iterations):
            # Apply current transformation
            transformed_points = points1_h.dot(current_transform.T)[:, :2]
            
            # Find nearest neighbors
            distances, indices = nbrs.kneighbors(transformed_points)
            distances = distances.ravel()
            
            # Filter outliers
            mask = distances < distance_threshold
            if np.sum(mask) < 3:  # Need at least 3 points for a valid transform
                if self.debug_level > 1:
                    print(f"[LoopClosureDetector] ICP failed: not enough valid correspondences")
                return None, np.array([]), 0
                
            p = transformed_points[mask]
            q = points2[indices[mask].ravel()]
            
            # Computer centroid
            p_centroid = np.mean(p, axis=0)
            q_centroid = np.mean(q, axis=0)
            
            # Center the points
            p_centered = p - p_centroid
            q_centered = q - q_centroid
            
            # Compute the covariance matrix
            H = p_centered.T.dot(q_centered)
            
            # Single Value Decomposition
            try:
                U, _, Vt = np.linalg.svd(H)
                
                # Calculate rotation
                R = Vt.T.dot(U.T)
                
                # Ensure proper rotation matrix (det=1)
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T.dot(U.T)
                
                # Calculate translation
                t = q_centroid - R.dot(p_centroid)
                
                # Create transformation matrix
                transform = np.eye(3)
                transform[:2, :2] = R
                transform[:2, 2] = t
                
                # Update current transformation
                current_transform = transform.dot(current_transform)
                
                # Calculate error
                current_error = np.mean(distances[mask])
                
                # Check for convergence
                if abs(prev_error - current_error) < convergence_threshold:
                    break
                    
                prev_error = current_error
                
            except Exception as e:
                if self.debug_level > 0:
                    print(f"[LoopClosureDetector] SVD failed in ICP: {e}")
                return None, np.array([]), 0
        
        return current_transform, distances, iteration + 1
    
    def apply_loop_closure_correction(self, trajectory, loop_closure, current_index):
        """
        Apply correction to trajectory based on the detected loop closure
        
        Args:
            trajectory: List of PoseEstimate objects
            loop_closure: Detected loop closure information
            current_index: Current index in the trajectory
            
        Returns:
            Corrected trajectory
        """
        # Extract loop closure information
        keyframe_idx = loop_closure['keyframe_idx']
        transform = loop_closure['transform']
        error_x = loop_closure['error_x']
        error_y = loop_closure['error_y']
        error_theta = loop_closure['error_theta']
        
        if self.debug_level > 0:
            print(f"[LoopClosureDetector] Applying correction: error_x={error_x:.3f}, "
                 f"error_y={error_y:.3f}, error_theta={error_theta:.3f} rad")
        
        # Create a copy of the trajectory
        corrected_trajectory = [pose.copy() for pose in trajectory]
        
        # Calculate the correction distribution factor - gradually apply more correction
        # to poses further from the loop closure point
        for i in range(keyframe_idx + 1, len(corrected_trajectory)):
            # Calculate correction factor: 0 at keyframe_idx, correction_ratio at current_index
            if i <= current_index:
                alpha = self.correction_ratio * (i - keyframe_idx) / (current_index - keyframe_idx + 1e-10)
            else:
                alpha = self.correction_ratio
            
            # Apply the position correction
            corrected_trajectory[i].x -= error_x * alpha
            corrected_trajectory[i].y -= error_y * alpha
            
            # Apply the rotation correction
            corrected_trajectory[i].theta -= error_theta * alpha
            corrected_trajectory[i].theta = normalize_angle(corrected_trajectory[i].theta)
            
            # Store the correction as a constraint for pose graph optimization
            if i % 10 == 0:  # Don't store too many constraints
                self.pose_constraints.append({
                    'pose_idx': i,
                    'reference_idx': keyframe_idx,
                    'transform': transform,
                    'correction': (error_x * alpha, error_y * alpha, error_theta * alpha),
                    'information': np.eye(3)  # Information matrix (inverse of covariance)
                })
        
        return corrected_trajectory
    
    def visualize_loop_closures(self, occupancy_grid=None, current_trajectory=None):
        """
        Visualize detected loop closures on the map
        
        Args:
            occupancy_grid: OccupancyGrid object
            current_trajectory: Current robot trajectory
            
        Returns:
            Matplotlib figure
        """
        if not self.loop_closures:
            print("[LoopClosureDetector] No loop closures to visualize")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the map if provided
        if occupancy_grid:
            # Custom colormap
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                occupancy_grid.get_grid_for_display(),
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-occupancy_grid.width/2, occupancy_grid.width/2, 
                       -occupancy_grid.height/2, occupancy_grid.height/2]
            )
        
        # Plot the trajectory if provided
        if current_trajectory:
            traj_x = [pose.x for pose in current_trajectory]
            traj_y = [pose.y for pose in current_trajectory]
            ax.plot(traj_x, traj_y, 'b-', linewidth=1, alpha=0.6, label='Trajectory')
            
            # Add markers for start and current position
            ax.scatter(traj_x[0], traj_y[0], c='green', s=100, marker='*', label='Start')
            ax.scatter(traj_x[-1], traj_y[-1], c='blue', s=100, marker='*', label='Current')
        
        # Plot keyframes
        keyframe_x = [kf['pose'].x for kf in self.keyframes]
        keyframe_y = [kf['pose'].y for kf in self.keyframes]
        ax.scatter(keyframe_x, keyframe_y, c='purple', s=30, marker='o', label='Keyframes')
        
        # Plot loop closures
        for i, lc in enumerate(self.loop_closures):
            keyframe = self.keyframes[lc['keyframe_id']]
            
            # Draw line connecting the loop closure points
            if current_trajectory:
                current_pose = current_trajectory[lc['current_idx']]
                ax.plot(
                    [keyframe['pose'].x, current_pose.x],
                    [keyframe['pose'].y, current_pose.y],
                    'r-', linewidth=2, alpha=0.7
                )
                
                # Add text label
                mid_x = (keyframe['pose'].x + current_pose.x) / 2
                mid_y = (keyframe['pose'].y + current_pose.y) / 2
                ax.text(mid_x, mid_y, f"LC{i+1}: {lc['error_magnitude']:.2f}m",
                       color='red', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7))
        
        # Add grid and labels
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Loop Closure Visualization')
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def optimize_pose_graph(self, trajectory):
        """
        Perform pose graph optimization to globally correct the trajectory
        
        Args:
            trajectory: List of PoseEstimate objects
            
        Returns:
            Optimized trajectory
        """
        if not self.pose_constraints:
            return trajectory
            
        if self.debug_level > 0:
            print(f"[LoopClosureDetector] Optimizing pose graph with {len(self.pose_constraints)} constraints")
        
        # Create a copy of the trajectory
        optimized_trajectory = [pose.copy() for pose in trajectory]
        
        # This is a simplified pose graph optimization
        # In a complete implementation, you would use a proper pose graph optimization
        # library like g2o or GTSAM
        
        # For now, we'll use a simple averaging approach to distribute corrections
        for constraint in self.pose_constraints:
            pose_idx = constraint['pose_idx']
            reference_idx = constraint['reference_idx']
            
            if pose_idx < len(optimized_trajectory) and reference_idx < len(optimized_trajectory):
                # Apply the stored correction
                correction_x, correction_y, correction_theta = constraint['correction']
                
                optimized_trajectory[pose_idx].x -= correction_x
                optimized_trajectory[pose_idx].y -= correction_y
                optimized_trajectory[pose_idx].theta = normalize_angle(
                    optimized_trajectory[pose_idx].theta - correction_theta)
        
        if self.debug_level > 0:
            print(f"[LoopClosureDetector] Pose graph optimization complete")
        
        return optimized_trajectory

#########################################################################
# IMPROVED SCAN MATCHING LOCALIZATION CLASS
#########################################################################

class ImprovedScanMatchingLocalization:
    """
    Improved implementation of scan matching localization using ICP algorithm
    with adaptive parameters, robust rotation handling, and loop closure detection.
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
        
        # Initialize robust rotation handler
        self.enable_robust_rotation = True
        self.rotation_handler = RobustRotationHandler(
            max_rotation_per_increment=0.15,  # 8.6 degrees per step
            num_increments=10,               # Up to 86 degrees of total rotation
            debug_level=debug_level
        )
        
        # Initialize loop closure detector
        self.enable_loop_closure = True
        self.loop_detector = LoopClosureDetector(
            min_loop_length=100,            # Minimum poses between potential loop closures
            distance_threshold=3.0,         # Maximum distance for loop closures
            scan_similarity_threshold=0.6,  # Minimum similarity score
            debug_level=debug_level
        )
        
        # Track loop closures
        self.detected_loop_closures = []
        
        if self.debug_level > 0:
            print("[ScanMatcher] Initialized with adaptive parameters and robust rotation handling")
            if self.enable_robust_rotation:
                print("[ScanMatcher] Robust rotation handling is ENABLED")
            else:
                print("[ScanMatcher] Robust rotation handling is DISABLED")
                
            if self.enable_loop_closure:
                print("[ScanMatcher] Loop closure detection is ENABLED")
            else:
                print("[ScanMatcher] Loop closure detection is DISABLED")
                
            print(f"[ScanMatcher]   - Base correspondence distance: {self.default_correspondence_distance}m (can increase to {self.max_possible_correspondence_distance}m)")
            print(f"[ScanMatcher]   - Base occupancy threshold: {self.occupancy_threshold} (can decrease to {self.min_occupancy_threshold})")
            print(f"[ScanMatcher]   - Base max translation: {self.max_translation_per_frame}m (can increase to {self.max_possible_translation}m)")
            print(f"[ScanMatcher]   - Alignment reset interval: {self.reset_interval} frames (drift threshold: {self.drift_threshold}m)")
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
            
            # Convert the scan points to array for later use
            scan_points = np.column_stack((scan_x, scan_y))
            
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
                
                # Normalize the orientation angle
                initial_guess.theta = normalize_angle(initial_guess.theta)
                
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
                    # Normal mode - Check for large rotation first
                    estimated_rotation = normalize_angle(initial_guess.theta - self.last_matched_pose.theta)
                    
                    # Handle large rotations if enabled and needed
                    robust_rotation_applied = False
                    
                    if self.enable_robust_rotation and abs(estimated_rotation) > self.max_rotation_per_frame:
                        # Try to handle the large rotation
                        rotation_result = self.rotation_handler.handle_large_rotation(
                            scan_points, self.map, self.last_matched_pose, estimated_rotation, 
                            occupancy_threshold=self.occupancy_threshold
                        )
                        
                        if rotation_result is not None:
                            # We got a good rotation result, use it as the initial guess
                            initial_guess = rotation_result
                            robust_rotation_applied = True
                            
                            if self.debug_level > 0:
                                print(f"\n[ScanMatcher] Applied robust rotation handling. "
                                     f"New initial guess: theta={initial_guess.theta:.4f}")
                    
                    # Match current scan against the map using ICP
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
                            
                        # Detect loop closures if enabled
                        if self.enable_loop_closure and i > self.loop_detector.min_loop_length:
                            # First, add this pose and scan as a potential keyframe
                            added_keyframe = self.loop_detector.add_keyframe(
                                matched_pose, scan_points, len(self.trajectory)-1, len(lidar_data),
                                timestamp=scan_data['timestamp']
                            )
                            
                            # Every 10 frames, check for loop closures
                            if i % 10 == 0:
                                loop_closures, corrected_traj = self.loop_detector.detect_loop_closures(
                                    matched_pose, scan_points, self.trajectory, len(self.trajectory)-1
                                )
                                
                                if loop_closures:
                                    self.detected_loop_closures.extend(loop_closures)
                                    
                                    # If we got a corrected trajectory, update our current one
                                    if corrected_traj:
                                        if self.debug_level > 0:
                                            print(f"\n[ScanMatcher] Applying loop closure correction")
                                        
                                        self.trajectory = corrected_traj
                                        self.last_matched_pose = self.trajectory[-1]
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
            
            if self.enable_loop_closure and self.detected_loop_closures:
                print(f"[ScanMatcher] Detected {len(self.detected_loop_closures)} loop closures.")
                
                # Final global optimization
                optimized_trajectory = self.loop_detector.optimize_pose_graph(self.trajectory)
                if optimized_trajectory:
                    self.trajectory = optimized_trajectory
                    print(f"[ScanMatcher] Applied final global optimization from loop closures.")
        
        return self.trajectory
    
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
        dtheta = normalize_angle(pose2.theta - pose1.theta)
        
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
        
        # Normalize the angle
        corrected_pose.theta = normalize_angle(corrected_pose.theta)
        
        return corrected_pose
    
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
        relative_dtheta = normalize_angle(odometry_pose.theta - last_odometry_pose.theta)
        
        # Create recovery pose by using odometry movement from last matched pose
        recovery_pose = last_matched_pose.copy()
        recovery_pose.x += relative_dx
        recovery_pose.y += relative_dy
        recovery_pose.theta = normalize_angle(recovery_pose.theta + relative_dtheta)
        
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
        odom_delta_theta = normalize_angle(odometry_pose.theta - previous_pose.theta)
        
        guess_delta_x = initial_guess.x - previous_pose.x
        guess_delta_y = initial_guess.y - previous_pose.y
        guess_delta_theta = normalize_angle(initial_guess.theta - previous_pose.theta)
        
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
            est_delta_theta = normalize_angle(estimated_pose.theta - previous_pose.theta)
            
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
            corr_delta_theta = normalize_angle(corrected_pose.theta - previous_pose.theta)
            
            print(f"CORRECTED POSE:   x={corrected_pose.x:.4f}, y={corrected_pose.y:.4f}, θ={corrected_pose.theta:.4f}")
            print(f"CORRECTED DELTA:  Δx={corr_delta_x:.4f}, Δy={corr_delta_y:.4f}, Δθ={corr_delta_theta:.4f}")
        
        print(f"{'='*100}\n")
    
    def visualize_map_and_trajectory(self, save_path=None):
        """
        Visualize the map and trajectory for final assessment
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot the map if we have one
        if self.map:
            # Custom colormap
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                self.map.get_grid_for_display(),
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-self.map.width/2, self.map.width/2, 
                       -self.map.height/2, self.map.height/2]
            )
            
            # Draw map boundaries
            ax.axhline(y=-self.map.height/2, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=self.map.height/2, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=-self.map.width/2, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.map.width/2, color='red', linestyle='--', alpha=0.5)
        
        # Plot odometry trajectory
        if self.odometry_trajectory:
            odom_x = [pose.x for pose in self.odometry_trajectory]
            odom_y = [pose.y for pose in self.odometry_trajectory]
            ax.plot(odom_x, odom_y, 'r--', linewidth=1.5, alpha=0.7, label='Odometry')
        
        # Plot matched trajectory
        if self.trajectory:
            traj_x = [pose.x for pose in self.trajectory]
            traj_y = [pose.y for pose in self.trajectory]
            ax.plot(traj_x, traj_y, 'b-', linewidth=2, label='Improved SLAM')
            
            # Add markers for start and end
            ax.scatter(traj_x[0], traj_y[0], c='green', s=150, marker='*', label='Start')
            ax.scatter(traj_x[-1], traj_y[-1], c='blue', s=150, marker='*', label='End')
        
        # Plot loop closures if detected
        if self.enable_loop_closure and self.loop_detector.loop_closures:
            for lc in self.loop_detector.loop_closures:
                if lc['keyframe_idx'] < len(self.trajectory) and lc['current_idx'] < len(self.trajectory):
                    # Get keyframe and current poses
                    keyframe_pose = self.trajectory[lc['keyframe_idx']]
                    current_pose = self.trajectory[lc['current_idx']]
                    
                    # Draw line connecting the loop closure points
                    ax.plot(
                        [keyframe_pose.x, current_pose.x],
                        [keyframe_pose.y, current_pose.y],
                        'g-', linewidth=2, alpha=0.7
                    )
                    
                    # Mark both points
                    ax.scatter(keyframe_pose.x, keyframe_pose.y, c='orange', s=80, marker='o')
                    ax.scatter(current_pose.x, current_pose.y, c='orange', s=80, marker='o')
            
            # Add to legend
            ax.scatter([], [], c='orange', s=80, marker='o', label='Loop Closures')
        
        # Add grid, labels and title
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Improved SLAM Map and Trajectory')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add metadata
        metadata_text = []
        if self.enable_robust_rotation:
            metadata_text.append("Robust Rotation Handling: Enabled")
        if self.enable_loop_closure:
            metadata_text.append(f"Loop Closure Detection: Enabled ({len(self.loop_detector.loop_closures)} detected)")
        if metadata_text:
            plt.figtext(0.02, 0.02, '\n'.join(metadata_text), fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.tight_layout()
        return fig

#########################################################################
# VISUALIZATION AND MAIN FUNCTIONS
#########################################################################

def visualize_lidar_data_realtime(file_path, max_entries=200, show_occupancy_grid=True, 
                             grid_resolution=0.05, save_grid=True, save_format='all',
                             enable_scan_matching=True, enable_robust_rotation=True,
                             enable_loop_closure=True):
    """
    Main function to visualize LiDAR data with improved SLAM algorithms
    
    Args:
        file_path: Path to the LiDAR data file
        max_entries: Maximum number of entries to read from the file
        show_occupancy_grid: Whether to show the occupancy grid visualization
        grid_resolution: Resolution of the occupancy grid in meters
        save_grid: Whether to save the final map to a file
        save_format: Format to save the grid ('png', 'npy', 'csv', or 'all')
        enable_scan_matching: Whether to use scan matching localization
        enable_robust_rotation: Whether to enable robust rotation handling
        enable_loop_closure: Whether to enable loop closure detection
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
    
    print(f"\nInitializing Improved SLAM system...")
    print(f"  Occupancy grid mapping: {'Enabled' if show_occupancy_grid else 'Disabled'}")
    if show_occupancy_grid:
        print(f"  Grid resolution: {grid_resolution}m")
    print(f"  Scan matching: {'Enabled' if enable_scan_matching else 'Disabled'}")
    print(f"  Robust rotation handling: {'Enabled' if enable_robust_rotation else 'Disabled'}")
    print(f"  Loop closure detection: {'Enabled' if enable_loop_closure else 'Disabled'}")
    
    # Assuming the LiDAR scan covers 180 degrees (π radians)
    angle_min = -math.pi/2
    angle_max = math.pi/2
    
    # Calculate map size from data
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
    
    # Add padding
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
    occupancy_grid = None
    if show_occupancy_grid:
        occupancy_grid = OccupancyGrid(resolution=grid_resolution, 
                                      width=grid_width, 
                                      height=grid_height)
    
    # Initialize SLAM system
    slam = ImprovedScanMatchingLocalization(occupancy_grid, debug_level=1)
    
    # Configure SLAM system based on parameters
    slam.enable_robust_rotation = enable_robust_rotation
    slam.enable_loop_closure = enable_loop_closure
    
    print(f"\nProcessing data with Improved SLAM...")
    
    # Process all the LiDAR data
    start_time = time.time()
    trajectory = slam.processSensorData(
        parsed_data_list,
        angle_min=angle_min,
        angle_max=angle_max,
        flip_x=False,
        flip_y=False,
        reverse_scan=True,
        flip_theta=False
    )
    processing_time = time.time() - start_time
    
    print(f"\nSLAM processing completed in {processing_time:.2f} seconds.")
    print(f"Generated trajectory with {len(trajectory)} poses.")
    
    # Create output directory for maps
    maps_dir = "maps"
    if save_grid and show_occupancy_grid and not os.path.exists(maps_dir):
        try:
            os.makedirs(maps_dir)
            print(f"Created directory for maps: {maps_dir}/")
        except Exception as e:
            print(f"Error creating maps directory: {e}")
    
    # Save the final map if requested
    if save_grid and show_occupancy_grid:
        # Generate a timestamp-based filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(maps_dir, f"improved_slam_map_{timestamp}")
        
        # Create path coordinates for saving
        robot_path_x = [pose.x for pose in trajectory]
        robot_path_y = [pose.y for pose in trajectory]
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
        
        print(f"\nImproved SLAM map saved to:")
        for file in saved_files:
            print(f"  - {file}")
    
    # Generate visualization
    print(f"\nGenerating final visualization...")
    
    # Save and display the final visualization
    visualization_path = os.path.join(maps_dir, f"improved_slam_visualization_{timestamp}.png") if save_grid else None
    fig = slam.visualize_map_and_trajectory(save_path=visualization_path)
    
    # Display results
    plt.figure(fig.number)
    plt.show()
    
    return slam, trajectory, occupancy_grid

def main():
    """Main function to parse arguments and run the improved SLAM system"""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Improved SLAM System with Robust Rotation and Loop Closure')
    
    # Add arguments
    parser.add_argument('--file', type=str, default="./lidar_slam/dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max_entries', type=int, default=100,
                       help='Maximum number of entries to read from the file')
    parser.add_argument('--grid', action='store_true', default=True,
                       help='Enable occupancy grid mapping')
    parser.add_argument('--resolution', type=float, default=0.05,
                       help='Resolution of the occupancy grid in meters')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save the final map')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'npy', 'csv', 'all'],
                       help='Format to save the grid')
    parser.add_argument('--scan_matching', action='store_true', default=True,
                       help='Enable scan matching localization')
    parser.add_argument('--robust_rotation', action='store_true', default=True,
                       help='Enable robust rotation handling')
    parser.add_argument('--loop_closure', action='store_true', default=True,
                       help='Enable loop closure detection')
    parser.add_argument('--debug', type=int, default=1, choices=[0, 1, 2, 3],
                       help='Debug level (0=none, 1=basic, 2=detailed, 3=verbose)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the improved SLAM system
    visualize_lidar_data_realtime(
        file_path=args.file,
        max_entries=args.max_entries,
        show_occupancy_grid=args.grid,
        grid_resolution=args.resolution,
        save_grid=args.save,
        save_format=args.format,
        enable_scan_matching=args.scan_matching,
        enable_robust_rotation=args.robust_rotation,
        enable_loop_closure=args.loop_closure
    )

if __name__ == "__main__":
    main()