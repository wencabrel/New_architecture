import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import os
import time
from scipy.ndimage import binary_dilation, binary_erosion
from collections import deque

class MotionCompensatedOccupancyGrid:
    """Enhanced occupancy grid with motion compensation to fix rounded walls during turns"""
    
    def __init__(self, resolution=0.05, initial_width=10, initial_height=10, 
                 expansion_factor=1.5, sensor_noise_variance=0.01):
        """
        Initialize an occupancy grid with motion compensation
        """
        self.resolution = resolution
        self.expansion_factor = expansion_factor
        self.sensor_noise_variance = sensor_noise_variance
        
        # Calculate initial grid dimensions
        self.grid_width = int(initial_width / resolution)
        self.grid_height = int(initial_height / resolution)
        
        # Initialize grid
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5
        
        # Track dimensions
        self.width = initial_width
        self.height = initial_height
        self.origin_x = initial_width / 2
        self.origin_y = initial_height / 2
        self.x_min = -self.origin_x
        self.x_max = self.origin_x
        self.y_min = -self.origin_y
        self.y_max = self.origin_y
        
        # Log odds grid
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))
        self.log_odds_occupied = np.log(0.75/0.25)
        self.log_odds_free = np.log(0.25/0.75)
        self.log_odds_min = np.log(0.01/0.99)
        self.log_odds_max = np.log(0.99/0.01)
        
        # MOTION COMPENSATION parameters
        self.enable_motion_compensation = True
        self.enable_turn_detection = True
        self.enable_velocity_estimation = True
        self.enable_scan_deskewing = True
        
        # Turn detection
        self.angular_velocity_threshold = 0.1  # rad/s - threshold for detecting turns
        self.linear_velocity_threshold = 0.05  # m/s - threshold for detecting motion
        self.turn_conservatism_factor = 0.01   # Reduce evidence during turns
        
        # Pose history for motion estimation
        self.pose_history = deque(maxlen=10)  # Keep last 10 poses
        self.timestamp_history = deque(maxlen=10)
        
        # LiDAR scan parameters (assuming typical values)
        self.lidar_scan_time = 0.1  # 100ms for full scan (10Hz)
        self.lidar_frequency = 10.0  # Hz
        self.num_scan_points = 180  # Typical number of points in 180° scan
        
        # Motion state tracking
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.is_turning = False
        self.is_moving = False
        
        # Scan point timing (for deskewing)
        self.enable_point_timing = True
        
        # Performance tracking
        self.update_count = 0
        self.last_resize_count = 0
        self.resize_frequency = 10
        
        # Statistics
        self.stats = {
            'free_cell_count': 0,
            'occupied_cell_count': 0,
            'unknown_cell_count': self.grid_width * self.grid_height,
            'updates': 0,
            'resizes': 0,
            'turn_compensated_updates': 0,
            'motion_compensated_points': 0
        }
        
        print("[MotionCompensatedGrid] Initialized with motion compensation")
        print(f"  - Turn detection threshold: {self.angular_velocity_threshold} rad/s")
        print(f"  - Motion compensation: {self.enable_motion_compensation}")
        print(f"  - Scan deskewing: {self.enable_scan_deskewing}")
        print(f"  - LiDAR scan time: {self.lidar_scan_time}s")
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        
        grid_x = np.floor((x_array + self.origin_x) / self.resolution).astype(int)
        grid_y = np.floor((y_array + self.origin_y) / self.resolution).astype(int)
        
        if np.isscalar(x) and np.isscalar(y):
            return int(grid_x.item()), int(grid_y.item())
        else:
            return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        grid_x_array = np.asarray(grid_x)
        grid_y_array = np.asarray(grid_y)
        
        x = grid_x_array * self.resolution + self.resolution/2 - self.origin_x
        y = grid_y_array * self.resolution + self.resolution/2 - self.origin_y
        
        if np.isscalar(grid_x) and np.isscalar(grid_y):
            return x.item(), y.item()
        else:
            return x, y
    
    def _expand_grid(self, direction):
        """Expand the occupancy grid in a specified direction"""
        # Similar implementation as before
        self.stats['resizes'] += 1
        
        old_height, old_width = self.grid.shape
        
        if direction in ['left', 'right']:
            new_width = int(old_width * self.expansion_factor)
            new_height = old_height
            additional_width = new_width - old_width
        else:
            new_height = int(old_height * self.expansion_factor)
            new_width = old_width
            additional_height = new_height - old_height
        
        # Create new grids
        new_grid = np.ones((new_height, new_width)) * 0.5
        new_log_odds = np.zeros((new_height, new_width))
        
        # Calculate placement
        if direction == 'left':
            x_offset = additional_width
            y_offset = 0
            self.x_min -= additional_width * self.resolution
        elif direction == 'right':
            x_offset = 0
            y_offset = 0
            self.x_max += additional_width * self.resolution
        elif direction == 'down':
            x_offset = 0
            y_offset = additional_height
            self.y_min -= additional_height * self.resolution
        else:  # 'up'
            x_offset = 0
            y_offset = 0
            self.y_max += additional_height * self.resolution
        
        # Copy data
        new_grid[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.grid
        new_log_odds[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.log_odds_grid
        
        # Update origin
        if direction == 'left':
            self.origin_x += additional_width * self.resolution
        elif direction == 'down':
            self.origin_y += additional_height * self.resolution
        
        # Update dimensions
        self.grid_width = new_width
        self.grid_height = new_height
        self.width = new_width * self.resolution
        self.height = new_height * self.resolution
        
        # Update grids
        self.grid = new_grid
        self.log_odds_grid = new_log_odds
    
    def check_and_expand_grid(self, world_x, world_y):
        """Check if points are outside current grid bounds and expand if necessary"""
        self.update_count += 1
        
        if self.update_count - self.last_resize_count < self.resize_frequency:
            return False
        
        self.last_resize_count = self.update_count
        
        x_array = np.asarray(world_x)
        y_array = np.asarray(world_y)
        
        min_x, max_x = np.min(x_array), np.max(x_array)
        min_y, max_y = np.min(y_array), np.max(y_array)
        
        padding = 1.0
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        expanded = False
        
        if min_x < self.x_min:
            self._expand_grid('left')
            expanded = True
        if max_x > self.x_max:
            self._expand_grid('right')
            expanded = True
        if min_y < self.y_min:
            self._expand_grid('down')
            expanded = True
        if max_y > self.y_max:
            self._expand_grid('up')
            expanded = True
        
        return expanded
    
    def estimate_velocity(self, current_pose, current_timestamp):
        """
        Estimate linear and angular velocity from pose history
        """
        if len(self.pose_history) < 2:
            return 0.0, 0.0
        
        # Get the most recent pose from history
        prev_pose = self.pose_history[-1]
        prev_timestamp = self.timestamp_history[-1]
        
        # Calculate time difference
        dt = current_timestamp - prev_timestamp
        if dt <= 0:
            return 0.0, 0.0
        
        # Calculate linear velocity
        dx = current_pose['x'] - prev_pose['x']
        dy = current_pose['y'] - prev_pose['y']
        linear_velocity = math.sqrt(dx*dx + dy*dy) / dt
        
        # Calculate angular velocity
        dtheta = current_pose['theta'] - prev_pose['theta']
        # Normalize angle difference to [-π, π]
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
        angular_velocity = dtheta / dt
        
        return linear_velocity, angular_velocity
    
    def detect_motion_state(self, linear_velocity, angular_velocity):
        """
        Detect if robot is turning, moving straight, or stationary
        """
        is_turning = abs(angular_velocity) > self.angular_velocity_threshold
        is_moving = linear_velocity > self.linear_velocity_threshold
        
        return is_turning, is_moving
    
    def deskew_scan_points(self, scan_x, scan_y, robot_pose, linear_velocity, angular_velocity):
        """
        Apply motion compensation to scan points (deskewing)
        
        This corrects for robot motion during the scan acquisition time
        """
        if not self.enable_scan_deskewing:
            return scan_x, scan_y
        
        if abs(linear_velocity) < 0.01 and abs(angular_velocity) < 0.01:
            # Robot not moving significantly, no need to deskew
            return scan_x, scan_y
        
        # Calculate timing for each scan point
        # Assume scan points are acquired sequentially over scan_time
        num_points = len(scan_x)
        
        corrected_x = []
        corrected_y = []
        
        for i in range(num_points):
            # Calculate time offset for this point (relative to scan start)
            point_time_offset = (i / num_points) * self.lidar_scan_time
            
            # Estimate robot pose at the time this point was acquired
            # Simple linear interpolation of motion
            time_fraction = point_time_offset / self.lidar_scan_time
            
            # Estimate position at point acquisition time
            point_robot_x = robot_pose['x'] - linear_velocity * math.cos(robot_pose['theta']) * point_time_offset
            point_robot_y = robot_pose['y'] - linear_velocity * math.sin(robot_pose['theta']) * point_time_offset
            point_robot_theta = robot_pose['theta'] - angular_velocity * point_time_offset
            
            # Calculate the scan point relative to the robot at that time
            # Convert from current world coordinates to robot frame at point time
            dx = scan_x[i] - robot_pose['x']
            dy = scan_y[i] - robot_pose['y']
            
            # Rotate to account for orientation change
            cos_dtheta = math.cos(angular_velocity * point_time_offset)
            sin_dtheta = math.sin(angular_velocity * point_time_offset)
            
            # Apply rotation correction
            dx_corrected = dx * cos_dtheta - dy * sin_dtheta
            dy_corrected = dx * sin_dtheta + dy * cos_dtheta
            
            # Calculate corrected world coordinates
            corrected_x.append(point_robot_x + dx_corrected)
            corrected_y.append(point_robot_y + dy_corrected)
        
        self.stats['motion_compensated_points'] += num_points
        return corrected_x, corrected_y
    
    def update_grid_with_motion_compensation(self, robot_x, robot_y, robot_theta, 
                                            scan_x, scan_y, timestamp=None):
        """
        Update the occupancy grid with motion compensation
        """
        # Create current pose dictionary
        current_pose = {'x': robot_x, 'y': robot_y, 'theta': robot_theta}
        
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time()
        
        # Estimate velocities if we have enough history
        if self.enable_velocity_estimation:
            linear_velocity, angular_velocity = self.estimate_velocity(current_pose, timestamp)
        else:
            linear_velocity = angular_velocity = 0.0
        
        # Update motion state
        self.current_linear_velocity = linear_velocity
        self.current_angular_velocity = angular_velocity
        self.is_turning, self.is_moving = self.detect_motion_state(linear_velocity, angular_velocity)
        
        # Add current pose to history
        self.pose_history.append(current_pose.copy())
        self.timestamp_history.append(timestamp)
        
        # Apply scan deskewing if enabled
        if self.enable_scan_deskewing and (self.is_turning or self.is_moving):
            corrected_scan_x, corrected_scan_y = self.deskew_scan_points(
                scan_x, scan_y, current_pose, linear_velocity, angular_velocity
            )
        else:
            corrected_scan_x, corrected_scan_y = scan_x, scan_y
        
        # Check if we need to expand the grid
        self.check_and_expand_grid(corrected_scan_x, corrected_scan_y)
        
        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        # Create temporary grid for updates
        temp_log_odds_grid = np.zeros_like(self.log_odds_grid)
        
        # Calculate update factors based on motion state
        if self.is_turning:
            # Reduce confidence during turns to prevent wall rounding
            evidence_factor = self.turn_conservatism_factor
            self.stats['turn_compensated_updates'] += 1
        else:
            evidence_factor = 1.0
        
        # Process each corrected scan point
        for i in range(len(corrected_scan_x)):
            scan_point_x, scan_point_y = corrected_scan_x[i], corrected_scan_y[i]
            
            # Convert to grid coordinates
            endpoint_grid_x, endpoint_grid_y = self.world_to_grid(scan_point_x, scan_point_y)
            
            # Skip if outside grid bounds
            if (endpoint_grid_x < 0 or endpoint_grid_x >= self.grid_width or 
                endpoint_grid_y < 0 or endpoint_grid_y >= self.grid_height):
                continue
            
            # Calculate distance for uncertainty modeling
            scan_distance = math.sqrt((robot_x - scan_point_x)**2 + (robot_y - scan_point_y)**2)
            distance_factor = 1.0 + scan_distance * self.sensor_noise_variance
            
            # Apply motion-compensated ray casting
            cells_x, cells_y = self._bresenham_line(robot_grid_x, robot_grid_y, endpoint_grid_x, endpoint_grid_y)
            
            # Mark free space along ray (with motion compensation)
            for j in range(len(cells_x) - 1):  # Exclude endpoint
                cell_x, cell_y = cells_x[j], cells_y[j]
                if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                    free_update = self.log_odds_free * evidence_factor
                    temp_log_odds_grid[cell_y, cell_x] += free_update
            
            # Mark endpoint as occupied (with motion compensation)
            if (0 <= endpoint_grid_x < self.grid_width and 0 <= endpoint_grid_y < self.grid_height):
                occupied_update = (self.log_odds_occupied * evidence_factor) / distance_factor
                temp_log_odds_grid[endpoint_grid_y, endpoint_grid_x] += occupied_update
        
        # Apply updates
        self.log_odds_grid += temp_log_odds_grid
        
        # Clamp values
        self.log_odds_grid = np.clip(self.log_odds_grid, self.log_odds_min, self.log_odds_max)
        
        # Convert to probabilities
        self.grid = 1 - 1 / (1 + np.exp(self.log_odds_grid))
        
        # Update statistics
        self.stats['updates'] += 1
        self.stats['free_cell_count'] = np.sum(self.grid < 0.4)
        self.stats['occupied_cell_count'] = np.sum(self.grid > 0.6)
        self.stats['unknown_cell_count'] = self.grid_width * self.grid_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count'])
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm"""
        x_cells = []
        y_cells = []
        
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            x_cells.append(x)
            y_cells.append(y)
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return x_cells, y_cells
    
    def get_grid_for_display(self, show_motion_state=False):
        """Get grid for display with motion information"""
        if show_motion_state:
            # Create a copy with motion state info
            display_grid = self.grid.copy()
            return display_grid
        else:
            return self.grid.copy()
    
    def get_motion_statistics(self):
        """Get current motion statistics"""
        return {
            'linear_velocity': self.current_linear_velocity,
            'angular_velocity': self.current_angular_velocity,
            'is_turning': self.is_turning,
            'is_moving': self.is_moving,
            'turn_compensated_updates': self.stats['turn_compensated_updates'],
            'motion_compensated_points': self.stats['motion_compensated_points'],
            'total_updates': self.stats['updates']
        }
    
    def save_to_file(self, filename, format='png', include_metadata=True, dpi=300, 
                    robot_path=None, start_position=None, current_position=None,
                    show_motion_info=True):
        """Save the motion compensated occupancy grid"""
        saved_files = []
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"motion_compensated_grid_{timestamp}"
        
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        if format.lower() == 'png' or format.lower() == 'all':
            img_filename = f"{filename}.png"
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Standard colormap
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            extent = [-self.origin_x, -self.origin_x + self.width,
                     -self.origin_y, -self.origin_y + self.height]
            
            im = ax.imshow(self.grid, cmap=cmap, norm=norm, origin='lower', extent=extent)
            ax.set_title('Motion Compensated Occupancy Grid')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.grid(True, alpha=0.3)
            
            # Add robot path and positions
            if robot_path:
                path_x, path_y = zip(*robot_path)
                ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7, label='Robot Path')
            if start_position:
                ax.scatter(start_position[0], start_position[1], c='green', s=100, marker='*', label='Start')
            if current_position:
                ax.scatter(current_position[0], current_position[1], c='blue', s=100, marker='*', label='End')
            
            if robot_path or start_position or current_position:
                ax.legend()
            
            # Add motion compensation info
            if include_metadata:
                motion_stats = self.get_motion_statistics()
                metadata_text = (
                    f"Resolution: {self.resolution:.3f}m/cell\n"
                    f"Motion Compensation: {self.enable_motion_compensation}\n"
                    f"Scan Deskewing: {self.enable_scan_deskewing}\n"
                    f"Turn Detection: {self.enable_turn_detection}\n"
                    f"Angular Velocity Threshold: {self.angular_velocity_threshold:.2f} rad/s\n"
                    f"Turn Compensated Updates: {motion_stats['turn_compensated_updates']}\n"
                    f"Motion Compensated Points: {motion_stats['motion_compensated_points']}\n"
                    f"Total Updates: {motion_stats['total_updates']}\n"
                    f"Current Linear Velocity: {motion_stats['linear_velocity']:.3f} m/s\n"
                    f"Current Angular Velocity: {motion_stats['angular_velocity']:.3f} rad/s"
                )
                plt.figtext(0.02, 0.02, metadata_text, wrap=True, fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(img_filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            saved_files.append(img_filename)
            print(f"Saved motion compensated grid: {img_filename}")
        
        return saved_files
    
    # Compatibility methods
    def update_grid(self, robot_x, robot_y, scan_x, scan_y):
        """Compatibility method (without timestamp)"""
        return self.update_grid_with_motion_compensation(robot_x, robot_y, 0, scan_x, scan_y)
    
    def update_grid_vectorized(self, robot_x, robot_y, scan_x, scan_y):
        """Compatibility method"""
        return self.update_grid_with_motion_compensation(robot_x, robot_y, 0, scan_x, scan_y)
    
    def update_grid_with_turn_handling(self, robot_x, robot_y, robot_theta, scan_x, scan_y, scan_ranges=None):
        """Enhanced compatibility method with full pose"""
        return self.update_grid_with_motion_compensation(robot_x, robot_y, robot_theta, scan_x, scan_y)