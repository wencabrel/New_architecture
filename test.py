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
    
    def update_grid(self, robot_x, robot_y, scan_x, scan_y):
        """
        Update the occupancy grid with a laser scan
        
        Args:
            robot_x: Robot's x position in world coordinates
            robot_y: Robot's y position in world coordinates
            scan_x: List of scan x points in world coordinates
            scan_y: List of scan y points in world coordinates
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
                # Ensure we're within grid bounds
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


def animate_lidar_data(parsed_data_list, flip_x=False, flip_y=True, reverse_scan=True, flip_theta=False, 
                      show_occupancy_grid=True, grid_resolution=0.05, save_grid=False,
                      save_format='png', save_path='maps/'):
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
    
def animate_lidar_data(parsed_data_list, flip_x=False, flip_y=True, reverse_scan=True, flip_theta=False, 
                      show_occupancy_grid=True, grid_resolution=0.05, save_grid=False,
                      save_format='png', save_path='maps/'):
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
            
        follow_button.on_clicked(toggle_follow)
        save_button.on_clicked(save_current_map)
        
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
            
            return (scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, settings_text, 
                   grid_img, grid_path_line, grid_start_point, grid_current_pos, grid_timestamp_text, grid_robot_id_text, 
                   grid_pose_text, grid_settings_text)
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
        theta = parsed_data['pose']['theta']
        if flip_theta:
            theta = -theta
        
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
            # Update the grid with current scan
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
            grid_pose_text.set_text(pose_str)
            
            return (scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, 
                   settings_text, arrow, grid_img, grid_path_line, grid_current_pos, grid_timestamp_text, 
                   grid_robot_id_text, grid_pose_text, grid_settings_text)
        else:
            return scatter, robot_pos, path_line, timestamp_text, robot_id_text, pose_text, settings_text, arrow
    
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
                             grid_resolution=0.05, save_grid=True, save_format='all'):
    """
    Main function to visualize LiDAR data in real-time with occupancy grid mapping
    
    Args:
        file_path: Path to the LiDAR data file
        max_entries: Maximum number of entries to read from the file
        show_occupancy_grid: Whether to show the occupancy grid visualization
        grid_resolution: Resolution of the occupancy grid in meters (smaller = more detail but slower)
        save_grid: Whether to save the final occupancy grid map to a file
        save_format: Format to save the grid ('png', 'npy', 'csv', or 'all')
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
        flip_y=False,           # Whether to flip the y-axis (common fix)
        reverse_scan=True,     # Whether to reverse the scan direction (common fix) 
        flip_theta=False,      # Whether to negate the orientation angle
        show_occupancy_grid=show_occupancy_grid,  # Whether to show occupancy grid
        grid_resolution=grid_resolution,          # Resolution of the grid in meters
        save_grid=save_grid,                      # Whether to save the final grid
        save_format=save_format,                  # Format to save the grid
        save_path=maps_dir                        # Directory to save the grid
    )

# Main execution
if __name__ == "__main__":
    # File path to read LiDAR data from
    file_path = "./DataSet/RawData/raw_data_zjnu20_21_3F.clf"
    
    # Run the visualization with occupancy grid mapping
    visualize_lidar_data_realtime(
        file_path, 
        max_entries=12430,
        show_occupancy_grid=True,         # Enable occupancy grid mapping
        grid_resolution=0.05,             # Grid resolution in meters (5cm per cell)
        save_grid=True,                   # Save the final occupancy grid map
        save_format='png'                 # Save in all available formats (png, npy, csv)
    )