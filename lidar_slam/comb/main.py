import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import math
import os
import time
from matplotlib.widgets import Button
import sys

# Import our utility functions and classes
from lidar_utility_functions import parse_lidar_data, convert_scans_to_cartesian, read_lidar_data_from_file
from occupancy_grid_class import OccupancyGrid
from ScanMatcher import ImprovedScanMatchingLocalization

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
    
    # Initialize occupancy grid with the enhanced version
    if show_occupancy_grid:
        # Use enhanced OccupancyGrid with better initial dimensions
        occupancy_grid = OccupancyGrid(
            resolution=grid_resolution, 
            initial_width=grid_width, 
            initial_height=grid_height,
            expansion_factor=1.5  # Allow for 50% expansion when needed
        )
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
        # Custom colormap: white (unknown), black (occupied), light gray (free), blue (enhanced), red (dynamic)
        cmap = colors.ListedColormap(['white', 'lightgray', 'black', 'darkblue', 'red'])
        bounds = [0, 0.4, 0.6, 0.8, 0.9, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Initialize the occupancy grid display with enhanced display 
        grid_img = ax2.imshow(occupancy_grid.get_grid_for_display(enhance=True), 
                             cmap=cmap, norm=norm, 
                             origin='lower', 
                             extent=[occupancy_grid.x_min, occupancy_grid.x_max, 
                                    occupancy_grid.y_min, occupancy_grid.y_max])
        
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
            
            # Save the grid with currently displayed robot path and positions (use enhanced display options)
            occupancy_grid.save_to_file(
                base_filename, 
                format=save_format, 
                include_metadata=True,
                robot_path=displayed_path_coords,
                start_position=start_pos,
                current_position=current_pos,
                enhance_for_display=True,
                show_dynamic=True
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
                localizer.plotMatchOverlay(scan_x, scan_y, current_pose, ax=overlay_ax, show_iterations=True)
                
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
            # Check which update method is available
            if hasattr(occupancy_grid, 'update_grid_with_turn_handling'):
                # Use the turn-aware update method from enhanced OccupancyGrid
                occupancy_grid.update_grid_with_turn_handling(
                    robot_x, robot_y, robot_theta, x_points, y_points, parsed_data['scan_ranges']
                )
                
                # Update the grid image with enhanced display options
                grid_img.set_data(occupancy_grid.get_grid_for_display(enhance=True))
                
            else:
                # Standard update method from original OccupancyGrid
                occupancy_grid.update_grid(robot_x, robot_y, x_points, y_points)
                grid_img.set_data(occupancy_grid.grid)
            
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
            
            # Display grid statistics if available
            if hasattr(occupancy_grid, 'stats'):
                stats_str = (f"Free: {occupancy_grid.stats['free_cell_count']} cells\n"
                            f"Occupied: {occupancy_grid.stats['occupied_cell_count']} cells\n"
                            f"Unknown: {occupancy_grid.stats['unknown_cell_count']} cells")
                grid_settings_text.set_text(stats_str)
            
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
    
    # For dual-plot layout with buttons, use a more flexible approach
    if show_occupancy_grid:
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, wspace=0.1)
    else:
        # For single plot, tight_layout works fine
        plt.tight_layout()
    plt.show()
    
    # Note: The save functionality is now handled by the Save Map button
    # If you still want to automatically save at the end, you can use:
    if save_grid and show_occupancy_grid:
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Generate a timestamp-based filename for the final map
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(save_path, f"final_map_{timestamp}")
        
        # Create final path coordinates for saving
        final_path_coords = list(zip(robot_path_x, robot_path_y))
        
        # Save the final grid state with enhanced display options
        saved_files = occupancy_grid.save_to_file(
            base_filename, 
            format=save_format, 
            include_metadata=True,
            robot_path=final_path_coords,
            start_position=(robot_path_x[0], robot_path_y[0]) if len(robot_path_x) > 0 else None,
            current_position=(robot_path_x[-1], robot_path_y[-1]) if len(robot_path_x) > 0 else None,
            enhance_for_display=True,
            show_dynamic=True
        )
        
        print(f"\nFinal occupancy grid map saved to:")
        for file in saved_files:
            print(f"  - {file}")
    
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
        if enable_scan_matching:
            print(f"  Scan matching is ENABLED - using ICP with adaptive parameters")
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
    
    # Start the animation with appropriate orientation settings
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
        enable_scan_matching=enable_scan_matching # Whether to use scan matching
    )

def main():
    """
    Main function to parse arguments and run the visualization
    """
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='LiDAR Visualization and Localization')
    
    # Add arguments
    parser.add_argument('--file', type=str, default="../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max_entries', type=int, default=100,
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