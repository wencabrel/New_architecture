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
from ScanMatcher import ImprovedScanMatchingLocalization, PoseEstimate
from loop_closure import integrate_with_scan_matcher, process_loop_closure, visualize_loop_closure_results

def visualize_lidar_data_realtime(file_path, max_entries=200, show_occupancy_grid=True, 
                             grid_resolution=0.05, save_grid=True, save_format='all',
                             enable_scan_matching=True, enable_loop_closure=True,
                             rebuild_map=True):
    """
    Main function to visualize LiDAR data in real-time with occupancy grid mapping, 
    scan matching, and loop closure detection
    
    Args:
        file_path: Path to the LiDAR data file
        max_entries: Maximum number of entries to read from the file
        show_occupancy_grid: Whether to show the occupancy grid visualization
        grid_resolution: Resolution of the occupancy grid in meters (smaller = more detail but slower)
        save_grid: Whether to save the final occupancy grid map to a file
        save_format: Format to save the grid ('png', 'npy', 'csv', or 'all')
        enable_scan_matching: Whether to use scan matching localization algorithm
        enable_loop_closure: Whether to enable loop closure detection
        rebuild_map: Whether to rebuild the map after loop closure optimization
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
        if enable_loop_closure and enable_scan_matching:
            print(f"  Loop closure detection is ENABLED - using Scan Context descriptors and pose graph optimization")
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
                                      initial_width=grid_width, 
                                      initial_height=grid_height, 
                                      expansion_factor=1.5, 
                                      sensor_noise_variance=0.01)
    else:
        occupancy_grid = None
    
    # Initialize scan matching localization if enabled
    if enable_scan_matching:
        print(f"  Using improved ICP scan matching algorithm with motion validation")
        localizer = ImprovedScanMatchingLocalization(occupancy_grid, debug_level=1)
        
        # Enable loop closure if requested
        if enable_loop_closure:
            localizer = integrate_with_scan_matcher(localizer, loop_closure_enabled=True)
        
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
        
        # Process loop closures after the entire trajectory has been processed
        if enable_loop_closure:
            print("\nChecking for loop closures in the trajectory...")
            optimized_trajectory, updated_map = process_loop_closure(
                localizer, parsed_data_list, occupancy_grid,
                angle_min=angle_min, angle_max=angle_max,
                flip_x=False, flip_y=False, 
                reverse_scan=True, flip_theta=False
            )
            
            if localizer.loop_closures_detected > 0 and localizer.is_optimized:
                print(f"Found and processed {localizer.loop_closures_detected} loop closures!")
                
                # Update map and trajectory if requested
                if rebuild_map:
                    occupancy_grid = updated_map
                
                # Update trajectory
                robot_path_x = [pose.x for pose in optimized_trajectory]
                robot_path_y = [pose.y for pose in optimized_trajectory]
            else:
                print("No loop closures were detected or processed.")
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
        
        # Add buttons for visualization and saving
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons
        
        # Button positions
        button_width = 0.15
        button_spacing = 0.02
        button_height = 0.04
        button_y = 0.05
        
        # Add a Save Map button
        save_button_ax = plt.axes([0.1, button_y, button_width, button_height])
        save_button = Button(save_button_ax, 'Save Map', color='lightblue', hovercolor='0.8')
        
        # Add a Visualize ICP Process button if scan matching is enabled
        if enable_scan_matching:
            icp_viz_button_ax = plt.axes([0.1 + button_width + button_spacing, button_y, button_width, button_height])
            icp_viz_button = Button(icp_viz_button_ax, 'Visualize ICP', color='lightgreen', hovercolor='0.8')
        
            # Add a Compare Paths button
            compare_button_ax = plt.axes([0.1 + 2 * (button_width + button_spacing), button_y, button_width, button_height])
            compare_button = Button(compare_button_ax, 'Compare Paths', color='lightcoral', hovercolor='0.8')
        
        # Add a Loop Closure Visualization button if loop closure is enabled
        if enable_loop_closure and enable_scan_matching:
            loop_viz_button_ax = plt.axes([0.1 + 3 * (button_width + button_spacing), button_y, button_width, button_height])
            loop_viz_button = Button(loop_viz_button_ax, 'Loop Closures', color='lightsalmon', hovercolor='0.8')
        
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
            saved_files = occupancy_grid.save_to_file(
                base_filename, 
                format=save_format, 
                include_metadata=True,
                robot_path=robot_path_coords,
                start_position=start_pos,
                current_position=current_pos
            )
            
            print(f"\nOccupancy grid map saved to:")
            for file in saved_files:
                print(f"  - {file}")
        
        save_button.on_clicked(save_map)
        
        if enable_scan_matching:
            def visualize_icp_process(event):
                # Create ICP process visualization
                if hasattr(localizer, 'visualizeIcpProcess'):
                    fig = localizer.visualizeIcpProcess()
                    if fig:
                        plt.figure(fig.number)
                        plt.show()
                    else:
                        print("No ICP visualization data available.")
                else:
                    print("ICP visualization is not available.")
            
            icp_viz_button.on_clicked(visualize_icp_process)
            
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
        
        # Add Loop Closure Visualization button handler
        if enable_loop_closure and enable_scan_matching:
            def show_loop_closures(event):
                if hasattr(localizer, 'loop_detector'):
                    if localizer.loop_closures_detected > 0:
                        visualize_loop_closure_results(localizer, occupancy_grid)
                    else:
                        print("No loop closures were detected.")
                else:
                    print("Loop closure detection is not available.")
            
            loop_viz_button.on_clicked(show_loop_closures)
        
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
        
        # Add loop closure status if enabled
        if enable_loop_closure:
            if hasattr(localizer, 'loop_closures_detected') and localizer.loop_closures_detected > 0:
                loop_text = ax.text(0.02, 0.94, f"Loop Closures Detected: {localizer.loop_closures_detected}", 
                                   transform=ax.transAxes, va='top', ha='left', 
                                   color='blue', fontsize=10,
                                   bbox=dict(facecolor='white', alpha=0.7))
            else:
                loop_text = ax.text(0.02, 0.94, "Loop Closure Detection Enabled", 
                                   transform=ax.transAxes, va='top', ha='left', 
                                   color='blue', fontsize=10,
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
            
    # Return key components for further use if needed
    return {
        'occupancy_grid': occupancy_grid,
        'localizer': localizer if enable_scan_matching else None,
        'robot_path': list(zip(robot_path_x, robot_path_y)),
        'scan_data': parsed_data_list
    }


def main():
    """
    Main function to parse arguments and run the visualization
    """
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='LiDAR Visualization, Localization, and Loop Closure')
    
    # Add arguments
    parser.add_argument('--file', type=str, default="../dataset/raw_data/laser_data_synchronized_short_u_turn_fast_processed_reduced180.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max_entries', type=int, default=3000,
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
    parser.add_argument('--loop_closure', action='store_true', default=True,
                       help='Enable loop closure detection')
    parser.add_argument('--rebuild_map', action='store_true', default=True,
                       help='Rebuild map after loop closure optimization')
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
        enable_scan_matching=args.scan_matching,
        enable_loop_closure=args.loop_closure,
        rebuild_map=args.rebuild_map
    )

# Main execution
if __name__ == "__main__":
    main()