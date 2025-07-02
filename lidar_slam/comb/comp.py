import numpy as np
import matplotlib.pyplot as plt
from motion_compensated_grid import MotionCompensatedOccupancyGrid
from occupancy_grid_class import OccupancyGrid  # Original grid
from lidar_utility_functions import convert_scans_to_cartesian
from lidar_range_utils import read_lidar_data_with_range, print_file_analysis
import math
import time

def compare_motion_compensation(file_path, entry_range=None, max_entries=200, resolution=0.05):
    """
    Compare occupancy grids with and without motion compensation
    Focus on areas where the robot turns to see the difference in wall shapes
    """
    print(f"Comparing motion compensation effects with: {file_path}")
    print_file_analysis(file_path)
    
    # Read LiDAR data
    parsed_data_list = read_lidar_data_with_range(file_path, entry_range=entry_range, max_entries=max_entries)
    
    if not parsed_data_list:
        print("No data was read from the file.")
        return
    
    print(f"Loaded {len(parsed_data_list)} LiDAR scans")
    
    # LiDAR scan parameters
    angle_min = -math.pi/2
    angle_max = math.pi/2
    
    # Calculate grid bounds
    all_x_points = []
    all_y_points = []
    for parsed_data in parsed_data_list:
        x_points, y_points = convert_scans_to_cartesian(
            parsed_data['scan_ranges'], angle_min, angle_max, parsed_data['pose'],
            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
        )
        all_x_points.extend(x_points)
        all_y_points.extend(y_points)
    
    x_min, x_max = min(all_x_points), max(all_x_points)
    y_min, y_max = min(all_y_points), max(all_y_points)
    
    x_padding = max(1.0, (x_max - x_min) * 0.2)
    y_padding = max(1.0, (y_max - y_min) * 0.2)
    
    grid_width = max(20, int(math.ceil((x_max - x_min + 2*x_padding))))
    grid_height = max(20, int(math.ceil((y_max - y_min + 2*y_padding))))
    
    # Initialize both grids
    print("Initializing standard occupancy grid...")
    standard_grid = OccupancyGrid(
        resolution=resolution,
        initial_width=grid_width,
        initial_height=grid_height,
        expansion_factor=1.5,
        sensor_noise_variance=0.01
    )
    
    print("Initializing motion compensated grid...")
    motion_grid = MotionCompensatedOccupancyGrid(
        resolution=resolution,
        initial_width=grid_width,
        initial_height=grid_height,
        expansion_factor=1.5,
        sensor_noise_variance=0.01
    )
    
    # Process scans and analyze motion
    robot_path_x = []
    robot_path_y = []
    robot_path_theta = []
    turning_points = []
    motion_analysis = []
    
    print("Processing scans with motion analysis...")
    start_time = time.time()
    
    for i, data in enumerate(parsed_data_list):
        if i % 20 == 0:
            print(f"  Processed {i}/{len(parsed_data_list)} scans...")
        
        # Get robot pose and timestamp
        robot_x = data['pose']['x']
        robot_y = data['pose']['y']
        robot_theta = data['pose']['theta']
        timestamp = data['timestamp']
        
        robot_path_x.append(robot_x)
        robot_path_y.append(robot_y)
        robot_path_theta.append(robot_theta)
        
        # Convert scan to Cartesian coordinates
        scan_x, scan_y = convert_scans_to_cartesian(
            data['scan_ranges'], angle_min, angle_max, data['pose'],
            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
        )
        
        # Update both grids
        standard_grid.update_grid(robot_x, robot_y, scan_x, scan_y)
        motion_grid.update_grid_with_motion_compensation(robot_x, robot_y, robot_theta, scan_x, scan_y, timestamp)
        
        # Analyze motion state
        motion_stats = motion_grid.get_motion_statistics()
        motion_analysis.append({
            'index': i,
            'linear_velocity': motion_stats['linear_velocity'],
            'angular_velocity': motion_stats['angular_velocity'],
            'is_turning': motion_stats['is_turning'],
            'is_moving': motion_stats['is_moving'],
            'position': (robot_x, robot_y)
        })
        
        # Mark turning points
        if motion_stats['is_turning']:
            turning_points.append((robot_x, robot_y, i))
    
    processing_time = time.time() - start_time
    print(f"Processing complete! ({processing_time:.2f} seconds)")
    
    # Analyze results
    print("\nAnalyzing motion and mapping results...")
    
    # Count turning frames
    turning_frames = [m for m in motion_analysis if m['is_turning']]
    moving_frames = [m for m in motion_analysis if m['is_moving']]
    
    print(f"Total frames: {len(motion_analysis)}")
    print(f"Turning frames: {len(turning_frames)} ({len(turning_frames)/len(motion_analysis)*100:.1f}%)")
    print(f"Moving frames: {len(moving_frames)} ({len(moving_frames)/len(motion_analysis)*100:.1f}%)")
    print(f"Turning points detected: {len(turning_points)}")
    
    # Calculate motion statistics
    max_angular_velocity = max([abs(m['angular_velocity']) for m in motion_analysis])
    max_linear_velocity = max([m['linear_velocity'] for m in motion_analysis])
    avg_angular_velocity = np.mean([abs(m['angular_velocity']) for m in motion_analysis])
    avg_linear_velocity = np.mean([m['linear_velocity'] for m in motion_analysis])
    
    print(f"Max angular velocity: {max_angular_velocity:.3f} rad/s ({math.degrees(max_angular_velocity):.1f}°/s)")
    print(f"Max linear velocity: {max_linear_velocity:.3f} m/s")
    print(f"Avg angular velocity: {avg_angular_velocity:.3f} rad/s ({math.degrees(avg_angular_velocity):.1f}°/s)")
    print(f"Avg linear velocity: {avg_linear_velocity:.3f} m/s")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Define extents
    standard_extent = [-standard_grid.origin_x, -standard_grid.origin_x + standard_grid.width,
                      -standard_grid.origin_y, -standard_grid.origin_y + standard_grid.height]
    motion_extent = [-motion_grid.origin_x, -motion_grid.origin_x + motion_grid.width,
                    -motion_grid.origin_y, -motion_grid.origin_y + motion_grid.height]
    
    # Plot 1: Standard grid
    ax1 = plt.subplot(2, 3, 1)
    cmap = plt.matplotlib.colors.ListedColormap(['white', 'lightgray', 'black'])
    bounds = [0, 0.4, 0.6, 1]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    im1 = ax1.imshow(standard_grid.grid, cmap=cmap, norm=norm, origin='lower', extent=standard_extent)
    ax1.plot(robot_path_x, robot_path_y, 'r-', linewidth=1, alpha=0.7, label='Robot Path')
    
    # Mark turning points
    if turning_points:
        turn_x, turn_y, _ = zip(*turning_points)
        ax1.scatter(turn_x, turn_y, c='orange', s=30, alpha=0.8, label='Turning Points')
    
    ax1.set_title('Standard Grid (with rounded walls)')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Motion compensated grid
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(motion_grid.grid, cmap=cmap, norm=norm, origin='lower', extent=motion_extent)
    ax2.plot(robot_path_x, robot_path_y, 'r-', linewidth=1, alpha=0.7, label='Robot Path')
    
    # Mark turning points
    if turning_points:
        ax2.scatter(turn_x, turn_y, c='orange', s=30, alpha=0.8, label='Turning Points')
    
    ax2.set_title('Motion Compensated Grid (straight walls)')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference map
    ax3 = plt.subplot(2, 3, 3)
    
    # Calculate difference (need to handle different grid sizes)
    min_height = min(standard_grid.grid_height, motion_grid.grid_height)
    min_width = min(standard_grid.grid_width, motion_grid.grid_width)
    
    diff_grid = motion_grid.grid[:min_height, :min_width] - standard_grid.grid[:min_height, :min_width]
    
    im3 = ax3.imshow(diff_grid, cmap='RdBu', vmin=-0.5, vmax=0.5, origin='lower')
    ax3.set_title('Difference (Motion - Standard)')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    plt.colorbar(im3, ax=ax3, label='Probability Difference')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Motion velocity over time
    ax4 = plt.subplot(2, 3, 4)
    frame_indices = [m['index'] for m in motion_analysis]
    angular_velocities = [m['angular_velocity'] for m in motion_analysis]
    linear_velocities = [m['linear_velocity'] for m in motion_analysis]
    
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(frame_indices, [abs(av) for av in angular_velocities], 'b-', label='Angular Velocity', linewidth=1)
    line2 = ax4_twin.plot(frame_indices, linear_velocities, 'r-', label='Linear Velocity', linewidth=1)
    
    # Mark turning threshold
    ax4.axhline(y=motion_grid.angular_velocity_threshold, color='blue', linestyle='--', alpha=0.5, label='Turn Threshold')
    ax4_twin.axhline(y=motion_grid.linear_velocity_threshold, color='red', linestyle='--', alpha=0.5, label='Motion Threshold')
    
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Angular Velocity (rad/s)', color='blue')
    ax4_twin.set_ylabel('Linear Velocity (m/s)', color='red')
    ax4.set_title('Robot Motion Over Time')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 5: Motion state visualization
    ax5 = plt.subplot(2, 3, 5)
    
    # Color code path by motion state
    path_colors = []
    for m in motion_analysis:
        if m['is_turning'] and m['is_moving']:
            path_colors.append('red')      # Turning while moving
        elif m['is_turning']:
            path_colors.append('orange')   # Turning only
        elif m['is_moving']:
            path_colors.append('blue')     # Moving only
        else:
            path_colors.append('gray')     # Stationary
    
    # Plot path segments with colors
    for i in range(len(robot_path_x) - 1):
        ax5.plot([robot_path_x[i], robot_path_x[i+1]], 
                [robot_path_y[i], robot_path_y[i+1]], 
                color=path_colors[i], linewidth=2, alpha=0.8)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Turning + Moving'),
        Line2D([0], [0], color='orange', lw=2, label='Turning Only'),
        Line2D([0], [0], color='blue', lw=2, label='Moving Only'),
        Line2D([0], [0], color='gray', lw=2, label='Stationary')
    ]
    ax5.legend(handles=legend_elements, loc='upper right')
    
    ax5.set_title('Robot Motion State')
    ax5.set_xlabel('X (meters)')
    ax5.set_ylabel('Y (meters)')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # Plot 6: Statistics comparison
    ax6 = plt.subplot(2, 3, 6)
    
    motion_stats = motion_grid.get_motion_statistics()
    stats_text = f"""
MOTION ANALYSIS RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MOTION STATISTICS:
• Total frames: {len(motion_analysis)}
• Turning frames: {len(turning_frames)} ({len(turning_frames)/len(motion_analysis)*100:.1f}%)
• Moving frames: {len(moving_frames)} ({len(moving_frames)/len(motion_analysis)*100:.1f}%)
• Turning points: {len(turning_points)}

VELOCITY ANALYSIS:
• Max angular velocity: {max_angular_velocity:.3f} rad/s ({math.degrees(max_angular_velocity):.1f}°/s)
• Max linear velocity: {max_linear_velocity:.3f} m/s
• Avg angular velocity: {avg_angular_velocity:.3f} rad/s ({math.degrees(avg_angular_velocity):.1f}°/s)
• Avg linear velocity: {avg_linear_velocity:.3f} m/s

MOTION COMPENSATION STATISTICS:
• Turn compensated updates: {motion_stats['turn_compensated_updates']}
• Motion compensated points: {motion_stats['motion_compensated_points']}
• Compensation rate: {motion_stats['motion_compensated_points']/motion_stats['total_updates']:.1f} points/update

GRID COMPARISON:
• Standard occupied cells: {standard_grid.stats['occupied_cell_count']}
• Motion comp. occupied cells: {motion_grid.stats['occupied_cell_count']}
• Difference: {motion_grid.stats['occupied_cell_count'] - standard_grid.stats['occupied_cell_count']:+d} cells

THRESHOLDS:
• Angular velocity threshold: {motion_grid.angular_velocity_threshold:.3f} rad/s
• Linear velocity threshold: {motion_grid.linear_velocity_threshold:.3f} m/s
• Turn conservatism factor: {motion_grid.turn_conservatism_factor:.1f}

PROCESSING TIME: {processing_time:.2f} seconds
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.suptitle(f'Motion Compensation Analysis - {len(parsed_data_list)} frames', fontsize=16)
    plt.tight_layout()
    
    # Save comparison
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_filename = f"motion_compensation_comparison_{timestamp}.png"
    plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved as: {comparison_filename}")
    
    plt.show()
    
    # Save individual grids
    robot_path_coords = list(zip(robot_path_x, robot_path_y))
    start_pos = (robot_path_x[0], robot_path_y[0])
    end_pos = (robot_path_x[-1], robot_path_y[-1])
    
    standard_grid.save_to_file(f"standard_grid_{timestamp}", format='png',
                              robot_path=robot_path_coords,
                              start_position=start_pos,
                              current_position=end_pos)
    
    motion_grid.save_to_file(f"motion_compensated_grid_{timestamp}", format='png',
                           robot_path=robot_path_coords,
                           start_position=start_pos,
                           current_position=end_pos)
    
    return {
        'standard_grid': standard_grid,
        'motion_grid': motion_grid,
        'motion_analysis': motion_analysis,
        'turning_points': turning_points,
        'processing_time': processing_time
    }

def analyze_wall_straightness(occupancy_grid, robot_path, title="Wall Analysis"):
    """
    Analyze how straight the walls are in the occupancy grid
    """
    # Create binary occupancy grid
    binary_grid = (occupancy_grid.grid > 0.6).astype(np.uint8)
    
    # Find wall edges using edge detection
    from scipy import ndimage
    
    # Edge detection
    edges_x = ndimage.sobel(binary_grid, axis=1)
    edges_y = ndimage.sobel(binary_grid, axis=0)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    
    # Find strong edges
    strong_edges = edges > np.percentile(edges[edges > 0], 75)
    
    # Analyze edge smoothness
    edge_points = np.where(strong_edges)
    
    if len(edge_points[0]) > 0:
        # Calculate local curvature approximation
        curvatures = []
        for i in range(5, len(edge_points[0]) - 5):
            y_coords = edge_points[0][i-5:i+6]
            x_coords = edge_points[1][i-5:i+6]
            
            # Fit a line and calculate deviation
            if len(np.unique(x_coords)) > 1:
                coeffs = np.polyfit(x_coords, y_coords, 1)
                fitted_line = np.polyval(coeffs, x_coords)
                deviation = np.mean(np.abs(y_coords - fitted_line))
                curvatures.append(deviation)
        
        avg_curvature = np.mean(curvatures) if curvatures else 0
        max_curvature = np.max(curvatures) if curvatures else 0
    else:
        avg_curvature = max_curvature = 0
    
    return {
        'avg_curvature': avg_curvature,
        'max_curvature': max_curvature,
        'edge_count': len(edge_points[0]),
        'wall_straightness_score': 1.0 / (1.0 + avg_curvature)  # Higher is straighter
    }

def test_motion_compensation_configurations(file_path, entry_range=None, max_entries=200):
    """
    Test different motion compensation configurations
    """
    print("Testing different motion compensation configurations...")
    
    # Read data once
    parsed_data_list = read_lidar_data_with_range(file_path, entry_range=entry_range, max_entries=max_entries)
    
    if not parsed_data_list:
        return
    
    # Test configurations
    configs = [
        {
            'name': 'No Compensation',
            'enable_motion_compensation': False,
            'enable_scan_deskewing': False,
            'enable_turn_detection': False
        },
        {
            'name': 'Basic Turn Detection',
            'enable_motion_compensation': True,
            'enable_scan_deskewing': False,
            'enable_turn_detection': True,
            'turn_conservatism_factor': 0.5
        },
        {
            'name': 'Scan Deskewing Only',
            'enable_motion_compensation': True,
            'enable_scan_deskewing': True,
            'enable_turn_detection': False
        },
        {
            'name': 'Full Compensation',
            'enable_motion_compensation': True,
            'enable_scan_deskewing': True,
            'enable_turn_detection': True,
            'turn_conservatism_factor': 0.3
        },
        {
            'name': 'Conservative Turns',
            'enable_motion_compensation': True,
            'enable_scan_deskewing': True,
            'enable_turn_detection': True,
            'turn_conservatism_factor': 0.1,
            'angular_velocity_threshold': 0.05
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        
        # Create grid with configuration
        if config['name'] == 'No Compensation':
            # Use standard grid for baseline
            grid = OccupancyGrid(resolution=0.05)
        else:
            grid = MotionCompensatedOccupancyGrid(resolution=0.05)
            # Apply configuration
            for key, value in config.items():
                if hasattr(grid, key):
                    setattr(grid, key, value)
        
        # Process data
        robot_path = []
        for i, data in enumerate(parsed_data_list):
            robot_x = data['pose']['x']
            robot_y = data['pose']['y']
            robot_theta = data['pose']['theta']
            timestamp = data['timestamp']
            
            robot_path.append((robot_x, robot_y))
            
            # Convert scan
            scan_x, scan_y = convert_scans_to_cartesian(
                data['scan_ranges'], -math.pi/2, math.pi/2, data['pose'],
                flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
            )
            
            # Update grid
            if hasattr(grid, 'update_grid_with_motion_compensation'):
                grid.update_grid_with_motion_compensation(robot_x, robot_y, robot_theta, scan_x, scan_y, timestamp)
            else:
                grid.update_grid(robot_x, robot_y, scan_x, scan_y)
        
        # Analyze wall straightness
        wall_analysis = analyze_wall_straightness(grid, robot_path, config['name'])
        
        results.append({
            'config': config,
            'grid': grid,
            'wall_analysis': wall_analysis,
            'robot_path': robot_path
        })
    
    # Display comparison
    print("\n" + "="*80)
    print("MOTION COMPENSATION CONFIGURATION COMPARISON")
    print("="*80)
    print(f"{'Configuration':<20} {'Wall Straightness':<18} {'Avg Curvature':<15} {'Occupied Cells':<15}")
    print("-" * 80)
    
    for result in results:
        config_name = result['config']['name']
        straightness = result['wall_analysis']['wall_straightness_score']
        curvature = result['wall_analysis']['avg_curvature']
        
        if hasattr(result['grid'], 'stats'):
            occupied_cells = result['grid'].stats['occupied_cell_count']
        else:
            # For standard grid, count manually
            occupied_cells = np.sum(result['grid'].grid > 0.6)
        
        print(f"{config_name:<20} {straightness:<18.3f} {curvature:<15.3f} {occupied_cells:<15}")
    
    return results

# Example usage
if __name__ == "__main__":
    file_path = "../dataset/raw_data/laser_data_synchronized_basement_loop_fast_speed_processed_reduced180.clf"
    
    # Example 1: Compare motion compensation for a specific range with turns
    print("=== Motion Compensation Comparison ===")
    comparison_results = compare_motion_compensation(
        file_path=file_path,
        entry_range=(0, 6000),  # Range where robot turns
        resolution=0.05
    )
    
    # Example 2: Test different configurations
    # print("\n=== Testing Different Motion Compensation Configurations ===")
    # config_results = test_motion_compensation_configurations(
    #     file_path=file_path,
    #     entry_range=(0, 2400),
    #     max_entries=150
    # )