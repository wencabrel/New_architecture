# Add this to your main.py file, or create a separate file to test the improved grid

import numpy as np
import matplotlib.pyplot as plt
from improved import ImprovedOccupancyGrid
from lidar_utility_functions import convert_scans_to_cartesian
from lidar_range_utils import read_lidar_data_with_range, print_file_analysis
import math

def test_improved_occupancy_grid(file_path, entry_range=None, max_entries=None):
    """
    Test the improved occupancy grid with persistent occupancy features
    
    Args:
        file_path: Path to LiDAR data file
        entry_range: Tuple of (start, end) indices, or single int for start, or None
        max_entries: Maximum entries to read (for backward compatibility)
    
    Examples:
        # Read entries 100-200
        test_improved_occupancy_grid(file_path, entry_range=(100, 200))
        
        # Read entries 50 to end
        test_improved_occupancy_grid(file_path, entry_range=(50, None))
        
        # Read from entry 100, max 50 entries
        test_improved_occupancy_grid(file_path, entry_range=100, max_entries=50)
    """
    print(f"Testing improved occupancy grid with: {file_path}")
    
    # Analyze the file first
    print_file_analysis(file_path)
    
    # Read LiDAR data with range support
    parsed_data_list = read_lidar_data_with_range(file_path, entry_range=entry_range, max_entries=max_entries)
    
    if not parsed_data_list:
        print("No data was read from the file.")
        return
    
    print(f"Loaded {len(parsed_data_list)} LiDAR scans")
    
    # LiDAR scan parameters
    angle_min = -math.pi/2
    angle_max = math.pi/2
    
    # Find bounds for grid initialization
    all_x_points = []
    all_y_points = []
    for parsed_data in parsed_data_list:
        x_points, y_points = convert_scans_to_cartesian(
            parsed_data['scan_ranges'], angle_min, angle_max, parsed_data['pose'],
            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
        )
        all_x_points.extend(x_points)
        all_y_points.extend(y_points)
    
    # Calculate grid dimensions
    x_min, x_max = min(all_x_points), max(all_x_points)
    y_min, y_max = min(all_y_points), max(all_y_points)
    
    # Add padding
    x_padding = max(1.0, (x_max - x_min) * 0.2)
    y_padding = max(1.0, (y_max - y_min) * 0.2)
    
    grid_width = max(20, int(math.ceil((x_max - x_min + 2*x_padding))))
    grid_height = max(20, int(math.ceil((y_max - y_min + 2*y_padding))))
    
    # Initialize the improved occupancy grid
    improved_grid = ImprovedOccupancyGrid(
        resolution=0.05,
        initial_width=grid_width,
        initial_height=grid_height,
        expansion_factor=1.5,
        sensor_noise_variance=0.01
    )
    
    # Process each scan
    robot_path_x = []
    robot_path_y = []
    
    print("Processing scans...")
    for i, data in enumerate(parsed_data_list):
        if i % 20 == 0:
            print(f"  Processed {i}/{len(parsed_data_list)} scans...")
        
        # Get robot pose
        robot_x = data['pose']['x']
        robot_y = data['pose']['y']
        robot_path_x.append(robot_x)
        robot_path_y.append(robot_y)
        
        # Convert scan to Cartesian coordinates
        scan_x, scan_y = convert_scans_to_cartesian(
            data['scan_ranges'], angle_min, angle_max, data['pose'],
            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
        )
        
        # Update the grid with persistent occupancy
        improved_grid.update_grid_with_persistence(robot_x, robot_y, scan_x, scan_y)
    
    print("Scan processing complete!")
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Final occupancy grid
    ax1 = axes[0, 0]
    extent = [-improved_grid.origin_x, -improved_grid.origin_x + improved_grid.width,
             -improved_grid.origin_y, -improved_grid.origin_y + improved_grid.height]
    
    cmap = plt.matplotlib.colors.ListedColormap(['white', 'lightgray', 'black'])
    bounds = [0, 0.4, 0.6, 1]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    
    im1 = ax1.imshow(improved_grid.grid, cmap=cmap, norm=norm, origin='lower', extent=extent)
    ax1.plot(robot_path_x, robot_path_y, 'r-', linewidth=1, alpha=0.7, label='Robot Path')
    ax1.scatter(robot_path_x[0], robot_path_y[0], c='green', s=100, marker='*', label='Start')
    ax1.scatter(robot_path_x[-1], robot_path_y[-1], c='blue', s=100, marker='*', label='End')
    ax1.set_title('Final Occupancy Grid (Improved)')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Occupancy strength
    ax2 = axes[0, 1]
    strength_display = improved_grid.get_grid_for_display(show_strength=True)
    im2 = ax2.imshow(strength_display, cmap='hot', origin='lower', extent=extent)
    ax2.set_title('Occupancy Strength')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    plt.colorbar(im2, ax=ax2, label='Normalized Strength')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Evidence count
    ax3 = axes[1, 0]
    evidence_display = improved_grid.get_grid_for_display(show_evidence=True)
    im3 = ax3.imshow(evidence_display, cmap='viridis', origin='lower', extent=extent)
    ax3.set_title('Evidence Count (Normalized)')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    plt.colorbar(im3, ax=ax3, label='Normalized Evidence')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Persistent cells
    ax4 = axes[1, 1]
    persistence_display = improved_grid.get_persistence_grid()
    im4 = ax4.imshow(persistence_display, cmap='Reds', origin='lower', extent=extent)
    ax4.set_title('Persistent Occupancy Cells')
    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Y (meters)')
    plt.colorbar(im4, ax=ax4, label='Persistent (1=Yes)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print("\n" + "="*50)
    print("IMPROVED OCCUPANCY GRID STATISTICS")
    print("="*50)
    print(f"Grid dimensions: {improved_grid.grid_width} x {improved_grid.grid_height} cells")
    print(f"Resolution: {improved_grid.resolution:.3f} m/cell")
    print(f"Total updates: {improved_grid.stats['updates']}")
    print(f"Occupied cells: {improved_grid.stats['occupied_cell_count']}")
    print(f"Free cells: {improved_grid.stats['free_cell_count']}")
    print(f"Unknown cells: {improved_grid.stats['unknown_cell_count']}")
    print(f"Persistent cells: {improved_grid.stats['persistent_cells']}")
    print(f"Percentage persistent: {improved_grid.stats['persistent_cells']/improved_grid.stats['occupied_cell_count']*100:.1f}% of occupied cells")
    print("="*50)
    
    # Save the improved grid
    saved_files = improved_grid.save_to_file(
        "improved_occupancy_test",
        format='all',
        robot_path=list(zip(robot_path_x, robot_path_y)),
        start_position=(robot_path_x[0], robot_path_y[0]),
        current_position=(robot_path_x[-1], robot_path_y[-1])
    )
    
    print(f"\nSaved improved grid to {len(saved_files)} files")
    
    plt.show()
    
    return improved_grid

# Example usage with different range options
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf"
    
    # Option 1: Test specific range (entries 100-200)
    print("Testing entries 100-200:")
    improved_grid = test_improved_occupancy_grid(file_path, entry_range=(150, 250))
    
    # Option 2: Test from entry 50 to end of file
    # print("Testing entries 50 to end:")
    # improved_grid = test_improved_occupancy_grid(file_path, entry_range=(50, None))
    
    # Option 3: Test from entry 100, maximum 50 entries
    # print("Testing 50 entries starting from entry 100:")
    # improved_grid = test_improved_occupancy_grid(file_path, entry_range=100, max_entries=50)
    
    # Option 4: Backward compatibility - first 200 entries
    # print("Testing first 200 entries (backward compatibility):")
    # improved_grid = test_improved_occupancy_grid(file_path, max_entries=200)