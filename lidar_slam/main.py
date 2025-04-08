#!/usr/bin/env python3
"""
LiDAR SLAM Framework - Main Entry Point

This script provides the main entry point for the LiDAR SLAM framework,
allowing for visualization and mapping using LiDAR data.

When run directly, it behaves the same as the original test.py script.
Components can also be imported and used independently in other modules.
"""

import os
import sys
import argparse
import time
import matplotlib.pyplot as plt

from lidar_processing import data_parser, scan_converter
from mapping import occupancy_grid
from visualization import lidar_visualizer
from utils import file_utils

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LiDAR Data Visualization and Mapping')
    
    parser.add_argument('--file', '-f', type=str, 
                        default="./dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                        help='Path to the LiDAR data file')
    
    parser.add_argument('--max-entries', '-m', type=int, default=500,
                        help='Maximum number of entries to read from the file')
    
    parser.add_argument('--grid', '-g', action='store_true', default=True,
                        help='Enable occupancy grid mapping')
    
    parser.add_argument('--no-grid', dest='grid', action='store_false',
                        help='Disable occupancy grid mapping')
    
    parser.add_argument('--resolution', '-r', type=float, default=0.05,
                        help='Grid resolution in meters (smaller = more detail but slower)')
    
    parser.add_argument('--save', '-s', action='store_true', default=True,
                        help='Save the final occupancy grid map')
    
    parser.add_argument('--no-save', dest='save', action='store_false',
                        help='Do not save the final occupancy grid map')
    
    parser.add_argument('--format', choices=['png', 'npy', 'csv', 'all'], default='png',
                        help='Format to save the grid (png, npy, csv, or all)')
    
    parser.add_argument('--output-dir', '-o', type=str, default='maps',
                        help='Directory to save output maps')
    
    parser.add_argument('--flip-x', action='store_true', default=False,
                        help='Flip the x-axis of the LiDAR scan')
    
    parser.add_argument('--flip-y', action='store_true', default=False,
                        help='Flip the y-axis of the LiDAR scan')
    
    parser.add_argument('--reverse-scan', action='store_true', default=True,
                        help='Reverse the scan direction of the LiDAR')
    
    parser.add_argument('--no-reverse-scan', dest='reverse_scan', action='store_false',
                        help='Do not reverse the scan direction of the LiDAR')
    
    parser.add_argument('--flip-theta', action='store_true', default=False,
                        help='Negate the orientation angle of the robot')
    
    return parser.parse_args()

def visualize_lidar_data_realtime(args):
    """
    Main function to visualize LiDAR data in real-time with occupancy grid mapping
    
    Args:
        args: Command line arguments
    """
    print(f"Reading LiDAR data from: {args.file}")
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist.")
        return
    
    # Read the data from file
    parsed_data_list = data_parser.read_lidar_data_from_file(args.file, args.max_entries)
    
    if not parsed_data_list:
        print("No data was read from the file.")
        return
    
    # Display data summary
    data_summary = data_parser.get_data_summary(parsed_data_list)
    
    print(f"\nData Summary:")
    print(f"  Number of entries: {data_summary['count']}")
    print(f"  Robot ID: {data_summary['robot_id']}")
    print(f"  Data duration: {data_summary['duration']:.2f} seconds")
    
    # Initialize occupancy grid if enabled
    grid_instance = None
    if args.grid:
        print(f"  Starting visualization with occupancy grid mapping (resolution: {args.resolution}m)...")
        if args.save:
            print(f"  The final occupancy grid will be saved in '{args.format}' format")
        
        # Find scan boundaries to determine grid size
        all_x_points = []
        all_y_points = []
        angle_min = -3.14159/2  # π/2
        angle_max = 3.14159/2   # π/2
        
        # Sample every 10th entry to calculate grid boundaries efficiently
        sample_step = max(1, len(parsed_data_list) // 10)
        for i in range(0, len(parsed_data_list), sample_step):
            parsed_data = parsed_data_list[i]
            x_points, y_points = scan_converter.convert_scans_to_cartesian(
                parsed_data['scan_ranges'], 
                angle_min, 
                angle_max, 
                parsed_data['pose'],
                flip_x=args.flip_x, 
                flip_y=args.flip_y, 
                reverse_scan=args.reverse_scan, 
                flip_theta=args.flip_theta
            )
            all_x_points.extend(x_points)
            all_y_points.extend(y_points)
        
        # Calculate grid dimensions based on data
        x_min, x_max, y_min, y_max = scan_converter.find_scan_boundaries(
            all_x_points, all_y_points, padding_percentage=0.3)
        
        grid_width = max(20, int(max(abs(x_min), abs(x_max)) * 2.5))
        grid_height = max(20, int(max(abs(y_min), abs(y_max)) * 2.5))
        
        # Make grid size a multiple of 2 for cleaner dimensions
        grid_width = (grid_width // 2) * 2
        grid_height = (grid_height // 2) * 2
        
        # Initialize the occupancy grid
        grid_instance = occupancy_grid.OccupancyGrid(
            resolution=args.resolution,
            width=grid_width,
            height=grid_height
        )
    else:
        print(f"  Starting visualization without occupancy grid mapping...")
    
    # Create output directory for maps if needed
    if args.save and args.grid:
        file_utils.ensure_directory_exists(args.output_dir)
    
    # Clear any existing plots to avoid multiple windows
    plt.close('all')
    
    # Create and run the visualizer
    viz = lidar_visualizer.LiDARVisualizer(
        parsed_data_list,
        occupancy_grid=grid_instance
    )
    
    # Configure the visualizer
    viz.config = {
        'flip_x': args.flip_x,
        'flip_y': args.flip_y,
        'reverse_scan': args.reverse_scan,
        'flip_theta': args.flip_theta
    }
    
    # Start the visualization
    print("\nStarting visualization. Close the window to exit.")
    print("Left-click: Zoom in | Right-click: Zoom out | Middle-click: Reset zoom")
    
    viz.show(
        interval=50,  # milliseconds between frames
        save_path=args.output_dir if args.save and args.grid else None
    )
    
    return grid_instance

def main():
    """Main entry point"""
    args = parse_arguments()
    grid = visualize_lidar_data_realtime(args)
    
    # Final message after visualization is closed
    print("\nVisualization closed.")
    
    if args.grid and args.save:
        print(f"Occupancy grid maps have been saved to '{args.output_dir}' directory.")
        print(f"You can use the saved maps for further processing or analysis.")
    
    return grid

if __name__ == "__main__":
    # If run as a script, execute the main function
    main()