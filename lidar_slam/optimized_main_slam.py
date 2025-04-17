#!/usr/bin/env python3
"""
Performance-Optimized LiDAR SLAM Framework - Main Entry Point

This script provides an optimized entry point for the LiDAR SLAM framework,
incorporating efficient scan matching and pose estimation for real-time mapping.
"""

import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

from lidar_processing import data_parser, scan_converter
from mapping import occupancy_grid
from visualization import lidar_visualizer
from utils import file_utils

# Import the optimized SLAM components
from lidar_processing.optimized_scan_matcher import OptimizedScanMatcher, MatchingAlgorithm
from pose_estimation.pose_estimator import PoseEstimator, PoseEstimationMethod, Pose2D
from optimized_slam_coordinator import OptimizedSLAMCoordinator, SLAMMode

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Performance-Optimized LiDAR SLAM System')
    
    parser.add_argument('--file', '-f', type=str, 
                        default="./dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                        help='Path to the LiDAR data file')
    
    parser.add_argument('--max-entries', '-m', type=int, default=1000,
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
    
    # SLAM parameters
    parser.add_argument('--slam-mode', choices=['odometry_only', 'scan_matching', 'full_slam'], 
                       default='scan_matching', help='SLAM operation mode')
    
    parser.add_argument('--matching-algorithm', choices=['icp', 'fast_icp', 'psm'], 
                       default='fast_icp', help='Scan matching algorithm to use')
    
    parser.add_argument('--pose-method', choices=['odometry_only', 'scan_match_only', 
                                               'ekf_fusion', 'weighted_average'], 
                       default='ekf_fusion', help='Pose estimation method')
    
    # Performance optimization parameters
    parser.add_argument('--scan-match-interval', type=int, default=5,
                       help='Process every Nth scan for matching (higher = faster)')
    
    parser.add_argument('--downsample-factor', type=int, default=3,
                       help='Downsample scan points by factor (higher = faster)')
    
    parser.add_argument('--max-iterations', type=int, default=15,
                       help='Maximum iterations for scan matching')
    
    parser.add_argument('--max-match-distance', type=float, default=0.5,
                       help='Maximum correspondence distance for matching (meters)')
    
    parser.add_argument('--visualization-interval', type=int, default=50,
                       help='Milliseconds between visualization frames (higher = faster)')
    
    return parser.parse_args()

def visualize_lidar_slam(args):
    """
    Main function to visualize LiDAR data with optimized SLAM
    
    Args:
        args: Command line arguments
    """
    print(f"Reading LiDAR data from: {args.file}")
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist.")
        return
    
    # Read the data from file
    print(f"Reading data file (max {args.max_entries} entries)...")
    start_time = time.time()
    parsed_data_list = data_parser.read_lidar_data_from_file(args.file, args.max_entries)
    read_time = time.time() - start_time
    
    if not parsed_data_list:
        print("No data was read from the file.")
        return
    
    print(f"Read {len(parsed_data_list)} entries in {read_time:.2f} seconds")
    
    # Display data summary
    data_summary = data_parser.get_data_summary(parsed_data_list)
    
    print(f"\nData Summary:")
    print(f"  Number of entries: {data_summary['count']}")
    print(f"  Robot ID: {data_summary['robot_id']}")
    print(f"  Data duration: {data_summary['duration']:.2f} seconds")
    
    # Configure scan matching and pose estimation methods
    slam_mode = SLAMMode(args.slam_mode)
    matching_algorithm = MatchingAlgorithm(args.matching_algorithm)
    pose_method = PoseEstimationMethod(args.pose_method)
    
    # Initialize the SLAM coordinator
    print(f"\nInitializing optimized SLAM system...")
    print(f"  SLAM Mode: {slam_mode.value}")
    print(f"  Scan Matching Algorithm: {matching_algorithm.value}")
    print(f"  Pose Estimation Method: {pose_method.value}")
    print(f"  Performance Settings:")
    print(f"    Scan Match Interval: {args.scan_match_interval}")
    print(f"    Downsample Factor: {args.downsample_factor}")
    print(f"    Max Iterations: {args.max_iterations}")
    print(f"    Max Match Distance: {args.max_match_distance}")
    
    coordinator = OptimizedSLAMCoordinator(mode=slam_mode, grid_resolution=args.resolution)
    
    # Configure scan parameters
    scan_config = {
        'flip_x': args.flip_x,
        'flip_y': args.flip_y,
        'reverse_scan': args.reverse_scan,
        'flip_theta': args.flip_theta
    }
    coordinator.set_scan_config(scan_config)
    
    # Configure performance parameters
    coordinator.set_scan_match_interval(args.scan_match_interval)
    coordinator.set_scan_matcher_params(
        downsample_factor=args.downsample_factor,
        max_iterations=args.max_iterations,
        max_distance=args.max_match_distance
    )
    
    # Process the dataset
    print("\nProcessing dataset with optimized SLAM...")
    processed_data_list = coordinator.process_dataset(parsed_data_list)
    
    # Get the updated occupancy grid
    grid_instance = coordinator.get_occupancy_grid()
    
    # Create output directory for maps if needed
    if args.save and args.grid:
        file_utils.ensure_directory_exists(args.output_dir)
    
    # Clear any existing plots to avoid multiple windows
    plt.close('all')
    
    # Create and run the visualizer with the processed data
    print("\nStarting visualization with SLAM-corrected poses.")
    print("Left-click: Zoom in | Right-click: Zoom out | Middle-click: Reset zoom")
    
    # Use the visualizer with the processed data and occupancy grid
    viz = lidar_visualizer.LiDARVisualizer(
        processed_data_list,
        occupancy_grid=grid_instance
    )
    
    # Configure the visualizer
    viz.config = scan_config
    
    # Start the visualization
    viz.show(
        interval=args.visualization_interval,  # milliseconds between frames
        save_path=args.output_dir if args.save and args.grid else None
    )
    
    # Return results for further analysis
    return grid_instance, coordinator

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Start timer
    start_time = time.time()
    
    # Run the SLAM system
    grid, coordinator = visualize_lidar_slam(args)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    
    # Final message after visualization is closed
    print("\nVisualization closed.")
    print(f"Total runtime: {total_time:.2f} seconds")
    
    if args.grid and args.save:
        print(f"Occupancy grid maps have been saved to '{args.output_dir}' directory.")
    
    # Print performance statistics
    print("\nPerformance Statistics:")
    stats = coordinator.get_performance_stats()
    print(f"  Total scans processed: {stats['total_scans']}")
    print(f"  Scan matches performed: {stats['scan_matches']}")
    print(f"  Successful matches: {stats['successful_matches']} "
          f"({stats['successful_matches']/stats['scan_matches']*100:.1f}% success rate)")
    print(f"  Average processing time per scan: {stats['avg_processing_time']*1000:.2f} ms")
    print(f"  Average matching time: {stats['avg_matching_time']*1000:.2f} ms")
    print(f"  Max processing time: {stats['max_processing_time']*1000:.2f} ms")
    print(f"  Max matching time: {stats['max_matching_time']*1000:.2f} ms")
    
    return grid, coordinator

if __name__ == "__main__":
    # If run as a script, execute the main function
    main()