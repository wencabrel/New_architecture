#!/usr/bin/env python3
"""
Enhanced LiDAR SLAM Framework - Main Entry Point

This script provides the main entry point for the enhanced LiDAR SLAM framework,
incorporating scan matching and pose estimation for improved mapping.
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

# Import the new SLAM components
from lidar_processing.scan_matcher import ScanMatcher, MatchingAlgorithm
from pose_estimation.pose_estimator import PoseEstimator, PoseEstimationMethod, Pose2D
from slam_coordinator import SLAMCoordinator, SLAMMode

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced LiDAR SLAM Visualization and Mapping')
    
    parser.add_argument('--file', '-f', type=str, 
                        default="./dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                        help='Path to the LiDAR data file')
    
    parser.add_argument('--max-entries', '-m', type=int, default=100,
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
    
    # New arguments for scan matching and pose estimation
    parser.add_argument('--slam-mode', choices=['odometry_only', 'scan_matching', 'full_slam'], 
                       default='scan_matching', help='SLAM operation mode')
    
    parser.add_argument('--matching-algorithm', choices=['icp', 'psm', 'ndt'], 
                       default='icp', help='Scan matching algorithm to use')
    
    parser.add_argument('--pose-method', choices=['odometry_only', 'scan_match_only', 
                                               'ekf_fusion', 'weighted_average'], 
                       default='ekf_fusion', help='Pose estimation method')
    
    parser.add_argument('--scan-match-interval', type=int, default=1,
                       help='Process every Nth scan for matching (1 = all scans)')
    
    parser.add_argument('--show-corrections', action='store_true', default=False,
                       help='Show both raw odometry and corrected paths')
    
    return parser.parse_args()

def visualize_lidar_slam(args):
    """
    Main function to visualize LiDAR data with SLAM processing for improved mapping
    
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
    
    # Configure scan matching and pose estimation methods
    slam_mode = SLAMMode(args.slam_mode)
    matching_algorithm = MatchingAlgorithm(args.matching_algorithm)
    pose_method = PoseEstimationMethod(args.pose_method)
    
    # Initialize the SLAM coordinator
    print(f"\nInitializing SLAM system...")
    print(f"  SLAM Mode: {slam_mode.value}")
    print(f"  Scan Matching Algorithm: {matching_algorithm.value}")
    print(f"  Pose Estimation Method: {pose_method.value}")
    
    coordinator = SLAMCoordinator(mode=slam_mode, grid_resolution=args.resolution)
    
    # Configure scan parameters
    scan_config = {
        'flip_x': args.flip_x,
        'flip_y': args.flip_y,
        'reverse_scan': args.reverse_scan,
        'flip_theta': args.flip_theta
    }
    coordinator.set_scan_config(scan_config)
    coordinator.scan_match_interval = args.scan_match_interval
    
    # Process the dataset
    print("\nProcessing dataset with SLAM...")
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
        interval=50,  # milliseconds between frames
        save_path=args.output_dir if args.save and args.grid else None
    )
    
    return grid_instance, coordinator

def main():
    """Main entry point"""
    args = parse_arguments()
    grid, coordinator = visualize_lidar_slam(args)
    
    # Final message after visualization is closed
    print("\nVisualization closed.")
    
    if args.grid and args.save:
        print(f"Occupancy grid maps have been saved to '{args.output_dir}' directory.")
        print(f"You can use the saved maps for further processing or analysis.")
    
    # Print a summary of the SLAM results
    if coordinator and coordinator.scan_match_results:
        successful_matches = sum(1 for result in coordinator.scan_match_results if result.success)
        total_matches = len(coordinator.scan_match_results)
        success_rate = successful_matches / total_matches * 100 if total_matches > 0 else 0
        
        print(f"\nSLAM Statistics:")
        print(f"  Scan Matches: {successful_matches}/{total_matches} successful ({success_rate:.1f}%)")
        
        if successful_matches > 0:
            avg_fitness = np.mean([r.fitness_score for r in coordinator.scan_match_results if r.success])
            avg_rmse = np.mean([r.inlier_rmse for r in coordinator.scan_match_results if r.success])
            avg_time = np.mean([r.computation_time for r in coordinator.scan_match_results if r.success])
            
            print(f"  Average Fitness Score: {avg_fitness:.4f}")
            print(f"  Average Inlier RMSE: {avg_rmse:.4f}")
            print(f"  Average Computation Time: {avg_time:.4f} seconds per match")
    
    return grid, coordinator

if __name__ == "__main__":
    # If run as a script, execute the main function
    main()