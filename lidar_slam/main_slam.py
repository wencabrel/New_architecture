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

# Import the SLAM components
from lidar_processing.scan_matcher import ScanMatcher
from pose_estimation.pose_estimator import PoseEstimator, PoseEstimationMethod, Pose2D
from slam_coordinator import SLAMCoordinator, SLAMMode

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced LiDAR SLAM Visualization and Mapping')
    
    parser.add_argument('--file', '-f', type=str, 
                        default="./dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                        help='Path to the LiDAR data file')
    
    parser.add_argument('--max-entries', '-m', type=int, default=3162,
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
    
    # Arguments for scan matching and pose estimation
    parser.add_argument('--slam-mode', choices=['odometry_only', 'scan_matching', 'full_slam'], 
                       default='scan_matching', help='SLAM operation mode')
    
    parser.add_argument('--pose-method', choices=['odometry_only', 'scan_match_only', 
                                               'ekf_fusion', 'weighted_average'], 
                       default='ekf_fusion', help='Pose estimation method')
    
    parser.add_argument('--scan-match-interval', type=int, default=1,
                       help='Process every Nth scan for matching (1 = all scans)')
    
    parser.add_argument('--show-corrections', action='store_true', default=False,
                       help='Show both raw odometry and corrected paths')
    
    # Advanced scan matcher parameters
    parser.add_argument('--search-radius', type=float, default=1.4,
                       help='Maximum search radius for scan matching in meters')
    
    parser.add_argument('--search-angle', type=float, default=0.25,
                       help='Maximum search angle for scan matching in radians')
    
    parser.add_argument('--coarse-factor', type=int, default=5,
                       help='Coarse-to-fine resolution reduction factor')
    
    parser.add_argument('--min-confidence', type=float, default=0.1,
                       help='Minimum confidence threshold for scan matching')
    
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
    
    # Configure SLAM mode
    slam_mode = SLAMMode(args.slam_mode)
    pose_method = PoseEstimationMethod(args.pose_method)
    
    # Initialize the SLAM coordinator
    print(f"\nInitializing SLAM system...")
    print(f"  SLAM Mode: {slam_mode.value}")
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
    
    # Configure scan matcher parameters
    coordinator.scan_matcher_params = {
        'search_radius': args.search_radius,
        'search_half_rad': args.search_angle,
        'scan_sigma_in_num_grid': 2,  # Default parameter
        'move_r_sigma': 0.1,  # Default parameter
        'max_move_deviation': args.search_radius * 0.2,  # 20% of search radius
        'turn_sigma': 0.3,  # Default parameter
        'mismatch_prob_at_coarse': args.min_confidence,
        'coarse_factor': args.coarse_factor
    }
    
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

def plot_slam_results(coordinator, output_dir=None):
    """
    Plot additional SLAM results for analysis
    
    Args:
        coordinator: SLAMCoordinator with processed data
        output_dir: Optional directory to save plots
    """
    if not coordinator or not coordinator.scan_match_results:
        print("No scan matching results to plot")
        return
    
    # Plot the confidence over time
    plt.figure(figsize=(10, 6))
    confidences = [result['confidence'] for result in coordinator.scan_match_results]
    plt.plot(confidences, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="50% Confidence")
    plt.title("Scan Matching Confidence Over Time")
    plt.xlabel("Scan Number")
    plt.ylabel("Confidence Score")
    plt.grid(True)
    plt.tight_layout()
    
    if output_dir:
        file_utils.ensure_directory_exists(output_dir)
        plt.savefig(os.path.join(output_dir, "scan_match_confidence.png"), dpi=300)
    
    # Plot the corrections compared to odometry
    if coordinator.pose_corrections:
        plt.figure(figsize=(10, 6))
        
        # Extract original and corrected coordinates
        odom_x = [corr['odom_x'] for corr in coordinator.pose_corrections]
        odom_y = [corr['odom_y'] for corr in coordinator.pose_corrections]
        corr_x = [corr['corrected_x'] for corr in coordinator.pose_corrections]
        corr_y = [corr['corrected_y'] for corr in coordinator.pose_corrections]
        
        # Plot both paths
        plt.plot(odom_x, odom_y, 'r-', alpha=0.7, linewidth=1.5, label="Odometry")
        plt.plot(corr_x, corr_y, 'b-', linewidth=2, label="SLAM Corrected")
        
        # Highlight start and end points
        plt.scatter(odom_x[0], odom_y[0], c='g', s=100, marker='*', label="Start")
        plt.scatter(corr_x[-1], corr_y[-1], c='m', s=100, marker='*', label="End")
        
        # Connect corresponding points to show corrections
        for i in range(0, len(odom_x), 10):  # Show every 10th correction to avoid clutter
            plt.plot([odom_x[i], corr_x[i]], [odom_y[i], corr_y[i]], 'k--', alpha=0.3)
        
        plt.title("Odometry vs SLAM-Corrected Path")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "path_comparison.png"), dpi=300)
    
    # Show the plots if not saving
    if not output_dir:
        plt.show()

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
        successful_matches = sum(1 for result in coordinator.scan_match_results 
                               if result['confidence'] > args.min_confidence)
        total_matches = len(coordinator.scan_match_results)
        success_rate = successful_matches / total_matches * 100 if total_matches > 0 else 0
        
        print(f"\nSLAM Statistics:")
        print(f"  Scan Matches: {successful_matches}/{total_matches} successful ({success_rate:.1f}%)")
        
        if successful_matches > 0:
            avg_confidence = np.mean([r['confidence'] for r in coordinator.scan_match_results])
            print(f"  Average Confidence: {avg_confidence:.4f}")
            
            if coordinator.pose_corrections:
                # Calculate average correction
                correction_distances = [
                    np.sqrt((corr['corrected_x'] - corr['odom_x'])**2 + 
                            (corr['corrected_y'] - corr['odom_y'])**2)
                    for corr in coordinator.pose_corrections
                ]
                avg_correction = np.mean(correction_distances)
                print(f"  Average Pose Correction: {avg_correction:.4f} meters")
        
        # Plot additional SLAM results if requested
        if args.show_corrections:
            plot_slam_results(coordinator, args.output_dir)
    
    return grid, coordinator

if __name__ == "__main__":
    # If run as a script, execute the main function
    main()