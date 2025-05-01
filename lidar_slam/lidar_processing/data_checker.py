#!/usr/bin/env python3
"""
LiDAR Data Validation Script

This script checks if LiDAR data is being parsed and converted correctly.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adjust this path to point to your lidar_slam directory
# sys.path.append(os.path.abspath('lidar_slam'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lidar_processing import data_parser, scan_converter

def validate_lidar_data(file_path, max_entries=10):
    """
    Parse and validate LiDAR data from a file
    
    Args:
        file_path (str): Path to the LiDAR data file
        max_entries (int): Maximum number of entries to check
    """
    print(f"Reading data from: {file_path}")
    
    # Read data
    parsed_data_list = data_parser.read_lidar_data_from_file(file_path, max_entries)
    
    if not parsed_data_list:
        print("ERROR: No data was read from the file!")
        return False
        
    # Print data summary
    summary = data_parser.get_data_summary(parsed_data_list)
    print(f"\nData Summary:")
    print(f"  Entries: {summary['count']}")
    print(f"  Robot ID: {summary['robot_id']}")
    print(f"  Duration: {summary['duration']:.2f} seconds")
    
    # Check scan sizes and ranges
    print("\nScan Validation:")
    for i, data in enumerate(parsed_data_list):
        scan_ranges = data['scan_ranges']
        valid_points = [r for r in scan_ranges if r < 11.0]  # Typical max range
        
        print(f"  Scan {i+1}: {len(scan_ranges)} points, {len(valid_points)} valid points")
        print(f"    Range: min={min(scan_ranges):.2f}, max={max(scan_ranges):.2f}")
        print(f"    Pose: x={data['pose']['x']:.4f}, y={data['pose']['y']:.4f}, theta={data['pose']['theta']:.4f}")
        
        # Check if scan has enough valid points
        if len(valid_points) < 50:
            print(f"    WARNING: Scan has very few valid points!")
    
    # Try converting scans
    print("\nConversion Test:")
    
    # Create test configurations
    configs = [
        {'flip_x': False, 'flip_y': False, 'reverse_scan': False, 'flip_theta': False},
        {'flip_x': False, 'flip_y': False, 'reverse_scan': True, 'flip_theta': False},
        {'flip_x': True, 'flip_y': False, 'reverse_scan': False, 'flip_theta': False},
        {'flip_x': False, 'flip_y': True, 'reverse_scan': False, 'flip_theta': False}
    ]
    
    # Test with first scan
    if parsed_data_list:
        data = parsed_data_list[0]
        angle_min = -np.pi/2
        angle_max = np.pi/2
        
        # Set up plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        for i, config in enumerate(configs):
            # Convert scan
            x_points, y_points = scan_converter.convert_scans_to_cartesian(
                data['scan_ranges'], 
                angle_min, 
                angle_max, 
                data['pose'],
                **config
            )
            
            # Plot points
            axs[i].scatter(x_points, y_points, s=2)
            axs[i].scatter([0], [0], c='r', s=50)  # Origin reference
            
            # Plot robot position and orientation
            robot_x, robot_y = data['pose']['x'], data['pose']['y']
            if config['flip_x']:
                robot_x = -robot_x
            if config['flip_y']:
                robot_y = -robot_y
                
            theta = data['pose']['theta']
            if config['flip_theta']:
                theta = -theta
                
            arrow_length = 0.5
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            
            axs[i].scatter([robot_x], [robot_y], c='b', s=100)
            axs[i].arrow(robot_x, robot_y, dx, dy, head_width=0.1, head_length=0.1, fc='r', ec='r')
            
            # Add config info to plot
            config_str = f"flip_x={config['flip_x']}, flip_y={config['flip_y']}\n"
            config_str += f"reverse_scan={config['reverse_scan']}, flip_theta={config['flip_theta']}"
            axs[i].set_title(config_str)
            axs[i].grid(True)
            axs[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('lidar_scan_configs.png')
        print("  Saved conversion tests to 'lidar_scan_configs.png'")
        
        # Additionally, check consecutive scans
        if len(parsed_data_list) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Use the best config from above
            best_config = configs[1]  # reverse_scan=True is typically best
            
            colors = ['b', 'g', 'r', 'c', 'm']
            for i in range(min(5, len(parsed_data_list))):
                data = parsed_data_list[i]
                x_points, y_points = scan_converter.convert_scans_to_cartesian(
                    data['scan_ranges'], 
                    angle_min, 
                    angle_max, 
                    data['pose'],
                    **best_config
                )
                
                # Plot points with different colors
                ax.scatter(x_points, y_points, s=2, c=colors[i % len(colors)], 
                          label=f"Scan {i+1}")
                
                # Plot robot position
                robot_x, robot_y = data['pose']['x'], data['pose']['y']
                if best_config['flip_x']:
                    robot_x = -robot_x
                if best_config['flip_y']:
                    robot_y = -robot_y
                    
                ax.scatter([robot_x], [robot_y], c=colors[i % len(colors)], s=100, marker='*')
            
            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_title("Multiple Consecutive Scans")
            ax.legend()
            
            plt.tight_layout()
            plt.savefig('consecutive_scans.png')
            print("  Saved consecutive scans to 'consecutive_scans.png'")
        
        print("\nValidation completed. Check the generated images to verify scan conversions.")
        return True
    
    return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Validate LiDAR data')
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='Path to the LiDAR data file')
    parser.add_argument('--max-entries', '-m', type=int, default=100,
                       help='Maximum number of entries to check')
    
    args = parser.parse_args()
    validate_lidar_data(args.file, args.max_entries)