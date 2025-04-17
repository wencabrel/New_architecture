"""
Scan Converter Module

This module provides functions for converting LiDAR scan data between
polar and Cartesian coordinate systems, with various transformation options.
"""

import math
import numpy as np

def convert_scans_to_cartesian(scan_ranges, angle_min, angle_max, pose, 
                               flip_x=False, flip_y=False, reverse_scan=False, flip_theta=False):
    """
    Convert scan ranges to Cartesian coordinates based on robot's pose
    
    Args:
        scan_ranges (list): List of LiDAR distance measurements
        angle_min (float): Starting angle of the scan (radians)
        angle_max (float): Ending angle of the scan (radians)
        pose (dict): Dictionary with 'x', 'y', 'theta' keys for robot's pose
        flip_x (bool): Whether to flip the x-axis
        flip_y (bool): Whether to flip the y-axis
        reverse_scan (bool): Whether to reverse the scan direction
        flip_theta (bool): Whether to negate the orientation angle
        
    Returns:
        tuple: Two lists of (x_world, y_world) coordinates of valid scan points
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

def find_scan_boundaries(scans_x, scans_y, padding_percentage=0.2):
    """
    Find the boundaries of scan data with optional padding
    
    Args:
        scans_x (list): List of x-coordinates from all scans
        scans_y (list): List of y-coordinates from all scans
        padding_percentage (float): Padding to add as percentage of range (0.2 = 20%)
        
    Returns:
        tuple: (x_min, x_max, y_min, y_max) boundaries with padding
    """
    if not scans_x or not scans_y:
        return -5, 5, -5, 5  # Default values if no data is provided
    
    x_min, x_max = min(scans_x), max(scans_x)
    y_min, y_max = min(scans_y), max(scans_y)
    
    # Add padding
    x_padding = max(1.0, (x_max - x_min) * padding_percentage)
    y_padding = max(1.0, (y_max - y_min) * padding_percentage)
    
    return (x_min - x_padding, x_max + x_padding, 
            y_min - y_padding, y_max + y_padding)

def extract_robot_path(parsed_data_list, flip_x=False, flip_y=False):
    """
    Extract robot path from parsed LiDAR data
    
    Args:
        parsed_data_list (list): List of parsed LiDAR data dictionaries
        flip_x (bool): Whether to flip the x coordinates
        flip_y (bool): Whether to flip the y coordinates
        
    Returns:
        tuple: Two lists (x_coords, y_coords) of robot positions
    """
    path_x = []
    path_y = []
    
    for data in parsed_data_list:
        x, y = data['pose']['x'], data['pose']['y']
        if flip_x:
            x = -x
        if flip_y:
            y = -y
        path_x.append(x)
        path_y.append(y)
    
    return path_x, path_y