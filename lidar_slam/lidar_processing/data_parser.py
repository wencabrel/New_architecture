"""
LiDAR Data Parser Module

This module provides functions for parsing raw LiDAR data from files or strings.
"""

import os
import time

def parse_lidar_data(data_string):
    """Parse the LiDAR data string and extract all relevant information
    
    Args:
        data_string (str): Raw LiDAR data string
        
    Returns:
        dict: Dictionary containing parsed LiDAR data with keys:
            - header: LiDAR sensor type
            - num_points: Number of scan points
            - scan_ranges: List of distance measurements
            - pose: Dictionary with 'x', 'y', 'theta' keys
            - timestamp: Unix timestamp of the scan
            - robot_id: Robot identifier
            - last_value: Additional data (possibly speed)
    """
    parts = data_string.strip().split()
    
    # Extract header information
    header = parts[0]  # 'LiDAR_E300'
    
    # Extract number of scan points
    num_points = int(parts[1])  # 180
    
    # Extract scan ranges (LiDAR measurements)
    scan_ranges = [float(parts[i]) for i in range(2, 2 + num_points)]
    
    # Extract pose information (x, y, theta)
    pose_idx = 2 + num_points
    x = float(parts[pose_idx])        # -0.001035
    y = float(parts[pose_idx + 1])    # -0.000620
    theta = float(parts[pose_idx + 2]) # 0.383864
    
    # Extract timestamp and robot ID
    timestamp = float(parts[pose_idx + 6])  # 1742988203.467810
    robot_id = parts[pose_idx + 7]          # zjnu-R1
    
    # Last value (possibly speed or other data)
    last_value = float(parts[pose_idx + 8])  # 0.000246
    
    return {
        'header': header,
        'num_points': num_points,
        'scan_ranges': scan_ranges,
        'pose': {
            'x': x,
            'y': y,
            'theta': theta
        },
        'timestamp': timestamp,
        'robot_id': robot_id,
        'last_value': last_value
    }

def read_lidar_data_from_file(file_path, max_entries=200):
    """Read LiDAR data from a file, up to max_entries
    
    Args:
        file_path (str): Path to the file containing LiDAR data
        max_entries (int, optional): Maximum number of entries to read. Defaults to 200.
        
    Returns:
        list: List of parsed LiDAR data dictionaries
    """
    parsed_data_list = []
    
    try:
        with open(file_path, 'r') as file:
            count = 0
            for line in file:
                if count >= max_entries:
                    break
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    parsed_data = parse_lidar_data(line)
                    parsed_data_list.append(parsed_data)
                    count += 1
                except Exception as e:
                    print(f"Error parsing line {count + 1}: {e}")
                    continue
            
            print(f"Successfully read {len(parsed_data_list)} entries from {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return parsed_data_list

def get_data_summary(parsed_data_list):
    """Generate a summary of the parsed LiDAR data
    
    Args:
        parsed_data_list (list): List of parsed LiDAR data dictionaries
        
    Returns:
        dict: Summary information including count, robot ID, and duration
    """
    if not parsed_data_list:
        return {
            "count": 0,
            "robot_id": None,
            "duration": 0,
            "start_time": None,
            "end_time": None
        }
    
    first_timestamp = parsed_data_list[0]['timestamp']
    last_timestamp = parsed_data_list[-1]['timestamp']
    duration = last_timestamp - first_timestamp
    
    return {
        "count": len(parsed_data_list),
        "robot_id": parsed_data_list[0]['robot_id'],
        "duration": duration,
        "start_time": first_timestamp,
        "end_time": last_timestamp
    }