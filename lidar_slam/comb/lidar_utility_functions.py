import numpy as np
import math

def parse_lidar_data(data_string):
    """Parse the LiDAR data string and extract all relevant information"""
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

def convert_scans_to_cartesian(scan_ranges, angle_min, angle_max, pose, 
                              flip_x=False, flip_y=False, reverse_scan=False, flip_theta=False):
    """
    Convert scan ranges to Cartesian coordinates based on robot's pose
    
    Args:
        scan_ranges: List of LiDAR distance measurements
        angle_min: Starting angle of the scan (radians)
        angle_max: Ending angle of the scan (radians)
        pose: Dictionary with 'x', 'y', 'theta' keys for robot's pose
        flip_x: Whether to flip the x-axis
        flip_y: Whether to flip the y-axis
        reverse_scan: Whether to reverse the scan direction
        flip_theta: Whether to negate the orientation angle
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

def read_lidar_data_from_file(file_path, max_entries=200):
    """Read LiDAR data from a file, up to max_entries"""
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