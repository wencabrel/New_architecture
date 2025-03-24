#!/usr/bin/env python3
# filepath: /home/goldenbwuoy/Research/ROS&SLAM/code/New_architecture/DataPreprocessAlog/format_data.py

import sys
import numpy as np
import re

def format_flaser_line(data, pose=None):
    """
    Format input data into FLASER format.
    Replace inf and nan values with max range (11.9).
    
    Expected FLASER format:
    FLASER <num_readings> <range_readings...> <x> <y> <theta> <x2> <y2> <theta2> <timestamp> <hostname> <timestamp_in_seconds>
    
    Args:
        data: Array of range readings
        pose: Optional tuple (x, y, theta) for pose information
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=float)
    
    # Replace inf and nan values with 11.9 (max range)
    data = np.nan_to_num(data, nan=11.9, posinf=11.9, neginf=11.9)
    
    # Number of laser readings is determined by the data length
    num_readings = len(data)
    
    # Format the range readings with 2 decimal places
    range_readings_str = ' '.join([f'{reading:.2f}' for reading in data])
    
    # Default values
    x = 0.000000
    y = 0.000000
    theta = -0.002458
    timestamp = 976052857.337530
    hostname = "zjnu-R1"
    timestamp_seconds = 0.000246
    
    # Use provided pose if available
    if pose and len(pose) >= 3:
        x = pose[0]
        y = pose[1]
        theta = pose[2]
    
    # Use same values for x2, y2, theta2 as requested
    x2 = x
    y2 = y
    theta2 = theta
    
    # Construct the FLASER line
    flaser_line = (f"LiDAR_E300 {num_readings} {range_readings_str} {x:.6f} {y:.6f} {theta:.6f} "
                  f"{x2:.6f} {y2:.6f} {theta2:.6f} {timestamp} {hostname} {timestamp_seconds}")
    
    return flaser_line

def load_data_from_file(filename):
    """
    Load data from file, supporting different formats.
    Returns a list of tuples (data_array, pose_tuple) where pose_tuple is (x,y,theta) if available
    """
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            entries = []
            
            # Look for patterns like (range_data) pose_data
            # Pattern for data in parentheses followed by pose data
            pattern = r'\(([\d\.\s,\-infa]+)\)\s*([\d\.\-\s]+)'
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            
            if matches:
                for match in matches:
                    range_data_str, pose_data_str = match
                    
                    # Parse range data
                    range_values = []
                    for val in range_data_str.split(','):
                        val = val.strip()
                        if val.lower() == 'inf' or val.lower() == 'infinity':
                            range_values.append(float('inf'))
                        elif val.lower() == 'nan' or val.lower() == '-nan':
                            range_values.append(float('nan'))
                        else:
                            try:
                                range_values.append(float(val))
                            except ValueError:
                                print(f"Warning: Could not convert '{val}' to float. Using max range instead.")
                                range_values.append(11.9)
                    
                    # Parse pose data
                    pose_values = []
                    for val in pose_data_str.split():
                        try:
                            pose_values.append(float(val))
                        except ValueError:
                            print(f"Warning: Could not convert pose value '{val}' to float. Using default instead.")
                            continue
                    
                    # Ensure we have valid pose data (x,y,theta)
                    pose = None
                    if len(pose_values) >= 3:
                        pose = tuple(pose_values[:3])
                    
                    entries.append((np.array(range_values), pose))
                
                return entries
            
            # If no parentheses pattern found, try to parse blocks of data
            lines = content.splitlines()
            
            # Identify blocks separated by empty lines
            current_block = []
            current_pose = None
            blocks = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    # Empty line indicates a new block
                    if current_block:
                        blocks.append((current_block, current_pose))
                        current_block = []
                        current_pose = None
                    continue
                
                # Check if this might be pose data (typically 3-6 numbers)
                if re.match(r'^[\d\.\-\s]+$', line):
                    values = [float(x) for x in line.split() if x.strip()]
                    if 3 <= len(values) <= 6 and current_block:
                        # This looks like pose data following range data
                        current_pose = tuple(values[:3])
                        continue
                
                # Otherwise treat as range data
                if ',' in line:
                    values = [float(x.strip()) if x.strip().lower() not in ('inf', 'infinity', 'nan', '-nan') 
                             else (float('inf') if x.strip().lower() in ('inf', 'infinity') else float('nan')) 
                             for x in line.split(',') if x.strip()]
                else:
                    values = [float(x) if x.lower() not in ('inf', 'infinity', 'nan', '-nan') 
                             else (float('inf') if x.lower() in ('inf', 'infinity') else float('nan')) 
                             for x in line.split() if x.strip()]
                
                current_block.extend(values)
            
            # Add the last block if it exists
            if current_block:
                blocks.append((current_block, current_pose))
            
            # Convert to the expected format
            for block, pose in blocks:
                entries.append((np.array(block), pose))
            
            return entries
                
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def main():
    # Check if input is provided as a file
    if len(sys.argv) > 1:
        data_entries = load_data_from_file(sys.argv[1])
        if not data_entries:
            print("No valid data found. Exiting.")
            return
    else:
        # Generate sample data if no file is provided
        print("No input file provided. Generating sample data...")
        # Generate two sets of sample data with pose
        data_entries = [
            (np.linspace(1.05, 5.86, 180), (1.2, 3.4, -0.5)),  # First entry: (range_data, pose)
            (np.linspace(1.10, 6.00, 180), (2.5, 1.7, 0.3))    # Second entry
        ]
        # Add some inf and nan values to test the replacement
        data_entries[0][0][10] = float('inf')
        data_entries[0][0][20] = float('nan')
    
    print(f"Processing {len(data_entries)} independent data entries...")
    
    # Format each set of readings as a separate FLASER line
    flaser_lines = []
    for i, (data, pose) in enumerate(data_entries):
        flaser_line = format_flaser_line(data, pose)
        flaser_lines.append(flaser_line)
        pose_str = f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})" if pose else "default"
        print(f"Entry {i+1}: {len(data)} readings, pose: {pose_str}")
        print(f"  First 50 chars: {flaser_line[:50]}...")
    
    # Optionally write to output file
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as f:
            for line in flaser_lines:
                f.write(line + "\n")
        print(f"Output written to {sys.argv[2]} ({len(flaser_lines)} entries)")

if __name__ == "__main__":
    main()