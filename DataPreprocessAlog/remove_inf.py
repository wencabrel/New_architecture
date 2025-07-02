#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

def replace_inf_with_max_range(input_file_path, output_file_path, max_range=25.0):
    """
    Process a data file and replace any 'inf' values in range readings with max_range
    
    Args:
        input_file_path: Path to the input data file
        output_file_path: Path to save the processed output file
        max_range: Maximum range value to replace 'inf' with
    """
    print("Processing file: {}".format(input_file_path))
    print("Output will be saved to: {}".format(output_file_path))
    
    # Check if input file exists
    if not os.path.exists(input_file_path):
        print("Error: Input file does not exist!")
        return False
    
    # Open input and output files
    with open(input_file_path, 'r') as in_file, open(output_file_path, 'w') as out_file:
        # Copy header if it exists (lines starting with #)
        line = in_file.readline()
        if line.startswith('#'):
            out_file.write(line)
            line = in_file.readline()
        
        # Process each line
        lines_processed = 0
        inf_values_replaced = 0
        
        while line:
            if line.startswith('FLASER'):
                parts = line.split()
                # Extract necessary parts
                flaser_token = parts[0]
                num_readings = int(parts[1])
                
                # The range readings start at index 2 and end at index 2+num_readings-1
                range_readings = parts[2:2+num_readings]
                remaining_parts = parts[2+num_readings:]
                
                # Process range readings
                new_range_readings = []
                for reading in range_readings:
                    try:
                        # Convert to float and check if it's infinite
                        value = float(reading)
                        if np.isinf(value):
                            new_range_readings.append("{:.2f}".format(max_range))
                            inf_values_replaced += 1
                        else:
                            new_range_readings.append("{:.2f}".format(value))
                    except ValueError:
                        # If conversion fails, keep the original value
                        # print("Warning: Non-numeric value found in range readings: {}".format(reading))
                        new_range_readings.append(reading)
                
                # Reassemble the line
                new_line = "{} {} {} {}\n".format(
                    flaser_token,
                    num_readings,
                    ' '.join(new_range_readings),
                    ' '.join(remaining_parts)
                )
                out_file.write(new_line)
            else:
                # For non-FLASER lines, just copy them
                out_file.write(line)
            
            lines_processed += 1
            line = in_file.readline()
    
    print("Processing complete!")
    print("Processed {} lines".format(lines_processed))
    print("Replaced {} 'inf' values with {}".format(inf_values_replaced, max_range))
    return True

def main():
    # Define file paths
    base_path = '/home/goldenbwuoy/Research/ROS&SLAM/code/New_architecture'
    input_file = os.path.join(base_path, 'DataSet/RawData/laser_data_synchronized_data_second_try.clf')
    output_file = os.path.join(base_path, 'lidar_slam/dataset/raw_data/laser_data_synchronized_data_second_try_processed_full.clf')
    
    # Process the file
    replace_inf_with_max_range(input_file, output_file, 25.0)

if __name__ == '__main__':
    main()