import re
import os

def format_flaser_data(input_filename, output_filename, inf_replacement=29.90, decimal_places=2):
    """
    Format FLASER data from input file and write to output file.
    
    Args:
        input_filename: Path to the input file
        output_filename: Path to the output file
        inf_replacement: Value to replace 'inf' readings with (default: 30.00)
        decimal_places: Number of decimal places for formatting floating point values (default: 2)
    """
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Extract command (FLASER) and count
            command_match = re.match(r'^(\w+)\s+(\d+)', line)
            if not command_match:
                continue
            
            # Extract readings part (inside parentheses)
            readings_match = re.search(r'\((.*?)\)', line)
            if not readings_match:
                continue
                
            readings_str = readings_match.group(1)
            readings = readings_str.split(', ')
            
            # Extract data after parentheses
            after_match = re.search(r'\)\s+(.*)', line)
            if not after_match:
                continue
                
            after_data = after_match.group(1).split()
            
            # Format the readings - replace 'inf' with the specified value
            formatted_readings = []
            format_str = f"{{:.{decimal_places}f}}"
            
            for reading in readings:
                if reading == 'inf' or reading == 'nan':
                    formatted_readings.append(format_str.format(inf_replacement))
                else:
                    # Convert to float and format to specified decimal places
                    try:
                        value = float(reading)
                        formatted_readings.append(format_str.format(value))
                    except ValueError:
                        # If we can't parse as float, keep the original value
                        formatted_readings.append(reading)
            
            # Create the formatted output line
            # Format: FLASER num_readings [range_readings] x y theta odom_x odom_y odom_theta timestamp hostname 0.000246
            output_line = f"LiDAR_E300 {len(readings)} {' '.join(formatted_readings)} {' '.join(after_data)}"
            outfile.write(output_line + '\n')
            
    print(f"Processing complete. Formatted data written to {output_filename}")

# Script configuration - MODIFY THESE VALUES
if __name__ == "__main__":
    # File paths - change these to your desired input and output paths
    input_file = "../DataSet/RawData/laser_data.clf"
    output_file = "../DataSet/RawData/raw_data_zjnu20_21_3F_big_one.clf"
    
    # Settings
    inf_replacement = 81.83  # Value to replace 'inf' with
    decimal_places = 2      # Number of decimal places for formatted numbers
    
    print(f"Processing {input_file}...")
    print(f"Replacing 'inf' values with {inf_replacement}")
    print(f"Using {decimal_places} decimal places for formatting")
    
    format_flaser_data(input_file, output_file, 
                      inf_replacement=inf_replacement,
                      decimal_places=decimal_places)
    
    print(f"Success! Output written to {output_file}")
    
    # Print a sample of the output
    try:
        with open(output_file, 'r') as f:
            sample = f.readline().strip()
            if len(sample) > 100:
                sample = sample[:100] + "..."
            print(f"\nSample of output: {sample}")
    except Exception as e:
        print(f"Could not read sample from output file: {e}")