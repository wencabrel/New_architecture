def read_odom_data(filename):
    """Read ODOM rows and extract 4th and 5th columns."""
    odom_data = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('ODOM'):
                parts = line.split()
                # Get 4th and 5th columns (index 4 and 5 since we split the line)

                # odom_data.append((float(parts[4]), float(parts[5]), float(parts[7]), float(parts[9])))
                odom_data.append((float(parts[7]), float(parts[9])))
    return odom_data

def process_files(sample_file, robot_file, output_file):
    """Process the files and create merged output."""
    # Read the sample file ODOM data (4th and 5th columns)
    sample_odom = read_odom_data(sample_file)
    
    # Process robot file and create merged output
    with open(robot_file, 'r') as robot, open(output_file, 'w') as output:
        odom_index = 0
        
        for line in robot:
            if line.startswith('ODOM'):
                parts = line.split()
                if odom_index < len(sample_odom):
                    # Replace 4th and 5th columns with values from sample file

                    # new_val4, new_val5, new_val7, new_val9 = sample_odom[odom_index]
                    new_val7, new_val9 = sample_odom[odom_index]
                    
                    # parts[4] = f"{new_val4}"
                    # parts[5] = f"{new_val5}"
                    parts[7] = f"{new_val7}"
                    parts[9] = f"{new_val9}"
                    odom_index += 1
                # Reconstruct the line with replaced values
                new_line = ' '.join(parts) + '\n'
                output.write(new_line)
            else:
                # Write non-ODOM lines as they are
                output.write(line)

# Use the function
try:
    process_files('sample_dataset.clf', 'robot_path_meters_modified.clf', 'merged.clf')
    print("Merge completed successfully. Output saved to merged.clf")
except Exception as e:
    print(f"An error occurred: {e}")