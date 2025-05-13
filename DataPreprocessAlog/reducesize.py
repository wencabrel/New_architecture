def process_lidar_data(input_data, degree_increment=2):
    """
    Process a single LiDAR scan to extract values at 2-degree increments.
    
    Note: For a 360-degree scan with 2-degree increments, we should get 181 values:
    0°, 2°, 4°, ..., 358°, 360°
    """
    """
    Process a single LiDAR scan to extract values at 2-degree increments.
    
    Args:
        input_data (str): Space-separated LiDAR data string
        degree_increment (int): Degree increment for sampling
    
    Returns:
        str: Processed LiDAR data with values at specified degree increments
    """
    # Split the data into tokens
    tokens = input_data.strip().split()
    
    # Extract metadata and values
    identifier = tokens[0]  # LiDAR_E300
    total_values = int(tokens[1])  # 803
    
    # Extract the actual distance measurements
    distances = [float(value) for value in tokens[2:2+total_values]]
    
    # Extract the trailing metadata after the distances
    trailing_metadata = tokens[2+total_values:]
    
    # Calculate the values to keep based on 2-degree increments
    # For 360 degrees with 2-degree increments, we'll have 181 values (0, 2, 4, ..., 358, 360)
    # We need to map these 181 values to the 803 values in the original data
    
    # The original data has an angle increment of 360 / (803-1) = 0.44888777 degrees
    original_increment = 360 / (total_values - 1)
    
    # Calculate indices to keep
    indices_to_keep = []
    for degree in range(0, 360, degree_increment):
        # Convert degree to index in the original array
        index = round(degree / original_increment)
        if index < total_values:  # Ensure we don't exceed array bounds
            indices_to_keep.append(index)
    
    # Extract the values at the calculated indices
    new_distances = [distances[i] for i in indices_to_keep]
    
    # Build the new data string
    new_data = [identifier, str(len(new_distances))]
    new_data.extend([str(distance) for distance in new_distances])
    new_data.extend(trailing_metadata)
    
    return ' '.join(new_data)

def process_file(file_path, output_path, degree_increment=2):
    """
    Process a file containing multiple LiDAR scans.
    
    Args:
        file_path (str): Path to the input file
        output_path (str): Path for the output file
        degree_increment (int): Degree increment for sampling
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the content by LiDAR_E300 to get each scan
    parts = content.split('FLASER')
    
    # Process each scan
    processed_parts = []
    for i, part in enumerate(parts):
        if i == 0 and not part.strip():
            # Skip empty first part
            continue
        
        # Reconstruct the original scan data with the header
        scan_data = 'FLASER' + part
        processed_scan = process_lidar_data(scan_data, degree_increment)
        processed_parts.append(processed_scan)
    
    # Join the processed scans and write to output file
    processed_content = '\n'.join(processed_parts)
    
    with open(output_path, 'w') as file:
        file.write(processed_content)
    
    # Count and verify the number of values per scan
    for i, scan in enumerate(processed_parts):
        tokens = scan.split()
        scan_id = tokens[0]
        num_values = int(tokens[1])
        print(f"Scan {i+1}: {scan_id} - Contains {num_values} values at 2-degree increments")
    
    print(f"\nProcessed data saved to {output_path}")
    print(f"Original data had {len(parts)-1} scans, each with 803 values")
    print(f"Each processed scan should have 181 values (from 0° to 360° in 2° increments)")

# Example usage
if __name__ == "__main__":
    input_file = "../lidar_slam/dataset/raw_data/laser_data_synchronized_data_second_try_processed_full.clf"
    output_file = "../lidar_slam/dataset/raw_data/laser_data_synchronized_data_second_try_processed_reduced180.clf"
    process_file(input_file, output_file, 2)




