import json


laser_scan_input_path = "../DataSet/RawData/intel.clf"
processed_data_output_path = "../DataSet/DataPreprocessed/intel_clf"

# An empty dictionary to store our laser scan measurements
laser_scan_measurements = {}

# We open and read the input file containing laser scanner data
with open(laser_scan_input_path, "r") as laser_scan_file:
    # We iterate through each line in the file
    for scan_line in laser_scan_file:
        # Here we check if this line contains laser scanner data (FLASER format)
        if scan_line.startswith('FLASER'):
            # We split the line into individual data elements
            scan_data_elements = scan_line.split()
            
            # extract the number of range measurements in this scan
            num_range_measurements = int(scan_data_elements[1])
            
            # extract the range measurements and convert them to float values
            range_measurements = scan_data_elements[2: num_range_measurements + 2]
            range_measurements = [float(measurement) for measurement in range_measurements]
            
            # extract robot pose (position and orientation) and timestamp
            robot_x = float(scan_data_elements[num_range_measurements + 2])
            robot_y = float(scan_data_elements[num_range_measurements + 3])
            robot_theta = float(scan_data_elements[num_range_measurements + 4])
            scan_timestamp = float(scan_data_elements[num_range_measurements + 8])
            
            # We store all measurements in our dictionary, indexed by timestamp
            laser_scan_measurements[scan_timestamp] = {
                'x': robot_x,
                'y': robot_y,
                'theta': robot_theta,
                'range': range_measurements
            }

# prepare our data structure for JSON export
processed_data = {
    'map': laser_scan_measurements,
}

# save the processed data to a JSON file
with open(processed_data_output_path, 'w') as output_file:
    # We write the data with proper formatting for readability
    json.dump(processed_data, output_file, sort_keys=True, indent=4)