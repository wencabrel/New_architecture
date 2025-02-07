import json


reference_data_path = "../DataSet/DataPreprocessed/intel_raw_refTime"
with open(reference_data_path, 'r') as reference_file:
    reference_input = json.load(reference_file)
    reference_map = reference_input['map']

# Mapping range measurements to reference timestamps
reference_range_mapping = {}
for timestamp in reference_map:
    # Using range measurements as key to find corresponding reference timestamps
    reference_range_mapping[tuple(reference_map[timestamp]['range'])] = timestamp


gfs_input_path = "../DataSet/RawData/intel.gfs"
corrected_log_input_path = "../DataSet/RawData/intel.gfs.log"
gfs_output_path = "../DataSet/DataPreprocessed/intel-gfs"
corrected_log_output_path = "../DataSet/DataPreprocessed/intel_corrected-log"


gfs_measurements = {}      # Stores raw GFS measurements
corrected_measurements = {}  # Stores corrected measurements
gfs_to_reference_time_mapping = {}  # Maps GFS timestamps to reference timestamps

# Processing GFS file (raw measurements)
with open(gfs_input_path, "r") as gfs_file:
    for measurement_line in gfs_file:
        # Looking for laser reading entries
        if measurement_line.startswith('LASER_READING'):
            # Parse the measurement line into components
            measurement_data = measurement_line.split()
            num_measurements = int(measurement_data[1])
            
            # Extract range measurements
            range_readings = measurement_data[2: num_measurements + 2]
            range_readings = [float(reading) for reading in range_readings]
            
            # Extract robot pose and timestamp
            robot_x = float(measurement_data[num_measurements + 2])
            robot_y = float(measurement_data[num_measurements + 3])
            robot_theta = float(measurement_data[num_measurements + 4])
            gfs_timestamp = float(measurement_data[num_measurements + 5])
            
            # Find corresponding reference timestamp using range measurements
            reference_timestamp = reference_range_mapping[tuple(range_readings)]
            
            # Store timestamp mappings and measurements
            gfs_to_reference_time_mapping[gfs_timestamp] = reference_timestamp
            gfs_measurements[reference_timestamp] = {
                'x': robot_x,
                'y': robot_y,
                'theta': robot_theta,
                'range': range_readings
            }

# Save processed GFS data
gfs_processed_data = {'map': gfs_measurements}
with open(gfs_output_path, 'w') as output_file:
    json.dump(gfs_processed_data, output_file, sort_keys=True, indent=4)

# Processing corrected log file
with open(corrected_log_input_path, "r") as log_file:
    for log_line in log_file:
        # Looking for FLASER entries
        if log_line.startswith('FLASER'):
            # Parse the log line into components
            log_data = log_line.split()
            num_measurements = int(log_data[1])
            
            # Extract range measurements
            range_readings = log_data[2: num_measurements + 2]
            range_readings = [float(reading) for reading in range_readings]
            
            # Extract robot pose and timestamp
            robot_x = float(log_data[num_measurements + 2])
            robot_y = float(log_data[num_measurements + 3])
            robot_theta = float(log_data[num_measurements + 4])
            gfs_timestamp = float(log_data[num_measurements + 8])
            
            # Map to reference timestamp and store measurements
            reference_timestamp = gfs_to_reference_time_mapping[gfs_timestamp]
            corrected_measurements[reference_timestamp] = {
                'x': robot_x,
                'y': robot_y,
                'theta': robot_theta,
                'range': range_readings
            }

# Save processed corrected log data
corrected_log_data = {'map': corrected_measurements}
with open(corrected_log_output_path, 'w') as output_file:
    json.dump(corrected_log_data, output_file, sort_keys=True, indent=4)