# # Define the input and output file paths
# input_file_path = "../DataSet/RawData/intel.clf"
# output_file_path = "../DataSet/RawData/flaser_entries.clf"

# # Open the input file for reading and the output file for writing
# with open(input_file_path, "r") as infile, open(output_file_path, "w") as outfile:
#     # Iterate through each line in the input file
#     for line in infile:
#         # Check if the line starts with "FLASER"
#         if line.startswith("FLASER"):
#             # Write the line to the output file
#             outfile.write(line)

# print(f"FLASER entries have been extracted and saved to {output_file_path}")

import json

# Input and output file paths
input_file_path = "../DataSet/DataPreprocessed/intel-gfs"
output_file_path = "../DataSet/RawData/formatted_intel.clf"

# Load the JSON-like data
with open(input_file_path, "r") as infile:
    data = json.load(infile)

# Open the output file for writing
with open(output_file_path, "w") as outfile:
    for timestamp, values in data["map"].items():
        if "range" in values:
            # Extract the range readings
            range_readings = " ".join(map(str, values["range"]))
            # Format the FLASER line
            flaser_line = f"FLASER {len(values['range'])} {range_readings} {values.get('x', 0.0)} {values.get('y', 0.0)} {values.get('theta', 0.0)} {values.get('x', 0.0)} {values.get('y', 0.0)} {values.get('theta', 0.0)} {timestamp} nohost 0.0\n"
            # Write to the output file
            outfile.write(flaser_line)

print(f"Formatted FLASER data has been saved to {output_file_path}")