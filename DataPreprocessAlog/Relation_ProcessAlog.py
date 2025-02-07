import json


relation_input_path = "../DataSet/RawData/intel.relations"
processed_output_path = "../DataSet/DataPreprocessed/intel_relation-processed"


# First vault stores relationships indexed by the first timestamp
temporal_relations_by_first = {}
# Second vault stores the same relationships but indexed by the second timestamp
temporal_relations_by_second = {}

# Let's dive into the data file and extract those precious relationships
with open(relation_input_path, "r") as relation_file:
   # Each line holds a piece of the puzzle - let's decode them one by one
   for relation_line in relation_file:
         # We split the line into individual data elements
       relation_elements = relation_line.split()
       
       # Extracting the juicy details - position, orientation, and their temporal anchors
       relative_x = float(relation_elements[2])        # The x-coordinate of relative pose
       relative_y = float(relation_elements[3])        # The y-coordinate of relative pose
       relative_theta = float(relation_elements[7])    # The angular relationship between poses
       first_timestamp = float(relation_elements[0])   # When the first observation happened
       second_timestamp = float(relation_elements[1])  # When the second observation occurred

       # Storing this relationship twice - it's all about perspective!
       # First from the viewpoint of the first timestamp
       temporal_relations_by_first[first_timestamp] = {
           'x': relative_x,
           'y': relative_y,
           'theta': relative_theta,
           'timeStamp2': second_timestamp
       }
       
       # Then from the viewpoint of the second timestamp
       temporal_relations_by_second[second_timestamp] = {
           'x': relative_x,
           'y': relative_y,
           'theta': relative_theta,
           'timeStamp1': first_timestamp
       }

# Packaging our processed relationships into a neat JSON structure
processed_relations = {
   'relation_timeStamp1': temporal_relations_by_first,
   'relation_timeStamp2': temporal_relations_by_second,
}

# Time to save our work! Writing everything to a JSON file
# The sort_keys and indent parameters make it human-readable - because we're nice like that
with open(processed_output_path, 'w') as output_file:
   json.dump(processed_relations, output_file, sort_keys=True, indent=4)