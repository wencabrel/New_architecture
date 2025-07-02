import numpy as np
import math

def read_lidar_data_from_file_range(file_path, start_entry=0, end_entry=None, max_total=None):
    """
    Read LiDAR data from a file within a specified range
    
    Args:
        file_path: Path to the LiDAR data file
        start_entry: Starting entry index (0-based, inclusive)
        end_entry: Ending entry index (0-based, exclusive). If None, reads to end of file
        max_total: Maximum total entries to read (optional limit)
    
    Returns:
        List of parsed LiDAR data dictionaries from the specified range
    """
    parsed_data_list = []
    
    try:
        with open(file_path, 'r') as file:
            current_index = 0
            entries_read = 0
            
            # Skip lines until we reach start_entry
            for line in file:
                if current_index < start_entry:
                    current_index += 1
                    continue
                
                # Check if we've reached the end_entry
                if end_entry is not None and current_index >= end_entry:
                    break
                
                # Check if we've read enough entries (optional limit)
                if max_total is not None and entries_read >= max_total:
                    break
                
                # Skip empty lines
                if not line.strip():
                    current_index += 1
                    continue
                
                try:
                    # Parse the current line
                    from lidar_utility_functions import parse_lidar_data
                    parsed_data = parse_lidar_data(line)
                    parsed_data_list.append(parsed_data)
                    entries_read += 1
                    current_index += 1
                except Exception as e:
                    print(f"Error parsing line {current_index}: {e}")
                    current_index += 1
                    continue
            
            print(f"Successfully read {len(parsed_data_list)} entries from range [{start_entry}:{end_entry if end_entry else 'end'}] in {file_path}")
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return parsed_data_list

def read_lidar_data_with_range(file_path, entry_range=None, max_entries=None):
    """
    Read LiDAR data from a file with flexible range specification
    
    Args:
        file_path: Path to the LiDAR data file
        entry_range: Tuple of (start, end) indices, or single int for start index, or None for all
        max_entries: Maximum entries to read (for backward compatibility)
    
    Returns:
        List of parsed LiDAR data dictionaries
    
    Examples:
        # Read first 200 entries (backward compatibility)
        data = read_lidar_data_with_range(file_path, max_entries=200)
        
        # Read entries 100-200
        data = read_lidar_data_with_range(file_path, entry_range=(100, 200))
        
        # Read entries 50 to end of file
        data = read_lidar_data_with_range(file_path, entry_range=(50, None))
        
        # Read from entry 100, maximum 50 entries
        data = read_lidar_data_with_range(file_path, entry_range=100, max_entries=50)
    """
    
    # Handle different input formats
    if entry_range is None:
        # Backward compatibility: read from start
        start_entry = 0
        end_entry = max_entries if max_entries else None
    elif isinstance(entry_range, tuple):
        # Range specified as (start, end)
        start_entry, end_entry = entry_range
    elif isinstance(entry_range, int):
        # Single start index
        start_entry = entry_range
        end_entry = None
    else:
        raise ValueError("entry_range must be None, int, or tuple of (start, end)")
    
    # If max_entries is specified and we don't have an end_entry, calculate it
    if max_entries is not None and end_entry is None:
        end_entry = start_entry + max_entries
    
    # Call the core function
    return read_lidar_data_from_file_range(file_path, start_entry, end_entry)

def analyze_file_info(file_path, sample_size=10):
    """
    Analyze a LiDAR file to get basic information about its contents
    
    Args:
        file_path: Path to the LiDAR data file
        sample_size: Number of entries to sample for analysis
    
    Returns:
        Dictionary with file information
    """
    info = {
        'total_lines': 0,
        'valid_entries': 0,
        'sample_timestamps': [],
        'robot_ids': set(),
        'estimated_duration': 0,
        'entries_per_second': 0
    }
    
    try:
        with open(file_path, 'r') as file:
            sample_data = []
            
            for line_num, line in enumerate(file):
                info['total_lines'] += 1
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Try to parse a sample of entries
                if len(sample_data) < sample_size:
                    try:
                        from lidar_utility_functions import parse_lidar_data
                        parsed_data = parse_lidar_data(line)
                        sample_data.append(parsed_data)
                        info['sample_timestamps'].append(parsed_data['timestamp'])
                        info['robot_ids'].add(parsed_data['robot_id'])
                        info['valid_entries'] += 1
                    except:
                        pass
                else:
                    # Continue counting valid entries without parsing
                    try:
                        parts = line.strip().split()
                        if len(parts) > 10:  # Basic check for valid format
                            info['valid_entries'] += 1
                    except:
                        pass
        
        # Calculate timing information if we have samples
        if len(info['sample_timestamps']) >= 2:
            timestamps = sorted(info['sample_timestamps'])
            time_span = timestamps[-1] - timestamps[0]
            entries_in_span = len(timestamps)
            
            if time_span > 0:
                # Estimate total duration and rate
                info['entries_per_second'] = entries_in_span / time_span
                info['estimated_duration'] = info['valid_entries'] / info['entries_per_second']
        
    except Exception as e:
        print(f"Error analyzing file {file_path}: {e}")
    
    return info

# Example usage functions
def print_file_analysis(file_path):
    """Print analysis of a LiDAR data file"""
    print(f"Analyzing file: {file_path}")
    print("-" * 50)
    
    info = analyze_file_info(file_path)
    
    print(f"Total lines in file: {info['total_lines']}")
    print(f"Valid entries: {info['valid_entries']}")
    print(f"Robot IDs found: {list(info['robot_ids'])}")
    
    if info['estimated_duration'] > 0:
        print(f"Estimated duration: {info['estimated_duration']:.2f} seconds")
        print(f"Estimated rate: {info['entries_per_second']:.2f} entries/second")
    
    if info['sample_timestamps']:
        print(f"Sample timestamp range: {min(info['sample_timestamps']):.3f} - {max(info['sample_timestamps']):.3f}")
    
    print("-" * 50)
    print("Example usage:")
    print(f"# Read entries 100-200:")
    print(f"data = read_lidar_data_with_range('{file_path}', entry_range=(100, 200))")
    print(f"# Read last 100 entries:")
    print(f"data = read_lidar_data_with_range('{file_path}', entry_range=({info['valid_entries']-100}, None))")
    print(f"# Read middle section:")
    print(f"data = read_lidar_data_with_range('{file_path}', entry_range=({info['valid_entries']//3}, {info['valid_entries']*2//3}))")

def extract_time_based_range(file_path, start_time=None, end_time=None, relative_start=None, duration=None):
    """
    Extract LiDAR data based on time ranges instead of entry indices
    
    Args:
        file_path: Path to the LiDAR data file
        start_time: Absolute start timestamp (optional)
        end_time: Absolute end timestamp (optional)
        relative_start: Start time relative to first entry in seconds (optional)
        duration: Duration in seconds from start_time or relative_start (optional)
    
    Returns:
        List of parsed LiDAR data dictionaries
    """
    # First, read a small sample to get the time range
    sample_data = read_lidar_data_from_file_range(file_path, 0, 100)
    
    if not sample_data:
        print("Could not read sample data for time analysis")
        return []
    
    first_timestamp = sample_data[0]['timestamp']
    
    # Calculate actual start and end times
    if relative_start is not None:
        actual_start_time = first_timestamp + relative_start
    elif start_time is not None:
        actual_start_time = start_time
    else:
        actual_start_time = first_timestamp
    
    if duration is not None:
        actual_end_time = actual_start_time + duration
    elif end_time is not None:
        actual_end_time = end_time
    else:
        actual_end_time = None
    
    # Now read all data and filter by time
    all_data = read_lidar_data_from_file_range(file_path, 0, None)
    
    filtered_data = []
    for data in all_data:
        timestamp = data['timestamp']
        
        if timestamp < actual_start_time:
            continue
        
        if actual_end_time is not None and timestamp > actual_end_time:
            break
        
        filtered_data.append(data)
    
    print(f"Extracted {len(filtered_data)} entries from time range {actual_start_time:.3f} to {actual_end_time if actual_end_time else 'end':.3f}")
    
    return filtered_data

# Advanced range utilities
def find_entry_by_timestamp(file_path, target_timestamp, tolerance=1.0):
    """
    Find the entry index closest to a target timestamp
    
    Args:
        file_path: Path to the LiDAR data file
        target_timestamp: Target timestamp to find
        tolerance: Maximum time difference to accept (seconds)
    
    Returns:
        Entry index closest to target timestamp, or None if not found
    """
    try:
        with open(file_path, 'r') as file:
            best_index = None
            best_diff = float('inf')
            
            for index, line in enumerate(file):
                if not line.strip():
                    continue
                
                try:
                    from lidar_utility_functions import parse_lidar_data
                    parsed_data = parse_lidar_data(line)
                    timestamp = parsed_data['timestamp']
                    
                    diff = abs(timestamp - target_timestamp)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_index = index
                    
                    # Early exit if we've passed the target time by more than tolerance
                    if timestamp > target_timestamp + tolerance:
                        break
                        
                except:
                    continue
            
            if best_diff <= tolerance:
                print(f"Found entry {best_index} at timestamp {target_timestamp} (diff: {best_diff:.3f}s)")
                return best_index
            else:
                print(f"No entry found within {tolerance}s of timestamp {target_timestamp}")
                return None
                
    except Exception as e:
        print(f"Error searching file: {e}")
        return None

def get_range_around_position(file_path, target_x, target_y, radius=2.0, max_entries=200):
    """
    Get entries where the robot was within a certain radius of a target position
    
    Args:
        file_path: Path to the LiDAR data file
        target_x, target_y: Target position coordinates
        radius: Search radius in meters
        max_entries: Maximum entries to check
    
    Returns:
        List of (index, data) tuples for entries within radius
    """
    matching_entries = []
    
    try:
        with open(file_path, 'r') as file:
            for index, line in enumerate(file):
                if len(matching_entries) >= max_entries:
                    break
                
                if not line.strip():
                    continue
                
                try:
                    from lidar_utility_functions import parse_lidar_data
                    parsed_data = parse_lidar_data(line)
                    
                    robot_x = parsed_data['pose']['x']
                    robot_y = parsed_data['pose']['y']
                    
                    distance = math.sqrt((robot_x - target_x)**2 + (robot_y - target_y)**2)
                    
                    if distance <= radius:
                        matching_entries.append((index, parsed_data))
                        
                except:
                    continue
    
    except Exception as e:
        print(f"Error searching for position: {e}")
    
    print(f"Found {len(matching_entries)} entries within {radius}m of ({target_x}, {target_y})")
    return matching_entries