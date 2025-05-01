import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import math
import copy  # Add import for deepcopy
import time  # Add import for timestamp generation

class LiDARScan:
    """Class representing a single LiDAR scan from the Intel dataset."""
    
    def __init__(self, line):
        """Parse a line from the Intel dataset file."""
        parts = line.strip().split(' ')
        
        # Header information
        self.sensor_name = parts[0]  # LiDAR_E300
        self.num_readings = 180  # Always ensure 180 readings
        
        # Parse range readings (ensure exactly 180 values)
        expected_ranges = 180
        range_start_idx = 2
        range_end_idx = range_start_idx + expected_ranges
        
        # Check if we have enough range values
        if len(parts) < range_end_idx:
            print(f"Warning: Not enough range values in line. Expected {expected_ranges}, found {len(parts) - range_start_idx}.")
            # Extend the parts list with max range values if needed
            parts.extend(['11.9'] * (range_end_idx - len(parts)))
        
        # Get the ranges
        self.ranges = np.array([float(r) for r in parts[range_start_idx:range_end_idx]])
        
        # Parse pose information (after the ranges)
        try:
            offset = range_end_idx
            self.x = float(parts[offset])
            self.y = float(parts[offset + 1])
            self.theta = float(parts[offset + 2])
            
            # Always use original timestamp if possible
            if len(parts) > offset + 6:
                self.timestamp = float(parts[offset + 6])
            else:
                # Only as fallback
                print("Warning: Original timestamp not found, using placeholder.")
                self.timestamp = 0.0
                
            # Get robot ID and additional time value
            if len(parts) > offset + 7:
                self.robot_id = parts[offset + 7]
            else:
                self.robot_id = "zjnu-R1"
                
            # Get the last time value after robot ID
            if len(parts) > offset + 8:
                self.additional_time = float(parts[offset + 8])
            else:
                self.additional_time = 0.0
                
        except (IndexError, ValueError) as e:
            print(f"Warning: Error parsing position data. Using defaults. Error: {e}")
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0
            self.timestamp = 0.0
            self.robot_id = "zjnu-R1"
            self.additional_time = 0.0
        
        # Calculate angles for each reading (180 degrees FOV)
        angle_min = -np.pi/2  # -90 degrees
        angle_max = np.pi/2   # 90 degrees
        self.angles = np.linspace(angle_min, angle_max, self.num_readings)
        
        # Calculate cartesian coordinates for each range reading
        self.points = self.calculate_points()
    
    def calculate_points(self):
        """Convert range readings to cartesian coordinates."""
        points = []
        max_range = 11.0  # Slightly less than the 11.9 max range value
        
        for i, r in enumerate(self.ranges):
            # Skip invalid readings (11.9 seems to be the max range or invalid reading)
            if r < max_range:
                # Calculate global angle
                global_angle = self.theta + self.angles[i]
                
                # Calculate point coordinates in global frame
                point_x = self.x + r * np.cos(global_angle)
                point_y = self.y + r * np.sin(global_angle)
                points.append((point_x, point_y))
        
        return np.array(points) if points else np.array([])

def read_intel_dataset(file_path, max_scans=None):
    """Read LiDAR scans from the Intel dataset file."""
    scans = []
    
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_scans is not None and i >= max_scans:
                    break
                
                if line.strip():  # Skip empty lines
                    try:
                        scan = LiDARScan(line)
                        scans.append(scan)
                    except Exception as e:
                        print(f"Error parsing line {i}: {e}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        # Create a sample scan from the example line
        example_line = "LiDAR_E300 180 11.9 11.9 11.9 11.9 11.9 11.9 2.69 2.72 2.47 2.74 2.8 2.84 2.99 11.9 11.9 3.03 3.05 3.05 3.05 3.05 3.04 3.04 3.05 3.05 3.06 3.06 3.07 3.07 3.08 3.09 3.1 3.11 11.9 3.15 3.17 3.2 3.21 3.51 3.54 3.58 3.61 3.66 3.69 3.74 3.8 3.84 11.9 11.9 11.9 11.9 11.9 11.9 11.9 4.46 4.34 4.26 4.36 4.46 4.58 4.7 4.82 5.39 5.54 5.72 5.66 5.71 5.93 6.23 6.42 11.9 7.02 11.9 11.9 11.9 11.9 11.9 11.9 11.9 11.9 6.18 11.9 8.62 8.61 8.57 8.77 8.77 11.9 8.77 8.78 8.79 8.82 8.81 8.82 8.89 8.87 8.72 11.9 11.9 11.9 11.9 11.9 11.9 11.9 3.72 3.74 3.77 11.9 11.9 2.79 2.64 2.55 2.42 2.34 2.26 2.18 2.1 2.03 1.98 1.92 1.87 1.82 1.77 1.73 1.69 1.66 1.62 1.6 1.56 1.54 1.51 1.49 1.46 1.44 1.41 1.4 1.38 1.36 1.35 1.33 1.32 1.3 1.3 1.29 1.28 1.27 1.26 1.25 1.24 1.24 1.23 1.23 1.22 1.22 1.22 1.21 1.21 1.21 1.21 1.21 1.21 1.22 1.22 1.22 1.23 1.23 1.24 1.24 1.25 1.26 1.27 1.27 1.29 1.3 1.31 1.32 1.33 1.34 1.33 11.9 11.9 -0.001035 -0.000620 0.383864 -0.001035 -0.000620 0.383864 1742988203.467810 zjnu-R1 0.000246"
        scan = LiDARScan(example_line)
        scans.append(scan)
        print("Using example scan provided instead.")
    
    print(f"Read {len(scans)} scans from dataset.")
    return scans

def icp_align(source_points, target_points, max_iterations=50, tolerance=1e-5):
    """
    Implements the Iterative Closest Point (ICP) algorithm for point cloud alignment.
    
    Args:
        source_points: Source point cloud (Nx2 array)
        target_points: Target point cloud (Mx2 array)
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold
        
    Returns:
        R_matrix: Rotation matrix (2x2)
        t_vector: Translation vector (2x1)
        transformed_source: Transformed source points
        error: Final mean squared error
    """
    if len(source_points) == 0 or len(target_points) == 0:
        return np.eye(2), np.zeros(2), source_points, float('inf')
    
    # Initialize transformation
    R_matrix = np.eye(2)
    t_vector = np.zeros(2)
    
    # Copy source points to avoid modifying the original
    source_points_current = source_points.copy()
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Find closest points
        closest_indices = []
        for point in source_points_current:
            distances = np.sqrt(np.sum((target_points - point)**2, axis=1))
            closest_idx = np.argmin(distances)
            closest_indices.append(closest_idx)
        
        # Get matching points
        matched_target_points = target_points[closest_indices]
        
        # Compute centroids
        source_centroid = np.mean(source_points_current, axis=0)
        target_centroid = np.mean(matched_target_points, axis=0)
        
        # Center the point clouds
        source_centered = source_points_current - source_centroid
        target_centered = matched_target_points - target_centroid
        
        # Compute covariance matrix
        H = source_centered.T @ target_centered
        
        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Calculate rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = target_centroid - R @ source_centroid
        
        # Apply transformation
        source_points_current = (R @ source_points_current.T).T + t
        
        # Compute error
        error = np.mean(np.sum((matched_target_points - source_points_current)**2, axis=1))
        
        # Check for convergence
        if abs(prev_error - error) < tolerance:
            break
        
        prev_error = error
        
        # Update transformation matrices
        R_matrix = R @ R_matrix
        t_vector = R @ t_vector + t
    
    return R_matrix, t_vector, source_points_current, error

def convert_points_to_ranges(points, reference_x, reference_y, reference_theta, angles, max_range=11.9):
    """
    Convert point cloud to range readings relative to a reference pose.
    
    Args:
        points: Array of (x, y) points in global coordinates
        reference_x, reference_y: Reference position
        reference_theta: Reference orientation
        angles: Array of angles for the range readings
        max_range: Maximum range value
        
    Returns:
        Array of range readings (exactly 180 values)
    """
    # Initialize with max range values (180 readings)
    num_angles = 180
    ranges = np.full(num_angles, max_range)
    
    # If angles array doesn't match expected length, create a new one
    if len(angles) != num_angles:
        angle_min = -np.pi/2  # -90 degrees
        angle_max = np.pi/2   # 90 degrees
        angles = np.linspace(angle_min, angle_max, num_angles)
    
    if len(points) == 0:
        return ranges
    
    # Calculate global angles for each reading angle
    global_angles = reference_theta + angles
    
    # For each point, find the closest angle and update the range
    for point in points:
        # Vector from reference to point
        dx = point[0] - reference_x
        dy = point[1] - reference_y
        
        # Calculate distance and angle to this point
        distance = np.sqrt(dx**2 + dy**2)
        point_angle = np.arctan2(dy, dx) - reference_theta
        
        # Normalize angle to be within [-pi, pi]
        point_angle = (point_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Find closest angle index
        angle_diffs = np.abs(angles - point_angle)
        closest_idx = np.argmin(angle_diffs)
        
        # Update range if closer than current value
        if distance < ranges[closest_idx]:
            ranges[closest_idx] = distance
    
    # Ensure we have exactly 180 values
    if len(ranges) > 180:
        ranges = ranges[:180]
    elif len(ranges) < 180:
        # Pad with max range values if needed
        ranges = np.pad(ranges, (0, 180 - len(ranges)), 'constant', constant_values=max_range)
    
    return ranges

def save_aligned_scan_to_file(scan, aligned_points, output_path, append=False):
    """
    Save aligned scan to file in the original format.
    
    Args:
        scan: Original scan object
        aligned_points: Aligned points array
        output_path: Path to save the file
        append: Whether to append to existing file (True) or create new file (False)
    """
    # Create a copy of the scan
    aligned_scan = scan
    
    # Convert aligned points to ranges - ensure exactly 180 values
    aligned_ranges = convert_points_to_ranges(
        aligned_points, 
        scan.x, 
        scan.y, 
        scan.theta, 
        scan.angles
    )
    
    # Verify we have exactly 180 range values
    if len(aligned_ranges) != 180:
        print(f"Warning: Range count mismatch. Expected 180, got {len(aligned_ranges)}. Adjusting...")
        if len(aligned_ranges) > 180:
            aligned_ranges = aligned_ranges[:180]
        else:
            aligned_ranges = np.pad(aligned_ranges, (0, 180 - len(aligned_ranges)), 
                                   'constant', constant_values=11.9)  # Pad with max range
    
    # Create a line in the original format
    line_parts = [scan.sensor_name, str(180)]  # Force number of readings to be 180
    
    # Add range readings
    for r in aligned_ranges:
        line_parts.append(f"{r:.2f}")
    
    # Add position data
    line_parts.extend([
        f"{scan.x:.6f}", 
        f"{scan.y:.6f}", 
        f"{scan.theta:.6f}",
        f"{scan.x:.6f}", 
        f"{scan.y:.6f}", 
        f"{scan.theta:.6f}",
        f"{scan.timestamp:.6f}",
        f"{scan.robot_id}",
        f"{scan.additional_time:.6f}"  # Include the additional time value after robot ID
    ])
    
    # Join parts with spaces
    line = " ".join(line_parts)
    
    # Verify total number of elements in the line
    expected_elements = 1 + 1 + 180 + 9  # sensor + count + ranges + position/metadata (now including additional time)
    actual_elements = len(line_parts)
    if actual_elements != expected_elements:
        print(f"Warning: Element count mismatch. Expected {expected_elements}, got {actual_elements}")
    
    # Write to file (append or create new)
    mode = 'a' if append else 'w'
    with open(output_path, mode) as f:
        f.write(line + '\n')
    
    # Print short notification
    if not append:
        print(f"Created new file: {output_path}")
    
    # Return the line for reference
    return line

def visualize_scan_matching(scan1, scan2, aligned_scan=None):
    """Visualize the results of scan matching."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot environments and original scans
    
    # Determine plot limits based on scan points
    all_points = []
    if len(scan1.points) > 0:
        all_points.extend(scan1.points)
    if len(scan2.points) > 0:
        all_points.extend(scan2.points)
    
    if all_points:
        all_points = np.array(all_points)
        min_x, max_x = np.min(all_points[:, 0]) - 1, np.max(all_points[:, 0]) + 1
        min_y, max_y = np.min(all_points[:, 1]) - 1, np.max(all_points[:, 1]) + 1
    else:
        min_x, max_x = -10, 10
        min_y, max_y = -10, 10
    
    # Draw robot positions
    ax1.plot(scan1.x, scan1.y, 'ro', markersize=8, label='Robot Pos 1')
    ax1.plot(scan2.x, scan2.y, 'bo', markersize=8, label='Robot Pos 2')
    
    # Draw scan points
    if len(scan1.points) > 0:
        ax1.scatter(scan1.points[:, 0], scan1.points[:, 1], c='r', s=10, alpha=0.5, label='Scan 1')
    if len(scan2.points) > 0:
        ax1.scatter(scan2.points[:, 0], scan2.points[:, 1], c='b', s=10, alpha=0.5, label='Scan 2')
    
    ax1.set_title('Original Scans')
    ax1.legend()
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot aligned scans
    ax2.plot(scan1.x, scan1.y, 'ro', markersize=8, label='Robot Pos 1')
    
    # Draw scan points
    if len(scan1.points) > 0:
        ax2.scatter(scan1.points[:, 0], scan1.points[:, 1], c='r', s=10, alpha=0.5, label='Scan 1')
    if aligned_scan is not None and len(aligned_scan) > 0:
        ax2.scatter(aligned_scan[:, 0], aligned_scan[:, 1], c='g', s=10, alpha=0.7, label='Aligned Scan 2')
    else:
        if len(scan2.points) > 0:
            ax2.scatter(scan2.points[:, 0], scan2.points[:, 1], c='b', s=10, alpha=0.5, label='Scan 2')
    
    ax2.set_title('Aligned Scans')
    ax2.legend()
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def visualize_map_building(scans, interval=20, use_scan_matching=True):
    """Visualize progressive map building using scan matching."""
    fig, ax = plt.subplots(figsize=(10, 10))
    all_points = []
    
    # Use the first scan as reference
    reference_scan = scans[0]
    aligned_scans = [reference_scan]
    
    # Perform scan matching for each subsequent scan
    if use_scan_matching and len(scans) > 1:
        for i in range(1, len(scans)):
            source_scan = scans[i]
            target_scan = aligned_scans[-1]
            
            # Skip if not enough points
            if len(source_scan.points) < 10 or len(target_scan.points) < 10:
                aligned_scans.append(source_scan)
                continue
            
            # Perform ICP alignment
            R, t, aligned_points, error = icp_align(source_scan.points, target_scan.points)
            
            # Create a new scan with aligned points
            aligned_scan = scans[i]
            aligned_scan.points = aligned_points
            aligned_scans.append(aligned_scan)
            
            print(f"Scan {i}: error={error:.4f}")
    else:
        aligned_scans = scans
    
    # Determine plot limits based on all scan points
    for scan in aligned_scans:
        if len(scan.points) > 0:
            all_points.extend(scan.points)
    
    if all_points:
        all_points = np.array(all_points)
        min_x, max_x = np.min(all_points[:, 0]) - 1, np.max(all_points[:, 0]) + 1
        min_y, max_y = np.min(all_points[:, 1]) - 1, np.max(all_points[:, 1]) + 1
    else:
        min_x, max_x = -10, 10
        min_y, max_y = -10, 10
    
    def update(frame):
        ax.clear()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Draw robot trajectory
        robot_xs = [scan.x for scan in aligned_scans[:frame+1]]
        robot_ys = [scan.y for scan in aligned_scans[:frame+1]]
        ax.plot(robot_xs, robot_ys, 'k-', linewidth=1, alpha=0.7)
        
        # Draw current robot position
        ax.plot(aligned_scans[frame].x, aligned_scans[frame].y, 'ro', markersize=8)
        
        # Draw accumulated map points
        for i in range(frame + 1):
            if len(aligned_scans[i].points) > 0:
                ax.scatter(aligned_scans[i].points[:, 0], aligned_scans[i].points[:, 1], 
                           c='b', s=5, alpha=0.5)
        
        ax.set_title(f'Scan {frame+1}/{len(aligned_scans)}')
        return ax,
    
    ani = FuncAnimation(fig, update, frames=len(aligned_scans), 
                        interval=interval, blit=False, repeat=False)
    
    plt.tight_layout()
    plt.show()

def run_scan_matching():
    """Run scan matching on the Intel dataset with progressive alignment."""
    # Path to the Intel dataset - updated to use the specified path
    file_path = './dataset/raw_data/raw_data_zjnu20_21_3F_short.clf'
    output_path = 'aligned_scans.clf'
    
    # Read the dataset
    max_scans = 3160  # Adjust as needed
    scans = read_intel_dataset(file_path, max_scans=max_scans)
    
    if len(scans) < 2:
        print("Not enough scans for matching.")
        return
    
    print(f"Processing {len(scans)} scans...")
    
    # Create a new output file
    open(output_path, 'w').close()
    
    # First scan is our reference and doesn't need alignment
    reference_scan = scans[0]
    print(f"Using scan 0 as reference: ({reference_scan.x:.2f}, {reference_scan.y:.2f})")
    
    # Save the reference scan
    line = save_aligned_scan_to_file(reference_scan, reference_scan.points, output_path, append=False)
    print(f"Saved reference scan (0): {line[:50]}...")
    
    # Process all consecutive pairs
    aligned_scans = [reference_scan]
    
    for i in range(1, len(scans)):
        source_scan = scans[i]
        target_scan = aligned_scans[-1]
        
        print(f"\nProcessing scan pair {i-1} -> {i}")
        print(f"Scan {i-1} position: ({target_scan.x:.2f}, {target_scan.y:.2f}, {target_scan.theta:.2f} rad)")
        print(f"Scan {i} position: ({source_scan.x:.2f}, {source_scan.y:.2f}, {source_scan.theta:.2f} rad)")
        
        # Skip if not enough points
        if len(source_scan.points) < 10 or len(target_scan.points) < 10:
            print(f"Not enough points for scan {i}, skipping...")
            aligned_scans.append(source_scan)
            # Save unaligned scan
            line = save_aligned_scan_to_file(source_scan, source_scan.points, output_path, append=True)
            print(f"Saved unaligned scan ({i}): {line[:50]}...")
            continue
        
        # Run ICP algorithm
        R, t, aligned_points, error = icp_align(source_scan.points, target_scan.points)
        
        # Calculate estimated transformation
        theta_est = np.arctan2(R[1, 0], R[0, 0])
        
        print(f"ICP Results for scan {i}:")
        print(f"Transformation: dx={t[0]:.4f}, dy={t[1]:.4f}, dtheta={theta_est:.4f} rad")
        print(f"Error: {error:.4f}")
        
        # Save the aligned scan
        line = save_aligned_scan_to_file(source_scan, aligned_points, output_path, append=True)
        print(f"Saved aligned scan ({i}): {line[:50]}...")
        
        # Visualize if desired (disabled by default to avoid too many plots)
        if i <= 2:  # Only visualize first few pairs
            visualize_scan_matching(target_scan, source_scan, aligned_points)
        
        # Create a modified scan with aligned points for next iteration
        modified_scan = copy.deepcopy(source_scan)
        modified_scan.points = aligned_points
        aligned_scans.append(modified_scan)
    
    print(f"\nProcessing complete. All aligned scans saved to {output_path}")
    
    # Visualize the full map
    visualize_map_building(aligned_scans, interval=500, use_scan_matching=False)

if __name__ == "__main__":
    run_scan_matching()