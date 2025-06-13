import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from scipy.linalg import svd

# Import utility functions if available, otherwise provide alternative implementations
try:
    from lidar_utility_functions import convert_scans_to_cartesian, parse_lidar_data
except ImportError:
    print("Warning: lidar_utility_functions.py not found. Using simplified implementations.")
    
    def convert_scans_to_cartesian(scan_ranges, angle_min, angle_max, pose, 
                             flip_x=False, flip_y=False, reverse_scan=False, flip_theta=False):
        """
        Convert scan ranges to Cartesian coordinates (simplified implementation)
        """
        num_points = len(scan_ranges)
        
        # Generate angle array for each measurement
        if reverse_scan:
            angles = np.linspace(angle_max, angle_min, num_points)
        else:
            angles = np.linspace(angle_min, angle_max, num_points)
        
        # Convert from polar to Cartesian coordinates (in robot's local frame)
        x_local = [r * math.cos(angle) for r, angle in zip(scan_ranges, angles)]
        y_local = [r * math.sin(angle) for r, angle in zip(scan_ranges, angles)]
        
        # Apply any coordinate flips to local coordinates
        if flip_x:
            x_local = [-x for x in x_local]
        if flip_y:
            y_local = [-y for y in y_local]
        
        # Get orientation angle
        theta = pose['theta'] if isinstance(pose, dict) else pose.theta
        if flip_theta:
            theta = -theta
        
        # Robot position
        robot_x = pose['x'] if isinstance(pose, dict) else pose.x
        robot_y = pose['y'] if isinstance(pose, dict) else pose.y
        
        # Transform to world coordinates based on robot pose
        x_world = [robot_x + x_l * math.cos(theta) - y_l * math.sin(theta) for x_l, y_l in zip(x_local, y_local)]
        y_world = [robot_y + x_l * math.sin(theta) + y_l * math.cos(theta) for x_l, y_l in zip(x_local, y_local)]
        
        return x_world, y_world


class FeatureExtractor:
    """
    LiDAR feature extraction for SLAM and loop closure detection.
    Extracts various geometric features from LiDAR scans to aid in scan matching and place recognition.
    """
    
    def __init__(self, debug_level=1):
        """
        Initialize the feature extractor with the specified parameters
        
        Args:
            debug_level: Level of debug information (0=None, 1=Basic, 2=Verbose)
        """
        self.debug_level = debug_level
        
        # Feature extraction parameters
        self.curvature_window = 5  # Points to consider when calculating curvature
        self.corner_threshold = 0.2  # Threshold for corner detection based on curvature
        self.plane_threshold = 0.02  # Threshold for plane detection based on curvature
        self.min_line_length = 5  # Minimum number of points to form a line segment
        self.line_threshold = 0.05  # Maximum deviation for points to be considered on a line
        self.feature_neighborhood = 3  # Size of neighborhood for feature extraction
        
        # DBSCAN clustering parameters
        self.cluster_eps = 0.2  # Maximum distance between points in a cluster
        self.min_samples = 5  # Minimum number of points to form a cluster
        
        if self.debug_level > 0:
            print("[FeatureExtractor] Initialized with parameters:")
            print(f"  - Curvature window: {self.curvature_window}")
            print(f"  - Corner threshold: {self.corner_threshold}")
            print(f"  - Plane threshold: {self.plane_threshold}")
            print(f"  - Min line length: {self.min_line_length}")
            print(f"  - Line threshold: {self.line_threshold}")
    
    def extract_features(self, scan_ranges, angle_min, angle_max, robot_pose, 
                         flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False):
        """
        Extract features from a LiDAR scan
        
        Args:
            scan_ranges: List of LiDAR distance measurements
            angle_min: Starting angle of the scan (radians)
            angle_max: Ending angle of the scan (radians)
            robot_pose: Robot pose dictionary or object with x, y, theta attributes
            flip_x, flip_y, reverse_scan, flip_theta: Orientation parameters
        
        Returns:
            Dictionary containing extracted features
        """
        # First filter out invalid range measurements
        max_range = 11.9  # Same as in lidar_utility_functions.py
        valid_ranges_mask = np.array([r < max_range and r > 0.1 for r in scan_ranges])
        valid_ranges = np.array(scan_ranges)[valid_ranges_mask]
        
        # Generate angles for valid ranges
        num_points = len(scan_ranges)
        if reverse_scan:
            angles = np.linspace(angle_max, angle_min, num_points)
        else:
            angles = np.linspace(angle_min, angle_max, num_points)
        valid_angles = angles[valid_ranges_mask]
        
        # Now convert filtered ranges to Cartesian coordinates
        scan_x, scan_y = convert_scans_to_cartesian(
            valid_ranges, angle_min, angle_max, robot_pose,
            flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
        )
        
        # Create points array
        points = np.column_stack((scan_x, scan_y))
        
        if len(points) < 10:
            if self.debug_level > 0:
                print("[FeatureExtractor] Warning: Too few valid points to extract features")
            return {"corners": [], "lines": [], "planes": [], "keypoints": []}
        
        # Extract various features
        corners = self.extract_corners(points)
        lines = self.extract_line_segments(points)
        planes = self.extract_planes(points, valid_ranges)  
        keypoints = self.extract_keypoints(points)
        
        # Calculate point normals and curvature
        normals, curvatures = self.compute_normals_and_curvature(points)
        
        return {
            "corners": corners,
            "lines": lines,
            "planes": planes,
            "keypoints": keypoints,
            "normals": normals,
            "curvatures": curvatures,
            "points": points
        }
    
    def compute_normals_and_curvature(self, points):
        """
        Compute surface normals and curvature for each point
        
        Args:
            points: Array of [x, y] points
            
        Returns:
            Tuple of (normals, curvatures) arrays
        """
        if len(points) < self.curvature_window + 1:
            # Not enough points to calculate normals
            return np.zeros((len(points), 2)), np.zeros(len(points))
        
        # Create a KD-tree for efficient neighborhood queries
        tree = KDTree(points)
        
        normals = np.zeros((len(points), 2))
        curvatures = np.zeros(len(points))
        
        for i, point in enumerate(points):
            # Find neighboring points
            indices = tree.query(point, k=self.curvature_window+1)[1]
            neighbors = points[indices]
            
            # Calculate covariance matrix for PCA
            mean = np.mean(neighbors, axis=0)
            centered = neighbors - mean
            cov = np.dot(centered.T, centered) / len(neighbors)
            
            # Perform SVD (Principal Component Analysis)
            try:
                u, s, vh = svd(cov)
                
                # Normal is the eigenvector corresponding to the smallest eigenvalue
                normal = vh[-1]
                
                # Make normal point outward from the scan center
                center_to_point = point - np.mean(points, axis=0)
                if np.dot(normal, center_to_point) < 0:
                    normal = -normal
                
                # Curvature estimation is ratio of smallest eigenvalue to sum
                if np.sum(s) > 0:
                    curvature = s[-1] / np.sum(s)
                else:
                    curvature = 0
                
                normals[i] = normal
                curvatures[i] = curvature
                
            except np.linalg.LinAlgError:
                # If SVD fails, use a default normal
                normals[i] = np.array([1, 0])
                curvatures[i] = 0
        
        return normals, curvatures
    
    def extract_corners(self, points):
        """
        Extract corner features based on curvature
        
        Args:
            points: Array of [x, y] points
            
        Returns:
            List of corner points
        """
        if len(points) < self.curvature_window + 1:
            return []
        
        _, curvatures = self.compute_normals_and_curvature(points)
        
        # Points with high curvature are corner candidates
        corner_indices = np.where(curvatures > self.corner_threshold)[0]
        
        # Apply non-maximum suppression to get local maxima
        corners = []
        for i in corner_indices:
            # Get neighborhood within feature_neighborhood points
            start_idx = max(0, i - self.feature_neighborhood)
            end_idx = min(len(curvatures), i + self.feature_neighborhood + 1)
            neighborhood = curvatures[start_idx:end_idx]
            
            # Check if current point is a local maximum
            if curvatures[i] == np.max(neighborhood):
                corners.append({
                    'position': points[i],
                    'curvature': curvatures[i],
                    'index': i
                })
        
        if self.debug_level > 0:
            print(f"[FeatureExtractor] Extracted {len(corners)} corners")
            
        return corners
    
    def extract_line_segments(self, points):
        """
        Extract line segments using iterative endpoint fitting
        
        Args:
            points: Array of [x, y] points
            
        Returns:
            List of line segments, each with start, end, and points indices
        """
        if len(points) < self.min_line_length:
            return []
        
        # Start with all points
        remaining_points = list(range(len(points)))
        line_segments = []
        
        # Iteratively extract line segments
        while len(remaining_points) >= self.min_line_length:
            # Try to fit a line to the remaining points
            best_line = self._fit_line_segment(points, remaining_points)
            
            if best_line is None or len(best_line['inliers']) < self.min_line_length:
                break
                
            # Add line segment to results
            line_segments.append({
                'start': points[best_line['inliers'][0]],
                'end': points[best_line['inliers'][-1]],
                'points': best_line['inliers'],
                'direction': best_line['direction'],
                'center': best_line['center']
            })
            
            # Remove used points
            for idx in best_line['inliers']:
                if idx in remaining_points:
                    remaining_points.remove(idx)
        
        if self.debug_level > 0:
            print(f"[FeatureExtractor] Extracted {len(line_segments)} line segments")
            
        return line_segments
    
    def _fit_line_segment(self, points, point_indices, ransac_iterations=20):
        """
        Fit a line segment to a subset of points using RANSAC
        
        Args:
            points: Array of all [x, y] points
            point_indices: Indices of points to consider
            ransac_iterations: Number of RANSAC iterations
            
        Returns:
            Dictionary with line segment parameters and inlier indices
        """
        if len(point_indices) < self.min_line_length:
            return None
            
        best_inliers = []
        best_direction = None
        best_center = None
        
        for _ in range(ransac_iterations):
            # Randomly select two points
            if len(point_indices) < 2:
                continue
                
            idx1, idx2 = np.random.choice(point_indices, 2, replace=False)
            p1, p2 = points[idx1], points[idx2]
            
            # Calculate line direction and center
            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 1e-6:  # Too close to be a line
                continue
                
            direction = direction / direction_norm
            center = (p1 + p2) / 2
            
            # Find inliers by projecting points onto the line
            inliers = []
            for idx in point_indices:
                p = points[idx]
                # Vector from center to point
                v = p - center
                # Project v onto the line direction
                proj = np.dot(v, direction) * direction
                # Calculate perpendicular distance
                dist = np.linalg.norm(v - proj)
                
                if dist < self.line_threshold:
                    inliers.append(idx)
            
            # If this line has more inliers, update the best line
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_direction = direction
                best_center = center
        
        if len(best_inliers) < self.min_line_length:
            return None
            
        # Sort inliers by their position along the line
        best_inliers = sorted(best_inliers, key=lambda idx: 
                              np.dot(points[idx] - best_center, best_direction))
        
        return {
            'inliers': best_inliers,
            'direction': best_direction,
            'center': best_center
        }
    
    def extract_planes(self, points, ranges):
        """
        Extract planar regions (in 2D these are line segments with consistent range)
        
        Args:
            points: Array of [x, y] points
            ranges: Array of range measurements
            
        Returns:
            List of planar segments, each with points, center, and normal
        """
        if len(points) < self.min_line_length:
            return []
            
        _, curvatures = self.compute_normals_and_curvature(points)
        
        # Points with low curvature are planar
        planar_indices = np.where(curvatures < self.plane_threshold)[0]
        
        if len(planar_indices) < self.min_line_length:
            return []
            
        # Cluster planar points into segments
        planar_points = points[planar_indices]
        
        # Use DBSCAN to cluster points
        if len(planar_points) >= self.min_samples:
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples).fit(planar_points)
            labels = clustering.labels_
        else:
            labels = np.array([-1] * len(planar_points))
            
        # Extract each cluster as a plane segment
        planes = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue
                
            # Get indices of points in this cluster
            cluster_mask = labels == label
            cluster_point_indices = planar_indices[cluster_mask]
            
            if len(cluster_point_indices) < self.min_line_length:
                continue
                
            cluster_points = points[cluster_point_indices]
            
            # Calculate cluster center and normal
            center = np.mean(cluster_points, axis=0)
            
            # Calculate covariance matrix for PCA
            centered = cluster_points - center
            cov = np.dot(centered.T, centered) / len(cluster_points)
            
            # Perform SVD to get the normal (eigenvector of smallest eigenvalue)
            try:
                u, s, vh = svd(cov)
                normal = vh[-1]
                
                # Make normal point outward from the origin
                if np.dot(normal, center) < 0:
                    normal = -normal
                    
                planes.append({
                    'points': cluster_point_indices,
                    'center': center,
                    'normal': normal,
                    'size': len(cluster_point_indices),
                    'extent': np.max(np.linalg.norm(cluster_points - center, axis=1))
                })
                
            except np.linalg.LinAlgError:
                # Skip if SVD fails
                continue
        
        if self.debug_level > 0:
            print(f"[FeatureExtractor] Extracted {len(planes)} planar segments")
            
        return planes
    
    def extract_keypoints(self, points):
        """
        Extract distinctive keypoints for place recognition
        
        Args:
            points: Array of [x, y] points
            
        Returns:
            List of keypoint features
        """
        # Create a KD-tree for efficient neighbor searching
        tree = KDTree(points)
        
        # Extract points that are local extrema or distinctive in some way
        keypoints = []
        
        # Find points with largest distance to their neighbors
        for i, point in enumerate(points):
            # Find closest neighbors
            distances, indices = tree.query(point, k=self.feature_neighborhood+1)
            
            # Skip the first index (self)
            distances = distances[1:]
            indices = indices[1:]
            
            # Calculate distinctiveness based on distance to neighbors
            if len(distances) > 0:
                distinctiveness = np.mean(distances)
                
                # Accept points that are more isolated
                if distinctiveness > np.percentile(distances, 75):  # Top 25% most distinctive
                    keypoints.append({
                        'position': point,
                        'index': i,
                        'distinctiveness': float(distinctiveness)
                    })
        
        # Sort keypoints by distinctiveness
        keypoints.sort(key=lambda kp: kp['distinctiveness'], reverse=True)
        
        # Limit number of keypoints to ensure balanced distribution
        max_keypoints = min(30, len(points) // 10)
        keypoints = keypoints[:max_keypoints]
        
        if self.debug_level > 0:
            print(f"[FeatureExtractor] Extracted {len(keypoints)} keypoints")
            
        return keypoints
    
    def visualize_features(self, scan_points, features, title="Extracted Features"):
        """
        Visualize the extracted features
        
        Args:
            scan_points: Array of [x, y] scan points
            features: Dictionary of extracted features
            title: Plot title
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot all scan points
        ax.scatter(scan_points[:, 0], scan_points[:, 1], c='lightgray', s=5, alpha=0.5, label='Scan Points')
        
        # Plot corner features
        if 'corners' in features and features['corners']:
            corner_points = np.array([corner['position'] for corner in features['corners']])
            ax.scatter(corner_points[:, 0], corner_points[:, 1], c='red', s=100, marker='x', 
                      label=f'Corners ({len(features["corners"])})')
        
        # Plot line segments
        if 'lines' in features and features['lines']:
            for i, line in enumerate(features['lines']):
                ax.plot([line['start'][0], line['end'][0]], 
                       [line['start'][1], line['end'][1]], 
                       'b-', linewidth=2)
                
                # Plot the center of the line segment
                ax.scatter(line['center'][0], line['center'][1], c='blue', s=30, alpha=0.7)
            
            # Add to legend once
            ax.plot([], [], 'b-', linewidth=2, label=f'Lines ({len(features["lines"])})')
        
        # Plot planar segments
        if 'planes' in features and features['planes']:
            for plane in features['planes']:
                # Get the points in this plane
                plane_points = scan_points[plane['points']]
                ax.scatter(plane_points[:, 0], plane_points[:, 1], c='green', s=20, alpha=0.7)
                
                # Plot the normal vector at the center
                normal_length = 0.5  # Length of normal vector for visualization
                normal_end = plane['center'] + normal_length * plane['normal']
                ax.arrow(plane['center'][0], plane['center'][1], 
                        normal_end[0] - plane['center'][0], normal_end[1] - plane['center'][1],
                        head_width=0.1, head_length=0.1, fc='green', ec='green')
            
            # Add to legend once
            ax.scatter([], [], c='green', s=20, label=f'Planes ({len(features["planes"])})')
        
        # Plot keypoints
        if 'keypoints' in features and features['keypoints']:
            keypoint_points = np.array([kp['position'] for kp in features['keypoints']])
            ax.scatter(keypoint_points[:, 0], keypoint_points[:, 1], c='purple', s=80, marker='*', 
                      label=f'Keypoints ({len(features["keypoints"])})')
        
        # Plot point normals (subsampled)
        if 'normals' in features and len(features['normals']) > 0:
            # Subsample for visibility
            step = max(1, len(scan_points) // 50)
            for i in range(0, len(scan_points), step):
                if i < len(features['normals']):
                    normal = features['normals'][i]
                    point = scan_points[i]
                    
                    # Scale normal for visibility
                    normal_length = 0.2
                    ax.arrow(point[0], point[1], 
                            normal[0] * normal_length, normal[1] * normal_length,
                            head_width=0.05, head_length=0.05, fc='orange', ec='orange', alpha=0.5)
        
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(title)
        ax.legend()
        
        return fig
    
    def compute_feature_descriptor(self, features, resolution=20, radius=10.0):
        """
        Compute a global feature descriptor for the scan
        
        Args:
            features: Dictionary of extracted features
            resolution: Resolution of the descriptor grid
            radius: Maximum radius to consider
            
        Returns:
            Descriptor array
        """
        # Initialize descriptor grid (polar coordinates: rings x sectors)
        descriptor = np.zeros((resolution, resolution))
        
        # Handle cases where features are missing or empty
        if 'points' not in features or len(features['points']) == 0:
            return descriptor
            
        points = features['points']
        
        # Calculate centroid of all points
        centroid = np.mean(points, axis=0)
        
        # For each point, calculate polar coordinates relative to centroid
        for point in points:
            # Vector from centroid to point
            v = point - centroid
            
            # Calculate polar coordinates
            distance = np.linalg.norm(v)
            angle = np.arctan2(v[1], v[0])
            
            # Normalize angle to [0, 2π]
            angle = (angle + 2 * np.pi) % (2 * np.pi)
            
            # Skip if distance is beyond radius
            if distance > radius:
                continue
                
            # Calculate grid indices
            ring_idx = min(int(distance / radius * resolution), resolution - 1)
            sector_idx = min(int(angle / (2 * np.pi) * resolution), resolution - 1)
            
            # Update descriptor - increment the corresponding cell
            descriptor[ring_idx, sector_idx] += 1
        
        # Normalize descriptor
        if np.sum(descriptor) > 0:
            descriptor = descriptor / np.sum(descriptor)
            
        return descriptor
    
    def match_feature_descriptors(self, descriptor1, descriptor2, method='cosine'):
        """
        Match two feature descriptors and return a similarity score
        
        Args:
            descriptor1, descriptor2: Feature descriptors to compare
            method: Matching method ('cosine' or 'euclidean')
            
        Returns:
            Similarity score [0-1] where 1 means identical
        """
        if method == 'cosine':
            # Flatten descriptors
            d1_flat = descriptor1.flatten()
            d2_flat = descriptor2.flatten()
            
            # Compute cosine similarity
            dot_product = np.dot(d1_flat, d2_flat)
            norm1 = np.linalg.norm(d1_flat)
            norm2 = np.linalg.norm(d2_flat)
            
            if norm1 == 0 or norm2 == 0:
                return 0
                
            similarity = dot_product / (norm1 * norm2)
            return similarity
            
        elif method == 'euclidean':
            # Calculate Euclidean distance
            distance = np.linalg.norm(descriptor1 - descriptor2)
            
            # Convert to similarity score (1 for identical, 0 for completely different)
            max_distance = np.sqrt(2)  # Maximum possible distance for normalized descriptors
            similarity = 1 - min(distance / max_distance, 1.0)
            return similarity
            
        else:
            raise ValueError(f"Unknown matching method: {method}")
    
    def extract_and_visualize(self, scan_data, angle_min=-math.pi/2, angle_max=math.pi/2,
                            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False):
        """
        Extract features from scan data and visualize them in one step
        
        Args:
            scan_data: Dictionary containing scan data with 'scan_ranges' and 'pose'
            angle_min, angle_max: Angle range of the scan
            flip_x, flip_y, reverse_scan, flip_theta: Orientation parameters
            
        Returns:
            Tuple of (features, figure)
        """
        # Extract features
        features = self.extract_features(
            scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
            flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
        )
        
        # Convert scan to Cartesian coordinates
        scan_x, scan_y = convert_scans_to_cartesian(
            scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
            flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
        )
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Visualize features
        fig = self.visualize_features(scan_points, features)
        
        return features, fig


def test_feature_extraction_sequence(file_path, start_index=0, num_scans=10, step=5, 
                               angle_min=-math.pi/2, angle_max=math.pi/2,
                               flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False, 
                               max_entries=300):
    """
    Test feature extraction on a sequence of scans and visualize the robot path
    
    Args:
        file_path: Path to the LiDAR data file
        start_index: Starting scan index
        num_scans: Number of scans to process
        step: Step size between scans
        angle_min, angle_max: Angle range of the scan
        flip_x, flip_y, reverse_scan, flip_theta: Orientation parameters
        max_entries: Maximum number of entries to read from file
    """
    try:
        # Import utility functions if available
        from lidar_utility_functions import read_lidar_data_from_file
        
        # Read the data from file
        parsed_data_list = read_lidar_data_from_file(file_path, max_entries=max_entries)
        
        if not parsed_data_list:
            print(f"Error: Could not read data from file {file_path}")
            return
            
        # Make sure we have enough scans
        max_index = start_index + (num_scans-1) * step
        if max_index >= len(parsed_data_list):
            print(f"Warning: Not enough scans available. Adjusting num_scans.")
            num_scans = (len(parsed_data_list) - start_index) // step
            if num_scans <= 0:
                print("Error: Not enough scans to process.")
                return
                
    except ImportError:
        print("Error: lidar_utility_functions.py not found. Cannot process multiple scans.")
        return
        
    # Initialize feature extractor
    extractor = FeatureExtractor(debug_level=1)
    
    # Create figure for visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # Initialize robot path
    robot_path_x = []
    robot_path_y = []
    
    # Process each scan
    all_features = []
    all_descriptors = []
    
    for i in range(num_scans):
        scan_index = start_index + i * step
        scan_data = parsed_data_list[scan_index]
        
        # Add to robot path
        if isinstance(scan_data['pose'], dict):
            robot_path_x.append(scan_data['pose']['x'])
            robot_path_y.append(scan_data['pose']['y'])
        else:
            robot_path_x.append(scan_data['pose'].x)
            robot_path_y.append(scan_data['pose'].y)
        
        # Extract features
        features = extractor.extract_features(
            scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
            flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
        )
        
        # Compute descriptor
        descriptor = extractor.compute_feature_descriptor(features)
        
        all_features.append(features)
        all_descriptors.append(descriptor)
        
        # Show progress
        print(f"Processed scan {scan_index} ({i+1}/{num_scans})")
    
    # Plot the robot path in the first subplot
    ax1.plot(robot_path_x, robot_path_y, 'b-', linewidth=2)
    ax1.scatter(robot_path_x, robot_path_y, c='blue', s=30)
    ax1.scatter(robot_path_x[0], robot_path_y[0], c='green', s=100, marker='*', label='Start')
    ax1.scatter(robot_path_x[-1], robot_path_y[-1], c='red', s=100, marker='*', label='End')
    
    # Set plot properties
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('Robot Path and Environment')
    ax1.legend()
    
    # Add an interactive element to view different scans
    scan_index = 0
    
    # Current scan data
    current_scan = all_features[scan_index]
    current_descriptor = all_descriptors[scan_index]
    
    # Plot the first scan's features
    scan_scatter = ax1.scatter(
        current_scan['points'][:, 0], 
        current_scan['points'][:, 1], 
        c='gray', s=5, alpha=0.5
    )
    
    # Plot corners
    if current_scan['corners']:
        corner_points = np.array([corner['position'] for corner in current_scan['corners']])
        corner_scatter = ax1.scatter(
            corner_points[:, 0], corner_points[:, 1], 
            c='red', s=50, marker='x'
        )
    else:
        corner_scatter = ax1.scatter([], [], c='red', s=50, marker='x')
    
    # Plot keypoints
    if current_scan['keypoints']:
        keypoint_points = np.array([kp['position'] for kp in current_scan['keypoints']])
        keypoint_scatter = ax1.scatter(
            keypoint_points[:, 0], keypoint_points[:, 1], 
            c='purple', s=80, marker='*'
        )
    else:
        keypoint_scatter = ax1.scatter([], [], c='purple', s=80, marker='*')
    
    # Plot current robot position
    current_pos_scatter = ax1.scatter(
        robot_path_x[scan_index], robot_path_y[scan_index],
        c='blue', s=150, marker='o', edgecolor='black'
    )
    
    # Plot descriptor in the second subplot
    descriptor_img = ax2.imshow(current_descriptor, cmap='viridis')
    ax2.set_title(f'Feature Descriptor (Scan {start_index + scan_index * step})')
    
    # Add colorbar
    plt.colorbar(descriptor_img, ax=ax2)
    
    # Create a slider for changing the scan
    plt.subplots_adjust(bottom=0.2)
    slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
    scan_slider = plt.Slider(
        slider_ax, 'Scan Index', 0, num_scans-1,
        valinit=0, valstep=1
    )
    
    # Function to update the plot when the slider changes
    def update_scan(val):
        scan_idx = int(scan_slider.val)
        scan = all_features[scan_idx]
        
        # Update scan points
        scan_scatter.set_offsets(scan['points'])
        
        # Update corners
        if scan['corners']:
            corner_points = np.array([corner['position'] for corner in scan['corners']])
            corner_scatter.set_offsets(corner_points)
        else:
            corner_scatter.set_offsets(np.zeros((0, 2)))
        
        # Update keypoints
        if scan['keypoints']:
            keypoint_points = np.array([kp['position'] for kp in scan['keypoints']])
            keypoint_scatter.set_offsets(keypoint_points)
        else:
            keypoint_scatter.set_offsets(np.zeros((0, 2)))
        
        # Update current position
        current_pos_scatter.set_offsets([[robot_path_x[scan_idx], robot_path_y[scan_idx]]])
        
        # Update descriptor
        descriptor_img.set_data(all_descriptors[scan_idx])
        ax2.set_title(f'Feature Descriptor (Scan {start_index + scan_idx * step})')
        
        fig.canvas.draw_idle()
    
    # Connect the slider to the update function
    scan_slider.on_changed(update_scan)
    
    # Add a button to visualize feature matching between consecutive scans
    match_button_ax = plt.axes([0.8, 0.1, 0.15, 0.05])
    match_button = plt.Button(match_button_ax, 'Match Features', color='lightblue')
    
    def show_matching(event):
        scan_idx = int(scan_slider.val)
        if scan_idx < num_scans - 1:
            descriptor1 = all_descriptors[scan_idx]
            descriptor2 = all_descriptors[scan_idx + 1]
            
            # Compute similarity
            similarity = extractor.match_feature_descriptors(descriptor1, descriptor2)
            
            # Create a new figure
            match_fig, (match_ax1, match_ax2, match_ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot the two descriptors
            match_ax1.imshow(descriptor1, cmap='viridis')
            match_ax1.set_title(f'Descriptor for Scan {start_index + scan_idx * step}')
            
            match_ax2.imshow(descriptor2, cmap='viridis')
            match_ax2.set_title(f'Descriptor for Scan {start_index + (scan_idx+1) * step}')
            
            # Plot the absolute difference
            diff = np.abs(descriptor1 - descriptor2)
            match_ax3.imshow(diff, cmap='hot')
            match_ax3.set_title(f'Difference (Similarity: {similarity:.4f})')
            
            plt.tight_layout()
            plt.show()
    
    match_button.on_clicked(show_matching)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return all_features, all_descriptors


if __name__ == "__main__":
    # Default settings
    lidar_data_file = "../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf"
    max_entries = 300
    
    import argparse
    parser = argparse.ArgumentParser(description='LiDAR Feature Extraction')
    parser.add_argument('--file', type=str, default=lidar_data_file, help='Path to the LiDAR data file')
    parser.add_argument('--scan', type=int, default=0, help='Index of the scan to process')
    parser.add_argument('--angle_min', type=float, default=-math.pi/2, help='Minimum scan angle (radians)')
    parser.add_argument('--angle_max', type=float, default=math.pi/2, help='Maximum scan angle (radians)')
    parser.add_argument('--flip_x', action='store_true', help='Flip the x-axis')
    parser.add_argument('--flip_y', action='store_true', help='Flip the y-axis')
    parser.add_argument('--reverse_scan', action='store_true', default=True, help='Reverse the scan direction')
    parser.add_argument('--flip_theta', action='store_true', help='Flip the orientation angle')
    parser.add_argument('--sequence', action='store_true', help='Process a sequence of scans')
    parser.add_argument('--num_scans', type=int, default=10, help='Number of scans to process in sequence mode')
    parser.add_argument('--step', type=int, default=5, help='Step size between scans in sequence mode')
    
    args = parser.parse_args()
    
    # Run the test
    if args.sequence:
        test_feature_extraction_sequence(
            args.file, args.scan,
            num_scans=args.num_scans,
            step=args.step,
            angle_min=args.angle_min, 
            angle_max=args.angle_max,
            flip_x=args.flip_x, 
            flip_y=args.flip_y, 
            reverse_scan=args.reverse_scan, 
            flip_theta=args.flip_theta,
            max_entries=max_entries
        )
    # else:
    #     test_feature_extraction(
    #         args.file, args.scan,
    #         args.angle_min, args.angle_max,
    #         args.flip_x, args.flip_y, args.reverse_scan, args.flip_theta,
    #         max_entries=max_entries
    #     )