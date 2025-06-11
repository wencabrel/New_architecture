import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import svd

from ScanMatcher import PoseEstimate
from lidar_utility_functions import convert_scans_to_cartesian


class LoopClosureDetector:
    """
    Loop closure detector using Scan Context descriptors and ICP verification
    """
    def __init__(self, distance_threshold=2.0, similarity_threshold=0.7, 
                 min_loop_size=50, descriptor_size=60):
        """
        Initialize the loop closure detector
        
        Args:
            distance_threshold: Minimum distance (meters) between poses to consider as a loop
            similarity_threshold: Minimum similarity score to accept a loop closure
            min_loop_size: Minimum number of poses between potential loop closures
            descriptor_size: Size of the scan context descriptor
        """
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.min_loop_size = min_loop_size
        self.descriptor_size = descriptor_size
        
        # Database of scan descriptors and their associated poses
        self.scan_database = []  # List of {descriptor, pose_index, pose} entries
        
        # Store scan points for verification
        self.scan_points_database = []  # List of scan point arrays
        
        # History of detected loop closures
        self.detected_loops = []  # List of {from_idx, to_idx, transform, score} entries
        
        # For visualization and debugging
        self.current_match_data = None
        
        # Performance optimization - only check some frames
        self.check_interval = 5  # Only check every 5 frames
        
    def create_scan_descriptor(self, scan_x, scan_y, robot_pose):
        """
        Create a scan context descriptor from LiDAR scan points
        
        Args:
            scan_x, scan_y: LiDAR scan points in world frame
            robot_pose: Current robot pose (PoseEstimate object)
            
        Returns:
            Scan context descriptor (2D array)
        """
        # Create points array
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Transform to local robot frame
        local_points = self._transform_to_local_frame(scan_points, robot_pose)
        
        # Create scan context descriptor
        # Divide the space around the robot into sectors and rings
        num_rings = self.descriptor_size // 2
        num_sectors = self.descriptor_size // 2
        
        # Initialize descriptor
        descriptor = np.zeros((num_rings, num_sectors))
        
        # Max range for normalization
        max_range = 15.0  # meters
        
        # Compute distances and angles to each point
        distances = np.sqrt(local_points[:, 0]**2 + local_points[:, 1]**2)
        angles = np.arctan2(local_points[:, 1], local_points[:, 0])
        
        # Normalize angles to [0, 2π]
        angles = (angles + 2*np.pi) % (2*np.pi)
        
        # Assign points to descriptor cells
        for i, (distance, angle) in enumerate(zip(distances, angles)):
            if distance > max_range:
                continue
                
            # Determine ring and sector indices
            ring_idx = min(int(distance / max_range * num_rings), num_rings - 1)
            sector_idx = min(int(angle / (2*np.pi) * num_sectors), num_sectors - 1)
            
            # Update descriptor - store the maximum distance in each cell
            descriptor[ring_idx, sector_idx] = max(descriptor[ring_idx, sector_idx], distance)
        
        # Normalize the descriptor
        with np.errstate(divide='ignore', invalid='ignore'):
            descriptor = descriptor / max_range
            descriptor = np.nan_to_num(descriptor)
            
        return descriptor

    def _transform_to_local_frame(self, points, pose):
        """
        Transform points from world frame to robot's local frame
        
        Args:
            points: Array of [x, y] points in world frame
            pose: Robot pose (PoseEstimate object)
            
        Returns:
            Array of transformed points in robot's local frame
        """
        # Extract pose components
        x, y, theta = pose.x, pose.y, pose.theta
        
        # Create rotation matrix for inverse transform
        c = math.cos(theta)
        s = math.sin(theta)
        R_inv = np.array([[c, s], [-s, c]])
        
        # Apply translation and rotation
        centered_points = points - np.array([x, y])
        local_points = np.dot(centered_points, R_inv.T)
        
        return local_points
        
    def detect_loop_closure(self, scan_x, scan_y, current_pose, current_pose_index):
        """
        Detect if the current scan closes a loop with a previous scan
        
        Args:
            scan_x, scan_y: Current LiDAR scan points
            current_pose: Current robot pose estimate
            current_pose_index: Index of the current pose in the trajectory
            
        Returns:
            True if loop closure detected, False otherwise
        """
        # Create points array for current scan
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Create descriptor for current scan
        current_descriptor = self.create_scan_descriptor(scan_x, scan_y, current_pose)
        
        # Store current scan in database
        self.scan_database.append({
            'descriptor': current_descriptor,
            'pose_index': current_pose_index,
            'pose': copy.deepcopy(current_pose)
        })
        
        # Store scan points
        self.scan_points_database.append(scan_points.copy())
        
        # Only check for loop closures after minimum trajectory length
        if current_pose_index < self.min_loop_size:
            return False
            
        # Only check periodically for performance
        if current_pose_index % self.check_interval != 0:
            return False
        
        # Find potential loop closures by comparing with database entries
        matches = []
        
        # Only compare with scans that are sufficiently far in trajectory
        for i, entry in enumerate(self.scan_database[:-self.min_loop_size]):
            # Skip if the poses are too close in trajectory
            if abs(entry['pose_index'] - current_pose_index) < self.min_loop_size:
                continue
                
            # Calculate Euclidean distance between poses
            pose_distance = math.sqrt(
                (entry['pose'].x - current_pose.x)**2 + 
                (entry['pose'].y - current_pose.y)**2
            )
            
            # Only consider poses that are physically close enough
            if pose_distance < self.distance_threshold:
                # Calculate similarity between descriptors
                similarity = self._compare_descriptors(current_descriptor, entry['descriptor'])
                
                if similarity > self.similarity_threshold:
                    matches.append({
                        'database_index': i,
                        'pose_index': entry['pose_index'],
                        'similarity': similarity,
                        'pose': entry['pose'],
                        'distance': pose_distance
                    })
        
        # If no matches found, return False
        if not matches:
            return False
            
        # Sort matches by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take the best match
        best_match = matches[0]
        
        # Verify match with ICP
        transformation, inlier_ratio = self._verify_with_icp(
            scan_points,
            self.scan_points_database[best_match['database_index']],
            best_match['pose'],
            current_pose
        )
        
        # If ICP verification passes
        if inlier_ratio > 0.5:  # At least 50% inlier points
            # Calculate relative transformation between poses
            rel_pose = self._calculate_relative_pose(best_match['pose'], current_pose)
            
            # Record loop closure
            loop_closure = {
                'from_index': current_pose_index,
                'to_index': best_match['pose_index'],
                'transform': transformation,
                'relative_pose': rel_pose,
                'score': best_match['similarity'] * inlier_ratio,
                'from_pose': copy.deepcopy(current_pose),
                'to_pose': copy.deepcopy(best_match['pose'])
            }
            
            self.detected_loops.append(loop_closure)
            
            # Store current match data for visualization
            self.current_match_data = {
                'current_descriptor': current_descriptor,
                'matched_descriptor': self.scan_database[best_match['database_index']]['descriptor'],
                'current_pose': current_pose,
                'matched_pose': best_match['pose'],
                'similarity': best_match['similarity'],
                'inlier_ratio': inlier_ratio
            }
            
            return True
        
        return False

    def _compare_descriptors(self, desc1, desc2):
        """
        Compare two scan descriptors using cosine similarity
        
        Args:
            desc1, desc2: Scan descriptors to compare
            
        Returns:
            Similarity score [0-1] where 1 means identical
        """
        # Flatten the descriptors
        flat_desc1 = desc1.flatten()
        flat_desc2 = desc2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(flat_desc1, flat_desc2)
        norm1 = np.linalg.norm(flat_desc1)
        norm2 = np.linalg.norm(flat_desc2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        similarity = dot_product / (norm1 * norm2)
        
        return similarity

    def _verify_with_icp(self, scan_points1, scan_points2, pose1, pose2):
        """
        Verify loop closure candidate with ICP registration
        
        Args:
            scan_points1, scan_points2: Scan points to compare
            pose1, pose2: Poses associated with the scans
            
        Returns:
            Tuple of (transformation matrix, inlier ratio)
        """
        # Transform scan_points1 to local frame of pose1
        local_points1 = self._transform_to_local_frame(scan_points1, pose1)
        
        # Transform scan_points2 to local frame of pose2
        local_points2 = self._transform_to_local_frame(scan_points2, pose2)
        
        # Randomly downsample points to speed up ICP
        if len(local_points1) > 200:
            indices = np.random.choice(len(local_points1), 200, replace=False)
            local_points1 = local_points1[indices]
            
        if len(local_points2) > 200:
            indices = np.random.choice(len(local_points2), 200, replace=False)
            local_points2 = local_points2[indices]
        
        # Run ICP
        transformation, distances = self._icp(local_points1, local_points2, max_iterations=20)
        
        # Count inliers
        inlier_threshold = 0.2  # 20 cm threshold
        inliers = np.sum(distances < inlier_threshold)
        inlier_ratio = inliers / len(distances) if len(distances) > 0 else 0
        
        return transformation, inlier_ratio

    def _icp(self, source, target, max_iterations=20, tolerance=0.001):
        """
        Iterative Closest Point algorithm for aligning point clouds
        
        Args:
            source: Source point cloud [N, 2]
            target: Target point cloud [M, 2]
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (transformation matrix, distances)
        """
        # Initialize transformation
        transformation = np.eye(3)
        
        # Create copy of source points
        source_points = source.copy()
        
        # Create nearest neighbors structures for target
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(target)
        
        # ICP loop
        for iteration in range(max_iterations):
            # Find nearest neighbors
            distances, indices = nn.kneighbors(source_points)
            distances = distances.flatten()
            
            # Get corresponding points
            target_points = target[indices.flatten()]
            
            # Compute centroids
            source_centroid = np.mean(source_points, axis=0)
            target_centroid = np.mean(target_points, axis=0)
            
            # Center the point clouds
            source_centered = source_points - source_centroid
            target_centered = target_points - target_centroid
            
            # Compute covariance matrix
            H = np.dot(source_centered.T, target_centered)
            
            # Singular Value Decomposition
            U, S, Vt = svd(H)
            
            # Rotation matrix
            R = np.dot(Vt.T, U.T)
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            # Translation vector
            t = target_centroid - np.dot(source_centroid, R.T)
            
            # Transform source points
            source_points = np.dot(source_points, R.T) + t
            
            # Update transformation matrix
            current_transformation = np.eye(3)
            current_transformation[:2, :2] = R
            current_transformation[:2, 2] = t
            transformation = np.dot(current_transformation, transformation)
            
            # Check for convergence
            mean_error = np.mean(distances)
            if mean_error < tolerance:
                break
        
        return transformation, distances

    def _calculate_relative_pose(self, from_pose, to_pose):
        """
        Calculate the relative pose between two poses
        
        Args:
            from_pose: First pose (PoseEstimate)
            to_pose: Second pose (PoseEstimate)
            
        Returns:
            Relative pose as PoseEstimate
        """
        # Calculate the difference in position
        dx = to_pose.x - from_pose.x
        dy = to_pose.y - from_pose.y
        
        # Rotate the difference to the first pose's frame
        c = math.cos(from_pose.theta)
        s = math.sin(from_pose.theta)
        
        x_rel = c * dx + s * dy
        y_rel = -s * dx + c * dy
        
        # Difference in orientation
        dtheta = to_pose.theta - from_pose.theta
        # Normalize to [-π, π]
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
        
        return PoseEstimate(x_rel, y_rel, dtheta)

    def visualize_loop_closures(self, trajectory, map_obj=None, ax=None):
        """
        Visualize detected loop closures
        
        Args:
            trajectory: List of PoseEstimate objects
            map_obj: OccupancyGrid object (optional)
            ax: Matplotlib axis (optional)
            
        Returns:
            Matplotlib axis with the plot
        """
        if not self.detected_loops:
            print("No loop closures to visualize.")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the map if we have one
        if map_obj is not None:
            # Custom colormap
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                map_obj.get_grid_for_display(),
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-map_obj.width/2, map_obj.width/2, -map_obj.height/2, map_obj.height/2]
            )
        
        # Plot the trajectory
        trajectory_x = [pose.x for pose in trajectory]
        trajectory_y = [pose.y for pose in trajectory]
        ax.plot(trajectory_x, trajectory_y, 'g-', linewidth=2, label='Robot Trajectory')
        
        # Plot loop closures
        for loop in self.detected_loops:
            from_pose = loop['from_pose']
            to_pose = loop['to_pose']
            
            # Plot the loop closure connection
            ax.plot([from_pose.x, to_pose.x], [from_pose.y, to_pose.y], 'b-', linewidth=1.5, alpha=0.7)
            
            # Highlight the loop closure points
            ax.scatter(from_pose.x, from_pose.y, c='blue', s=80, alpha=0.7, marker='o')
            ax.scatter(to_pose.x, to_pose.y, c='red', s=80, alpha=0.7, marker='o')
        
        # Finish the plot
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'Loop Closure Visualization - {len(self.detected_loops)} Detected')
        ax.legend(loc='upper right')
        
        return ax
        
    def visualize_descriptors(self, descriptor1=None, descriptor2=None):
        """
        Visualize scan context descriptors
        
        Args:
            descriptor1, descriptor2: Descriptors to compare (optional)
            
        Returns:
            Matplotlib figure
        """
        if descriptor1 is None and self.current_match_data is not None:
            descriptor1 = self.current_match_data['current_descriptor']
            descriptor2 = self.current_match_data['matched_descriptor']
        
        if descriptor1 is None:
            print("No descriptors to visualize.")
            return None
        
        if descriptor2 is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot first descriptor
            im1 = ax1.imshow(descriptor1, cmap='viridis', origin='lower')
            ax1.set_title('Current Scan Descriptor')
            plt.colorbar(im1, ax=ax1)
            
            # Plot second descriptor
            im2 = ax2.imshow(descriptor2, cmap='viridis', origin='lower')
            ax2.set_title('Matched Scan Descriptor')
            plt.colorbar(im2, ax=ax2)
            
            # Add similarity score if available
            if self.current_match_data is not None:
                similarity = self.current_match_data['similarity']
                fig.suptitle(f'Descriptor Comparison - Similarity: {similarity:.4f}')
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot the descriptor
            im = ax.imshow(descriptor1, cmap='viridis', origin='lower')
            ax.set_title('Scan Descriptor')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        return fig


class PoseGraphOptimizer:
    """
    Optimizer for pose graphs with odometry and loop closure constraints
    """
    def __init__(self):
        """
        Initialize the pose graph optimizer
        """
        self.poses = []  # List of PoseEstimate objects
        self.constraints = []  # List of constraints between poses
        self.original_poses = []  # Backup of original poses
        
    def add_pose(self, pose):
        """
        Add a pose to the graph
        
        Args:
            pose: PoseEstimate object
        
        Returns:
            Index of the added pose
        """
        self.poses.append(copy.deepcopy(pose))
        self.original_poses.append(copy.deepcopy(pose))
        return len(self.poses) - 1
        
    def add_odometry_constraint(self, from_idx, to_idx, relative_pose, information_matrix=None):
        """
        Add an odometry constraint between two poses
        
        Args:
            from_idx: Index of the first pose
            to_idx: Index of the second pose
            relative_pose: Relative transformation between poses
            information_matrix: Certainty of the constraint (optional)
        """
        if information_matrix is None:
            # Default information matrix (identity)
            information_matrix = np.eye(3)
        
        constraint = {
            'type': 'odometry',
            'from_idx': from_idx,
            'to_idx': to_idx,
            'relative_pose': copy.deepcopy(relative_pose),
            'information': information_matrix
        }
        
        self.constraints.append(constraint)
        
    def add_loop_closure_constraint(self, from_idx, to_idx, relative_pose, information_matrix=None):
        """
        Add a loop closure constraint between two poses
        
        Args:
            from_idx: Index of the first pose
            to_idx: Index of the second pose
            relative_pose: Relative transformation between poses
            information_matrix: Certainty of the constraint (optional)
        """
        if information_matrix is None:
            # Default information matrix, with lower certainty than odometry
            information_matrix = np.eye(3) * 0.5
        
        constraint = {
            'type': 'loop_closure',
            'from_idx': from_idx,
            'to_idx': to_idx,
            'relative_pose': copy.deepcopy(relative_pose),
            'information': information_matrix
        }
        
        self.constraints.append(constraint)
    
    def optimize(self, max_iterations=20):
        """
        Optimize the pose graph
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Optimized poses
        """
        # Keep a copy of original poses for comparison
        self.original_poses = [pose.copy() for pose in self.poses]
        
        # Identify loop closure constraints
        loop_closures = [c for c in self.constraints if c['type'] == 'loop_closure']
        
        if not loop_closures:
            print("No loop closures to optimize.")
            return self.poses
            
        print(f"Optimizing pose graph with {len(loop_closures)} loop closures...")
        
        # For each iteration
        for iteration in range(max_iterations):
            max_error = 0.0
            
            # Process each loop closure constraint
            for constraint in loop_closures:
                from_idx = constraint['from_idx']
                to_idx = constraint['to_idx']
                
                # Get the poses
                from_pose = self.poses[from_idx]
                to_pose = self.poses[to_idx]
                
                # Calculate the expected relative pose
                expected_rel_pose = self._calculate_relative_pose(from_pose, to_pose)
                constraint_rel_pose = constraint['relative_pose']
                
                # Calculate the error
                error = self._calculate_pose_error(expected_rel_pose, constraint_rel_pose)
                
                # Calculate error magnitude
                error_magnitude = np.linalg.norm(error)
                max_error = max(max_error, error_magnitude)
                
                # If error is small, continue
                if error_magnitude < 1e-6:
                    continue
                    
                # Distribute the error along the trajectory
                self._distribute_error(from_idx, to_idx, error)
            
            # Check for convergence
            if max_error < 0.01:
                print(f"Pose graph optimization converged after {iteration+1} iterations.")
                break
                
            # Progress indicator
            if (iteration + 1) % 5 == 0:
                print(f"  Iteration {iteration+1}/{max_iterations}, max error: {max_error:.6f}")
        
        return self.poses
    
    def _calculate_relative_pose(self, from_pose, to_pose):
        """
        Calculate the relative pose between two poses
        
        Args:
            from_pose: First pose (PoseEstimate)
            to_pose: Second pose (PoseEstimate)
            
        Returns:
            Relative pose as a PoseEstimate
        """
        # Calculate the difference in position
        dx = to_pose.x - from_pose.x
        dy = to_pose.y - from_pose.y
        
        # Rotate the difference to the first pose's frame
        c = math.cos(from_pose.theta)
        s = math.sin(from_pose.theta)
        
        x_rel = c * dx + s * dy
        y_rel = -s * dx + c * dy
        
        # Difference in orientation
        dtheta = to_pose.theta - from_pose.theta
        # Normalize to [-π, π]
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
        
        return PoseEstimate(x_rel, y_rel, dtheta)
    
    def _calculate_pose_error(self, pose1, pose2):
        """
        Calculate the error between two poses
        
        Args:
            pose1, pose2: Poses to compare (PoseEstimate)
            
        Returns:
            Error vector [dx, dy, dtheta]
        """
        dx = pose1.x - pose2.x
        dy = pose1.y - pose2.y
        dtheta = pose1.theta - pose2.theta
        
        # Normalize angle difference
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
        
        return np.array([dx, dy, dtheta])
    
    def _distribute_error(self, from_idx, to_idx, error):
        """
        Distribute error along the trajectory between two poses
        
        Args:
            from_idx: Start pose index
            to_idx: End pose index
            error: Error to distribute [dx, dy, dtheta]
        """
        # Ensure from_idx is less than to_idx
        if from_idx > to_idx:
            from_idx, to_idx = to_idx, from_idx
            error = -error
        
        # Number of poses to distribute the error over
        num_poses = to_idx - from_idx + 1
        
        # Calculate error per pose (proportional to distance from from_idx)
        for i in range(from_idx, to_idx + 1):
            # Calculate interpolation factor (0 at from_idx, 1 at to_idx)
            factor = (i - from_idx) / num_poses
            
            # Apply weighted correction
            self.poses[i].x += error[0] * factor
            self.poses[i].y += error[1] * factor
            self.poses[i].theta += error[2] * factor
            
            # Normalize angle
            self.poses[i].theta = (self.poses[i].theta + math.pi) % (2 * math.pi) - math.pi
    
    def visualize_optimization(self, map_obj=None, ax=None):
        """
        Visualize the effect of optimization on the trajectory
        
        Args:
            map_obj: OccupancyGrid object (optional)
            ax: Matplotlib axis (optional)
            
        Returns:
            Matplotlib axis with the plot
        """
        if not self.original_poses or not self.poses:
            print("No poses to visualize.")
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the map if we have one
        if map_obj is not None:
            # Custom colormap
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                map_obj.get_grid_for_display(),
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-map_obj.width/2, map_obj.width/2, -map_obj.height/2, map_obj.height/2]
            )
        
        # Plot the original trajectory
        original_x = [pose.x for pose in self.original_poses]
        original_y = [pose.y for pose in self.original_poses]
        ax.plot(original_x, original_y, 'r--', linewidth=1.5, alpha=0.7, label='Original Trajectory')
        
        # Plot the optimized trajectory
        optimized_x = [pose.x for pose in self.poses]
        optimized_y = [pose.y for pose in self.poses]
        ax.plot(optimized_x, optimized_y, 'g-', linewidth=2, label='Optimized Trajectory')
        
        # Plot the loop closure constraints
        loop_closures = [c for c in self.constraints if c['type'] == 'loop_closure']
        for constraint in loop_closures:
            from_idx = constraint['from_idx']
            to_idx = constraint['to_idx']
            
            # Get the poses
            from_pose = self.poses[from_idx]
            to_pose = self.poses[to_idx]
            
            # Plot the loop closure connection
            ax.plot([from_pose.x, to_pose.x], [from_pose.y, to_pose.y], 'b-', linewidth=1, alpha=0.5)
            
            # Highlight the loop closure points
            ax.scatter(from_pose.x, from_pose.y, c='blue', s=50, alpha=0.7, marker='o')
            ax.scatter(to_pose.x, to_pose.y, c='red', s=50, alpha=0.7, marker='o')
        
        # Finish the plot
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Trajectory Optimization')
        ax.legend(loc='upper right')
        
        return ax


def update_map_with_optimized_poses(occupancy_grid, lidar_data, angle_min, angle_max, 
                                   optimized_trajectory, flip_x=False, flip_y=False, 
                                   reverse_scan=True, flip_theta=False):
    """
    Update the occupancy grid with optimized poses after loop closure
    
    Args:
        occupancy_grid: OccupancyGrid object
        lidar_data: List of LiDAR data dictionaries
        angle_min, angle_max: LiDAR scan angle range
        optimized_trajectory: List of optimized PoseEstimate objects
        flip_x, flip_y, reverse_scan, flip_theta: Orientation parameters
    """
    from lidar_utility_functions import convert_scans_to_cartesian
    
    print("Updating map with optimized poses after loop closure...")
    
    # Reset the occupancy grid
    occupancy_grid.grid = np.ones((occupancy_grid.grid_height, occupancy_grid.grid_width)) * 0.5
    occupancy_grid.log_odds_grid = np.zeros((occupancy_grid.grid_height, occupancy_grid.grid_width))
    
    # Make sure we have the same number of poses and lidar data
    num_frames = min(len(lidar_data), len(optimized_trajectory))
    
    # Update the map with each scan using optimized poses
    for i in range(num_frames):
        # Get data and pose
        data = lidar_data[i]
        pose = optimized_trajectory[i]
        
        # Convert scan to Cartesian coordinates
        scan_x, scan_y = convert_scans_to_cartesian(
            data['scan_ranges'], angle_min, angle_max, data['pose'],
            flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
        )
        
        # Update grid with optimized pose
        try:
            # Use update_grid_with_turn_handling if available
            if hasattr(occupancy_grid, 'update_grid_with_turn_handling'):
                occupancy_grid.update_grid_with_turn_handling(
                    pose.x, pose.y, pose.theta, scan_x, scan_y, data['scan_ranges']
                )
            else:
                # Fall back to standard update_grid
                occupancy_grid.update_grid(pose.x, pose.y, scan_x, scan_y)
                
            # Progress indicator
            if i % 20 == 0:
                print(f"  Updated {i}/{num_frames} frames...")
                
        except Exception as e:
            print(f"Error updating map at frame {i}: {e}")
    
    print("Map update complete!")
    return occupancy_grid


def integrate_with_scan_matcher(scan_matcher_obj, loop_closure_enabled=True):
    """
    Integrate loop closure with the ScanMatcher class
    
    Args:
        scan_matcher_obj: ImprovedScanMatchingLocalization object
        loop_closure_enabled: Whether to enable loop closure
        
    Returns:
        Updated scan_matcher_obj with loop closure capabilities
    """
    # Add loop closure detector to scan matcher
    if loop_closure_enabled:
        scan_matcher_obj.loop_detector = LoopClosureDetector()
        scan_matcher_obj.pose_graph = PoseGraphOptimizer()
        scan_matcher_obj.enable_loop_closure = True
        scan_matcher_obj.loop_closures_detected = 0
        scan_matcher_obj.is_optimized = False
        scan_matcher_obj.original_trajectory = []
        
        print("[ScanMatcher] Loop closure detection enabled")
    
    return scan_matcher_obj


def process_loop_closure(scan_matcher_obj, lidar_data, map_obj, angle_min=-math.pi/2, angle_max=math.pi/2,
                        flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False):
    """
    Process loop closures and optimize the trajectory
    
    Args:
        scan_matcher_obj: ImprovedScanMatchingLocalization object with loop_detector
        lidar_data: List of LiDAR data dictionaries
        map_obj: OccupancyGrid object
        angle_min, angle_max: LiDAR scan angle range
        flip_x, flip_y, reverse_scan, flip_theta: Orientation parameters
        
    Returns:
        Tuple of (optimized_trajectory, updated_map)
    """
    if not hasattr(scan_matcher_obj, 'loop_detector') or not scan_matcher_obj.enable_loop_closure:
        print("Loop closure is not enabled on this scan matcher object.")
        return scan_matcher_obj.trajectory, map_obj
    
    # Extract current trajectory
    trajectory = scan_matcher_obj.trajectory
    
    # Loop through each frame and detect loop closures
    for i, (pose, data) in enumerate(zip(trajectory, lidar_data)):
        if i % 5 == 0:  # Check every 5 frames for performance
            # Convert scan to Cartesian coordinates
            scan_x, scan_y = convert_scans_to_cartesian(
                data['scan_ranges'], angle_min, angle_max, data['pose'],
                flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
            )
            
            # Detect loop closure
            loop_detected = scan_matcher_obj.loop_detector.detect_loop_closure(
                scan_x, scan_y, pose, i
            )
            
            if loop_detected:
                # Get the latest loop closure
                latest_loop = scan_matcher_obj.loop_detector.detected_loops[-1]
                
                print(f"Loop closure detected between frames {latest_loop['from_index']} and {latest_loop['to_index']} with score {latest_loop['score']:.4f}")
                
                # Add loop closure constraint to pose graph
                scan_matcher_obj.pose_graph.add_loop_closure_constraint(
                    latest_loop['to_index'],
                    latest_loop['from_index'],
                    latest_loop['relative_pose']
                )
                
                scan_matcher_obj.loop_closures_detected += 1
    
    # If we detected enough loop closures, optimize the trajectory
    min_loops_for_optimization = 1  # Can be increased for stability
    if scan_matcher_obj.loop_closures_detected >= min_loops_for_optimization:
        print(f"Performing global optimization with {scan_matcher_obj.loop_closures_detected} loop closures...")
        
        # Initialize pose graph with all poses
        for i, pose in enumerate(trajectory):
            scan_matcher_obj.pose_graph.add_pose(pose)
            
            # Add odometry constraints between consecutive poses
            if i > 0:
                relative_pose = scan_matcher_obj.loop_detector._calculate_relative_pose(
                    trajectory[i-1], trajectory[i]
                )
                scan_matcher_obj.pose_graph.add_odometry_constraint(i-1, i, relative_pose)
        
        # Store original trajectory for comparison
        scan_matcher_obj.original_trajectory = copy.deepcopy(trajectory)
        
        # Optimize the pose graph
        optimized_poses = scan_matcher_obj.pose_graph.optimize()
        
        # Update trajectory with optimized poses
        scan_matcher_obj.trajectory = optimized_poses
        
        # Update the map with optimized poses
        updated_map = update_map_with_optimized_poses(
            map_obj, lidar_data, angle_min, angle_max,
            optimized_poses, flip_x, flip_y, reverse_scan, flip_theta
        )
        
        scan_matcher_obj.is_optimized = True
        
        print("Global optimization and map update completed!")
        
        return optimized_poses, updated_map
    else:
        print(f"Not enough loop closures detected ({scan_matcher_obj.loop_closures_detected}/{min_loops_for_optimization} required). Skipping optimization.")
        return trajectory, map_obj


# Example usage functions
def visualize_loop_closure_results(scan_matcher_obj, occupancy_grid):
    """
    Visualize loop closure detection and optimization results
    """
    if not hasattr(scan_matcher_obj, 'loop_detector') or not scan_matcher_obj.enable_loop_closure:
        print("Loop closure is not enabled on this scan matcher object.")
        return
    
    # Only plot if we have loop closures detected
    if not scan_matcher_obj.loop_detector.detected_loops:
        print("No loop closures detected to visualize.")
        return
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    
    # Trajectory comparison (original vs. optimized)
    ax1 = fig.add_subplot(121)
    scan_matcher_obj.pose_graph.visualize_optimization(occupancy_grid, ax1)
    
    # Loop closure detection visualization
    ax2 = fig.add_subplot(122)
    scan_matcher_obj.loop_detector.visualize_loop_closures(
        scan_matcher_obj.trajectory, occupancy_grid, ax2
    )
    
    plt.tight_layout()
    plt.show()
    
    # If there's at least one loop closure, also visualize descriptors
    if scan_matcher_obj.loop_detector.current_match_data is not None:
        scan_matcher_obj.loop_detector.visualize_descriptors()