"""
Optimized Scan Matcher Module

This module provides an efficient scan matching implementation for LiDAR SLAM,
based on an optimized ICP algorithm for aligning consecutive scans.
"""

import numpy as np
from enum import Enum
import time

class MatchingAlgorithm(Enum):
    """Enumeration of available scan matching algorithms"""
    ICP = "icp"  # Iterative Closest Point
    FAST_ICP = "fast_icp"  # Optimized ICP version
    PSM = "psm"  # Polar Scan Matching (placeholder)

class MatchResult:
    """Class to store and manage scan matching results"""
    
    def __init__(self, transform_matrix, translation, rotation, fitness_score, inlier_rmse, iterations, success=True, message=""):
        """
        Initialize a match result
        
        Args:
            transform_matrix (np.ndarray): 3x3 homogeneous transformation matrix
            translation (tuple): (x, y) translation components
            rotation (float): Rotation angle in radians
            fitness_score (float): Measure of alignment quality (higher is better)
            inlier_rmse (float): Root mean square error of inlier point correspondences
            iterations (int): Number of iterations performed
            success (bool): Whether the matching was successful
            message (str): Additional information about the matching process
        """
        self.transform_matrix = transform_matrix
        self.translation = translation
        self.rotation = rotation
        self.fitness_score = fitness_score
        self.inlier_rmse = inlier_rmse
        self.iterations = iterations
        self.success = success
        self.message = message
        self.computation_time = 0  # Will be set by the caller
    
    def __str__(self):
        """String representation of the match result"""
        return (
            f"Match Result:\n"
            f"  Success: {self.success}\n"
            f"  Translation: ({self.translation[0]:.4f}, {self.translation[1]:.4f})\n"
            f"  Rotation: {np.degrees(self.rotation):.4f} degrees\n"
            f"  Fitness Score: {self.fitness_score:.4f}\n"
            f"  Inlier RMSE: {self.inlier_rmse:.4f}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Computation Time: {self.computation_time:.4f} seconds\n"
            f"  Message: {self.message}"
        )
    
    def get_transform_matrix(self):
        """Get the transformation matrix"""
        return self.transform_matrix.copy()
    
    def get_inverse_transform(self):
        """Get the inverse transformation matrix"""
        return np.linalg.inv(self.transform_matrix)
    
    def apply_transform(self, points):
        """
        Apply the transformation to a set of points
        
        Args:
            points (np.ndarray): Array of shape (N, 2) containing points to transform
            
        Returns:
            np.ndarray: Transformed points
        """
        # Create homogeneous coordinates
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Apply transformation
        transformed_points = np.dot(homogeneous_points, self.transform_matrix.T)
        
        # Return to 2D coordinates
        return transformed_points[:, :2]
    
    def is_reliable(self, min_fitness=0.7, max_rmse=0.2):
        """
        Check if the match result is reliable based on quality metrics
        
        Args:
            min_fitness (float): Minimum acceptable fitness score
            max_rmse (float): Maximum acceptable RMSE
            
        Returns:
            bool: True if the result is considered reliable
        """
        return (self.success and 
                self.fitness_score >= min_fitness and 
                self.inlier_rmse <= max_rmse)


class OptimizedScanMatcher:
    """Class for efficient scan matching algorithms"""
    
    def __init__(self, algorithm=MatchingAlgorithm.FAST_ICP):
        """
        Initialize the scan matcher
        
        Args:
            algorithm (MatchingAlgorithm): Algorithm to use for scan matching
        """
        self.algorithm = algorithm
        self.debug_mode = False
        self.last_source_points = None
        self.last_target_points = None
        self.last_transform = None
        
        # Performance optimization parameters
        self.downsample_factor = 2  # Use every Nth point (higher = faster but less accurate)
        self.max_iterations = 20     # Maximum iterations for convergence
        self.convergence_tolerance = 1e-4  # Threshold for convergence
        self.max_correspondence_distance = 1.0  # Maximum distance for point correspondences
        self.min_points_required = 10  # Minimum points required for matching
    
    def set_optimization_params(self, downsample_factor=None, max_iterations=None, 
                               tolerance=None, max_correspondence_distance=None):
        """
        Configure optimization parameters
        
        Args:
            downsample_factor (int): Use every Nth point
            max_iterations (int): Maximum iterations for convergence
            tolerance (float): Convergence threshold
            max_correspondence_distance (float): Maximum correspondence distance
        """
        if downsample_factor is not None:
            self.downsample_factor = max(1, int(downsample_factor))
        if max_iterations is not None:
            self.max_iterations = max(5, int(max_iterations))
        if tolerance is not None:
            self.convergence_tolerance = max(1e-6, float(tolerance))
        if max_correspondence_distance is not None:
            self.max_correspondence_distance = max(0.1, float(max_correspondence_distance))
    
    def downsample_points(self, points):
        """
        Downsample point cloud for faster processing
        
        Args:
            points (np.ndarray): Array of shape (N, 2) containing points
            
        Returns:
            np.ndarray: Downsampled points
        """
        if len(points) <= self.min_points_required:
            return points  # Don't downsample if already too few points
        
        # Simple stride-based downsampling
        return points[::self.downsample_factor]
    
    def match_scans(self, source_points, target_points, initial_guess=None):
        """
        Match two scans and find the transformation between them
        
        Args:
            source_points (np.ndarray): Array of shape (N, 2) containing the source scan points
            target_points (np.ndarray): Array of shape (M, 2) containing the target scan points
            initial_guess (tuple): Optional (translation_x, translation_y, rotation) initial guess
            
        Returns:
            MatchResult: Result of the matching process
        """
        # Ensure we have numpy arrays
        source_points = np.asarray(source_points)
        target_points = np.asarray(target_points)
        
        # Check if we have enough points
        if (len(source_points) < self.min_points_required or 
            len(target_points) < self.min_points_required):
            return MatchResult(
                np.eye(3), (0, 0), 0, 0, float('inf'), 0, 
                False, "Not enough points for matching"
            )
        
        # Store for visualization
        self.last_source_points = source_points.copy()
        self.last_target_points = target_points.copy()
        
        # Downsample points for faster processing if needed
        if self.downsample_factor > 1:
            source_points = self.downsample_points(source_points)
            target_points = self.downsample_points(target_points)
        
        # Dispatch to appropriate algorithm
        start_time = time.time()
        if self.algorithm == MatchingAlgorithm.ICP or self.algorithm == MatchingAlgorithm.FAST_ICP:
            result = self._fast_icp_match(source_points, target_points, initial_guess)
        elif self.algorithm == MatchingAlgorithm.PSM:
            result = self._psm_match(source_points, target_points, initial_guess)
        else:
            result = MatchResult(
                np.eye(3), (0, 0), 0, 0, float('inf'), 0, 
                False, f"Unknown algorithm: {self.algorithm}"
            )
        
        # Record computation time
        result.computation_time = time.time() - start_time
        
        # Store for visualization
        self.last_transform = result.transform_matrix
        
        return result
    
    def _fast_icp_match(self, source_points, target_points, initial_guess=None):
        """
        Optimized Iterative Closest Point (ICP) scan matching algorithm.
        Based on the implementation provided.
        
        Args:
            source_points (np.ndarray): Array of shape (N, 2) containing the source scan points
            target_points (np.ndarray): Array of shape (M, 2) containing the target scan points
            initial_guess (tuple): Optional (translation_x, translation_y, rotation) initial guess
            
        Returns:
            MatchResult: Result of the matching process
        """
        # Initialize transformation matrix
        current_transform = np.eye(3)
        
        # Apply initial guess if provided
        if initial_guess is not None:
            tx, ty, theta = initial_guess
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            
            # Create 2D rotation matrix
            R_init = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])
            
            # Apply initial rotation and translation to source points
            source_points_current = (R_init @ source_points.T).T + np.array([tx, ty])
        else:
            source_points_current = source_points.copy()
        
        # Variables to track convergence
        prev_error = float('inf')
        iteration = 0
        
        # ICP loop
        for iteration in range(self.max_iterations):
            # Find closest points
            closest_indices = []
            corresponding_distances = []
            
            for point in source_points_current:
                distances = np.sqrt(np.sum((target_points - point)**2, axis=1))
                closest_idx = np.argmin(distances)
                min_distance = distances[closest_idx]
                
                # Only use correspondences within max distance
                if min_distance <= self.max_correspondence_distance:
                    closest_indices.append(closest_idx)
                    corresponding_distances.append(min_distance)
            
            # Skip if not enough correspondences
            if len(closest_indices) < self.min_points_required:
                return MatchResult(
                    current_transform, 
                    (current_transform[0, 2], current_transform[1, 2]),
                    np.arctan2(current_transform[1, 0], current_transform[0, 0]),
                    0, float('inf'), iteration, 
                    False, "Not enough correspondences found"
                )
            
            # Get matching points
            matched_target_points = target_points[closest_indices]
            matched_source_points = source_points_current[list(range(len(closest_indices)))]
            
            # Compute centroids
            source_centroid = np.mean(matched_source_points, axis=0)
            target_centroid = np.mean(matched_target_points, axis=0)
            
            # Center the point clouds
            source_centered = matched_source_points - source_centroid
            target_centered = matched_target_points - target_centroid
            
            # Compute covariance matrix
            H = source_centered.T @ target_centered
            
            # Singular Value Decomposition
            try:
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
                error = np.mean(np.square(corresponding_distances))
                
                # Update iteration transform
                iteration_transform = np.eye(3)
                iteration_transform[:2, :2] = R
                iteration_transform[0, 2] = t[0]
                iteration_transform[1, 2] = t[1]
                
                # Update current transformation
                current_transform = iteration_transform @ current_transform
                
                # Check for convergence
                if abs(prev_error - error) < self.convergence_tolerance:
                    break
                
                prev_error = error
                
            except np.linalg.LinAlgError:
                # SVD computation failed
                return MatchResult(
                    current_transform, 
                    (current_transform[0, 2], current_transform[1, 2]),
                    np.arctan2(current_transform[1, 0], current_transform[0, 0]),
                    0, float('inf'), iteration, 
                    False, "SVD computation failed"
                )
        
        # Calculate final metrics
        tx, ty = current_transform[0, 2], current_transform[1, 2]
        theta = np.arctan2(current_transform[1, 0], current_transform[0, 0])
        
        # Calculate fitness score (0-1, higher is better)
        # Proportion of inlier correspondences
        inlier_proportion = len(closest_indices) / len(source_points)
        error_score = 1.0 / (1.0 + prev_error) if prev_error != float('inf') else 0
        fitness_score = 0.5 * inlier_proportion + 0.5 * error_score
        
        # Calculate RMSE for inliers
        inlier_rmse = np.sqrt(prev_error) if prev_error != float('inf') else float('inf')
        
        return MatchResult(
            current_transform, (tx, ty), theta, 
            fitness_score, inlier_rmse, iteration + 1, 
            True, "Fast ICP matching successful"
        )
    
    def _psm_match(self, source_points, target_points, initial_guess=None):
        """
        Polar Scan Matching (PSM) algorithm (placeholder)
        
        Args:
            source_points (np.ndarray): Array of shape (N, 2) containing the source scan points
            target_points (np.ndarray): Array of shape (M, 2) containing the target scan points
            initial_guess (tuple): Optional (translation_x, translation_y, rotation) initial guess
            
        Returns:
            MatchResult: Result of the matching process
        """
        # This is a placeholder for future implementation
        return MatchResult(
            np.eye(3), (0, 0), 0, 0, float('inf'), 0, 
            False, "PSM algorithm not implemented yet"
        )
    
    def create_transform_matrix(self, translation_x, translation_y, rotation):
        """
        Create a 2D homogeneous transformation matrix
        
        Args:
            translation_x (float): X translation component
            translation_y (float): Y translation component
            rotation (float): Rotation angle in radians
            
        Returns:
            np.ndarray: 3x3 homogeneous transformation matrix
        """
        cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
        
        transform_matrix = np.array([
            [cos_theta, -sin_theta, translation_x],
            [sin_theta, cos_theta, translation_y],
            [0, 0, 1]
        ])
        
        return transform_matrix
    
    def get_last_transform(self):
        """Get the last computed transformation matrix"""
        return self.last_transform.copy() if self.last_transform is not None else np.eye(3)
    
    def transform_points(self, points, transform_matrix):
        """
        Apply a transformation to a set of points
        
        Args:
            points (np.ndarray): Array of shape (N, 2) containing points to transform
            transform_matrix (np.ndarray): 3x3 homogeneous transformation matrix
            
        Returns:
            np.ndarray: Transformed points
        """
        # Create homogeneous coordinates
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Apply transformation
        transformed_points = np.dot(homogeneous_points, transform_matrix.T)
        
        # Return to 2D coordinates
        return transformed_points[:, :2]
    
    def visualize_match(self, show_correspondences=True, title=None):
        """
        Visualize the matching result using matplotlib
        
        Args:
            show_correspondences (bool): Whether to show point correspondences
            title (str): Optional title for the plot
        """
        import matplotlib.pyplot as plt
        
        if (self.last_source_points is None or 
            self.last_target_points is None or 
            self.last_transform is None):
            print("No matching data to visualize")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Plot target points
        plt.scatter(self.last_target_points[:, 0], self.last_target_points[:, 1], 
                   c='blue', s=10, label='Target Points')
        
        # Plot original source points
        plt.scatter(self.last_source_points[:, 0], self.last_source_points[:, 1], 
                   c='red', s=10, label='Source Points')
        
        # Transform source points
        transformed_source = self.transform_points(self.last_source_points, self.last_transform)
        
        # Plot transformed source points
        plt.scatter(transformed_source[:, 0], transformed_source[:, 1], 
                   c='green', s=10, label='Transformed Source')
        
        # Show correspondences if requested
        if show_correspondences:
            # Find closest points
            for i, point in enumerate(transformed_source):
                # Only show a subset of correspondences for clarity
                if i % 5 == 0:  # Show every 5th point
                    distances = np.sqrt(np.sum((self.last_target_points - point)**2, axis=1))
                    closest_idx = np.argmin(distances)
                    closest_point = self.last_target_points[closest_idx]
                    
                    plt.plot([point[0], closest_point[0]], [point[1], closest_point[1]],
                            'k-', linewidth=0.5, alpha=0.3)
        
        # Extract transformation parameters
        tx, ty = self.last_transform[0, 2], self.last_transform[1, 2]
        theta = np.arctan2(self.last_transform[1, 0], self.last_transform[0, 0])
        
        # Add transformation text
        transform_text = f"Translation: ({tx:.4f}, {ty:.4f}), Rotation: {np.degrees(theta):.4f}°"
        plt.annotate(transform_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        # Set plot properties
        if title:
            plt.title(title)
        else:
            plt.title(f"Scan Matching Result using {self.algorithm.value.upper()}")
        
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.show()


# Helper functions for scan conversion
def convert_scan_to_cartesian(ranges, angles, min_range=0.1, max_range=30.0):
    """
    Convert a laser scan from polar to Cartesian coordinates
    
    Args:
        ranges (np.ndarray): Array of range measurements
        angles (np.ndarray): Array of corresponding angles
        min_range (float): Minimum valid range value
        max_range (float): Maximum valid range value
        
    Returns:
        np.ndarray: Array of (x, y) points
    """
    # Filter out invalid measurements
    valid_indices = np.logical_and(ranges >= min_range, ranges <= max_range)
    valid_ranges = ranges[valid_indices]
    valid_angles = angles[valid_indices]
    
    # Convert to Cartesian coordinates
    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
    
    return np.column_stack((x, y))


# Example usage
if __name__ == "__main__":
    # Create some test data
    source_x = np.random.uniform(-5, 5, 100)
    source_y = np.random.uniform(-5, 5, 100)
    source_points = np.column_stack((source_x, source_y))
    
    # Create a transformation
    tx, ty, theta = 1.0, 0.5, np.radians(10)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    # Apply transformation
    R = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    target_points = (R @ source_points.T).T + np.array([tx, ty])
    
    # Add noise
    target_points += np.random.normal(0, 0.05, target_points.shape)
    
    # Create matcher
    matcher = OptimizedScanMatcher(algorithm=MatchingAlgorithm.FAST_ICP)
    
    # Configure optimization params
    matcher.set_optimization_params(downsample_factor=1, max_iterations=20)
    
    # Match scans
    result = matcher.match_scans(source_points, target_points)
    
    # Print result
    print(result)
    
    # Visualize match
    matcher.visualize_match()