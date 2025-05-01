"""
Pose Estimator Module

This module provides pose estimation capabilities for LiDAR SLAM systems,
fusing odometry data with scan matching results to estimate robot pose.
"""

import numpy as np
import math
from enum import Enum

class PoseEstimationMethod(Enum):
    """Enumeration of available pose estimation methods"""
    ODOMETRY_ONLY = "odometry_only"  # Use only odometry data
    SCAN_MATCH_ONLY = "scan_match_only"  # Use only scan matching
    EKF_FUSION = "ekf_fusion"  # Extended Kalman Filter fusion
    WEIGHTED_AVERAGE = "weighted_average"  # Simple weighted average

class Pose2D:
    """Class to represent and manipulate 2D poses (x, y, theta)"""
    
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """
        Initialize a 2D pose
        
        Args:
            x (float): X position in meters
            y (float): Y position in meters
            theta (float): Orientation in radians
        """
        self.x = x
        self.y = y
        self.theta = self._normalize_angle(theta)
    
    def _normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi"""
        return ((angle + math.pi) % (2 * math.pi)) - math.pi
    
    def __str__(self):
        """String representation of the pose"""
        return f"Pose2D(x={self.x:.4f}, y={self.y:.4f}, theta={math.degrees(self.theta):.2f}°)"
    
    def copy(self):
        """Create a copy of this pose"""
        return Pose2D(self.x, self.y, self.theta)
    
    def to_dict(self):
        """Convert to dictionary format"""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta
        }
    
    def to_transform_matrix(self):
        """Convert pose to homogeneous transformation matrix"""
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        return np.array([
            [cos_theta, -sin_theta, self.x],
            [sin_theta, cos_theta, self.y],
            [0, 0, 1]
        ])
    
    @staticmethod
    def from_transform_matrix(matrix):
        """Create a Pose2D object from a transformation matrix"""
        x = matrix[0, 2]
        y = matrix[1, 2]
        theta = math.atan2(matrix[1, 0], matrix[0, 0])
        
        return Pose2D(x, y, theta)
    
    @staticmethod
    def from_dict(data):
        """Create a Pose2D object from a dictionary"""
        return Pose2D(
            data.get('x', 0.0),
            data.get('y', 0.0),
            data.get('theta', 0.0)
        )
    
    def apply_transform(self, transform_matrix):
        """
        Apply a transformation matrix to this pose
        
        Args:
            transform_matrix (np.ndarray): 3x3 homogeneous transformation matrix
            
        Returns:
            Pose2D: New pose after applying the transformation
        """
        # Convert current pose to matrix
        pose_matrix = self.to_transform_matrix()
        
        # Apply transformation
        new_matrix = np.dot(transform_matrix, pose_matrix)
        
        # Convert back to pose
        return Pose2D.from_transform_matrix(new_matrix)
    
    def relative_to(self, reference_pose):
        """
        Compute the relative pose with respect to a reference pose
        
        Args:
            reference_pose (Pose2D): Reference pose
            
        Returns:
            Pose2D: Relative pose
        """
        # Inverse of reference pose
        ref_cos = math.cos(reference_pose.theta)
        ref_sin = math.sin(reference_pose.theta)
        
        # Translate to reference frame
        dx = self.x - reference_pose.x
        dy = self.y - reference_pose.y
        
        # Rotate to reference frame
        rel_x = ref_cos * dx + ref_sin * dy
        rel_y = -ref_sin * dx + ref_cos * dy
        rel_theta = self._normalize_angle(self.theta - reference_pose.theta)
        
        return Pose2D(rel_x, rel_y, rel_theta)


class PoseWithCovariance:
    """Pose with associated covariance matrix for uncertainty representation"""
    
    def __init__(self, pose, covariance=None):
        """
        Initialize a pose with covariance
        
        Args:
            pose (Pose2D): The pose estimate
            covariance (np.ndarray): 3x3 covariance matrix, identity if None
        """
        self.pose = pose
        
        # Default covariance if not provided (high uncertainty)
        if covariance is None:
            self.covariance = np.eye(3) * 0.1
        else:
            self.covariance = covariance
    
    def __str__(self):
        """String representation of the pose with covariance"""
        return (
            f"Pose: {self.pose}\n"
            f"Covariance:\n{self.covariance}"
        )
    
    def copy(self):
        """Create a copy of this pose with covariance"""
        return PoseWithCovariance(self.pose.copy(), self.covariance.copy())


class PoseEstimator:
    """Class for estimating robot pose using various methods"""
    
    def __init__(self, method=PoseEstimationMethod.EKF_FUSION, initial_pose=None):
        """
        Initialize the pose estimator
        
        Args:
            method (PoseEstimationMethod): Method to use for pose estimation
            initial_pose (Pose2D): Initial robot pose, origin if None
        """
        self.method = method
        self.current_pose = initial_pose or Pose2D()
        
        # For storing pose history
        self.pose_history = [self.current_pose.copy()]
        
        # For EKF
        self.current_pose_with_cov = PoseWithCovariance(self.current_pose.copy())
        
        # Motion model parameters (for EKF prediction)
        self.alpha1 = 0.1  # Rotational error from rotational motion
        self.alpha2 = 0.1  # Rotational error from translational motion
        self.alpha3 = 0.1  # Translational error from translational motion
        self.alpha4 = 0.1  # Translational error from rotational motion
        
        # Scan matching parameters
        self.scan_match_base_noise = np.diag([0.05, 0.05, 0.02])  # x, y, theta in meters/radians
        
        # Debug information
        self.last_odom_update = None
        self.last_scan_match_result = None
        self.last_scan_match_covariance = None
        
        # Confidence tracking for scan matching
        self.scan_match_confidence_history = []
        self.min_confidence_threshold = 0.1  # Minimum confidence to consider a match valid
    
    def update_from_odometry(self, delta_x, delta_y, delta_theta):
        """
        Update pose using odometry data
        
        Args:
            delta_x (float): Change in x position in robot frame (meters)
            delta_y (float): Change in y position in robot frame (meters)
            delta_theta (float): Change in orientation (radians)
            
        Returns:
            Pose2D: Updated pose
        """
        # Store for debugging
        self.last_odom_update = (delta_x, delta_y, delta_theta)
        
        # Simple trigonometry to convert from robot frame to world frame
        cos_theta = math.cos(self.current_pose.theta)
        sin_theta = math.sin(self.current_pose.theta)
        
        # Translate to world frame
        world_delta_x = cos_theta * delta_x - sin_theta * delta_y
        world_delta_y = sin_theta * delta_x + cos_theta * delta_y
        
        # Apply updates
        new_x = self.current_pose.x + world_delta_x
        new_y = self.current_pose.y + world_delta_y
        new_theta = self.current_pose.theta + delta_theta
        
        # Create new pose
        new_pose = Pose2D(new_x, new_y, new_theta)
        
        # If using odometry only, update the current pose
        if self.method == PoseEstimationMethod.ODOMETRY_ONLY:
            self.current_pose = new_pose
            self.pose_history.append(self.current_pose.copy())
        
        # For EKF, predict step
        if self.method == PoseEstimationMethod.EKF_FUSION:
            self._ekf_predict(delta_x, delta_y, delta_theta)
        
        return new_pose
    
    def _ekf_predict(self, delta_x, delta_y, delta_theta):
        """
        EKF prediction step using odometry data
        
        Args:
            delta_x (float): Change in x position in robot frame (meters)
            delta_y (float): Change in y position in robot frame (meters)
            delta_theta (float): Change in orientation (radians)
        """
        # Current state
        x = self.current_pose_with_cov.pose.x
        y = self.current_pose_with_cov.pose.y
        theta = self.current_pose_with_cov.pose.theta
        
        # Convert to world frame motion
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Calculate motion in world frame
        delta_x_world = cos_theta * delta_x - sin_theta * delta_y
        delta_y_world = sin_theta * delta_x + cos_theta * delta_y
        
        # Update mean
        new_x = x + delta_x_world
        new_y = y + delta_y_world
        new_theta = theta + delta_theta
        
        # Calculate motion noise
        trans = math.sqrt(delta_x_world**2 + delta_y_world**2)
        
        # Calculate motion angles for noise model
        if abs(trans) > 1e-5:
            rot1 = math.atan2(delta_y, delta_x)
            rot2 = delta_theta - rot1
        else:
            # Near-zero motion case
            rot1 = 0
            rot2 = 0
        
        # Noise variances based on motion model
        sigma_rot1 = self.alpha1 * abs(rot1) + self.alpha2 * trans
        sigma_trans = self.alpha3 * trans + self.alpha4 * (abs(rot1) + abs(rot2))
        sigma_rot2 = self.alpha1 * abs(rot2) + self.alpha2 * trans
        
        # Motion noise covariance in control space
        M = np.diag([sigma_rot1**2, sigma_trans**2, sigma_rot2**2])
        
        # Jacobian of motion model with respect to state
        G = np.eye(3)
        G[0, 2] = -delta_y_world
        G[1, 2] = delta_x_world
        
        # Jacobian of motion model with respect to controls
        V = np.zeros((3, 3))
        V[0, 0] = -sin_theta * delta_x - cos_theta * delta_y
        V[0, 1] = cos_theta
        V[0, 2] = 0
        V[1, 0] = cos_theta * delta_x - sin_theta * delta_y
        V[1, 1] = sin_theta
        V[1, 2] = 0
        V[2, 0] = 0
        V[2, 1] = 0
        V[2, 2] = 1
        
        # Update covariance
        P = self.current_pose_with_cov.covariance
        P_pred = np.dot(np.dot(G, P), G.T) + np.dot(np.dot(V, M), V.T)
        
        # Create predicted pose with covariance
        predicted_pose = Pose2D(new_x, new_y, new_theta)
        self.current_pose_with_cov = PoseWithCovariance(predicted_pose, P_pred)
        
        # Update the current pose and history
        self.current_pose = predicted_pose
        self.pose_history.append(self.current_pose.copy())
    
    def update_from_scan_match(self, scan_match_result, reliability_weight=0.7):
        """
        Update pose using scan matching result
        
        Args:
            scan_match_result: Result from scan matcher (tuple with corrected pose and confidence)
            reliability_weight (float): Weight for scan matching vs odometry (0-1)
            
        Returns:
            Pose2D: Updated pose
        """
        # Parse the scan match result - now expecting (matched_pose, confidence) tuple
        if isinstance(scan_match_result, tuple) and len(scan_match_result) == 2:
            matched_pose, confidence = scan_match_result
        else:
            # For backward compatibility
            matched_pose = scan_match_result
            confidence = getattr(scan_match_result, 'fitness_score', 0.5)
        
        # Store for debugging
        self.last_scan_match_result = matched_pose
        self.scan_match_confidence_history.append(confidence)
        
        # Check if the match was successful and reliable
        if confidence < self.min_confidence_threshold:
            # If scan matching result is not reliable, keep the current pose
            return self.current_pose.copy()
        
        # Convert dictionary pose to Pose2D if needed
        if isinstance(matched_pose, dict):
            scan_match_pose = Pose2D.from_dict(matched_pose)
        else:
            scan_match_pose = matched_pose
        
        # Calculate covariance based on matching quality
        # Higher confidence -> lower uncertainty
        uncertainty_scale = 1.0 - min(0.95, confidence)  # Cap at 0.95 to ensure some minimum uncertainty
        pos_variance = 0.02 + uncertainty_scale * 0.2  # meters (min 0.02, max 0.22)
        rot_variance = 0.01 + uncertainty_scale * 0.1  # radians (min 0.01, max 0.11)
        
        scan_match_covariance = np.diag([pos_variance, pos_variance, rot_variance])
        self.last_scan_match_covariance = scan_match_covariance
        
        # Handle different update methods
        if self.method == PoseEstimationMethod.SCAN_MATCH_ONLY:
            # Directly use the scan match result
            self.current_pose = scan_match_pose
            self.pose_history.append(self.current_pose.copy())
            
        elif self.method == PoseEstimationMethod.WEIGHTED_AVERAGE:
            # Simple weighted average between odometry and scan matching
            # Get the last odometry-based pose (should be current_pose)
            odom_pose = self.current_pose
            
            # Adjust reliability weight based on confidence
            adjusted_weight = reliability_weight * confidence
            
            # Weighted average (linear interpolation)
            avg_x = odom_pose.x * (1 - adjusted_weight) + scan_match_pose.x * adjusted_weight
            avg_y = odom_pose.y * (1 - adjusted_weight) + scan_match_pose.y * adjusted_weight
            
            # For angles, we need to handle wrapping
            delta_theta = scan_match_pose.theta - odom_pose.theta
            # Normalize to [-pi, pi]
            delta_theta = ((delta_theta + math.pi) % (2 * math.pi)) - math.pi
            avg_theta = odom_pose.theta + delta_theta * adjusted_weight
            
            # Update the pose
            self.current_pose = Pose2D(avg_x, avg_y, avg_theta)
            self.pose_history.append(self.current_pose.copy())
            
        elif self.method == PoseEstimationMethod.EKF_FUSION:
            # Create relative transform from EKF prediction to scan match
            rel_dx = scan_match_pose.x - self.current_pose.x
            rel_dy = scan_match_pose.y - self.current_pose.y
            rel_dtheta = scan_match_pose.theta - self.current_pose.theta
            
            # Normalize angle difference
            rel_dtheta = ((rel_dtheta + math.pi) % (2 * math.pi)) - math.pi
            
            # EKF update using scan matching as measurement
            self._ekf_update(rel_dx, rel_dy, rel_dtheta, scan_match_covariance, confidence)
        
        return self.current_pose.copy()
    
    def _ekf_update(self, dx, dy, dtheta, measurement_cov, confidence):
        """
        EKF update step using scan matching result
        
        Args:
            dx (float): X translation from scan matching
            dy (float): Y translation from scan matching
            dtheta (float): Rotation from scan matching
            measurement_cov (np.ndarray): Measurement covariance matrix
            confidence (float): Confidence in the scan matching result (0-1)
        """
        # Current state and covariance
        predicted_pose = self.current_pose_with_cov.pose
        P = self.current_pose_with_cov.covariance
        
        # Create measurement from scan matching (relative transformation)
        z = np.array([dx, dy, dtheta])
        
        # Measurement matrix (identity for direct measurements)
        H = np.eye(3)
        
        # Adjust measurement covariance based on confidence
        confidence_factor = max(0.1, confidence)  # Ensure minimum factor
        R = measurement_cov / confidence_factor  # Lower covariance for higher confidence
        
        # Calculate Kalman gain
        S = np.dot(np.dot(H, P), H.T) + R
        try:
            S_inv = np.linalg.inv(S)
            K = np.dot(np.dot(P, H.T), S_inv)
        except np.linalg.LinAlgError:
            # Handle numerical issues - fall back to simple update
            print("Warning: Matrix inversion failed in EKF update. Using simplified update.")
            K = np.eye(3) * 0.5  # Simplified gain
        
        # Innovation (difference between measurement and prediction)
        innovation = z
        
        # Update state
        update = np.dot(K, innovation)
        
        # Apply update
        updated_x = predicted_pose.x + update[0]
        updated_y = predicted_pose.y + update[1]
        updated_theta = predicted_pose.theta + update[2]
        
        # Update covariance
        I = np.eye(3)
        updated_P = np.dot(I - np.dot(K, H), P)
        
        # Create updated pose with covariance
        updated_pose = Pose2D(updated_x, updated_y, updated_theta)
        self.current_pose_with_cov = PoseWithCovariance(updated_pose, updated_P)
        
        # Update the current pose and history
        self.current_pose = updated_pose
        self.pose_history.append(self.current_pose.copy())
    
    def get_current_pose(self):
        """Get the current pose estimate"""
        return self.current_pose.copy()
    
    def get_current_pose_with_covariance(self):
        """Get the current pose with covariance"""
        return self.current_pose_with_cov.copy()
    
    def get_pose_history(self):
        """Get the history of poses"""
        return [pose.copy() for pose in self.pose_history]
    
    def get_confidence_history(self):
        """Get the history of scan matching confidence values"""
        return self.scan_match_confidence_history.copy()
    
    def reset(self, initial_pose=None):
        """
        Reset the estimator to initial state
        
        Args:
            initial_pose (Pose2D): Initial pose to reset to, origin if None
        """
        self.current_pose = initial_pose or Pose2D()
        self.pose_history = [self.current_pose.copy()]
        self.current_pose_with_cov = PoseWithCovariance(self.current_pose.copy())
        self.last_odom_update = None
        self.last_scan_match_result = None
        self.last_scan_match_covariance = None
        self.scan_match_confidence_history = []