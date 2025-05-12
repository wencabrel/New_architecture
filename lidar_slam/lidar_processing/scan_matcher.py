#!/usr/bin/env python3
"""
Improved Scan Matcher Module

This module implements a sophisticated scan matcher using ICP (Iterative Closest Point)
algorithm with adaptive parameters, aggressive resampling, and motion validation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
from matplotlib.patches import Circle

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lidar_slam.lidar_processing import scan_converter, data_parser
from lidar_slam.utils import file_utils
from lidar_slam.mapping import occupancy_grid

class PoseEstimate:
    """Class to represent a robot pose estimate with uncertainty"""
    
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """
        Initialize a pose estimate
        
        Args:
            x (float): X position in meters
            y (float): Y position in meters
            theta (float): Orientation in radians
        """
        self.x = x
        self.y = y
        self.theta = theta
        
        # Covariance matrix for x, y, theta
        self.covariance = np.eye(3) * 0.01  # Default small uncertainty
        
    def to_dict(self):
        """Convert pose to dictionary format compatible with existing code"""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta
        }
    
    def from_dict(self, pose_dict):
        """Set pose from dictionary format"""
        self.x = pose_dict['x']
        self.y = pose_dict['y']
        self.theta = pose_dict['theta']
        return self
    
    def copy(self):
        """Create a copy of this pose estimate"""
        new_pose = PoseEstimate(self.x, self.y, self.theta)
        new_pose.covariance = self.covariance.copy()
        return new_pose
    
    def normalize_angle(self):
        """Normalize angle to [-π, π]"""
        self.theta = ((self.theta + math.pi) % (2 * math.pi)) - math.pi
        return self


class ScanMatcher:
    """
    Improved implementation of scan matching using ICP algorithm
    with adaptive parameters, alignment reset, and aggressive resampling.
    """
    
    def __init__(self, occupancy_grid_map=None, debug_level=1, **kwargs):
        """
        Initialize the improved scan matcher
        
        Args:
            occupancy_grid_map: OccupancyGrid object representing the map
            debug_level: 0=none, 1=basic info, 2=detailed, 3=verbose
            **kwargs: Additional parameters for backward compatibility with original ScanMatcher
                - search_radius: Maps to max_correspondence_distance
                - search_half_rad: Maps to max_rotation_per_frame
                - mismatch_prob_at_coarse: Maps to match_quality_threshold
                - scan_sigma_in_num_grid: Not directly used, but stored for compatibility
                - move_r_sigma: Maps to default_max_translation
                - max_move_deviation: Additional constraint on translation
                - turn_sigma: Additional constraint on rotation
                - coarse_factor: Not directly used, but stored for compatibility
        """
        self.map = occupancy_grid_map
        self.debug_level = debug_level
        
        # Default ICP parameters (will be adjusted adaptively)
        self.max_iterations = 15
        self.convergence_threshold = 0.001
        self.max_correspondence_distance = 1.5  # meters
        
        # Parameters for adaptive adjustment
        self.default_correspondence_distance = 1.5  # Starting value
        self.max_possible_correspondence_distance = 3.0  # Maximum allowed value
        self.min_correspondence_distance = 1.0  # Minimum allowed value
        
        # For aggressive resampling (when no correspondences found)
        self.aggressive_max_correspondence_distance = 5.0  # Much larger search radius
        self.aggressive_max_resampling_attempts = 5  # How many times to try resampling
        
        # Occupancy threshold (will be adjusted adaptively)
        self.occupancy_threshold = 0.55
        self.default_occupancy_threshold = 0.55  # Starting value
        self.min_occupancy_threshold = 0.45  # Minimum allowed value
        
        # For aggressive resampling
        self.aggressive_min_occupancy_threshold = 0.35  # Much lower threshold
        
        # Motion validation parameters (will be adjusted adaptively)
        self.max_translation_per_frame = 0.5  # meters
        self.default_max_translation = 0.5  # Starting value
        self.max_possible_translation = 1.0  # Maximum allowed value
        
        self.max_rotation_per_frame = 0.5  # radians (~28 degrees)
        self.default_max_rotation = 0.5  # Starting value
        self.max_possible_rotation = 0.8  # Maximum allowed value
        
        # Handle old parameter names for backward compatibility
        if 'search_radius' in kwargs:
            self.max_correspondence_distance = kwargs['search_radius']
            self.default_correspondence_distance = kwargs['search_radius']
            
        if 'search_half_rad' in kwargs:
            self.max_rotation_per_frame = kwargs['search_half_rad']
            self.default_max_rotation = kwargs['search_half_rad']
            
        if 'mismatch_prob_at_coarse' in kwargs:
            self.match_quality_threshold = kwargs['mismatch_prob_at_coarse']
            
        if 'move_r_sigma' in kwargs:
            self.default_max_translation = kwargs['move_r_sigma'] * 5  # Scale appropriately
            self.max_translation_per_frame = kwargs['move_r_sigma'] * 5
            
        if 'max_move_deviation' in kwargs:
            # This parameter affects how much deviation is allowed - keep it for reference
            self.max_move_deviation = kwargs['max_move_deviation']
            
        if 'turn_sigma' in kwargs:
            # Affects the allowed rotation error - store it but use our own parameters
            self.turn_sigma = kwargs['turn_sigma']
            
        # Store other parameters for compatibility but don't directly use them
        self.scan_sigma_in_num_grid = kwargs.get('scan_sigma_in_num_grid', 2)
        self.coarse_factor = kwargs.get('coarse_factor', 5)
        
        # Current estimated trajectory
        self.trajectory = []  # List of PoseEstimate objects
        
        # Last matched pose (used as reference for next match)
        self.last_matched_pose = None
        
        # Keep track of odometry poses for comparison
        self.odometry_trajectory = []
        
        # Flag to determine if we've loaded a map or are building it
        self.mapping_mode = True if occupancy_grid_map is None else False
        
        # For visualization purposes
        self.current_visualization_data = None
        
        # Keep track of match count for logging
        self.match_count = 0
        
        # Flag to track if we've built the map enough
        self.map_built = False
        
        # Track the building progress for the map
        self.map_build_progress = 0
        
        # Alignment reset tracking
        self.frames_since_last_reset = 0
        self.reset_interval = 50  # Check alignment every 50 frames
        self.drift_threshold = 0.7  # Trigger reset if drift exceeds 0.7m
        
        # Match quality tracking
        self.match_qualities = []  # Track recent match qualities
        self.quality_history_size = 5  # How many recent matches to consider
        
        # Health monitoring
        self.consecutive_poor_matches = 0
        self.match_quality_threshold = 0.4  # Threshold for a "good" match
        
        # Recovery mode
        self.in_recovery_mode = False
        self.recovery_counter = 0
        self.recovery_frames = 5  # How many frames to stay in recovery mode
        
        # Map expansion tracking
        self.should_expand_map = False
        self.force_map_expansion = False  # Used for emergency expansion
        
        # Resampling stats
        self.resampling_attempts = 0  # Track how many times we've had to resample
        self.frames_with_resampling = 0  # Track how many frames needed resampling
        
        # For scan conversion
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        self.scan_config = {
            'flip_x': False,
            'flip_y': False,
            'reverse_scan': True,
            'flip_theta': False
        }
        
        # Parameters derived from occupancy grid
        self.angular_step = 0.01  # Small angular step for fine rotations
        self.lidar_max_range = 10.0  # Default max range, can be updated
        self.lidar_fov = math.pi  # Default FOV, can be updated
        self.num_samples_per_rev = 180  # Default number of samples, can be updated
        
        if self.debug_level > 0:
            print("[ScanMatcher] Initialized with adaptive parameters and aggressive resampling")
            print(f"[ScanMatcher]   - Base correspondence distance: {self.default_correspondence_distance}m (can increase to {self.max_possible_correspondence_distance}m)")
            print(f"[ScanMatcher]   - Base occupancy threshold: {self.occupancy_threshold} (can decrease to {self.min_occupancy_threshold})")
            print(f"[ScanMatcher]   - Base max translation: {self.max_translation_per_frame}m (can increase to {self.max_possible_translation}m)")
            print(f"[ScanMatcher]   - Alignment reset interval: {self.reset_interval} frames (drift threshold: {self.drift_threshold}m)")
            print(f"[ScanMatcher]   - Aggressive resampling enabled (max search radius: {self.aggressive_max_correspondence_distance}m, min threshold: {self.aggressive_min_occupancy_threshold})")
            print("\n" + "="*80)
            print("        POSE INFORMATION FOR EACH SCAN MATCH WILL BE PRINTED BELOW")
            print("="*80 + "\n")
    
    def set_scan_config(self, config):
        """
        Set the scan configuration parameters
        
        Args:
            config (dict): Dictionary with scan configuration parameters
        """
        self.scan_config = config
    
    def set_lidar_params(self, max_range, field_of_view, num_samples):
        """
        Set LiDAR parameters for scan matching
        
        Args:
            max_range (float): Maximum range of the LiDAR in meters
            field_of_view (float): Field of view in radians
            num_samples (int): Number of samples per revolution
        """
        self.lidar_max_range = max_range
        self.lidar_fov = field_of_view
        self.num_samples_per_rev = num_samples
        self.angular_step = field_of_view / num_samples
    
    def process_dataset(self, parsed_data_list, initial_pose=None):
        """
        Process a sequence of LiDAR scans to localize the robot
        
        Args:
            parsed_data_list: List of parsed LiDAR data dictionaries
            initial_pose: Initial pose estimate (PoseEstimate object or None)
            
        Returns:
            List of updated pose estimates (trajectory)
        """
        if self.debug_level > 0:
            print("[ScanMatcher] Processing sensor data with ICP scan matching...")
        
        # Initialize trajectory with initial pose if provided
        if initial_pose:
            self.last_matched_pose = initial_pose
            self.trajectory = [initial_pose.copy()]
            self.odometry_trajectory = [initial_pose.copy()]
        else:
            # Use the first scan's pose as initial estimate
            first_pose_dict = parsed_data_list[0]['pose']
            initial_pose = PoseEstimate(
                first_pose_dict['x'], 
                first_pose_dict['y'], 
                first_pose_dict['theta']
            )
            self.last_matched_pose = initial_pose
            self.trajectory = [initial_pose.copy()]
            self.odometry_trajectory = [initial_pose.copy()]
        
        # Set the minimum number of frames to build the map before matching
        map_build_frames = 20  # Frames dedicated to building the initial map
        
        # Process each scan
        for i, scan_data in enumerate(parsed_data_list):
            if i == 0 and initial_pose:
                # Skip first scan if we already set the initial pose
                continue
            
            if self.debug_level > 0:
                print(f"\r[ScanMatcher] Processing scan {i+1}/{len(parsed_data_list)}", end="")
            
            # First, store the odometry pose
            odometry_pose = PoseEstimate().from_dict(scan_data['pose'])
            self.odometry_trajectory.append(odometry_pose.copy())
            
            # Convert from polar to Cartesian coordinates
            scan_x, scan_y = scan_converter.convert_scans_to_cartesian(
                scan_data['scan_ranges'], 
                self.angle_min, self.angle_max, 
                scan_data['pose'],
                **self.scan_config
            )
            
            # Update the map with current scan
            if self.map:
                # Check if we need to expand the map
                if (self.should_expand_map or self.force_map_expansion) and hasattr(self.map, 'expand_grid'):
                    self.map.expand_grid()
                    self.should_expand_map = False
                    self.force_map_expansion = False
                
                # Update the map with the current scan
                try:
                    self.map.update_grid(
                        odometry_pose.x,  # Use odometry pose for mapping
                        odometry_pose.y,
                        scan_x,
                        scan_y
                    )
                except Exception as e:
                    print(f"\n[ScanMatcher] Warning: Error updating map: {e}")
                    # If we get grid bounds errors, force map expansion next frame
                    self.force_map_expansion = True
                
                # Track the building progress
                self.map_build_progress = (i * 100) // map_build_frames if i <= map_build_frames else 100
                
                # DEBUG: Print map statistics every 10 frames
                if i % 10 == 0 and self.debug_level > 1:
                    try:
                        occupied_cells = np.sum(self.map.grid > self.occupancy_threshold)
                        total_cells = self.map.grid_width * self.map.grid_height
                        print(f"\n[ScanMatcher] Map stats: Min={np.min(self.map.grid):.3f}, "
                              f"Max={np.max(self.map.grid):.3f}, "
                              f"Mean={np.mean(self.map.grid):.3f}, "
                              f"Occupied={occupied_cells}/{total_cells} cells "
                              f"({occupied_cells/total_cells*100:.2f}%)")
                    except Exception as e:
                        print(f"\n[ScanMatcher] Error calculating map stats: {e}")
            
            # Check if we've built enough of the map
            if i >= map_build_frames and not self.map_built:
                self.map_built = True
                print(f"\n[ScanMatcher] Map building complete after {i+1} frames. Starting scan matching.")
            
            # Only perform scan matching if we have a map and have built it enough
            if self.map_built:
                # Check if we need to reset alignment between odometry and scan matcher
                alignment_reset = self.checkAndResetAlignment()
                if alignment_reset:
                    print(f"[ScanMatcher] Alignment reset performed at frame {i}")
                    continue  # Skip this frame's scan matching
                
                # Now adapt parameters based on matching history
                self.adaptParametersBasedOnMatchQuality()
                
                # Get the relative odometry movement since last frame
                relative_dx = odometry_pose.x - self.odometry_trajectory[-2].x
                relative_dy = odometry_pose.y - self.odometry_trajectory[-2].y
                relative_dtheta = odometry_pose.theta - self.odometry_trajectory[-2].theta
                
                # Apply this relative movement to our last matched pose as the initial guess
                initial_guess = self.last_matched_pose.copy()
                initial_guess.x += relative_dx 
                initial_guess.y += relative_dy
                initial_guess.theta += relative_dtheta
                
                # Print poses before matching
                self.print_pose_comparison(
                    match_num=self.match_count+1,
                    previous_pose=self.last_matched_pose,
                    odometry_pose=odometry_pose,
                    initial_guess=initial_guess,
                    stage="BEFORE MATCHING"
                )
                
                # Check for emergency alignment reset if drift is extreme
                odom_guess_diff_x = odometry_pose.x - initial_guess.x
                odom_guess_diff_y = odometry_pose.y - initial_guess.y
                odom_guess_diff_dist = np.sqrt(odom_guess_diff_x**2 + odom_guess_diff_y**2)
                
                if odom_guess_diff_dist > 2.0:  # More than 2 meters difference is extreme
                    print(f"\n[ScanMatcher] EMERGENCY: Extreme drift detected ({odom_guess_diff_dist:.2f}m). "
                          f"Forcing immediate alignment reset.")
                    
                    # Create reset pose using odometry position but keeping matched orientation
                    reset_pose = PoseEstimate(
                        odometry_pose.x,
                        odometry_pose.y,
                        self.last_matched_pose.theta
                    )
                    
                    # Update trajectory
                    self.trajectory.append(reset_pose.copy())
                    self.last_matched_pose = reset_pose
                    
                    # Reset counters and flags
                    self.frames_since_last_reset = 0
                    self.consecutive_poor_matches = 0
                    
                    # Skip to next frame
                    self.match_count += 1
                    continue
                
                # Check if we're in recovery mode
                if self.in_recovery_mode:
                    # In recovery mode, use odometry directly for a few frames
                    self.recovery_counter += 1
                    
                    # Print recovery mode status
                    print(f"\n[ScanMatcher] In recovery mode (frame {self.recovery_counter}/{self.recovery_frames})")
                    
                    # Create a pose that's a blend between odometry and last matched
                    recovery_pose = self.createRecoveryPose(odometry_pose, self.last_matched_pose)
                    match_info = {
                        'iterations': 0,
                        'final_score': 0.5,  # Arbitrary middle score
                        'error': 0.0,
                        'correspondences': 0,
                        'resampling_attempts': 0
                    }
                    
                    # Print recovery pose
                    self.print_pose_comparison(
                        match_num=self.match_count+1,
                        previous_pose=self.last_matched_pose,
                        odometry_pose=odometry_pose,
                        initial_guess=initial_guess,
                        estimated_pose=recovery_pose,
                        match_info=match_info,
                        stage="RECOVERY MODE"
                    )
                    
                    # Update trajectory with recovery pose
                    self.trajectory.append(recovery_pose.copy())
                    self.last_matched_pose = recovery_pose
                    
                    # Exit recovery mode after enough frames
                    if self.recovery_counter >= self.recovery_frames:
                        self.in_recovery_mode = False
                        self.recovery_counter = 0
                        self.consecutive_poor_matches = 0
                        print(f"\n[ScanMatcher] Exiting recovery mode")
                else:
                    # Normal mode - match current scan against the map using ICP
                    matched_pose, match_info = self.matchScan(scan_x, scan_y, initial_guess)
                    
                    # Check if we have the special case of resampling
                    if match_info['resampling_attempts'] > 0:
                        resampling_str = f"[ScanMatcher] Used aggressive resampling - {match_info['resampling_attempts']} attempts needed"
                        if match_info['correspondences'] > 0:
                            resampling_str += f", found {match_info['correspondences']} correspondences"
                        print(f"\n{resampling_str}")
                        
                        self.frames_with_resampling += 1
                        self.resampling_attempts += match_info['resampling_attempts']
                        
                    # Check for boundary issues - if many points are out of bounds, flag for map expansion
                    if self.checkForMapBoundaryIssues(scan_x, scan_y, matched_pose):
                        self.should_expand_map = True
                        
                    # Validate the match - check if the movement is reasonable
                    is_valid = self.validateMatch(matched_pose, self.last_matched_pose, match_info)
                    
                    # Update parameters based on match quality for next frame
                    self.adaptParametersBasedOnMatchQuality(match_info)
                    
                    if is_valid:
                        # Reset the consecutive failures counter
                        self.consecutive_poor_matches = 0
                        
                        # Print final matched pose
                        self.print_pose_comparison(
                            match_num=self.match_count+1,
                            previous_pose=self.last_matched_pose,
                            odometry_pose=odometry_pose,
                            initial_guess=initial_guess,
                            estimated_pose=matched_pose,
                            match_info=match_info,
                            stage="AFTER MATCHING (VALID)"
                        )
                        
                        # Update trajectory with the matched pose
                        self.trajectory.append(matched_pose.copy())
                        self.last_matched_pose = matched_pose
                        
                        if self.debug_level > 1:
                            print(f"\n[ScanMatcher] Valid match found. Score: {match_info['final_score']:.4f}")
                    else:
                        # Increment the consecutive failures counter
                        self.consecutive_poor_matches += 1
                        
                        # If match is invalid, use the odometry pose with small correction
                        corrected_pose = self.applySmallCorrection(odometry_pose, self.last_matched_pose)
                        
                        # Print corrected pose 
                        self.print_pose_comparison(
                            match_num=self.match_count+1,
                            previous_pose=self.last_matched_pose,
                            odometry_pose=odometry_pose,
                            initial_guess=initial_guess,
                            estimated_pose=matched_pose,
                            corrected_pose=corrected_pose,
                            match_info=match_info,
                            stage="AFTER MATCHING (INVALID - USING CORRECTION)"
                        )
                        
                        self.trajectory.append(corrected_pose.copy())
                        self.last_matched_pose = corrected_pose
                        
                        # Check if we need to enter recovery mode
                        if self.consecutive_poor_matches >= 3:
                            print(f"\n[ScanMatcher] ⚠️ {self.consecutive_poor_matches} consecutive match failures! Entering recovery mode.")
                            self.in_recovery_mode = True
                            self.recovery_counter = 0
                        else:
                            if self.debug_level > 0:
                                print(f"\n[ScanMatcher] ⚠️ Invalid match rejected! Using odometry with correction.")
                
                # Increment match count
                self.match_count += 1
            else:
                # In mapping mode or early frames, use odometry for trajectory
                current_pose = PoseEstimate().from_dict(scan_data['pose'])
                self.trajectory.append(current_pose.copy())
                self.last_matched_pose = current_pose
                
                # Print the building progress
                if i % 5 == 0:
                    print(f"\n[ScanMatcher] Building map... {self.map_build_progress}% complete")
        
        if self.debug_level > 0:
            print(f"\n[ScanMatcher] Processed {len(parsed_data_list)} scans. Trajectory contains {len(self.trajectory)} poses.")
            if self.frames_with_resampling > 0:
                print(f"[ScanMatcher] Aggressive resampling was used in {self.frames_with_resampling} frames " 
                      f"({self.frames_with_resampling/self.match_count*100:.1f}% of matches).")
                print(f"[ScanMatcher] Average of {self.resampling_attempts/self.frames_with_resampling:.1f} " 
                      f"resampling attempts per frame when needed.")
        return self.trajectory

    def match_scan(self, scan_data, initial_pose, est_moving_dist=0, est_moving_theta=None, match_max=True):
        """
        Match current scan against occupancy grid - interface compatible with original ScanMatcher
        
        Args:
            scan_data (dict): Current sensor reading with scan ranges
            initial_pose (dict): Initial pose estimate dictionary
            est_moving_dist (float): Estimated movement distance
            est_moving_theta (float): Estimated movement angle
            match_max (bool): Whether to use maximum likelihood matching
            
        Returns:
            tuple: (matched_pose, confidence) - Best match pose dictionary and confidence score
        """
        # Convert the initial pose dict to PoseEstimate if needed
        if isinstance(initial_pose, dict):
            initial_pose_obj = PoseEstimate(
                initial_pose['x'],
                initial_pose['y'],
                initial_pose['theta']
            )
        else:
            initial_pose_obj = initial_pose
            
        # Extract scan points from scan data
        # Use existing scan_ranges or range key based on what's available
        if 'scan_ranges' in scan_data:
            scan_ranges = scan_data['scan_ranges']
        elif 'range' in scan_data:
            scan_ranges = scan_data['range']
        else:
            raise ValueError("Scan data must contain 'scan_ranges' or 'range' field")
        
        # If the est_moving_dist and est_moving_theta are provided, use them to
        # improve the initial guess by applying the estimated motion
        if est_moving_dist > 0:
            # Update the max_translation_per_frame for this match based on estimated movement
            self.max_translation_per_frame = max(
                self.default_max_translation,
                est_moving_dist * 1.2  # Allow 20% more than estimated
            )
            
            # If we have estimated motion parameters, use them to adjust the initial pose
            if est_moving_theta is not None:
                # Update max_rotation_per_frame based on estimated rotation
                self.max_rotation_per_frame = max(
                    self.default_max_rotation,
                    abs(est_moving_theta) * 1.2  # Allow 20% more than estimated
                )
            
            if self.debug_level > 1:
                print(f"[ScanMatcher] Using estimated motion: dist={est_moving_dist:.3f}m, theta={est_moving_theta:.3f}rad")
                print(f"[ScanMatcher] Adjusted motion limits: trans={self.max_translation_per_frame:.3f}m, rot={self.max_rotation_per_frame:.3f}rad")
            
        # Convert scan to Cartesian coordinates
        scan_x, scan_y = scan_converter.convert_scans_to_cartesian(
            scan_ranges, 
            self.angle_min, self.angle_max, 
            initial_pose_obj.to_dict() if isinstance(initial_pose_obj, PoseEstimate) else initial_pose_obj,
            **self.scan_config
        )
            
        # Call the internal matchScan method
        matched_pose, match_info = self.matchScan(scan_x, scan_y, initial_pose_obj)
        
        # Convert the matched pose to a dictionary for compatibility
        matched_pose_dict = matched_pose.to_dict()
        
        # Return the same format as the original ScanMatcher
        return matched_pose_dict, match_info['final_score']
    
    def adaptParametersBasedOnMatchQuality(self, match_info=None):
        """
        Adaptively adjust parameters based on recent match quality
        
        Args:
            match_info: Information from the last match attempt
        """
        # If we have match info, add it to our history
        if match_info is not None:
            self.match_qualities.append({
                'score': match_info['final_score'],
                'correspondences': match_info['correspondences'],
                'error': match_info['error']
            })
            
            # Keep only the most recent N matches
            if len(self.match_qualities) > self.quality_history_size:
                self.match_qualities.pop(0)
        
        # If we don't have enough history yet, use default settings
        if len(self.match_qualities) < 2:
            return
        
        # Calculate the average match quality
        avg_score = sum(q['score'] for q in self.match_qualities) / len(self.match_qualities)
        avg_correspondences = sum(q['correspondences'] for q in self.match_qualities) / len(self.match_qualities)
        
        # Check if we're having matching problems
        poor_match = avg_score < self.match_quality_threshold or avg_correspondences < 10
        
        # Get the most recent match result
        last_match = self.match_qualities[-1]
        
        # Calculate adaptive parameter adjustments
        if poor_match:
            # Don't increment the consecutive_poor_matches counter here, as it's done in the main loop
            
            # Adaptively increase search parameters based on consecutive poor matches
            adjustment_factor = min(1.0, 0.2 * self.consecutive_poor_matches)  # Up to 100% adjustment
            
            # Increase search radius
            self.max_correspondence_distance = min(
                self.max_possible_correspondence_distance,
                self.default_correspondence_distance * (1.0 + adjustment_factor)
            )
            
            # Lower occupancy threshold
            self.occupancy_threshold = max(
                self.min_occupancy_threshold,
                self.default_occupancy_threshold * (1.0 - adjustment_factor * 0.3)
            )
            
            # Increase motion limits
            self.max_translation_per_frame = min(
                self.max_possible_translation,
                self.default_max_translation * (1.0 + adjustment_factor)
            )
            
            self.max_rotation_per_frame = min(
                self.max_possible_rotation,
                self.default_max_rotation * (1.0 + adjustment_factor * 0.5)
            )
            
            if self.debug_level > 1 and self.consecutive_poor_matches > 0:
                print(f"\n[ScanMatcher] Low match quality detected ({self.consecutive_poor_matches} consecutive). Adapting parameters:")
                print(f"  - Correspondence distance: {self.max_correspondence_distance:.2f}m")
                print(f"  - Occupancy threshold: {self.occupancy_threshold:.2f}")
                print(f"  - Max translation: {self.max_translation_per_frame:.2f}m")
        else:
            # Good match, gradually return to default values
            # Don't reset consecutive_poor_matches here, as it's done in the main loop
            
            # Gradually move back towards defaults (10% step)
            self.max_correspondence_distance = self.max_correspondence_distance * 0.9 + self.default_correspondence_distance * 0.1
            self.occupancy_threshold = self.occupancy_threshold * 0.9 + self.default_occupancy_threshold * 0.1
            self.max_translation_per_frame = self.max_translation_per_frame * 0.9 + self.default_max_translation * 0.1
            self.max_rotation_per_frame = self.max_rotation_per_frame * 0.9 + self.default_max_rotation * 0.1
    
    def checkAndResetAlignment(self):
        """
        Check alignment between odometry and scan matcher, and reset if necessary
        
        Returns:
            bool: True if alignment was reset, False otherwise
        """
        # Make sure we have enough data
        if len(self.odometry_trajectory) < 2 or len(self.trajectory) < 1:
            return False
        
        # Increment counter for frames since last reset
        self.frames_since_last_reset += 1
        
        # Only check at specified interval
        if self.frames_since_last_reset < self.reset_interval:
            return False
        
        # Get the most recent odometry pose
        current_odom = self.odometry_trajectory[-1]
        
        # Get the current scan-matched pose
        current_matched = self.trajectory[-1]
        
        # Calculate drift between odometry and scan matcher
        drift_x = current_odom.x - current_matched.x
        drift_y = current_odom.y - current_matched.y
        drift_dist = math.sqrt(drift_x**2 + drift_y**2)
        
        if self.debug_level > 0:
            print(f"\n[ScanMatcher] Alignment check - Current drift: {drift_dist:.2f}m between odometry and scan matcher")
        
        # Reset alignment if drift exceeds threshold
        if drift_dist > self.drift_threshold:
            if self.debug_level > 0:
                print(f"[ScanMatcher] Excessive drift detected! Odometry: ({current_odom.x:.2f}, {current_odom.y:.2f}), "
                      f"Matched: ({current_matched.x:.2f}, {current_matched.y:.2f})")
            
            # Create a new pose that uses the odometry position but keeps the scan matcher's orientation
            reset_pose = PoseEstimate(
                current_odom.x, 
                current_odom.y,
                current_matched.theta  # Keep the scan matcher's orientation estimate
            )
            
            # Update the last matched pose
            self.last_matched_pose = reset_pose
            
            # Add to trajectory
            self.trajectory.append(reset_pose.copy())
            
            # Reset the counter
            self.frames_since_last_reset = 0
            
            # Reset parameters to defaults when realigning
            self.max_correspondence_distance = self.default_correspondence_distance
            self.occupancy_threshold = self.default_occupancy_threshold
            self.max_translation_per_frame = self.default_max_translation
            self.max_rotation_per_frame = self.default_max_rotation
            
            if self.debug_level > 0:
                print(f"[ScanMatcher] ALIGNMENT RESET to odometry position: ({reset_pose.x:.2f}, {reset_pose.y:.2f})")
                print(f"[ScanMatcher] Parameters reset to defaults")
            
            return True
        
        # If we performed a check but didn't reset, still reset the counter
        self.frames_since_last_reset = 0
        return False
    
    def createRecoveryPose(self, odometry_pose, last_matched_pose):
        """
        Create a recovery pose by blending odometry and last matched pose
        
        Args:
            odometry_pose: Current odometry pose
            last_matched_pose: Last matched pose
            
        Returns:
            PoseEstimate: Recovery pose
        """
        # Calculate relative movement from odometry
        if len(self.odometry_trajectory) < 2:
            return odometry_pose.copy()
            
        last_odometry_pose = self.odometry_trajectory[-2]
        relative_dx = odometry_pose.x - last_odometry_pose.x
        relative_dy = odometry_pose.y - last_odometry_pose.y
        relative_dtheta = odometry_pose.theta - last_odometry_pose.theta
        
        # Create recovery pose by using odometry movement from last matched pose
        recovery_pose = last_matched_pose.copy()
        recovery_pose.x += relative_dx
        recovery_pose.y += relative_dy
        recovery_pose.theta += relative_dtheta
        
        return recovery_pose
    
    def checkForMapBoundaryIssues(self, scan_x, scan_y, pose):
        """
        Check if the current scan is near map boundaries
        
        Args:
            scan_x, scan_y: Scan points
            pose: Current pose
            
        Returns:
            bool: True if map expansion is needed
        """
        if self.map is None:
            return False
            
        # Create points array from scan
        scan_points = np.column_stack((np.array(scan_x), np.array(scan_y)))
        
        # Transform points to world frame
        world_points = self.transformPointsToWorld(scan_points, pose)
        
        # Count how many points are near the boundary
        buffer = 2.0  # 2 meter buffer
        boundary_points = 0
        
        map_width = self.map.width
        map_height = self.map.height
        
        for point in world_points:
            # Check if point is near the boundary
            if (abs(point[0]) >= map_width/2 - buffer or 
                abs(point[1]) >= map_height/2 - buffer):
                boundary_points += 1
        
        # If more than 20% of points are near boundary, suggest expansion
        if boundary_points > 0.2 * len(world_points):
            if self.debug_level > 0:
                print(f"\n[ScanMatcher] Warning: {boundary_points} scan points ({boundary_points/len(world_points)*100:.1f}%) "
                     f"are near map boundaries. Map expansion recommended.")
            return True
            
        return False
    
    def print_pose_comparison(self, match_num, previous_pose, odometry_pose, initial_guess, 
                            estimated_pose=None, corrected_pose=None, match_info=None, stage=""):
        """
        Print a detailed comparison of poses for debugging
        
        Args:
            match_num: The match number (for tracking)
            previous_pose: The previous matched pose
            odometry_pose: The current odometry pose
            initial_guess: The initial guess for ICP 
            estimated_pose: The estimated pose from ICP (if available)
            corrected_pose: The corrected pose (if applicable)
            match_info: Match information dictionary
            stage: Description of the matching stage
        """
        # Calculate deltas from previous pose
        odom_delta_x = odometry_pose.x - previous_pose.x
        odom_delta_y = odometry_pose.y - previous_pose.y
        odom_delta_theta = self.normalize_angle(odometry_pose.theta - previous_pose.theta)
        
        guess_delta_x = initial_guess.x - previous_pose.x
        guess_delta_y = initial_guess.y - previous_pose.y
        guess_delta_theta = self.normalize_angle(initial_guess.theta - previous_pose.theta)
        
        # Print header
        print(f"\n{'='*100}")
        print(f"MATCH #{match_num}: {stage}")
        print(f"{'-'*100}")
        
        # Print previous pose
        print(f"PREVIOUS POSE:    x={previous_pose.x:.4f}, y={previous_pose.y:.4f}, θ={previous_pose.theta:.4f}")
        
        # Print odometry pose and delta
        print(f"ODOMETRY POSE:    x={odometry_pose.x:.4f}, y={odometry_pose.y:.4f}, θ={odometry_pose.theta:.4f}")
        print(f"ODOMETRY DELTA:   Δx={odom_delta_x:.4f}, Δy={odom_delta_y:.4f}, Δθ={odom_delta_theta:.4f}")
        
        # Print initial guess
        print(f"INITIAL GUESS:    x={initial_guess.x:.4f}, y={initial_guess.y:.4f}, θ={initial_guess.theta:.4f}")
        print(f"GUESS DELTA:      Δx={guess_delta_x:.4f}, Δy={guess_delta_y:.4f}, Δθ={guess_delta_theta:.4f}")
        
        # Calculate the difference between odometry and initial guess
        odom_guess_diff_x = odometry_pose.x - initial_guess.x
        odom_guess_diff_y = odometry_pose.y - initial_guess.y
        odom_guess_diff_dist = np.sqrt(odom_guess_diff_x**2 + odom_guess_diff_y**2)
        
        # Print the difference
        print(f"ODOM-GUESS DIFF:  Δx={odom_guess_diff_x:.4f}, Δy={odom_guess_diff_y:.4f}, dist={odom_guess_diff_dist:.4f}")
        
        # Print estimated pose if available
        if estimated_pose:
            est_delta_x = estimated_pose.x - previous_pose.x
            est_delta_y = estimated_pose.y - previous_pose.y
            est_delta_theta = self.normalize_angle(estimated_pose.theta - previous_pose.theta)
            
            print(f"ESTIMATED POSE:   x={estimated_pose.x:.4f}, y={estimated_pose.y:.4f}, θ={estimated_pose.theta:.4f}")
            print(f"ESTIMATED DELTA:  Δx={est_delta_x:.4f}, Δy={est_delta_y:.4f}, Δθ={est_delta_theta:.4f}")
            
            # Print match info if available
            if match_info:
                # Include resampling info
                if match_info.get('resampling_attempts', 0) > 0:
                    print(f"MATCH INFO:       Score={match_info['final_score']:.4f}, Iterations={match_info['iterations']}, "
                          f"Error={match_info['error']:.6f}, Correspondences={match_info['correspondences']}, "
                          f"Resampling Attempts={match_info['resampling_attempts']}")
                else:
                    print(f"MATCH INFO:       Score={match_info['final_score']:.4f}, Iterations={match_info['iterations']}, "
                          f"Error={match_info['error']:.6f}, Correspondences={match_info['correspondences']}")
        
        # Print corrected pose if available
        if corrected_pose:
            corr_delta_x = corrected_pose.x - previous_pose.x
            corr_delta_y = corrected_pose.y - previous_pose.y
            corr_delta_theta = self.normalize_angle(corrected_pose.theta - previous_pose.theta)
            
            print(f"CORRECTED POSE:   x={corrected_pose.x:.4f}, y={corrected_pose.y:.4f}, θ={corrected_pose.theta:.4f}")
            print(f"CORRECTED DELTA:  Δx={corr_delta_x:.4f}, Δy={corr_delta_y:.4f}, Δθ={corr_delta_theta:.4f}")
        
        print(f"{'='*100}\n")
    
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        return ((angle + math.pi) % (2 * math.pi)) - math.pi
    
    def matchScan(self, scan_x, scan_y, initial_pose):
        """
        Match the current scan against the map using ICP algorithm with aggressive resampling
        
        Args:
            scan_x: List of scan x coordinates
            scan_y: List of scan y coordinates
            initial_pose: Initial pose estimate (PoseEstimate object)
            
        Returns:
            Updated pose estimate (PoseEstimate object) and match info dictionary
        """
        # Create points array from scan
        scan_points = np.column_stack((np.array(scan_x), np.array(scan_y)))
        
        # Transform scan points to world frame using initial pose
        transformed_points = self.transformPointsToWorld(scan_points, initial_pose)
        
        # Try with normal parameters first
        current_pose = initial_pose.copy()
        correspondences, mean_error = self.findCorrespondences(transformed_points)
        
        # If we don't have enough correspondences, use aggressive resampling
        resampling_attempts = 0
        original_max_correspondence_distance = self.max_correspondence_distance
        original_occupancy_threshold = self.occupancy_threshold
        
        if len(correspondences) < 5:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Only {len(correspondences)} correspondences found initially. Starting aggressive resampling.")
            
            # Gradually increase search parameters until we find enough correspondences
            for attempt in range(1, self.aggressive_max_resampling_attempts + 1):
                resampling_attempts += 1
                
                # Calculate more aggressive parameters based on attempt number
                progress = attempt / self.aggressive_max_resampling_attempts
                
                # Increase search radius dramatically
                search_radius = self.max_correspondence_distance + progress * (self.aggressive_max_correspondence_distance - self.max_correspondence_distance)
                
                # Lower occupancy threshold dramatically
                occupancy_threshold = self.occupancy_threshold - progress * (self.occupancy_threshold - self.aggressive_min_occupancy_threshold)
                
                if self.debug_level > 1:
                    print(f"[ScanMatcher] Resampling attempt {attempt}: search radius={search_radius:.2f}m, "
                          f"occupancy threshold={occupancy_threshold:.2f}")
                
                # Temporarily set the new parameters
                self.max_correspondence_distance = search_radius
                self.occupancy_threshold = occupancy_threshold
                
                # Try to find correspondences with these more aggressive parameters
                correspondences, mean_error = self.findCorrespondences(transformed_points)
                
                if len(correspondences) >= 5:
                    if self.debug_level > 1:
                        print(f"[ScanMatcher] Found {len(correspondences)} correspondences after {attempt} resampling attempts.")
                    break
            
            # Restore original parameters
            self.max_correspondence_distance = original_max_correspondence_distance
            self.occupancy_threshold = original_occupancy_threshold
        
        # If we still don't have enough correspondences, use the initial pose
        if len(correspondences) < 5:
            if self.debug_level > 0:
                print(f"\n[ScanMatcher] Warning: Still only found {len(correspondences)} correspondences "
                      f"after {resampling_attempts} resampling attempts.")
            
            # Use initial pose but flag it as a poor match
            return initial_pose.copy(), {
                'iterations': 0,
                'final_score': 0.3,  # Low score to indicate it's not a good match
                'error': float('inf'),
                'correspondences': len(correspondences),
                'resampling_attempts': resampling_attempts
            }
        
        # Now proceed with ICP using the found correspondences
        iterations_data = []
        prev_error = mean_error
        
        # Main ICP loop
        for iteration in range(self.max_iterations):
            # Store the current pose before updating
            prev_pose = current_pose.copy()
            
            # Estimate new transformation that minimizes the distance between corresponding points
            updated_pose = self.estimateTransformation(scan_points, correspondences, current_pose)
            
            # Store iteration data for visualization
            iterations_data.append({
                'iteration': iteration,
                'pose': updated_pose.copy(),
                'error': mean_error,
                'correspondences': len(correspondences),
                'transformed_points': transformed_points.copy()
            })
            
            # Update current pose
            current_pose = updated_pose
            
            # Transform scan points to world frame using the updated pose
            transformed_points = self.transformPointsToWorld(scan_points, current_pose)
            
            # Find new correspondences
            correspondences, mean_error = self.findCorrespondences(transformed_points)
            
            # If we lost too many correspondences, stop
            if len(correspondences) < 5:
                if self.debug_level > 1:
                    print(f"\n[ScanMatcher] Lost correspondences during ICP (down to {len(correspondences)}). Stopping.")
                break
            
            # Check for convergence
            if abs(prev_error - mean_error) < self.convergence_threshold:
                if self.debug_level > 2:
                    print(f"\n[ScanMatcher] ICP converged after {iteration+1} iterations. Error: {mean_error:.6f}")
                break
                
            prev_error = mean_error
        
        # Score the final match
        final_score = self.scoreFinalMatch(scan_points, current_pose)
        
        # Store visualization data
        self.current_visualization_data = {
            'iterations': iterations_data,
            'final_pose': current_pose,
            'initial_pose': initial_pose,
            'scan_points': scan_points,
            'final_score': final_score,
            'resampling_attempts': resampling_attempts
        }
        
        # Return the matched pose and match information
        match_info = {
            'iterations': len(iterations_data),
            'final_score': final_score,
            'error': prev_error,
            'correspondences': len(correspondences),
            'resampling_attempts': resampling_attempts
        }
        
        return current_pose, match_info
    
    def transformPointsToWorld(self, points, pose):
        """
        Transform points from robot frame to world frame
        
        Args:
            points: Array of [x, y] points in robot's local frame
            pose: Robot pose (PoseEstimate object)
            
        Returns:
            Array of transformed points in world frame
        """
        # Extract pose components
        x, y, theta = pose.x, pose.y, pose.theta
        
        # Create rotation matrix
        c = math.cos(theta)
        s = math.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        # Apply rotation
        rotated_points = np.dot(points, rotation_matrix.T)
        
        # Apply translation
        transformed_points = rotated_points + np.array([x, y])
        
        return transformed_points
    
    def findCorrespondences(self, transformed_points):
        """
        Find corresponding points between the transformed scan and the map
        
        Args:
            transformed_points: Array of scan points in world frame
            
        Returns:
            tuple: List of correspondences and mean error
        """
        if self.map is None:
            return [], float('inf')
        
        correspondences = []
        total_error = 0.0
        
        for point in transformed_points:
            # Skip points outside the map with a buffer
            buffer = 1.0  # 1 meter buffer
            if (abs(point[0]) >= self.map.width/2 - buffer or 
                abs(point[1]) >= self.map.height/2 - buffer):
                continue
                
            # Convert to grid coordinates
            grid_x, grid_y = self.map.world_to_grid(point[0], point[1])
            
            # Make sure the grid coordinates are valid
            if not (0 <= grid_x < self.map.grid_width and 0 <= grid_y < self.map.grid_height):
                continue
                
            # Find closest occupied cell within search radius
            closest_cell, distance = self.findClosestOccupiedCell(grid_x, grid_y)
            
            if closest_cell is not None and distance < self.max_correspondence_distance / self.map.resolution:
                # Convert back to world coordinates
                world_x, world_y = self.map.grid_to_world(closest_cell[0], closest_cell[1])
                
                # Add correspondence
                correspondences.append({
                    'scan_point': point,
                    'map_point': np.array([world_x, world_y]),
                    'distance': distance * self.map.resolution  # Convert to meters
                })
                
                total_error += distance * self.map.resolution
        
        mean_error = total_error / len(correspondences) if correspondences else float('inf')
        
        return correspondences, mean_error
    
    def findClosestOccupiedCell(self, grid_x, grid_y):
        """
        Find the closest occupied cell to the given grid coordinates
        
        Args:
            grid_x, grid_y: Grid coordinates to search from
            
        Returns:
            tuple: Closest occupied cell coordinates and distance, or (None, inf) if none found
        """
        # Define search radius (in grid cells)
        search_radius = int(self.max_correspondence_distance / self.map.resolution)
        
        min_distance = float('inf')
        closest_cell = None
        
        # Simple grid search in a square area
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                
                # Check if within grid bounds
                if (0 <= nx < self.map.grid_width and 0 <= ny < self.map.grid_height):
                    # Check if this cell is occupied - USING CURRENT THRESHOLD
                    if self.map.grid[ny, nx] > self.occupancy_threshold:
                        # Calculate Euclidean distance
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_cell = (nx, ny)
        
        return closest_cell, min_distance
    
    def estimateTransformation(self, scan_points, correspondences, current_pose):
        """
        Estimate the transformation that minimizes the distance between corresponding points
        
        Args:
            scan_points: Original scan points in robot frame
            correspondences: List of correspondences between scan and map
            current_pose: Current pose estimate
            
        Returns:
            PoseEstimate: Updated pose estimate
        """
        if not correspondences:
            return current_pose.copy()
        
        try:
            # Extract corresponding points
            scan_points_array = np.array([corr['scan_point'] for corr in correspondences])
            map_points_array = np.array([corr['map_point'] for corr in correspondences])
            
            # Calculate centroids
            scan_centroid = np.mean(scan_points_array, axis=0)
            map_centroid = np.mean(map_points_array, axis=0)
            
            # Center the points
            centered_scan = scan_points_array - scan_centroid
            centered_map = map_points_array - map_centroid
            
            # Compute the covariance matrix
            H = np.dot(centered_scan.T, centered_map)
            
            # Singular Value Decomposition
            U, S, Vt = np.linalg.svd(H)
            
            # Calculate rotation matrix
            R = np.dot(Vt.T, U.T)
            
            # Ensure proper rotation matrix (det=1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            # Calculate translation
            t = map_centroid - np.dot(scan_centroid, R.T)
            
            # Extract rotation angle from rotation matrix
            theta = math.atan2(R[1, 0], R[0, 0])
            
            # Create updated pose
            updated_pose = current_pose.copy()
            updated_pose.x = t[0]
            updated_pose.y = t[1]
            updated_pose.theta = theta
            
            return updated_pose
            
        except Exception as e:
            # If there's any error in the estimation, return the original pose
            if self.debug_level > 0:
                print(f"\n[ScanMatcher] Error in pose estimation: {e}")
            return current_pose.copy()
    
    def calculatePoseChange(self, pose1, pose2):
        """
        Calculate the change between two poses
        
        Args:
            pose1, pose2: PoseEstimate objects
            
        Returns:
            dict: Dictionary with dx, dy, dtheta, distance
        """
        dx = pose2.x - pose1.x
        dy = pose2.y - pose1.y
        dtheta = self.normalize_angle(pose2.theta - pose1.theta)
        
        return {
            'dx': dx,
            'dy': dy,
            'dtheta': dtheta,
            'distance': math.sqrt(dx**2 + dy**2)
        }
    
    def scoreFinalMatch(self, scan_points, pose):
        """
        Score the final match quality
        
        Args:
            scan_points: Original scan points in robot frame
            pose: Final pose estimate
            
        Returns:
            float: Match quality score (higher is better)
        """
        if self.map is None:
            return 0.0
        
        # Transform points to world frame
        world_points = self.transformPointsToWorld(scan_points, pose)
        
        total_score = 0.0
        valid_points = 0
        
        for point in world_points:
            # Skip points outside the map
            if (abs(point[0]) >= self.map.width/2 or abs(point[1]) >= self.map.height/2):
                continue
                
            # Convert to grid coordinates
            grid_x, grid_y = self.map.world_to_grid(point[0], point[1])
            
            # Ensure grid coordinates are valid
            if not (0 <= grid_x < self.map.grid_width and 0 <= grid_y < self.map.grid_height):
                continue
                
            # Get the occupancy value at this point
            occupancy = self.map.grid[grid_y, grid_x]
            
            # Score higher for points that land on occupied cells
            # and lower for points that land on free space
            if occupancy > self.occupancy_threshold:  # Occupied
                total_score += 1.0
            elif occupancy < 0.3:  # Free
                total_score -= 0.5
            
            valid_points += 1
        
        # Normalize score between 0 and 1
        if valid_points > 0:
            normalized_score = (total_score / valid_points + 0.5) / 1.5
            return max(0.0, min(1.0, normalized_score))
        else:
            # If no valid points, return a very low score
            return 0.1
    
    def validateMatch(self, matched_pose, previous_pose, match_info):
        """
        Validate if the match is reasonable
        
        Args:
            matched_pose: New matched pose
            previous_pose: Previous pose
            match_info: Information about the match
            
        Returns:
            bool: True if the match is valid, False otherwise
        """
        # If there were resampling attempts but still few correspondences, be stricter
        min_required_correspondences = 5
        if match_info['resampling_attempts'] > 0:
            min_required_correspondences = 3 + match_info['resampling_attempts']
            
        # If no correspondences were found, match is invalid
        if match_info['correspondences'] < min_required_correspondences:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Too few correspondences ({match_info['correspondences']} < {min_required_correspondences})")
            return False
        
        # Calculate pose change
        pose_change = self.calculatePoseChange(previous_pose, matched_pose)
        
        # Check if the translation is within limits
        if pose_change['distance'] > self.max_translation_per_frame:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Translation too large ({pose_change['distance']:.3f}m > {self.max_translation_per_frame}m)")
            return False
        
        # Check if the rotation is within limits
        if abs(pose_change['dtheta']) > self.max_rotation_per_frame:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Rotation too large ({abs(pose_change['dtheta']):.3f}rad > {self.max_rotation_per_frame}rad)")
            return False
        
        # Check if the match score is reasonable - be more lenient if we had to resample
        score_threshold = 0.3
        if match_info['resampling_attempts'] > 0:
            score_threshold = max(0.2, 0.3 - 0.02 * match_info['resampling_attempts'])
            
        if match_info['final_score'] < score_threshold:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Score too low ({match_info['final_score']:.3f} < {score_threshold})")
            return False
        
        # All checks passed
        return True
    
    def applySmallCorrection(self, odometry_pose, previous_matched_pose):
        """
        Apply a small correction to the odometry pose based on the previous matched pose
        
        Args:
            odometry_pose: Current odometry pose
            previous_matched_pose: Previous matched pose
            
        Returns:
            PoseEstimate: Corrected pose
        """
        # Make sure we have enough history
        if len(self.odometry_trajectory) < 2:
            return odometry_pose.copy()
            
        # Calculate the odometry change from the previous frame
        last_odometry_pose = self.odometry_trajectory[-2]
        odom_change = self.calculatePoseChange(last_odometry_pose, odometry_pose)
        
        # Apply the same change to the previous matched pose, with a slight correction factor
        # (this helps prevent drift by applying a small correction towards the matched trajectory)
        correction_factor = 0.9  # Apply 90% of the odometry change
        
        corrected_pose = previous_matched_pose.copy()
        corrected_pose.x += odom_change['dx'] * correction_factor
        corrected_pose.y += odom_change['dy'] * correction_factor
        corrected_pose.theta += odom_change['dtheta'] * correction_factor
        
        return corrected_pose
    
    def process_scan_sequence(self, scan_data_list, initial_pose=None, update_og=False):
        """
        Process a sequence of scans with tracking - compatible with original API
        
        Args:
            scan_data_list: List of scan data dictionaries
            initial_pose: Optional initial pose dictionary (or None to use first scan's pose)
            update_og: Whether to update occupancy grid with matched scans
            
        Returns:
            tuple: (corrected_poses, confidences) Lists of corrected poses and match confidences
        """
        # Convert the initial pose if provided
        if initial_pose:
            initial_pose_obj = PoseEstimate(
                initial_pose['x'],
                initial_pose['y'],
                initial_pose['theta']
            )
        else:
            initial_pose_obj = None
        
        # Process the dataset with our new implementation
        # Reset any existing state first
        self.trajectory = []
        self.odometry_trajectory = []
        self.last_matched_pose = None
        self.match_qualities = []
        self.match_count = 0
        self.map_built = False
        self.in_recovery_mode = False
        self.consecutive_poor_matches = 0
        
        # Process the entire dataset
        self.process_dataset(scan_data_list, initial_pose_obj)
        
        # Convert the results back to the expected format
        corrected_poses = [pose.to_dict() for pose in self.trajectory]
        
        # Create confidence list - use match qualities if available, otherwise default values
        if self.match_qualities:
            # Create confidence values from match scores
            confidences = [1.0]  # First pose is always confident
            for q in self.match_qualities:
                confidences.append(q['score'])
        else:
            # Default confidences if we don't have match qualities
            confidences = [1.0] * len(corrected_poses)
        
        # Make sure the number of confidences matches the number of poses
        if len(confidences) < len(corrected_poses):
            confidences.extend([0.5] * (len(corrected_poses) - len(confidences)))
        elif len(confidences) > len(corrected_poses):
            confidences = confidences[:len(corrected_poses)]
        
        if self.debug_level > 0:
            print(f"[ScanMatcher] Processed {len(scan_data_list)} scans. Returning {len(corrected_poses)} poses.")
            
            # Calculate statistics on trajectory
            if len(corrected_poses) > 1:
                total_distance = 0
                for i in range(1, len(corrected_poses)):
                    dx = corrected_poses[i]['x'] - corrected_poses[i-1]['x']
                    dy = corrected_poses[i]['y'] - corrected_poses[i-1]['y']
                    total_distance += math.sqrt(dx**2 + dy**2)
                
                print(f"[ScanMatcher] Total trajectory length: {total_distance:.2f} meters")
                print(f"[ScanMatcher] Average confidence: {sum(confidences)/len(confidences):.4f}")
        
        return corrected_poses, confidences
    
    def plotMatchOverlay(self, scan_x, scan_y, pose, ax=None, show_iterations=False):
        """
        Plot the scan overlaid on the map to visualize the match quality
        
        Args:
            scan_x: List of scan x coordinates
            scan_y: List of scan y coordinates
            pose: Current pose estimate (PoseEstimate object)
            ax: Matplotlib axis to plot on (or None to create new figure)
            show_iterations: Whether to show the ICP iterations
            
        Returns:
            Matplotlib axis with the plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create scan points array
        scan_points = np.column_stack((np.array(scan_x), np.array(scan_y)))
        
        # Transform scan points using the pose
        transformed_points = self.transformPointsToWorld(scan_points, pose)
        
        # Plot the map if we have one
        if self.map:
            # Handle different grid access methods
            if hasattr(self.map, 'get_grid_for_display'):
                grid_data = self.map.get_grid_for_display()
            else:
                grid_data = self.map.get_grid()
                
            # Custom colormap: white (unknown), black (occupied), light gray (free)
            cmap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                grid_data,
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-self.map.width/2, self.map.width/2, -self.map.height/2, self.map.height/2]
            )
            
            # Draw map boundaries
            ax.axhline(y=-self.map.height/2, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=self.map.height/2, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=-self.map.width/2, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.map.width/2, color='red', linestyle='--', alpha=0.5)
        
        # Plot the transformed scan points
        ax.scatter(transformed_points[:, 0], transformed_points[:, 1], c='red', s=3, label='Scan Points')
        
        # Plot the robot position
        ax.scatter(pose.x, pose.y, c='blue', s=100, marker='*', label='Robot Position')
        
        # Plot orientation arrow
        arrow_length = 0.5
        dx = arrow_length * math.cos(pose.theta)
        dy = arrow_length * math.sin(pose.theta)
        
        ax.arrow(
            pose.x, pose.y, dx, dy,
            head_width=0.1, head_length=0.1, fc='blue', ec='blue'
        )
        
        # Plot search radius circle to visualize correspondence distance
        search_circle = Circle((pose.x, pose.y), 
                              self.max_correspondence_distance,
                              color='blue', fill=False, alpha=0.3)
        ax.add_patch(search_circle)
        
        # If we have visualization data and want to show iterations
        if show_iterations and self.current_visualization_data:
            data = self.current_visualization_data
            
            # Plot initial pose
            ax.scatter(
                data['initial_pose'].x, 
                data['initial_pose'].y, 
                c='orange', s=100, marker='o', 
                label='Initial Pose'
            )
            
            # Plot iteration poses with color gradient
            iterations = data['iterations']
            if iterations:
                colors_iter = plt.cm.viridis(np.linspace(0, 1, len(iterations)))
                
                for i, iter_data in enumerate(iterations):
                    iter_pose = iter_data['pose']
                    ax.scatter(
                        iter_pose.x, iter_pose.y, 
                        c=[colors_iter[i]], s=50, alpha=0.7,
                        marker='x'
                    )
                
                # Add a custom legend entry for iterations
                ax.scatter([], [], c='green', marker='x', s=50, label='ICP Iterations')
            
            # Add match score and resampling info to the plot
            info_text = f"Match Score: {data['final_score']:.3f}"
            if data.get('resampling_attempts', 0) > 0:
                info_text += f"\nResampling Attempts: {data['resampling_attempts']}"
                
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    va='top', ha='left', color='blue', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # If aggressive resampling was used, also show the aggressive search radius
            if data.get('resampling_attempts', 0) > 0:
                aggressive_circle = Circle((pose.x, pose.y), 
                                        self.aggressive_max_correspondence_distance,
                                        color='red', fill=False, alpha=0.2, linestyle='--')
                ax.add_patch(aggressive_circle)
                ax.scatter([], [], c='red', marker='o', s=0, label=f'Aggressive Search ({self.aggressive_max_correspondence_distance}m)', 
                          linestyle='--', alpha=0.2)
        
        # Add grid and labels
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Scan-Map Match Overlay')
        ax.legend(loc='upper right')
        
        return ax
    
    def visualize_trajectory(self, pose_list, confidences=None):
        """
        Visualize the trajectory and occupancy grid - compatible with original API
        
        Args:
            pose_list: List of corrected poses (dictionaries)
            confidences: Optional list of match confidences
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        if not pose_list:
            print("No poses to visualize")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Setup for trajectory visualization
        colors = iter(cm.rainbow(np.linspace(1, 0, len(pose_list) + 1)))
        
        # Extract path coordinates
        x_trajectory = [pose['x'] for pose in pose_list]
        y_trajectory = [pose['y'] for pose in pose_list]
        
        # Plot trajectory
        plt.plot(x_trajectory, y_trajectory, 'k-', linewidth=1.5, label='Robot Path')
        
        # Highlight start and end
        plt.scatter(x_trajectory[0], y_trajectory[0], color='green', s=100, marker='*', label='Start')
        plt.scatter(x_trajectory[-1], y_trajectory[-1], color='red', s=100, marker='*', label='End')
        
        # Plot intermediate positions with colors reflecting confidence if available
        if confidences:
            # Normalize confidences for colormap
            norm_conf = np.array(confidences)
            norm_conf = (norm_conf - norm_conf.min()) / (norm_conf.max() - norm_conf.min() + 1e-10)
            
            # Plot each position
            for i, (x, y, conf) in enumerate(zip(x_trajectory, y_trajectory, norm_conf)):
                if i % 10 == 0:  # Plot every 10th point to avoid clutter
                    plt.scatter(x, y, color=cm.jet(conf), s=30, alpha=0.7)
        else:
            # Plot every 10th position with rainbow colors
            for i, (x, y) in enumerate(zip(x_trajectory, y_trajectory)):
                if i % 10 == 0:
                    plt.scatter(x, y, color=next(colors), s=30)
        
        # Plot occupancy grid as background if available
        if self.map:
            self.plot_occupancy_grid(plt.gca())
        
        plt.title("Robot Trajectory on Occupancy Grid")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_occupancy_grid(self, ax=None):
        """
        Plot the occupancy grid - compatible with original API
        
        Args:
            ax: Matplotlib axis to plot on (creates a new figure if None)
            
        Returns:
            Matplotlib image object
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        if not self.map:
            print("No occupancy grid to plot")
            return None
        
        # Get grid data
        if hasattr(self.map, 'get_grid_for_display'):
            grid_data = self.map.get_grid_for_display()
        else:
            grid_data = self.map.get_grid()
        
        # Custom colormap: white (unknown), black (occupied), light gray (free)
        colormap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'black'])
        bounds = [0, 0.4, 0.6, 1]
        norm = plt.cm.colors.BoundaryNorm(bounds, colormap.N)
        
        # Plot the grid
        width = self.map.width
        height = self.map.height
        img = ax.imshow(grid_data, 
                       cmap=colormap, 
                       norm=norm, 
                       origin='lower',
                       extent=[-width/2, width/2, -height/2, height/2],
                       alpha=0.7)  # Make slightly transparent
        
        # Add gridlines
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add colorbar if this is a standalone plot
        if ax is None:
            plt.colorbar(img, ax=ax, label='Occupancy Probability')
            ax.set_title('Occupancy Grid Map')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            plt.tight_layout()
            
        return img
    
    def visualizeIcpProcess(self):
        """
        Create a comprehensive visualization of the ICP process
        
        Returns:
            Matplotlib figure with the visualization
        """
        import matplotlib.pyplot as plt
        
        if not self.current_visualization_data:
            print("[ScanMatcher] No visualization data available.")
            return None
        
        data = self.current_visualization_data
        iterations = data['iterations']
        
        if not iterations:
            print("[ScanMatcher] No iteration data available.")
            return None
        
        # Create figure with multiple subplots
        n_iterations = min(4, len(iterations))  # Show at most 4 iterations
        fig, axes = plt.subplots(1, n_iterations + 1, figsize=(5 * (n_iterations + 1), 5))
        
        # Handle the case where n_iterations is 0 (single plot)
        if n_iterations == 0:
            axes = [axes]
        
        # Plot the map in all subplots
        if self.map:
            # Handle different grid access methods
            if hasattr(self.map, 'get_grid_for_display'):
                grid_data = self.map.get_grid_for_display()
            else:
                grid_data = self.map.get_grid()
                
            cmap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            for ax in axes:
                ax.imshow(
                    grid_data,
                    cmap=cmap, norm=norm,
                    origin='lower',
                    extent=[-self.map.width/2, self.map.width/2, -self.map.height/2, self.map.height/2]
                )
                ax.set_aspect('equal')
                ax.grid(True)
        
        # Plot initial state
        axes[0].scatter(
            data['initial_pose'].x, 
            data['initial_pose'].y, 
            c='orange', s=100, marker='o', 
            label='Initial Pose'
        )
        
        # Transform points using initial pose
        initial_transformed = self.transformPointsToWorld(data['scan_points'], data['initial_pose'])
        axes[0].scatter(
            initial_transformed[:, 0], 
            initial_transformed[:, 1], 
            c='orange', s=3, alpha=0.7,
            label='Initial Points'
        )
        
        # Show search radius
        search_circle = Circle(
            (data['initial_pose'].x, data['initial_pose'].y), 
            self.max_correspondence_distance,
            color='blue', fill=False, alpha=0.3
        )
        axes[0].add_patch(search_circle)
        
        # If resampling was used, show that info
        if data.get('resampling_attempts', 0) > 0:
            info_text = f"Initial State\nResampling: {data['resampling_attempts']} attempts"
            
            # Also show aggressive search radius
            aggressive_circle = Circle(
                (data['initial_pose'].x, data['initial_pose'].y), 
                self.aggressive_max_correspondence_distance,
                color='red', fill=False, alpha=0.2, linestyle='--'
            )
            axes[0].add_patch(aggressive_circle)
        else:
            info_text = "Initial State"
            
        axes[0].set_title(info_text)
        axes[0].legend()
        
        # Plot iteration states
        if len(iterations) > 0:
            selected_indices = np.linspace(0, len(iterations) - 1, n_iterations, dtype=int)
            
            for i, idx in enumerate(selected_indices):
                if i >= len(axes) - 1:  # Skip if we don't have enough axes
                    break
                    
                iter_data = iterations[idx]
                ax = axes[i + 1]
                
                # Plot the transformed points
                if 'transformed_points' in iter_data:
                    ax.scatter(
                        iter_data['transformed_points'][:, 0], 
                        iter_data['transformed_points'][:, 1], 
                        c='red', s=3, alpha=0.7,
                        label='Scan Points'
                    )
                
                # Plot the pose
                ax.scatter(
                    iter_data['pose'].x, 
                    iter_data['pose'].y, 
                    c='blue', s=100, marker='*', 
                    label='Robot Pose'
                )
                
                # Plot orientation arrow
                arrow_length = 0.5
                dx = arrow_length * math.cos(iter_data['pose'].theta)
                dy = arrow_length * math.sin(iter_data['pose'].theta)
                
                ax.arrow(
                    iter_data['pose'].x, iter_data['pose'].y, dx, dy,
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue'
                )
                
                # Add iteration info
                info_text = (
                    f"Iteration {iter_data['iteration'] + 1}\n"
                    f"Error: {iter_data['error']:.4f}\n"
                    f"Correspondences: {iter_data['correspondences']}"
                )
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                        va='top', ha='left', color='blue', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7))
                
                ax.set_title(f"Iteration {iter_data['iteration'] + 1}")
                ax.legend()
        
        plt.tight_layout()
        return fig


def main():
    """Main function to demonstrate the improved scan matcher"""
    import argparse
    import matplotlib.pyplot as plt
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Improved Scan Matcher Example')
    parser.add_argument('--file', '-f', type=str, 
                       default="../dataset/raw_data/raw_data_zjnu20_21_3F.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max-entries', '-m', type=int, default=1000,
                       help='Maximum number of entries to read from the file')
    parser.add_argument('--resolution', '-r', type=float, default=0.05,
                       help='Grid resolution in meters')
    parser.add_argument('--visualize', '-v', action='store_true', default=True,
                       help='Visualize the results')
    parser.add_argument('--debug-level', '-d', type=int, default=1, choices=[0, 1, 2, 3],
                       help='Debug level: 0=none, 1=basic, 2=detailed, 3=verbose')
    args = parser.parse_args()
    
    # Read data from file
    print(f"Reading LiDAR data from: {args.file}")
    sys.path.append('lidar_slam')
    data_list = data_parser.read_lidar_data_from_file(args.file, args.max_entries)
    
    if not data_list:
        print("No data was read from the file.")
        return
    
    # Create occupancy grid
    print(f"Creating occupancy grid with resolution {args.resolution}m")
    grid = occupancy_grid.OccupancyGrid(
        resolution=args.resolution,
        width=30,
        height=30
    )
    
    # Create improved scan matcher
    print("Initializing improved scan matcher")
    matcher = ScanMatcher(
        occupancy_grid_map=grid,
        debug_level=args.debug_level
    )
    
    # Set LiDAR parameters
    matcher.set_lidar_params(
        max_range=10.0,
        field_of_view=math.pi,
        num_samples=len(data_list[0]['scan_ranges'])
    )
    
    # Configure scan orientation
    matcher.set_scan_config({
        'flip_x': False,
        'flip_y': False,
        'reverse_scan': True,
        'flip_theta': False
    })
    
    # Process scans with scan matching
    print("Processing scans with scan matching...")
    start_time = time.time()
    corrected_poses, confidences = matcher.process_scan_sequence(
        data_list, 
        update_og=True
    )
    processing_time = time.time() - start_time
    
    # Print statistics
    print(f"Processed {len(data_list)} scans in {processing_time:.2f} seconds")
    print(f"Average processing time: {processing_time/len(data_list)*1000:.2f} ms per scan")
    
    # Calculate trajectory length
    total_distance = 0
    for i in range(1, len(corrected_poses)):
        dx = corrected_poses[i]['x'] - corrected_poses[i-1]['x']
        dy = corrected_poses[i]['y'] - corrected_poses[i-1]['y']
        total_distance += math.sqrt(dx**2 + dy**2)
    
    print(f"Total trajectory length: {total_distance:.2f} meters")
    
    # Save results
    print("Saving corrected map...")
    maps_dir = "maps"
    file_utils.ensure_directory_exists(maps_dir)
    timestamp = int(time.time())
    filename = os.path.join(maps_dir, f"scan_matcher_map_{timestamp}")
    
    # Extract path
    path = [(pose['x'], pose['y']) for pose in corrected_poses]
    start_pos = path[0] if path else None
    end_pos = path[-1] if path else None
    
    # Save as image
    file_utils.save_grid_as_image(
        grid.get_grid(),
        filename,
        resolution=grid.resolution,
        width=grid.width,
        height=grid.height,
        robot_path=path,
        start_position=start_pos,
        current_position=end_pos
    )
    
    # Save grid data
    grid.save_to_file(filename, format='png', include_metadata=True)
    
    # Visualize if requested
    if args.visualize:
        print("Visualizing results...")
        matcher.visualize_trajectory(corrected_poses, confidences)
        
        # Also create a visualization of a specific scan match if we have data
        if matcher.current_visualization_data:
            print("Creating ICP process visualization...")
            icp_fig = matcher.visualizeIcpProcess()
            if icp_fig:
                plt.figure(icp_fig.number)
                plt.savefig(os.path.join(maps_dir, f"icp_process_{timestamp}.png"), dpi=300)
    
    print(f"Results saved to {filename}")
    print("Done!")

if __name__ == "__main__":
    main()