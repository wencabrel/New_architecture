import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import math
import os
import time
from matplotlib.widgets import Button
import sys

# Import our utility functions and classes
from lidar_utility_functions import parse_lidar_data, convert_scans_to_cartesian, read_lidar_data_from_file
from occupancy_grid_class import OccupancyGrid

class PoseEstimate:
    """Class to represent a robot pose estimate with uncertainty"""
    
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """Initialize a pose estimate"""
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

class ImprovedScanMatchingLocalization:
    """
    Improved implementation of scan matching localization using ICP algorithm
    with aggressive resampling to never accept unmapped regions.
    """
    
    def __init__(self, occupancy_grid=None, debug_level=1):
        """
        Initialize the improved scan matching system
        
        Args:
            occupancy_grid: OccupancyGrid object representing the map
            debug_level: 0=none, 1=basic info, 2=detailed, 3=verbose
        """
        self.map = occupancy_grid
        self.debug_level = debug_level
        
        # Default ICP parameters 
        self.max_iterations = 15
        self.convergence_threshold = 0.001
        self.max_correspondence_distance = 1.5  # meters
        
        # Parameters for adaptive adjustment
        self.default_correspondence_distance = 1.5  # Starting value
        self.max_possible_correspondence_distance = 5.0  # Maximum allowed value - INCREASED for resampling
        self.min_correspondence_distance = 1.0  # Minimum allowed value
        
        # Occupancy threshold 
        self.occupancy_threshold = 0.55
        self.default_occupancy_threshold = 0.55  # Starting value
        self.min_occupancy_threshold = 0.03  # Minimum allowed value - LOWERED for resampling
        
        # Motion validation parameters 
        self.max_translation_per_frame = 0.05  # meters
        self.default_max_translation = 0.05  # Starting value
        self.max_possible_translation = 1.0  # Maximum allowed value
        
        self.max_rotation_per_frame = 0.05  # radians (~28 degrees)
        self.default_max_rotation = 0.05  # Starting value
        self.max_possible_rotation = 0.08  # Maximum allowed value
        
        # Resampling parameters
        self.max_resampling_attempts = 5  # Maximum number of resampling attempts
        self.resampling_radius_multiplier = 1.5  # How much to increase search radius each attempt
        self.resampling_threshold_reduction = 0.1  # How much to lower threshold each attempt
        
        # Current estimated trajectory
        self.trajectory = []  # List of PoseEstimate objects
        
        # Last matched pose (used as reference for next match)
        self.last_matched_pose = None
        
        # Keep track of odometry poses for comparison
        self.odometry_trajectory = []
        
        # Flag to determine if we've loaded a map or are building it
        self.mapping_mode = True if occupancy_grid is None else False
        
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
        
        # Performance tracking for resampling
        self.resampling_count = 0
        self.resampling_success_rate = 0
        
        if self.debug_level > 0:
            print("[ScanMatcher] Initialized with aggressive resampling strategy")
            print(f"[ScanMatcher]   - Base correspondence distance: {self.default_correspondence_distance}m (can increase to {self.max_possible_correspondence_distance}m)")
            print(f"[ScanMatcher]   - Base occupancy threshold: {self.occupancy_threshold} (can decrease to {self.min_occupancy_threshold})")
            print(f"[ScanMatcher]   - Max resampling attempts: {self.max_resampling_attempts}")
            print(f"[ScanMatcher]   - Alignment reset interval: {self.reset_interval} frames (drift threshold: {self.drift_threshold}m)")
            print("\n" + "="*80)
            print("        POSE INFORMATION FOR EACH SCAN MATCH WILL BE PRINTED BELOW")
            print("="*80 + "\n")
    
    def processSensorData(self, lidar_data, initial_pose=None, angle_min=-math.pi/2, angle_max=math.pi/2, 
                         flip_x=False, flip_y=False, reverse_scan=False, flip_theta=False):
        """
        Process a sequence of LiDAR scans to localize the robot
        
        Args:
            lidar_data: List of parsed LiDAR data dictionaries
            initial_pose: Initial pose estimate (PoseEstimate object or None)
            angle_min: Starting angle of the scan (radians)
            angle_max: Ending angle of the scan (radians)
            flip_x: Whether to flip the x-axis
            flip_y: Whether to flip the y-axis
            reverse_scan: Whether to reverse the scan direction
            flip_theta: Whether to negate the orientation angle
            
        Returns:
            List of updated pose estimates (trajectory)
        """
        if self.debug_level > 0:
            print("[ScanMatcher] Processing sensor data with ICP scan matching (aggressive resampling)...")
        
        # Initialize trajectory with initial pose if provided
        if initial_pose:
            self.last_matched_pose = initial_pose
            self.trajectory = [initial_pose.copy()]
            self.odometry_trajectory = [initial_pose.copy()]
        else:
            # Use the first scan's pose as initial estimate
            first_pose_dict = lidar_data[0]['pose']
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
        for i, scan_data in enumerate(lidar_data):
            if i == 0 and initial_pose:
                # Skip first scan if we already set the initial pose
                continue
            
            if self.debug_level > 0:
                print(f"\r[ScanMatcher] Processing scan {i+1}/{len(lidar_data)}", end="")
            
            # First, store the odometry pose
            odometry_pose = PoseEstimate().from_dict(scan_data['pose'])
            self.odometry_trajectory.append(odometry_pose.copy())
            
            # Convert from polar to Cartesian coordinates
            scan_x, scan_y = convert_scans_to_cartesian(
                scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
                flip_x=flip_x, flip_y=flip_y, reverse_scan=reverse_scan, flip_theta=flip_theta
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
                        'final_score': 0.5,
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
                    # Normal mode - match current scan against the map using ICP with resampling
                    matched_pose, match_info = self.matchScanWithResampling(scan_x, scan_y, initial_guess)
                    
                    # Check if we should expand the map based on boundary points
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
            print(f"\n[ScanMatcher] Processed {len(lidar_data)} scans. Trajectory contains {len(self.trajectory)} poses.")
            if self.resampling_count > 0:
                print(f"[ScanMatcher] Resampling stats: {self.resampling_count} attempts, "
                      f"{self.resampling_success_rate/self.resampling_count*100:.1f}% success rate")
        return self.trajectory
    
    def matchScanWithResampling(self, scan_x, scan_y, initial_pose):
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
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Save original parameters for restoring later
        original_correspondence_distance = self.max_correspondence_distance
        original_occupancy_threshold = self.occupancy_threshold
        
        # Initialize resampling variables
        resampling_attempts = 0
        current_correspondence_distance = self.max_correspondence_distance
        current_occupancy_threshold = self.occupancy_threshold
        
        # Resampling loop - keep trying with more aggressive parameters until we find correspondences
        while resampling_attempts <= self.max_resampling_attempts:
            if resampling_attempts > 0:
                # Increase search radius and decrease threshold for each resampling attempt
                current_correspondence_distance = min(
                    self.max_possible_correspondence_distance,
                    current_correspondence_distance * self.resampling_radius_multiplier
                )
                current_occupancy_threshold = max(
                    self.min_occupancy_threshold,
                    current_occupancy_threshold - self.resampling_threshold_reduction
                )
                
                # Set the new parameters
                self.max_correspondence_distance = current_correspondence_distance
                self.occupancy_threshold = current_occupancy_threshold
                
                if self.debug_level > 0:
                    print(f"\n[ScanMatcher] Resampling attempt {resampling_attempts}/{self.max_resampling_attempts} with: "
                          f"distance = {current_correspondence_distance:.2f}m, threshold = {current_occupancy_threshold:.2f}")
            
            # Try matching with current parameters
            matched_pose, match_info = self.matchScan(scan_x, scan_y, initial_pose)
            
            # Check if we found correspondences
            if match_info['correspondences'] >= 5:  # Minimum acceptable number of correspondences
                # Resampling succeeded
                self.resampling_count += resampling_attempts
                self.resampling_success_rate += 1
                
                # Add resampling info to match_info
                match_info['resampling_attempts'] = resampling_attempts
                
                if resampling_attempts > 0 and self.debug_level > 0:
                    print(f"[ScanMatcher] ✓ Resampling succeeded after {resampling_attempts} attempt(s). "
                          f"Found {match_info['correspondences']} correspondences.")
                
                # Restore original parameters
                self.max_correspondence_distance = original_correspondence_distance
                self.occupancy_threshold = original_occupancy_threshold
                
                return matched_pose, match_info
            
            # Increment resampling attempts
            resampling_attempts += 1
        
        # If we get here, all resampling attempts failed
        self.resampling_count += resampling_attempts
        
        if self.debug_level > 0:
            print(f"\n[ScanMatcher] ✗ Resampling failed after {resampling_attempts} attempts. "
                  f"Using most aggressive attempt result anyway.")
        
        # Use the final match info from the most aggressive attempt
        match_info['resampling_attempts'] = resampling_attempts
        
        # Restore original parameters
        self.max_correspondence_distance = original_correspondence_distance
        self.occupancy_threshold = original_occupancy_threshold
        
        return matched_pose, match_info
    
    def matchScan(self, scan_x, scan_y, initial_pose):
        """
        Match the current scan against the map using ICP algorithm
        
        Args:
            scan_x: List of scan x coordinates
            scan_y: List of scan y coordinates
            initial_pose: Initial pose estimate (PoseEstimate object)
            
        Returns:
            Updated pose estimate (PoseEstimate object) and match info dictionary
        """
        # Create points array from scan
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Initialize transformation from initial pose
        current_pose = initial_pose.copy()
        prev_error = float('inf')
        
        # For visualization
        iterations_data = []
        
        # Main ICP loop
        for iteration in range(self.max_iterations):
            # Transform scan points to world frame using current pose
            transformed_points = self.transformPointsToWorld(scan_points, current_pose)
            
            # Find correspondences between scan points and map
            correspondences, mean_error = self.findCorrespondences(transformed_points)
            
            # Debug print for correspondences
            if self.debug_level > 2 and iteration == 0:
                n_points = len(transformed_points)
                n_correspondences = len(correspondences)
                print(f"\n[ScanMatcher] Iteration {iteration}: Found {n_correspondences}/{n_points} "
                      f"correspondences ({n_correspondences/max(1,n_points)*100:.1f}%)")
            
            if len(correspondences) < 3:  # Not enough correspondences
                if self.debug_level > 1:
                    print(f"\n[ScanMatcher] Warning: Only {len(correspondences)} correspondences found in iteration {iteration+1}.")
                break
            
            # Store the current pose before updating
            prev_pose = current_pose.copy()
            
            # Estimate new transformation that minimizes the distance between corresponding points
            updated_pose = self.estimateTransformation(scan_points, correspondences, current_pose)
            
            # Calculate the change in pose
            pose_change = self.calculatePoseChange(current_pose, updated_pose)
            
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
            
            # Check for convergence
            if abs(prev_error - mean_error) < self.convergence_threshold:
                if self.debug_level > 2:
                    print(f"\n[ScanMatcher] ICP converged after {iteration+1} iterations. Error: {mean_error:.6f}")
                break
                
            prev_error = mean_error
        
        # Score the final match - FIXED to handle cases with no valid points
        final_score = self.scoreFinalMatch(scan_points, current_pose)
        
        # Store visualization data
        self.current_visualization_data = {
            'iterations': iterations_data,
            'final_pose': current_pose,
            'initial_pose': initial_pose,
            'scan_points': scan_points,
            'final_score': final_score
        }
        
        # Return the matched pose and match information
        match_info = {
            'iterations': len(iterations_data),
            'final_score': final_score,
            'error': prev_error,
            'correspondences': len(correspondences) if correspondences is not None else 0
        }
        
        return current_pose, match_info
    
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
                'error': match_info['error'],
                'resampling_attempts': match_info.get('resampling_attempts', 0)
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
        avg_resampling = sum(q.get('resampling_attempts', 0) for q in self.match_qualities) / len(self.match_qualities)
        
        # Check if we're having matching problems
        poor_match = avg_score < self.match_quality_threshold or avg_correspondences < 10 or avg_resampling > 1
        
        # Get the most recent match result
        last_match = self.match_qualities[-1]
        
        # Calculate adaptive parameter adjustments
        if poor_match:
            self.consecutive_poor_matches += 1
            
            # Adaptively increase search parameters based on consecutive poor matches
            adjustment_factor = min(1.0, 0.2 * self.consecutive_poor_matches + 0.1 * avg_resampling)
            
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
            self.consecutive_poor_matches = 0
            
            # Gradually move back towards defaults (10% step)
            self.max_correspondence_distance = self.max_correspondence_distance * 0.9 + self.default_correspondence_distance * 0.1
            self.occupancy_threshold = self.occupancy_threshold * 0.9 + self.default_occupancy_threshold * 0.1
            self.max_translation_per_frame = self.max_translation_per_frame * 0.9 + self.default_max_translation * 0.1
            self.max_rotation_per_frame = self.max_rotation_per_frame * 0.9 + self.default_max_rotation * 0.1
    
    def checkAndResetAlignment(self):
        """
        Check alignment between odometry and scan matcher, and reset if necessary
        
        Returns:
            True if alignment was reset, False otherwise
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
            Recovery pose
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
            True if map expansion is needed
        """
        if self.map is None:
            return False
            
        # Create points array from scan
        scan_points = np.column_stack((scan_x, scan_y))
        
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
                # Include resampling attempts info if available
                resampling_attempts = match_info.get('resampling_attempts', 0)
                if resampling_attempts > 0:
                    print(f"MATCH INFO:       Score={match_info['final_score']:.4f}, Iterations={match_info['iterations']}, "
                          f"Error={match_info['error']:.6f}, Correspondences={match_info['correspondences']}, "
                          f"Resampling attempts={resampling_attempts}")
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
            List of correspondences and mean error
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
            Closest occupied cell coordinates and distance
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
                    # Check if this cell is occupied - USING LOWER THRESHOLD
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
            Updated pose estimate
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
            Dictionary with dx, dy, dtheta
        """
        dx = pose2.x - pose1.x
        dy = pose2.y - pose1.y
        dtheta = (pose2.theta - pose1.theta + math.pi) % (2 * math.pi) - math.pi  # Normalize angle difference
        
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
            Match quality score (higher is better)
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
        
        # Normalize score between 0 and 1 - FIXED to handle zero valid points
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
            Boolean indicating if the match is valid
        """
        # If resampling was needed, but still found few correspondences, consider invalid
        if match_info.get('resampling_attempts', 0) >= self.max_resampling_attempts - 1 and match_info['correspondences'] < 10:
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Required {match_info.get('resampling_attempts')} resampling attempts "
                      f"but only found {match_info['correspondences']} correspondences.")
            return False
            
        # If no correspondences were found after resampling, match is invalid
        if match_info['correspondences'] < 3:  # Lowered from 5 to 3 due to resampling
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Too few correspondences ({match_info['correspondences']} < 3)")
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
        
        # Check if the match score is reasonable
        if match_info['final_score'] < 0.2:  # Lowered threshold from 0.3 to 0.2 due to resampling
            if self.debug_level > 1:
                print(f"\n[ScanMatcher] Match rejected: Score too low ({match_info['final_score']:.3f} < 0.2)")
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
            Corrected pose
        """
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create scan points array
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Transform scan points using the pose
        transformed_points = self.transformPointsToWorld(scan_points, pose)
        
        # Plot the map if we have one
        if self.map:
            # Custom colormap: white (unknown), black (occupied), light gray (free)
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                self.map.get_grid_for_display(),
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
        search_circle = plt.Circle((pose.x, pose.y), 
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
            
            # Add match score to the plot
            score_text = f"Match Score: {data['final_score']:.3f}"
            ax.text(0.02, 0.98, score_text, transform=ax.transAxes, 
                    va='top', ha='left', color='blue', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # Add grid and labels
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Scan-Map Match Overlay')
        ax.legend(loc='upper right')
        
        return ax
    
    def visualizeIcpProcess(self):
        """
        Create a comprehensive visualization of the ICP process
        
        Returns:
            Matplotlib figure with the visualization
        """
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
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            for ax in axes:
                ax.imshow(
                    self.map.get_grid_for_display(),
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
        search_circle = plt.Circle(
            (data['initial_pose'].x, data['initial_pose'].y), 
            self.max_correspondence_distance,
            color='blue', fill=False, alpha=0.3
        )
        axes[0].add_patch(search_circle)
        
        # If resampling was used, show that info
        if data.get('resampling_attempts', 0) > 0:
            info_text = f"Initial State\nResampling: {data['resampling_attempts']} attempts"
            
            # Also show aggressive search radius
            aggressive_circle = plt.Circle(
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

    def visualize_map_and_scan(self, scan_x, scan_y, pose):
        """Create a visualization of the map and current scan for debugging"""
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create points array from scan
        scan_points = np.column_stack((scan_x, scan_y))
        
        # Transform points to world frame
        world_points = self.transformPointsToWorld(scan_points, pose)
        
        # Plot the map
        if self.map:
            # Custom colormap
            cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            ax.imshow(
                self.map.get_grid_for_display(),
                cmap=cmap, norm=norm,
                origin='lower',
                extent=[-self.map.width/2, self.map.width/2, -self.map.height/2, self.map.height/2]
            )
            
            # Count the number of occupied cells
            try:
                occupied_cells = np.sum(self.map.grid > self.occupancy_threshold)
                total_cells = self.map.grid_width * self.map.grid_height
                
                ax.set_title(f"Map Visualization - {occupied_cells} occupied cells ({occupied_cells/total_cells*100:.2f}%)")
            except:
                ax.set_title("Map Visualization")
        
        # Plot odometry trajectory
        odom_x = [pose.x for pose in self.odometry_trajectory]
        odom_y = [pose.y for pose in self.odometry_trajectory]
        ax.plot(odom_x, odom_y, 'r-', linewidth=1, alpha=0.5, label='Odometry')
        
        # Plot matched trajectory
        matched_x = [pose.x for pose in self.trajectory]
        matched_y = [pose.y for pose in self.trajectory]
        ax.plot(matched_x, matched_y, 'g-', linewidth=1, label='Matched')
        
        # Plot scan points
        ax.scatter(world_points[:, 0], world_points[:, 1], c='blue', s=3, alpha=0.5, label='Current Scan')
        
        # Plot the current position from both odometry and matched pose
        if len(self.odometry_trajectory) > 0:
            ax.scatter(self.odometry_trajectory[-1].x, self.odometry_trajectory[-1].y, 
                      c='red', s=100, marker='*', label='Odometry Position')
        
        if len(self.trajectory) > 0:
            ax.scatter(self.trajectory[-1].x, self.trajectory[-1].y, 
                      c='green', s=100, marker='*', label='Matched Position')
        
        # Draw map boundaries
        ax.axhline(y=-self.map.height/2, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=self.map.height/2, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=-self.map.width/2, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=self.map.width/2, color='red', linestyle='--', alpha=0.5)
        
        # Add search radius visualization around current matched position
        if len(self.trajectory) > 0:
            current_pos = self.trajectory[-1]
            search_circle = plt.Circle((current_pos.x, current_pos.y), 
                                      self.max_correspondence_distance,
                                      color='blue', fill=False, alpha=0.3)
            ax.add_patch(search_circle)
            
            # Also show the aggressive search radius
            aggressive_circle = plt.Circle((current_pos.x, current_pos.y), 
                                        self.aggressive_max_correspondence_distance,
                                        color='red', fill=False, alpha=0.2, linestyle='--')
            ax.add_patch(aggressive_circle)
        
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save the figure to a file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"map_scan_debug_{timestamp}.png", dpi=150)
        
        print(f"\n[ScanMatcher] Map visualization saved to map_scan_debug_{timestamp}.png")
        
        # Close the figure to free memory
        plt.close(fig)