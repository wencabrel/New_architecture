"""
SLAM Coordinator Module

This module coordinates the interaction between data parsing, scan matching,
pose estimation, and mapping components for a complete LiDAR SLAM system.
"""

import numpy as np
import os
import sys
import time
from enum import Enum

# Add the parent directory to the path to import other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules from existing system
from lidar_processing import scan_converter
from lidar_processing.scan_matcher import ScanMatcher
from pose_estimation.pose_estimator import PoseEstimator, PoseEstimationMethod, Pose2D
from mapping import occupancy_grid

class SLAMMode(Enum):
    """Enumeration of available SLAM operation modes"""
    ODOMETRY_ONLY = "odometry_only"  # Use only odometry for mapping
    SCAN_MATCHING = "scan_matching"  # Use scan matching for pose correction
    FULL_SLAM = "full_slam"  # Use full SLAM with loop closure (future)

class SLAMCoordinator:
    """Class to coordinate the full SLAM pipeline"""
    
    def __init__(self, mode=SLAMMode.SCAN_MATCHING, grid_resolution=0.05):
        """
        Initialize the SLAM coordinator
        
        Args:
            mode (SLAMMode): Operation mode for SLAM
            grid_resolution (float): Resolution of the occupancy grid in meters
        """
        self.mode = mode
        self.grid_resolution = grid_resolution
        
        # Occupancy grid will be initialized when first scan is processed
        self.occupancy_grid = None
        
        # Choose pose estimation method based on SLAM mode
        pose_method = PoseEstimationMethod.ODOMETRY_ONLY
        if mode == SLAMMode.SCAN_MATCHING:
            pose_method = PoseEstimationMethod.EKF_FUSION
        
        self.pose_estimator = PoseEstimator(method=pose_method)
        
        # Initialize scan matcher (will be set up properly after grid is created)
        self.scan_matcher = None
        self.scan_matcher_params = {
            'search_radius': 1.4,
            'search_half_rad': 0.25,
            'scan_sigma_in_num_grid': 2,
            'move_r_sigma': 0.1,
            'max_move_deviation': 0.25,
            'turn_sigma': 0.3,
            'mismatch_prob_at_coarse': 0.15,
            'coarse_factor': 5
        }
        
        # For storing previous scan data
        self.previous_scan = None
        self.previous_scan_points = None
        
        # For debugging and visualization
        self.scan_match_results = []
        self.pose_corrections = []
        
        # Scan matching parameters
        self.min_points_for_matching = 50  # Minimum number of points needed for scan matching
        self.scan_match_interval = 1  # Process every Nth scan (1 = every scan)
        self.scan_counter = 0
        
        # LiDAR configuration
        self.angle_min = -np.pi/2
        self.angle_max = np.pi/2
        self.scan_config = {
            'flip_x': False,
            'flip_y': False,
            'reverse_scan': True,
            'flip_theta': False
        }
    
    def set_scan_config(self, config):
        """
        Set the scan configuration parameters
        
        Args:
            config (dict): Dictionary of configuration parameters
        """
        self.scan_config = config
    
    def _initialize_scan_matcher(self):
        """Initialize the scan matcher with the occupancy grid"""
        if self.occupancy_grid and not self.scan_matcher:
            # Create the scan matcher with our occupancy grid
            self.scan_matcher = ScanMatcher(
                occupancy_grid_map=self.occupancy_grid,
                **self.scan_matcher_params
            )
            
            # Set LiDAR parameters based on what we've seen
            lidar_max_range = 10.0  # Default range
            num_samples = 180  # Default number of samples
            
            if self.previous_scan:
                # Estimate max range from the data
                max_range_in_data = max(self.previous_scan['scan_ranges'])
                if max_range_in_data > 0:
                    lidar_max_range = max(10.0, max_range_in_data * 1.1)  # Add 10% margin
                
                # Get number of samples from the data
                num_samples = len(self.previous_scan['scan_ranges'])
            
            # Set LiDAR parameters in scan matcher
            self.scan_matcher.set_lidar_params(
                max_range=lidar_max_range,
                field_of_view=self.angle_max - self.angle_min,
                num_samples=num_samples
            )
            
            # Also set in occupancy grid
            self.occupancy_grid.set_lidar_params(
                max_range=lidar_max_range,
                field_of_view=self.angle_max - self.angle_min,
                num_samples=num_samples
            )
    
    def process_scan(self, parsed_data):
        """
        Process a single LiDAR scan and update the SLAM system
        
        Args:
            parsed_data (dict): Dictionary containing LiDAR scan data
            
        Returns:
            dict: Updated data with corrected pose
        """
        self.scan_counter += 1
        
        # Extract scan data
        scan_ranges = parsed_data['scan_ranges']
        raw_pose = parsed_data['pose']
        
        # Convert current pose to our internal format
        current_pose = Pose2D(raw_pose['x'], raw_pose['y'], raw_pose['theta'])
        
        # Initialize occupancy grid if not already done
        if self.occupancy_grid is None:
            # Estimate grid size based on potential scan range
            # This is a conservative estimate and may need adjustment for your environment
            max_range = 30.0  # Typical max range for LiDAR in meters
            grid_width = int(max_range * 4 / self.grid_resolution)
            grid_height = int(max_range * 4 / self.grid_resolution)
            
            # Ensure grid size is a multiple of 2 for clean centering
            grid_width = (grid_width // 2) * 2
            grid_height = (grid_height // 2) * 2
            
            # Create the occupancy grid
            self.occupancy_grid = occupancy_grid.OccupancyGrid(
                resolution=self.grid_resolution,
                width=grid_width * self.grid_resolution,
                height=grid_height * self.grid_resolution,
                init_position=raw_pose  # Store initial position in metadata
            )
            
            # Initialize scan matcher now that we have a grid
            self._initialize_scan_matcher()
            
            # Set initial pose as the first grid update reference
            self.initial_pose = current_pose.copy()
        
        # Convert scan from polar to Cartesian coordinates
        x_points, y_points = scan_converter.convert_scans_to_cartesian(
            scan_ranges, 
            self.angle_min, 
            self.angle_max, 
            raw_pose,
            **self.scan_config
        )
        
        # Combine into a single array of points
        current_scan_points = np.column_stack((x_points, y_points))
        
        # Calculate odometry update (relative movement since last scan)
        delta_pose = None
        if self.previous_scan is not None:
            prev_pose = Pose2D(
                self.previous_scan['pose']['x'],
                self.previous_scan['pose']['y'],
                self.previous_scan['pose']['theta']
            )
            
            # Calculate relative movement in robot frame
            rel_pose = current_pose.relative_to(prev_pose)
            delta_pose = (rel_pose.x, rel_pose.y, rel_pose.theta)
            
            # Update pose estimator with odometry
            self.pose_estimator.update_from_odometry(rel_pose.x, rel_pose.y, rel_pose.theta)
        
        # Perform scan matching if in appropriate mode and we have a previous scan
        corrected_pose = current_pose.copy()
        match_result = None
        
        if (self.mode != SLAMMode.ODOMETRY_ONLY and 
            self.previous_scan is not None and 
            self.scan_counter % self.scan_match_interval == 0 and
            len(current_scan_points) >= self.min_points_for_matching and
            len(self.previous_scan_points) >= self.min_points_for_matching):
            
            # Use scan matcher to align current scan with the map
            try:
                # Get estimated pose from odometry or previous correction
                initial_pose = self.pose_estimator.get_current_pose().to_dict()
                
                # Previous and current motion information for scan matcher
                est_moving_dist = 0
                est_moving_theta = None
                
                if delta_pose:
                    est_moving_dist = np.sqrt(delta_pose[0]**2 + delta_pose[1]**2)
                    if est_moving_dist > 0.1:  # Significant movement
                        est_moving_theta = np.arctan2(delta_pose[1], delta_pose[0])
                
                # Perform scan matching to align the scan with the map
                match_result = self.scan_matcher.match_scan(
                    parsed_data,
                    initial_pose,
                    est_moving_dist,
                    est_moving_theta
                )
                
                # Store scan matching information for debugging
                matched_pose, confidence = match_result
                match_info = {
                    'matched_pose': matched_pose,
                    'confidence': confidence,
                    'time': time.time(),
                    'frame': self.scan_counter
                }
                self.scan_match_results.append(match_info)
                
                # Update pose using scan matching result
                corrected_pose = self.pose_estimator.update_from_scan_match(match_result)
                
                # Store the difference between odometry and corrected pose
                odom_pose = self.pose_estimator.get_current_pose()
                self.pose_corrections.append({
                    'odom_x': current_pose.x,
                    'odom_y': current_pose.y,
                    'odom_theta': current_pose.theta,
                    'corrected_x': corrected_pose.x,
                    'corrected_y': corrected_pose.y,
                    'corrected_theta': corrected_pose.theta,
                    'confidence': confidence
                })
                
            except Exception as e:
                print(f"Scan matching failed: {e}")
                # Fall back to odometry if scan matching fails
                corrected_pose = self.pose_estimator.get_current_pose()
        else:
            # Just use the latest pose from the estimator
            corrected_pose = self.pose_estimator.get_current_pose()
        
        # Convert corrected pose back to the format used by the original system
        corrected_pose_dict = {
            'x': corrected_pose.x,
            'y': corrected_pose.y,
            'theta': corrected_pose.theta
        }
        
        # Update the occupancy grid with the corrected pose and scan
        if self.occupancy_grid:
            # Convert scan to world coordinates using corrected pose
            corrected_x_points, corrected_y_points = scan_converter.convert_scans_to_cartesian(
                scan_ranges, 
                self.angle_min, 
                self.angle_max, 
                corrected_pose_dict,
                **self.scan_config
            )
            
            # Update the grid
            self.occupancy_grid.update_grid(
                corrected_pose.x, 
                corrected_pose.y, 
                corrected_x_points, 
                corrected_y_points
            )
        
        # Store the current scan for next iteration
        self.previous_scan = parsed_data.copy()
        self.previous_scan_points = current_scan_points.copy()
        
        # Create updated data with corrected pose
        updated_data = parsed_data.copy()
        updated_data['original_pose'] = parsed_data['pose'].copy()  # Store original
        updated_data['pose'] = corrected_pose_dict  # Update with corrected
        
        # Add matching information if available
        if match_result:
            matched_pose, confidence = match_result
            updated_data['scan_match_info'] = {
                'confidence': confidence,
                'pose_change_x': matched_pose['x'] - initial_pose['x'],
                'pose_change_y': matched_pose['y'] - initial_pose['y'],
                'pose_change_theta': matched_pose['theta'] - initial_pose['theta']
            }
        
        return updated_data
    
    def process_dataset(self, parsed_data_list):
        """
        Process an entire dataset of scans
        
        Args:
            parsed_data_list (list): List of dictionaries with LiDAR scan data
            
        Returns:
            list: Updated data with corrected poses
        """
        # Reset the system
        self.reset()
        
        # Process each scan
        updated_data_list = []
        
        print(f"Processing {len(parsed_data_list)} scans...")
        start_time = time.time()
        
        for i, parsed_data in enumerate(parsed_data_list):
            # Print progress every 10%
            if (i + 1) % max(1, len(parsed_data_list) // 10) == 0:
                progress = (i + 1) / len(parsed_data_list) * 100
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i + 1) * len(parsed_data_list)
                remaining = estimated_total - elapsed
                
                print(f"Progress: {progress:.1f}% ({i + 1}/{len(parsed_data_list)}) - "
                     f"Time: {elapsed:.1f}s / Est. remaining: {remaining:.1f}s")
            
            # Process the scan
            updated_data = self.process_scan(parsed_data)
            updated_data_list.append(updated_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        scans_per_second = len(parsed_data_list) / processing_time
        
        print(f"Processing complete. Processed {len(parsed_data_list)} scans in {processing_time:.2f} seconds")
        print(f"Average processing speed: {scans_per_second:.2f} scans/second")
        
        # Print scan matching statistics if any matches were performed
        if self.scan_match_results:
            confidences = [result['confidence'] for result in self.scan_match_results]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"Average scan matching confidence: {avg_confidence:.4f}")
            
            if self.pose_corrections:
                # Calculate average correction distance
                correction_distances = [
                    np.sqrt((corr['corrected_x'] - corr['odom_x'])**2 + 
                            (corr['corrected_y'] - corr['odom_y'])**2)
                    for corr in self.pose_corrections
                ]
                avg_correction = sum(correction_distances) / len(correction_distances)
                print(f"Average pose correction: {avg_correction:.4f} meters")
        
        return updated_data_list
    
    def get_occupancy_grid(self):
        """Get the current occupancy grid"""
        return self.occupancy_grid
    
    def get_pose_history(self):
        """Get the history of poses from the estimator"""
        return self.pose_estimator.get_pose_history()
    
    def get_confidence_history(self):
        """Get the history of scan matching confidence values"""
        if hasattr(self.pose_estimator, 'get_confidence_history'):
            return self.pose_estimator.get_confidence_history()
        return [result['confidence'] for result in self.scan_match_results]
    
    def reset(self):
        """Reset the SLAM system to initial state"""
        self.pose_estimator.reset()
        
        # Reset occupancy grid
        if self.occupancy_grid:
            grid_width = self.occupancy_grid.width
            grid_height = self.occupancy_grid.height
            self.occupancy_grid = occupancy_grid.OccupancyGrid(
                resolution=self.grid_resolution,
                width=grid_width,
                height=grid_height
            )
            
            # Re-initialize scan matcher
            self.scan_matcher = None
            self._initialize_scan_matcher()
        
        # Reset scan data
        self.previous_scan = None
        self.previous_scan_points = None
        
        # Reset counters and debug info
        self.scan_counter = 0
        self.scan_match_results = []
        self.pose_corrections = []