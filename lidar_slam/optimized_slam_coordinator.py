"""
Optimized SLAM Coordinator Module

This module provides an efficient implementation of SLAM coordination,
optimized for performance while integrating scan matching and pose estimation.
"""

import numpy as np
import os
import sys
import time
from enum import Enum

# Add the parent directory to the path to import other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules from existing system
from lidar_processing import data_parser, scan_converter
from lidar_processing.optimized_scan_matcher import OptimizedScanMatcher, MatchingAlgorithm, MatchResult
from pose_estimation.pose_estimator import PoseEstimator, PoseEstimationMethod, Pose2D
from mapping import occupancy_grid

class SLAMMode(Enum):
    """Enumeration of available SLAM operation modes"""
    ODOMETRY_ONLY = "odometry_only"  # Use only odometry for mapping
    SCAN_MATCHING = "scan_matching"  # Use scan matching for pose correction
    FULL_SLAM = "full_slam"  # Use full SLAM with loop closure (future)

class OptimizedSLAMCoordinator:
    """Class to coordinate the full SLAM pipeline with performance optimization"""
    
    def __init__(self, mode=SLAMMode.SCAN_MATCHING, grid_resolution=0.05):
        """
        Initialize the SLAM coordinator
        
        Args:
            mode (SLAMMode): Operation mode for SLAM
            grid_resolution (float): Resolution of the occupancy grid in meters
        """
        self.mode = mode
        self.grid_resolution = grid_resolution
        
        # Initialize components
        self.scan_matcher = OptimizedScanMatcher(algorithm=MatchingAlgorithm.FAST_ICP)
        
        # Configure scan matcher for performance
        self.scan_matcher.set_optimization_params(
            downsample_factor=3,  # Use every 3rd point
            max_iterations=15,     # Limit iterations
            tolerance=1e-4,        # Convergence threshold
            max_correspondence_distance=0.5  # Max correspondence distance
        )
        
        # Choose pose estimation method based on SLAM mode
        pose_method = PoseEstimationMethod.ODOMETRY_ONLY
        if mode == SLAMMode.SCAN_MATCHING:
            pose_method = PoseEstimationMethod.EKF_FUSION
        
        self.pose_estimator = PoseEstimator(method=pose_method)
        
        # Occupancy grid will be initialized when first scan is processed
        self.occupancy_grid = None
        
        # For storing previous scan data
        self.previous_scan = None
        self.previous_scan_points = None
        
        # For debugging and visualization
        self.scan_match_results = []
        self.pose_corrections = []
        
        # Scan matching parameters
        self.min_points_for_matching = 20  # Minimum number of points needed
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
        
        # Performance metrics
        self.processing_times = []
        self.matching_times = []
    
    def set_scan_config(self, config):
        """
        Set the scan configuration parameters
        
        Args:
            config (dict): Dictionary of configuration parameters
        """
        self.scan_config = config
    
    def set_scan_matcher_params(self, downsample_factor=None, max_iterations=None, 
                               tolerance=None, max_distance=None):
        """
        Configure scan matcher parameters for performance optimization
        
        Args:
            downsample_factor (int): Use every Nth point
            max_iterations (int): Maximum iterations for matching
            tolerance (float): Convergence threshold
            max_distance (float): Maximum correspondence distance
        """
        self.scan_matcher.set_optimization_params(
            downsample_factor=downsample_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
            max_correspondence_distance=max_distance
        )
    
    def set_scan_match_interval(self, interval):
        """
        Set the interval for scan matching (process every Nth scan)
        
        Args:
            interval (int): Scan matching interval
        """
        self.scan_match_interval = max(1, int(interval))
    
    def process_scan(self, parsed_data):
        """
        Process a single LiDAR scan and update the SLAM system
        
        Args:
            parsed_data (dict): Dictionary containing LiDAR scan data
            
        Returns:
            dict: Updated data with corrected pose
        """
        start_time = time.time()
        self.scan_counter += 1
        
        # Extract scan data
        scan_ranges = parsed_data['scan_ranges']
        raw_pose = parsed_data['pose']
        
        # Convert current pose to our internal format
        current_pose = Pose2D(raw_pose['x'], raw_pose['y'], raw_pose['theta'])
        
        # Initialize occupancy grid if not already done
        if self.occupancy_grid is None:
            # Estimate grid size based on potential scan range
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
                height=grid_height * self.grid_resolution
            )
            
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
        
        # Apply coordinate transformations if necessary
        if self.scan_config['flip_x']:
            current_scan_points[:, 0] = -current_scan_points[:, 0]
        if self.scan_config['flip_y']:
            current_scan_points[:, 1] = -current_scan_points[:, 1]
        
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
            self.previous_scan_points is not None and 
            self.scan_counter % self.scan_match_interval == 0 and
            len(current_scan_points) >= self.min_points_for_matching and
            len(self.previous_scan_points) >= self.min_points_for_matching):
            
            match_start_time = time.time()
            try:
                # Use odometry as initial guess for scan matcher
                initial_guess = None
                if delta_pose is not None:
                    initial_guess = delta_pose
                
                # Perform scan matching
                match_result = self.scan_matcher.match_scans(
                    self.previous_scan_points, 
                    current_scan_points,
                    initial_guess
                )
                
                # Store result for debugging
                self.scan_match_results.append(match_result)
                
                # Update pose using scan matching result
                corrected_pose = self.pose_estimator.update_from_scan_match(match_result)
                
                # Store the difference between odometry and corrected pose
                odom_only_pose = self.pose_estimator.get_current_pose()
                self.pose_corrections.append({
                    'odom_x': odom_only_pose.x,
                    'odom_y': odom_only_pose.y,
                    'odom_theta': odom_only_pose.theta,
                    'corrected_x': corrected_pose.x,
                    'corrected_y': corrected_pose.y,
                    'corrected_theta': corrected_pose.theta,
                    'scan_match_success': match_result.success,
                    'fitness_score': match_result.fitness_score
                })
                
                # Record matching time
                self.matching_times.append(time.time() - match_start_time)
                
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
            self.occupancy_grid.update_grid(
                corrected_pose.x, 
                corrected_pose.y, 
                x_points, 
                y_points
            )
        
        # Store the current scan for next iteration
        self.previous_scan = parsed_data.copy()
        self.previous_scan_points = current_scan_points.copy()
        
        # Create updated data with corrected pose
        updated_data = parsed_data.copy()
        updated_data['original_pose'] = parsed_data['pose'].copy()  # Store original
        updated_data['pose'] = corrected_pose_dict  # Update with corrected
        
        # Add matching information if available
        if match_result is not None:
            updated_data['scan_match_info'] = {
                'success': match_result.success,
                'fitness_score': match_result.fitness_score,
                'inlier_rmse': match_result.inlier_rmse,
                'iterations': match_result.iterations,
                'computation_time': match_result.computation_time
            }
        
        # Record processing time
        self.processing_times.append(time.time() - start_time)
        
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
        last_progress_report = time.time()
        progress_interval = 2.0  # Report progress every 2 seconds
        
        for i, parsed_data in enumerate(parsed_data_list):
            # Print progress at regular intervals
            current_time = time.time()
            if current_time - last_progress_report > progress_interval:
                progress = (i + 1) / len(parsed_data_list) * 100
                elapsed = current_time - start_time
                remaining = (len(parsed_data_list) - i - 1) * elapsed / (i + 1) if i > 0 else 0
                scans_per_second = (i + 1) / elapsed if elapsed > 0 else 0
                
                print(f"Progress: {progress:.1f}% ({i + 1}/{len(parsed_data_list)}) | "
                      f"Speed: {scans_per_second:.1f} scans/sec | "
                      f"Est. remaining: {remaining:.1f} seconds")
                
                last_progress_report = current_time
            
            # Process the scan
            updated_data = self.process_scan(parsed_data)
            updated_data_list.append(updated_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        scans_per_second = len(parsed_data_list) / processing_time
        
        print(f"\nProcessing complete!")
        print(f"Processed {len(parsed_data_list)} scans in {processing_time:.2f} seconds")
        print(f"Average processing speed: {scans_per_second:.2f} scans/second")
        
        # Print performance statistics
        if self.scan_match_results:
            successful_matches = sum(1 for r in self.scan_match_results if r.success)
            avg_iterations = np.mean([r.iterations for r in self.scan_match_results if r.success]) if successful_matches else 0
            avg_time = np.mean([r.computation_time for r in self.scan_match_results if r.success]) if successful_matches else 0
            
            print(f"\nScan Matching Statistics:")
            print(f"  Successful matches: {successful_matches}/{len(self.scan_match_results)} "
                  f"({successful_matches/len(self.scan_match_results)*100:.1f}%)")
            print(f"  Average iterations: {avg_iterations:.1f}")
            print(f"  Average matching time: {avg_time*1000:.2f} ms")
        
        return updated_data_list
    
    def get_occupancy_grid(self):
        """Get the current occupancy grid"""
        return self.occupancy_grid
    
    def get_pose_history(self):
        """Get the history of poses from the estimator"""
        return self.pose_estimator.get_pose_history()
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {
            'total_scans': self.scan_counter,
            'scan_matches': len(self.scan_match_results),
            'successful_matches': sum(1 for r in self.scan_match_results if r.success),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'avg_matching_time': np.mean(self.matching_times) if self.matching_times else 0,
            'max_processing_time': max(self.processing_times) if self.processing_times else 0,
            'max_matching_time': max(self.matching_times) if self.matching_times else 0,
        }
        return stats
    
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
        
        # Reset scan data
        self.previous_scan = None
        self.previous_scan_points = None
        
        # Reset counters and debug info
        self.scan_counter = 0
        self.scan_match_results = []
        self.pose_corrections = []
        self.processing_times = []
        self.matching_times = []


# Example usage
if __name__ == "__main__":
    # This will be extended to work with your existing system
    print("Optimized SLAM Coordinator module loaded.")
    print("Run this through main_slam.py with appropriate modifications.")