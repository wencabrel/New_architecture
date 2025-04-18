#!/usr/bin/env python3
"""
Improved Scan Matcher Example

This module implements a sophisticated scan matcher based on the ScanMatcher_OGBased implementation.
It uses a multi-resolution approach with coarse-to-fine matching and Bayesian updates.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
from scipy.ndimage import gaussian_filter

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lidar_slam.lidar_processing import data_parser, scan_converter
from lidar_slam.mapping import occupancy_grid
from lidar_slam.utils import file_utils

class ScanMatcher:
    """
    Implements scan matching for LIDAR data against an occupancy grid map.
    Uses a multi-resolution approach with coarse-to-fine matching strategy.
    """
    
    def __init__(self, occupancy_grid_map, search_radius=1.4, search_half_rad=0.25, 
                 scan_sigma_in_num_grid=2, move_r_sigma=0.1, max_move_deviation=0.25, 
                 turn_sigma=0.3, mismatch_prob_at_coarse=0.15, coarse_factor=5):
        """
        Initialize scan matcher with given parameters.
        
        Args:
            occupancy_grid_map: OccupancyGrid object to match against
            search_radius (float): Maximum search radius in meters
            search_half_rad (float): Maximum rotation search range in radians
            scan_sigma_in_num_grid (float): Standard deviation for scan correlation in grid cells
            move_r_sigma (float): Standard deviation for movement in meters
            max_move_deviation (float): Maximum allowed movement deviation in meters
            turn_sigma (float): Standard deviation for rotation in radians
            mismatch_prob_at_coarse (float): Probability threshold for mismatches in coarse search
            coarse_factor (int): Factor for coarse-to-fine resolution reduction
        """
        self.search_radius = search_radius
        self.search_half_rad = search_half_rad
        self.og = occupancy_grid_map
        self.scan_sigma_in_num_grid = scan_sigma_in_num_grid
        self.coarse_factor = coarse_factor
        self.move_r_sigma = move_r_sigma
        self.turn_sigma = turn_sigma
        self.mismatch_prob_at_coarse = mismatch_prob_at_coarse
        self.max_move_deviation = max_move_deviation
        
        # For scan conversion
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        self.config = {
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
    
    def set_lidar_params(self, max_range, field_of_view, num_samples):
        """Set LiDAR parameters for scan matching"""
        self.lidar_max_range = max_range
        self.lidar_fov = field_of_view
        self.num_samples_per_rev = num_samples
    
    def frame_search_space(self, estimated_x, estimated_y, unit_length, sigma, mismatch_prob):
        """
        Create and initialize the search space for scan matching.
        
        Args:
            estimated_x (float): Estimated x position
            estimated_y (float): Estimated y position
            unit_length (float): Grid cell size for search space
            sigma (float): Standard deviation for Gaussian blurring
            mismatch_prob (float): Base probability for mismatches
            
        Returns:
            tuple: (x_range, y_range, probability_search_space)
                  Search space boundaries and probability distribution
        """
        # Calculate maximum scan radius including search area
        max_scan_radius = 1.1 * self.lidar_max_range + self.search_radius
        
        # Define search space boundaries
        x_range = [estimated_x - max_scan_radius, estimated_x + max_scan_radius]
        y_range = [estimated_y - max_scan_radius, estimated_y + max_scan_radius]
        
        # Calculate grid dimensions
        idx_end_x = int((x_range[1] - x_range[0]) / unit_length)
        idx_end_y = int((y_range[1] - y_range[0]) / unit_length)
        
        # Initialize search space with mismatch probability
        search_space = math.log(mismatch_prob) * np.ones((idx_end_y + 1, idx_end_x + 1))
        
        # Ensure our search space is covered by the occupancy grid
        # We'll check if the occupancy grid needs to be expanded to cover our search
        self._ensure_grid_coverage(x_range, y_range)
        
        # Extract and process relevant portion of occupancy grid
        og_map, og_x, og_y = self._extract_grid_section(x_range, y_range)
        
        # Convert occupied cells to search space indices
        if len(og_x) > 0 and len(og_y) > 0:
            og_idx = self._convert_xy_to_search_space_idx(
                og_x, og_y, x_range[0], y_range[0], unit_length
            )
            
            # Mark occupied cells in search space
            valid_indices = (
                (og_idx[1] >= 0) & (og_idx[1] < search_space.shape[0]) &
                (og_idx[0] >= 0) & (og_idx[0] < search_space.shape[1])
            )
            
            if np.any(valid_indices):
                row_indices = og_idx[1][valid_indices]
                col_indices = og_idx[0][valid_indices]
                search_space[row_indices, col_indices] = 0
        
        # Generate probability distribution using Gaussian filter
        prob_search_space = self._generate_prob_search_space(search_space, sigma)
        
        return x_range, y_range, prob_search_space
    
    def _ensure_grid_coverage(self, x_range, y_range):
        """Ensure occupancy grid covers the search range (placeholder)"""
        # In a complete implementation, this would expand the occupancy grid if needed
        # For now, we just check if the OG covers our search area
        pass
    
    def _extract_grid_section(self, x_range, y_range):
        """
        Extract relevant section from occupancy grid
        
        Args:
            x_range (list): [min_x, max_x] range
            y_range (list): [min_y, max_y] range
            
        Returns:
            tuple: (binary_map, og_x, og_y) - occupied cells and their coordinates
        """
        # Convert search range to grid coordinates
        x_min_idx = max(0, int((x_range[0] + self.og.width/2) / self.og.resolution))
        x_max_idx = min(self.og.grid_width - 1, int((x_range[1] + self.og.width/2) / self.og.resolution))
        y_min_idx = max(0, int((y_range[0] + self.og.height/2) / self.og.resolution))
        y_max_idx = min(self.og.grid_height - 1, int((y_range[1] + self.og.height/2) / self.og.resolution))
        
        # Extract the grid section
        grid_section = self.og.get_grid()[y_min_idx:y_max_idx, x_min_idx:x_max_idx]
        
        # Find occupied cells (value > 0.55)
        occupied = grid_section > 0.55
        
        if not np.any(occupied):
            return np.array([]), np.array([]), np.array([])
            
        # Get indices of occupied cells
        occupied_y, occupied_x = np.where(occupied)
        
        # Convert to global indices
        occupied_x += x_min_idx
        occupied_y += y_min_idx
        
        # Convert to world coordinates
        og_x = (occupied_x * self.og.resolution) - self.og.width/2
        og_y = (occupied_y * self.og.resolution) - self.og.height/2
        
        return occupied, og_x, og_y
    
    def _generate_prob_search_space(self, search_space, sigma):
        """
        Generate probability distribution for search space using Gaussian filtering.
        
        Args:
            search_space (numpy.ndarray): Initial search space
            sigma (float): Standard deviation for Gaussian filter
            
        Returns:
            numpy.ndarray: Probability distribution over search space
        """
        # Apply Gaussian filter
        prob_space = gaussian_filter(search_space, sigma=sigma)
        
        # Threshold probabilities
        prob_min = prob_space.min()
        prob_space[prob_space > 0.5 * prob_min] = 0
        
        return prob_space
    
    def match_scan(self, scan_data, initial_pose, est_moving_dist=0, est_moving_theta=None, 
                   match_max=True):
        """
        Match current scan against occupancy grid using two-stage approach.
        
        Args:
            scan_data (dict): Current sensor reading
            initial_pose (dict): Initial pose estimate with 'x', 'y', 'theta' keys
            est_moving_dist (float): Estimated movement distance
            est_moving_theta (float): Estimated movement angle
            match_max (bool): Whether to use maximum likelihood (True) or sampling (False)
            
        Returns:
            tuple: (matched_pose, confidence)
                  Best match pose and confidence score
        """
        # Extract pose data
        estimated_x = initial_pose['x']
        estimated_y = initial_pose['y']
        estimated_theta = initial_pose['theta']
        
        # Convert scan data format if needed
        if 'range' in scan_data:
            r_measure = np.array(scan_data['range'])
        else:
            r_measure = np.array(scan_data['scan_ranges'])
        
        # Coarse Search Stage
        coarse_search_step = self.coarse_factor * self.og.resolution
        coarse_sigma = self.scan_sigma_in_num_grid / self.coarse_factor
        
        x_range, y_range, prob_space = self.frame_search_space(
            estimated_x, estimated_y, coarse_search_step,
            coarse_sigma, self.mismatch_prob_at_coarse
        )
        
        matched_px, matched_py, matched_pose, conv_total, coarse_confidence = \
            self._search_to_match(
                prob_space, estimated_x, estimated_y, estimated_theta, r_measure,
                x_range, y_range, self.search_radius, self.search_half_rad,
                coarse_search_step, est_moving_dist, est_moving_theta,
                fine_search=False, match_max=match_max
            )
        
        # Fine Search Stage
        fine_search_step = self.og.resolution
        fine_sigma = self.scan_sigma_in_num_grid
        fine_search_half_rad = self.search_half_rad / 2  # Smaller rotation search for fine stage
        fine_mismatch_prob = self.mismatch_prob_at_coarse ** (2 / self.coarse_factor)
        
        x_range, y_range, prob_space = self.frame_search_space(
            matched_pose['x'], matched_pose['y'],
            fine_search_step, fine_sigma, fine_mismatch_prob
        )
        
        matched_px, matched_py, matched_pose, conv_total, fine_confidence = \
            self._search_to_match(
                prob_space, matched_pose['x'], matched_pose['y'],
                matched_pose['theta'], r_measure,
                x_range, y_range, coarse_search_step, fine_search_half_rad,
                fine_search_step, est_moving_dist, est_moving_theta,
                fine_search=True, match_max=True
            )
        
        return matched_pose, fine_confidence
    
    def _convert_measure_to_xy(self, estimated_x, estimated_y, estimated_theta, r_measure):
        """
        Convert polar LiDAR measurements to Cartesian coordinates.
        
        Args:
            estimated_x (float): Estimated x position
            estimated_y (float): Estimated y position
            estimated_theta (float): Estimated orientation
            r_measure (numpy.ndarray): Range measurements
            
        Returns:
            tuple: (px, py) Arrays of x and y coordinates
        """
        # Calculate measurement angles
        angles = np.linspace(
            estimated_theta - self.lidar_fov / 2,
            estimated_theta + self.lidar_fov / 2,
            num=self.num_samples_per_rev
        )
        
        # Filter valid measurements
        valid_idx = r_measure < self.lidar_max_range
        r_measure_valid = r_measure[valid_idx]
        angles_valid = angles[valid_idx]
        
        # Convert to Cartesian coordinates
        px = estimated_x + np.cos(angles_valid) * r_measure_valid
        py = estimated_y + np.sin(angles_valid) * r_measure_valid
        
        return px, py
    
    def _search_to_match(self, prob_space, estimated_x, estimated_y, estimated_theta,
                        r_measure, x_range, y_range, search_radius, search_half_rad,
                        unit_length, est_moving_dist, est_moving_theta,
                        fine_search=False, match_max=True):
        """
        Search for best match by trying different positions and orientations.
        
        Args:
            prob_space (numpy.ndarray): Probability distribution over search space
            estimated_x, estimated_y (float): Estimated position
            estimated_theta (float): Estimated orientation
            r_measure (numpy.ndarray): Range measurements
            x_range, y_range (list): Search space boundaries
            search_radius (float): Maximum search radius
            search_half_rad (float): Maximum rotation search range
            unit_length (float): Grid cell size
            est_moving_dist (float): Estimated movement distance
            est_moving_theta (float): Estimated movement angle
            fine_search (bool): Whether this is fine search stage
            match_max (bool): Whether to use maximum likelihood or sampling
            
        Returns:
            tuple: (matched_px, matched_py, matched_pose, conv_total, confidence)
                  Matched points, pose, correlation results, and confidence
        """
        # Convert measurements to Cartesian coordinates
        px, py = self._convert_measure_to_xy(estimated_x, estimated_y, estimated_theta, r_measure)
        
        # Calculate search grid
        num_cells_radius = int(search_radius / unit_length)
        x_moving_range = np.arange(-num_cells_radius, num_cells_radius + 1)
        y_moving_range = np.arange(-num_cells_radius, num_cells_radius + 1)
        x_grid, y_grid = np.meshgrid(x_moving_range, y_moving_range)
        
        # Initialize correlation weights
        if fine_search:
            range_weight = np.zeros(x_grid.shape)
            theta_weight = np.zeros(x_grid.shape)
        else:
            # Calculate range correlation weight
            range_weight = -(1 / (2 * self.move_r_sigma ** 2)) * \
                         (np.sqrt((x_grid * unit_length) ** 2 + 
                                 (y_grid * unit_length) ** 2) - est_moving_dist) ** 2
            
            # Apply maximum deviation constraint
            range_deviation = np.abs(np.sqrt((x_grid * unit_length) ** 2 + 
                                          (y_grid * unit_length) ** 2) - est_moving_dist)
            range_weight[range_deviation > self.max_move_deviation] = -100
            
            # Calculate rotation correlation weight
            if est_moving_theta is not None:
                dist_grid = np.sqrt(np.square(x_grid) + np.square(y_grid))
                dist_grid[dist_grid == 0] = 0.0001
                theta_grid = np.arccos(np.clip(
                    (x_grid * math.cos(est_moving_theta) + y_grid * math.sin(est_moving_theta)) / dist_grid,
                    -1.0, 1.0  # Clip to valid arccos range
                ))
                theta_weight = -1 / (2 * self.turn_sigma ** 2) * np.square(theta_grid)
            else:
                theta_weight = np.zeros(x_grid.shape)
        
        # Reshape search grids for broadcasting
        x_grid = x_grid.reshape((x_grid.shape[0], x_grid.shape[1], 1))
        y_grid = y_grid.reshape((y_grid.shape[0], y_grid.shape[1], 1))
        
        # Define rotation search range
        theta_range = np.arange(
            -search_half_rad, 
            search_half_rad + self.angular_step, 
            self.angular_step
        )
        
        # Initialize correlation results
        conv_total = np.zeros((len(theta_range), x_grid.shape[0], x_grid.shape[1]))
        
        # Search over all rotations
        for i, theta in enumerate(theta_range):
            # Rotate scan points
            rotated_px, rotated_py = self._rotate_points(
                (estimated_x, estimated_y), 
                (px, py), 
                theta
            )
            
            # Convert to search space indices
            rotated_px_idx, rotated_py_idx = self._convert_xy_to_search_space_idx(
                rotated_px, rotated_py, x_range[0], y_range[0], unit_length
            )
            
            # Remove points outside search space
            valid_indices = (
                (rotated_px_idx >= 0) & (rotated_px_idx < prob_space.shape[1]) &
                (rotated_py_idx >= 0) & (rotated_py_idx < prob_space.shape[0])
            )
            
            if not np.any(valid_indices):
                continue
                
            rotated_px_idx = rotated_px_idx[valid_indices]
            rotated_py_idx = rotated_py_idx[valid_indices]
            
            # Remove duplicate points
            if len(rotated_px_idx) > 0:
                unique_points = np.unique(np.column_stack((rotated_px_idx, rotated_py_idx)), axis=0)
                rotated_px_idx, rotated_py_idx = unique_points[:, 0], unique_points[:, 1]
                
                # Reshape for broadcasting
                rotated_px_idx = rotated_px_idx.reshape(1, 1, -1)
                rotated_py_idx = rotated_py_idx.reshape(1, 1, -1)
                
                # Apply translation search
                translated_px_idx = rotated_px_idx + x_grid
                translated_py_idx = rotated_py_idx + y_grid
                
                # Ensure indices are within bounds
                valid_x = (translated_px_idx >= 0) & (translated_px_idx < prob_space.shape[1])
                valid_y = (translated_py_idx >= 0) & (translated_py_idx < prob_space.shape[0])
                valid = valid_x & valid_y
                
                # Calculate correlation (set invalid indices to 0)
                conv_result = np.zeros_like(valid, dtype=float)
                conv_result[valid] = prob_space[translated_py_idx[valid], translated_px_idx[valid]]
                
                # Sum along scan points axis
                conv_sum = np.sum(conv_result, axis=2)
                
                # Combine correlation with motion weights
                conv_sum = conv_sum + range_weight + theta_weight
                conv_total[i, :, :] = conv_sum
        
        # Select best match based on strategy
        if match_max:
            # Maximum likelihood approach
            max_idx = np.unravel_index(np.argmax(conv_total), conv_total.shape)
        else:
            # Probabilistic sampling approach
            conv_total_flat = np.reshape(conv_total, -1)
            conv_max = np.max(conv_total_flat)
            conv_exp = np.exp(conv_total_flat - conv_max)  # Subtract max for numerical stability
            conv_total_prob = conv_exp / np.sum(conv_exp)
            max_idx = np.unravel_index(
                np.random.choice(np.arange(conv_total_flat.size), 1, p=conv_total_prob)[0],
                conv_total.shape
            )
        
        # Calculate confidence score
        # Use log-sum-exp trick for numerical stability
        max_val = np.max(conv_total)
        confidence = np.exp(max_val) * np.sum(np.exp(conv_total - max_val))
        
        # Convert best match to transformations
        dx = x_moving_range[max_idx[2]] * unit_length
        dy = y_moving_range[max_idx[1]] * unit_length
        dtheta = theta_range[max_idx[0]]
        
        # Create matched pose
        matched_pose = {
            "x": estimated_x + dx,
            "y": estimated_y + dy,
            "theta": estimated_theta + dtheta
        }
        
        # Transform scan points to matched position
        matched_px, matched_py = self._rotate_points(
            (estimated_x, estimated_y), 
            (px, py), 
            dtheta
        )
        matched_px = matched_px + dx
        matched_py = matched_py + dy
        
        return matched_px, matched_py, matched_pose, conv_total, confidence
    
    def _rotate_points(self, origin, points, angle):
        """
        Rotate point(s) around origin by given angle.
        
        Args:
            origin (tuple): (x, y) coordinates of rotation center
            points (tuple): (x_arr, y_arr) arrays of points to rotate
            angle (float): Rotation angle in radians (counterclockwise)
            
        Returns:
            tuple: (rotated_x, rotated_y) Rotated point coordinates
        """
        ox, oy = origin
        px, py = points
        
        # Apply rotation matrix
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        
        return qx, qy
    
    def _convert_xy_to_search_space_idx(self, px, py, begin_x, begin_y, unit_length):
        """
        Convert real-world coordinates to search space indices.
        
        Args:
            px, py (numpy.ndarray): Point coordinates
            begin_x, begin_y (float): Search space origin
            unit_length (float): Grid cell size
            
        Returns:
            tuple: (x_idx, y_idx) Grid indices
        """
        x_idx = ((px - begin_x) / unit_length).astype(int)
        y_idx = ((py - begin_y) / unit_length).astype(int)
        return x_idx, y_idx
    
    def plot_match_overlay(self, prob_space, matched_px, matched_py, matched_pose, 
                         x_range, y_range, unit_length):
        """
        Visualize scan matching results overlaid on probability space.
        
        Args:
            prob_space (numpy.ndarray): Probability distribution over search space
            matched_px, matched_py (numpy.ndarray): Matched scan points
            matched_pose (dict): Matched sensor reading
            x_range, y_range (list): Search space boundaries
            unit_length (float): Grid cell size
        """
        plt.figure(figsize=(12, 10))
        
        # Plot probability space
        plt.imshow(prob_space, origin='lower')
        
        # Convert and plot matched scan points
        px_idx, py_idx = self._convert_xy_to_search_space_idx(
            matched_px, matched_py, x_range[0], y_range[0], unit_length
        )
        
        # Filter valid indices
        valid = (
            (px_idx >= 0) & (px_idx < prob_space.shape[1]) &
            (py_idx >= 0) & (py_idx < prob_space.shape[0])
        )
        
        plt.scatter(px_idx[valid], py_idx[valid], c='r', s=5)
        
        # Convert and plot matched position
        pose_x_idx, pose_y_idx = self._convert_xy_to_search_space_idx(
            matched_pose['x'], matched_pose['y'],
            x_range[0], y_range[0], unit_length
        )
        plt.scatter(pose_x_idx, pose_y_idx, color='blue', s=50)
        
        plt.colorbar(label='Correlation Score')
        plt.title('Scan Matching Result')
        plt.xlabel('X (search space pixels)')
        plt.ylabel('Y (search space pixels)')
        plt.show()
    
    def process_scan_sequence(self, scan_data_list, initial_pose=None, update_og=False):
        """
        Process a sequence of scans with tracking.
        
        Args:
            scan_data_list (list): List of scan data dictionaries
            initial_pose (dict): Optional initial pose (uses first scan's pose if None)
            update_og (bool): Whether to update occupancy grid with matched scans
            
        Returns:
            tuple: (corrected_poses, confidences) Lists of corrected poses and match confidences
        """
        if not scan_data_list:
            print("No scan data provided")
            return [], []
        
        corrected_poses = []
        confidences = []
        
        # Use first scan's pose if no initial pose provided
        if initial_pose is None:
            if 'pose' in scan_data_list[0]:
                current_pose = scan_data_list[0]['pose'].copy()
            else:
                # Default pose at origin
                current_pose = {'x': 0, 'y': 0, 'theta': 0}
            
            corrected_poses.append(current_pose)
            confidences.append(1.0)  # Perfect confidence for initial pose
        else:
            current_pose = initial_pose.copy()
            corrected_poses.append(current_pose)
            confidences.append(1.0)
        
        # Variables for motion estimation
        prev_pose = current_pose
        prev_moving_theta = None
        
        # Process subsequent scans
        for i in range(1, len(scan_data_list)):
            current_scan = scan_data_list[i]
            
            # Extract raw pose from scan
            if 'pose' in current_scan:
                raw_pose = current_scan['pose']
            else:
                # If no pose in scan data, estimate from previous motions
                raw_pose = {
                    'x': prev_pose['x'],
                    'y': prev_pose['y'],
                    'theta': prev_pose['theta']
                }
            
            # Estimate motion from previous and current raw poses
            dx = raw_pose['x'] - scan_data_list[i-1]['pose']['x']
            dy = raw_pose['y'] - scan_data_list[i-1]['pose']['y']
            dtheta = raw_pose['theta'] - scan_data_list[i-1]['pose']['theta']
            
            # Estimate parameters for scan matching
            est_moving_dist = math.sqrt(dx**2 + dy**2)
            est_moving_theta = None
            raw_moving_theta = None
            
            if est_moving_dist > 0.3:  # Only calculate theta for significant movement
                raw_moving_theta = math.atan2(dy, dx)
                
                if prev_moving_theta is not None:
                    raw_turn_theta = raw_moving_theta - prev_moving_theta
                    if hasattr(self, 'prev_matched_moving_theta') and self.prev_matched_moving_theta is not None:
                        est_moving_theta = self.prev_matched_moving_theta + raw_turn_theta
            
            # Create initial pose estimate based on previous corrected pose and raw motion
            initial_estimate = {
                'x': prev_pose['x'] + dx,
                'y': prev_pose['y'] + dy,
                'theta': prev_pose['theta'] + dtheta
            }
            
            # Perform scan matching
            matched_pose, confidence = self.match_scan(
                current_scan, 
                initial_estimate, 
                est_moving_dist, 
                est_moving_theta
            )
            
            # Update occupancy grid if requested
            if update_og:
                # Convert scan to world coordinates using matched pose
                if 'range' in current_scan:
                    r_measure = current_scan['range']
                else:
                    r_measure = current_scan['scan_ranges']
                    
                px, py = self._convert_measure_to_xy(
                    matched_pose['x'], 
                    matched_pose['y'], 
                    matched_pose['theta'], 
                    np.array(r_measure)
                )
                
                # Update grid
                self.og.update_grid(matched_pose['x'], matched_pose['y'], px, py)
            
            # Save results
            corrected_poses.append(matched_pose)
            confidences.append(confidence)
            
            # Update for next iteration
            prev_pose = matched_pose
            prev_moving_theta = raw_moving_theta
            
            # Calculate matched movement theta for next iteration
            if est_moving_dist > 0.3:
                if i > 1:
                    prev_matched_x = corrected_poses[i-1]['x']
                    prev_matched_y = corrected_poses[i-1]['y']
                    dx_matched = matched_pose['x'] - prev_matched_x
                    dy_matched = matched_pose['y'] - prev_matched_y
                    if dx_matched**2 + dy_matched**2 > 0:
                        self.prev_matched_moving_theta = math.atan2(dy_matched, dx_matched)
        
        return corrected_poses, confidences
    
    def visualize_trajectory(self, pose_list, confidences=None):
        """
        Visualize the trajectory and occupancy grid
        
        Args:
            pose_list (list): List of corrected poses
            confidences (list): Optional list of match confidences
        """
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
            norm_conf = (norm_conf - norm_conf.min()) / (norm_conf.max() - norm_conf.min())
            
            # Plot each position
            for i, (x, y, conf) in enumerate(zip(x_trajectory, y_trajectory, norm_conf)):
                if i % 10 == 0:  # Plot every 10th point to avoid clutter
                    plt.scatter(x, y, color=cm.jet(conf), s=30, alpha=0.7)
        else:
            # Plot every 10th position with rainbow colors
            for i, (x, y) in enumerate(zip(x_trajectory, y_trajectory)):
                if i % 10 == 0:
                    plt.scatter(x, y, color=next(colors), s=30)
        
        # Plot occupancy grid as background
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
        Plot the occupancy grid
        
        Args:
            ax: Matplotlib axis to plot on (creates a new figure if None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get grid data
        grid_data = self.og.get_grid()
        
        # Custom colormap: white (unknown), black (occupied), light gray (free)
        colormap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'black'])
        bounds = [0, 0.4, 0.6, 1]
        norm = plt.cm.colors.BoundaryNorm(bounds, colormap.N)
        
        # Plot the grid
        width = self.og.width
        height = self.og.height
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


def update_estimated_pose(current_raw_reading, prev_matched_reading, prev_raw_reading, 
                          prev_raw_moving_theta, prev_matched_moving_theta):
    """
    Update estimated pose based on previous readings and movements.
    
    Args:
        current_raw_reading (dict): Current raw sensor reading
        prev_matched_reading (dict): Previous matched reading
        prev_raw_reading (dict): Previous raw reading
        prev_raw_moving_theta (float): Previous raw movement angle
        prev_matched_moving_theta (float): Previous matched movement angle
        
    Returns:
        tuple: (estimated_reading, est_moving_dist, est_moving_theta, raw_moving_theta)
    """
    # Calculate estimated orientation
    estimated_theta = (prev_matched_reading['theta'] + 
                      current_raw_reading['theta'] - 
                      prev_raw_reading['theta'])
    
    # Create estimated reading
    estimated_reading = {
        'x': prev_matched_reading['x'],
        'y': prev_matched_reading['y'],
        'theta': estimated_theta,
        'range': current_raw_reading['range'] if 'range' in current_raw_reading 
                else current_raw_reading['scan_ranges']
    }
    
    # Calculate movement
    dx = current_raw_reading['x'] - prev_raw_reading['x']
    dy = current_raw_reading['y'] - prev_raw_reading['y']
    est_moving_dist = math.sqrt(dx**2 + dy**2)
    
    # Calculate movement angles for significant movements
    raw_moving_theta = None
    est_moving_theta = None
    
    if est_moving_dist > 0.3:
        raw_moving_theta = math.atan2(dy, dx)
        
        if prev_raw_moving_theta is not None:
            raw_turn_theta = raw_moving_theta - prev_raw_moving_theta
            est_moving_theta = prev_matched_moving_theta + raw_turn_theta
    
    return estimated_reading, est_moving_dist, est_moving_theta, raw_moving_theta


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
    
    # Create scan matcher
    print("Initializing improved scan matcher")
    matcher = ScanMatcher(
        occupancy_grid_map=grid,
        search_radius=1.4,
        search_half_rad=0.25,
        scan_sigma_in_num_grid=2,
        move_r_sigma=0.1,
        max_move_deviation=0.25,
        turn_sigma=0.3,
        mismatch_prob_at_coarse=0.15,
        coarse_factor=5
    )
    
    # Set LiDAR parameters
    matcher.set_lidar_params(
        max_range=10.0,
        field_of_view=math.pi,
        num_samples=len(data_list[0]['scan_ranges'])
    )
    
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
    
    print(f"Results saved to {filename}")
    print("Done!")

if __name__ == "__main__":
    main()