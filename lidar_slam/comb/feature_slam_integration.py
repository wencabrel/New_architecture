# feature_slam_integration.py - Integration with existing SLAM system

import numpy as np
import math
import copy
from feature_extraction import LiDARFeatureExtractor, LiDARFeature
from data_association import DataAssociation, FeatureMatch
from ScanMatcher import PoseEstimate, ImprovedScanMatchingLocalization

class FeatureBasedScanMatcher(ImprovedScanMatchingLocalization):
    """
    Enhanced scan matcher that uses features for improved performance
    Extends the existing ImprovedScanMatchingLocalization class
    """
    
    def __init__(self, occupancy_grid=None, debug_level=1, use_features=True, 
                 hybrid_mode=True):
        """
        Initialize feature-based scan matcher
        
        Args:
            occupancy_grid: OccupancyGrid object
            debug_level: Debug verbosity level
            use_features: Whether to use feature-based matching
            hybrid_mode: Whether to combine feature and point-based approaches
        """
        # Initialize parent class
        super().__init__(occupancy_grid, debug_level)
        
        # Feature extraction and matching components
        self.use_features = use_features
        self.hybrid_mode = hybrid_mode
        
        if self.use_features:
            self.feature_extractor = LiDARFeatureExtractor()
            self.data_associator = DataAssociation()
            
            # Feature-specific parameters
            self.min_features_for_matching = 3
            self.feature_weight = 0.7  # Weight for feature-based pose estimation
            self.point_weight = 0.3    # Weight for point-based refinement
            
            # Feature tracking
            self.previous_features = []
            self.feature_database = []  # For loop closure
            self.feature_match_history = []
            
            # Performance tracking
            self.feature_stats = {
                'features_extracted': 0,
                'successful_matches': 0,
                'failed_matches': 0,
                'hybrid_corrections': 0
            }
        
        if self.debug_level > 0:
            mode_str = "Feature + Point Hybrid" if hybrid_mode else "Feature-only"
            print(f"[FeatureSLAM] Initialized with {mode_str} matching")
    
    def matchScan(self, scan_x, scan_y, initial_pose):
        """
        Override parent matchScan to use feature-based approach
        
        Args:
            scan_x, scan_y: Scan points in world frame
            initial_pose: Initial pose estimate
            
        Returns:
            Tuple of (matched_pose, match_info)
        """
        if not self.use_features:
            # Fall back to parent implementation
            return super().matchScan(scan_x, scan_y, initial_pose)
        
        # Extract features from current scan
        current_features = self.feature_extractor.extract_features(
            scan_x, scan_y, initial_pose
        )
        
        self.feature_stats['features_extracted'] += len(current_features)
        
        if self.debug_level > 2:
            print(f"[FeatureSLAM] Extracted {len(current_features)} features: "
                  f"{sum(1 for f in current_features if f.feature_type == 'corner')} corners, "
                  f"{sum(1 for f in current_features if f.feature_type == 'line')} lines, "
                  f"{sum(1 for f in current_features if f.feature_type == 'curve')} curves")
        
        # Initialize match info
        match_info = {
            'iterations': 0,
            'final_score': 0.0,
            'error': float('inf'),
            'correspondences': 0,
            'resampling_attempts': 0,
            'feature_matches': 0,
            'method': 'feature_based'
        }
        
        # Try feature-based matching first
        feature_pose = None
        feature_matches = []
        
        if (len(current_features) >= self.min_features_for_matching and 
            len(self.previous_features) >= self.min_features_for_matching):
            
            # Predict feature positions based on motion model
            predicted_transform = self._predict_motion_transform(initial_pose)
            
            # Match features
            feature_matches = self.data_associator.match_with_prediction(
                self.previous_features, current_features, predicted_transform
            )
            
            if self.debug_level > 2:
                print(f"[FeatureSLAM] Found {len(feature_matches)} initial feature matches")
            
            # Verify geometric consistency
            consistent_matches = self.data_associator.verify_geometric_consistency(feature_matches)
            
            if len(consistent_matches) >= self.min_features_for_matching:
                # Estimate pose from feature matches
                relative_pose = self.data_associator.estimate_relative_pose(consistent_matches)
                
                if relative_pose and relative_pose['confidence'] > 0.3:
                    # Convert relative pose to absolute pose
                    feature_pose = self._apply_relative_pose(
                        self.last_matched_pose, relative_pose
                    )
                    
                    match_info['feature_matches'] = len(consistent_matches)
                    match_info['correspondences'] = len(consistent_matches)
                    match_info['final_score'] = relative_pose['confidence']
                    match_info['error'] = 1.0 - relative_pose['confidence']
                    
                    self.feature_stats['successful_matches'] += 1
                    
                    if self.debug_level > 1:
                        print(f"[FeatureSLAM] Feature-based pose estimate: "
                              f"x={feature_pose.x:.3f}, y={feature_pose.y:.3f}, "
                              f"theta={feature_pose.theta:.3f} "
                              f"(confidence: {relative_pose['confidence']:.3f})")
        
        # Handle different scenarios based on feature matching success
        if feature_pose is not None and self.hybrid_mode:
            # Hybrid approach: use features for initial estimate, refine with points
            final_pose, point_match_info = super().matchScan(scan_x, scan_y, feature_pose)
            
            # Combine match information
            match_info.update(point_match_info)
            match_info['method'] = 'hybrid'
            match_info['feature_matches'] = len(feature_matches)
            
            # Weight the final pose based on confidence
            if point_match_info['final_score'] > 0.5:
                # Point matching also succeeded, blend the estimates
                blended_pose = self._blend_pose_estimates(
                    feature_pose, final_pose, 
                    self.feature_weight, self.point_weight
                )
                final_pose = blended_pose
                match_info['method'] = 'hybrid_blended'
                self.feature_stats['hybrid_corrections'] += 1
            
            if self.debug_level > 1:
                print(f"[FeatureSLAM] Hybrid matching successful")
                
        elif feature_pose is not None:
            # Feature-only mode
            final_pose = feature_pose
            match_info['method'] = 'feature_only'
            
            if self.debug_level > 1:
                print(f"[FeatureSLAM] Feature-only matching successful")
                
        else:
            # Fall back to point-based matching
            final_pose, point_match_info = super().matchScan(scan_x, scan_y, initial_pose)
            match_info.update(point_match_info)
            match_info['method'] = 'point_fallback'
            match_info['feature_matches'] = 0
            
            self.feature_stats['failed_matches'] += 1
            
            if self.debug_level > 1:
                print(f"[FeatureSLAM] Fell back to point-based matching")
        
        # Store features for next iteration
        self.previous_features = current_features
        
        # Store in feature database for loop closure
        if len(current_features) > 0:
            self.feature_database.append({
                'features': copy.deepcopy(current_features),
                'pose': copy.deepcopy(final_pose),
                'frame_id': len(self.trajectory)
            })
        
        # Store match history
        self.feature_match_history.append({
            'feature_matches': feature_matches,
            'pose': copy.deepcopy(final_pose),
            'match_info': copy.deepcopy(match_info)
        })
        
        return final_pose, match_info
    
    def _predict_motion_transform(self, current_pose):
        """
        Predict motion transformation based on recent trajectory
        """
        if len(self.trajectory) < 2:
            return None
        
        # Calculate motion from last two poses
        prev_pose = self.trajectory[-1]
        
        dx = current_pose.x - prev_pose.x
        dy = current_pose.y - prev_pose.y  
        dtheta = current_pose.theta - prev_pose.theta
        
        # Normalize angle
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi
        
        return {
            'x': dx,
            'y': dy,
            'theta': dtheta
        }
    
    def _apply_relative_pose(self, base_pose, relative_pose):
        """
        Apply relative transformation to get absolute pose
        """
        # Apply relative transformation
        cos_theta = math.cos(base_pose.theta)
        sin_theta = math.sin(base_pose.theta)
        
        # Rotate relative translation
        dx_global = (relative_pose['x'] * cos_theta - 
                    relative_pose['y'] * sin_theta)
        dy_global = (relative_pose['x'] * sin_theta + 
                    relative_pose['y'] * cos_theta)
        
        # Create new pose
        new_pose = PoseEstimate(
            x=base_pose.x + dx_global,
            y=base_pose.y + dy_global,
            theta=base_pose.theta + relative_pose['theta']
        )
        
        # Normalize angle
        new_pose.theta = (new_pose.theta + math.pi) % (2 * math.pi) - math.pi
        
        return new_pose
    
    def _blend_pose_estimates(self, pose1, pose2, weight1, weight2):
        """
        Blend two pose estimates with given weights
        """
        # Normalize weights
        total_weight = weight1 + weight2
        w1 = weight1 / total_weight
        w2 = weight2 / total_weight
        
        # Blend position
        x = w1 * pose1.x + w2 * pose2.x
        y = w1 * pose1.y + w2 * pose2.y
        
        # Blend orientation (handle angle wraparound)
        theta1 = pose1.theta
        theta2 = pose2.theta
        
        # Find the shortest angular path
        diff = theta2 - theta1
        if diff > math.pi:
            theta2 -= 2 * math.pi
        elif diff < -math.pi:
            theta2 += 2 * math.pi
        
        theta = w1 * theta1 + w2 * theta2
        
        # Normalize final angle
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        
        return PoseEstimate(x, y, theta)
    
    def detect_loop_closure_with_features(self, current_features, similarity_threshold=0.7):
        """
        Detect loop closure using feature similarity
        Enhanced version of loop closure that uses features
        """
        if len(self.feature_database) < 10:  # Need some history
            return None
        
        best_match = None
        best_similarity = 0
        
        # Compare with features from earlier in the trajectory
        for i, db_entry in enumerate(self.feature_database[:-10]):  # Skip recent frames
            # Calculate feature similarity
            similarity = self._calculate_feature_similarity(
                current_features, db_entry['features']
            )
            
            if similarity > best_similarity and similarity > similarity_threshold:
                best_similarity = similarity
                best_match = {
                    'frame_id': db_entry['frame_id'],
                    'pose': db_entry['pose'],
                    'features': db_entry['features'],
                    'similarity': similarity
                }
        
        return best_match
    
    def _calculate_feature_similarity(self, features1, features2):
        """
        Calculate similarity between two sets of features
        """
        if not features1 or not features2:
            return 0.0
        
        # Find matches between feature sets
        matches = self.data_associator.match_features(features1, features2)
        
        if not matches:
            return 0.0
        
        # Calculate similarity based on number and quality of matches
        max_features = max(len(features1), len(features2))
        match_ratio = len(matches) / max_features
        
        # Weight by match confidence
        avg_confidence = np.mean([match.confidence for match in matches])
        
        similarity = match_ratio * avg_confidence
        
        return similarity
    
    def get_feature_statistics(self):
        """
        Get statistics about feature extraction and matching performance
        """
        total_attempts = (self.feature_stats['successful_matches'] + 
                         self.feature_stats['failed_matches'])
        
        if total_attempts == 0:
            success_rate = 0
        else:
            success_rate = self.feature_stats['successful_matches'] / total_attempts
        
        stats = {
            'total_features_extracted': self.feature_stats['features_extracted'],
            'successful_matches': self.feature_stats['successful_matches'],
            'failed_matches': self.feature_stats['failed_matches'],
            'success_rate': success_rate,
            'hybrid_corrections': self.feature_stats['hybrid_corrections'],
            'features_per_frame': (self.feature_stats['features_extracted'] / 
                                 max(1, len(self.trajectory))),
            'current_features': len(self.previous_features)
        }
        
        return stats
    
    def visualize_feature_matches(self, scan_x, scan_y, pose, show_features=True):
        """
        Visualize current scan with extracted features and matches
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the scan points
        ax.scatter(scan_x, scan_y, c='lightblue', s=3, alpha=0.6, label='Scan Points')
        
        if show_features and len(self.previous_features) > 0:
            # Plot previous features
            for feature in self.previous_features:
                color = {'corner': 'red', 'line': 'green', 'curve': 'orange'}
                marker = {'corner': 'o', 'line': 's', 'curve': '^'}
                
                ax.scatter(feature.position[0], feature.position[1], 
                          c=color.get(feature.feature_type, 'blue'),
                          marker=marker.get(feature.feature_type, 'o'),
                          s=50, alpha=0.8,
                          label=f'Previous {feature.feature_type}' if feature.feature_type == 'corner' else "")
            
            # Extract and plot current features
            current_features = self.feature_extractor.extract_features(scan_x, scan_y, pose)
            
            for feature in current_features:
                color = {'corner': 'darkred', 'line': 'darkgreen', 'curve': 'darkorange'}
                marker = {'corner': 'o', 'line': 's', 'curve': '^'}
                
                ax.scatter(feature.position[0], feature.position[1], 
                          c=color.get(feature.feature_type, 'darkblue'),
                          marker=marker.get(feature.feature_type, 'o'),
                          s=80, alpha=0.9,
                          label=f'Current {feature.feature_type}' if feature.feature_type == 'corner' else "")
            
            # Show feature matches if available
            if len(self.feature_match_history) > 0:
                latest_matches = self.feature_match_history[-1]['feature_matches']
                
                for match in latest_matches:
                    # Draw line between matched features
                    ax.plot([match.feature1.position[0], match.feature2.position[0]],
                           [match.feature1.position[1], match.feature2.position[1]],
                           'purple', linewidth=1, alpha=0.6)
        
        # Plot robot position and orientation
        ax.scatter(pose.x, pose.y, c='blue', s=100, marker='*', label='Robot')
        
        # Orientation arrow
        arrow_length = 0.5
        dx = arrow_length * math.cos(pose.theta)
        dy = arrow_length * math.sin(pose.theta)
        ax.arrow(pose.x, pose.y, dx, dy, head_width=0.1, head_length=0.1, 
                fc='blue', ec='blue')
        
        # Formatting
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Feature-Based Scan Matching Visualization')
        ax.legend()
        
        # Add feature statistics
        stats = self.get_feature_statistics()
        stats_text = (f"Features: {stats['current_features']}\n"
                     f"Success Rate: {stats['success_rate']:.1%}\n"
                     f"Avg Features/Frame: {stats['features_per_frame']:.1f}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                va='top', ha='left', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig

# Example integration function
def integrate_features_with_existing_slam(slam_system):
    """
    Integrate feature-based approach with existing SLAM system
    
    Args:
        slam_system: Existing ImprovedScanMatchingLocalization instance
        
    Returns:
        Enhanced FeatureBasedScanMatcher instance
    """
    # Create new feature-based matcher with same parameters
    feature_slam = FeatureBasedScanMatcher(
        occupancy_grid=slam_system.map,
        debug_level=slam_system.debug_level,
        use_features=True,
        hybrid_mode=True
    )
    
    # Copy relevant state from existing system
    if hasattr(slam_system, 'trajectory'):
        feature_slam.trajectory = slam_system.trajectory.copy()
    if hasattr(slam_system, 'last_matched_pose'):
        feature_slam.last_matched_pose = slam_system.last_matched_pose
    if hasattr(slam_system, 'odometry_trajectory'):
        feature_slam.odometry_trajectory = slam_system.odometry_trajectory.copy()
    
    # Copy adaptive parameters
    feature_slam.max_correspondence_distance = slam_system.max_correspondence_distance
    feature_slam.occupancy_threshold = slam_system.occupancy_threshold
    feature_slam.max_translation_per_frame = slam_system.max_translation_per_frame
    feature_slam.max_rotation_per_frame = slam_system.max_rotation_per_frame
    
    print("[Integration] Successfully integrated feature-based matching with existing SLAM system")
    
    return feature_slam

# Performance comparison utility
def compare_slam_performance(point_based_slam, feature_based_slam, 
                           lidar_data_list, angle_min, angle_max):
    """
    Compare performance between point-based and feature-based SLAM
    """
    from lidar_utility_functions import convert_scans_to_cartesian
    import time
    
    results = {
        'point_based': {'times': [], 'poses': [], 'scores': []},
        'feature_based': {'times': [], 'poses': [], 'scores': []}
    }
    
    print("Comparing SLAM performance...")
    
    # Test with subset of data
    test_data = lidar_data_list[:50]  # First 50 scans
    
    for i, scan_data in enumerate(test_data):
        if i == 0:
            continue  # Skip first scan
            
        # Convert scan to Cartesian
        scan_x, scan_y = convert_scans_to_cartesian(
            scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
        )
        
        # Initial guess
        odometry_pose = PoseEstimate().from_dict(scan_data['pose'])
        
        # Test point-based approach
        start_time = time.time()
        point_pose, point_info = point_based_slam.matchScan(scan_x, scan_y, odometry_pose)
        point_time = time.time() - start_time
        
        results['point_based']['times'].append(point_time)
        results['point_based']['poses'].append(point_pose)
        results['point_based']['scores'].append(point_info['final_score'])
        
        # Test feature-based approach
        start_time = time.time()
        feature_pose, feature_info = feature_based_slam.matchScan(scan_x, scan_y, odometry_pose)
        feature_time = time.time() - start_time
        
        results['feature_based']['times'].append(feature_time)
        results['feature_based']['poses'].append(feature_pose)
        results['feature_based']['scores'].append(feature_info['final_score'])
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(test_data)} scans...")
    
    # Calculate statistics
    point_avg_time = np.mean(results['point_based']['times'])
    feature_avg_time = np.mean(results['feature_based']['times'])
    
    point_avg_score = np.mean(results['point_based']['scores'])
    feature_avg_score = np.mean(results['feature_based']['scores'])
    
    print(f"\nPerformance Comparison Results:")
    print(f"Point-based SLAM:")
    print(f"  Average time per scan: {point_avg_time:.4f}s")
    print(f"  Average match score: {point_avg_score:.3f}")
    
    print(f"Feature-based SLAM:")
    print(f"  Average time per scan: {feature_avg_time:.4f}s") 
    print(f"  Average match score: {feature_avg_score:.3f}")
    
    print(f"Speedup: {point_avg_time/feature_avg_time:.1f}x")
    print(f"Score improvement: {((feature_avg_score-point_avg_score)/point_avg_score)*100:.1f}%")
    
    return results

if __name__ == "__main__":
    print("Feature-based SLAM integration module")
    print("Use integrate_features_with_existing_slam() to enhance your current SLAM system")