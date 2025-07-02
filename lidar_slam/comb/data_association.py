# data_association.py - Feature Matching and Data Association

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import math

class FeatureMatch:
    """Represents a match between two features"""
    def __init__(self, feature1, feature2, distance, confidence=1.0):
        self.feature1 = feature1
        self.feature2 = feature2
        self.distance = distance
        self.confidence = confidence
        self.geometric_consistency = True

class DataAssociation:
    """Handle feature matching and data association between scans"""
    
    def __init__(self, max_descriptor_distance=0.3, max_spatial_distance=1.0,
                 ratio_test_threshold=0.8, geometric_consistency_threshold=0.2):
        """
        Initialize data association parameters
        
        Args:
            max_descriptor_distance: Maximum allowed descriptor distance for matches
            max_spatial_distance: Maximum spatial distance for feature matches
            ratio_test_threshold: Ratio test threshold (Lowe's ratio test)
            geometric_consistency_threshold: Threshold for geometric consistency check
        """
        self.max_descriptor_distance = max_descriptor_distance
        self.max_spatial_distance = max_spatial_distance
        self.ratio_test_threshold = ratio_test_threshold
        self.geometric_consistency_threshold = geometric_consistency_threshold
        
        # Parameters for different matching strategies
        self.use_hungarian_assignment = True
        self.use_geometric_verification = True
        self.min_matches_for_transform = 3
        
    def match_features(self, features1, features2, method='hybrid'):
        """
        Match features between two scans
        
        Args:
            features1: List of features from first scan
            features2: List of features from second scan
            method: Matching method ('nearest_neighbor', 'hungarian', 'hybrid')
            
        Returns:
            List of FeatureMatch objects
        """
        if not features1 or not features2:
            return []
        
        if method == 'nearest_neighbor':
            return self._nearest_neighbor_matching(features1, features2)
        elif method == 'hungarian':
            return self._hungarian_matching(features1, features2)
        elif method == 'hybrid':
            return self._hybrid_matching(features1, features2)
        else:
            raise ValueError(f"Unknown matching method: {method}")
    
    def _nearest_neighbor_matching(self, features1, features2):
        """
        Nearest neighbor matching with ratio test
        """
        matches = []
        
        if not features1 or not features2:
            return matches
        
        # Group features by type for more efficient matching
        feature_groups1 = self._group_features_by_type(features1)
        feature_groups2 = self._group_features_by_type(features2)
        
        for feature_type in feature_groups1.keys():
            if feature_type not in feature_groups2:
                continue
                
            group1 = feature_groups1[feature_type]
            group2 = feature_groups2[feature_type]
            
            if not group1 or not group2:
                continue
            
            # Check if all features have valid descriptors
            valid_group1 = [f for f in group1 if f.descriptor is not None and len(f.descriptor) > 0]
            valid_group2 = [f for f in group2 if f.descriptor is not None and len(f.descriptor) > 0]
            
            if not valid_group1 or not valid_group2:
                continue
            
            try:
                # Compute descriptor distances
                descriptors1 = np.array([f.descriptor for f in valid_group1])
                descriptors2 = np.array([f.descriptor for f in valid_group2])
                
                if descriptors1.size == 0 or descriptors2.size == 0:
                    continue
                
                # Ensure all descriptors have the same length
                desc_len1 = descriptors1.shape[1] if len(descriptors1.shape) > 1 else len(descriptors1[0])
                desc_len2 = descriptors2.shape[1] if len(descriptors2.shape) > 1 else len(descriptors2[0])
                
                if desc_len1 != desc_len2:
                    print(f"Warning: Descriptor length mismatch for {feature_type}: {desc_len1} vs {desc_len2}")
                    continue
                
                # Compute distance matrix
                dist_matrix = cdist(descriptors1, descriptors2, metric='euclidean')
                
                # For each feature in group1, find best matches in group2
                for i, feature1 in enumerate(valid_group1):
                    distances = dist_matrix[i]
                    sorted_indices = np.argsort(distances)
                    
                    if len(sorted_indices) == 0:
                        continue
                    
                    best_idx = sorted_indices[0]
                    best_distance = distances[best_idx]
                    
                    # Apply distance threshold
                    if best_distance > self.max_descriptor_distance:
                        continue
                    
                    # Apply ratio test (if we have at least 2 candidates)
                    if len(sorted_indices) > 1:
                        second_best_idx = sorted_indices[1]
                        second_best_distance = distances[second_best_idx]
                        
                        if second_best_distance > 0:
                            ratio = best_distance / second_best_distance
                            if ratio > self.ratio_test_threshold:
                                continue
                    
                    # Check spatial distance constraint
                    feature2 = valid_group2[best_idx]
                    spatial_distance = np.linalg.norm(feature1.position - feature2.position)
                    
                    if spatial_distance > self.max_spatial_distance:
                        continue
                    
                    # Create match
                    match = FeatureMatch(
                        feature1=feature1,
                        feature2=feature2,
                        distance=best_distance,
                        confidence=1.0 - (best_distance / self.max_descriptor_distance)
                    )
                    matches.append(match)
                    
            except Exception as e:
                print(f"Error in nearest neighbor matching for {feature_type}: {e}")
                continue
        
        return matches
    
    def _hungarian_matching(self, features1, features2):
        """
        Hungarian algorithm for optimal assignment
        """
        matches = []
        
        # Group features by type
        feature_groups1 = self._group_features_by_type(features1)
        feature_groups2 = self._group_features_by_type(features2)
        
        for feature_type in feature_groups1.keys():
            if feature_type not in feature_groups2:
                continue
                
            group1 = feature_groups1[feature_type]
            group2 = feature_groups2[feature_type]
            
            if not group1 or not group2:
                continue
            
            # Compute cost matrix (descriptor + spatial distance)
            cost_matrix = self._compute_cost_matrix(group1, group2)
            
            # Check if cost matrix is feasible (has at least one finite value)
            if not np.any(np.isfinite(cost_matrix)):
                # No valid matches possible, skip Hungarian algorithm
                continue
            
            # Check if cost matrix has valid matches within threshold
            valid_costs = cost_matrix[cost_matrix < self.max_descriptor_distance]
            if len(valid_costs) == 0:
                # No matches within threshold, skip
                continue
            
            try:
                # Apply Hungarian algorithm
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                # Create matches from assignments
                for row_idx, col_idx in zip(row_indices, col_indices):
                    cost = cost_matrix[row_idx, col_idx]
                    
                    # Only accept matches below threshold
                    if cost < self.max_descriptor_distance:
                        feature1 = group1[row_idx]
                        feature2 = group2[col_idx]
                        
                        match = FeatureMatch(
                            feature1=feature1,
                            feature2=feature2,
                            distance=cost,
                            confidence=1.0 - (cost / self.max_descriptor_distance)
                        )
                        matches.append(match)
                        
            except ValueError as e:
                # Hungarian algorithm failed, fall back to nearest neighbor for this group
                print(f"Hungarian algorithm failed for {feature_type} features: {e}")
                print(f"Falling back to nearest neighbor matching for this feature type")
                
                # Use nearest neighbor as fallback
                nn_matches = self._nearest_neighbor_matching(group1, group2)
                matches.extend(nn_matches)
        
        return matches
    
    def _hybrid_matching(self, features1, features2):
        """
        Hybrid approach: Hungarian for high-quality features, NN for others
        """
        if not features1 or not features2:
            return []
        
        # Separate features by quality
        high_quality1 = [f for f in features1 if f.quality > 0.7]
        low_quality1 = [f for f in features1 if f.quality <= 0.7]
        
        high_quality2 = [f for f in features2 if f.quality > 0.7]
        low_quality2 = [f for f in features2 if f.quality <= 0.7]
        
        matches = []
        
        # Use Hungarian for high-quality features
        if high_quality1 and high_quality2:
            try:
                hq_matches = self._hungarian_matching(high_quality1, high_quality2)
                matches.extend(hq_matches)
            except Exception as e:
                print(f"Hungarian matching failed, falling back to nearest neighbor: {e}")
                # Fall back to nearest neighbor for high-quality features
                hq_matches = self._nearest_neighbor_matching(high_quality1, high_quality2)
                matches.extend(hq_matches)
        
        # Use nearest neighbor for remaining features
        # Remove already matched features
        matched_features1 = set(m.feature1 for m in matches)
        matched_features2 = set(m.feature2 for m in matches)
        
        remaining1 = [f for f in low_quality1 if f not in matched_features1]
        remaining2 = [f for f in low_quality2 if f not in matched_features2]
        
        if remaining1 and remaining2:
            try:
                nn_matches = self._nearest_neighbor_matching(remaining1, remaining2)
                matches.extend(nn_matches)
            except Exception as e:
                print(f"Nearest neighbor matching failed: {e}")
                # Continue without these matches
        
        return matches
    
    def _group_features_by_type(self, features):
        """Group features by their type"""
        groups = {}
        for feature in features:
            if feature.feature_type not in groups:
                groups[feature.feature_type] = []
            groups[feature.feature_type].append(feature)
        return groups
    
    def _compute_cost_matrix(self, features1, features2):
        """
        Compute cost matrix combining descriptor and spatial distances
        """
        n1, n2 = len(features1), len(features2)
        cost_matrix = np.full((n1, n2), np.inf)
        
        # If either group is empty, return the inf matrix
        if n1 == 0 or n2 == 0:
            return cost_matrix
        
        for i, f1 in enumerate(features1):
            for j, f2 in enumerate(features2):
                # Check if descriptors exist and are valid
                if (f1.descriptor is None or f2.descriptor is None or 
                    len(f1.descriptor) == 0 or len(f2.descriptor) == 0):
                    continue
                
                # Ensure descriptors have the same length
                if len(f1.descriptor) != len(f2.descriptor):
                    continue
                
                try:
                    # Descriptor distance
                    desc_dist = np.linalg.norm(f1.descriptor - f2.descriptor)
                    
                    # Spatial distance
                    spatial_dist = np.linalg.norm(f1.position - f2.position)
                    
                    # Check if distances are reasonable
                    if (np.isfinite(desc_dist) and np.isfinite(spatial_dist) and
                        desc_dist <= self.max_descriptor_distance and 
                        spatial_dist <= self.max_spatial_distance):
                        
                        # Combined cost (weighted sum)
                        cost_matrix[i, j] = (0.7 * desc_dist + 
                                           0.3 * spatial_dist / self.max_spatial_distance)
                        
                except Exception as e:
                    # Skip this pair if there's any computation error
                    continue
        
        return cost_matrix
    
    def verify_geometric_consistency(self, matches):
        """
        Verify geometric consistency of matches using RANSAC
        """
        if len(matches) < self.min_matches_for_transform:
            return matches
        
        # Extract point correspondences
        points1 = np.array([match.feature1.position for match in matches])
        points2 = np.array([match.feature2.position for match in matches])
        
        # RANSAC for transformation estimation
        best_inliers = []
        best_transform = None
        max_iterations = 100
        
        for _ in range(max_iterations):
            if len(matches) < 3:
                break
                
            # Sample minimum set for transformation estimation
            sample_indices = np.random.choice(len(matches), 3, replace=False)
            sample_points1 = points1[sample_indices]
            sample_points2 = points2[sample_indices]
            
            # Estimate transformation (similarity transform)
            transform = self._estimate_similarity_transform(sample_points1, sample_points2)
            
            if transform is None:
                continue
            
            # Count inliers
            inliers = []
            for i, (p1, p2) in enumerate(zip(points1, points2)):
                # Apply transformation to p1
                p1_transformed = self._apply_transform(p1, transform)
                
                # Check if it's close to p2
                error = np.linalg.norm(p1_transformed - p2)
                
                if error < self.geometric_consistency_threshold:
                    inliers.append(i)
            
            # Update best if this is better
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_transform = transform
        
        # Mark matches as geometrically consistent or not
        for i, match in enumerate(matches):
            match.geometric_consistency = i in best_inliers
        
        # Return only geometrically consistent matches if verification is enabled
        if self.use_geometric_verification:
            return [match for match in matches if match.geometric_consistency]
        else:
            return matches
    
    def _estimate_similarity_transform(self, points1, points2):
        """
        Estimate similarity transformation (rotation, translation, scale) between point sets
        """
        if len(points1) != len(points2) or len(points1) < 2:
            return None
        
        # Center the points
        centroid1 = np.mean(points1, axis=0)
        centroid2 = np.mean(points2, axis=0)
        
        centered1 = points1 - centroid1
        centered2 = points2 - centroid2
        
        # Compute scale
        scale1 = np.sqrt(np.sum(centered1**2))
        scale2 = np.sqrt(np.sum(centered2**2))
        
        if scale1 == 0 or scale2 == 0:
            return None
        
        scale = scale2 / scale1
        
        # Normalize for rotation estimation
        normalized1 = centered1 / scale1
        normalized2 = centered2 / scale2
        
        # Estimate rotation using SVD
        H = np.dot(normalized1.T, normalized2)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Compute translation
        t = centroid2 - scale * np.dot(centroid1, R.T)
        
        return {
            'rotation': R,
            'translation': t,
            'scale': scale
        }
    
    def _apply_transform(self, point, transform):
        """Apply similarity transformation to a point"""
        R = transform['rotation']
        t = transform['translation']
        s = transform['scale']
        
        return s * np.dot(point, R.T) + t
    
    def estimate_relative_pose(self, matches):
        """
        Estimate relative pose from feature matches
        
        Returns:
            Dictionary with 'x', 'y', 'theta' for relative transformation
        """
        if len(matches) < 3:
            return None
        
        # Extract point correspondences
        points1 = np.array([match.feature1.position for match in matches])
        points2 = np.array([match.feature2.position for match in matches])
        
        # Estimate transformation
        transform = self._estimate_similarity_transform(points1, points2)
        
        if transform is None:
            return None
        
        # Extract pose parameters
        R = transform['rotation']
        t = transform['translation']
        
        # Rotation angle
        theta = math.atan2(R[1, 0], R[0, 0])
        
        return {
            'x': t[0],
            'y': t[1],
            'theta': theta,
            'confidence': len(matches) / max(len(points1), len(points2))
        }
    
    def match_with_prediction(self, features1, features2, predicted_transform=None):
        """
        Match features with motion prediction to improve associations
        
        Args:
            features1: Features from previous scan
            features2: Features from current scan  
            predicted_transform: Predicted transformation from motion model
            
        Returns:
            List of FeatureMatch objects
        """
        if predicted_transform is not None:
            # Transform features1 using prediction
            predicted_features1 = []
            for feature in features1:
                new_feature = LiDARFeature(
                    position=self._apply_transform_dict(feature.position, predicted_transform),
                    feature_type=feature.feature_type,
                    descriptor=feature.descriptor.copy(),
                    quality=feature.quality
                )
                predicted_features1.append(new_feature)
            
            # Use transformed features for matching
            return self.match_features(predicted_features1, features2)
        else:
            # Standard matching without prediction
            return self.match_features(features1, features2)
    
    def _apply_transform_dict(self, point, transform_dict):
        """Apply transformation from dictionary format"""
        x, y, theta = transform_dict['x'], transform_dict['y'], transform_dict['theta']
        
        # Rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Apply transformation
        rotated_x = point[0] * cos_theta - point[1] * sin_theta
        rotated_y = point[0] * sin_theta + point[1] * cos_theta
        
        return np.array([rotated_x + x, rotated_y + y])
    
    def filter_matches_by_consistency(self, matches):
        """
        Filter matches based on various consistency checks
        """
        if not matches:
            return matches
        
        filtered_matches = []
        
        # Group matches by feature type
        type_groups = {}
        for match in matches:
            feature_type = match.feature1.feature_type
            if feature_type not in type_groups:
                type_groups[feature_type] = []
            type_groups[feature_type].append(match)
        
        # Apply type-specific filtering
        for feature_type, group_matches in type_groups.items():
            if feature_type == 'corner':
                # For corners, ensure spatial distribution
                filtered = self._filter_corner_matches(group_matches)
            elif feature_type == 'line':
                # For lines, check orientation consistency
                filtered = self._filter_line_matches(group_matches)
            elif feature_type == 'curve':
                # For curves, check curvature consistency
                filtered = self._filter_curve_matches(group_matches)
            else:
                filtered = group_matches
            
            filtered_matches.extend(filtered)
        
        return filtered_matches
    
    def _filter_corner_matches(self, matches):
        """Filter corner matches for spatial consistency"""
        # Remove corners that are too close to each other
        filtered = []
        min_corner_distance = 0.3  # 30cm minimum between corners
        
        for match in matches:
            is_valid = True
            for existing in filtered:
                dist1 = np.linalg.norm(match.feature1.position - existing.feature1.position)
                dist2 = np.linalg.norm(match.feature2.position - existing.feature2.position)
                
                if dist1 < min_corner_distance or dist2 < min_corner_distance:
                    # Keep the match with higher confidence
                    if match.confidence > existing.confidence:
                        filtered.remove(existing)
                    else:
                        is_valid = False
                    break
            
            if is_valid:
                filtered.append(match)
        
        return filtered
    
    def _filter_line_matches(self, matches):
        """Filter line matches for orientation consistency"""
        # Check that line orientations are consistent with overall transformation
        if len(matches) < 2:
            return matches
        
        # Compute orientation differences
        orientation_diffs = []
        for match in matches:
            if (hasattr(match.feature1, 'direction') and 
                hasattr(match.feature2, 'direction')):
                
                angle1 = math.atan2(match.feature1.direction[1], match.feature1.direction[0])
                angle2 = math.atan2(match.feature2.direction[1], match.feature2.direction[0])
                
                diff = angle2 - angle1
                # Normalize to [-pi, pi]
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                orientation_diffs.append(diff)
        
        if not orientation_diffs:
            return matches
        
        # Find consensus orientation change
        median_diff = np.median(orientation_diffs)
        std_diff = np.std(orientation_diffs)
        
        # Filter matches that don't agree with consensus
        filtered = []
        for i, match in enumerate(matches):
            if i < len(orientation_diffs):
                diff = orientation_diffs[i]
                if abs(diff - median_diff) <= 2 * std_diff:
                    filtered.append(match)
            else:
                filtered.append(match)
        
        return filtered
    
    def _filter_curve_matches(self, matches):
        """Filter curve matches for curvature consistency"""
        # Simple filtering - could be enhanced with curvature analysis
        return matches

# Integration with existing LiDARFeature class
from feature_extraction import LiDARFeature

# Example usage and testing
def demo_data_association():
    """Demonstrate data association between two sets of features"""
    
    # Create sample features for two scans
    features1 = [
        LiDARFeature([1.0, 2.0], 'corner', np.array([0.8, 0.2, 0.1, 0.9]), 0.9),
        LiDARFeature([3.0, 1.0], 'line', np.array([0.7, 0.7, 0.5, 0.8]), 0.8),
        LiDARFeature([2.0, 3.0], 'corner', np.array([0.6, 0.3, 0.2, 0.7]), 0.7)
    ]
    
    # Second scan with slight transformation
    features2 = [
        LiDARFeature([1.1, 2.1], 'corner', np.array([0.8, 0.2, 0.1, 0.9]), 0.9),
        LiDARFeature([3.1, 1.1], 'line', np.array([0.7, 0.7, 0.5, 0.8]), 0.8),
        LiDARFeature([2.1, 3.1], 'corner', np.array([0.6, 0.3, 0.2, 0.7]), 0.7),
        LiDARFeature([4.0, 4.0], 'corner', np.array([0.5, 0.4, 0.3, 0.6]), 0.6)  # New feature
    ]
    
    # Perform data association
    associator = DataAssociation()
    
    # Test different matching methods
    methods = ['nearest_neighbor', 'hungarian', 'hybrid']
    
    for method in methods:
        print(f"\n{method.upper()} MATCHING:")
        matches = associator.match_features(features1, features2, method=method)
        
        print(f"Found {len(matches)} matches:")
        for i, match in enumerate(matches):
            print(f"  Match {i}: {match.feature1.feature_type} "
                  f"({match.feature1.position}) -> ({match.feature2.position}) "
                  f"distance: {match.distance:.3f}, confidence: {match.confidence:.3f}")
        
        # Verify geometric consistency
        consistent_matches = associator.verify_geometric_consistency(matches)
        print(f"Geometrically consistent matches: {len(consistent_matches)}")
        
        # Estimate relative pose
        pose = associator.estimate_relative_pose(consistent_matches)
        if pose:
            print(f"Estimated relative pose: x={pose['x']:.3f}, y={pose['y']:.3f}, "
                  f"theta={pose['theta']:.3f}, confidence={pose['confidence']:.3f}")

if __name__ == "__main__":
    demo_data_association()