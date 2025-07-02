# feature_extraction.py - Phase 1 Implementation

import numpy as np
import math
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import cv2

class LiDARFeature:
    """Base class for LiDAR features"""
    def __init__(self, position, feature_type, descriptor=None, quality=1.0):
        self.position = np.array(position)  # [x, y]
        self.feature_type = feature_type    # 'corner', 'line', 'curve'
        self.descriptor = descriptor        # Feature descriptor
        self.quality = quality             # Detection confidence
        self.id = None                     # For tracking across frames

class LiDARFeatureExtractor:
    """Extract geometric features from 2D LiDAR scans"""
    
    def __init__(self, corner_threshold=0.1, line_min_points=5, 
                 line_max_distance=0.05, curve_min_points=8):
        self.corner_threshold = corner_threshold
        self.line_min_points = line_min_points
        self.line_max_distance = line_max_distance
        self.curve_min_points = curve_min_points
        
        # Parameters for adaptive extraction
        self.min_point_distance = 0.02  # Minimum distance between consecutive points
        self.max_scan_range = 10.0      # Maximum valid scan range
        
    def extract_features(self, scan_x, scan_y, robot_pose=None):
        """
        Extract all types of features from a LiDAR scan
        
        Args:
            scan_x, scan_y: LiDAR scan points in world frame
            robot_pose: Current robot pose (for quality assessment)
            
        Returns:
            List of LiDARFeature objects
        """
        # Preprocess scan points
        points = self._preprocess_scan(scan_x, scan_y)
        
        if len(points) < 5:
            return []
        
        features = []
        
        # Extract different types of features
        corners = self._extract_corners(points)
        lines = self._extract_lines(points)
        curves = self._extract_curves(points)
        
        features.extend(corners)
        features.extend(lines)
        features.extend(curves)
        
        # Assign unique IDs and compute descriptors
        for i, feature in enumerate(features):
            feature.id = i
            feature.descriptor = self._compute_descriptor(feature, points)
        
        # Quality assessment and filtering
        features = self._filter_features_by_quality(features, robot_pose)
        
        return features
    
    def _preprocess_scan(self, scan_x, scan_y):
        """
        Preprocess scan points: remove outliers, sort by angle
        """
        points = np.column_stack((scan_x, scan_y))
        
        # Remove points that are too close to origin or too far
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        valid_mask = (distances > 0.1) & (distances < self.max_scan_range)
        points = points[valid_mask]
        
        if len(points) < 3:
            return points
        
        # Sort points by angle from robot
        angles = np.arctan2(points[:, 1], points[:, 0])
        sorted_indices = np.argsort(angles)
        points = points[sorted_indices]
        
        # Remove points that are too close together
        if len(points) > 1:
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            keep_mask = np.concatenate(([True], distances > self.min_point_distance))
            points = points[keep_mask]
        
        return points
    
    def _extract_corners(self, points):
        """
        Extract corner features using curvature analysis
        """
        if len(points) < 5:
            return []
        
        corners = []
        window_size = 3  # Points on each side for curvature calculation
        
        for i in range(window_size, len(points) - window_size):
            # Calculate curvature at point i
            curvature = self._calculate_curvature(points, i, window_size)
            
            # Check if this is a corner (high curvature)
            if abs(curvature) > self.corner_threshold:
                # Verify it's a local maximum
                is_local_max = True
                for j in range(max(0, i-2), min(len(points), i+3)):
                    if j != i:
                        other_curvature = self._calculate_curvature(points, j, window_size)
                        if abs(other_curvature) > abs(curvature):
                            is_local_max = False
                            break
                
                if is_local_max:
                    corner = LiDARFeature(
                        position=points[i],
                        feature_type='corner',
                        quality=abs(curvature)
                    )
                    corners.append(corner)
        
        # Remove corners that are too close to each other
        corners = self._remove_duplicate_features(corners, min_distance=0.2)
        
        return corners
    
    def _calculate_curvature(self, points, center_idx, window_size):
        """
        Calculate curvature at a point using neighboring points
        """
        start_idx = max(0, center_idx - window_size)
        end_idx = min(len(points), center_idx + window_size + 1)
        
        if end_idx - start_idx < 3:
            return 0.0
        
        # Fit a circle through the points and calculate curvature
        local_points = points[start_idx:end_idx]
        
        # Simple approximation: angle change
        if len(local_points) < 3:
            return 0.0
        
        # Calculate vectors
        v1 = local_points[len(local_points)//2] - local_points[0]
        v2 = local_points[-1] - local_points[len(local_points)//2]
        
        # Calculate angle between vectors
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Return curvature (higher for sharper corners)
        return math.pi - angle
    
    def _extract_lines(self, points):
        """
        Extract line segment features using RANSAC
        """
        if len(points) < self.line_min_points:
            return []
        
        lines = []
        remaining_points = points.copy()
        remaining_indices = np.arange(len(points))
        
        while len(remaining_points) >= self.line_min_points:
            # RANSAC for line fitting
            best_line = None
            best_inliers = []
            best_score = 0
            
            for _ in range(50):  # RANSAC iterations
                if len(remaining_points) < 2:
                    break
                
                # Sample two random points
                sample_indices = np.random.choice(len(remaining_points), 2, replace=False)
                p1, p2 = remaining_points[sample_indices]
                
                # Find inliers
                inliers = []
                line_vec = p2 - p1
                line_length = np.linalg.norm(line_vec)
                
                if line_length < 0.1:  # Too short
                    continue
                
                line_unit = line_vec / line_length
                
                for j, point in enumerate(remaining_points):
                    # Distance from point to line
                    point_vec = point - p1
                    proj_length = np.dot(point_vec, line_unit)
                    
                    # Check if projection is within line segment bounds
                    if 0 <= proj_length <= line_length:
                        proj_point = p1 + proj_length * line_unit
                        distance = np.linalg.norm(point - proj_point)
                        
                        if distance < self.line_max_distance:
                            inliers.append(j)
                
                # Score this line
                if len(inliers) > best_score and len(inliers) >= self.line_min_points:
                    best_score = len(inliers)
                    best_inliers = inliers.copy()
                    
                    # Fit line to all inliers
                    inlier_points = remaining_points[inliers]
                    centroid = np.mean(inlier_points, axis=0)
                    
                    # PCA to find line direction
                    centered_points = inlier_points - centroid
                    cov_matrix = np.cov(centered_points.T)
                    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                    
                    # Direction of maximum variance
                    direction = eigenvecs[:, np.argmax(eigenvals)]
                    
                    # Find line endpoints
                    projections = np.dot(centered_points, direction)
                    min_proj, max_proj = np.min(projections), np.max(projections)
                    
                    start_point = centroid + min_proj * direction
                    end_point = centroid + max_proj * direction
                    
                    best_line = {
                        'start': start_point,
                        'end': end_point,
                        'direction': direction,
                        'centroid': centroid,
                        'length': max_proj - min_proj
                    }
            
            # Add the best line if found
            if best_line and best_score >= self.line_min_points:
                line_feature = LiDARFeature(
                    position=best_line['centroid'],
                    feature_type='line',
                    quality=best_score / len(remaining_points)
                )
                # Store line-specific data
                line_feature.start_point = best_line['start']
                line_feature.end_point = best_line['end']
                line_feature.direction = best_line['direction']
                line_feature.length = best_line['length']
                
                lines.append(line_feature)
                
                # Remove inliers from remaining points
                remaining_points = np.delete(remaining_points, best_inliers, axis=0)
                remaining_indices = np.delete(remaining_indices, best_inliers)
            else:
                break
        
        return lines
    
    def _extract_curves(self, points):
        """
        Extract curved features (simplified implementation)
        """
        # For now, implement a simple curve detector
        # This could be enhanced with spline fitting, arc detection, etc.
        curves = []
        
        if len(points) < self.curve_min_points:
            return curves
        
        # Look for sequences of points with consistent curvature
        window_size = 5
        curve_points = []
        
        for i in range(len(points) - window_size):
            window_points = points[i:i+window_size]
            
            # Check if points form a smooth curve
            if self._is_smooth_curve(window_points):
                curve_points.extend(range(i, i+window_size))
        
        # Group consecutive curve points
        if curve_points:
            # Remove duplicates and sort
            curve_points = sorted(list(set(curve_points)))
            
            # Find continuous segments
            segments = []
            current_segment = [curve_points[0]]
            
            for i in range(1, len(curve_points)):
                if curve_points[i] == curve_points[i-1] + 1:
                    current_segment.append(curve_points[i])
                else:
                    if len(current_segment) >= self.curve_min_points:
                        segments.append(current_segment)
                    current_segment = [curve_points[i]]
            
            # Don't forget the last segment
            if len(current_segment) >= self.curve_min_points:
                segments.append(current_segment)
            
            # Create curve features
            for segment in segments:
                curve_points_segment = points[segment]
                centroid = np.mean(curve_points_segment, axis=0)
                
                curve_feature = LiDARFeature(
                    position=centroid,
                    feature_type='curve',
                    quality=len(segment) / len(points)
                )
                curve_feature.curve_points = curve_points_segment
                curves.append(curve_feature)
        
        return curves
    
    def _is_smooth_curve(self, points):
        """
        Check if a sequence of points forms a smooth curve
        """
        if len(points) < 3:
            return False
        
        # Calculate curvatures
        curvatures = []
        for i in range(1, len(points) - 1):
            curvature = self._calculate_curvature(points, i, 1)
            curvatures.append(abs(curvature))
        
        # Check if curvatures are consistent (not too variable)
        if len(curvatures) < 2:
            return False
        
        curvature_std = np.std(curvatures)
        curvature_mean = np.mean(curvatures)
        
        # Curve should have consistent, moderate curvature
        return (curvature_mean > 0.05 and curvature_mean < 0.5 and 
                curvature_std < curvature_mean * 0.5)
    
    def _compute_descriptor(self, feature, scan_points):
        """
        Compute a descriptor for the feature based on local geometry
        """
        try:
            if feature.feature_type == 'corner':
                return self._compute_corner_descriptor(feature, scan_points)
            elif feature.feature_type == 'line':
                return self._compute_line_descriptor(feature)
            elif feature.feature_type == 'curve':
                return self._compute_curve_descriptor(feature)
            else:
                # Fallback: simple position-based descriptor
                return np.array([feature.position[0] % 1.0, feature.position[1] % 1.0, 
                               feature.quality, 0.0], dtype=np.float32)
        except Exception as e:
            print(f"Error computing descriptor for {feature.feature_type}: {e}")
            # Return a simple fallback descriptor
            return np.array([feature.position[0] % 1.0, feature.position[1] % 1.0, 
                           feature.quality, 0.0], dtype=np.float32)
    
    def _compute_corner_descriptor(self, corner, scan_points):
        """
        Compute descriptor for corner feature
        """
        try:
            # Find points near the corner
            distances = np.sqrt(np.sum((scan_points - corner.position)**2, axis=1))
            nearby_mask = distances < 0.5  # 50cm radius
            nearby_points = scan_points[nearby_mask]
            
            if len(nearby_points) < 3:
                # Not enough nearby points, return simple descriptor
                return np.array([corner.quality, 0.0, 0.0, 0.0], dtype=np.float32)
            
            # Compute local geometry descriptor
            # - Angles to nearby points
            # - Distance distribution
            # - Local density
            
            relative_points = nearby_points - corner.position
            angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
            distances = np.sqrt(np.sum(relative_points**2, axis=1))
            
            # Histogram of angles (4 bins)
            angle_hist, _ = np.histogram(angles, bins=4, range=(-np.pi, np.pi))
            angle_hist = angle_hist / np.sum(angle_hist) if np.sum(angle_hist) > 0 else angle_hist
            
            return angle_hist.astype(np.float32)
            
        except Exception as e:
            print(f"Error computing corner descriptor: {e}")
            # Return fallback descriptor
            return np.array([corner.quality, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def _compute_line_descriptor(self, line):
        """
        Compute descriptor for line feature
        """
        try:
            # Check if line has required attributes
            if not hasattr(line, 'direction') or not hasattr(line, 'length'):
                # Fallback descriptor
                return np.array([0.0, 0.0, line.quality, 0.0], dtype=np.float32)
            
            # Line descriptor: [orientation, length, normalized_position]
            orientation = np.arctan2(line.direction[1], line.direction[0])
            # Normalize orientation to [0, pi]
            if orientation < 0:
                orientation += np.pi
            
            # Normalize length (relative to typical scan range)
            normalized_length = min(line.length / 2.0, 1.0)
            
            descriptor = np.array([
                np.cos(orientation),
                np.sin(orientation),
                normalized_length,
                line.quality
            ], dtype=np.float32)
            
            # Ensure all values are finite
            if not np.all(np.isfinite(descriptor)):
                return np.array([0.0, 0.0, line.quality, 0.0], dtype=np.float32)
            
            return descriptor
            
        except Exception as e:
            print(f"Error computing line descriptor: {e}")
            return np.array([0.0, 0.0, line.quality, 0.0], dtype=np.float32)
    
    def _compute_curve_descriptor(self, curve):
        """
        Compute descriptor for curve feature
        """
        try:
            if not hasattr(curve, 'curve_points') or len(curve.curve_points) < 3:
                return np.array([0.0, 0.0, 0.0, curve.quality], dtype=np.float32)
            
            # Curve descriptor: curvature statistics
            curvatures = []
            points = curve.curve_points
            
            for i in range(1, len(points) - 1):
                curvature = self._calculate_curvature(points, i, 1)
                if np.isfinite(curvature):
                    curvatures.append(curvature)
            
            if not curvatures:
                return np.array([0.0, 0.0, 0.0, curve.quality], dtype=np.float32)
            
            curvatures = np.array(curvatures)
            descriptor = np.array([
                np.mean(curvatures),
                np.std(curvatures),
                min(len(points) / 20.0, 1.0),  # Normalized length
                curve.quality
            ], dtype=np.float32)
            
            # Ensure all values are finite
            if not np.all(np.isfinite(descriptor)):
                return np.array([0.0, 0.0, 0.0, curve.quality], dtype=np.float32)
            
            return descriptor
            
        except Exception as e:
            print(f"Error computing curve descriptor: {e}")
            return np.array([0.0, 0.0, 0.0, curve.quality], dtype=np.float32)
    
    def _filter_features_by_quality(self, features, robot_pose=None):
        """
        Filter features based on quality metrics
        """
        if not features:
            return features
        
        # Sort by quality
        features.sort(key=lambda f: f.quality, reverse=True)
        
        # Keep top features and remove low-quality ones
        min_quality = 0.1
        max_features = 50  # Limit to prevent computational overload
        
        filtered = []
        for feature in features:
            if feature.quality >= min_quality and len(filtered) < max_features:
                filtered.append(feature)
        
        return filtered
    
    def _remove_duplicate_features(self, features, min_distance=0.1):
        """
        Remove features that are too close to each other
        """
        if len(features) <= 1:
            return features
        
        filtered = []
        
        for feature in features:
            is_duplicate = False
            for existing in filtered:
                distance = np.linalg.norm(feature.position - existing.position)
                if distance < min_distance:
                    # Keep the higher quality feature
                    if feature.quality > existing.quality:
                        filtered.remove(existing)
                    else:
                        is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(feature)
        
        return filtered

# Example usage:
def demo_feature_extraction():
    """Demo of feature extraction on sample LiDAR data"""
    
    # Create sample LiDAR scan (a room with corners)
    angles = np.linspace(-np.pi/2, np.pi/2, 180)
    ranges = np.ones_like(angles) * 5.0
    
    # Add some walls and corners
    ranges[angles < -np.pi/4] = 3.0  # Left wall
    ranges[angles > np.pi/4] = 4.0   # Right wall
    ranges[np.abs(angles) < 0.1] = 6.0  # Front opening
    
    # Convert to Cartesian
    scan_x = ranges * np.cos(angles)
    scan_y = ranges * np.sin(angles)
    
    # Extract features
    extractor = LiDARFeatureExtractor()
    features = extractor.extract_features(scan_x, scan_y)
    
    print(f"Extracted {len(features)} features:")
    for i, feature in enumerate(features):
        print(f"  {i}: {feature.feature_type} at {feature.position} "
              f"(quality: {feature.quality:.3f})")
    
    return features

if __name__ == "__main__":
    demo_feature_extraction()