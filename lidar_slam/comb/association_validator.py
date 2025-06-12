from __future__ import annotations
import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import existing components and association module
try:
    from feature_extractor import FeatureSet, LiDARFeature, FeatureType
    from pose_estimate import PoseEstimate
    from feature_association import FeatureDescriptor, AssociationScore, FeatureAssociationEngine
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("Warning: Some dependencies not available. Association validation will have limited functionality.")
    DEPENDENCIES_AVAILABLE = False


class ValidationResult:
    """Container for association validation results"""
    
    def __init__(self):
        self.is_valid = False
        self.confidence = 0.0
        self.pose_estimate = None  # PoseEstimate object
        self.inlier_count = 0
        self.total_associations = 0
        self.geometric_consistency = 0.0
        self.temporal_consistency = 0.0
        self.validation_method = "unknown"
        self.error_metrics = {}
        
        # Detailed results
        self.inlier_associations = []  # List of association indices that are inliers
        self.outlier_associations = []  # List of association indices that are outliers
        self.ransac_iterations = 0
        self.best_model_score = 0.0


class TemporalConsistencyChecker:
    """
    Checks temporal consistency of associations across multiple scans
    """
    5
    def __init__(self, history_length: int = 5, consistency_threshold: float = 0.7):
        """
        Initialize temporal consistency checker
        
        Args:
            history_length: Number of previous scans to consider
            consistency_threshold: Minimum consistency score to pass validation
        """
        self.history_length = history_length
        self.consistency_threshold = consistency_threshold
        
        # Store association history
        self.association_history = []  # List of association sets from previous scans
        self.feature_track_history = {}  # Track individual features across time
        
    def add_associations(self, associations: List["AssociationScore"], 
                        current_descriptors: List["FeatureDescriptor"],
                        previous_descriptors: List["FeatureDescriptor"]):
        """
        Add new associations to history and update feature tracks
        
        Args:
            associations: Current associations
            current_descriptors: Current scan descriptors
            previous_descriptors: Previous scan descriptors
        """
        # Store associations
        self.association_history.append(associations.copy())
        
        # Maintain history length
        if len(self.association_history) > self.history_length:
            self.association_history.pop(0)
        
        # Update feature tracks
        for assoc in associations:
            curr_desc = current_descriptors[assoc.feature_idx1]
            prev_desc = previous_descriptors[assoc.feature_idx2]
            
            # Create or update feature track
            track_id = f"track_{len(self.feature_track_history)}"
            
            # Try to find existing track for previous feature
            existing_track = None
            for tid, track in self.feature_track_history.items():
                if len(track) > 0 and track[-1]['descriptor_id'] == id(prev_desc):
                    existing_track = tid
                    break
            
            if existing_track:
                # Continue existing track
                self.feature_track_history[existing_track].append({
                    'descriptor_id': id(curr_desc),
                    'position': curr_desc.base_feature.point_world.copy(),
                    'feature_type': curr_desc.base_feature.feature_type,
                    'association_score': assoc.score,
                    'scan_index': curr_desc.scan_index
                })
            else:
                # Start new track
                self.feature_track_history[track_id] = [
                    {
                        'descriptor_id': id(prev_desc),
                        'position': prev_desc.base_feature.point_world.copy(),
                        'feature_type': prev_desc.base_feature.feature_type,
                        'association_score': 1.0,  # Initial score
                        'scan_index': prev_desc.scan_index
                    },
                    {
                        'descriptor_id': id(curr_desc),
                        'position': curr_desc.base_feature.point_world.copy(),
                        'feature_type': curr_desc.base_feature.feature_type,
                        'association_score': assoc.score,
                        'scan_index': curr_desc.scan_index
                    }
                ]
        
        # Clean old tracks (remove tracks that haven't been updated recently)
        current_scan_index = current_descriptors[0].scan_index if current_descriptors else 0
        tracks_to_remove = []
        for track_id, track in self.feature_track_history.items():
            if len(track) > 0 and current_scan_index - track[-1]['scan_index'] > self.history_length:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.feature_track_history[track_id]
    
    def validate_temporal_consistency(self, associations: List["AssociationScore"],
                                    current_descriptors: List["FeatureDescriptor"],
                                    previous_descriptors: List["FeatureDescriptor"]) -> float:
        """
        Validate temporal consistency of associations
        
        Args:
            associations: Associations to validate
            current_descriptors: Current scan descriptors
            previous_descriptors: Previous scan descriptors
            
        Returns:
            Temporal consistency score [0, 1]
        """
        if len(self.association_history) < 2:
            return 1.0  # Can't validate with insufficient history
        
        consistent_associations = 0
        total_checkable = 0
        
        # Check each association for consistency with history
        for assoc in associations:
            curr_desc = current_descriptors[assoc.feature_idx1]
            prev_desc = previous_descriptors[assoc.feature_idx2]
            
            # Find corresponding feature in previous associations
            previous_associations = self.association_history[-1] if self.association_history else []
            
            # Look for feature that matches previous descriptor
            found_consistent = False
            for prev_assoc in previous_associations:
                # Check if this is the same feature continuing from previous scan
                if self._features_likely_same(prev_desc, prev_assoc, len(self.association_history) - 1):
                    # Check motion consistency
                    if self._check_motion_consistency(curr_desc, prev_desc, prev_assoc):
                        found_consistent = True
                        consistent_associations += 1
                    break
            
            total_checkable += 1
        
        # Calculate consistency score
        if total_checkable == 0:
            return 1.0
        
        consistency_score = consistent_associations / total_checkable
        return consistency_score
    
    def _features_likely_same(self, descriptor: "FeatureDescriptor", 
                            prev_association: "AssociationScore", history_index: int) -> bool:
        """
        Check if a descriptor likely corresponds to a feature from previous association
        
        Args:
            descriptor: Current descriptor to check
            prev_association: Previous association
            history_index: Index in association history
            
        Returns:
            True if features are likely the same
        """
        # Simple check based on feature type and approximate position
        # In a full implementation, this would use more sophisticated tracking
        
        # Check feature type consistency
        if len(self.association_history) > history_index:
            # This is a simplified check - in practice you'd track features more carefully
            return True  # Placeholder for more sophisticated tracking
        
        return False
    
    def _check_motion_consistency(self, current_desc: "FeatureDescriptor",
                                previous_desc: "FeatureDescriptor",
                                prev_association: "AssociationScore") -> bool:
        """
        Check if the motion of a feature is consistent with expectations
        
        Args:
            current_desc: Current feature descriptor
            previous_desc: Previous feature descriptor
            prev_association: Previous association for reference
            
        Returns:
            True if motion is consistent
        """
        # Calculate motion vector
        motion = current_desc.base_feature.point_world - previous_desc.base_feature.point_world
        motion_magnitude = np.linalg.norm(motion)
        
        # Check if motion is reasonable (not too large)
        max_reasonable_motion = 2.0  # meters per scan
        if motion_magnitude > max_reasonable_motion:
            return False
        
        # Check if motion direction is consistent with overall robot motion
        # (This would require robot motion estimate - simplified for now)
        
        return True


class AssociationValidator:
    """
    Main validator for feature associations using RANSAC and consistency checks
    """
    
    def __init__(self, ransac_threshold: float = 0.2,
                 ransac_iterations: int = 100,
                 min_inliers: int = 3,
                 temporal_consistency_weight: float = 0.3,
                 geometric_consistency_weight: float = 0.7,
                 debug_level: int = 1):
        """
        Initialize the association validator
        
        Args:
            ransac_threshold: Distance threshold for RANSAC inliers (meters)
            ransac_iterations: Number of RANSAC iterations
            min_inliers: Minimum number of inliers required
            temporal_consistency_weight: Weight for temporal consistency in final score
            geometric_consistency_weight: Weight for geometric consistency in final score
            debug_level: Debug output level
        """
        self.ransac_threshold = ransac_threshold
        self.ransac_iterations = ransac_iterations
        self.min_inliers = min_inliers
        self.temporal_consistency_weight = temporal_consistency_weight
        self.geometric_consistency_weight = geometric_consistency_weight
        self.debug_level = debug_level
        
        # Initialize temporal consistency checker
        self.temporal_checker = TemporalConsistencyChecker()
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'average_inlier_ratio': 0.0,
            'average_confidence': 0.0
        }
        
        if self.debug_level > 0:
            print(f"[AssociationValidator] Initialized with RANSAC threshold={ransac_threshold}m, "
                  f"iterations={ransac_iterations}, min_inliers={min_inliers}")
    
    def validate_associations(self, associations: List["AssociationScore"],
                            current_descriptors: List["FeatureDescriptor"],
                            previous_descriptors: List["FeatureDescriptor"],
                            previous_motion_estimate: Optional["PoseEstimate"] = None) -> ValidationResult:
        """
        Validate a set of feature associations
        
        Args:
            associations: Feature associations to validate
            current_descriptors: Current scan descriptors
            previous_descriptors: Previous scan descriptors
            previous_motion_estimate: Previous motion estimate for consistency checking
            
        Returns:
            ValidationResult with validation outcome and metrics
        """
        result = ValidationResult()
        result.total_associations = len(associations)
        
        # Update validation statistics
        self.validation_stats['total_validations'] += 1
        
        if len(associations) < self.min_inliers:
            if self.debug_level > 1:
                print(f"[AssociationValidator] Insufficient associations ({len(associations)}) for validation")
            return result
        
        # Step 1: RANSAC geometric validation
        geometric_result = self._ransac_pose_estimation(
            associations, current_descriptors, previous_descriptors
        )
        
        # Step 2: Temporal consistency validation
        temporal_score = self.temporal_checker.validate_temporal_consistency(
            associations, current_descriptors, previous_descriptors
        )
        
        # Step 3: Motion consistency check
        motion_consistency = 1.0
        if previous_motion_estimate is not None and geometric_result.pose_estimate is not None:
            motion_consistency = self._check_motion_consistency(
                geometric_result.pose_estimate, previous_motion_estimate
            )
        
        # Step 4: Combine validation scores
        geometric_score = geometric_result.inlier_count / len(associations) if associations else 0.0
        
        final_confidence = (
            self.geometric_consistency_weight * geometric_score +
            self.temporal_consistency_weight * temporal_score
        )
        
        # Apply motion consistency as a multiplier
        final_confidence *= motion_consistency
        
        # Determine validation result
        result.is_valid = (
            geometric_result.inlier_count >= self.min_inliers and
            final_confidence > 0.5 and
            temporal_score > self.temporal_checker.consistency_threshold
        )
        
        # Fill result details
        result.confidence = final_confidence
        result.pose_estimate = geometric_result.pose_estimate
        result.inlier_count = geometric_result.inlier_count
        result.geometric_consistency = geometric_score
        result.temporal_consistency = temporal_score
        result.validation_method = "RANSAC + Temporal"
        result.inlier_associations = geometric_result.inlier_associations
        result.outlier_associations = geometric_result.outlier_associations
        result.ransac_iterations = geometric_result.ransac_iterations
        result.best_model_score = geometric_result.best_model_score
        
        result.error_metrics = {
            'motion_consistency': motion_consistency,
            'inlier_ratio': geometric_score,
            'temporal_score': temporal_score
        }
        
        # Update temporal history
        if result.is_valid:
            self.temporal_checker.add_associations(associations, current_descriptors, previous_descriptors)
        
        # Update statistics
        if result.is_valid:
            self.validation_stats['successful_validations'] += 1
        
        # Update running averages
        total = self.validation_stats['total_validations']
        self.validation_stats['average_inlier_ratio'] = (
            (self.validation_stats['average_inlier_ratio'] * (total - 1) + geometric_score) / total
        )
        self.validation_stats['average_confidence'] = (
            (self.validation_stats['average_confidence'] * (total - 1) + final_confidence) / total
        )
        
        if self.debug_level > 1:
            print(f"[AssociationValidator] Validation result: valid={result.is_valid}, "
                  f"confidence={final_confidence:.3f}, inliers={result.inlier_count}/{len(associations)}")
        
        return result
    
    def _ransac_pose_estimation(self, associations: List["AssociationScore"],
                              current_descriptors: List["FeatureDescriptor"],
                              previous_descriptors: List["FeatureDescriptor"]) -> ValidationResult:
        """
        Estimate pose using RANSAC on feature associations
        
        Args:
            associations: Feature associations
            current_descriptors: Current scan descriptors
            previous_descriptors: Previous scan descriptors
            
        Returns:
            Geometric validation result
        """
        result = ValidationResult()
        result.total_associations = len(associations)
        
        if len(associations) < 3:
            return result
        
        best_inlier_count = 0
        best_pose = None
        best_inliers = []
        best_score = 0.0
        
        # Extract point correspondences
        current_points = []
        previous_points = []
        
        for assoc in associations:
            curr_point = current_descriptors[assoc.feature_idx1].base_feature.point_world
            prev_point = previous_descriptors[assoc.feature_idx2].base_feature.point_world
            current_points.append(curr_point)
            previous_points.append(prev_point)
        
        current_points = np.array(current_points)
        previous_points = np.array(previous_points)
        
        # RANSAC loop
        for iteration in range(self.ransac_iterations):
            # Sample minimum set (3 points for 2D rigid transformation)
            sample_indices = random.sample(range(len(associations)), min(3, len(associations)))
            
            # Estimate pose from sample
            sample_current = current_points[sample_indices]
            sample_previous = previous_points[sample_indices]
            
            pose_estimate = self._estimate_pose_from_correspondences(
                sample_current, sample_previous
            )
            
            if pose_estimate is None:
                continue
            
            # Count inliers
            inliers, outliers, inlier_count = self._count_inliers(
                pose_estimate, current_points, previous_points, associations
            )
            
            # Score this model
            model_score = inlier_count + 0.1 * len(inliers)  # Bonus for more inliers
            
            if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and model_score > best_score):
                best_inlier_count = inlier_count
                best_pose = pose_estimate
                best_inliers = inliers
                best_score = model_score
        
        # Refine pose using all inliers if we found a good model
        if best_inlier_count >= self.min_inliers and best_inliers:
            inlier_current = current_points[best_inliers]
            inlier_previous = previous_points[best_inliers]
            
            refined_pose = self._estimate_pose_from_correspondences(inlier_current, inlier_previous)
            if refined_pose is not None:
                best_pose = refined_pose
        
        # Fill result
        result.pose_estimate = best_pose
        result.inlier_count = best_inlier_count
        result.ransac_iterations = self.ransac_iterations
        result.best_model_score = best_score
        
        if best_inliers:
            result.inlier_associations = best_inliers
            result.outlier_associations = [i for i in range(len(associations)) if i not in best_inliers]
        
        return result
    
    def _estimate_pose_from_correspondences(self, current_points: np.ndarray, 
                                          previous_points: np.ndarray) -> Optional["PoseEstimate"]:
        """
        Estimate 2D rigid transformation from point correspondences
        
        Args:
            current_points: Current points [N, 2]
            previous_points: Previous points [N, 2]
            
        Returns:
            PoseEstimate or None if estimation fails
        """
        if len(current_points) < 2 or len(previous_points) < 2:
            return None
        
        try:
            # Calculate centroids
            current_centroid = np.mean(current_points, axis=0)
            previous_centroid = np.mean(previous_points, axis=0)
            
            # Center the points
            current_centered = current_points - current_centroid
            previous_centered = previous_points - previous_centroid
            
            # Calculate rotation using SVD
            H = np.dot(previous_centered.T, current_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            
            # Ensure proper rotation matrix
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)
            
            # Calculate translation
            t = current_centroid - np.dot(previous_centroid, R.T)
            
            # Extract rotation angle
            theta = math.atan2(R[1, 0], R[0, 0])
            
            return PoseEstimate(t[0], t[1], theta)
            
        except Exception as e:
            if self.debug_level > 1:
                print(f"[AssociationValidator] Pose estimation failed: {e}")
            return None
    
    def _count_inliers(self, pose_estimate: "PoseEstimate", 
                      current_points: np.ndarray, previous_points: np.ndarray,
                      associations: List["AssociationScore"]) -> Tuple[List[int], List[int], int]:
        """
        Count inliers for a given pose estimate
        
        Args:
            pose_estimate: Pose to test
            current_points: Current scan points
            previous_points: Previous scan points
            associations: Original associations
            
        Returns:
            Tuple of (inlier_indices, outlier_indices, inlier_count)
        """
        inliers = []
        outliers = []
        
        # Transform previous points according to pose estimate
        c = math.cos(pose_estimate.theta)
        s = math.sin(pose_estimate.theta)
        R = np.array([[c, -s], [s, c]])
        
        transformed_previous = np.dot(previous_points, R.T) + np.array([pose_estimate.x, pose_estimate.y])
        
        # Calculate distances and count inliers
        for i, (curr_pt, trans_prev_pt) in enumerate(zip(current_points, transformed_previous)):
            distance = np.linalg.norm(curr_pt - trans_prev_pt)
            
            if distance < self.ransac_threshold:
                inliers.append(i)
            else:
                outliers.append(i)
        
        return inliers, outliers, len(inliers)
    
    def _check_motion_consistency(self, estimated_pose: "PoseEstimate", 
                                previous_motion: "PoseEstimate") -> float:
        """
        Check consistency between estimated motion and previous motion
        
        Args:
            estimated_pose: Current motion estimate
            previous_motion: Previous motion estimate
            
        Returns:
            Consistency score [0, 1]
        """
        # Calculate differences in motion
        dx_diff = abs(estimated_pose.x - previous_motion.x)
        dy_diff = abs(estimated_pose.y - previous_motion.y)
        dtheta_diff = abs(estimated_pose.theta - previous_motion.theta)
        
        # Normalize angle difference
        dtheta_diff = min(dtheta_diff, 2*math.pi - dtheta_diff)
        
        # Calculate consistency scores
        position_consistency = np.exp(-2.0 * (dx_diff + dy_diff))  # Exponential decay
        rotation_consistency = np.exp(-3.0 * dtheta_diff)
        
        # Combined consistency
        consistency = 0.7 * position_consistency + 0.3 * rotation_consistency
        
        return min(1.0, max(0.0, consistency))
    
    def filter_associations_by_validation(self, associations: List["AssociationScore"],
                                        validation_result: "ValidationResult") -> List["AssociationScore"]:
        """
        Filter associations to keep only validated inliers
        
        Args:
            associations: Original associations
            validation_result: Validation result with inlier information
            
        Returns:
            Filtered list of associations (inliers only)
        """
        if not validation_result.is_valid or not validation_result.inlier_associations:
            return []
        
        # Mark inlier associations as validated
        validated_associations = []
        for inlier_idx in validation_result.inlier_associations:
            if 0 <= inlier_idx < len(associations):
                association = associations[inlier_idx]
                association.validated = True
                association.confidence = min(1.0, association.confidence * validation_result.confidence)
                validated_associations.append(association)
        
        return validated_associations
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive validation statistics
        
        Returns:
            Dictionary with validation performance metrics
        """
        stats = self.validation_stats.copy()
        
        # Calculate success rate
        if stats['total_validations'] > 0:
            stats['validation_success_rate'] = stats['successful_validations'] / stats['total_validations']
        else:
            stats['validation_success_rate'] = 0.0
        
        # Add current parameters
        stats['ransac_threshold'] = self.ransac_threshold
        stats['min_inliers'] = self.min_inliers
        stats['ransac_iterations'] = self.ransac_iterations
        
        return stats
    
    def reset_statistics(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'average_inlier_ratio': 0.0,
            'average_confidence': 0.0
        }
        
        # Also reset temporal checker
        self.temporal_checker = TemporalConsistencyChecker()
        
        if self.debug_level > 0:
            print("[AssociationValidator] Statistics and history reset")


# Utility functions for integration
def validate_feature_associations(associations: List["AssociationScore"],
                                current_features: FeatureSet,
                                previous_features: FeatureSet,
                                validator: Optional[AssociationValidator] = None) -> ValidationResult:
    """
    Convenience function to validate feature associations
    
    Args:
        associations: Feature associations to validate
        current_features: Current scan features
        previous_features: Previous scan features
        validator: Existing validator (creates new if None)
        
    Returns:
        ValidationResult
    """
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Dependencies not available for association validation")
        return ValidationResult()
    
    if validator is None:
        validator = AssociationValidator()
    
    # Create descriptors (simplified - in practice you'd reuse existing descriptors)
    engine = FeatureAssociationEngine()
    current_descriptors = engine.create_descriptors(current_features, scan_index=1)
    previous_descriptors = engine.create_descriptors(previous_features, scan_index=0)
    
    # Perform validation
    return validator.validate_associations(associations, current_descriptors, previous_descriptors)


def get_robust_associations(associations: List["AssociationScore"],
                          current_features: FeatureSet,
                          previous_features: FeatureSet,
                          validator: Optional[AssociationValidator] = None) -> Tuple[List["AssociationScore"], ValidationResult]:
    """
    Get validated associations with outliers removed
    
    Args:
        associations: Raw associations
        current_features: Current scan features
        previous_features: Previous scan features
        validator: Existing validator (creates new if None)
        
    Returns:
        Tuple of (robust_associations, validation_result)
    """
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Dependencies not available for robust association filtering")
        return associations, ValidationResult()
    
    # Validate associations
    validation_result = validate_feature_associations(
        associations, current_features, previous_features, validator
    )
    
    # Filter to keep only inliers
    if validator is None:
        validator = AssociationValidator()
    
    robust_associations = validator.filter_associations_by_validation(associations, validation_result)
    
    return robust_associations, validation_result