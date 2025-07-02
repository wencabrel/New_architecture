import numpy as np
import math
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import copy
from pose_estimate import PoseEstimate

# Import existing components (assuming they're available)
try:
    from feature_extractor import FeatureSet, LiDARFeature, FeatureType
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("Warning: Some dependencies not available. Feature association will have limited functionality.")
    DEPENDENCIES_AVAILABLE = False


@dataclass
class FeatureDescriptor:
    """
    Enhanced feature representation for robust association
    Extends the existing LiDARFeature with additional descriptors
    """
    base_feature: Any = None  # LiDARFeature object
    
    # Geometric Context
    local_density: float = 0.0  # Number of neighboring points within radius
    surface_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    geometric_context: str = "unknown"  # "edge", "corner", "plane", "isolated"
    
    # Spatial Descriptor
    scan_angle: float = 0.0  # Angle within the scan (relative to robot front)
    distance_from_center: float = 0.0  # Distance from scan center
    sector_position: float = 0.0  # Position within sector (0-1)
    
    # Temporal Information
    scan_index: int = -1  # Which scan this feature belongs to
    predicted_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    prediction_confidence: float = 0.0
    
    # Association History
    match_history: List[int] = field(default_factory=list)  # Indices of previous matches
    association_quality_history: List[float] = field(default_factory=list)
    consecutive_matches: int = 0
    last_match_confidence: float = 0.0
    
    # Descriptor Vector (for fast similarity comparison)
    descriptor_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        """Compute derived descriptors after initialization"""
        if self.base_feature is not None:
            self._compute_descriptors()
    
    def _compute_descriptors(self):
        """Compute the descriptor vector from feature properties"""
        if self.base_feature is None:
            return
        
        # Create a descriptor vector combining multiple attributes
        # This will be used for fast similarity computation
        descriptor_components = [
            self.base_feature.curvature,
            self.base_feature.strength,
            self.base_feature.distance,
            self.scan_angle,
            self.distance_from_center,
            self.local_density,
            self.sector_position
        ]
        
        # Add surface normal components if available
        if len(self.surface_normal) == 2:
            descriptor_components.extend(self.surface_normal.tolist())
        
        self.descriptor_vector = np.array(descriptor_components)
    
    def compute_similarity(self, other: 'FeatureDescriptor') -> float:
        """
        Compute similarity score between two feature descriptors
        
        Args:
            other: Another FeatureDescriptor to compare with
            
        Returns:
            Similarity score [0, 1] where 1 is most similar
        """
        if self.base_feature is None or other.base_feature is None:
            return 0.0
        
        # Feature type compatibility (same type gets bonus)
        type_similarity = 1.0 if self.base_feature.feature_type == other.base_feature.feature_type else 0.5
        
        # Geometric similarity (curvature and strength)
        curvature_diff = abs(self.base_feature.curvature - other.base_feature.curvature)
        strength_diff = abs(self.base_feature.strength - other.base_feature.strength)
        geometric_similarity = np.exp(-5.0 * (curvature_diff + strength_diff))
        
        # Spatial similarity (relative positions should be similar)
        angle_diff = abs(self.scan_angle - other.scan_angle)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Handle wraparound
        spatial_similarity = np.exp(-2.0 * angle_diff)
        
        # Distance consistency (features at similar distances are more likely to match)
        distance_ratio = min(self.base_feature.distance, other.base_feature.distance) / \
                        max(self.base_feature.distance, other.base_feature.distance)
        distance_similarity = distance_ratio
        
        # Context similarity
        context_similarity = 1.0 if self.geometric_context == other.geometric_context else 0.7
        
        # Combine similarities with weights
        total_similarity = (
            0.3 * type_similarity +
            0.25 * geometric_similarity +
            0.2 * spatial_similarity +
            0.15 * distance_similarity +
            0.1 * context_similarity
        )
        
        return min(1.0, max(0.0, total_similarity))
    
    def predict_next_position(self, motion_estimate: "PoseEstimate") -> np.ndarray:
        """
        Predict where this feature should appear in the next scan
        
        Args:
            motion_estimate: Estimated motion between scans
            
        Returns:
            Predicted position in world coordinates
        """
        if self.base_feature is None:
            return np.array([0.0, 0.0])
        
        # Current position in world frame
        current_pos = self.base_feature.point_world
        
        # Apply inverse motion to predict where the feature will be
        # (since the robot moves, the feature appears to move in opposite direction)
        dx = -motion_estimate.x
        dy = -motion_estimate.y
        dtheta = -motion_estimate.theta
        
        # Rotate by negative rotation
        c = math.cos(dtheta)
        s = math.sin(dtheta)
        
        rotated_x = current_pos[0] * c - current_pos[1] * s
        rotated_y = current_pos[0] * s + current_pos[1] * c
        
        predicted_pos = np.array([rotated_x + dx, rotated_y + dy])
        
        # Update internal prediction
        self.predicted_position = predicted_pos
        self.prediction_confidence = min(1.0, self.consecutive_matches / 5.0)  # Higher confidence with more matches
        
        return predicted_pos


class AssociationScore:
    """Container for association scoring information"""
    
    def __init__(self, feature_idx1: int, feature_idx2: int, score: float,
                 distance: float, similarity: float, confidence: float):
        self.feature_idx1 = feature_idx1  # Index in first feature set
        self.feature_idx2 = feature_idx2  # Index in second feature set
        self.score = score  # Overall association score
        self.distance = distance  # Spatial distance
        self.similarity = similarity  # Descriptor similarity
        self.confidence = confidence  # Association confidence
        self.validated = False  # Whether this association passed validation


class FeatureAssociationEngine:
    """
    Enhanced Global Nearest Neighbor feature association with adaptive gating
    """
    
    def __init__(self, max_association_distance: float = 2.0,
                 min_similarity_threshold: float = 0.3,
                 enable_temporal_prediction: bool = True,
                 enable_adaptive_gating: bool = True,
                 debug_level: int = 1):
        """
        Initialize the feature association engine
        
        Args:
            max_association_distance: Maximum distance for feature association (meters)
            min_similarity_threshold: Minimum similarity score for association
            enable_temporal_prediction: Whether to use motion prediction
            enable_adaptive_gating: Whether to adapt thresholds based on quality
            debug_level: Debug output level (0=none, 1=basic, 2=detailed)
        """
        self.max_association_distance = max_association_distance
        self.min_similarity_threshold = min_similarity_threshold
        self.enable_temporal_prediction = enable_temporal_prediction
        self.enable_adaptive_gating = enable_adaptive_gating
        self.debug_level = debug_level
        
        # Adaptive parameters (will be adjusted based on feature quality)
        self.current_distance_threshold = max_association_distance
        self.current_similarity_threshold = min_similarity_threshold
        
        # Feature type hierarchy (order of matching priority)
        self.feature_type_hierarchy = [
            FeatureType.SHARP_EDGE,
            FeatureType.LESS_SHARP_EDGE,
            FeatureType.PLANAR,
            FeatureType.LESS_PLANAR
        ]
        
        # Association history for analysis
        self.association_history = []
        self.association_stats = {
            'total_attempts': 0,
            'successful_associations': 0,
            'average_score': 0.0,
            'feature_type_success': {ft: 0 for ft in FeatureType}
        }
        
        # Previous scan data for temporal consistency
        self.previous_descriptors = []
        self.previous_feature_set = None
        self.motion_estimate = PoseEstimate(0, 0, 0)
        
        if self.debug_level > 0:
            print(f"[FeatureAssociation] Initialized with max_distance={max_association_distance}m, "
                  f"min_similarity={min_similarity_threshold}")
    
    def create_descriptors(self, feature_set: FeatureSet, scan_index: int = 0) -> List["FeatureDescriptor"]:
        """
        Create enhanced descriptors from a feature set
        
        Args:
            feature_set: FeatureSet object from feature extraction
            scan_index: Index of this scan in the sequence
            
        Returns:
            List of FeatureDescriptor objects
        """
        descriptors = []
        
        for i, feature in enumerate(feature_set.features):
            descriptor = FeatureDescriptor(base_feature=feature)
            descriptor.scan_index = scan_index
            
            # Compute spatial descriptors
            descriptor.scan_angle = feature.angle
            descriptor.distance_from_center = feature.distance
            descriptor.sector_position = (feature.scan_index % 30) / 30.0  # Normalize within sector
            
            # Compute local density (simplified - count nearby features)
            nearby_count = 0
            for other_feature in feature_set.features:
                if other_feature is not feature:
                    dist = np.linalg.norm(feature.point_world - other_feature.point_world)
                    if dist < 0.5:  # Within 0.5 meters
                        nearby_count += 1
            descriptor.local_density = nearby_count
            
            # Estimate geometric context based on curvature and neighbors
            if feature.curvature > 0.15:
                descriptor.geometric_context = "edge" if nearby_count > 2 else "isolated"
            elif feature.curvature < 0.05:
                descriptor.geometric_context = "plane"
            else:
                descriptor.geometric_context = "corner"
            
            # Compute surface normal (simplified - from local neighborhood)
            descriptor.surface_normal = self._estimate_surface_normal(feature, feature_set.features)
            
            # Temporal prediction if we have motion estimate
            if self.enable_temporal_prediction and scan_index > 0:
                descriptor.predict_next_position(self.motion_estimate)
            
            # Finalize descriptor computation
            descriptor._compute_descriptors()
            
            descriptors.append(descriptor)
        
        return descriptors
    
    def _estimate_surface_normal(self, target_feature: Any, all_features: List[Any]) -> np.ndarray:
        """
        Estimate surface normal for a feature based on local neighborhood
        
        Args:
            target_feature: The feature to estimate normal for
            all_features: All features in the scan
            
        Returns:
            Estimated surface normal as 2D vector
        """
        # Find nearby features
        nearby_features = []
        for feature in all_features:
            if feature is not target_feature:
                dist = np.linalg.norm(target_feature.point_world - feature.point_world)
                if dist < 1.0:  # Within 1 meter
                    nearby_features.append(feature)
        
        if len(nearby_features) < 2:
            # Not enough neighbors, use angle-based estimate
            angle = target_feature.angle + math.pi/2  # Perpendicular to radial direction
            return np.array([math.cos(angle), math.sin(angle)])
        
        # Fit line to nearby points and compute normal
        points = np.array([f.point_world for f in nearby_features[:5]])  # Use up to 5 nearest
        if len(points) >= 2:
            # Simple line fitting
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            
            # PCA to find principal direction
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # Normal is perpendicular to principal direction
            principal_direction = eigenvectors[:, np.argmax(eigenvalues)]
            normal = np.array([-principal_direction[1], principal_direction[0]])
            
            return normal / np.linalg.norm(normal)
        
        # Fallback
        angle = target_feature.angle + math.pi/2
        return np.array([math.cos(angle), math.sin(angle)])
    
    def adapt_parameters(self, feature_set: FeatureSet):
        """
        Adapt association parameters based on feature quality metrics
        
        Args:
            feature_set: Current feature set with quality metrics
        """
        if not self.enable_adaptive_gating or not hasattr(feature_set, 'quality_metrics'):
            return
        
        quality_metrics = feature_set.quality_metrics
        
        # Get overall quality score
        overall_quality = quality_metrics.get('overall_quality', 0.5)
        spatial_quality = quality_metrics.get('spatial_distribution_quality', 0.5)
        feature_count = quality_metrics.get('total_features', 0)
        
        # Adapt distance threshold based on spatial distribution
        distance_factor = 1.0 + (1.0 - spatial_quality) * 0.5  # Increase threshold if poor distribution
        self.current_distance_threshold = min(
            self.max_association_distance * distance_factor,
            self.max_association_distance * 2.0  # Cap at 2x original
        )
        
        # Adapt similarity threshold based on overall quality
        similarity_factor = 0.8 + overall_quality * 0.4  # Range [0.8, 1.2]
        self.current_similarity_threshold = max(
            self.min_similarity_threshold * similarity_factor,
            0.1  # Minimum threshold
        )
        
        # Adapt based on feature density
        if feature_count < 10:
            # Few features - be more permissive
            self.current_distance_threshold *= 1.2
            self.current_similarity_threshold *= 0.9
        elif feature_count > 50:
            # Many features - be more selective
            self.current_distance_threshold *= 0.8
            self.current_similarity_threshold *= 1.1
        
        if self.debug_level > 1:
            print(f"[FeatureAssociation] Adapted thresholds: distance={self.current_distance_threshold:.2f}m, "
                  f"similarity={self.current_similarity_threshold:.3f}")
    
    def associate_features(self, current_descriptors: List["FeatureDescriptor"],
                          previous_descriptors: List["FeatureDescriptor"],
                          motion_estimate: Optional["PoseEstimate"] = None) -> List["AssociationScore"]:
        """
        Associate features between current and previous scans using Enhanced GNN
        
        Args:
            current_descriptors: Current scan feature descriptors
            previous_descriptors: Previous scan feature descriptors
            motion_estimate: Estimated motion between scans
            
        Returns:
            List of valid associations
        """
        if not current_descriptors or not previous_descriptors:
            return []
        
        # Update motion estimate for temporal prediction
        if motion_estimate:
            self.motion_estimate = motion_estimate
        
        # Update association attempt statistics
        self.association_stats['total_attempts'] += 1
        
        # Hierarchical association by feature type
        all_associations = []
        used_current = set()
        used_previous = set()
        
        # Process each feature type in priority order
        for feature_type in self.feature_type_hierarchy:
            # Get features of this type
            current_type_features = [
                (i, desc) for i, desc in enumerate(current_descriptors)
                if desc.base_feature.feature_type == feature_type and i not in used_current
            ]
            
            previous_type_features = [
                (i, desc) for i, desc in enumerate(previous_descriptors)
                if desc.base_feature.feature_type == feature_type and i not in used_previous
            ]
            
            if not current_type_features or not previous_type_features:
                continue
            
            # Perform GNN association for this feature type
            type_associations = self._gnn_associate_type(
                current_type_features, previous_type_features
            )
            
            # Add valid associations and mark features as used
            for assoc in type_associations:
                if assoc.score >= self.current_similarity_threshold:
                    all_associations.append(assoc)
                    used_current.add(assoc.feature_idx1)
                    used_previous.add(assoc.feature_idx2)
                    
                    # Update statistics
                    self.association_stats['feature_type_success'][feature_type] += 1
        
        # Update overall statistics
        if all_associations:
            avg_score = np.mean([a.score for a in all_associations])
            self.association_stats['successful_associations'] += len(all_associations)
            self.association_stats['average_score'] = (
                (self.association_stats['average_score'] * (self.association_stats['total_attempts'] - 1) + avg_score) /
                self.association_stats['total_attempts']
            )
        
        # Store for next iteration
        self.previous_descriptors = current_descriptors.copy()
        
        if self.debug_level > 1:
            print(f"[FeatureAssociation] Found {len(all_associations)} associations from "
                  f"{len(current_descriptors)} current and {len(previous_descriptors)} previous features")
        
        return all_associations
    
    def _gnn_associate_type(self, current_features: List[Tuple[int, FeatureDescriptor]],
                           previous_features: List[Tuple[int, FeatureDescriptor]]) -> List["AssociationScore"]:
        """
        Global Nearest Neighbor association for a specific feature type
        
        Args:
            current_features: List of (index, descriptor) for current scan
            previous_features: List of (index, descriptor) for previous scan
            
        Returns:
            List of associations for this feature type
        """
        associations = []
        
        # Compute all pairwise scores
        score_matrix = np.zeros((len(current_features), len(previous_features)))
        distance_matrix = np.zeros((len(current_features), len(previous_features)))
        
        for i, (curr_idx, curr_desc) in enumerate(current_features):
            for j, (prev_idx, prev_desc) in enumerate(previous_features):
                # Spatial distance
                spatial_dist = np.linalg.norm(
                    curr_desc.base_feature.point_world - prev_desc.base_feature.point_world
                )
                
                # Skip if too far apart
                if spatial_dist > self.current_distance_threshold:
                    score_matrix[i, j] = 0.0
                    distance_matrix[i, j] = float('inf')
                    continue
                
                # Descriptor similarity
                similarity = curr_desc.compute_similarity(prev_desc)
                
                # Temporal consistency bonus if prediction available
                temporal_bonus = 0.0
                if self.enable_temporal_prediction and len(prev_desc.predicted_position) == 2:
                    prediction_error = np.linalg.norm(
                        curr_desc.base_feature.point_world - prev_desc.predicted_position
                    )
                    temporal_bonus = np.exp(-prediction_error) * 0.2  # Up to 20% bonus
                
                # Combined score
                score = similarity + temporal_bonus
                
                # Distance penalty (prefer closer matches when scores are similar)
                distance_penalty = np.exp(-spatial_dist / self.current_distance_threshold) * 0.1
                score += distance_penalty
                
                score_matrix[i, j] = score
                distance_matrix[i, j] = spatial_dist
        
        # Global Nearest Neighbor assignment
        used_current = set()
        used_previous = set()
        
        # Find best associations greedily
        while True:
            # Find the best unassigned association
            best_score = 0.0
            best_i, best_j = -1, -1
            
            for i in range(len(current_features)):
                if i in used_current:
                    continue
                for j in range(len(previous_features)):
                    if j in used_previous:
                        continue
                    
                    if score_matrix[i, j] > best_score:
                        best_score = score_matrix[i, j]
                        best_i, best_j = i, j
            
            # Stop if no good association found
            if best_score < self.current_similarity_threshold:
                break
            
            # Create association
            curr_idx, curr_desc = current_features[best_i]
            prev_idx, prev_desc = previous_features[best_j]
            
            association = AssociationScore(
                feature_idx1=curr_idx,
                feature_idx2=prev_idx,
                score=best_score,
                distance=distance_matrix[best_i, best_j],
                similarity=curr_desc.compute_similarity(prev_desc),
                confidence=min(1.0, best_score)
            )
            
            associations.append(association)
            
            # Mark as used
            used_current.add(best_i)
            used_previous.add(best_j)
        
        return associations
    
    def estimate_motion_from_associations(self, associations: List["AssociationScore"],
                                        current_descriptors: List["FeatureDescriptor"],
                                        previous_descriptors: List["FeatureDescriptor"]) -> Optional["PoseEstimate"]:
        """
        Estimate motion between scans using feature associations
        
        Args:
            associations: Valid feature associations
            current_descriptors: Current scan descriptors
            previous_descriptors: Previous scan descriptors
            
        Returns:
            Estimated motion or None if insufficient associations
        """
        if len(associations) < 3:
            if self.debug_level > 1:
                print(f"[FeatureAssociation] Insufficient associations ({len(associations)}) for motion estimation")
            return None
        
        # Extract corresponding points
        current_points = []
        previous_points = []
        
        for assoc in associations:
            curr_point = current_descriptors[assoc.feature_idx1].base_feature.point_world
            prev_point = previous_descriptors[assoc.feature_idx2].base_feature.point_world
            
            current_points.append(curr_point)
            previous_points.append(prev_point)
        
        current_points = np.array(current_points)
        previous_points = np.array(previous_points)
        
        # Simple motion estimation using centroids (can be improved with more sophisticated methods)
        current_centroid = np.mean(current_points, axis=0)
        previous_centroid = np.mean(previous_points, axis=0)
        
        # Translation estimate
        translation = current_centroid - previous_centroid
        
        # Rotation estimate (simplified - using average rotation of point pairs)
        rotations = []
        for curr_pt, prev_pt in zip(current_points, previous_points):
            curr_angle = math.atan2(curr_pt[1] - current_centroid[1], curr_pt[0] - current_centroid[0])
            prev_angle = math.atan2(prev_pt[1] - previous_centroid[1], prev_pt[0] - previous_centroid[0])
            
            rotation = curr_angle - prev_angle
            # Normalize to [-π, π]
            rotation = (rotation + math.pi) % (2 * math.pi) - math.pi
            rotations.append(rotation)
        
        # Use median rotation to reduce outlier influence
        median_rotation = np.median(rotations)
        
        motion_estimate = PoseEstimate(translation[0], translation[1], median_rotation)
        
        if self.debug_level > 1:
            print(f"[FeatureAssociation] Estimated motion: dx={translation[0]:.3f}, "
                  f"dy={translation[1]:.3f}, dtheta={median_rotation:.3f}")
        
        return motion_estimate
    
    def get_association_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive association statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.association_stats.copy()
        
        # Calculate success rate
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_associations'] / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0
        
        # Add current parameter values
        stats['current_distance_threshold'] = self.current_distance_threshold
        stats['current_similarity_threshold'] = self.current_similarity_threshold
        
        # Add feature type success rates
        stats['feature_type_success_rates'] = {}
        total_success = sum(stats['feature_type_success'].values())
        if total_success > 0:
            for ft, count in stats['feature_type_success'].items():
                stats['feature_type_success_rates'][ft.value] = count / total_success
        
        return stats
    
    def reset_statistics(self):
        """Reset association statistics"""
        self.association_stats = {
            'total_attempts': 0,
            'successful_associations': 0,
            'average_score': 0.0,
            'feature_type_success': {ft: 0 for ft in FeatureType}
        }
        
        if self.debug_level > 0:
            print("[FeatureAssociation] Statistics reset")


# Utility functions for integration
def create_descriptors_from_feature_set(feature_set: FeatureSet, scan_index: int = 0) -> List["FeatureDescriptor"]:
    """
    Convenience function to create descriptors from a FeatureSet
    
    Args:
        feature_set: FeatureSet object from feature extraction
        scan_index: Index of this scan in the sequence
        
    Returns:
        List of FeatureDescriptor objects
    """
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Dependencies not available for descriptor creation")
        return []
    
    engine = FeatureAssociationEngine()
    return engine.create_descriptors(feature_set, scan_index)


def associate_consecutive_scans(current_features: FeatureSet, 
                              previous_features: FeatureSet,
                              motion_estimate: Optional["PoseEstimate"] = None,
                              engine: Optional[FeatureAssociationEngine] = None) -> Tuple[List["AssociationScore"], Optional["PoseEstimate"]]:
    """
    Convenience function to associate features between consecutive scans
    
    Args:
        current_features: Current scan features
        previous_features: Previous scan features
        motion_estimate: Estimated motion between scans
        engine: Existing association engine (creates new if None)
        
    Returns:
        Tuple of (associations, estimated_motion)
    """
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Dependencies not available for feature association")
        return [], None
    
    if engine is None:
        engine = FeatureAssociationEngine()
    
    # Create descriptors
    current_descriptors = engine.create_descriptors(current_features, scan_index=1)
    previous_descriptors = engine.create_descriptors(previous_features, scan_index=0)
    
    # Adapt parameters based on current feature quality
    engine.adapt_parameters(current_features)
    
    # Perform association
    associations = engine.associate_features(current_descriptors, previous_descriptors, motion_estimate)
    
    # Estimate motion from associations
    estimated_motion = engine.estimate_motion_from_associations(associations, current_descriptors, previous_descriptors)
    
    return associations, estimated_motion