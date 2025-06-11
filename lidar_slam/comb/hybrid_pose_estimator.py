from __future__ import annotations
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import existing components
try:
    from feature_extractor import FeatureSet, LiDARFeature, FeatureType
    from pose_estimate import PoseEstimate
    from feature_association import FeatureDescriptor, AssociationScore, FeatureAssociationEngine
    from association_validator import ValidationResult, AssociationValidator
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("Warning: Some dependencies not available. Hybrid pose estimation will have limited functionality.")
    DEPENDENCIES_AVAILABLE = False


class PoseSource(Enum):
    """Enumeration for different pose sources"""
    FEATURE_BASED = "feature_based"
    ICP_BASED = "icp_based"
    HYBRID = "hybrid"
    ODOMETRY = "odometry"
    RECOVERY = "recovery"


@dataclass
class PoseEstimateWithConfidence:
    """
    Pose estimate with confidence and source information
    """
    pose: Any = None  # PoseEstimate object
    confidence: float = 0.0
    source: PoseSource = PoseSource.ODOMETRY
    
    # Detailed confidence breakdown
    geometric_confidence: float = 0.0
    temporal_confidence: float = 0.0
    feature_confidence: float = 0.0
    convergence_confidence: float = 0.0
    
    # Supporting information
    num_features_used: int = 0
    num_inliers: int = 0
    association_quality: float = 0.0
    validation_passed: bool = False
    
    # Error estimates
    position_uncertainty: float = 0.0  # meters
    orientation_uncertainty: float = 0.0  # radians
    
    # Additional metadata
    processing_time: float = 0.0  # milliseconds
    iteration_count: int = 0
    fallback_reason: str = ""


class EnvironmentClassifier:
    """
    Classifies the current environment to guide adaptive pose estimation
    """
    
    def __init__(self):
        self.environment_history = []
        self.current_classification = "unknown"
        
    def classify_environment(self, feature_set: FeatureSet, 
                           motion_estimate: Optional[PoseEstimate] = None) -> Dict[str, float]:
        """
        Classify the current environment based on feature characteristics
        
        Args:
            feature_set: Current feature set
            motion_estimate: Current motion estimate
            
        Returns:
            Dictionary with environment classification scores
        """
        if not hasattr(feature_set, 'quality_metrics'):
            return {'unknown': 1.0}
        
        quality_metrics = feature_set.quality_metrics
        
        # Feature density analysis
        feature_density = quality_metrics.get('total_features', 0) / 50.0  # Normalize by typical count
        feature_density = min(1.0, feature_density)
        
        # Spatial distribution analysis
        spatial_quality = quality_metrics.get('spatial_distribution_quality', 0.5)
        sector_balance = quality_metrics.get('sector_balance', 0.5)
        
        # Feature type distribution
        edge_features = quality_metrics.get('sharp_edges', 0) + quality_metrics.get('less_sharp_edges', 0)
        planar_features = quality_metrics.get('planar_features', 0) + quality_metrics.get('less_planar_features', 0)
        total_features = quality_metrics.get('total_features', 1)
        
        edge_ratio = edge_features / total_features if total_features > 0 else 0
        planar_ratio = planar_features / total_features if total_features > 0 else 0
        
        # Environment classification
        classifications = {}
        
        # Structured environment (high edge ratio, good distribution)
        classifications['structured'] = min(1.0, edge_ratio * 2.0 * spatial_quality)
        
        # Open/sparse environment (low feature density, poor balance)
        classifications['sparse'] = (1.0 - feature_density) * (1.0 - sector_balance)
        
        # Corridor/tunnel (poor spatial distribution, but reasonable features)
        classifications['corridor'] = (1.0 - spatial_quality) * feature_density * 0.8
        
        # Feature-rich environment (high density, good balance)
        classifications['feature_rich'] = feature_density * spatial_quality * sector_balance
        
        # Planar environment (high planar ratio)
        classifications['planar'] = planar_ratio * (1.0 - edge_ratio)
        
        # Dynamic environment (low temporal consistency - would need motion analysis)
        if motion_estimate:
            motion_magnitude = math.sqrt(motion_estimate.x**2 + motion_estimate.y**2)
            classifications['dynamic'] = min(1.0, motion_magnitude / 2.0)  # High motion indicates dynamic
        else:
            classifications['dynamic'] = 0.0
        
        # Normalize to sum to 1
        total_score = sum(classifications.values())
        if total_score > 0:
            classifications = {k: v/total_score for k, v in classifications.items()}
        else:
            classifications = {'unknown': 1.0}
        
        # Store classification
        self.environment_history.append(classifications)
        if len(self.environment_history) > 10:  # Keep recent history
            self.environment_history.pop(0)
        
        # Determine current dominant classification
        self.current_classification = max(classifications.items(), key=lambda x: x[1])[0]
        
        return classifications


class AdaptiveWeightCalculator:
    """
    Calculates adaptive weights for combining feature-based and ICP poses
    """
    
    def __init__(self):
        self.weight_history = []
        
    def calculate_weights(self, feature_confidence: float, 
                         icp_confidence: float,
                         environment_classification: Dict[str, float],
                         validation_result: Optional[ValidationResult] = None) -> Dict[str, float]:
        """
        Calculate adaptive weights for pose combination
        
        Args:
            feature_confidence: Confidence in feature-based pose
            icp_confidence: Confidence in ICP-based pose
            environment_classification: Environment classification scores
            validation_result: Feature association validation result
            
        Returns:
            Dictionary with weights and metadata
        """
        # Base weights from confidence
        base_feature_weight = feature_confidence
        base_icp_weight = icp_confidence
        
        # Environment-based adjustments
        env_adjustments = self._get_environment_adjustments(environment_classification)
        
        # Validation-based adjustments
        validation_adjustment = 1.0
        if validation_result and validation_result.is_valid:
            validation_adjustment = 1.0 + 0.3 * validation_result.confidence  # Up to 30% boost
        elif validation_result and not validation_result.is_valid:
            validation_adjustment = 0.5  # Reduce feature weight if validation failed
        
        # Apply adjustments
        adjusted_feature_weight = base_feature_weight * env_adjustments['feature_factor'] * validation_adjustment
        adjusted_icp_weight = base_icp_weight * env_adjustments['icp_factor']
        
        # Normalize weights
        total_weight = adjusted_feature_weight + adjusted_icp_weight
        
        if total_weight > 0:
            feature_weight = adjusted_feature_weight / total_weight
            icp_weight = adjusted_icp_weight / total_weight
        else:
            # Fallback to equal weights
            feature_weight = 0.5
            icp_weight = 0.5
        
        # Safety bounds
        feature_weight = max(0.1, min(0.9, feature_weight))  # Keep between 10% and 90%
        icp_weight = 1.0 - feature_weight
        
        weights = {
            'feature_weight': feature_weight,
            'icp_weight': icp_weight,
            'base_feature_confidence': base_feature_weight,
            'base_icp_confidence': base_icp_weight,
            'environment_factor': env_adjustments,
            'validation_factor': validation_adjustment,
            'reasoning': self._get_weight_reasoning(feature_weight, environment_classification)
        }
        
        # Store for analysis
        self.weight_history.append(weights)
        if len(self.weight_history) > 20:
            self.weight_history.pop(0)
        
        return weights
    
    def _get_environment_adjustments(self, environment_classification: Dict[str, float]) -> Dict[str, float]:
        """
        Get environment-specific adjustment factors
        
        Args:
            environment_classification: Environment classification scores
            
        Returns:
            Dictionary with adjustment factors
        """
        # Default factors
        feature_factor = 1.0
        icp_factor = 1.0
        
        # Structured environments favor features
        structured_score = environment_classification.get('structured', 0.0)
        feature_factor += structured_score * 0.5  # Up to 50% boost for features
        
        # Feature-rich environments favor features
        feature_rich_score = environment_classification.get('feature_rich', 0.0)
        feature_factor += feature_rich_score * 0.4  # Up to 40% boost
        
        # Sparse environments favor ICP (more robust to few features)
        sparse_score = environment_classification.get('sparse', 0.0)
        icp_factor += sparse_score * 0.6  # Up to 60% boost for ICP
        feature_factor *= (1.0 - sparse_score * 0.3)  # Reduce feature weight
        
        # Corridor environments need careful balance
        corridor_score = environment_classification.get('corridor', 0.0)
        icp_factor += corridor_score * 0.3  # Slight ICP preference
        
        # Planar environments favor ICP (fewer distinctive features)
        planar_score = environment_classification.get('planar', 0.0)
        icp_factor += planar_score * 0.4
        feature_factor *= (1.0 - planar_score * 0.2)
        
        # Dynamic environments favor features (more robust to changing map)
        dynamic_score = environment_classification.get('dynamic', 0.0)
        feature_factor += dynamic_score * 0.3
        
        return {
            'feature_factor': feature_factor,
            'icp_factor': icp_factor,
            'structured_bonus': structured_score * 0.5,
            'sparse_penalty': sparse_score * 0.3,
            'corridor_adjustment': corridor_score * 0.3,
            'planar_penalty': planar_score * 0.2,
            'dynamic_bonus': dynamic_score * 0.3
        }
    
    def _get_weight_reasoning(self, feature_weight: float, 
                            environment_classification: Dict[str, float]) -> str:
        """
        Generate human-readable reasoning for weight decisions
        
        Args:
            feature_weight: Final feature weight
            environment_classification: Environment classification
            
        Returns:
            String explaining the weighting decision
        """
        dominant_env = max(environment_classification.items(), key=lambda x: x[1])
        env_name, env_score = dominant_env
        
        if feature_weight > 0.7:
            return f"Feature-dominant ({feature_weight:.2f}) - {env_name} environment ({env_score:.2f}) favors features"
        elif feature_weight < 0.3:
            return f"ICP-dominant ({1-feature_weight:.2f}) - {env_name} environment ({env_score:.2f}) favors dense matching"
        else:
            return f"Balanced ({feature_weight:.2f}/{1-feature_weight:.2f}) - {env_name} environment ({env_score:.2f})"


class HybridPoseEstimator:
    """
    Main class for intelligent combination of feature-based and ICP poses
    """
    
    def __init__(self, enable_adaptive_weighting: bool = True,
                 enable_environment_classification: bool = True,
                 fallback_to_odometry: bool = True,
                 debug_level: int = 1):
        """
        Initialize the hybrid pose estimator
        
        Args:
            enable_adaptive_weighting: Whether to use adaptive weight calculation
            enable_environment_classification: Whether to classify environment
            fallback_to_odometry: Whether to fall back to odometry when both estimates fail
            debug_level: Debug output level
        """
        self.enable_adaptive_weighting = enable_adaptive_weighting
        self.enable_environment_classification = enable_environment_classification
        self.fallback_to_odometry = fallback_to_odometry
        self.debug_level = debug_level
        
        # Initialize components
        self.environment_classifier = EnvironmentClassifier() if enable_environment_classification else None
        self.weight_calculator = AdaptiveWeightCalculator() if enable_adaptive_weighting else None
        
        # Statistics
        self.estimation_stats = {
            'total_estimations': 0,
            'feature_dominant': 0,
            'icp_dominant': 0,
            'balanced': 0,
            'fallback_to_odometry': 0,
            'average_confidence': 0.0,
            'environment_distribution': {}
        }
        
        # Pose history for smoothing and consistency
        self.pose_history = []
        self.confidence_history = []
        
        if self.debug_level > 0:
            print(f"[HybridPoseEstimator] Initialized with adaptive_weighting={enable_adaptive_weighting}, "
                  f"environment_classification={enable_environment_classification}")
    
    def estimate_hybrid_pose(self, feature_pose: Optional[PoseEstimateWithConfidence],
                           icp_pose: Optional[PoseEstimateWithConfidence],
                           odometry_pose: Optional[PoseEstimate],
                           current_features: Optional[FeatureSet] = None,
                           validation_result: Optional[ValidationResult] = None,
                           motion_estimate: Optional[PoseEstimate] = None) -> PoseEstimateWithConfidence:
        """
        Estimate hybrid pose by intelligently combining available pose sources
        
        Args:
            feature_pose: Feature-based pose estimate with confidence
            icp_pose: ICP-based pose estimate with confidence
            odometry_pose: Odometry pose estimate
            current_features: Current feature set for environment analysis
            validation_result: Feature association validation result
            motion_estimate: Motion estimate for environment classification
            
        Returns:
            Hybrid pose estimate with confidence
        """
        import time
        start_time = time.time()
        
        # Update statistics
        self.estimation_stats['total_estimations'] += 1
        
        # Initialize result
        result = PoseEstimateWithConfidence()
        result.source = PoseSource.HYBRID
        
        # Check what poses are available
        has_feature_pose = feature_pose is not None and feature_pose.pose is not None
        has_icp_pose = icp_pose is not None and icp_pose.pose is not None
        has_odometry_pose = odometry_pose is not None
        
        # Environment classification
        environment_classification = {'unknown': 1.0}
        if self.enable_environment_classification and current_features:
            environment_classification = self.environment_classifier.classify_environment(
                current_features, motion_estimate
            )
            
            # Update environment statistics
            dominant_env = max(environment_classification.items(), key=lambda x: x[1])[0]
            if dominant_env not in self.estimation_stats['environment_distribution']:
                self.estimation_stats['environment_distribution'][dominant_env] = 0
            self.estimation_stats['environment_distribution'][dominant_env] += 1
        
        # Case 1: Neither feature nor ICP pose available
        if not has_feature_pose and not has_icp_pose:
            if has_odometry_pose and self.fallback_to_odometry:
                result.pose = PoseEstimate(odometry_pose.x, odometry_pose.y, odometry_pose.theta)
                result.confidence = 0.3  # Low confidence for odometry-only
                result.source = PoseSource.ODOMETRY
                result.fallback_reason = "No feature or ICP poses available"
                self.estimation_stats['fallback_to_odometry'] += 1
            else:
                # Return previous pose if available
                if self.pose_history:
                    last_pose = self.pose_history[-1]
                    result.pose = PoseEstimate(last_pose.x, last_pose.y, last_pose.theta)
                    result.confidence = 0.2  # Very low confidence
                    result.source = PoseSource.RECOVERY
                    result.fallback_reason = "No poses available, using previous"
                else:
                    result.pose = PoseEstimate(0, 0, 0)
                    result.confidence = 0.1
                    result.source = PoseSource.RECOVERY
                    result.fallback_reason = "No poses or history available"
        
        # Case 2: Only feature pose available
        elif has_feature_pose and not has_icp_pose:
            result = self._copy_pose_with_confidence(feature_pose)
            result.source = PoseSource.FEATURE_BASED
            
            # Apply environment-based confidence adjustment
            if self.enable_environment_classification:
                env_factor = self._get_single_pose_environment_factor(environment_classification, 'feature')
                result.confidence *= env_factor
        
        # Case 3: Only ICP pose available
        elif not has_feature_pose and has_icp_pose:
            result = self._copy_pose_with_confidence(icp_pose)
            result.source = PoseSource.ICP_BASED
            
            # Apply environment-based confidence adjustment
            if self.enable_environment_classification:
                env_factor = self._get_single_pose_environment_factor(environment_classification, 'icp')
                result.confidence *= env_factor
        
        # Case 4: Both poses available - hybrid estimation
        else:
            result = self._combine_poses(feature_pose, icp_pose, environment_classification, validation_result)
        
        # Post-processing
        result = self._apply_temporal_smoothing(result)
        result = self._calculate_uncertainty_estimates(result, feature_pose, icp_pose)
        
        # Update statistics
        if result.source == PoseSource.HYBRID:
            if hasattr(result, 'feature_weight'):
                if result.feature_weight > 0.6:
                    self.estimation_stats['feature_dominant'] += 1
                elif result.feature_weight < 0.4:
                    self.estimation_stats['icp_dominant'] += 1
                else:
                    self.estimation_stats['balanced'] += 1
        
        # Update running average confidence
        total = self.estimation_stats['total_estimations']
        self.estimation_stats['average_confidence'] = (
            (self.estimation_stats['average_confidence'] * (total - 1) + result.confidence) / total
        )
        
        # Store in history
        if result.pose:
            self.pose_history.append(result.pose)
            self.confidence_history.append(result.confidence)
            
            # Maintain history length
            if len(self.pose_history) > 10:
                self.pose_history.pop(0)
                self.confidence_history.pop(0)
        
        # Calculate processing time
        result.processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if self.debug_level > 1:
            self._log_estimation_result(result, environment_classification, has_feature_pose, has_icp_pose)
        
        return result
    
    def _copy_pose_with_confidence(self, source_pose: PoseEstimateWithConfidence) -> PoseEstimateWithConfidence:
        """
        Create a copy of a pose with confidence
        
        Args:
            source_pose: Source pose to copy
            
        Returns:
            Copy of the pose
        """
        result = PoseEstimateWithConfidence()
        result.pose = PoseEstimate(source_pose.pose.x, source_pose.pose.y, source_pose.pose.theta)
        result.confidence = source_pose.confidence
        result.geometric_confidence = source_pose.geometric_confidence
        result.temporal_confidence = source_pose.temporal_confidence
        result.feature_confidence = source_pose.feature_confidence
        result.convergence_confidence = source_pose.convergence_confidence
        result.num_features_used = source_pose.num_features_used
        result.num_inliers = source_pose.num_inliers
        result.association_quality = source_pose.association_quality
        result.validation_passed = source_pose.validation_passed
        result.iteration_count = source_pose.iteration_count
        
        return result
    
    def _combine_poses(self, feature_pose: PoseEstimateWithConfidence,
                      icp_pose: PoseEstimateWithConfidence,
                      environment_classification: Dict[str, float],
                      validation_result: Optional[ValidationResult]) -> PoseEstimateWithConfidence:
        """
        Combine feature and ICP poses using adaptive weighting
        
        Args:
            feature_pose: Feature-based pose
            icp_pose: ICP-based pose
            environment_classification: Environment classification
            validation_result: Validation result
            
        Returns:
            Combined pose estimate
        """
        # Calculate adaptive weights
        if self.enable_adaptive_weighting and self.weight_calculator:
            weights = self.weight_calculator.calculate_weights(
                feature_pose.confidence, icp_pose.confidence,
                environment_classification, validation_result
            )
            feature_weight = weights['feature_weight']
            icp_weight = weights['icp_weight']
        else:
            # Simple confidence-based weighting
            total_confidence = feature_pose.confidence + icp_pose.confidence
            if total_confidence > 0:
                feature_weight = feature_pose.confidence / total_confidence
                icp_weight = icp_pose.confidence / total_confidence
            else:
                feature_weight = 0.5
                icp_weight = 0.5
        
        # Combine poses
        combined_x = feature_weight * feature_pose.pose.x + icp_weight * icp_pose.pose.x
        combined_y = feature_weight * feature_pose.pose.y + icp_weight * icp_pose.pose.y
        
        # Handle angle combination (circular mean)
        feature_angle = feature_pose.pose.theta
        icp_angle = icp_pose.pose.theta
        
        # Convert to unit vectors and combine
        feature_vec = np.array([math.cos(feature_angle), math.sin(feature_angle)]) * feature_weight
        icp_vec = np.array([math.cos(icp_angle), math.sin(icp_angle)]) * icp_weight
        
        combined_vec = feature_vec + icp_vec
        combined_angle = math.atan2(combined_vec[1], combined_vec[0])
        
        # Create result
        result = PoseEstimateWithConfidence()
        result.pose = PoseEstimate(combined_x, combined_y, combined_angle)
        result.source = PoseSource.HYBRID
        
        # Combine confidence metrics
        result.confidence = feature_weight * feature_pose.confidence + icp_weight * icp_pose.confidence
        result.geometric_confidence = feature_weight * feature_pose.geometric_confidence + icp_weight * icp_pose.geometric_confidence
        result.temporal_confidence = feature_weight * feature_pose.temporal_confidence + icp_weight * icp_pose.temporal_confidence
        result.feature_confidence = feature_pose.feature_confidence
        result.convergence_confidence = icp_pose.convergence_confidence
        
        # Combine other metrics
        result.num_features_used = feature_pose.num_features_used
        result.num_inliers = max(feature_pose.num_inliers, icp_pose.num_inliers)
        result.association_quality = feature_pose.association_quality
        result.validation_passed = validation_result.is_valid if validation_result else False
        result.iteration_count = icp_pose.iteration_count
        
        # Store weighting information for analysis
        result.feature_weight = feature_weight
        result.icp_weight = icp_weight
        
        return result
    
    def _get_single_pose_environment_factor(self, environment_classification: Dict[str, float], 
                                          pose_type: str) -> float:
        """
        Get environment adjustment factor for single pose type
        
        Args:
            environment_classification: Environment classification
            pose_type: 'feature' or 'icp'
            
        Returns:
            Adjustment factor for confidence
        """
        factor = 1.0
        
        if pose_type == 'feature':
            # Features work better in structured environments
            factor += environment_classification.get('structured', 0.0) * 0.3
            factor += environment_classification.get('feature_rich', 0.0) * 0.2
            factor -= environment_classification.get('sparse', 0.0) * 0.3
            factor -= environment_classification.get('planar', 0.0) * 0.2
        else:  # icp
            # ICP works better in dense environments
            factor += environment_classification.get('sparse', 0.0) * 0.2
            factor += environment_classification.get('planar', 0.0) * 0.2
            factor -= environment_classification.get('dynamic', 0.0) * 0.2
        
        return max(0.5, min(1.5, factor))  # Keep within reasonable bounds
    
    def _apply_temporal_smoothing(self, current_estimate: PoseEstimateWithConfidence) -> PoseEstimateWithConfidence:
        """
        Apply temporal smoothing to reduce jitter
        
        Args:
            current_estimate: Current pose estimate
            
        Returns:
            Smoothed pose estimate
        """
        if not self.pose_history or current_estimate.confidence > 0.8:
            return current_estimate  # No smoothing for high-confidence estimates
        
        # Calculate smoothing factor based on confidence
        smoothing_factor = 0.1 * (1.0 - current_estimate.confidence)  # More smoothing for low confidence
        
        if smoothing_factor > 0 and self.pose_history:
            last_pose = self.pose_history[-1]
            
            # Smooth position
            current_estimate.pose.x = (1 - smoothing_factor) * current_estimate.pose.x + smoothing_factor * last_pose.x
            current_estimate.pose.y = (1 - smoothing_factor) * current_estimate.pose.y + smoothing_factor * last_pose.y
            
            # Smooth orientation (handle wraparound)
            angle_diff = current_estimate.pose.theta - last_pose.theta
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # Normalize
            smoothed_angle_diff = (1 - smoothing_factor) * angle_diff
            current_estimate.pose.theta = last_pose.theta + smoothed_angle_diff
        
        return current_estimate
    
    def _calculate_uncertainty_estimates(self, result: PoseEstimateWithConfidence,
                                       feature_pose: Optional[PoseEstimateWithConfidence],
                                       icp_pose: Optional[PoseEstimateWithConfidence]) -> PoseEstimateWithConfidence:
        """
        Calculate uncertainty estimates for the hybrid pose
        
        Args:
            result: Current result
            feature_pose: Feature pose (if available)
            icp_pose: ICP pose (if available)
            
        Returns:
            Result with uncertainty estimates
        """
        # Base uncertainty on confidence
        base_position_uncertainty = 0.5 * (1.0 - result.confidence)  # meters
        base_orientation_uncertainty = 0.2 * (1.0 - result.confidence)  # radians
        
        # If we have both poses, estimate uncertainty from their disagreement
        if feature_pose and icp_pose and feature_pose.pose and icp_pose.pose:
            position_disagreement = math.sqrt(
                (feature_pose.pose.x - icp_pose.pose.x)**2 +
                (feature_pose.pose.y - icp_pose.pose.y)**2
            )
            
            angle_disagreement = abs(feature_pose.pose.theta - icp_pose.pose.theta)
            angle_disagreement = min(angle_disagreement, 2*math.pi - angle_disagreement)
            
            # Uncertainty increases with disagreement
            result.position_uncertainty = max(base_position_uncertainty, position_disagreement * 0.5)
            result.orientation_uncertainty = max(base_orientation_uncertainty, angle_disagreement * 0.5)
        else:
            result.position_uncertainty = base_position_uncertainty
            result.orientation_uncertainty = base_orientation_uncertainty
        
        return result
    
    def _log_estimation_result(self, result: PoseEstimateWithConfidence,
                             environment_classification: Dict[str, float],
                             has_feature_pose: bool, has_icp_pose: bool):
        """
        Log detailed information about the estimation result
        
        Args:
            result: Estimation result
            environment_classification: Environment classification
            has_feature_pose: Whether feature pose was available
            has_icp_pose: Whether ICP pose was available
        """
        dominant_env = max(environment_classification.items(), key=lambda x: x[1])
        
        print(f"[HybridPoseEstimator] Estimation result:")
        print(f"  Source: {result.source.value}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Pose: x={result.pose.x:.3f}, y={result.pose.y:.3f}, θ={result.pose.theta:.3f}")
        print(f"  Environment: {dominant_env[0]} ({dominant_env[1]:.3f})")
        print(f"  Available: Feature={has_feature_pose}, ICP={has_icp_pose}")
        
        if hasattr(result, 'feature_weight'):
            print(f"  Weights: Feature={result.feature_weight:.3f}, ICP={result.icp_weight:.3f}")
        
        if result.fallback_reason:
            print(f"  Fallback reason: {result.fallback_reason}")
    
    def get_estimation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive estimation statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.estimation_stats.copy()
        
        # Calculate percentages
        total = stats['total_estimations']
        if total > 0:
            stats['feature_dominant_pct'] = stats['feature_dominant'] / total * 100
            stats['icp_dominant_pct'] = stats['icp_dominant'] / total * 100
            stats['balanced_pct'] = stats['balanced'] / total * 100
            stats['fallback_pct'] = stats['fallback_to_odometry'] / total * 100
        
        # Add current configuration
        stats['adaptive_weighting_enabled'] = self.enable_adaptive_weighting
        stats['environment_classification_enabled'] = self.enable_environment_classification
        
        # Add weight history statistics if available
        if self.weight_calculator and self.weight_calculator.weight_history:
            weights = [w['feature_weight'] for w in self.weight_calculator.weight_history]
            stats['average_feature_weight'] = np.mean(weights)
            stats['feature_weight_std'] = np.std(weights)
        
        return stats
    
    def reset_statistics(self):
        """Reset estimation statistics and history"""
        self.estimation_stats = {
            'total_estimations': 0,
            'feature_dominant': 0,
            'icp_dominant': 0,
            'balanced': 0,
            'fallback_to_odometry': 0,
            'average_confidence': 0.0,
            'environment_distribution': {}
        }
        
        self.pose_history = []
        self.confidence_history = []
        
        if self.weight_calculator:
            self.weight_calculator.weight_history = []
        
        if self.environment_classifier:
            self.environment_classifier.environment_history = []
        
        if self.debug_level > 0:
            print("[HybridPoseEstimator] Statistics and history reset")


# Utility functions for integration
def create_pose_with_confidence(pose: PoseEstimate, confidence: float, 
                               source: PoseSource = PoseSource.ODOMETRY) -> PoseEstimateWithConfidence:
    """
    Convenience function to create a PoseEstimateWithConfidence
    
    Args:
        pose: Base pose estimate
        confidence: Confidence level
        source: Pose source
        
    Returns:
        PoseEstimateWithConfidence object
    """
    result = PoseEstimateWithConfidence()
    result.pose = PoseEstimate(pose.x, pose.y, pose.theta) if pose else None
    result.confidence = confidence
    result.source = source
    return result


def estimate_pose_from_associations(associations: List["AssociationScore"],
                                  current_descriptors: List["FeatureDescriptor"],
                                  previous_descriptors: List["FeatureDescriptor"],
                                  validation_result: Optional[ValidationResult] = None) -> PoseEstimateWithConfidence:
    """
    Estimate pose from feature associations
    
    Args:
        associations: Feature associations
        current_descriptors: Current scan descriptors
        previous_descriptors: Previous scan descriptors
        validation_result: Validation result
        
    Returns:
        Feature-based pose estimate with confidence
    """
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Dependencies not available for pose estimation from associations")
        return PoseEstimateWithConfidence()
    
    # Use association engine to estimate motion
    engine = FeatureAssociationEngine()
    motion_estimate = engine.estimate_motion_from_associations(
        associations, current_descriptors, previous_descriptors
    )
    
    if motion_estimate is None:
        return PoseEstimateWithConfidence()
    
    # Calculate confidence based on validation and association quality
    confidence = 0.5  # Base confidence
    
    if validation_result and validation_result.is_valid:
        confidence += 0.3 * validation_result.confidence
    
    if associations:
        avg_association_score = np.mean([a.score for a in associations])
        confidence += 0.2 * avg_association_score
    
    confidence = min(1.0, confidence)
    
    # Create result
    result = create_pose_with_confidence(motion_estimate, confidence, PoseSource.FEATURE_BASED)
    result.num_features_used = len(associations)
    result.association_quality = np.mean([a.score for a in associations]) if associations else 0.0
    result.validation_passed = validation_result.is_valid if validation_result else False
    
    if validation_result:
        result.num_inliers = validation_result.inlier_count
        result.geometric_confidence = validation_result.geometric_consistency
        result.temporal_confidence = validation_result.temporal_consistency
    
    return result