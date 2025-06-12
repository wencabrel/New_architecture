#!/usr/bin/env python3
"""
Comprehensive testing suite for feature association system
Tests all components with real data and extensive visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import sys
import argparse
from typing import List, Dict, Tuple, Optional, Any

# Import existing components
try:
    from lidar_utility_functions import parse_lidar_data, convert_scans_to_cartesian, read_lidar_data_from_file
    from feature_extractor import FeatureExtractor, FeatureSet, FeatureType
    from ScanMatcher import PoseEstimate
    
    # Import our new association system
    from feature_association import (
        FeatureDescriptor, AssociationScore, FeatureAssociationEngine,
        create_descriptors_from_feature_set, associate_consecutive_scans
    )
    from association_validator import (
        ValidationResult, AssociationValidator, TemporalConsistencyChecker,
        validate_feature_associations, get_robust_associations
    )
    from hybrid_pose_estimator import (
        PoseEstimateWithConfidence, HybridPoseEstimator, EnvironmentClassifier,
        create_pose_with_confidence, estimate_pose_from_associations
    )
    from association_visualizer import (
        AssociationVisualizer, VisualizationConfig, create_debug_visualizer,
        quick_association_plot
    )
    
    DEPENDENCIES_AVAILABLE = True
    
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Make sure all modules are in the same directory:")
    print("- lidar_utility_functions.py")
    print("- feature_extractor.py") 
    print("- ScanMatcher.py")
    print("- feature_association.py")
    print("- association_validator.py")
    print("- hybrid_pose_estimator.py")
    print("- association_visualizer.py")
    DEPENDENCIES_AVAILABLE = False
    sys.exit(1)


class FeatureAssociationTestSuite:
    """
    Comprehensive test suite for feature association system
    """
    
    def __init__(self, output_dir: str = "association_test_results", 
                 save_plots: bool = True, debug_level: int = 2):
        """
        Initialize the test suite
        
        Args:
            output_dir: Directory to save test results
            save_plots: Whether to save plots
            debug_level: Debug output level
        """
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.debug_level = debug_level
        
        # Create output directory
        if self.save_plots and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"[TestSuite] Created output directory: {self.output_dir}")
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(debug_level=max(0, debug_level-1))
        self.association_engine = FeatureAssociationEngine(debug_level=debug_level)
        self.validator = AssociationValidator(debug_level=debug_level)
        self.hybrid_estimator = HybridPoseEstimator(debug_level=debug_level)
        self.visualizer = create_debug_visualizer(save_plots, self.output_dir)
        
        # Test data storage
        self.test_results = {
            'feature_extraction': [],
            'associations': [],
            'validations': [],
            'pose_estimates': [],
            'performance_metrics': {},
            'environment_classifications': []
        }
        
        print(f"[TestSuite] Initialized with debug_level={debug_level}")
        print(f"[TestSuite] Output directory: {self.output_dir}")
    
    def load_test_data(self, file_path: str, max_entries: int = 100) -> List[Dict]:
        """
        Load LiDAR data for testing
        
        Args:
            file_path: Path to LiDAR data file
            max_entries: Maximum number of scans to load
            
        Returns:
            List of parsed LiDAR data
        """
        print(f"[TestSuite] Loading test data from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"[TestSuite] ERROR: File {file_path} does not exist")
            return []
        
        # Read LiDAR data
        lidar_data = read_lidar_data_from_file(file_path, max_entries)
        
        if not lidar_data:
            print(f"[TestSuite] ERROR: No data loaded from {file_path}")
            return []
        
        print(f"[TestSuite] Successfully loaded {len(lidar_data)} scans")
        print(f"[TestSuite] Duration: {lidar_data[-1]['timestamp'] - lidar_data[0]['timestamp']:.2f} seconds")
        
        return lidar_data
    
    def extract_features_from_data(self, lidar_data: List[Dict], 
                                 angle_min: float = -math.pi/2, 
                                 angle_max: float = math.pi/2) -> List[FeatureSet]:
        """
        Extract features from all scans
        
        Args:
            lidar_data: LiDAR scan data
            angle_min: Minimum scan angle
            angle_max: Maximum scan angle
            
        Returns:
            List of FeatureSet objects
        """
        print(f"[TestSuite] Extracting features from {len(lidar_data)} scans...")
        
        feature_sets = []
        
        for i, scan_data in enumerate(lidar_data):
            try:
                # Convert scan to Cartesian coordinates
                scan_x, scan_y = convert_scans_to_cartesian(
                    scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
                    flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
                )
                
                # Create pose estimate
                pose = PoseEstimate(
                    scan_data['pose']['x'],
                    scan_data['pose']['y'], 
                    scan_data['pose']['theta']
                )
                
                # Extract features
                feature_set = self.feature_extractor.extract_features(
                    scan_x, scan_y, scan_data['scan_ranges'],
                    robot_pose=pose, scan_timestamp=scan_data['timestamp']
                )
                
                feature_sets.append(feature_set)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"[TestSuite] Extracted features from {i+1}/{len(lidar_data)} scans")
                
            except Exception as e:
                print(f"[TestSuite] WARNING: Error extracting features from scan {i}: {e}")
                continue
        
        print(f"[TestSuite] Feature extraction complete: {len(feature_sets)} feature sets")
        
        # Store results
        self.test_results['feature_extraction'] = feature_sets
        
        return feature_sets
    
    def test_feature_association(self, feature_sets: List[FeatureSet], 
                                test_pairs: int = 20) -> List[Tuple[List[AssociationScore], ValidationResult]]:
        """
        Test feature association between consecutive scans
        
        Args:
            feature_sets: List of feature sets
            test_pairs: Number of consecutive pairs to test
            
        Returns:
            List of (associations, validation_result) tuples
        """
        print(f"[TestSuite] Testing feature association on {min(test_pairs, len(feature_sets)-1)} pairs...")
        
        association_results = []
        
        # Test consecutive scan pairs
        num_pairs = min(test_pairs, len(feature_sets) - 1)
        
        for i in range(num_pairs):
            current_features = feature_sets[i + 1]
            previous_features = feature_sets[i]
            
            try:
                print(f"\n[TestSuite] --- Testing pair {i+1}/{num_pairs} ---")
                
                # Create descriptors
                current_descriptors = self.association_engine.create_descriptors(current_features, i+1)
                previous_descriptors = self.association_engine.create_descriptors(previous_features, i)
                
                print(f"[TestSuite] Created {len(current_descriptors)} current and {len(previous_descriptors)} previous descriptors")
                
                # Adapt parameters based on feature quality
                self.association_engine.adapt_parameters(current_features)
                
                # Perform association
                associations = self.association_engine.associate_features(
                    current_descriptors, previous_descriptors
                )
                
                print(f"[TestSuite] Found {len(associations)} associations")
                
                # Validate associations
                validation_result = self.validator.validate_associations(
                    associations, current_descriptors, previous_descriptors
                )
                
                print(f"[TestSuite] Validation: {'PASSED' if validation_result.is_valid else 'FAILED'} "
                      f"(confidence: {validation_result.confidence:.3f}, "
                      f"inliers: {validation_result.inlier_count}/{validation_result.total_associations})")
                
                # Store results
                association_results.append((associations, validation_result))
                
                # Create visualization for first few pairs
                if i < 5:
                    self._visualize_single_association(
                        current_descriptors, previous_descriptors, 
                        associations, validation_result, i+1
                    )
                
            except Exception as e:
                print(f"[TestSuite] ERROR in association test {i+1}: {e}")
                continue
        
        # Store results
        self.test_results['associations'] = [assoc for assoc, _ in association_results]
        self.test_results['validations'] = [val for _, val in association_results]
        
        return association_results
    
    def test_hybrid_pose_estimation(self, feature_sets: List[FeatureSet],
                                   association_results: List[Tuple[List[AssociationScore], ValidationResult]],
                                   test_count: int = 15) -> List[PoseEstimateWithConfidence]:
        """
        Test hybrid pose estimation
        
        Args:
            feature_sets: Feature sets for environment classification
            association_results: Association and validation results
            test_count: Number of poses to estimate
            
        Returns:
            List of hybrid pose estimates
        """
        print(f"[TestSuite] Testing hybrid pose estimation on {min(test_count, len(association_results))} samples...")
        
        pose_estimates = []
        
        num_tests = min(test_count, len(association_results))
        
        for i in range(num_tests):
            try:
                associations, validation_result = association_results[i]
                current_features = feature_sets[i + 1] if i + 1 < len(feature_sets) else feature_sets[-1]
                
                print(f"\n[TestSuite] --- Testing pose estimation {i+1}/{num_tests} ---")
                
                # Create descriptors for pose estimation
                current_descriptors = self.association_engine.create_descriptors(current_features, i+1)
                previous_descriptors = self.association_engine.create_descriptors(feature_sets[i], i) if i < len(feature_sets) else []
                
                # Estimate feature-based pose
                feature_pose = estimate_pose_from_associations(
                    associations, current_descriptors, previous_descriptors, validation_result
                )
                
                print(f"[TestSuite] Feature pose: x={feature_pose.pose.x:.3f}, y={feature_pose.pose.y:.3f}, "
                      f"θ={feature_pose.pose.theta:.3f}, conf={feature_pose.confidence:.3f}")
                
                # Create mock ICP pose (simulate ICP result)
                icp_pose = self._create_mock_icp_pose(feature_pose, i)
                
                print(f"[TestSuite] Mock ICP pose: x={icp_pose.pose.x:.3f}, y={icp_pose.pose.y:.3f}, "
                      f"θ={icp_pose.pose.theta:.3f}, conf={icp_pose.confidence:.3f}")
                
                # Create odometry pose
                odometry_pose = PoseEstimate(
                    current_features.robot_pose.x if current_features.robot_pose else 0,
                    current_features.robot_pose.y if current_features.robot_pose else 0,
                    current_features.robot_pose.theta if current_features.robot_pose else 0
                )
                
                # Estimate hybrid pose
                hybrid_pose = self.hybrid_estimator.estimate_hybrid_pose(
                    feature_pose=feature_pose,
                    icp_pose=icp_pose,
                    odometry_pose=odometry_pose,
                    current_features=current_features,
                    validation_result=validation_result
                )
                
                print(f"[TestSuite] Hybrid pose: x={hybrid_pose.pose.x:.3f}, y={hybrid_pose.pose.y:.3f}, "
                      f"θ={hybrid_pose.pose.theta:.3f}, conf={hybrid_pose.confidence:.3f}, "
                      f"source={hybrid_pose.source.value}")
                
                if hasattr(hybrid_pose, 'feature_weight'):
                    print(f"[TestSuite] Weights: feature={hybrid_pose.feature_weight:.3f}, "
                          f"icp={hybrid_pose.icp_weight:.3f}")
                
                pose_estimates.append(hybrid_pose)
                
            except Exception as e:
                print(f"[TestSuite] ERROR in pose estimation test {i+1}: {e}")
                continue
        
        # Store results
        self.test_results['pose_estimates'] = pose_estimates
        
        return pose_estimates
    
    def _create_mock_icp_pose(self, feature_pose: PoseEstimateWithConfidence, index: int) -> PoseEstimateWithConfidence:
        """
        Create a mock ICP pose for testing (simulates ICP output)
        
        Args:
            feature_pose: Feature-based pose to base mock on
            index: Test index for variation
            
        Returns:
            Mock ICP pose estimate
        """
        # Add some noise to simulate ICP differences
        noise_scale = 0.1 + 0.05 * (index % 3)  # Varying noise levels
        
        x_noise = np.random.normal(0, noise_scale)
        y_noise = np.random.normal(0, noise_scale)
        theta_noise = np.random.normal(0, noise_scale * 0.5)
        
        mock_pose = PoseEstimate(
            feature_pose.pose.x + x_noise,
            feature_pose.pose.y + y_noise,
            feature_pose.pose.theta + theta_noise
        )
        
        # Mock ICP confidence (varies based on simulated convergence)
        mock_confidence = 0.6 + 0.3 * np.random.random()
        
        icp_pose = create_pose_with_confidence(mock_pose, mock_confidence)
        icp_pose.source = PoseEstimateWithConfidence().source  # ICP_BASED
        icp_pose.convergence_confidence = mock_confidence
        icp_pose.iteration_count = np.random.randint(5, 15)
        
        return icp_pose
    
    def _visualize_single_association(self, current_descriptors: List[FeatureDescriptor],
                                    previous_descriptors: List[FeatureDescriptor],
                                    associations: List[AssociationScore],
                                    validation_result: ValidationResult,
                                    pair_index: int):
        """
        Create visualization for a single association test
        
        Args:
            current_descriptors: Current scan descriptors
            previous_descriptors: Previous scan descriptors
            associations: Feature associations
            validation_result: Validation result
            pair_index: Index of this test pair
        """
        try:
            fig = self.visualizer.visualize_associations(
                current_descriptors, previous_descriptors, associations, validation_result,
                title=f"Association Test {pair_index}"
            )
            
            if self.save_plots and fig:
                filename = f"{self.output_dir}/association_test_{pair_index:03d}.png"
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"[TestSuite] Saved association plot: {filename}")
            
        except Exception as e:
            print(f"[TestSuite] WARNING: Could not create visualization for pair {pair_index}: {e}")
    
    def create_comprehensive_performance_report(self):
        """
        Create comprehensive performance analysis and visualizations
        """
        print(f"\n[TestSuite] Creating comprehensive performance report...")
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Create performance visualizations
        self._create_performance_visualizations()
        
        # Create summary report
        self._create_summary_report()
        
        print(f"[TestSuite] Performance report complete!")
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        
        # Feature extraction metrics
        if self.test_results['feature_extraction']:
            feature_sets = self.test_results['feature_extraction']
            
            total_features = sum(len(fs.features) for fs in feature_sets)
            avg_features_per_scan = total_features / len(feature_sets)
            
            extraction_times = [fs.extraction_time for fs in feature_sets]
            avg_extraction_time = np.mean(extraction_times)
            
            quality_scores = [fs.quality_metrics.get('overall_quality', 0) for fs in feature_sets]
            avg_quality = np.mean(quality_scores)
            
            self.test_results['performance_metrics']['feature_extraction'] = {
                'total_features': total_features,
                'avg_features_per_scan': avg_features_per_scan,
                'avg_extraction_time': avg_extraction_time,
                'avg_quality_score': avg_quality,
                'extraction_time_std': np.std(extraction_times),
                'quality_score_std': np.std(quality_scores)
            }
        
        # Association metrics
        if self.test_results['associations']:
            associations = self.test_results['associations']
            
            total_associations = sum(len(assoc_list) for assoc_list in associations)
            avg_associations_per_pair = total_associations / len(associations) if associations else 0
            
            all_scores = []
            all_distances = []
            for assoc_list in associations:
                all_scores.extend([a.score for a in assoc_list])
                all_distances.extend([a.distance for a in assoc_list])
            
            self.test_results['performance_metrics']['associations'] = {
                'total_associations': total_associations,
                'avg_associations_per_pair': avg_associations_per_pair,
                'avg_association_score': np.mean(all_scores) if all_scores else 0,
                'avg_association_distance': np.mean(all_distances) if all_distances else 0,
                'score_std': np.std(all_scores) if all_scores else 0,
                'distance_std': np.std(all_distances) if all_distances else 0
            }
        
        # Validation metrics
        if self.test_results['validations']:
            validations = self.test_results['validations']
            
            successful_validations = sum(1 for v in validations if v.is_valid)
            validation_success_rate = successful_validations / len(validations)
            
            avg_confidence = np.mean([v.confidence for v in validations])
            avg_inlier_ratio = np.mean([v.inlier_count / max(1, v.total_associations) for v in validations])
            
            self.test_results['performance_metrics']['validation'] = {
                'total_validations': len(validations),
                'successful_validations': successful_validations,
                'validation_success_rate': validation_success_rate,
                'avg_confidence': avg_confidence,
                'avg_inlier_ratio': avg_inlier_ratio,
                'avg_geometric_consistency': np.mean([v.geometric_consistency for v in validations]),
                'avg_temporal_consistency': np.mean([v.temporal_consistency for v in validations])
            }
        
        # Pose estimation metrics
        if self.test_results['pose_estimates']:
            poses = self.test_results['pose_estimates']
            
            avg_confidence = np.mean([p.confidence for p in poses])
            avg_processing_time = np.mean([p.processing_time for p in poses])
            
            source_distribution = {}
            for pose in poses:
                source = pose.source.value
                source_distribution[source] = source_distribution.get(source, 0) + 1
            
            # Calculate average weights for hybrid poses
            feature_weights = [getattr(p, 'feature_weight', 0.5) for p in poses]
            avg_feature_weight = np.mean(feature_weights)
            
            self.test_results['performance_metrics']['pose_estimation'] = {
                'total_estimates': len(poses),
                'avg_confidence': avg_confidence,
                'avg_processing_time': avg_processing_time,
                'source_distribution': source_distribution,
                'avg_feature_weight': avg_feature_weight,
                'feature_weight_std': np.std(feature_weights),
                'avg_position_uncertainty': np.mean([p.position_uncertainty for p in poses]),
                'avg_orientation_uncertainty': np.mean([p.orientation_uncertainty for p in poses])
            }
    
    def _create_performance_visualizations(self):
        """Create comprehensive performance visualizations"""
        
        # Feature extraction performance
        if self.test_results['feature_extraction']:
            self._plot_feature_extraction_performance()
        
        # Association performance
        if self.test_results['associations'] and self.test_results['validations']:
            self._plot_association_performance()
        
        # Pose estimation performance
        if self.test_results['pose_estimates']:
            self._plot_pose_estimation_performance()
        
        # Combined performance overview
        self._plot_combined_performance_overview()
    
    def _plot_feature_extraction_performance(self):
        """Plot feature extraction performance metrics"""
        
        feature_sets = self.test_results['feature_extraction']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Extraction Performance')
        
        # Feature count over time
        feature_counts = [len(fs.features) for fs in feature_sets]
        axes[0, 0].plot(feature_counts, 'b-', linewidth=2)
        axes[0, 0].set_title('Feature Count per Scan')
        axes[0, 0].set_xlabel('Scan Number')
        axes[0, 0].set_ylabel('Feature Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Quality scores over time
        quality_scores = [fs.quality_metrics.get('overall_quality', 0) for fs in feature_sets]
        axes[0, 1].plot(quality_scores, 'g-', linewidth=2)
        axes[0, 1].set_title('Feature Quality Score')
        axes[0, 1].set_xlabel('Scan Number')
        axes[0, 1].set_ylabel('Quality Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Extraction time distribution
        extraction_times = [fs.extraction_time for fs in feature_sets]
        axes[1, 0].hist(extraction_times, bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('Extraction Time Distribution')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature type distribution
        feature_type_counts = {'sharp_edges': [], 'less_sharp_edges': [], 'planar_features': [], 'less_planar_features': []}
        for fs in feature_sets:
            counts = fs.get_feature_count_by_type()
            for ftype in feature_type_counts:
                feature_type_counts[ftype].append(counts.get(ftype, 0))
        
        x = range(len(feature_sets))
        bottom = np.zeros(len(feature_sets))
        colors = ['red', 'orange', 'blue', 'lightblue']
        
        for i, (ftype, counts) in enumerate(feature_type_counts.items()):
            axes[1, 1].bar(x, counts, bottom=bottom, label=ftype.replace('_', ' ').title(), 
                          alpha=0.7, color=colors[i])
            bottom += counts
        
        axes[1, 1].set_title('Feature Type Distribution')
        axes[1, 1].set_xlabel('Scan Number')
        axes[1, 1].set_ylabel('Feature Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.output_dir}/feature_extraction_performance.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"[TestSuite] Saved feature extraction performance plot: {filename}")
            plt.close(fig)
    
    def _plot_association_performance(self):
        """Plot association and validation performance"""
        
        associations = self.test_results['associations']
        validations = self.test_results['validations']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Feature Association Performance')
        
        # Association count
        association_counts = [len(assoc_list) for assoc_list in associations]
        axes[0, 0].plot(association_counts, 'b-', linewidth=2)
        axes[0, 0].set_title('Associations per Scan Pair')
        axes[0, 0].set_xlabel('Pair Number')
        axes[0, 0].set_ylabel('Association Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Association scores
        all_scores = []
        for assoc_list in associations:
            all_scores.extend([a.score for a in assoc_list])
        
        axes[0, 1].hist(all_scores, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Association Score Distribution')
        axes[0, 1].set_xlabel('Association Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation success rate
        validation_success = [1 if v.is_valid else 0 for v in validations]
        cumulative_success = np.cumsum(validation_success) / np.arange(1, len(validation_success) + 1)
        axes[0, 2].plot(cumulative_success, 'r-', linewidth=2)
        axes[0, 2].set_title('Cumulative Validation Success Rate')
        axes[0, 2].set_xlabel('Pair Number')
        axes[0, 2].set_ylabel('Success Rate')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Validation confidence
        validation_confidences = [v.confidence for v in validations]
        axes[1, 0].plot(validation_confidences, 'purple', linewidth=2)
        axes[1, 0].set_title('Validation Confidence')
        axes[1, 0].set_xlabel('Pair Number')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Inlier ratios
        inlier_ratios = [v.inlier_count / max(1, v.total_associations) for v in validations]
        axes[1, 1].plot(inlier_ratios, 'brown', linewidth=2)
        axes[1, 1].set_title('RANSAC Inlier Ratio')
        axes[1, 1].set_xlabel('Pair Number')
        axes[1, 1].set_ylabel('Inlier Ratio')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Geometric vs temporal consistency
        geometric_consistency = [v.geometric_consistency for v in validations]
        temporal_consistency = [v.temporal_consistency for v in validations]
        axes[1, 2].scatter(geometric_consistency, temporal_consistency, alpha=0.6)
        axes[1, 2].set_title('Geometric vs Temporal Consistency')
        axes[1, 2].set_xlabel('Geometric Consistency')
        axes[1, 2].set_ylabel('Temporal Consistency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.output_dir}/association_performance.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"[TestSuite] Saved association performance plot: {filename}")
            plt.close(fig)
    
    def _plot_pose_estimation_performance(self):
        """Plot pose estimation performance"""
        
        poses = self.test_results['pose_estimates']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Hybrid Pose Estimation Performance')
        
        # Confidence over time
        confidences = [p.confidence for p in poses]
        axes[0, 0].plot(confidences, 'b-', linewidth=2)
        axes[0, 0].set_title('Pose Confidence')
        axes[0, 0].set_xlabel('Estimation Number')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Source distribution
        sources = [p.source.value for p in poses]
        unique_sources = list(set(sources))
        source_counts = [sources.count(source) for source in unique_sources]
        axes[0, 1].pie(source_counts, labels=[s.replace('_', ' ').title() for s in unique_sources], 
                      autopct='%1.1f%%')
        axes[0, 1].set_title('Pose Source Distribution')
        
        # Feature vs ICP weighting (for hybrid poses)
        feature_weights = [getattr(p, 'feature_weight', 0.5) for p in poses]
        axes[0, 2].plot(feature_weights, 'g-', linewidth=2, label='Feature Weight')
        axes[0, 2].plot([1-w for w in feature_weights], 'r-', linewidth=2, label='ICP Weight')
        axes[0, 2].set_title('Feature vs ICP Weighting')
        axes[0, 2].set_xlabel('Estimation Number')
        axes[0, 2].set_ylabel('Weight')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Processing time
        processing_times = [p.processing_time for p in poses]
        axes[1, 0].plot(processing_times, 'purple', linewidth=2)
        axes[1, 0].set_title('Processing Time')
        axes[1, 0].set_xlabel('Estimation Number')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Position uncertainty
        position_uncertainties = [p.position_uncertainty for p in poses]
        axes[1, 1].plot(position_uncertainties, 'orange', linewidth=2)
        axes[1, 1].set_title('Position Uncertainty')
        axes[1, 1].set_xlabel('Estimation Number')
        axes[1, 1].set_ylabel('Uncertainty (m)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Confidence vs uncertainty correlation
        axes[1, 2].scatter(confidences, position_uncertainties, alpha=0.6)
        axes[1, 2].set_title('Confidence vs Position Uncertainty')
        axes[1, 2].set_xlabel('Confidence')
        axes[1, 2].set_ylabel('Position Uncertainty (m)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.output_dir}/pose_estimation_performance.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"[TestSuite] Saved pose estimation performance plot: {filename}")
            plt.close(fig)
    
    def _plot_combined_performance_overview(self):
        """Create a combined performance overview"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get key metrics
        metrics = self.test_results['performance_metrics']
        
        categories = []
        values = []
        colors = []
        
        if 'feature_extraction' in metrics:
            categories.append('Feature\nExtraction\nTime (ms)')
            values.append(metrics['feature_extraction']['avg_extraction_time'])
            colors.append('blue')
            
            categories.append('Feature\nQuality\nScore')
            values.append(metrics['feature_extraction']['avg_quality_score'] * 100)  # Scale to 0-100
            colors.append('green')
        
        if 'associations' in metrics:
            categories.append('Avg\nAssociations\nper Pair')
            values.append(metrics['associations']['avg_associations_per_pair'])
            colors.append('orange')
            
            categories.append('Association\nScore\n(x100)')
            values.append(metrics['associations']['avg_association_score'] * 100)
            colors.append('red')
        
        if 'validation' in metrics:
            categories.append('Validation\nSuccess Rate\n(%)')
            values.append(metrics['validation']['validation_success_rate'] * 100)
            colors.append('purple')
            
            categories.append('Validation\nConfidence\n(x100)')
            values.append(metrics['validation']['avg_confidence'] * 100)
            colors.append('brown')
        
        if 'pose_estimation' in metrics:
            categories.append('Pose\nConfidence\n(x100)')
            values.append(metrics['pose_estimation']['avg_confidence'] * 100)
            colors.append('pink')
            
            categories.append('Processing\nTime (ms)')
            values.append(metrics['pose_estimation']['avg_processing_time'])
            colors.append('gray')
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Feature Association System - Performance Overview', fontsize=16, fontweight='bold')
        ax.set_ylabel('Metric Value')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if self.save_plots:
            filename = f"{self.output_dir}/performance_overview.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"[TestSuite] Saved performance overview: {filename}")
            plt.close(fig)
    
    def _create_summary_report(self):
        """Create a text summary report"""
        
        report_filename = f"{self.output_dir}/test_summary_report.txt"
        
        with open(report_filename, 'w') as f:
            f.write("FEATURE ASSOCIATION SYSTEM - TEST SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            # Feature extraction summary
            if 'feature_extraction' in self.test_results['performance_metrics']:
                metrics = self.test_results['performance_metrics']['feature_extraction']
                f.write("FEATURE EXTRACTION PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total features extracted: {metrics['total_features']}\n")
                f.write(f"Average features per scan: {metrics['avg_features_per_scan']:.1f}\n")
                f.write(f"Average extraction time: {metrics['avg_extraction_time']:.2f} ms\n")
                f.write(f"Average quality score: {metrics['avg_quality_score']:.3f}\n")
                
                if metrics['avg_extraction_time'] <= 15.0:
                    f.write("✓ PERFORMANCE: EXCELLENT (≤15ms)\n")
                elif metrics['avg_extraction_time'] <= 25.0:
                    f.write("⚠ PERFORMANCE: GOOD (15-25ms)\n")
                else:
                    f.write("⚠ PERFORMANCE: NEEDS OPTIMIZATION (>25ms)\n")
                
                f.write("\n")
            
            # Association summary
            if 'associations' in self.test_results['performance_metrics']:
                metrics = self.test_results['performance_metrics']['associations']
                f.write("FEATURE ASSOCIATION PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total associations: {metrics['total_associations']}\n")
                f.write(f"Average associations per pair: {metrics['avg_associations_per_pair']:.1f}\n")
                f.write(f"Average association score: {metrics['avg_association_score']:.3f}\n")
                f.write(f"Average association distance: {metrics['avg_association_distance']:.3f} m\n")
                f.write("\n")
            
            # Validation summary
            if 'validation' in self.test_results['performance_metrics']:
                metrics = self.test_results['performance_metrics']['validation']
                f.write("VALIDATION PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total validations: {metrics['total_validations']}\n")
                f.write(f"Successful validations: {metrics['successful_validations']}\n")
                f.write(f"Validation success rate: {metrics['validation_success_rate']*100:.1f}%\n")
                f.write(f"Average confidence: {metrics['avg_confidence']:.3f}\n")
                f.write(f"Average inlier ratio: {metrics['avg_inlier_ratio']:.3f}\n")
                
                if metrics['validation_success_rate'] >= 0.8:
                    f.write("✓ VALIDATION: EXCELLENT (≥80% success)\n")
                elif metrics['validation_success_rate'] >= 0.6:
                    f.write("✓ VALIDATION: GOOD (60-80% success)\n")
                else:
                    f.write("⚠ VALIDATION: NEEDS IMPROVEMENT (<60% success)\n")
                
                f.write("\n")
            
            # Pose estimation summary
            if 'pose_estimation' in self.test_results['performance_metrics']:
                metrics = self.test_results['performance_metrics']['pose_estimation']
                f.write("POSE ESTIMATION PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total pose estimates: {metrics['total_estimates']}\n")
                f.write(f"Average confidence: {metrics['avg_confidence']:.3f}\n")
                f.write(f"Average processing time: {metrics['avg_processing_time']:.2f} ms\n")
                f.write(f"Average feature weight: {metrics['avg_feature_weight']:.3f}\n")
                f.write(f"Average position uncertainty: {metrics['avg_position_uncertainty']:.3f} m\n")
                
                f.write("\nPose source distribution:\n")
                for source, count in metrics['source_distribution'].items():
                    percentage = count / metrics['total_estimates'] * 100
                    f.write(f"  {source.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n")
                
                f.write("\n")
            
            # Overall assessment
            f.write("OVERALL ASSESSMENT:\n")
            f.write("-" * 40 + "\n")
            
            # Calculate overall scores
            scores = []
            if 'feature_extraction' in self.test_results['performance_metrics']:
                fe_metrics = self.test_results['performance_metrics']['feature_extraction']
                if fe_metrics['avg_extraction_time'] <= 15.0:
                    scores.append(('Feature Extraction Performance', 'EXCELLENT'))
                elif fe_metrics['avg_extraction_time'] <= 25.0:
                    scores.append(('Feature Extraction Performance', 'GOOD'))
                else:
                    scores.append(('Feature Extraction Performance', 'NEEDS WORK'))
            
            if 'validation' in self.test_results['performance_metrics']:
                val_metrics = self.test_results['performance_metrics']['validation']
                if val_metrics['validation_success_rate'] >= 0.8:
                    scores.append(('Association Validation', 'EXCELLENT'))
                elif val_metrics['validation_success_rate'] >= 0.6:
                    scores.append(('Association Validation', 'GOOD'))
                else:
                    scores.append(('Association Validation', 'NEEDS WORK'))
            
            for metric, score in scores:
                f.write(f"{metric}: {score}\n")
            
            if len([s for _, s in scores if s == 'EXCELLENT']) >= len(scores) * 0.7:
                f.write("\n✓ SYSTEM READY FOR INTEGRATION\n")
            elif len([s for _, s in scores if s in ['EXCELLENT', 'GOOD']]) >= len(scores) * 0.8:
                f.write("\n⚠ SYSTEM MOSTLY READY - MINOR TUNING RECOMMENDED\n")
            else:
                f.write("\n⚠ SYSTEM NEEDS OPTIMIZATION BEFORE INTEGRATION\n")
        
        print(f"[TestSuite] Created summary report: {report_filename}")
    
    def run_comprehensive_test(self, file_path: str, max_entries: int = 50,
                             test_pairs: int = 20, test_poses: int = 15) -> bool:
        """
        Run the complete test suite
        
        Args:
            file_path: Path to LiDAR data file
            max_entries: Maximum scans to load
            test_pairs: Number of association pairs to test
            test_poses: Number of pose estimates to test
            
        Returns:
            True if tests completed successfully
        """
        print(f"\n{'='*80}")
        print(f"FEATURE ASSOCIATION SYSTEM - COMPREHENSIVE TEST SUITE")
        print(f"{'='*80}")
        
        try:
            # Step 1: Load test data
            lidar_data = self.load_test_data(file_path, max_entries)
            if not lidar_data:
                return False
            
            # Step 2: Extract features
            feature_sets = self.extract_features_from_data(lidar_data)
            if not feature_sets:
                return False
            
            # Step 3: Test feature association
            association_results = self.test_feature_association(feature_sets, test_pairs)
            if not association_results:
                return False
            
            # Step 4: Test hybrid pose estimation
            pose_estimates = self.test_hybrid_pose_estimation(feature_sets, association_results, test_poses)
            
            # Step 5: Create performance report
            self.create_comprehensive_performance_report()
            
            print(f"\n{'='*80}")
            print(f"COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
            print(f"Results saved to: {self.output_dir}")
            print(f"{'='*80}")
            
            return True
            
        except Exception as e:
            print(f"\n[TestSuite] ERROR: Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """
    Main function for standalone testing
    """
    parser = argparse.ArgumentParser(description='Feature Association System Test Suite')
    
    parser.add_argument('--file', type=str,
                       default="../dataset/raw_data/laser_data_synchronized_short_u_turn_fast_processed_reduced180.clf",
                       help='Path to LiDAR data file')
    parser.add_argument('--max_entries', type=int, default=50,
                       help='Maximum number of scans to process')
    parser.add_argument('--test_pairs', type=int, default=20,
                       help='Number of consecutive pairs to test for association')
    parser.add_argument('--test_poses', type=int, default=15,
                       help='Number of pose estimates to test')
    parser.add_argument('--output_dir', type=str, default='association_test_results',
                       help='Output directory for test results')
    parser.add_argument('--debug_level', type=int, default=2, choices=[0, 1, 2, 3],
                       help='Debug output level')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save plots (display only)')
    
    args = parser.parse_args()
    
    print("Feature Association System - Comprehensive Test Suite")
    print(f"Data file: {args.file}")
    print(f"Max entries: {args.max_entries}")
    print(f"Test pairs: {args.test_pairs}")
    print(f"Test poses: {args.test_poses}")
    print(f"Output directory: {args.output_dir}")
    print(f"Debug level: {args.debug_level}")
    print()
    
    # Initialize test suite
    test_suite = FeatureAssociationTestSuite(
        output_dir=args.output_dir,
        save_plots=not args.no_save,
        debug_level=args.debug_level
    )
    
    # Run comprehensive test
    success = test_suite.run_comprehensive_test(
        file_path=args.file,
        max_entries=args.max_entries,
        test_pairs=args.test_pairs,
        test_poses=args.test_poses
    )
    
    if success:
        print("\nTest suite completed successfully!")
        print(f"Check the results in: {args.output_dir}")
    else:
        print("\nTest suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()