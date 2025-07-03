#!/usr/bin/env python3
"""
Figure 6 Generator: Adaptive Weight Behavior Based on Feature Analysis
Generates a four-panel temporal analysis showing:
(a) feature vs. ICP weight evolution based on feature distribution quality
(b) feature type influence on fusion weight assignment
(c) correlation between feature classification confidence and weight adjustment  
(d) fusion confidence scores throughout operation with feature-specific contributions
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import math
import random
from collections import defaultdict, deque
from scipy import stats

# Import your existing modules
from feature_extractor import FeatureExtractor, FeatureType
from lidar_utility_functions import convert_scans_to_cartesian, read_lidar_data_from_file

class Figure6Generator:
    """
    Generates Figure 6: Adaptive Weight Behavior Based on Feature Analysis
    """
    
    def __init__(self, debug_level=1):
        self.debug_level = debug_level
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            debug_level=0,
            curvature_window_size=5,
            num_sectors=6,
            sharp_edge_threshold=0.15,
            planar_threshold=0.08,
            max_sharp_edges_per_sector=1,
            max_less_sharp_per_sector=8,
            max_planar_per_sector=2
        )
        
        # Feature type colors
        self.feature_colors = {
            FeatureType.SHARP_EDGE: '#FF4444',        # Red
            FeatureType.LESS_SHARP_EDGE: '#FF8844',   # Orange
            FeatureType.PLANAR: '#4488FF',            # Blue  
            FeatureType.LESS_PLANAR: '#44FF88'        # Green
        }
        
        self.feature_labels = {
            FeatureType.SHARP_EDGE: 'Sharp Edges',
            FeatureType.LESS_SHARP_EDGE: 'Less Sharp Edges', 
            FeatureType.PLANAR: 'Planar Features',
            FeatureType.LESS_PLANAR: 'Less Planar Features'
        }
        
        # Adaptive fusion parameters
        self.fusion_params = {
            'base_feature_weight': 0.5,
            'feature_quality_influence': 0.4,
            'spatial_distribution_influence': 0.3,
            'confidence_influence': 0.3,
            'min_weight': 0.1,
            'max_weight': 0.9
        }

    def simulate_adaptive_fusion_trajectory(self, file_path, max_entries=200, analysis_step=2):
        """
        Simulate adaptive fusion behavior over a trajectory
        """
        print("Simulating adaptive fusion behavior over trajectory...")
        parsed_data_list = read_lidar_data_from_file(file_path, max_entries)
        
        if len(parsed_data_list) < 10:
            raise ValueError("Need at least 10 scans for trajectory analysis")
        
        # Subsample for efficiency
        analysis_scans = parsed_data_list[::analysis_step]
        print(f"Analyzing {len(analysis_scans)} scans (every {analysis_step}nd scan)...")
        
        # Initialize trajectory data
        trajectory_data = {
            'scan_indices': [],
            'feature_weights': [],
            'icp_weights': [],
            'fusion_confidences': [],
            'feature_quality_scores': [],
            'spatial_distribution_scores': [],
            'feature_type_counts': defaultdict(list),
            'feature_type_confidences': defaultdict(list),
            'feature_type_contributions': defaultdict(list),
            'environment_classifications': [],
            'weight_adjustment_reasons': []
        }
        
        angle_min, angle_max = -math.pi/2, math.pi/2
        
        # Process each scan in the trajectory
        for i, data in enumerate(analysis_scans):
            if i % 10 == 0:
                print(f"  Processing scan {i}/{len(analysis_scans)}...")
            
            try:
                # Extract features
                x_points, y_points = convert_scans_to_cartesian(
                    data['scan_ranges'], angle_min, angle_max, data['pose'],
                    flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
                )
                
                feature_set = self.feature_extractor.extract_features(x_points, y_points, data['scan_ranges'])
                
                if not feature_set.features:
                    continue
                
                # Analyze feature characteristics for this scan
                scan_analysis = self._analyze_scan_features(feature_set, i)
                
                # Calculate adaptive weights
                fusion_weights = self._calculate_adaptive_weights(scan_analysis)
                
                # Store trajectory data
                trajectory_data['scan_indices'].append(i)
                trajectory_data['feature_weights'].append(fusion_weights['feature_weight'])
                trajectory_data['icp_weights'].append(fusion_weights['icp_weight'])
                trajectory_data['fusion_confidences'].append(fusion_weights['fusion_confidence'])
                trajectory_data['feature_quality_scores'].append(scan_analysis['feature_quality'])
                trajectory_data['spatial_distribution_scores'].append(scan_analysis['spatial_distribution'])
                trajectory_data['environment_classifications'].append(scan_analysis['environment_type'])
                trajectory_data['weight_adjustment_reasons'].append(fusion_weights['adjustment_reason'])
                
                # Store feature-type specific data
                for ftype in FeatureType:
                    count = scan_analysis['feature_counts'][ftype]
                    confidence = scan_analysis['feature_confidences'][ftype]
                    contribution = scan_analysis['feature_contributions'][ftype]
                    
                    trajectory_data['feature_type_counts'][ftype].append(count)
                    trajectory_data['feature_type_confidences'][ftype].append(confidence)
                    trajectory_data['feature_type_contributions'][ftype].append(contribution)
                
            except Exception as e:
                if self.debug_level > 1:
                    print(f"    Failed to process scan {i}: {e}")
                continue
        
        print(f"Successfully processed {len(trajectory_data['scan_indices'])} scans")
        return trajectory_data

    def _analyze_scan_features(self, feature_set, scan_index):
        """
        Comprehensive analysis of features in a single scan
        """
        features = feature_set.features
        
        # Count features by type
        feature_counts = defaultdict(int)
        feature_confidences = defaultdict(list)
        
        for feature in features:
            feature_counts[feature.feature_type] += 1
            # Use curvature as proxy for feature confidence
            confidence = min(1.0, abs(feature.curvature) * 5)  # Scale curvature to confidence
            feature_confidences[feature.feature_type].append(confidence)
        
        # Calculate average confidence by type
        avg_confidences = {}
        for ftype in FeatureType:
            if feature_confidences[ftype]:
                avg_confidences[ftype] = np.mean(feature_confidences[ftype])
            else:
                avg_confidences[ftype] = 0.0
        
        # Calculate feature quality score
        total_features = len(features)
        feature_quality = self._calculate_feature_quality(feature_counts, avg_confidences, total_features)
        
        # Calculate spatial distribution score
        spatial_distribution = self._calculate_spatial_distribution(features)
        
        # Classify environment type
        environment_type = self._classify_environment(feature_counts, feature_quality, spatial_distribution)
        
        # Calculate feature type contributions to fusion
        feature_contributions = self._calculate_feature_contributions(feature_counts, avg_confidences)
        
        return {
            'feature_counts': feature_counts,
            'feature_confidences': avg_confidences,
            'feature_contributions': feature_contributions,
            'feature_quality': feature_quality,
            'spatial_distribution': spatial_distribution,
            'environment_type': environment_type,
            'total_features': total_features
        }

    def _calculate_feature_quality(self, feature_counts, avg_confidences, total_features):
        """Calculate overall feature quality score (0-1)"""
        if total_features == 0:
            return 0.0
        
        # Weight different feature types by their reliability
        type_weights = {
            FeatureType.SHARP_EDGE: 1.0,      # Most reliable
            FeatureType.LESS_SHARP_EDGE: 0.8,
            FeatureType.PLANAR: 0.6,
            FeatureType.LESS_PLANAR: 0.4      # Least reliable
        }
        
        quality_score = 0.0
        total_weighted_count = 0.0
        
        for ftype, count in feature_counts.items():
            if count > 0:
                type_weight = type_weights.get(ftype, 0.5)
                confidence = avg_confidences.get(ftype, 0.5)
                weighted_contribution = count * type_weight * confidence
                
                quality_score += weighted_contribution
                total_weighted_count += count * type_weight
        
        if total_weighted_count > 0:
            normalized_quality = quality_score / total_weighted_count
            # Scale by feature density (more features = better)
            density_factor = min(1.0, total_features / 20.0)  # Saturate at 20 features
            return normalized_quality * density_factor
        
        return 0.0

    def _calculate_spatial_distribution(self, features):
        """Calculate spatial distribution uniformity score (0-1)"""
        if not features:
            return 0.0
        
        # Check sector distribution
        sector_counts = defaultdict(int)
        for feature in features:
            sector_counts[feature.sector] += 1
        
        # Calculate uniformity (lower variance = better distribution)
        sector_values = [sector_counts[i] for i in range(6)]  # 6 sectors
        if sum(sector_values) == 0:
            return 0.0
        
        mean_per_sector = sum(sector_values) / 6
        variance = np.var(sector_values)
        
        # Convert variance to uniformity score
        if mean_per_sector > 0:
            coefficient_of_variation = np.sqrt(variance) / mean_per_sector
            uniformity = 1.0 / (1.0 + coefficient_of_variation)
        else:
            uniformity = 0.0
        
        # Bonus for occupied sectors
        occupied_sectors = sum(1 for count in sector_values if count > 0)
        sector_coverage = occupied_sectors / 6.0
        
        return (uniformity * 0.7 + sector_coverage * 0.3)

    def _classify_environment(self, feature_counts, feature_quality, spatial_distribution):
        """Classify environment type based on feature characteristics"""
        total_features = sum(feature_counts.values())
        
        # Calculate feature type ratios
        edge_ratio = (feature_counts[FeatureType.SHARP_EDGE] + 
                     feature_counts[FeatureType.LESS_SHARP_EDGE]) / max(1, total_features)
        planar_ratio = (feature_counts[FeatureType.PLANAR] + 
                       feature_counts[FeatureType.LESS_PLANAR]) / max(1, total_features)
        
        # Classify based on feature characteristics
        if feature_quality > 0.7 and edge_ratio > 0.4:
            return 'structured'  # High quality with many edges (corners, intersections)
        elif planar_ratio > 0.6 and spatial_distribution < 0.4:
            return 'corridor'    # Dominated by planar features (hallways)
        elif total_features < 10:
            return 'sparse'      # Few features available
        else:
            return 'complex'     # Mixed characteristics

    def _calculate_feature_contributions(self, feature_counts, avg_confidences):
        """Calculate how much each feature type contributes to fusion confidence"""
        contributions = {}
        total_weighted_features = 0.0
        
        # Weight contributions by count and confidence
        for ftype in FeatureType:
            count = feature_counts[ftype]
            confidence = avg_confidences[ftype]
            weighted_count = count * confidence
            total_weighted_features += weighted_count
        
        # Normalize contributions
        for ftype in FeatureType:
            count = feature_counts[ftype]
            confidence = avg_confidences[ftype]
            weighted_count = count * confidence
            
            if total_weighted_features > 0:
                contributions[ftype] = weighted_count / total_weighted_features
            else:
                contributions[ftype] = 0.0
        
        return contributions

    def _calculate_adaptive_weights(self, scan_analysis):
        """Calculate adaptive fusion weights based on scan analysis"""
        
        feature_quality = scan_analysis['feature_quality']
        spatial_distribution = scan_analysis['spatial_distribution']
        environment_type = scan_analysis['environment_type']
        total_features = scan_analysis['total_features']
        
        # Base weight assignment by environment type
        base_weights = {
            'structured': 0.7,   # Favor features in structured environments
            'complex': 0.6,      # Balanced in complex environments
            'corridor': 0.4,     # Favor ICP in corridors (planar-heavy)
            'sparse': 0.3        # Favor ICP when few features available
        }
        
        base_feature_weight = base_weights.get(environment_type, 0.5)
        
        # Adjust based on feature quality
        quality_adjustment = (feature_quality - 0.5) * self.fusion_params['feature_quality_influence']
        
        # Adjust based on spatial distribution
        distribution_adjustment = (spatial_distribution - 0.5) * self.fusion_params['spatial_distribution_influence']
        
        # Adjust based on feature count
        if total_features < 5:
            count_adjustment = -0.2  # Favor ICP with few features
        elif total_features > 20:
            count_adjustment = 0.1   # Slight favor to features with many
        else:
            count_adjustment = 0.0
        
        # Calculate final feature weight
        feature_weight = base_feature_weight + quality_adjustment + distribution_adjustment + count_adjustment
        
        # Clamp to valid range
        feature_weight = max(self.fusion_params['min_weight'], 
                           min(self.fusion_params['max_weight'], feature_weight))
        
        icp_weight = 1.0 - feature_weight
        
        # Calculate fusion confidence
        fusion_confidence = self._calculate_fusion_confidence(scan_analysis, feature_weight, icp_weight)
        
        # Determine adjustment reason
        adjustment_reason = self._get_adjustment_reason(
            environment_type, feature_quality, spatial_distribution, total_features
        )
        
        return {
            'feature_weight': feature_weight,
            'icp_weight': icp_weight,
            'fusion_confidence': fusion_confidence,
            'adjustment_reason': adjustment_reason
        }

    def _calculate_fusion_confidence(self, scan_analysis, feature_weight, icp_weight):
        """Calculate overall fusion confidence"""
        feature_quality = scan_analysis['feature_quality']
        spatial_distribution = scan_analysis['spatial_distribution']
        
        # Base confidence from feature quality
        feature_confidence = feature_quality
        
        # ICP confidence (assumed constant for simulation)
        icp_confidence = 0.75
        
        # Weighted combination
        weighted_confidence = (feature_weight * feature_confidence + 
                             icp_weight * icp_confidence)
        
        # Bonus for good spatial distribution
        distribution_bonus = spatial_distribution * 0.1
        
        # Penalty for extreme weights (balanced is more confident)
        balance_factor = 1.0 - abs(feature_weight - 0.5) * 0.2
        
        final_confidence = (weighted_confidence + distribution_bonus) * balance_factor
        
        return min(1.0, max(0.0, final_confidence))

    def _get_adjustment_reason(self, environment_type, feature_quality, spatial_distribution, total_features):
        """Determine the primary reason for weight adjustment"""
        if total_features < 5:
            return 'low_feature_count'
        elif feature_quality > 0.8:
            return 'high_feature_quality'
        elif feature_quality < 0.3:
            return 'low_feature_quality'
        elif environment_type == 'structured':
            return 'structured_environment'
        elif environment_type == 'sparse':
            return 'sparse_environment'
        elif spatial_distribution > 0.7:
            return 'good_distribution'
        elif spatial_distribution < 0.3:
            return 'poor_distribution'
        else:
            return 'balanced_conditions'

    def create_panel_a(self, ax, trajectory_data):
        """Panel (a): Feature vs. ICP weight evolution based on feature distribution quality"""
        
        scan_indices = trajectory_data['scan_indices']
        feature_weights = trajectory_data['feature_weights']
        icp_weights = trajectory_data['icp_weights']
        feature_quality = trajectory_data['feature_quality_scores']
        
        # Main weight evolution plot
        ax.plot(scan_indices, feature_weights, 'b-', linewidth=2, label='Feature Weight', alpha=0.8)
        ax.plot(scan_indices, icp_weights, 'r-', linewidth=2, label='ICP Weight', alpha=0.8)
        
        # Create secondary y-axis for feature quality
        ax2 = ax.twinx()
        ax2.plot(scan_indices, feature_quality, 'g--', linewidth=1.5, alpha=0.6, label='Feature Quality')
        
        # Color background based on which method is dominant
        for i in range(len(scan_indices) - 1):
            if feature_weights[i] > icp_weights[i]:
                ax.axvspan(scan_indices[i], scan_indices[i+1], alpha=0.1, color='blue')
            else:
                ax.axvspan(scan_indices[i], scan_indices[i+1], alpha=0.1, color='red')
        
        ax.set_xlabel('Scan Index')
        ax.set_ylabel('Fusion Weight')
        ax2.set_ylabel('Feature Quality Score')
        ax.set_title('(a) Adaptive Weight Evolution Based on Feature Quality')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1)

    def create_panel_b(self, ax, trajectory_data):
        """Panel (b): Feature type influence on fusion weight assignment"""
        
        # Analyze correlation between feature types and weights
        feature_weights = trajectory_data['feature_weights']
        
        # Create scatter plots for each feature type count vs feature weight
        for ftype in FeatureType:
            counts = trajectory_data['feature_type_counts'][ftype]
            
            if counts:
                ax.scatter(counts, feature_weights, 
                          color=self.feature_colors[ftype], 
                          label=self.feature_labels[ftype],
                          alpha=0.6, s=20)
        
        # Add trend lines
        for ftype in FeatureType:
            counts = trajectory_data['feature_type_counts'][ftype]
            if counts and len(counts) > 5:
                # Calculate correlation and trend line
                slope, intercept, r_value, p_value, std_err = stats.linregress(counts, feature_weights)
                
                if abs(r_value) > 0.3:  # Only show significant correlations
                    x_trend = np.array([min(counts), max(counts)])
                    y_trend = slope * x_trend + intercept
                    ax.plot(x_trend, y_trend, '--', 
                           color=self.feature_colors[ftype], alpha=0.8, linewidth=1.5)
        
        ax.set_xlabel('Feature Count')
        ax.set_ylabel('Feature Weight')
        ax.set_title('(b) Feature Type Influence on Weight Assignment')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def create_panel_c(self, ax, trajectory_data):
        """Panel (c): Correlation between feature classification confidence and weight adjustment"""
        
        # Calculate overall feature confidence for each scan
        scan_indices = trajectory_data['scan_indices']
        feature_weights = trajectory_data['feature_weights']
        
        overall_confidences = []
        for i in range(len(scan_indices)):
            # Weighted average of feature type confidences
            total_confidence = 0.0
            total_weight = 0.0
            
            for ftype in FeatureType:
                if i < len(trajectory_data['feature_type_confidences'][ftype]):
                    confidence = trajectory_data['feature_type_confidences'][ftype][i]
                    count = trajectory_data['feature_type_counts'][ftype][i]
                    
                    total_confidence += confidence * count
                    total_weight += count
            
            if total_weight > 0:
                overall_confidences.append(total_confidence / total_weight)
            else:
                overall_confidences.append(0.0)
        
        # Create scatter plot with color coding by environment type
        env_colors = {
            'structured': 'blue',
            'corridor': 'green', 
            'sparse': 'red',
            'complex': 'orange'
        }
        
        for env_type in env_colors.keys():
            env_confidences = []
            env_weights = []
            
            for i, env in enumerate(trajectory_data['environment_classifications']):
                if env == env_type and i < len(overall_confidences):
                    env_confidences.append(overall_confidences[i])
                    env_weights.append(feature_weights[i])
            
            if env_confidences:
                ax.scatter(env_confidences, env_weights, 
                          color=env_colors[env_type], 
                          label=env_type.capitalize(),
                          alpha=0.7, s=25)
        
        # Add correlation trend line
        if overall_confidences:
            slope, intercept, r_value, p_value, std_err = stats.linregress(overall_confidences, feature_weights)
            x_trend = np.array([min(overall_confidences), max(overall_confidences)])
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'k--', alpha=0.8, linewidth=2)
            
            # Add correlation coefficient
            ax.text(0.05, 0.95, f'Correlation: r = {r_value:.3f}', 
                   transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Feature Classification Confidence')
        ax.set_ylabel('Feature Weight')
        ax.set_title('(c) Confidence vs Weight Adjustment Correlation')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

    def create_panel_d(self, ax, trajectory_data):
        """Panel (d): Fusion confidence scores with feature-specific contributions"""
        
        scan_indices = trajectory_data['scan_indices']
        fusion_confidences = trajectory_data['fusion_confidences']
        
        # Plot overall fusion confidence
        ax.plot(scan_indices, fusion_confidences, 'k-', linewidth=3, label='Overall Fusion Confidence')
        
        # Plot feature-type specific contributions (stacked area)
        bottom = np.zeros(len(scan_indices))
        
        for ftype in FeatureType:
            contributions = trajectory_data['feature_type_contributions'][ftype]
            
            if contributions:
                # Scale contributions by fusion confidence
                scaled_contributions = [conf * contrib for conf, contrib in 
                                      zip(fusion_confidences, contributions)]
                
                ax.fill_between(scan_indices, bottom, 
                               np.array(bottom) + np.array(scaled_contributions),
                               color=self.feature_colors[ftype], 
                               alpha=0.6, 
                               label=f'{self.feature_labels[ftype]} Contribution')
                
                bottom = np.array(bottom) + np.array(scaled_contributions)
        
        # Add confidence statistics
        avg_confidence = np.mean(fusion_confidences)
        std_confidence = np.std(fusion_confidences)
        
        stats_text = f'Average Confidence: {avg_confidence:.3f}±{std_confidence:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               verticalalignment='top')
        
        ax.set_xlabel('Scan Index')
        ax.set_ylabel('Fusion Confidence Score')
        ax.set_title('(d) Fusion Confidence with Feature-Specific Contributions')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def generate_figure6(self, file_path, output_path='figure6_adaptive_weights.png', 
                        max_entries=200, figsize=(16, 12)):
        """
        Generate Figure 6: Adaptive Weight Behavior Based on Feature Analysis
        """
        print("Generating Figure 6: Adaptive Weight Behavior Based on Feature Analysis")
        print(f"Data file: {file_path}")
        
        # Simulate adaptive fusion trajectory
        trajectory_data = self.simulate_adaptive_fusion_trajectory(file_path, max_entries)
        
        if not trajectory_data['scan_indices']:
            raise ValueError("No scans processed successfully")
        
        # Create figure with four panels (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Create each panel
        print("\nGenerating adaptive fusion analysis panels...")
        self.create_panel_a(axes[0, 0], trajectory_data)
        self.create_panel_b(axes[0, 1], trajectory_data)
        self.create_panel_c(axes[1, 0], trajectory_data)
        self.create_panel_d(axes[1, 1], trajectory_data)
        
        # Add overall title
        fig.suptitle('Figure 6: Adaptive Weight Behavior Based on Feature Analysis', 
                    fontsize=14, y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.5)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\nFigure 6 saved to: {output_path}")
        
        # Print summary statistics
        print(f"\nFigure 6 Summary:")
        print(f"  Trajectory scans analyzed: {len(trajectory_data['scan_indices'])}")
        
        feature_weights = trajectory_data['feature_weights']
        icp_weights = trajectory_data['icp_weights']
        fusion_confidences = trajectory_data['fusion_confidences']
        
        print(f"  Feature weight range: {min(feature_weights):.3f} - {max(feature_weights):.3f}")
        print(f"  Average feature weight: {np.mean(feature_weights):.3f}")
        print(f"  Average fusion confidence: {np.mean(fusion_confidences):.3f}")
        
        # Count environment types
        env_counts = {}
        for env in trajectory_data['environment_classifications']:
            env_counts[env] = env_counts.get(env, 0) + 1
        
        print(f"  Environment distribution:")
        for env_type, count in env_counts.items():
            percentage = count / len(trajectory_data['environment_classifications']) * 100
            print(f"    {env_type}: {count} scans ({percentage:.1f}%)")
        
        plt.show()
        
        return fig

def main():
    """
    Main function to generate Figure 6
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Figure 6: Adaptive Weight Behavior')
    parser.add_argument('--file', type=str, 
                       default="../../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--output', type=str, default='figure6_adaptive_weights.png',
                       help='Output filename for Figure 6')
    parser.add_argument('--max_entries', type=int, default=200,
                       help='Maximum number of entries to analyze')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Create generator and generate figure
    generator = Figure6Generator(debug_level=1)
    
    try:
        generator.generate_figure6(
            file_path=args.file,
            output_path=args.output,
            max_entries=args.max_entries
        )
        print("\n✓ Figure 6 generated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error generating Figure 6: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()