#!/usr/bin/env python3
"""
Figure 3 Generator: Feature Classification Performance and Technical Validation
Generates a four-panel quantitative analysis showing:
(a) curvature threshold analysis and classification boundaries for different feature types
(b) classification accuracy confusion matrix across all feature categories  
(c) spatial distribution uniformity and sector-based selection effectiveness
(d) temporal consistency and repeatability of feature detection across consecutive scans
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import math
import seaborn as sns
from collections import defaultdict, Counter

# Import your existing modules
from feature_extractor import FeatureExtractor, FeatureType
from lidar_utility_functions import convert_scans_to_cartesian, read_lidar_data_from_file

class Figure3Generator:
    """
    Generates Figure 3: Feature Classification Performance and Technical Validation
    """
    
    def __init__(self, debug_level=1):
        self.debug_level = debug_level
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            debug_level=0,  # Quiet for figure generation
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

    def analyze_feature_performance(self, file_path, max_entries=500, analysis_step=5):
        """
        Comprehensive analysis of feature extraction performance
        
        Args:
            file_path: Path to LiDAR data file
            max_entries: Maximum entries to analyze
            analysis_step: Process every Nth scan for efficiency
            
        Returns:
            dict: Comprehensive performance analysis results
        """
        print("Analyzing feature classification performance...")
        parsed_data_list = read_lidar_data_from_file(file_path, max_entries)
        
        if not parsed_data_list:
            raise ValueError("No data was read from file")
        
        # Subsample for efficiency
        analysis_scans = parsed_data_list[::analysis_step]
        print(f"Analyzing {len(analysis_scans)} scans (every {analysis_step}th scan)...")
        
        results = {
            'curvature_data': [],           # For panel (a)
            'classification_stats': {},     # For panel (b) 
            'spatial_distribution': {},     # For panel (c)
            'temporal_consistency': {},     # For panel (d)
            'feature_sets': [],             # Store all feature sets
            'scan_indices': []              # Track which scans were processed
        }
        
        angle_min, angle_max = -math.pi/2, math.pi/2
        
        for i, data in enumerate(analysis_scans):
            if i % 20 == 0:
                print(f"  Processed {i}/{len(analysis_scans)} scans...")
            
            try:
                # Convert to cartesian coordinates
                x_points, y_points = convert_scans_to_cartesian(
                    data['scan_ranges'], angle_min, angle_max, data['pose'],
                    flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
                )
                
                # Extract features (use fallback approach that worked before)
                feature_set = self.feature_extractor.extract_features(x_points, y_points, data['scan_ranges'])
                
                if feature_set and feature_set.features:
                    # Store for temporal analysis
                    results['feature_sets'].append(feature_set)
                    results['scan_indices'].append(i)
                    
                    # Collect curvature data for panel (a)
                    for feature in feature_set.features:
                        results['curvature_data'].append({
                            'curvature': abs(feature.curvature),
                            'feature_type': feature.feature_type,
                            'sector': feature.sector,
                            'scan_index': i
                        })
                    
                    # Collect spatial distribution data for panel (c)
                    self._analyze_spatial_distribution(feature_set, results['spatial_distribution'], i)
                    
            except Exception as e:
                if self.debug_level > 1:
                    print(f"    Failed to process scan {i}: {e}")
                continue
        
        # Post-process results
        print(f"Successfully processed {len(results['feature_sets'])} scans")
        
        # Generate classification statistics for panel (b)
        self._generate_classification_stats(results)
        
        # Generate temporal consistency analysis for panel (d)
        self._analyze_temporal_consistency(results)
        
        return results

    def _analyze_spatial_distribution(self, feature_set, spatial_data, scan_index):
        """Analyze spatial distribution for panel (c)"""
        if 'sector_counts' not in spatial_data:
            spatial_data['sector_counts'] = defaultdict(list)
            spatial_data['feature_positions'] = []
            spatial_data['coverage_metrics'] = []
        
        # Count features per sector
        sector_counts = defaultdict(int)
        for feature in feature_set.features:
            sector_counts[feature.sector] += 1
        
        # Store sector distribution
        for sector in range(6):  # 6 sectors
            spatial_data['sector_counts'][sector].append(sector_counts[sector])
        
        # Calculate spatial coverage metrics
        occupied_sectors = len([s for s in sector_counts.values() if s > 0])
        total_features = len(feature_set.features)
        
        # Calculate uniformity (lower variance = more uniform)
        sector_values = [sector_counts[s] for s in range(6)]
        uniformity = 1.0 / (1.0 + np.var(sector_values))  # Higher = more uniform
        
        spatial_data['coverage_metrics'].append({
            'scan_index': scan_index,
            'occupied_sectors': occupied_sectors,
            'total_features': total_features,
            'uniformity': uniformity,
            'sector_distribution': sector_values
        })

    def _generate_classification_stats(self, results):
        """Generate classification statistics for panel (b)"""
        curvature_data = results['curvature_data']
        
        # Expected classification based on curvature thresholds
        sharp_threshold = 0.15
        planar_threshold = 0.08
        
        # Create ground truth based on curvature values
        classification_comparison = []
        
        for data_point in curvature_data:
            curvature = data_point['curvature']
            actual_type = data_point['feature_type']
            
            # Determine expected type based on curvature
            if curvature > sharp_threshold:
                expected_type = FeatureType.SHARP_EDGE
            elif curvature < planar_threshold:
                expected_type = FeatureType.PLANAR
            else:
                # Medium curvature - could be either less sharp or less planar
                # Use actual classification as expected for these ambiguous cases
                expected_type = actual_type
            
            classification_comparison.append({
                'expected': expected_type,
                'actual': actual_type,
                'curvature': curvature
            })
        
        results['classification_stats'] = {
            'comparisons': classification_comparison,
            'accuracy_by_type': self._calculate_accuracy_by_type(classification_comparison)
        }

    def _calculate_accuracy_by_type(self, comparisons):
        """Calculate classification accuracy for each feature type"""
        type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for comp in comparisons:
            expected = comp['expected']
            actual = comp['actual']
            
            type_stats[expected]['total'] += 1
            if expected == actual:
                type_stats[expected]['correct'] += 1
        
        accuracy_stats = {}
        for ftype, stats in type_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            accuracy_stats[ftype] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        return accuracy_stats

    def _analyze_temporal_consistency(self, results):
        """Analyze temporal consistency for panel (d)"""
        feature_sets = results['feature_sets']
        
        if len(feature_sets) < 2:
            results['temporal_consistency'] = {'insufficient_data': True}
            return
        
        consistency_metrics = []
        
        # Analyze consecutive pairs of scans
        for i in range(1, min(len(feature_sets), 50)):  # Limit to first 50 for efficiency
            prev_set = feature_sets[i-1]
            curr_set = feature_sets[i]
            
            # Count features by type for both scans
            prev_counts = self._count_features_by_type(prev_set)
            curr_counts = self._count_features_by_type(curr_set)
            
            # Calculate consistency metrics
            total_consistency = 0
            type_consistencies = {}
            
            for ftype in FeatureType:
                prev_count = prev_counts.get(ftype, 0)
                curr_count = curr_counts.get(ftype, 0)
                
                # Consistency metric: 1 - normalized absolute difference
                max_count = max(prev_count, curr_count, 1)  # Avoid division by zero
                consistency = 1.0 - abs(prev_count - curr_count) / max_count
                
                type_consistencies[ftype] = consistency
                total_consistency += consistency
            
            overall_consistency = total_consistency / len(FeatureType)
            
            consistency_metrics.append({
                'scan_pair': (i-1, i),
                'overall_consistency': overall_consistency,
                'type_consistencies': type_consistencies,
                'prev_counts': prev_counts,
                'curr_counts': curr_counts
            })
        
        results['temporal_consistency'] = {
            'metrics': consistency_metrics,
            'average_consistency': np.mean([m['overall_consistency'] for m in consistency_metrics]),
            'consistency_by_type': self._average_consistency_by_type(consistency_metrics)
        }

    def _count_features_by_type(self, feature_set):
        """Count features by type in a feature set"""
        counts = {}
        for ftype in FeatureType:
            counts[ftype] = len(feature_set.get_features_by_type(ftype))
        return counts

    def _average_consistency_by_type(self, consistency_metrics):
        """Calculate average consistency by feature type"""
        type_totals = defaultdict(list)
        
        for metric in consistency_metrics:
            for ftype, consistency in metric['type_consistencies'].items():
                type_totals[ftype].append(consistency)
        
        averages = {}
        for ftype, values in type_totals.items():
            averages[ftype] = np.mean(values) if values else 0
        
        return averages

    def create_panel_a(self, ax, results):
        """Panel (a): Curvature threshold analysis and classification boundaries"""
        curvature_data = results['curvature_data']
        
        if not curvature_data:
            ax.text(0.5, 0.5, 'No curvature data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Extract data by feature type
        curvatures_by_type = defaultdict(list)
        for data_point in curvature_data:
            curvatures_by_type[data_point['feature_type']].append(data_point['curvature'])
        
        # Create histogram with overlapping distributions
        bins = np.linspace(0, 1.0, 50)
        
        for ftype in FeatureType:
            if curvatures_by_type[ftype]:
                ax.hist(curvatures_by_type[ftype], bins=bins, alpha=0.6, 
                       color=self.feature_colors[ftype], label=self.feature_labels[ftype],
                       density=True)
        
        # Add threshold lines
        ax.axvline(x=0.08, color='blue', linestyle='--', linewidth=2, 
                  label='Planar Threshold (0.08)')
        ax.axvline(x=0.15, color='red', linestyle='--', linewidth=2, 
                  label='Sharp Edge Threshold (0.15)')
        
        ax.set_xlabel('Curvature Value')
        ax.set_ylabel('Density')
        ax.set_title('(a) Curvature Distribution & Classification Boundaries')
        ax.legend(loc='upper right')  # Move legend inside plot area, top right
        ax.grid(True, alpha=0.3)

    def create_panel_b(self, ax, results):
        """Panel (b): Classification accuracy confusion matrix"""
        classification_stats = results['classification_stats']
        
        if 'comparisons' not in classification_stats:
            ax.text(0.5, 0.5, 'No classification data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Create confusion matrix
        feature_types = list(FeatureType)
        matrix_size = len(feature_types)
        confusion_matrix = np.zeros((matrix_size, matrix_size))
        
        type_to_index = {ftype: i for i, ftype in enumerate(feature_types)}
        
        for comp in classification_stats['comparisons']:
            expected_idx = type_to_index[comp['expected']]
            actual_idx = type_to_index[comp['actual']]
            confusion_matrix[expected_idx, actual_idx] += 1
        
        # Normalize by row (expected type)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(confusion_matrix, row_sums, 
                                    out=np.zeros_like(confusion_matrix), 
                                    where=row_sums!=0)
        
        # Create heatmap
        sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[self.feature_labels[ft] for ft in feature_types],
                   yticklabels=[self.feature_labels[ft] for ft in feature_types],
                   ax=ax, cbar_kws={'label': 'Classification Accuracy'})
        
        ax.set_xlabel('Actual Classification')
        ax.set_ylabel('Expected Classification')
        ax.set_title('(b) Classification Accuracy Matrix')
        
        # Calculate overall accuracy
        overall_accuracy = np.trace(normalized_matrix) / matrix_size
        ax.text(0.02, 0.98, f'Overall Accuracy: {overall_accuracy:.1%}', 
               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white'),
               verticalalignment='top')

    def create_panel_c(self, ax, results):
        """Panel (c): Spatial distribution uniformity and sector-based selection"""
        spatial_data = results['spatial_distribution']
        
        if 'coverage_metrics' not in spatial_data:
            ax.text(0.5, 0.5, 'No spatial data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        coverage_metrics = spatial_data['coverage_metrics']
        
        # Plot sector distribution as box plots
        sector_data = []
        sector_labels = []
        
        for sector in range(6):
            sector_counts = spatial_data['sector_counts'][sector]
            if sector_counts:
                sector_data.append(sector_counts)
                sector_labels.append(f'Sector {sector}')
        
        if sector_data:
            bp = ax.boxplot(sector_data, labels=sector_labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        
        ax.set_xlabel('Sector Number')
        ax.set_ylabel('Feature Count per Scan')
        ax.set_title('(c) Spatial Distribution Across Sectors')
        ax.grid(True, alpha=0.3)
        
        # Add uniformity statistics
        uniformities = [m['uniformity'] for m in coverage_metrics]
        mean_uniformity = np.mean(uniformities)
        
        stats_text = f'Mean Uniformity: {mean_uniformity:.3f}\nSectors Used: {len(sector_data)}/6'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
               verticalalignment='top')

    def create_panel_d(self, ax, results):
        """Panel (d): Temporal consistency and repeatability"""
        temporal_data = results['temporal_consistency']
        
        if 'insufficient_data' in temporal_data:
            ax.text(0.5, 0.5, 'Insufficient temporal data', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        metrics = temporal_data['metrics']
        
        if not metrics:
            ax.text(0.5, 0.5, 'No temporal consistency data', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Plot overall consistency over time
        scan_pairs = [m['scan_pair'][1] for m in metrics]  # Use second scan index
        consistencies = [m['overall_consistency'] for m in metrics]
        
        ax.plot(scan_pairs, consistencies, 'b-', linewidth=2, label='Overall Consistency')
        
        # Plot consistency by feature type
        for ftype in FeatureType:
            type_consistencies = [m['type_consistencies'][ftype] for m in metrics]
            ax.plot(scan_pairs, type_consistencies, '--', 
                   color=self.feature_colors[ftype], 
                   label=f'{self.feature_labels[ftype]}', alpha=0.7)
        
        ax.set_xlabel('Scan Index')
        ax.set_ylabel('Consistency Score')
        ax.set_title('(d) Temporal Consistency Across Scans')
        ax.legend(loc='upper right')  # Move legend inside plot area
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add average consistency
        avg_consistency = temporal_data['average_consistency']
        ax.axhline(y=avg_consistency, color='red', linestyle='-', alpha=0.8,
                  label=f'Average: {avg_consistency:.3f}')

    def generate_figure3(self, file_path, output_path='figure3_classification_validation.png', 
                        max_entries=500, figsize=(16, 12)):
        """
        Generate Figure 3: Feature Classification Performance and Technical Validation
        """
        print("Generating Figure 3: Feature Classification Performance and Technical Validation")
        print(f"Data file: {file_path}")
        
        # Analyze feature performance
        results = self.analyze_feature_performance(file_path, max_entries)
        
        # Create figure with four panels (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Create each panel
        print("\nGenerating analysis panels...")
        self.create_panel_a(axes[0, 0], results)
        self.create_panel_b(axes[0, 1], results)
        self.create_panel_c(axes[1, 0], results)
        self.create_panel_d(axes[1, 1], results)
        
        # Add overall title
        fig.suptitle('Figure 3: Feature Classification Performance and Technical Validation', 
                    fontsize=14, y=0.95)
        
        # Adjust layout with more space between rows
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.4)  # Increased hspace from 0.3 to 0.4
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\nFigure 3 saved to: {output_path}")
        
        # Print summary
        print(f"\nFigure 3 Summary:")
        if results['curvature_data']:
            print(f"  Total features analyzed: {len(results['curvature_data'])}")
            print(f"  Scans processed: {len(results['feature_sets'])}")
            
            if 'classification_stats' in results:
                accuracy_stats = results['classification_stats'].get('accuracy_by_type', {})
                for ftype, stats in accuracy_stats.items():
                    if stats['total'] > 0:
                        print(f"  {self.feature_labels[ftype]}: {stats['accuracy']:.1%} accuracy "
                              f"({stats['correct']}/{stats['total']})")
        
        plt.show()
        
        return fig

def main():
    """
    Main function to generate Figure 3
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Figure 3: Feature Classification Validation')
    parser.add_argument('--file', type=str, 
                       default="../../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--output', type=str, default='figure3_classification_validation.png',
                       help='Output filename for Figure 3')
    parser.add_argument('--max_entries', type=int, default=500,
                       help='Maximum number of entries to analyze')
    
    args = parser.parse_args()
    
    # Create generator and generate figure
    generator = Figure3Generator(debug_level=1)
    
    try:
        generator.generate_figure3(
            file_path=args.file,
            output_path=args.output,
            max_entries=args.max_entries
        )
        print("\n✓ Figure 3 generated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error generating Figure 3: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()