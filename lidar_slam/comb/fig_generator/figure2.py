#!/usr/bin/env python3
"""
Figure 2 Generator: Progressive Feature Analysis and Classification
Generates a three-panel demonstration of curvature-based feature extraction:
(a) Raw LiDAR scan with complete feature extraction overlay
(b) Same scan with features color-coded by classification type  
(c) Quality-weighted visualization with features sized by strength/reliability metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import your existing modules
from feature_extractor import FeatureExtractor, FeatureType
from lidar_utility_functions import convert_scans_to_cartesian, read_lidar_data_from_file

# Try to import PoseEstimate with fallback
try:
    from pose_estimate import PoseEstimate
except ImportError:
    print("Warning: Could not import PoseEstimate. Will try to import during execution.")
    PoseEstimate = None

class Figure2Generator:
    """
    Generates Figure 2: Progressive Feature Analysis and Classification
    """
    
    def __init__(self, debug_level=1):
        self.debug_level = debug_level
        
        # Initialize feature extractor with same parameters as your main system
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
        
        # Define colors for different feature types
        self.feature_colors = {
            FeatureType.SHARP_EDGE: '#FF4444',        # Bright red
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

    def select_representative_scan(self, file_path, max_entries=2000, manual_index=None):
        """
        Select a single representative scan with good feature diversity
        
        Args:
            file_path: Path to LiDAR data file
            max_entries: Maximum entries to analyze  
            manual_index: Optional manual scan index
        
        Returns:
            dict: Contains scan data and metadata
        """
        print("Reading LiDAR data to select representative scan...")
        parsed_data_list = read_lidar_data_from_file(file_path, max_entries)
        
        if not parsed_data_list:
            raise ValueError("No data was read from file")
        
        # If manual index provided, use it directly
        if manual_index is not None:
            if 0 <= manual_index < len(parsed_data_list):
                print(f"Using manually specified scan #{manual_index}")
                return {
                    'index': manual_index,
                    'data': parsed_data_list[manual_index],
                    'total_scans': len(parsed_data_list)
                }
            else:
                raise ValueError(f"Manual index {manual_index} is out of range (0-{len(parsed_data_list)-1})")
        
        print(f"Analyzing {len(parsed_data_list)} scans to find representative scan...")
        
        # Analyze scans to find one with good feature diversity
        scan_analysis = []
        angle_min, angle_max = -math.pi/2, math.pi/2
        
        for i, data in enumerate(parsed_data_list):
            # Convert to cartesian for analysis
            x_points, y_points = convert_scans_to_cartesian(
                data['scan_ranges'], angle_min, angle_max, data['pose'],
                flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
            )
            
            # Basic scan quality metrics
            valid_points = [(x, y) for x, y in zip(x_points, y_points) 
                          if not (math.isnan(x) or math.isnan(y))]
            
            if len(valid_points) < 20:
                continue
                
            # Calculate geometric diversity metrics
            ranges = [math.sqrt(x**2 + y**2) for x, y in valid_points]
            range_std = np.std(ranges)
            point_density = len(valid_points)
            
            # Calculate geometric complexity (range variation)
            range_changes = sum(abs(ranges[i] - ranges[i-1]) for i in range(1, len(ranges)))
            complexity = range_changes / len(ranges) if len(ranges) > 1 else 0
            
            # Calculate angular coverage
            angles = [math.atan2(y, x) for x, y in valid_points]
            angular_span = max(angles) - min(angles)
            
            # Combined quality score
            quality_score = (
                min(point_density / 100.0, 1.0) * 0.3 +  # Point density (normalized)
                min(range_std / 2.0, 1.0) * 0.3 +        # Range variation
                min(complexity / 10.0, 1.0) * 0.2 +      # Geometric complexity
                min(angular_span / math.pi, 1.0) * 0.2   # Angular coverage
            )
            
            scan_analysis.append({
                'index': i,
                'data': data,
                'points': valid_points,
                'point_density': point_density,
                'range_std': range_std,
                'complexity': complexity,
                'angular_span': angular_span,
                'quality_score': quality_score
            })
        
        # Select scan with best overall quality score
        if not scan_analysis:
            raise ValueError("No valid scans found for analysis")
        
        best_scan = max(scan_analysis, key=lambda x: x['quality_score'])
        
        print(f"Selected scan #{best_scan['index']} with quality score {best_scan['quality_score']:.3f}")
        print(f"  Point density: {best_scan['point_density']}")
        print(f"  Range variation: {best_scan['range_std']:.2f}")
        print(f"  Geometric complexity: {best_scan['complexity']:.2f}")
        print(f"  Angular span: {best_scan['angular_span']:.2f} rad")
        
        return {
            'index': best_scan['index'],
            'data': best_scan['data'],
            'total_scans': len(parsed_data_list),
            'quality_metrics': {
                'point_density': best_scan['point_density'],
                'range_std': best_scan['range_std'],
                'complexity': best_scan['complexity'],
                'angular_span': best_scan['angular_span'],
                'quality_score': best_scan['quality_score']
            }
        }

    def extract_features_for_scan(self, scan_data):
        """Extract features from the selected scan"""
        angle_min, angle_max = -math.pi/2, math.pi/2
        
        # Convert to cartesian coordinates
        x_points, y_points = convert_scans_to_cartesian(
            scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
        )
        
        # Create scan points list compatible with your feature extractor
        scan_points = []
        scan_ranges = scan_data['scan_ranges']
        
        for i, (x, y, range_val) in enumerate(zip(x_points, y_points, scan_ranges)):
            if not (math.isnan(x) or math.isnan(y) or math.isnan(range_val)):
                scan_points.append((x, y, i, range_val))
        
        if len(scan_points) < 10:
            print(f"Warning: Only {len(scan_points)} valid points for feature extraction")
        
        # Extract features using your existing feature extractor
        try:
            if self.debug_level > 0:
                print(f"  Attempting feature extraction on {len(scan_points)} points...")
            
            # Create pose estimate object - try multiple approaches
            pose = None
            try:
                # Try importing and creating PoseEstimate
                if PoseEstimate is None:
                    from pose_estimate import PoseEstimate
                pose = PoseEstimate(
                    scan_data['pose']['x'],
                    scan_data['pose']['y'], 
                    scan_data['pose']['theta']
                )
                if self.debug_level > 1:
                    print(f"  Created PoseEstimate object successfully")
            except (ImportError, NameError, AttributeError) as e:
                if self.debug_level > 0:
                    print(f"  Warning: Could not create PoseEstimate: {e}")
                    print("  Trying with pose dictionary...")
                pose = scan_data['pose']  # Use dict as fallback
            except Exception as e:
                if self.debug_level > 0:
                    print(f"  Warning: Unexpected error with PoseEstimate: {e}")
                    print("  Trying with None pose...")
                pose = None  # Last resort fallback
            
            # Use original scan data (not filtered scan_points)
            angle_min, angle_max = -math.pi/2, math.pi/2
            scan_x, scan_y = convert_scans_to_cartesian(
                scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
                flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
            )
            scan_ranges = scan_data['scan_ranges']  # Use original ranges
            
            if self.debug_level > 1:
                print(f"  Prepared data: {len(scan_x)} x coords, {len(scan_y)} y coords, {len(scan_ranges)} ranges")
                print(f"  Pose type: {type(pose)}")
                if hasattr(pose, 'x'):
                    print(f"  Pose: x={pose.x:.3f}, y={pose.y:.3f}, theta={pose.theta:.3f}")
                elif isinstance(pose, dict):
                    print(f"  Pose dict: x={pose['x']:.3f}, y={pose['y']:.3f}, theta={pose['theta']:.3f}")
                else:
                    print(f"  Pose: {pose}")
            
            # Try different parameter combinations for feature extraction
            feature_set = None
            try:
                # First try with all parameters
                feature_set = self.feature_extractor.extract_features(
                    scan_x, scan_y, scan_ranges, 
                    robot_pose=pose, 
                    scan_timestamp=scan_data.get('timestamp', 0.0)
                )
            except Exception as e1:
                if self.debug_level > 0:
                    print(f"  Failed with full parameters: {e1}")
                    print("  Trying without timestamp...")
                try:
                    # Try without timestamp
                    feature_set = self.feature_extractor.extract_features(
                        scan_x, scan_y, scan_ranges, robot_pose=pose
                    )
                except Exception as e2:
                    if self.debug_level > 0:
                        print(f"  Failed without timestamp: {e2}")
                        print("  Trying without pose...")
                    # Try without pose
                    feature_set = self.feature_extractor.extract_features(
                        scan_x, scan_y, scan_ranges
                    )
            
            if self.debug_level > 0:
                features_count = len(feature_set.features) if feature_set and feature_set.features else 0
                print(f"  Success: Extracted {features_count} features")
            
            return scan_points, feature_set
            
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            print(f"Debug info:")
            print(f"  scan_x length: {len(scan_x) if 'scan_x' in locals() else 'N/A'}")
            print(f"  scan_y length: {len(scan_y) if 'scan_y' in locals() else 'N/A'}")
            print(f"  scan_ranges length: {len(scan_ranges) if 'scan_ranges' in locals() else 'N/A'}")
            print(f"  pose type: {type(pose) if 'pose' in locals() else 'N/A'}")
            print(f"  original scan_ranges length: {len(scan_data['scan_ranges'])}")
            
            import traceback
            traceback.print_exc()
            
            # Return empty feature set if extraction fails
            from feature_extractor import FeatureSet
            empty_feature_set = FeatureSet()
            return scan_points, empty_feature_set

    def create_panel_a(self, ax, scan_points, feature_set):
        """Panel A: Raw LiDAR scan with complete feature extraction overlay"""
        
        # Plot all scan points in gray
        if scan_points:
            scan_x = [p[0] for p in scan_points]
            scan_y = [p[1] for p in scan_points]
            ax.scatter(scan_x, scan_y, c='lightgray', s=12, alpha=0.7, label='LiDAR Points')
        
        # Plot ALL features as simple red dots (no classification)
        all_features = feature_set.features
        if all_features:
            feature_x = [f.point_world[0] for f in all_features]
            feature_y = [f.point_world[1] for f in all_features]
            ax.scatter(feature_x, feature_y, c='red', s=25, alpha=0.8, 
                      label=f'Detected Features ({len(all_features)})',
                      edgecolors='darkred', linewidth=0.5, zorder=5)
        
        # Add robot position
        ax.scatter(0, 0, c='blue', marker='*', s=200, label='Robot', 
                  edgecolors='black', linewidth=1, zorder=10)
        
        # Formatting
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('(a) Raw Scan + Feature Detection\nComplete extraction overlay', 
                    fontsize=11, pad=10)
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
        
        return len(all_features)

    def create_panel_b(self, ax, scan_points, feature_set):
        """Panel B: Same scan with features color-coded by classification type"""
        
        # Plot all scan points in light gray
        if scan_points:
            scan_x = [p[0] for p in scan_points]
            scan_y = [p[1] for p in scan_points]
            ax.scatter(scan_x, scan_y, c='lightgray', s=8, alpha=0.5, label='LiDAR Points')
        
        # Plot features by type with color coding
        feature_counts = {}
        total_features = 0
        
        for feature_type in [FeatureType.SHARP_EDGE, FeatureType.LESS_SHARP_EDGE, 
                           FeatureType.PLANAR, FeatureType.LESS_PLANAR]:
            
            type_features = feature_set.get_features_by_type(feature_type)
            feature_counts[feature_type] = len(type_features)
            total_features += len(type_features)
            
            if type_features:
                feature_x = [f.point_world[0] for f in type_features]
                feature_y = [f.point_world[1] for f in type_features]
                
                # Different markers for different feature types
                markers = {
                    FeatureType.SHARP_EDGE: '^',         # Triangle up
                    FeatureType.LESS_SHARP_EDGE: 's',    # Square  
                    FeatureType.PLANAR: 'o',             # Circle
                    FeatureType.LESS_PLANAR: 'D'         # Diamond
                }
                
                ax.scatter(feature_x, feature_y, 
                          c=self.feature_colors[feature_type],
                          marker=markers[feature_type],
                          s=40, 
                          label=f'{self.feature_labels[feature_type]} ({len(type_features)})',
                          edgecolors='black',
                          linewidth=0.3,
                          alpha=0.8,
                          zorder=5)
        
        # Add robot position
        ax.scatter(0, 0, c='blue', marker='*', s=200, label='Robot', 
                  edgecolors='black', linewidth=1, zorder=10)
        
        # Formatting
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('(b) Feature Classification\nColor-coded by geometric type', 
                    fontsize=11, pad=10)
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
        
        return feature_counts

    def create_panel_c(self, ax, scan_points, feature_set):
        """Panel C: Quality-weighted visualization with features sized by strength"""
        
        # Initialize qualities list
        qualities = []
        
        # Plot all scan points in light gray
        if scan_points:
            scan_x = [p[0] for p in scan_points]
            scan_y = [p[1] for p in scan_points]
            ax.scatter(scan_x, scan_y, c='lightgray', s=6, alpha=0.4, label='LiDAR Points')
        
        # Plot features sized by quality/strength
        all_features = feature_set.features
        if all_features:
            feature_x = [f.point_world[0] for f in all_features]
            feature_y = [f.point_world[1] for f in all_features]
            
            # Get quality metrics (strength, curvature, or distance-based)
            colors = []
            for f in all_features:
                # Use strength if available, otherwise use curvature
                if hasattr(f, 'strength') and f.strength > 0:
                    quality = f.strength
                else:
                    quality = abs(f.curvature) if hasattr(f, 'curvature') else 0.5
                qualities.append(quality)
                colors.append(self.feature_colors[f.feature_type])
            
            # Normalize qualities to reasonable size range
            if qualities:
                min_quality = min(qualities)
                max_quality = max(qualities)
                if max_quality > min_quality:
                    normalized_qualities = [(q - min_quality) / (max_quality - min_quality) 
                                          for q in qualities]
                else:
                    normalized_qualities = [0.5] * len(qualities)
                
                # Map to size range (20-100)
                sizes = [20 + 80 * nq for nq in normalized_qualities]
            else:
                sizes = [40] * len(all_features)
            
            # Create scatter plot with varying sizes
            scatter = ax.scatter(feature_x, feature_y, 
                               c=colors, s=sizes,
                               alpha=0.7, edgecolors='black', linewidth=0.3, zorder=5)
        
        # Add robot position
        ax.scatter(0, 0, c='blue', marker='*', s=200, label='Robot', 
                  edgecolors='black', linewidth=1, zorder=10)
        
        # Formatting
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('(c) Quality Assessment\nFeature size = reliability/strength', 
                    fontsize=11, pad=10)
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
        
        # Add quality legend only if we have features and qualities
        if all_features and qualities:
            quality_text = f'Quality Range:\nMin: {min(qualities):.3f}\nMax: {max(qualities):.3f}'
            ax.text(0.02, 0.98, quality_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=9)
        elif not all_features:
            # Add a note if no features were extracted
            ax.text(0.5, 0.5, 'No features extracted\nfrom this scan', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   fontsize=12)
        
        return qualities

    def set_common_axis_limits(self, axes, scan_points):
        """Set common axis limits for all panels"""
        if not scan_points:
            return
        
        scan_x = [p[0] for p in scan_points]
        scan_y = [p[1] for p in scan_points]
        x_min, x_max = min(scan_x), max(scan_x)
        y_min, y_max = min(scan_y), max(scan_y)
        
        # Add padding
        x_padding = max(0.5, (x_max - x_min) * 0.1)
        y_padding = max(0.5, (y_max - y_min) * 0.1)
        
        xlim = (x_min - x_padding, x_max + x_padding)
        ylim = (y_min - y_padding, y_max + y_padding)
        
        for ax in axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    def generate_figure2(self, file_path, output_path='figure2_progressive_analysis.png', 
                        max_entries=2000, figsize=(18, 6), manual_index=None):
        """
        Generate Figure 2: Progressive Feature Analysis and Classification
        
        Args:
            manual_index: Optional manual scan index to use
        """
        print("Generating Figure 2: Progressive Feature Analysis and Classification")
        print(f"Data file: {file_path}")
        
        # Select representative scan
        scan_info = self.select_representative_scan(file_path, max_entries, manual_index)
        scan_data = scan_info['data']
        
        print(f"\nProcessing scan #{scan_info['index']} for progressive analysis...")
        
        # Extract features
        scan_points, feature_set = self.extract_features_for_scan(scan_data)
        
        # Debug feature extraction results
        print(f"  Scan points: {len(scan_points)}")
        print(f"  Features extracted: {len(feature_set.features) if feature_set.features else 0}")
        
        if not feature_set.features:
            print("Warning: No features extracted from selected scan!")
            print("This could indicate:")
            print("  - Feature extractor interface mismatch")
            print("  - Insufficient geometric complexity in selected scan")
            print("  - Feature extraction thresholds too strict")
            print("Try using a different scan index with --manual_index")
        else:
            # Print feature breakdown
            feature_counts = feature_set.get_feature_count_by_type()
            for ftype, count in feature_counts.items():
                if count > 0:
                    print(f"    {ftype}: {count}")
        
        # Create figure with three panels
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Create each panel
        total_features_a = self.create_panel_a(axes[0], scan_points, feature_set)
        feature_counts_b = self.create_panel_b(axes[1], scan_points, feature_set)
        qualities_c = self.create_panel_c(axes[2], scan_points, feature_set)
        
        # Set common axis limits
        self.set_common_axis_limits(axes, scan_points)
        
        # Add overall title
        fig.suptitle('Figure 2: Progressive Feature Analysis and Classification', 
                    fontsize=14, y=0.95)
        
        # Create unified legend for panel B (most informative)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, -0.02),
                  ncol=len(handles), 
                  fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.88)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\nFigure 2 saved to: {output_path}")
        
        # Print summary
        print(f"\nFigure 2 Summary:")
        print(f"  Scan: #{scan_info['index']} from {scan_info['total_scans']} total scans")
        print(f"  Total features: {len(feature_set.features)}")
        for ftype, count in feature_counts_b.items():
            if count > 0:
                print(f"    {self.feature_labels[ftype]}: {count}")
        
        if qualities_c:
            print(f"  Quality range: {min(qualities_c):.3f} - {max(qualities_c):.3f}")
        
        plt.show()
        
        return fig

def main():
    """
    Main function to generate Figure 2
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Figure 2: Progressive Feature Analysis')
    parser.add_argument('--file', type=str, 
                       default="../../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--output', type=str, default='figure2_progressive_analysis.png',
                       help='Output filename for Figure 2')
    parser.add_argument('--max_entries', type=int, default=2000,
                       help='Maximum number of entries to analyze')
    parser.add_argument('--manual_index', type=int, default=None,
                       help='Manual scan index to use (overrides automatic selection)')
    
    args = parser.parse_args()
    
    # Create generator and generate figure
    generator = Figure2Generator(debug_level=1)
    
    try:
        generator.generate_figure2(
            file_path=args.file,
            output_path=args.output,
            max_entries=args.max_entries,
            manual_index=args.manual_index
        )
        print("\n✓ Figure 2 generated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error generating Figure 2: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()