#!/usr/bin/env python3
"""
Figure 1 Generator: Dataset Geometric Diversity Overview
Generates a three-panel figure showing representative scans with varying geometric complexity:
(a) scan with abundant sharp edges and corners
(b) scan dominated by planar surfaces and smooth curves  
(c) scan with mixed geometric features demonstrating classification challenges
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import math

# Import your existing modules
from feature_extractor import FeatureExtractor, FeatureType
from lidar_utility_functions import convert_scans_to_cartesian, read_lidar_data_from_file

class Figure1Generator:
    """
    Generates Figure 1: Dataset Geometric Diversity Overview
    """
    
    def __init__(self, debug_level=1):
        self.debug_level = debug_level
        
        # Initialize feature extractor for geometric analysis
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

    def analyze_geometric_complexity(self, file_path, max_entries=2000):
        """
        Analyze all scans to classify them by geometric complexity
        
        Args:
            file_path: Path to LiDAR data file
            max_entries: Maximum entries to analyze
        
        Returns:
            dict: Classified scans by complexity type
        """
        print("Reading LiDAR data for geometric complexity analysis...")
        parsed_data_list = read_lidar_data_from_file(file_path, max_entries)
        
        if not parsed_data_list:
            raise ValueError("No data was read from file")
        
        print(f"Analyzing geometric complexity of {len(parsed_data_list)} scans...")
        
        # Analyze each scan for geometric characteristics
        scan_analysis = []
        angle_min, angle_max = -math.pi/2, math.pi/2
        
        for i, data in enumerate(parsed_data_list):
            if i % 200 == 0:  # Progress indicator
                print(f"  Processed {i}/{len(parsed_data_list)} scans...")
            
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
            
            # Calculate geometric complexity metrics
            ranges = [math.sqrt(x**2 + y**2) for x, y in valid_points]
            
            # Range variation (higher = more geometric complexity)
            range_std = np.std(ranges)
            range_changes = sum(abs(ranges[j] - ranges[j-1]) for j in range(1, len(ranges)))
            geometric_complexity = range_changes / len(ranges) if len(ranges) > 1 else 0
            
            # Distance distribution analysis  
            short_range_count = sum(1 for r in ranges if r < 2.0)
            medium_range_count = sum(1 for r in ranges if 2.0 <= r < 5.0)
            long_range_count = sum(1 for r in ranges if r >= 5.0)
            
            # Calculate range uniformity (lower = more uniform/planar)
            range_uniformity = range_std / np.mean(ranges) if np.mean(ranges) > 0 else 0
            
            # Estimate feature diversity by trying feature extraction
            feature_diversity = 0
            try:
                # Quick feature extraction for classification - use fallback approach
                feature_set = None
                try:
                    # Try without pose (like what worked in Figure 2)
                    feature_set = self.feature_extractor.extract_features(x_points, y_points, data['scan_ranges'])
                except:
                    pass
                
                if feature_set and feature_set.features:
                    sharp_count = len(feature_set.sharp_edges)
                    planar_count = len(feature_set.planar_features)
                    total_count = len(feature_set.features)
                    
                    # Calculate feature diversity score
                    if total_count > 0:
                        sharp_ratio = sharp_count / total_count
                        planar_ratio = planar_count / total_count
                        # Higher diversity when we have sharp features
                        feature_diversity = sharp_ratio + 0.5 * (1 - abs(sharp_ratio - planar_ratio))
                    
            except Exception as e:
                if self.debug_level > 2:
                    print(f"    Feature extraction failed for scan {i}: {e}")
                feature_diversity = 0
            
            # Classify scan type based on metrics
            scan_type = self._classify_scan_complexity(
                geometric_complexity, range_uniformity, feature_diversity,
                short_range_count, medium_range_count, long_range_count, len(valid_points)
            )
            
            scan_analysis.append({
                'index': i,
                'data': data,
                'points': valid_points,
                'geometric_complexity': geometric_complexity,
                'range_uniformity': range_uniformity,
                'feature_diversity': feature_diversity,
                'point_density': len(valid_points),
                'short_range_ratio': short_range_count / len(ranges),
                'scan_type': scan_type,
                'quality_score': geometric_complexity * 0.4 + feature_diversity * 0.4 + range_uniformity * 0.2
            })
        
        # Group scans by type
        scans_by_type = {
            'edge_rich': [s for s in scan_analysis if s['scan_type'] == 'edge_rich'],
            'planar_rich': [s for s in scan_analysis if s['scan_type'] == 'planar_rich'],
            'mixed': [s for s in scan_analysis if s['scan_type'] == 'mixed']
        }
        
        print(f"\nGeometric complexity classification results:")
        for scan_type, scans in scans_by_type.items():
            print(f"  {scan_type}: {len(scans)} scans")
            if scans and self.debug_level > 0:
                # Show best example metrics for each type
                best = max(scans, key=lambda s: s['quality_score'])
                print(f"    Best example: scan #{best['index']} (geo={best['geometric_complexity']:.3f}, "
                      f"uniform={best['range_uniformity']:.3f}, diversity={best['feature_diversity']:.3f})")
        
        # If no edge_rich scans found, reclassify some high-complexity scans
        if len(scans_by_type['edge_rich']) == 0:
            print("\n  No edge_rich scans found with current criteria. Reclassifying...")
            # Find scans with highest geometric complexity and reclassify as edge_rich
            all_scans = scan_analysis
            all_scans.sort(key=lambda s: s['geometric_complexity'], reverse=True)
            
            # Take top 10% of most geometrically complex scans as edge_rich
            num_edge_rich = max(1, len(all_scans) // 20)  # At least 1, up to 5% of total
            for i in range(num_edge_rich):
                if i < len(all_scans):
                    all_scans[i]['scan_type'] = 'edge_rich'
            
            # Regroup after reclassification
            scans_by_type = {
                'edge_rich': [s for s in scan_analysis if s['scan_type'] == 'edge_rich'],
                'planar_rich': [s for s in scan_analysis if s['scan_type'] == 'planar_rich'],
                'mixed': [s for s in scan_analysis if s['scan_type'] == 'mixed']
            }
            
            print(f"  After reclassification:")
            for scan_type, scans in scans_by_type.items():
                print(f"    {scan_type}: {len(scans)} scans")
        
        return scans_by_type

    def _classify_scan_complexity(self, geometric_complexity, range_uniformity, 
                                 feature_diversity, short_count, medium_count, long_count, density):
        """Classify scan into one of three complexity types with relaxed thresholds"""
        
        # Normalize metrics for comparison
        total_points = short_count + medium_count + long_count
        short_ratio = short_count / total_points if total_points > 0 else 0
        
        # RELAXED THRESHOLDS - Edge-rich: Higher geometric complexity OR good feature diversity
        if (geometric_complexity > 0.3 and short_ratio > 0.2) or (feature_diversity > 0.1 and geometric_complexity > 0.2):
            return 'edge_rich'
        
        # Planar-rich: Very uniform ranges and low complexity
        elif range_uniformity < 0.4 and geometric_complexity < 0.3 and short_ratio < 0.3:
            return 'planar_rich'
        
        # Mixed: Everything else
        else:
            return 'mixed'

    def select_representative_scans(self, scans_by_type, manual_indices=None):
        """
        Select the best representative scan for each complexity type
        """
        if manual_indices:
            print("Using manually specified scan indices...")
            representative_scans = {}
            for scan_type, idx in manual_indices.items():
                if scan_type in ['edge_rich', 'planar_rich', 'mixed']:
                    # Find the scan with this index in any category
                    found_scan = None
                    for category_scans in scans_by_type.values():
                        for scan in category_scans:
                            if scan['index'] == idx:
                                found_scan = scan
                                break
                        if found_scan:
                            break
                    
                    if found_scan:
                        representative_scans[scan_type] = found_scan
                        print(f"  {scan_type}: Using manual scan #{idx}")
                    else:
                        print(f"  Warning: Manual index {idx} not found for {scan_type}")
            
            return representative_scans
        
        # Automatic selection - pick best representative from each category
        representative_scans = {}
        
        for scan_type, scans in scans_by_type.items():
            if not scans:
                print(f"  Warning: No scans found for {scan_type}")
                continue
            
            # Select based on different criteria for each type
            if scan_type == 'edge_rich':
                # Want highest geometric complexity and feature diversity
                best_scan = max(scans, key=lambda s: s['geometric_complexity'] + s['feature_diversity'])
            elif scan_type == 'planar_rich':
                # Want lowest range uniformity (most uniform/planar)
                best_scan = min(scans, key=lambda s: s['range_uniformity'])
            else:  # mixed
                # Want balanced/moderate complexity
                best_scan = max(scans, key=lambda s: s['quality_score'])
            
            representative_scans[scan_type] = best_scan
            
            print(f"  {scan_type}: Selected scan #{best_scan['index']}")
            print(f"    Geometric complexity: {best_scan['geometric_complexity']:.3f}")
            print(f"    Range uniformity: {best_scan['range_uniformity']:.3f}")
            print(f"    Feature diversity: {best_scan['feature_diversity']:.3f}")
        
        return representative_scans

    def create_single_panel(self, ax, scan_data, scan_type, panel_label, robot_centered=True):
        """Create a single panel showing geometric diversity"""
        
        angle_min, angle_max = -math.pi/2, math.pi/2
        
        if robot_centered:
            # Robot-centered view: convert to local coordinates (robot at origin)
            num_points = len(scan_data['scan_ranges'])
            angles = np.linspace(angle_min, angle_max, num_points)
            
            # Filter out max range values
            max_range = 11.9
            valid_indices = [i for i, r in enumerate(scan_data['scan_ranges']) if r < max_range]
            valid_ranges = [scan_data['scan_ranges'][i] for i in valid_indices]
            valid_angles = [angles[i] for i in valid_indices]
            
            # Convert to local Cartesian coordinates (robot at 0,0)
            x_points = [r * math.cos(angle) for r, angle in zip(valid_ranges, valid_angles)]
            y_points = [r * math.sin(angle) for r, angle in zip(valid_ranges, valid_angles)]
            
            robot_x, robot_y = 0, 0  # Robot at origin in local coordinates
        else:
            # Global coordinates (original approach)
            x_points, y_points = convert_scans_to_cartesian(
                scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
                flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
            )
            robot_x, robot_y = scan_data['pose']['x'], scan_data['pose']['y']
        
        # Filter valid points
        valid_points = [(x, y) for x, y in zip(x_points, y_points) 
                       if not (math.isnan(x) or math.isnan(y))]
        
        if valid_points:
            scan_x = [p[0] for p in valid_points]
            scan_y = [p[1] for p in valid_points]
            
            # Create gradient coloring based on distance from robot
            distances = [math.sqrt((x - robot_x)**2 + (y - robot_y)**2) for x, y in valid_points]
            
            # Plot scan points with distance-based coloring
            scatter = ax.scatter(scan_x, scan_y, c=distances, cmap='viridis', 
                               s=20, alpha=0.8, edgecolors='black', linewidth=0.1)
            
            # Add colorbar for distance reference
            plt.colorbar(scatter, ax=ax, label='Distance (m)', shrink=0.8)
        
        # Add robot position
        ax.scatter(robot_x, robot_y, c='red', marker='*', s=300, label='Robot', 
                  edgecolors='black', linewidth=2, zorder=10)
        
        # Formatting based on scan type
        titles = {
            'edge_rich': f'{panel_label} Sharp Edges & Corners\nAbundant geometric features',
            'planar_rich': f'{panel_label} Planar & Smooth Surfaces\nUniform geometric structure', 
            'mixed': f'{panel_label} Mixed Geometric Features\nClassification challenges'
        }
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(titles.get(scan_type, f'{panel_label} {scan_type}'), 
                    fontsize=11, pad=10)
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
        
        # Set axis limits based on data
        if valid_points:
            # Include robot position in bounds calculation
            all_x = scan_x + [robot_x]
            all_y = scan_y + [robot_y]
            
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            # Add padding
            x_padding = max(0.5, (x_max - x_min) * 0.1)
            y_padding = max(0.5, (y_max - y_min) * 0.1)
            
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Add stats text box BELOW the axis labels
        if valid_points:
            ranges = [math.sqrt((x - robot_x)**2 + (y - robot_y)**2) for x, y in valid_points]
            stats_text = (f'Points: {len(valid_points)} | '
                         f'Range: {min(ranges):.1f}-{max(ranges):.1f}m | '
                         f'Density: {len(valid_points)/max(1, max(ranges)):.1f} pts/m')
            
            # Place text well below the axis labels
            ax.text(0.5, -0.20, stats_text, 
                   transform=ax.transAxes, 
                   horizontalalignment='center',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
                   fontsize=9)
        
        return len(valid_points) if valid_points else 0

    def generate_figure1(self, file_path, output_path='figure1_dataset_diversity.png', 
                        max_entries=2000, figsize=(18, 6), manual_indices=None, robot_centered=True):
        """
        Generate Figure 1: Dataset Geometric Diversity Overview
        """
        print("Generating Figure 1: Dataset Geometric Diversity Overview")
        print(f"Data file: {file_path}")
        
        # Analyze geometric complexity of all scans
        scans_by_type = self.analyze_geometric_complexity(file_path, max_entries)
        
        # Select representative scans
        representative_scans = self.select_representative_scans(scans_by_type, manual_indices)
        
        if len(representative_scans) < 3:
            print("Warning: Could not find representatives for all complexity types")
            print("Available types:", list(representative_scans.keys()))
        
        # Create figure with three panels
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Panel order and labels
        panel_info = [
            ('edge_rich', '(a)'),
            ('planar_rich', '(b)'),
            ('mixed', '(c)')
        ]
        
        point_counts = {}
        
        for i, (scan_type, panel_label) in enumerate(panel_info):
            ax = axes[i]
            
            if scan_type in representative_scans:
                scan_info = representative_scans[scan_type]
                scan_data = scan_info['data']
                
                print(f"\nProcessing {panel_label} {scan_type} (scan #{scan_info['index']})...")
                
                # Create panel
                point_count = self.create_single_panel(ax, scan_data, scan_type, panel_label, robot_centered=robot_centered)
                point_counts[scan_type] = point_count
                
                print(f"  Points plotted: {point_count}")
            else:
                # Handle missing scan type
                ax.text(0.5, 0.5, f'No representative\n{scan_type} scan found', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       fontsize=12)
                ax.set_title(f'{panel_label} {scan_type.replace("_", " ").title()}')
        
        # Add overall title
        fig.suptitle('Figure 1: Dataset Geometric Diversity Overview', 
                    fontsize=14, y=0.95)
        
        # Adjust layout with extra space at bottom for stats
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.15)  # More space at bottom for stats
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\nFigure 1 saved to: {output_path}")
        
        # Print summary
        print(f"\nFigure 1 Summary:")
        print(f"  Dataset analyzed: {max_entries} scans")
        for scan_type, scan_info in representative_scans.items():
            count = point_counts.get(scan_type, 0)
            print(f"  {scan_type}: Scan #{scan_info['index']} ({count} points)")
        
        plt.show()
        
        return fig

def main():
    """
    Main function to generate Figure 1
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Figure 1: Dataset Geometric Diversity')
    parser.add_argument('--file', type=str, 
                       default="../../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--output', type=str, default='figure1_dataset_diversity.png',
                       help='Output filename for Figure 1')
    parser.add_argument('--max_entries', type=int, default=2000,
                       help='Maximum number of entries to analyze')
    parser.add_argument('--manual_edge', type=int, default=None,
                       help='Manual scan index for edge-rich environment')
    parser.add_argument('--manual_planar', type=int, default=None,
                       help='Manual scan index for planar-rich environment') 
    parser.add_argument('--manual_mixed', type=int, default=None,
                       help='Manual scan index for mixed environment')
    parser.add_argument('--global_coords', action='store_true', default=False,
                       help='Show scans in global coordinates instead of robot-centered')
    
    args = parser.parse_args()
    
    # Handle manual indices
    manual_indices = None
    if any([args.manual_edge, args.manual_planar, args.manual_mixed]):
        manual_indices = {}
        if args.manual_edge is not None:
            manual_indices['edge_rich'] = args.manual_edge
        if args.manual_planar is not None:
            manual_indices['planar_rich'] = args.manual_planar  
        if args.manual_mixed is not None:
            manual_indices['mixed'] = args.manual_mixed
        print(f"Using manual scan indices: {manual_indices}")
    
    # Determine coordinate system
    robot_centered = not args.global_coords  # Default to robot-centered unless global_coords specified
    
    # Create generator and generate figure
    generator = Figure1Generator(debug_level=1)
    
    try:
        generator.generate_figure1(
            file_path=args.file,
            output_path=args.output,
            max_entries=args.max_entries,
            manual_indices=manual_indices,
            robot_centered=robot_centered
        )
        print("\n✓ Figure 1 generated successfully!")
        
    except Exception as e:
        print(f"\n❌ Error generating Figure 1: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()