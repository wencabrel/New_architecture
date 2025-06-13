import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import math
import os
import time
from matplotlib.widgets import Button
import sys

# Import our utility functions and classes
from lidar_utility_functions import parse_lidar_data, convert_scans_to_cartesian, read_lidar_data_from_file
from occupancy_grid_class import OccupancyGrid
from ScanMatcher import ImprovedScanMatchingLocalization, PoseEstimate
from loop_closure import integrate_with_scan_matcher, process_loop_closure, visualize_loop_closure_results

# Import feature association components for UI and statistics (with graceful fallback)
try:
    from feature_association import (
        FeatureAssociationEngine, associate_consecutive_scans
    )
    from association_validator import (
        AssociationValidator, validate_feature_associations
    )
    from hybrid_pose_estimator import (
        HybridPoseEstimator, PoseSource
    )
    from association_visualizer import (
        AssociationVisualizer, create_debug_visualizer, quick_association_plot
    )
    ASSOCIATION_UI_AVAILABLE = True
except ImportError:
    print("Note: Feature association UI components not available. Association controls will be disabled.")
    ASSOCIATION_UI_AVAILABLE = False
    
def debug_hybrid_pose_issue(localizer):
    """Debug the hybrid pose statistics issue in detail"""
    print("\n=== DEBUGGING HYBRID POSE STATISTICS ISSUE ===")
    
    if not hasattr(localizer, 'hybrid_estimator'):
        print("❌ No hybrid_estimator found")
        return None
    
    estimator = localizer.hybrid_estimator
    stats = estimator.estimation_stats
    
    print(f"Raw statistics:")
    print(f"  total_estimations: {stats['total_estimations']}")
    print(f"  feature_dominant: {stats['feature_dominant']}")
    print(f"  icp_dominant: {stats['icp_dominant']}")
    print(f"  balanced: {stats['balanced']}")
    print(f"  fallback_to_odometry: {stats['fallback_to_odometry']}")
    print(f"  rejected_poses: {stats.get('rejected_poses', 'N/A')}")
    
    # Calculate what's tracked vs untracked
    total = stats['total_estimations']
    tracked = (stats['feature_dominant'] + stats['icp_dominant'] + 
              stats['balanced'] + stats['fallback_to_odometry'])
    untracked = total - tracked
    
    print(f"\nTracking analysis:")
    print(f"  Total: {total}")
    print(f"  Tracked: {tracked}")
    print(f"  Untracked: {untracked}")
    
    if untracked > 0:
        print(f"\n❌ PROBLEM: {untracked} estimations are untracked!")
        print(f"   This explains why all percentages are 0%")
    
    return stats

def fix_case1_statistics_tracking(localizer):
    """Fix the missing statistics tracking in recovery paths"""
    
    if not hasattr(localizer, 'hybrid_estimator'):
        print("❌ No hybrid estimator found")
        return False
    
    estimator = localizer.hybrid_estimator
    
    # Store original method if not already stored
    if not hasattr(estimator, '_original_estimate_hybrid_pose'):
        estimator._original_estimate_hybrid_pose = estimator.estimate_hybrid_pose
    
    def estimate_hybrid_pose_with_fixed_tracking(*args, **kwargs):
        """Wrapper that fixes the missing statistics tracking"""
        # Store stats before calling original method
        stats_before = estimator.estimation_stats.copy()
        
        # Call original method
        result = estimator._original_estimate_hybrid_pose(*args, **kwargs)
        
        # Check if statistics were properly updated
        stats_after = estimator.estimation_stats
        total_before = stats_before['total_estimations']
        total_after = stats_after['total_estimations']
        
        # If total increased but no category was incremented, fix it
        if total_after > total_before:
            tracked_before = (stats_before['feature_dominant'] + stats_before['icp_dominant'] + 
                            stats_before['balanced'] + stats_before['fallback_to_odometry'])
            tracked_after = (stats_after['feature_dominant'] + stats_after['icp_dominant'] + 
                           stats_after['balanced'] + stats_after['fallback_to_odometry'])
            
            # If no category was incremented, this was an untracked case
            if tracked_after == tracked_before:
                if result.source.value == 'recovery':
                    estimator.estimation_stats['fallback_to_odometry'] += 1
                    print(f"[FIX] Tracked recovery case as fallback")
                elif result.source.value == 'feature_based':
                    estimator.estimation_stats['feature_dominant'] += 1
                    print(f"[FIX] Tracked feature-based case")
                elif result.source.value == 'icp_based':
                    estimator.estimation_stats['icp_dominant'] += 1
                    print(f"[FIX] Tracked ICP-based case")
                else:
                    estimator.estimation_stats['fallback_to_odometry'] += 1
                    print(f"[FIX] Tracked unknown case as fallback")
        
        return result
    
    # Apply the fix
    estimator.estimate_hybrid_pose = estimate_hybrid_pose_with_fixed_tracking
    
    print("✅ Statistics tracking fix applied!")
    return True

# Modifications to main.py for feature extraction integration

def visualize_lidar_data_realtime(file_path, max_entries=200, show_occupancy_grid=True, 
                             grid_resolution=0.05, save_grid=True, save_format='all',
                             enable_scan_matching=True, enable_loop_closure=True,
                             enable_feature_extraction=False, enable_feature_association=False, 
                             rebuild_map=True):
    """
    Main function to visualize LiDAR data in real-time with occupancy grid mapping, 
    scan matching, loop closure detection, feature extraction, and feature association
    
    Args:
        file_path: Path to the LiDAR data file
        max_entries: Maximum number of entries to read from the file
        show_occupancy_grid: Whether to show the occupancy grid visualization
        grid_resolution: Resolution of the occupancy grid in meters (smaller = more detail but slower)
        save_grid: Whether to save the final occupancy grid map to a file
        save_format: Format to save the grid ('png', 'npy', 'csv', or 'all')
        enable_scan_matching: Whether to use scan matching localization algorithm
        enable_loop_closure: Whether to enable loop closure detection
        enable_feature_extraction: Whether to enable feature extraction alongside ICP
        enable_feature_association: Whether to enable feature association and hybrid poses
        rebuild_map: Whether to rebuild the map after loop closure optimization
    """
    
    print(f"Reading LiDAR data from: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    # Read the data from file
    parsed_data_list = read_lidar_data_from_file(file_path, max_entries)
    
    if not parsed_data_list:
        print("No data was read from the file.")
        return
    
    # Display data summary
    first_timestamp = parsed_data_list[0]['timestamp']
    last_timestamp = parsed_data_list[-1]['timestamp']
    duration = last_timestamp - first_timestamp
    
    print(f"\nData Summary:")
    print(f"  Number of entries: {len(parsed_data_list)}")
    print(f"  Robot ID: {parsed_data_list[0]['robot_id']}")
    print(f"  Data duration: {duration:.2f} seconds")
    
    if show_occupancy_grid:
        print(f"  Starting visualization with occupancy grid mapping (resolution: {grid_resolution}m)...")
        if enable_scan_matching:
            print(f"  Scan matching is ENABLED - using ICP with adaptive parameters")
        if enable_loop_closure and enable_scan_matching:
            print(f"  Loop closure detection is ENABLED - using Scan Context descriptors and pose graph optimization")
        if enable_feature_extraction and enable_scan_matching:
            print(f"  Feature extraction is ENABLED - using LOAM-style curvature-based features")
        if save_grid:
            print(f"  The final occupancy grid will be saved in '{save_format}' format")
    else:
        print(f"  Starting visualization with orientation correction...")
    
    # Create output directory for maps
    maps_dir = "maps"
    if save_grid and not os.path.exists(maps_dir):
        try:
            os.makedirs(maps_dir)
            print(f"  Created directory for maps: {maps_dir}/")
        except Exception as e:
            print(f"  Error creating maps directory: {e}")
    
    # Assuming the LiDAR scan covers 180 degrees (π radians)
    angle_min = -math.pi/2
    angle_max = math.pi/2
    
    # Find max range for consistent scaling by converting all data points
    all_x_points = []
    all_y_points = []
    for parsed_data in parsed_data_list:
        x_points, y_points = convert_scans_to_cartesian(
            parsed_data['scan_ranges'], angle_min, angle_max, parsed_data['pose'],
            flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
        )
        all_x_points.extend(x_points)
        all_y_points.extend(y_points)
    
    # Calculate proper axis limits for visualization
    x_min, x_max = min(all_x_points), max(all_x_points)
    y_min, y_max = min(all_y_points), max(all_y_points)
    
    # Add some padding (20%)
    x_padding = max(1.0, (x_max - x_min) * 0.2)
    y_padding = max(1.0, (y_max - y_min) * 0.2)
    
    # Set limits with padding
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Calculate grid dimensions based on data range
    grid_width = max(20, int(math.ceil((x_max - x_min) * 1.5)))  # Make grid at least 20m wide
    grid_height = max(20, int(math.ceil((y_max - y_min) * 1.5)))  # Make grid at least 20m tall
    
    # Initialize occupancy grid
    if show_occupancy_grid:
        occupancy_grid = OccupancyGrid(resolution=grid_resolution, 
                                      initial_width=grid_width, 
                                      initial_height=grid_height, 
                                      expansion_factor=1.5, 
                                      sensor_noise_variance=0.01)
    else:
        occupancy_grid = None
    
    # Initialize scan matching localization if enabled
    if enable_scan_matching:
        print(f"  Using improved ICP scan matching algorithm with motion validation")
        
        # Enhanced initialization message
        enhancement_features = []
        if enable_feature_extraction:
            enhancement_features.append("feature extraction")
        if enable_feature_association:
            enhancement_features.append("feature association")
        
        if enhancement_features:
            print(f"  Enhanced with: {', '.join(enhancement_features)}")
        
        # Initialize scan matcher with all enabled features
        localizer = ImprovedScanMatchingLocalization(
            occupancy_grid, 
            debug_level=1,
            enable_feature_extraction=enable_feature_extraction,
            enable_feature_association=enable_feature_association
        )
        
        # Enable loop closure if requested
        if enable_loop_closure:
            localizer = integrate_with_scan_matcher(localizer, loop_closure_enabled=True)
        
        # Process sensor data to build trajectory with scan matching
        localizer.processSensorData(
            parsed_data_list,
            angle_min=angle_min,
            angle_max=angle_max,
            flip_x=False,
            flip_y=False,
            reverse_scan=True,
            flip_theta=False
        )
        
        # Extract trajectory for visualization
        trajectory = localizer.trajectory
        robot_path_x = [pose.x for pose in trajectory]
        robot_path_y = [pose.y for pose in trajectory]
        
        # Also keep track of odometry trajectory for comparison
        odometry_trajectory = localizer.odometry_trajectory
        odometry_path_x = [pose.x for pose in odometry_trajectory]
        odometry_path_y = [pose.y for pose in odometry_trajectory]
        
        # Process loop closures after the entire trajectory has been processed
        if enable_loop_closure:
            print("\nChecking for loop closures in the trajectory...")
            optimized_trajectory, updated_map = process_loop_closure(
                localizer, parsed_data_list, occupancy_grid,
                angle_min=angle_min, angle_max=angle_max,
                flip_x=False, flip_y=False, 
                reverse_scan=True, flip_theta=False
            )
            
            if localizer.loop_closures_detected > 0 and localizer.is_optimized:
                print(f"Found and processed {localizer.loop_closures_detected} loop closures!")
                
                # Update map and trajectory if requested
                if rebuild_map:
                    occupancy_grid = updated_map
                
                # Update trajectory
                robot_path_x = [pose.x for pose in optimized_trajectory]
                robot_path_y = [pose.y for pose in optimized_trajectory]
            else:
                print("No loop closures were detected or processed.")
    else:
        # Use odometry-based trajectory without scan matching
        print(f"  Scan matching is DISABLED - using raw odometry")
        robot_path_x = []
        robot_path_y = []
        for data in parsed_data_list:
            x, y = data['pose']['x'], data['pose']['y']
            robot_path_x.append(x)
            robot_path_y.append(y)
        odometry_path_x = robot_path_x
        odometry_path_y = robot_path_y
        localizer = None  # No localizer available
    
    # Create a figure with two subplots side by side if showing occupancy grid
    if show_occupancy_grid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        
        # Set up the occupancy grid image
        # Custom colormap: white (unknown), black (occupied), light gray (free)
        cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
        bounds = [0, 0.4, 0.6, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Initialize the occupancy grid display
        grid_img = ax2.imshow(occupancy_grid.get_grid_for_display(), 
                             cmap=cmap, norm=norm, 
                             origin='lower', 
                             extent=[-occupancy_grid.width/2, occupancy_grid.width/2, 
                                     -occupancy_grid.height/2, occupancy_grid.height/2])
        
        # Add reference grid lines
        ax2.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Create a line for robot path on the occupancy grid
        grid_path_line, = ax2.plot(robot_path_x, robot_path_y, 'b-', linewidth=2, label='Matched Path')
        
        # Create a line for odometry path if scan matching is enabled
        if enable_scan_matching:
            grid_odom_line, = ax2.plot(odometry_path_x, odometry_path_y, 'r--', linewidth=1, alpha=0.6, label='Odometry Path')
        
        # Also plot the starting position on the grid
        if len(robot_path_x) > 0:
            grid_start_point = ax2.scatter(robot_path_x[0], robot_path_y[0], c='green', s=100, marker='*', label='Start')
            grid_current_pos = ax2.scatter(robot_path_x[-1], robot_path_y[-1], c='blue', s=100, marker='*', label='End')
        
        # Add buttons for visualization and saving
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons
        
        # Button positions
        button_width = 0.07
        button_spacing = 0.01
        button_height = 0.04
        button_y = 0.05
        button_x_start = 0.05
        
        # Add a Save Map button
        save_button_ax = plt.axes([button_x_start, button_y, button_width, button_height])
        save_button = Button(save_button_ax, 'Save Map', color='lightblue', hovercolor='0.8')
        
        # Add feature extraction related buttons if enabled
        if enable_feature_extraction and enable_scan_matching and localizer:
            # Feature Visualization button
            feature_viz_button_ax = plt.axes([button_x_start + (button_width + button_spacing), button_y, button_width, button_height])
            feature_viz_button = Button(feature_viz_button_ax, 'Show Features', color='lightgreen', hovercolor='0.8')
            
            # Feature Statistics button
            feature_stats_button_ax = plt.axes([button_x_start + 2*(button_width + button_spacing), button_y, button_width, button_height])
            feature_stats_button = Button(feature_stats_button_ax, 'Feature Stats', color='lightyellow', hovercolor='0.8')
            
            # Save Features button  
            save_features_button_ax = plt.axes([button_x_start + 3*(button_width + button_spacing), button_y, button_width, button_height])
            save_features_button = Button(save_features_button_ax, 'Save Features', color='lightcoral', hovercolor='0.8')
            
            # Feature Association Controls (if enabled)
            if enable_feature_association and ASSOCIATION_UI_AVAILABLE:
                print("Setting up feature association visualization controls...")
                
                # Calculate button positions following the existing pattern
                base_offset = 4  # Start after the 4 feature extraction buttons
                
                # Association Visualization button
                assoc_viz_button_ax = plt.axes([
                    button_x_start + base_offset*(button_width + button_spacing), 
                    button_y, 
                    button_width, 
                    button_height
                ])
                assoc_viz_button = Button(assoc_viz_button_ax, 'Associations', color='lightpink', hovercolor='0.8')
                
                # Association Statistics button
                assoc_stats_button_ax = plt.axes([
                    button_x_start + (base_offset + 1)*(button_width + button_spacing), 
                    button_y, 
                    button_width, 
                    button_height
                ])
                assoc_stats_button = Button(assoc_stats_button_ax, 'Assoc Stats', color='lightsteelblue', hovercolor='0.8')
                
                # Save Associations button
                assoc_save_button_ax = plt.axes([
                    button_x_start + (base_offset + 2)*(button_width + button_spacing), 
                    button_y, 
                    button_width, 
                    button_height
                ])
                assoc_save_button = Button(assoc_save_button_ax, 'Save Assoc', color='lightgoldenrodyellow', hovercolor='0.8')
                
                # Hybrid Analysis button
                hybrid_analysis_button_ax = plt.axes([
                    button_x_start + (base_offset + 3)*(button_width + button_spacing), 
                    button_y, 
                    button_width, 
                    button_height
                ])
                hybrid_analysis_button = Button(hybrid_analysis_button_ax, 'Hybrid Perf', color='lightcyan', hovercolor='0.8')
                
                def show_associations(event):
                    """
                    Enhanced callback function to access association data from existing components
                    """
                    try:
                        if hasattr(localizer, 'enable_feature_association') and localizer.enable_feature_association:
                            print("Attempting to create association visualization...")
                            
                            # Method 1: Try to get data from association_engine
                            associations = []
                            current_descriptors = []
                            previous_descriptors = []
                            validation_result = None
                            
                            if hasattr(localizer, 'association_engine') and localizer.association_engine:
                                engine = localizer.association_engine
                                print(f"Association engine found: {type(engine)}")
                                
                                # Check if engine has recent data stored
                                if hasattr(engine, 'last_associations'):
                                    associations = engine.last_associations or []
                                if hasattr(engine, 'last_current_descriptors'):
                                    current_descriptors = engine.last_current_descriptors or []
                                if hasattr(engine, 'last_previous_descriptors'):
                                    previous_descriptors = engine.last_previous_descriptors or []
                                    
                                print(f"Found {len(associations)} associations from engine")
                            
                            # Method 2: Try to get data from feature_history
                            if not associations and hasattr(localizer, 'feature_history') and localizer.feature_history:
                                print(f"Feature history available with {len(localizer.feature_history)} entries")
                                
                                # If we have at least 2 feature sets, try to create associations
                                if len(localizer.feature_history) >= 2:
                                    try:
                                        current_features = localizer.feature_history[-1]  # Most recent
                                        previous_features = localizer.feature_history[-2]  # Second most recent
                                        
                                        print(f"Creating associations from feature history...")
                                        print(f"Current features: {len(current_features.features) if hasattr(current_features, 'features') else 'unknown'}")
                                        print(f"Previous features: {len(previous_features.features) if hasattr(previous_features, 'features') else 'unknown'}")
                                        
                                        # Use the association engine to create associations
                                        engine = localizer.association_engine
                                        
                                        # Create descriptors
                                        current_descriptors = engine.create_descriptors(current_features, scan_index=1)
                                        previous_descriptors = engine.create_descriptors(previous_features, scan_index=0)
                                        
                                        print(f"Created {len(current_descriptors)} current descriptors")
                                        print(f"Created {len(previous_descriptors)} previous descriptors")
                                        
                                        # Perform association
                                        if current_descriptors and previous_descriptors:
                                            associations = engine.associate_features(
                                                current_descriptors, 
                                                previous_descriptors, 
                                                motion_estimate=None
                                            )
                                            print(f"Generated {len(associations)} associations")
                                            
                                            # Try to validate associations
                                            if hasattr(localizer, 'association_validator') and localizer.association_validator:
                                                try:
                                                    validation_result = localizer.association_validator.validate_associations(
                                                        associations, current_descriptors, previous_descriptors
                                                    )
                                                    print(f"Validation result: {validation_result.is_valid if validation_result else 'None'}")
                                                except Exception as e:
                                                    print(f"Validation failed: {e}")
                                        
                                    except Exception as e:
                                        print(f"Error creating associations from feature history: {e}")
                                        import traceback
                                        traceback.print_exc()
                            
                            # Method 3: Access any stored current/previous features directly
                            if not associations:
                                current_features = None
                                previous_features = None
                                
                                # Check for stored feature sets
                                if hasattr(localizer, 'current_features'):
                                    current_features = localizer.current_features
                                if hasattr(localizer, 'previous_features'):
                                    previous_features = localizer.previous_features
                                
                                if current_features and previous_features:
                                    print("Found current and previous features, creating visualization...")
                                    create_feature_only_visualization(current_features, previous_features)
                                    return
                            
                            print(f"Final result: {len(associations)} associations, {len(current_descriptors)} current, {len(previous_descriptors)} previous")
                            
                            # Create visualization if we have data
                            if associations and current_descriptors and previous_descriptors:
                                try:
                                    # Try to import and use the full visualizer
                                    try:
                                        from association_visualizer import AssociationVisualizer, VisualizationConfig
                                        
                                        config = VisualizationConfig()
                                        config.show_all_associations = True
                                        config.show_quality_metrics = True
                                        config.figure_size = (15, 10)
                                        
                                        visualizer = AssociationVisualizer(config=config, save_plots=False)
                                        
                                        fig = visualizer.visualize_associations(
                                            current_descriptors=current_descriptors,
                                            previous_descriptors=previous_descriptors,
                                            associations=associations,
                                            validation_result=validation_result,
                                            title=f"Feature Associations ({len(associations)} found)"
                                        )
                                        
                                        if fig:
                                            plt.show()
                                            print(f"Association visualization displayed with {len(associations)} associations.")
                                            return
                                        
                                    except ImportError as e:
                                        print(f"Full visualizer not available ({e}), using simple plot...")
                                    
                                    # Fallback to simple visualization
                                    create_simple_association_plot(associations, current_descriptors, previous_descriptors, validation_result)
                                    
                                except Exception as e:
                                    print(f"Error creating visualization: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            elif len(localizer.feature_history) >= 2:
                                # Show just features if we have them
                                current_features = localizer.feature_history[-1]
                                previous_features = localizer.feature_history[-2]
                                create_feature_only_visualization(current_features, previous_features)
                            else:
                                print("No usable data found for visualization.")
                                print("Suggestions:")
                                print("1. Make sure you've processed at least 2 scans")
                                print("2. Ensure feature extraction is working")
                                print("3. Check that association processing is enabled")
                                
                                # Show association statistics if available
                                if hasattr(localizer, 'get_association_statistics'):
                                    try:
                                        stats = localizer.get_association_statistics()
                                        print(f"\nCurrent association statistics:")
                                        for key, value in stats.items():
                                            print(f"  {key}: {value}")
                                    except Exception as e:
                                        print(f"Could not get association statistics: {e}")
                        else:
                            print("Feature association is not enabled.")
                            
                    except Exception as e:
                        print(f"Error in association visualization: {e}")
                        import traceback
                        traceback.print_exc()


                def create_simple_association_plot(associations, current_descriptors, previous_descriptors, validation_result=None):
                    """
                    Create a simple association plot when the full visualizer is not available
                    """
                    try:
                        fig, ax = plt.subplots(figsize=(15, 10))
                        
                        # Plot current features
                        if current_descriptors:
                            current_x = []
                            current_y = []
                            current_types = []
                            
                            for desc in current_descriptors:
                                if desc.base_feature and hasattr(desc.base_feature, 'point_world'):
                                    current_x.append(desc.base_feature.point_world[0])
                                    current_y.append(desc.base_feature.point_world[1])
                                    if hasattr(desc.base_feature, 'feature_type'):
                                        current_types.append(desc.base_feature.feature_type.value)
                                    else:
                                        current_types.append('unknown')
                            
                            if current_x:
                                ax.scatter(current_x, current_y, c='blue', s=50, alpha=0.8, 
                                        label=f'Current Features ({len(current_x)})', marker='o', edgecolors='black')
                        
                        # Plot previous features
                        if previous_descriptors:
                            prev_x = []
                            prev_y = []
                            
                            for desc in previous_descriptors:
                                if desc.base_feature and hasattr(desc.base_feature, 'point_world'):
                                    prev_x.append(desc.base_feature.point_world[0])
                                    prev_y.append(desc.base_feature.point_world[1])
                            
                            if prev_x:
                                ax.scatter(prev_x, prev_y, c='red', s=50, alpha=0.6, 
                                        label=f'Previous Features ({len(prev_x)})', marker='s', edgecolors='black')
                        
                        # Plot associations
                        association_count = 0
                        if associations and current_descriptors and previous_descriptors:
                            for i, assoc in enumerate(associations):
                                try:
                                    if (assoc.feature_idx1 < len(current_descriptors) and 
                                        assoc.feature_idx2 < len(previous_descriptors)):
                                        
                                        curr_desc = current_descriptors[assoc.feature_idx1]
                                        prev_desc = previous_descriptors[assoc.feature_idx2]
                                        
                                        if (curr_desc.base_feature and prev_desc.base_feature and
                                            hasattr(curr_desc.base_feature, 'point_world') and 
                                            hasattr(prev_desc.base_feature, 'point_world')):
                                            
                                            x1, y1 = curr_desc.base_feature.point_world[0], curr_desc.base_feature.point_world[1]
                                            x2, y2 = prev_desc.base_feature.point_world[0], prev_desc.base_feature.point_world[1]
                                            
                                            # Color based on association quality
                                            if hasattr(assoc, 'score'):
                                                if assoc.score > 0.7:
                                                    color = 'green'
                                                elif assoc.score > 0.5:
                                                    color = 'orange'
                                                else:
                                                    color = 'red'
                                                alpha = min(1.0, assoc.score + 0.3)
                                            else:
                                                color = 'gray'
                                                alpha = 0.5
                                            
                                            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=2)
                                            association_count += 1
                                            
                                except (IndexError, AttributeError) as e:
                                    continue
                        
                        # Add validation info if available
                        title = f'Feature Associations ({association_count} associations drawn)'
                        if validation_result:
                            status = "VALID" if validation_result.is_valid else "INVALID"
                            confidence = validation_result.confidence if hasattr(validation_result, 'confidence') else 0
                            title += f' - Validation: {status} (conf: {confidence:.2f})'
                        
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                        ax.set_xlabel('X (meters)')
                        ax.set_ylabel('Y (meters)')
                        ax.set_title(title)
                        ax.legend()
                        
                        # Add color legend for association quality
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], color='green', lw=2, label='High Quality (>0.7)'),
                            Line2D([0], [0], color='orange', lw=2, label='Medium Quality (0.5-0.7)'),
                            Line2D([0], [0], color='red', lw=2, label='Low Quality (<0.5)')
                        ]
                        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
                        
                        plt.tight_layout()
                        plt.show()
                        print(f"Association plot created with {association_count} associations displayed.")
                        
                    except Exception as e:
                        print(f"Failed to create association plot: {e}")
                        import traceback
                        traceback.print_exc()


                def create_feature_only_visualization(current_features, previous_features):
                    """
                    Create a visualization showing only features when association data is not available
                    """
                    try:
                        fig, ax = plt.subplots(figsize=(12, 10))
                        
                        # Plot current features
                        if current_features and hasattr(current_features, 'features'):
                            current_x = []
                            current_y = []
                            current_types = []
                            
                            for f in current_features.features:
                                if hasattr(f, 'point_world'):
                                    current_x.append(f.point_world[0])
                                    current_y.append(f.point_world[1])
                                    if hasattr(f, 'feature_type'):
                                        current_types.append(f.feature_type.value)
                                    else:
                                        current_types.append('unknown')
                            
                            if current_x:
                                ax.scatter(current_x, current_y, c='blue', s=40, alpha=0.8, 
                                        label=f'Current Features ({len(current_x)})', marker='o')
                        
                        # Plot previous features
                        if previous_features and hasattr(previous_features, 'features'):
                            prev_x = []
                            prev_y = []
                            
                            for f in previous_features.features:
                                if hasattr(f, 'point_world'):
                                    prev_x.append(f.point_world[0])
                                    prev_y.append(f.point_world[1])
                            
                            if prev_x:
                                ax.scatter(prev_x, prev_y, c='red', s=40, alpha=0.6, 
                                        label=f'Previous Features ({len(prev_x)})', marker='s')
                        
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                        ax.set_xlabel('X (meters)')
                        ax.set_ylabel('Y (meters)')
                        ax.set_title('Features Only (No Association Data Available)')
                        ax.legend()
                        
                        plt.tight_layout()
                        plt.show()
                        print("Feature-only visualization created.")
                        
                    except Exception as e:
                        print(f"Failed to create feature-only visualization: {e}")
                        import traceback
                        traceback.print_exc()


                # Debug function to inspect association engine
                def debug_association_engine(localizer):
                    """
                    Debug the association engine to see what data it contains
                    """
                    print("\n=== DEBUGGING ASSOCIATION ENGINE ===")
                    
                    if hasattr(localizer, 'association_engine') and localizer.association_engine:
                        engine = localizer.association_engine
                        print(f"Association engine type: {type(engine)}")
                        
                        # Check for data storage attributes
                        data_attrs = [
                            'last_associations', 'recent_associations', 'association_history',
                            'last_current_descriptors', 'last_previous_descriptors',
                            'current_descriptors', 'previous_descriptors'
                        ]
                        
                        for attr in data_attrs:
                            if hasattr(engine, attr):
                                obj = getattr(engine, attr)
                                if obj is not None:
                                    if isinstance(obj, list):
                                        print(f"  ✓ {attr}: List with {len(obj)} items")
                                    else:
                                        print(f"  ✓ {attr}: {type(obj)}")
                                else:
                                    print(f"  ○ {attr}: None")
                            else:
                                print(f"  ✗ {attr}: Not found")
                        
                        # Check statistics
                        if hasattr(engine, 'get_association_statistics'):
                            try:
                                stats = engine.get_association_statistics()
                                print(f"  Statistics: {stats}")
                            except Exception as e:
                                print(f"  Statistics error: {e}")
                    
                    print("=" * 40)


                # Usage: Add this line to your callback for debugging
                # debug_association_engine(localizer)
                
                def show_association_stats(event):
                    try:
                        if hasattr(localizer, 'get_association_statistics'):
                            stats = localizer.get_association_statistics()
                            
                            print(f"\n{'='*60}")
                            print(f"FEATURE ASSOCIATION STATISTICS")
                            print(f"{'='*60}")
                            
                            if stats.get('feature_association_enabled', False):
                                print(f"Association Success Rate: {stats.get('association_success_rate', 0)*100:.1f}%")
                                print(f"Validation Success Rate: {stats.get('validation_success_rate', 0)*100:.1f}%")
                                print(f"Average Association Time: {stats.get('average_association_time', 0):.2f}ms")
                                print(f"Total Hybrid Poses: {stats.get('total_hybrid_poses', 0)}")
                                print(f"Feature Dominant: {stats.get('feature_dominant_rate', 0)*100:.1f}%")
                                print(f"ICP Dominant: {stats.get('icp_dominant_rate', 0)*100:.1f}%")
                                print(f"Balanced: {stats.get('balanced_rate', 0)*100:.1f}%")
                                print(f"Fallback Rate: {stats.get('fallback_rate', 0)*100:.1f}%")
                                print(f"Budget Violations: {stats.get('budget_violations', 0)}")
                                
                                # Performance assessment
                                if stats.get('performance_excellent', False):
                                    print("Performance Status: ✅ EXCELLENT")
                                elif stats.get('performance_good', False):
                                    print("Performance Status: ✅ GOOD")
                                else:
                                    print("Performance Status: ⚠️ NEEDS OPTIMIZATION")
                                    
                                if stats.get('validation_excellent', False):
                                    print("Validation Status: ✅ EXCELLENT")
                                elif stats.get('validation_good', False):
                                    print("Validation Status: ✅ GOOD")
                                else:
                                    print("Validation Status: ⚠️ NEEDS IMPROVEMENT")
                            else:
                                print("Feature association is not enabled.")
                            
                            print(f"{'='*60}")
                            
                        else:
                            print("Association statistics not available.")
                    except Exception as e:
                        print(f"Error retrieving association statistics: {e}")
                
                def save_associations(event):
                    try:
                        if hasattr(localizer, 'enable_feature_association') and localizer.enable_feature_association:
                            # Create association output directory
                            maps_dir = "maps"
                            if not os.path.exists(maps_dir):
                                os.makedirs(maps_dir)
                            
                            assoc_dir = os.path.join(maps_dir, "associations")
                            if not os.path.exists(assoc_dir):
                                os.makedirs(assoc_dir)
                            
                            # Generate timestamp-based filename
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            
                            # Save association statistics
                            stats_filename = os.path.join(assoc_dir, f"association_stats_{timestamp}.txt")
                            if hasattr(localizer, 'get_association_statistics'):
                                stats = localizer.get_association_statistics()
                                
                                with open(stats_filename, 'w') as f:
                                    f.write("FEATURE ASSOCIATION STATISTICS REPORT\n")
                                    f.write("="*50 + "\n\n")
                                    f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                                    
                                    for key, value in stats.items():
                                        if isinstance(value, float):
                                            f.write(f"{key}: {value:.4f}\n")
                                        else:
                                            f.write(f"{key}: {value}\n")
                            
                            # Save association visualization
                            viz_filename = os.path.join(assoc_dir, f"associations_{timestamp}.png")
                            if hasattr(localizer, 'visualize_associations_with_trajectory'):
                                fig_save = plt.figure(figsize=(15, 10))
                                ax = localizer.visualize_associations_with_trajectory()
                                if ax:
                                    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
                                    plt.close(fig_save)
                            
                            print(f"\nAssociation data saved to:")
                            print(f"  - Statistics: {stats_filename}")
                            print(f"  - Visualization: {viz_filename}")
                            
                        else:
                            print("No association data available to save.")
                    except Exception as e:
                        print(f"Error saving association data: {e}")
                
                def analyze_hybrid_performance(event):
                    try:
                        if hasattr(localizer, 'get_association_statistics') and hasattr(localizer, 'get_feature_extraction_statistics'):
                            assoc_stats = localizer.get_association_statistics()
                            feature_stats = localizer.get_feature_extraction_statistics()
                            
                            print(f"\n{'='*70}")
                            print(f"HYBRID LOCALIZATION PERFORMANCE ANALYSIS")
                            print(f"{'='*70}")
                            
                            # Feature extraction performance
                            if feature_stats.get('feature_extraction_enabled', False):
                                print(f"\nFeature Extraction:")
                                print(f"  Average extraction time: {feature_stats.get('average_feature_time', 0):.2f}ms")
                                print(f"  Average features per scan: {feature_stats.get('average_features_per_scan', 0):.1f}")
                                print(f"  Average quality score: {feature_stats.get('average_quality_score', 0):.3f}")
                            
                            # Association performance
                            if assoc_stats.get('feature_association_enabled', False):
                                print(f"\nFeature Association:")
                                print(f"  Average association time: {assoc_stats.get('average_association_time', 0):.2f}ms")
                                print(f"  Validation success rate: {assoc_stats.get('validation_success_rate', 0)*100:.1f}%")
                                print(f"  Total processing budget: {feature_stats.get('average_feature_time', 0) + assoc_stats.get('average_association_time', 0):.2f}ms")
                            
                            # Pose source analysis
                            print(f"\nPose Source Distribution:")
                            print(f"  Feature dominant: {assoc_stats.get('feature_dominant_rate', 0)*100:.1f}%")
                            print(f"  ICP dominant: {assoc_stats.get('icp_dominant_rate', 0)*100:.1f}%")
                            print(f"  Balanced fusion: {assoc_stats.get('balanced_rate', 0)*100:.1f}%")
                            print(f"  Fallback used: {assoc_stats.get('fallback_rate', 0)*100:.1f}%")
                            
                            # Performance recommendations
                            print(f"\nPerformance Assessment:")
                            total_time = feature_stats.get('average_feature_time', 0) + assoc_stats.get('average_association_time', 0)
                            if total_time <= 20:
                                print("  ✅ EXCELLENT - Well within real-time budget")
                            elif total_time <= 30:
                                print("  ✅ GOOD - Acceptable for most applications")
                            elif total_time <= 50:
                                print("  ⚠️ ACCEPTABLE - May need optimization for high-frequency operation")
                            else:
                                print("  ❌ NEEDS OPTIMIZATION - Exceeds typical real-time budgets")
                            
                            print(f"{'='*70}")
                            
                        else:
                            print("Hybrid performance analysis requires both feature extraction and association to be enabled.")
                            
                    except Exception as e:
                        print(f"Error in hybrid performance analysis: {e}")
                
                # Connect button events
                assoc_viz_button.on_clicked(show_associations)
                assoc_stats_button.on_clicked(show_association_stats)
                assoc_save_button.on_clicked(save_associations)
                hybrid_analysis_button.on_clicked(analyze_hybrid_performance)
        
        # Add a Visualize ICP Process button if scan matching is enabled
        if enable_scan_matching:
            # Calculate offset based on what's already been added
            if enable_feature_extraction and enable_feature_association and ASSOCIATION_UI_AVAILABLE:
                button_offset = 8  # 4 feature buttons + 4 association buttons
            elif enable_feature_extraction:
                button_offset = 4  # 4 feature buttons only
            else:
                button_offset = 1  # Just the save map button
                
            icp_viz_button_ax = plt.axes([button_x_start + button_offset*(button_width + button_spacing), button_y, button_width, button_height])
            icp_viz_button = Button(icp_viz_button_ax, 'Visualize ICP', color='lightgreen', hovercolor='0.8')
        
            # Add a Compare Paths button
            if enable_feature_extraction and enable_feature_association and ASSOCIATION_UI_AVAILABLE:
                button_offset = 9  # 4 feature + 4 association + 1 ICP button
            elif enable_feature_extraction:
                button_offset = 5  # 4 feature + 1 ICP button
            else:
                button_offset = 2  # 1 save + 1 ICP button
                
            compare_button_ax = plt.axes([button_x_start + button_offset*(button_width + button_spacing), button_y, button_width, button_height])
            compare_button = Button(compare_button_ax, 'Compare Paths', color='lightcoral', hovercolor='0.8')
        
            # Add a Loop Closure Visualization button if loop closure is enabled
            if enable_loop_closure and enable_scan_matching:
                # Calculate offset based on what's already been added
                if enable_feature_extraction and enable_feature_association and ASSOCIATION_UI_AVAILABLE:
                    button_offset = 10  # 4 feature + 4 association + 1 ICP + 1 compare
                elif enable_feature_extraction:
                    button_offset = 6   # 4 feature + 1 ICP + 1 compare
                else:
                    button_offset = 3   # 1 save + 1 ICP + 1 compare
                    
                loop_viz_button_ax = plt.axes([button_x_start + button_offset*(button_width + button_spacing), button_y, button_width, button_height])
                loop_viz_button = Button(loop_viz_button_ax, 'Loop Closures', color='lightsalmon', hovercolor='0.8')
        
        def save_map(event):
            if not show_occupancy_grid:
                print("Cannot save map - occupancy grid is disabled.")
                return
                
            # Create the save directory if it doesn't exist
            if not os.path.exists(maps_dir):
                os.makedirs(maps_dir)
            
            # Generate a timestamp-based filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_filename = os.path.join(maps_dir, f"occupancy_grid_{timestamp}")
            
            # Create path coordinates for saving
            robot_path_coords = list(zip(robot_path_x, robot_path_y))
            
            # Get start and current positions from path
            start_pos = (robot_path_x[0], robot_path_y[0]) if len(robot_path_x) > 0 else None
            current_pos = (robot_path_x[-1], robot_path_y[-1]) if len(robot_path_x) > 0 else None
            
            # Save the grid with robot path and positions
            saved_files = occupancy_grid.save_to_file(
                base_filename, 
                format=save_format, 
                include_metadata=True,
                robot_path=robot_path_coords,
                start_position=start_pos,
                current_position=current_pos
            )
            
            print(f"\nOccupancy grid map saved to:")
            for file in saved_files:
                print(f"  - {file}")
        
        save_button.on_clicked(save_map)
        
        # Feature extraction button handlers
        if enable_feature_extraction and enable_scan_matching and localizer:
            def show_features(event):
                try:
                    # Create feature visualization
                    feature_fig, feature_ax = plt.subplots(figsize=(12, 10))
                    localizer.visualize_features_with_trajectory(ax=feature_ax, show_curvatures=False)
                    plt.title('Extracted Features with Robot Trajectory')
                    plt.tight_layout()
                    plt.show()
                    
                    # Also create curvature visualization
                    curvature_fig, curvature_ax = plt.subplots(figsize=(12, 10))
                    localizer.visualize_features_with_trajectory(ax=curvature_ax, show_curvatures=True)
                    plt.title('Features Colored by Curvature Strength')
                    plt.tight_layout()
                    plt.show()
                    
                except Exception as e:
                    print(f"Error creating feature visualization: {e}")
            
            def show_feature_stats(event):
                try:
                    stats = localizer.get_feature_extraction_statistics()
                    if stats.get('feature_extraction_enabled', False):
                        print(f"\n{'='*60}")
                        print(f"FEATURE EXTRACTION STATISTICS")
                        print(f"{'='*60}")
                        print(f"Total extractions: {stats['total_extractions']}")
                        print(f"Total features: {stats['total_features_extracted']}")
                        print(f"Average features per scan: {stats['average_features_per_scan']:.1f}")
                        print(f"Average extraction time: {stats['average_feature_time']:.2f} ms")
                        
                        if 'average_quality_score' in stats:
                            print(f"Average quality score: {stats['average_quality_score']:.3f}")
                            print(f"Quality range: {stats['min_quality_score']:.3f} - {stats['max_quality_score']:.3f}")
                            print(f"Feature count range: {stats['min_features_per_scan']} - {stats['max_features_per_scan']}")
                        
                        # Performance assessment
                        if stats.get('performance_excellent', False):
                            print("Performance: ✓ EXCELLENT (≤15ms per scan)")
                        elif stats.get('performance_good', False):
                            print("Performance: ⚠ GOOD (15-25ms per scan)")
                        else:
                            print("Performance: ⚠ NEEDS OPTIMIZATION (>25ms per scan)")
                        
                        # Quality assessment
                        if stats.get('quality_excellent', False):
                            print("Quality: ✓ EXCELLENT (≥0.7)")
                        elif stats.get('quality_good', False):
                            print("Quality: ✓ GOOD (0.5-0.7)")
                        elif stats.get('quality_acceptable', False):
                            print("Quality: ⚠ ACCEPTABLE (0.3-0.5)")
                        else:
                            print("Quality: ⚠ NEEDS IMPROVEMENT (<0.3)")
                        
                        print(f"{'='*60}")
                    else:
                        print("Feature extraction statistics not available.")
                except Exception as e:
                    print(f"Error retrieving feature statistics: {e}")
            
            def save_features(event):
                try:
                    if hasattr(localizer, 'feature_extractor') and localizer.feature_extractor:
                        # Create feature output directory
                        feature_dir = os.path.join(maps_dir, "features")
                        if not os.path.exists(feature_dir):
                            os.makedirs(feature_dir)
                        
                        # Generate timestamp-based filename
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        
                        # Save feature data in JSON format
                        json_filename = os.path.join(feature_dir, f"features_{timestamp}.json")
                        localizer.feature_extractor.save_feature_data(json_filename, format='json')
                        
                        # Save feature data in CSV format
                        csv_filename = os.path.join(feature_dir, f"features_{timestamp}.csv")
                        localizer.feature_extractor.save_feature_data(csv_filename, format='csv')
                        
                        print(f"\nFeature data saved to:")
                        print(f"  - {json_filename}")
                        print(f"  - {csv_filename}")
                    else:
                        print("No feature data available to save.")
                except Exception as e:
                    print(f"Error saving feature data: {e}")
            
            feature_viz_button.on_clicked(show_features)
            feature_stats_button.on_clicked(show_feature_stats)
            save_features_button.on_clicked(save_features)
        
        if enable_scan_matching:
            def visualize_icp_process(event):
                # Create ICP process visualization
                if hasattr(localizer, 'visualizeIcpProcess'):
                    fig = localizer.visualizeIcpProcess()
                    if fig:
                        plt.figure(fig.number)
                        plt.show()
                    else:
                        print("No ICP visualization data available.")
                else:
                    print("ICP visualization is not available.")
            
            icp_viz_button.on_clicked(visualize_icp_process)
            
            def compare_paths(event):
                # Create a figure to compare odometry and matched paths
                compare_fig, compare_ax = plt.subplots(figsize=(12, 10))
                
                # Plot the map
                if show_occupancy_grid:
                    compare_ax.imshow(
                        occupancy_grid.get_grid_for_display(),
                        cmap=cmap, norm=norm,
                        origin='lower',
                        extent=[-occupancy_grid.width/2, occupancy_grid.width/2, 
                               -occupancy_grid.height/2, occupancy_grid.height/2]
                    )
                
                # Plot both paths
                compare_ax.plot(odometry_path_x, odometry_path_y, 'r-', linewidth=2, label='Odometry Path')
                compare_ax.plot(robot_path_x, robot_path_y, 'b-', linewidth=2, label='Matched Path')
                
                # Plot start and end points
                compare_ax.scatter(odometry_path_x[0], odometry_path_y[0], c='green', s=100, marker='*', label='Start')
                compare_ax.scatter(odometry_path_x[-1], odometry_path_y[-1], c='red', s=100, marker='*', label='Odometry End')
                compare_ax.scatter(robot_path_x[-1], robot_path_y[-1], c='blue', s=100, marker='*', label='Matched End')
                
                # Overlay features if available
                if enable_feature_extraction and localizer and hasattr(localizer, 'feature_history'):
                    try:
                        # Add semi-transparent feature overlay
                        localizer.visualize_features_with_trajectory(ax=compare_ax, show_curvatures=False)
                        compare_ax.set_title('Path Comparison with Extracted Features')
                    except:
                        compare_ax.set_title('Odometry vs. Scan-Matched Path Comparison')
                else:
                    compare_ax.set_title('Odometry vs. Scan-Matched Path Comparison')
                
                # Add grid, labels, and legend
                compare_ax.grid(True)
                compare_ax.set_aspect('equal')
                compare_ax.set_xlabel('X (meters)')
                compare_ax.set_ylabel('Y (meters)')
                compare_ax.legend(loc='upper right')
                
                plt.tight_layout()
                plt.show()
            
            compare_button.on_clicked(compare_paths)
        
        # Add Loop Closure Visualization button handler
        if enable_loop_closure and enable_scan_matching:
            def show_loop_closures(event):
                if hasattr(localizer, 'loop_detector'):
                    if localizer.loop_closures_detected > 0:
                        visualize_loop_closure_results(localizer, occupancy_grid)
                    else:
                        print("No loop closures were detected.")
                else:
                    print("Loop closure detection is not available.")
            
            loop_viz_button.on_clicked(show_loop_closures)
        
        # Set occupancy grid plot properties
        ax2.set_title('Occupancy Grid Map')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.set_aspect('equal')
        
        # Add legend to grid map
        ax2.legend(loc='upper right')
        
        # Set the LiDAR scan plot in the first subplot
        ax = ax1
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a scatter plot for LiDAR points
    scan_x, scan_y = convert_scans_to_cartesian(
        parsed_data_list[-1]['scan_ranges'], angle_min, angle_max, parsed_data_list[-1]['pose'],
        flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
    )
    scatter = ax.scatter(scan_x, scan_y, c='blue', s=3, label='LiDAR Points')
    
    # Create a scatter plot for robot position
    last_x = robot_path_x[-1] if robot_path_x else 0
    last_y = robot_path_y[-1] if robot_path_y else 0
    robot_pos = ax.scatter(last_x, last_y, c='red', s=100, marker='*', label='Final Position')
    
    # Create a line for robot path
    path_line, = ax.plot(robot_path_x, robot_path_y, 'g-', linewidth=2, label='Robot Path')
    
    # Create a line for odometry path if scan matching is enabled
    if enable_scan_matching:
        odom_line, = ax.plot(odometry_path_x, odometry_path_y, 'r--', linewidth=1, alpha=0.6, label='Odometry Path')
    
    # Add scan matching status if enabled
    if enable_scan_matching:
        status_text = "Using Improved ICP Scan Matching"
        if enable_feature_extraction:
            status_text += " + Feature Extraction"
        
        match_text = ax.text(0.02, 0.98, status_text, 
                            transform=ax.transAxes, va='top', ha='left', 
                            color='green', fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.7))
        
        # Add loop closure status if enabled
        if enable_loop_closure:
            if hasattr(localizer, 'loop_closures_detected') and localizer.loop_closures_detected > 0:
                loop_text = ax.text(0.02, 0.94, f"Loop Closures Detected: {localizer.loop_closures_detected}", 
                                   transform=ax.transAxes, va='top', ha='left', 
                                   color='blue', fontsize=10,
                                   bbox=dict(facecolor='white', alpha=0.7))
            else:
                loop_text = ax.text(0.02, 0.94, "Loop Closure Detection Enabled", 
                                   transform=ax.transAxes, va='top', ha='left', 
                                   color='blue', fontsize=10,
                                   bbox=dict(facecolor='white', alpha=0.7))
        
        # Add feature extraction status if enabled
        if enable_feature_extraction and localizer:
            stats = localizer.get_feature_extraction_statistics()
            if stats.get('feature_extraction_enabled', False):
                feature_text = (f"Features: {stats['total_features_extracted']} total, "
                               f"{stats['average_features_per_scan']:.1f} avg/scan, "
                               f"{stats['average_feature_time']:.1f}ms avg time")
                
                y_pos = 0.90 if enable_loop_closure else 0.94
                feature_status_text = ax.text(0.02, y_pos, feature_text, 
                                             transform=ax.transAxes, va='top', ha='left', 
                                             color='purple', fontsize=10,
                                             bbox=dict(facecolor='white', alpha=0.7))
    
    # Add grid and labels
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('2D LiDAR Scan Visualization')
    
    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # If save_grid is enabled, save the final map
    if save_grid and show_occupancy_grid:
        # Generate a timestamp-based filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(maps_dir, f"final_map_{timestamp}")
        
        # Create path coordinates for saving
        robot_path_coords = list(zip(robot_path_x, robot_path_y))
        
        # Get start and current positions from path
        start_pos = (robot_path_x[0], robot_path_y[0]) if len(robot_path_x) > 0 else None
        current_pos = (robot_path_x[-1], robot_path_y[-1]) if len(robot_path_x) > 0 else None
        
        # Save the grid with robot path and positions
        saved_files = occupancy_grid.save_to_file(
            base_filename, 
            format=save_format, 
            include_metadata=True,
            robot_path=robot_path_coords,
            start_position=start_pos,
            current_position=current_pos
        )
        
        print(f"\nFinal occupancy grid map saved to:")
        for file in saved_files:
            print(f"  - {file}")
    
    # Save feature data if feature extraction was enabled
    if enable_feature_extraction and enable_scan_matching and localizer and hasattr(localizer, 'feature_extractor'):
        if save_grid:  # Only auto-save if user wants to save other data
            try:
                # Create feature output directory
                feature_dir = os.path.join(maps_dir, "features")
                if not os.path.exists(feature_dir):
                    os.makedirs(feature_dir)
                
                # Generate timestamp-based filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                feature_filename = os.path.join(feature_dir, f"features_{timestamp}")
                
                # Save in both formats
                localizer.feature_extractor.save_feature_data(feature_filename + ".json", format='json')
                localizer.feature_extractor.save_feature_data(feature_filename + ".csv", format='csv')
                
                print(f"\nFeature data auto-saved to:")
                print(f"  - {feature_filename}.json")
                print(f"  - {feature_filename}.csv")
            except Exception as e:
                print(f"Warning: Could not auto-save feature data: {e}")
            
    # Return key components for further use if needed
    return {
        'occupancy_grid': occupancy_grid,
        'localizer': localizer,
        'robot_path': list(zip(robot_path_x, robot_path_y)),
        'scan_data': parsed_data_list,
        'feature_extraction_enabled': enable_feature_extraction
    }


def main():
    """
    Main function to parse arguments and run the visualization
    """
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='LiDAR Visualization, Localization, Loop Closure, and Feature Extraction')
    
    # Add arguments
    parser.add_argument('--file', type=str, default="../dataset/raw_data/laser_data_synchronized_short_u_turn_fast_processed_reduced180.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max_entries', type=int, default=30,
                       help='Maximum number of entries to rea0d from the file')
    parser.add_argument('--grid', action='store_true', default=True,
                       help='Enable occupancy grid mapping')
    parser.add_argument('--resolution', type=float, default=0.05,
                       help='Resolution of the occupancy grid in meters')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save the final occupancy grid map')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'npy', 'csv', 'all'],
                       help='Format to save the grid')
    parser.add_argument('--scan_matching', action='store_true', default=True,
                       help='Enable scan matching localization algorithm')
    parser.add_argument('--loop_closure', action='store_true', default=True,
                       help='Enable loop closure detection')
    parser.add_argument('--feature_extraction', action='store_true', default=True,
                       help='Enable feature extraction alongside ICP scan matching')
    parser.add_argument('--feature_association', action='store_true', default=True,
                       help='Enable feature association and hybrid pose estimation')
    parser.add_argument('--rebuild_map', action='store_true', default=True,
                       help='Rebuild map after loop closure optimization')
    parser.add_argument('--debug', type=int, default=1, choices=[0, 1, 2, 3],
                       help='Debug level (0=none, 1=basic, 2=detailed, 3=verbose)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the visualization
    visualize_lidar_data_realtime(
        file_path=args.file,
        max_entries=args.max_entries,
        show_occupancy_grid=args.grid,
        grid_resolution=args.resolution,
        save_grid=args.save,
        save_format=args.format,
        enable_scan_matching=args.scan_matching,
        enable_loop_closure=args.loop_closure,
        enable_feature_extraction=args.feature_extraction,
        enable_feature_association=args.feature_association,
        rebuild_map=args.rebuild_map
    )

# Main execution
if __name__ == "__main__":
    main()