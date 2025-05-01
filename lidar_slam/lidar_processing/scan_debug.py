#!/usr/bin/env python3
"""
ScanMatcher_OGBased Debugger

This script debugs the standalone ScanMatcher_OGBased implementation.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import time
from matplotlib.animation import FuncAnimation

# Import the OccupancyGrid and ScanMatcher classes
# Make sure these files are in the same directory or adjust these imports
from Utils.OccupancyGrid import OccupancyGrid
from Utils.ScanMatcher_OGBased import ScanMatcher, updateEstimatedPose

def debug_scan_matcher(json_file, max_entries=100, visualize=True, save_dir='sm_debug_output'):
    """
    Debug the ScanMatcher_OGBased implementation with visualization
    
    Args:
        json_file (str): Path to the JSON file with sensor data
        max_entries (int): Maximum number of entries to process
        visualize (bool): Whether to visualize the results
        save_dir (str): Directory to save debug outputs
    """
    print(f"Reading data from: {json_file}")
    
    # Read the data
    if not os.path.exists(json_file):
        print(f"ERROR: File {json_file} does not exist!")
        return
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        if 'map' in data:
            sensor_data = data['map']
        else:
            sensor_data = data
            
        print(f"Successfully loaded data with {len(sensor_data)} entries")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Get sorted keys to process in order
    sorted_keys = sorted(sensor_data.keys())[:max_entries]
    
    if not sorted_keys:
        print("No data found in the file!")
        return
    
    # Get parameters from first entry
    first_entry = sensor_data[sorted_keys[0]]
    
    # Check data format
    if not all(key in first_entry for key in ['x', 'y', 'theta', 'range']):
        print(f"ERROR: Data format is not as expected. Keys found: {list(first_entry.keys())}")
        return
    
    # Extract parameters needed for grid initialization
    num_samples_per_rev = len(first_entry['range'])
    init_xy = {'x': first_entry['x'], 'y': first_entry['y']}
    
    print("\nLiDAR Data Info:")
    print(f"  Samples per scan: {num_samples_per_rev}")
    print(f"  Initial position: x={init_xy['x']:.4f}, y={init_xy['y']:.4f}, theta={first_entry['theta']:.4f}")
    
    # Create directory for saving debug outputs
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Test different scan matcher configurations
    sm_configs = [
        {
            'name': 'Default',
            'unit_grid_size': 0.05,
            'search_radius': 1.4,
            'search_half_rad': 0.25,
            'scan_sigma': 2,
            'move_r_sigma': 0.1,
            'max_move_deviation': 0.25,
            'turn_sigma': 0.3,
            'mismatch_prob': 0.15,
            'coarse_factor': 5
        },
        {
            'name': 'Enhanced',
            'unit_grid_size': 0.1,
            'search_radius': 2.0,
            'search_half_rad': 0.5,
            'scan_sigma': 3,
            'move_r_sigma': 0.2,
            'max_move_deviation': 0.5,
            'turn_sigma': 0.5,
            'mismatch_prob': 0.05,
            'coarse_factor': 4
        },
        {
            'name': 'Aggressive',
            'unit_grid_size': 0.2,
            'search_radius': 3.0,
            'search_half_rad': 0.7,
            'scan_sigma': 4,
            'move_r_sigma': 0.3,
            'max_move_deviation': 1.0,
            'turn_sigma': 0.7,
            'mismatch_prob': 0.01,
            'coarse_factor': 3
        }
    ]
    
    # Common parameters for all configurations
    init_map_x_length = 20  # meters - larger than default to accommodate more area
    init_map_y_length = 20  # meters
    lidar_fov = math.pi   # Field of view in radians
    lidar_max_range = 10.0  # Maximum range in meters
    wall_thickness = 0.15    # Wall thickness in meters
    
    sm_results = []
    
    # Process data with each configuration
    for config in sm_configs:
        print(f"\nTesting {config['name']} configuration")
        print(f"  Grid resolution: {config['unit_grid_size']}m/cell")
        print(f"  Search radius: {config['search_radius']}m, angle: {config['search_half_rad']} rad")
        
        # Initialize grid and scan matcher
        og = OccupancyGrid(
            mapXLength=init_map_x_length,
            mapYLength=init_map_y_length,
            initXY=init_xy,
            unitGridSize=config['unit_grid_size'],
            lidarFOV=lidar_fov,
            numSamplesPerRev=num_samples_per_rev,
            lidarMaxRange=lidar_max_range,
            wallThickness=wall_thickness
        )
        
        sm = ScanMatcher(
            og=og,
            searchRadius=config['search_radius'],
            searchHalfRad=config['search_half_rad'],
            scanSigmaInNumGrid=config['scan_sigma'],
            moveRSigma=config['move_r_sigma'],
            maxMoveDeviation=config['max_move_deviation'],
            turnSigma=config['turn_sigma'],
            missMatchProbAtCoarse=config['mismatch_prob'],
            coarseFactor=config['coarse_factor']
        )
        
        # Initialize tracking variables
        count = 0
        match_results = []
        
        # For trajectory visualization
        x_trajectory_raw = []
        y_trajectory_raw = []
        x_trajectory_corrected = []
        y_trajectory_corrected = []
        
        # Previous readings for tracking
        prev_raw_moving_theta = None
        prev_matched_moving_theta = None
        
        start_time = time.time()
        
        # Process entries
        for key in sorted_keys:
            count += 1
            
            if count % 10 == 0:
                print(f"  Processing entry {count}/{len(sorted_keys)}")
            
            try:
                if count == 1:
                    # First reading - just initialize
                    matched_reading = sensor_data[key].copy()
                    confidence = 1.0
                    
                    # Add to raw and corrected trajectory
                    x_trajectory_raw.append(matched_reading['x'])
                    y_trajectory_raw.append(matched_reading['y'])
                    x_trajectory_corrected.append(matched_reading['x'])
                    y_trajectory_corrected.append(matched_reading['y'])
                else:
                    # Get current reading
                    current_raw_reading = sensor_data[key].copy()
                    
                    # Add to raw trajectory
                    x_trajectory_raw.append(current_raw_reading['x'])
                    y_trajectory_raw.append(current_raw_reading['y'])
                    
                    # Estimate pose using odometry
                    estimated_reading, est_moving_dist, est_moving_theta, raw_moving_theta = \
                        updateEstimatedPose(
                            current_raw_reading, 
                            prev_matched_reading,
                            prev_raw_reading, 
                            prev_raw_moving_theta,
                            prev_matched_moving_theta
                        )
                    
                    # Perform scan matching
                    matched_reading, confidence = sm.matchScan(
                        estimated_reading, 
                        est_moving_dist,
                        est_moving_theta, 
                        count
                    )
                    
                    # Store match info
                    match_info = {
                        'count': count,
                        'timestamp': time.time(),
                        'raw_reading': current_raw_reading.copy(),
                        'estimated_reading': estimated_reading.copy(),
                        'matched_reading': matched_reading.copy(),
                        'confidence': confidence,
                        'est_moving_dist': est_moving_dist,
                        'est_moving_theta': est_moving_theta
                    }
                    match_results.append(match_info)
                    
                    # Update corrected trajectory
                    x_trajectory_corrected.append(matched_reading['x'])
                    y_trajectory_corrected.append(matched_reading['y'])
                    
                    # Update moving theta tracking
                    prev_raw_moving_theta = raw_moving_theta
                    
                    if count > 2:
                        # Calculate matched movement theta
                        dx_matched = matched_reading['x'] - x_trajectory_corrected[-2]
                        dy_matched = matched_reading['y'] - y_trajectory_corrected[-2]
                        move_dist = math.sqrt(dx_matched**2 + dy_matched**2)
                        
                        if move_dist > 0.05:  # Only track significant movements
                            prev_matched_moving_theta = math.atan2(dy_matched, dx_matched)
                
                # Update occupancy grid with matched reading
                og.updateOccupancyGrid(matched_reading)
                
                # Store readings for next iteration
                prev_matched_reading = matched_reading.copy()
                prev_raw_reading = sensor_data[key].copy()
                
            except Exception as e:
                print(f"  Error processing entry {count}: {e}")
                # Skip this entry and continue with the next one
        
        processing_time = time.time() - start_time
        
        # Calculate success rate
        if match_results:
            successful_matches = sum(1 for r in match_results if r['confidence'] > 0.1)
            success_rate = successful_matches / len(match_results) * 100
        else:
            successful_matches = 0
            success_rate = 0
        
        print(f"  Processed {count} entries in {processing_time:.2f} seconds")
        print(f"  Success rate: {successful_matches}/{len(match_results)} ({success_rate:.1f}%)")
        
        # Save result
        sm_results.append({
            'config': config,
            'og': og,
            'match_results': match_results,
            'x_trajectory_raw': x_trajectory_raw,
            'y_trajectory_raw': y_trajectory_raw,
            'x_trajectory_corrected': x_trajectory_corrected,
            'y_trajectory_corrected': y_trajectory_corrected,
            'processing_time': processing_time,
            'success_rate': success_rate
        })
    
    # Find best configuration based on success rate
    if sm_results:
        best_result = max(sm_results, key=lambda r: r['success_rate'])
        print(f"\nBest configuration: {best_result['config']['name']} with {best_result['success_rate']:.1f}% success rate")
    else:
        print("\nNo results to evaluate")
        return None
    
    if visualize and sm_results:
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # 1. Compare trajectories for each configuration
        fig, axes = plt.subplots(len(sm_results), 1, figsize=(10, 5 * len(sm_results)))
        
        if len(sm_results) == 1:
            axes = [axes]  # Make it iterable for a single configuration
        
        for i, result in enumerate(sm_results):
            config = result['config']
            
            # Plot raw vs corrected trajectories
            axes[i].plot(
                result['x_trajectory_raw'], 
                result['y_trajectory_raw'], 
                'r-', linewidth=1, alpha=0.7, label='Raw Odometry'
            )
            axes[i].plot(
                result['x_trajectory_corrected'], 
                result['y_trajectory_corrected'], 
                'g-', linewidth=2, label='Corrected Path'
            )
            
            # Highlight start and end points
            axes[i].scatter(
                result['x_trajectory_raw'][0], 
                result['y_trajectory_raw'][0], 
                c='blue', s=100, marker='*', label='Start'
            )
            axes[i].scatter(
                result['x_trajectory_corrected'][-1], 
                result['y_trajectory_corrected'][-1], 
                c='purple', s=100, marker='*', label='End'
            )
            
            # Connect corresponding points to show corrections
            for j in range(0, len(result['x_trajectory_raw']), 10):  # Show every 10th point
                if j < len(result['x_trajectory_corrected']):
                    axes[i].plot(
                        [result['x_trajectory_raw'][j], result['x_trajectory_corrected'][j]],
                        [result['y_trajectory_raw'][j], result['y_trajectory_corrected'][j]],
                        'k--', alpha=0.3
                    )
            
            axes[i].set_title(f"{config['name']} Configuration - Success Rate: {result['success_rate']:.1f}%")
            axes[i].set_xlabel('X (meters)')
            axes[i].set_ylabel('Y (meters)')
            axes[i].grid(True)
            axes[i].legend(loc='upper right')
            axes[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'trajectory_comparison.png'), dpi=300)
        print(f"  Saved trajectory comparison to {os.path.join(save_dir, 'trajectory_comparison.png')}")
        
        # 2. Plot confidence over time for each configuration
        fig, axes = plt.subplots(len(sm_results), 1, figsize=(12, 4 * len(sm_results)))
        
        if len(sm_results) == 1:
            axes = [axes]  # Make it iterable for a single configuration
        
        for i, result in enumerate(sm_results):
            config = result['config']
            
            if result['match_results']:
                # Extract confidence values
                confidences = [r['confidence'] for r in result['match_results']]
                
                # Plot confidence over time
                axes[i].plot(confidences, 'b-', linewidth=1.5)
                axes[i].axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
                
                # Add statistics
                mean_conf = np.mean(confidences)
                median_conf = np.median(confidences)
                
                axes[i].set_title(
                    f"{config['name']} Configuration - Mean Confidence: {mean_conf:.4f}, Median: {median_conf:.4f}"
                )
                axes[i].set_xlabel('Scan Number')
                axes[i].set_ylabel('Confidence Score')
                axes[i].grid(True)
                axes[i].legend()
            else:
                axes[i].text(0.5, 0.5, 'No match results available', ha='center', va='center')
                axes[i].set_title(f"{config['name']} Configuration")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_comparison.png'), dpi=300)
        print(f"  Saved confidence comparison to {os.path.join(save_dir, 'confidence_comparison.png')}")
        
        # 3. Plot final occupancy grid maps
        fig, axes = plt.subplots(1, len(sm_results), figsize=(6 * len(sm_results), 6))
        
        if len(sm_results) == 1:
            axes = [axes]  # Make it iterable for a single configuration
        
        for i, result in enumerate(sm_results):
            config = result['config']
            og = result['og']
            
            # Get occupancy grid
            occupancy_map = og.occupancyGridVisited / og.occupancyGridTotal
            occupancy_map = np.flipud(1 - occupancy_map)  # Flip for visualization
            
            # Get map extents
            x_min = og.mapXLim[0]
            x_max = og.mapXLim[1]
            y_min = og.mapYLim[0]
            y_max = og.mapYLim[1]
            
            # Plot grid
            axes[i].imshow(occupancy_map, cmap='gray', extent=[x_min, x_max, y_min, y_max])
            
            # Plot corrected path
            axes[i].plot(
                result['x_trajectory_corrected'], 
                result['y_trajectory_corrected'], 
                'r-', linewidth=1.5, label='Robot Path'
            )
            
            # Highlight start and end points
            axes[i].scatter(
                result['x_trajectory_corrected'][0], 
                result['y_trajectory_corrected'][0], 
                c='green', s=100, marker='*', label='Start'
            )
            axes[i].scatter(
                result['x_trajectory_corrected'][-1], 
                result['y_trajectory_corrected'][-1], 
                c='blue', s=100, marker='*', label='End'
            )
            
            axes[i].set_title(f"{config['name']} Map - Success Rate: {result['success_rate']:.1f}%")
            axes[i].set_xlabel('X (meters)')
            axes[i].set_ylabel('Y (meters)')
            axes[i].grid(True)
            axes[i].legend(loc='upper right')
            
            # Set consistent aspect ratio
            axes[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'grid_maps_comparison.png'), dpi=300)
        print(f"  Saved grid maps comparison to {os.path.join(save_dir, 'grid_maps_comparison.png')}")
        
        # 4. Visualize individual scan matches
        # Create an animation of the scan matching process for the best configuration
        print("\nCreating animation of scan matching process...")
        
        # Get the best configuration's results
        best_result = max(sm_results, key=lambda r: r['success_rate'])
        
        # We'll use the match results to recreate the scan matching process
        match_results = best_result['match_results']
        
        if match_results:
            # Initialize occupancy grid and scan matcher with best configuration
            config = best_result['config']
            
            # Create figure for animation
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Add subplot for confidence history
            confidence_ax = plt.axes([0.15, 0.02, 0.7, 0.1])  # [left, bottom, width, height]
            confidence_line, = confidence_ax.plot([], [], 'b-')
            confidence_ax.axhline(y=0.1, color='r', linestyle='--')
            confidence_ax.set_xlim(0, len(match_results))
            confidence_ax.set_ylim(0, 1)
            confidence_ax.set_xlabel('Scan Number')
            confidence_ax.set_ylabel('Confidence')
            
            # For storing confidence history
            confidence_history = []
            
            # Animation update function
            def update(frame):
                ax.clear()
                
                if frame < len(match_results):
                    result = match_results[frame]
                    
                    # Get readings
                    raw_reading = result['raw_reading']
                    estimated_reading = result['estimated_reading']
                    matched_reading = result['matched_reading']
                    confidence = result['confidence']
                    
                    # Add to confidence history
                    confidence_history.append(confidence)
                    confidence_line.set_data(range(len(confidence_history)), confidence_history)
                    
                    # Plot raw, estimated, and matched positions
                    ax.scatter(
                        raw_reading['x'], raw_reading['y'], 
                        c='red', s=100, marker='o', label='Raw Position'
                    )
                    ax.scatter(
                        estimated_reading['x'], estimated_reading['y'], 
                        c='blue', s=100, marker='x', label='Estimated Position'
                    )
                    ax.scatter(
                        matched_reading['x'], matched_reading['y'], 
                        c='green', s=100, marker='*', label='Matched Position'
                    )
                    
                    # Draw arrows to show orientations
                    arrow_len = 0.5
                    
                    # Raw orientation
                    dx = arrow_len * math.cos(raw_reading['theta'])
                    dy = arrow_len * math.sin(raw_reading['theta'])
                    ax.arrow(
                        raw_reading['x'], raw_reading['y'], dx, dy,
                        head_width=0.1, head_length=0.1, fc='red', ec='red'
                    )
                    
                    # Estimated orientation
                    dx = arrow_len * math.cos(estimated_reading['theta'])
                    dy = arrow_len * math.sin(estimated_reading['theta'])
                    ax.arrow(
                        estimated_reading['x'], estimated_reading['y'], dx, dy,
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue'
                    )
                    
                    # Matched orientation
                    dx = arrow_len * math.cos(matched_reading['theta'])
                    dy = arrow_len * math.sin(matched_reading['theta'])
                    ax.arrow(
                        matched_reading['x'], matched_reading['y'], dx, dy,
                        head_width=0.1, head_length=0.1, fc='green', ec='green'
                    )
                    
                    # Plot connections between positions
                    ax.plot(
                        [raw_reading['x'], estimated_reading['x']],
                        [raw_reading['y'], estimated_reading['y']],
                        'b--', alpha=0.5
                    )
                    ax.plot(
                        [estimated_reading['x'], matched_reading['x']],
                        [estimated_reading['y'], matched_reading['y']],
                        'g--', alpha=0.5
                    )
                    
                    # Plot partial trajectory up to this point
                    x_traj = best_result['x_trajectory_corrected'][:frame+2]  # +2 to include starting point
                    y_traj = best_result['y_trajectory_corrected'][:frame+2]
                    ax.plot(x_traj, y_traj, 'k-', linewidth=1, alpha=0.7)
                    
                    # Add information text
                    info_text = f"Scan: {result['count']}\n"
                    info_text += f"Confidence: {confidence:.4f}\n"
                    info_text += f"Est. Movement: {result['est_moving_dist']:.3f}m\n"
                    if result['est_moving_theta'] is not None:
                        info_text += f"Est. Direction: {math.degrees(result['est_moving_theta']):.1f}°"
                    
                    ax.text(
                        0.02, 0.98, info_text, transform=ax.transAxes,
                        va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7)
                    )
                    
                    # Set title
                    ax.set_title(f"Scan Matching - {config['name']} Configuration")
                    
                    # Set axis properties
                    ax.grid(True)
                    ax.set_aspect('equal')
                    ax.legend(loc='upper right')
                    
                    # Adjust view to follow the robot
                    margin = 2.0  # meters
                    ax.set_xlim(matched_reading['x'] - margin, matched_reading['x'] + margin)
                    ax.set_ylim(matched_reading['y'] - margin, matched_reading['y'] + margin)
                
                return [ax, confidence_line]
            
            # Create animation
            ani = FuncAnimation(
                fig, update, frames=min(100, len(match_results)), 
                blit=False, interval=200
            )
            
            # Save animation
            ani.save(os.path.join(save_dir, 'scan_matching_process.gif'), writer='pillow', fps=5)
            print(f"  Saved scan matching animation to {os.path.join(save_dir, 'scan_matching_process.gif')}")
        
        # 5. Visualize a specific scan match in detail
        # Find a scan match with reasonable confidence (not too high, not too low)
        good_matches = [r for r in best_result['match_results'] if 0.2 <= r['confidence'] <= 0.8]
        
        if good_matches:
            # Pick a match in the middle of the list
            sample_match = good_matches[len(good_matches) // 2]
            
            print("\nVisualizing details of a specific scan match...")
            
            # Let's visualize the search space of this match
            # For this, we need to recreate a specific match
            
            # Reset to the best configuration
            config = best_result['config']
            og = OccupancyGrid(
                mapXLength=init_map_x_length,
                mapYLength=init_map_y_length,
                initXY=init_xy,
                unitGridSize=config['unit_grid_size'],
                lidarFOV=lidar_fov,
                numSamplesPerRev=num_samples_per_rev,
                lidarMaxRange=lidar_max_range,
                wallThickness=wall_thickness
            )
            
            sm = ScanMatcher(
                og=og,
                searchRadius=config['search_radius'],
                searchHalfRad=config['search_half_rad'],
                scanSigmaInNumGrid=config['scan_sigma'],
                moveRSigma=config['move_r_sigma'],
                maxMoveDeviation=config['max_move_deviation'],
                turnSigma=config['turn_sigma'],
                missMatchProbAtCoarse=config['mismatch_prob'],
                coarseFactor=config['coarse_factor']
            )
            
            # Process some scans to build a partial map
            count = 0
            
            # Find index of the sample match
            sample_idx = [i for i, r in enumerate(best_result['match_results']) 
                         if r['count'] == sample_match['count']][0]
            
            # Process scans up to the sample
            for i in range(sample_idx + 20):  # +20 to process a few more for a better map
                if i < len(best_result['match_results']):
                    result = best_result['match_results'][i]
                    og.updateOccupancyGrid(result['matched_reading'])
            
            # Now let's visualize the search space and the match result
            estimated_reading = sample_match['estimated_reading']
            matched_reading = sample_match['matched_reading']
            est_moving_dist = sample_match['est_moving_dist']
            est_moving_theta = sample_match['est_moving_theta']
            
            # Create a coarse search space
            coarse_search_step = config['coarse_factor'] * config['unit_grid_size']
            coarse_sigma = config['scan_sigma'] / config['coarse_factor']
            
            x_range, y_range, prob_space = sm.frameSearchSpace(
                estimated_reading['x'], 
                estimated_reading['y'], 
                coarse_search_step,
                coarse_sigma, 
                config['mismatch_prob']
            )
            
            # Create a figure to visualize the search space
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot probability space
            im = ax.imshow(prob_space, origin='lower', cmap='viridis',
                         extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
            plt.colorbar(im, ax=ax, label='Log Probability')
            
            # Plot estimated position
            ax.scatter(
                estimated_reading['x'], estimated_reading['y'], 
                c='blue', s=100, marker='x', label='Estimated Position'
            )
            
            # Plot matched position
            ax.scatter(
                matched_reading['x'], matched_reading['y'], 
                c='green', s=100, marker='*', label='Matched Position'
            )
            
            # Draw arrows for orientations
            arrow_len = 0.5
            
            # Estimated orientation
            dx = arrow_len * math.cos(estimated_reading['theta'])
            dy = arrow_len * math.sin(estimated_reading['theta'])
            ax.arrow(
                estimated_reading['x'], estimated_reading['y'], dx, dy,
                head_width=0.1, head_length=0.1, fc='blue', ec='blue'
            )
            
            # Matched orientation
            dx = arrow_len * math.cos(matched_reading['theta'])
            dy = arrow_len * math.sin(matched_reading['theta'])
            ax.arrow(
                matched_reading['x'], matched_reading['y'], dx, dy,
                head_width=0.1, head_length=0.1, fc='green', ec='green'
            )
            
            # Connect estimated and matched positions
            ax.plot(
                [estimated_reading['x'], matched_reading['x']],
                [estimated_reading['y'], matched_reading['y']],
                'w--', linewidth=2
            )
            
            # Add search space boundary
            search_radius = config['search_radius']
            circle = plt.Circle(
                (estimated_reading['x'], estimated_reading['y']), 
                search_radius, 
                fill=False, color='r', linestyle='--'
            )
            ax.add_artist(circle)
            
            # Add information text
            info_text = f"Scan: {sample_match['count']}\n"
            info_text += f"Confidence: {sample_match['confidence']:.4f}\n"
            info_text += f"Est. Movement: {est_moving_dist:.3f}m\n"
            if est_moving_theta is not None:
                info_text += f"Est. Direction: {math.degrees(est_moving_theta):.1f}°\n"
            info_text += f"Delta Position: {matched_reading['x'] - estimated_reading['x']:.3f}, "
            info_text += f"{matched_reading['y'] - estimated_reading['y']:.3f}\n"
            info_text += f"Delta Theta: {math.degrees(matched_reading['theta'] - estimated_reading['theta']):.1f}°"
            
            ax.text(
                0.02, 0.98, info_text, transform=ax.transAxes,
                va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7)
            )
            
            ax.set_title(f"Scan Matching Search Space - Confidence: {sample_match['confidence']:.4f}")
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.grid(True)
            ax.legend(loc='upper right')
            
            plt.savefig(os.path.join(save_dir, 'search_space_detail.png'), dpi=300)
            print(f"  Saved search space detail to {os.path.join(save_dir, 'search_space_detail.png')}")
            
            # 6. Feature extraction visualization
            # Try to extract wall features from the best occupancy grid
            og = best_result['og']
            
            # Get occupancy grid
            occupancy_map = og.occupancyGridVisited / og.occupancyGridTotal
            
            # Threshold to get binary map
            binary_map = occupancy_map > 0.6
            
            # Plot original and binary map side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot original occupancy grid
            occupancy_map_viz = np.flipud(1 - occupancy_map)  # Flip for visualization
            ax1.imshow(occupancy_map_viz, cmap='gray', 
                     extent=[og.mapXLim[0], og.mapXLim[1], og.mapYLim[0], og.mapYLim[1]])
            ax1.set_title("Occupancy Grid (Probability)")
            ax1.set_xlabel('X (meters)')
            ax1.set_ylabel('Y (meters)')
            ax1.grid(True)
            
            # Plot binary map
            binary_map_viz = np.flipud(1 - binary_map)  # Flip for visualization
            ax2.imshow(binary_map_viz, cmap='gray',
                     extent=[og.mapXLim[0], og.mapXLim[1], og.mapYLim[0], og.mapYLim[1]])
            ax2.set_title("Binary Occupancy Grid (Thresholded)")
            ax2.set_xlabel('X (meters)')
            ax2.set_ylabel('Y (meters)')
            ax2.grid(True)
            
            # Plot robot path on both
            ax1.plot(
                best_result['x_trajectory_corrected'], 
                best_result['y_trajectory_corrected'], 
                'r-', linewidth=1.5
            )
            ax2.plot(
                best_result['x_trajectory_corrected'], 
                best_result['y_trajectory_corrected'], 
                'r-', linewidth=1.5
            )
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'binary_map_comparison.png'), dpi=300)
            print(f"  Saved binary map comparison to {os.path.join(save_dir, 'binary_map_comparison.png')}")
        
        plt.show()
    
    # Return the results for further use
    return sm_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Debug ScanMatcher_OGBased implementation')
    parser.add_argument('--file', '-f', type=str, help='Path to the JSON file with sensor data', default='../../DataSet/DataPreprocessed/intel-gfs')
    parser.add_argument('--max-entries', '-m', type=int, default=1000,
                      help='Maximum number of entries to process')
    parser.add_argument('--no-viz', action='store_true', default=False,
                      help='Disable visualization')
    parser.add_argument('--output-dir', '-o', type=str, default='maps',
                      help='Directory to save debug outputs')
    
    args = parser.parse_args()
    debug_scan_matcher(args.file, args.max_entries, not args.no_viz, args.output_dir)