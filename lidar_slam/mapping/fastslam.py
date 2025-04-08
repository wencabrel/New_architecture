#!/usr/bin/env python3
"""
FastSLAM with Scan Matcher Example

This example demonstrates an implementation of FastSLAM 1.0 that incorporates
the scan matcher for improved pose estimation and particle weighting.
"""

import os
import sys
import numpy as np
import math
import time
import random
from copy import deepcopy

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lidar_slam.lidar_processing import data_parser, scan_converter
from lidar_slam.mapping import occupancy_grid
from lidar_slam.mapping.scan_matcher import ScanMatcher
from lidar_slam.utils import file_utils

class Particle:
    """Particle for FastSLAM algorithm"""
    
    def __init__(self, pose, weight=1.0, grid_resolution=0.05, grid_width=20, grid_height=20):
        """
        Initialize a particle with pose and map
        
        Args:
            pose (dict): Particle pose with 'x', 'y', 'theta' keys
            weight (float): Particle weight
            grid_resolution (float): Resolution of the occupancy grid
            grid_width (float): Width of the occupancy grid in meters
            grid_height (float): Height of the occupancy grid in meters
        """
        self.pose = pose.copy()
        self.weight = weight
        
        # Each particle maintains its own map
        self.grid = occupancy_grid.OccupancyGrid(
            resolution=grid_resolution,
            width=grid_width,
            height=grid_height
        )
        
        # Keep track of path
        self.path = [(pose['x'], pose['y'])]
    
    def update_pose(self, new_pose):
        """
        Update particle pose
        
        Args:
            new_pose (dict): New pose with 'x', 'y', 'theta' keys
        """
        self.pose = new_pose.copy()
        
        # Add to path
        self.path.append((new_pose['x'], new_pose['y']))
    
    def predict_pose(self, dx, dy, dtheta, noise_level=0.1):
        """
        Predict new pose with motion model and noise
        
        Args:
            dx (float): Change in x position
            dy (float): Change in y position
            dtheta (float): Change in orientation
            noise_level (float): Noise level for motion model
            
        Returns:
            dict: Predicted pose
        """
        # Add noise to motion update (simple Gaussian noise)
        pos_noise = noise_level * self.grid.resolution
        angle_noise = noise_level * 0.05  # radians
        
        predicted_pose = {
            'x': self.pose['x'] + dx + random.gauss(0, pos_noise),
            'y': self.pose['y'] + dy + random.gauss(0, pos_noise),
            'theta': self.pose['theta'] + dtheta + random.gauss(0, angle_noise)
        }
        
        # Normalize theta to [-pi, pi]
        predicted_pose['theta'] = math.atan2(math.sin(predicted_pose['theta']), math.cos(predicted_pose['theta']))
        
        return predicted_pose
    
    def update_map(self, scan_x, scan_y):
        """
        Update particle's map with scan data
        
        Args:
            scan_x (list): X coordinates of scan points in world frame
            scan_y (list): Y coordinates of scan points in world frame
        """
        self.grid.update_grid(
            self.pose['x'],
            self.pose['y'],
            scan_x,
            scan_y
        )

class FastSLAM:
    """FastSLAM 1.0 implementation with scan matching"""
    
    def __init__(self, num_particles=30, grid_resolution=0.05, grid_width=20, grid_height=20):
        """
        Initialize FastSLAM
        
        Args:
            num_particles (int): Number of particles
            grid_resolution (float): Resolution of the occupancy grid
            grid_width (float): Width of the occupancy grid in meters
            grid_height (float): Height of the occupancy grid in meters
        """
        self.num_particles = num_particles
        self.grid_resolution = grid_resolution
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # For scan conversion
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        self.config = {
            'flip_x': False,
            'flip_y': False,
            'reverse_scan': True,
            'flip_theta': False
        }
        
        # Create scan matcher for pose correction
        self.scan_matcher = ScanMatcher(
            grid_resolution=grid_resolution,
            search_radius=0.5,
            search_angle=0.2
        )
        
        # Particles list
        self.particles = []
        
        # Best particle (will be updated during resampling)
        self.best_particle = None
        
    def initialize(self, initial_pose):
        """
        Initialize particles around the initial pose
        
        Args:
            initial_pose (dict): Initial pose with 'x', 'y', 'theta' keys
        """
        self.particles = []
        
        # Create particles with small random variations
        for _ in range(self.num_particles):
            # Add small noise to initial pose
            pose = {
                'x': initial_pose['x'] + random.gauss(0, 0.1),
                'y': initial_pose['y'] + random.gauss(0, 0.1),
                'theta': initial_pose['theta'] + random.gauss(0, 0.05)
            }
            
            # Create particle with uniform weight
            particle = Particle(
                pose=pose,
                weight=1.0 / self.num_particles,
                grid_resolution=self.grid_resolution,
                grid_width=self.grid_width,
                grid_height=self.grid_height
            )
            
            self.particles.append(particle)
        
        # Initialize best particle
        self.best_particle = self.particles[0]
    
    def update(self, scan_data, prev_scan_data=None):
        """
        Update FastSLAM with a new scan
        
        Args:
            scan_data (dict): Current LiDAR scan data
            prev_scan_data (dict): Previous LiDAR scan data for odometry
        """
        # If this is the first scan, just initialize
        if prev_scan_data is None:
            if not self.particles:
                self.initialize(scan_data['pose'])
                
                # Update maps with first scan for all particles
                for particle in self.particles:
                    self._update_particle_with_scan(particle, scan_data)
            return
        
        # Calculate odometry (change in pose)
        dx = scan_data['pose']['x'] - prev_scan_data['pose']['x']
        dy = scan_data['pose']['y'] - prev_scan_data['pose']['y']
        dtheta = scan_data['pose']['theta'] - prev_scan_data['pose']['theta']
        
        # Update all particles
        for particle in self.particles:
            # 1. Predict new pose with motion model
            predicted_pose = particle.predict_pose(dx, dy, dtheta)
            
            # 2. Use scan matcher to refine the predicted pose
            corrected_pose, match_score = self.scan_matcher.match_scan(
                scan_data, 
                particle.grid, 
                predicted_pose
            )
            
            # 3. Update particle pose with corrected pose
            particle.update_pose(corrected_pose)
            
            # 4. Update particle's map with the scan
            self._update_particle_with_scan(particle, scan_data)
            
            # 5. Update particle weight based on match score
            particle.weight *= (match_score + 0.1)  # Add small constant to avoid zero weights
        
        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # If all weights are zero, reset to uniform
            for particle in self.particles:
                particle.weight = 1.0 / self.num_particles
        
        # Find best particle
        self.best_particle = max(self.particles, key=lambda p: p.weight)
        
        # Resample particles if the effective sample size is too small
        n_eff = 1.0 / sum(p.weight**2 for p in self.particles)
        if n_eff < self.num_particles / 2:
            self._resample()
    
    def _update_particle_with_scan(self, particle, scan_data):
        """
        Update a particle with scan data
        
        Args:
            particle: Particle to update
            scan_data: LiDAR scan data
        """
        # Convert scan to world coordinates using particle's pose
        scan_x, scan_y = scan_converter.convert_scans_to_cartesian(
            scan_data['scan_ranges'],
            self.angle_min,
            self.angle_max,
            particle.pose,
            **self.config
        )
        
        # Update particle's map
        particle.update_map(scan_x, scan_y)
    
    def _resample(self):
        """Resample particles based on weights using low variance resampling"""
        new_particles = []
        
        # Determine how many of the best particles to always keep (elitism)
        elite_count = max(1, self.num_particles // 10)  # 10% of particles
        elite_particles = sorted(self.particles, key=lambda p: p.weight, reverse=True)[:elite_count]
        
        # Add elite particles to new set
        for elite in elite_particles:
            new_particles.append(elite)
        
        # Resample remaining particles
        remaining_count = self.num_particles - elite_count
        step = 1.0 / remaining_count
        r = random.uniform(0, step)
        c = self.particles[0].weight
        i = 0
        
        for m in range(remaining_count):
            u = r + m * step
            while u > c:
                i += 1
                if i >= len(self.particles):
                    i = len(self.particles) - 1
                c += self.particles[i].weight
            
            # Create a new particle (deep copy of map is expensive, so we only copy the pose)
            new_particle = Particle(
                self.particles[i].pose,
                weight=1.0 / self.num_particles,
                grid_resolution=self.grid_resolution,
                grid_width=self.grid_width,
                grid_height=self.grid_height
            )
            
            # Copy path
            new_particle.path = self.particles[i].path.copy()
            
            # Copy map (this is expensive but necessary)
            new_particle.grid = deepcopy(self.particles[i].grid)
            
            new_particles.append(new_particle)
        
        self.particles = new_particles
        
        # Update best particle (it should be the first one from elite selection)
        self.best_particle = self.particles[0]
    
    def get_best_pose(self):
        """
        Get the pose of the best particle
        
        Returns:
            dict: Pose dictionary with 'x', 'y', 'theta' keys
        """
        if self.best_particle:
            return self.best_particle.pose
        return None
    
    def get_best_map(self):
        """
        Get the map of the best particle
        
        Returns:
            OccupancyGrid: The occupancy grid of the best particle
        """
        if self.best_particle:
            return self.best_particle.grid
        return None
    
    def get_pose_estimate(self):
        """
        Get weighted average pose from all particles
        
        Returns:
            dict: Pose dictionary with 'x', 'y', 'theta' keys
        """
        if not self.particles:
            return None
        
        # Compute weighted average for x and y
        x_avg = sum(p.pose['x'] * p.weight for p in self.particles)
        y_avg = sum(p.pose['y'] * p.weight for p in self.particles)
        
        # For theta, we need to handle the circular mean properly
        sin_avg = sum(math.sin(p.pose['theta']) * p.weight for p in self.particles)
        cos_avg = sum(math.cos(p.pose['theta']) * p.weight for p in self.particles)
        theta_avg = math.atan2(sin_avg, cos_avg)
        
        return {
            'x': x_avg,
            'y': y_avg,
            'theta': theta_avg
        }
    
    def process_data(self, data_list, verbose=True):
        """
        Process a list of scan data, performing FastSLAM on each
        
        Args:
            data_list (list): List of parsed LiDAR data
            verbose (bool): Whether to print progress information
            
        Returns:
            list: Corrected pose history
        """
        if not data_list:
            return []
        
        # Initialize with first scan
        self.initialize(data_list[0]['pose'])
        
        # Update map with first scan
        self._update_particle_with_scan(self.best_particle, data_list[0])
        
        # Process each subsequent scan
        pose_history = [self.get_pose_estimate()]
        
        start_time = time.time()
        for i in range(1, len(data_list)):
            if verbose and i % 10 == 0:
                progress = i / len(data_list) * 100
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (len(data_list) - i)
                print(f"Processing scan {i}/{len(data_list)} ({progress:.1f}%) - " +
                     f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
            
            # Update with current scan
            self.update(data_list[i], data_list[i-1])
            
            # Save estimated pose
            pose_history.append(self.get_pose_estimate())
        
        if verbose:
            print(f"FastSLAM processing complete in {time.time() - start_time:.1f} seconds")
        
        return pose_history
    
    def save_map(self, filename=None, format='png'):
        """
        Save the best particle's map
        
        Args:
            filename (str): Base filename without extension
            format (str): Format to save in ('png', 'npy', 'csv', or 'all')
            
        Returns:
            list: List of saved filenames
        """
        if self.best_particle is None:
            print("No best particle available to save map")
            return []
        
        if filename is None:
            # Create a default filename
            maps_dir = "maps"
            file_utils.ensure_directory_exists(maps_dir)
            timestamp = int(time.time())
            filename = os.path.join(maps_dir, f"fastslam_map_{timestamp}")
        
        # Get path from best particle
        path = self.best_particle.path
        start_pos = path[0] if path else None
        end_pos = path[-1] if path else None
        
        # Save as image if requested
        if format == 'png' or format == 'all':
            file_utils.save_grid_as_image(
                self.best_particle.grid.get_grid(),
                filename,
                resolution=self.best_particle.grid.resolution,
                width=self.best_particle.grid.width,
                height=self.best_particle.grid.height,
                robot_path=path,
                start_position=start_pos,
                current_position=end_pos
            )
        
        # Save particle paths to visualize the filter
        if format == 'all':
            # Save all particle paths to a CSV file
            paths_file = f"{filename}_particle_paths.csv"
            with open(paths_file, 'w') as f:
                f.write("particle_id,weight,x,y\n")
                for i, particle in enumerate(self.particles):
                    for x, y in particle.path:
                        f.write(f"{i},{particle.weight},{x},{y}\n")
            print(f"Saved particle paths to {paths_file}")
        
        # Always save the raw grid data
        return self.best_particle.grid.save_to_file(filename, format=format, include_metadata=True)


def main():
    """Main function to demonstrate FastSLAM with Scan Matching"""
    import argparse
    import matplotlib.pyplot as plt
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='FastSLAM with Scan Matching Example')
    parser.add_argument('--file', '-f', type=str, 
                       default="../dataset/raw_data/raw_data_zjnu20_21_3F_short.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max-entries', '-m', type=int, default=200,
                       help='Maximum number of entries to read from the file')
    parser.add_argument('--particles', '-p', type=int, default=30,
                       help='Number of particles for FastSLAM')
    parser.add_argument('--resolution', '-r', type=float, default=0.05,
                       help='Grid resolution in meters')
    parser.add_argument('--visualize', '-v', action='store_true', default=True,
                       help='Visualize the final map')
    args = parser.parse_args()
    
    # Read data from file
    print(f"Reading LiDAR data from: {args.file}")
    sys.path.append('lidar_slam')
    data_list = data_parser.read_lidar_data_from_file(args.file, args.max_entries)
    
    if not data_list:
        print("No data was read from the file.")
        return
    
    # Create FastSLAM instance
    print(f"Initializing FastSLAM with {args.particles} particles...")
    slam = FastSLAM(
        num_particles=args.particles,
        grid_resolution=args.resolution,
        grid_width=30,  # Slightly larger grid for exploration
        grid_height=30
    )
    
    # Process data
    print("Processing data with FastSLAM (with scan matching)...")
    pose_history = slam.process_data(data_list)
    
    # Print stats
    print(f"Processed {len(data_list)} scans with {args.particles} particles")
    
    # Save the map
    print("Saving map...")
    saved_files = slam.save_map(format='png')
    print(f"Saved map to: {', '.join(saved_files)}")
    
    # Visualize final map if requested
    if args.visualize:
        best_map = slam.get_best_map()
        
        if best_map:
            # Extract path coordinates
            path_x = [pos[0] for pos in slam.best_particle.path]
            path_y = [pos[1] for pos in slam.best_particle.path]
            
            # Create figure for the map
            plt.figure(figsize=(12, 10))
            
            # Custom colormap: white (unknown), black (occupied), light gray (free)
            cmap = plt.cm.colors.ListedColormap(['white', 'lightgray', 'black'])
            bounds = [0, 0.4, 0.6, 1]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            # Plot the grid
            width = best_map.width
            height = best_map.height
            plt.imshow(best_map.get_grid(), cmap=cmap, norm=norm, origin='lower',
                      extent=[-width/2, width/2, -height/2, height/2])
            
            # Plot the path
            plt.plot(path_x, path_y, 'r-', linewidth=2, label='Robot Path')
            
            # Plot start and end positions
            if path_x:
                plt.scatter(path_x[0], path_y[0], c='green', s=100, marker='*', label='Start')
                plt.scatter(path_x[-1], path_y[-1], c='blue', s=100, marker='*', label='End')
            
            # Set up the plot
            plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            plt.title(f'FastSLAM Map with {args.particles} Particles')
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.legend(loc='upper right')
            
            # Add metadata as text
            metadata_text = (
                f"Resolution: {best_map.resolution:.3f}m/cell\n"
                f"Dimensions: {width:.1f}m × {height:.1f}m\n"
                f"Grid Size: {best_map.grid_width}×{best_map.grid_height} cells\n"
                f"Particles: {args.particles}"
            )
            plt.figtext(0.02, 0.02, metadata_text, wrap=True, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main()