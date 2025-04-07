"""
LiDAR Visualizer Module

This module provides visualization tools for LiDAR scan data, including
real-time animation of scans and robot movement.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import math
import os
import sys
import time
from matplotlib.widgets import Button

# Add the parent directory to the path to import other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lidar_processing import scan_converter
from visualization import grid_visualizer
from utils import file_utils

class LiDARVisualizer:
    """Class for visualizing LiDAR scan data with animation"""
    
    def __init__(self, parsed_data_list, occupancy_grid=None, figure_size=(18, 9)):
        """
        Initialize the LiDAR scan visualizer
        
        Args:
            parsed_data_list (list): List of parsed LiDAR data dictionaries
            occupancy_grid: Optional OccupancyGrid object for grid mapping
            figure_size (tuple): Size of the figure in inches
        """
        self.parsed_data_list = parsed_data_list
        self.occupancy_grid = occupancy_grid
        self.figure_size = figure_size
        
        # Configuration for scan conversion
        self.config = {
            'flip_x': False,
            'flip_y': False,
            'reverse_scan': True,
            'flip_theta': False
        }
        
        # Assuming the LiDAR scan covers 180 degrees (π radians)
        self.angle_min = -math.pi/2
        self.angle_max = math.pi/2
        
        # Initialize tracking variables
        self.current_frame_index = 0
        self.animation = None
        self.is_running = True
        
        # For storing robot path
        self.robot_path_x = []
        self.robot_path_y = []
        
        # Calculate time differences between frames
        if self.parsed_data_list:
            timestamps = [data['timestamp'] for data in self.parsed_data_list]
            self.start_time = timestamps[0]
            self.time_diffs = [t - self.start_time for t in timestamps]
        
        # Initialize figure and axes
        if occupancy_grid:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=self.figure_size)
            self.grid_visualizer = grid_visualizer.OccupancyGridVisualizer(
                occupancy_grid, 
                robot_path=[self.robot_path_x, self.robot_path_y],
                figure_size=None  # Don't create a new figure
            )
            self.grid_visualizer.fig = self.fig
            self.grid_visualizer.ax = self.ax2
        else:
            self.fig, self.ax1 = plt.subplots(figsize=self.figure_size)
            self.ax2 = None
            self.grid_visualizer = None
        
        # Initialize plot elements
        self.scatter = None
        self.robot_pos = None
        self.path_line = None
        self.arrow = None
        self.timestamp_text = None
        self.robot_id_text = None
        self.pose_text = None
        self.settings_text = None
    
    def _initialize_plot(self):
        """Initialize the plot elements"""
        # Find boundaries for all scans
        all_x_points = []
        all_y_points = []
        
        for parsed_data in self.parsed_data_list:
            x_points, y_points = scan_converter.convert_scans_to_cartesian(
                parsed_data['scan_ranges'], 
                self.angle_min, 
                self.angle_max, 
                parsed_data['pose'],
                **self.config
            )
            all_x_points.extend(x_points)
            all_y_points.extend(y_points)
        
        # Calculate scan boundaries
        boundaries = scan_converter.find_scan_boundaries(all_x_points, all_y_points)
        self.x_min, self.x_max, self.y_min, self.y_max = boundaries
        
        # Extract robot path
        self.robot_path_x, self.robot_path_y = scan_converter.extract_robot_path(
            self.parsed_data_list,
            flip_x=self.config['flip_x'],
            flip_y=self.config['flip_y']
        )
        
        # Initialize plot elements on the LiDAR scan axis
        self.scatter = self.ax1.scatter([], [], c='blue', s=3, label='LiDAR Points')
        self.robot_pos = self.ax1.scatter([], [], c='red', s=100, marker='*', label='Robot Position')
        self.path_line, = self.ax1.plot([], [], 'g-', linewidth=2, label='Robot Path')
        
        # Initialize text objects for information display
        self.timestamp_text = self.ax1.text(0.02, 0.98, "", transform=self.ax1.transAxes, va='top', ha='left')
        self.robot_id_text = self.ax1.text(0.02, 0.94, "", transform=self.ax1.transAxes, va='top', ha='left')
        self.pose_text = self.ax1.text(0.02, 0.90, "", transform=self.ax1.transAxes, va='top', ha='left')
        
        # Show orientation settings
        settings_str = (f"Settings: flip_x={self.config['flip_x']}, flip_y={self.config['flip_y']}, "
                        f"reverse_scan={self.config['reverse_scan']}, flip_theta={self.config['flip_theta']}")
        self.settings_text = self.ax1.text(0.02, 0.86, settings_str, transform=self.ax1.transAxes, va='top', ha='left')
        
        # Set limits, grid, and labels for the scan axis
        self.ax1.set_xlim(self.x_min, self.x_max)
        self.ax1.set_ylim(self.y_min, self.y_max)
        self.ax1.grid(True)
        self.ax1.set_aspect('equal')
        self.ax1.set_xlabel('X (meters)')
        self.ax1.set_ylabel('Y (meters)')
        self.ax1.set_title('2D LiDAR Scan Visualization')
        self.ax1.legend(loc='upper right')
        
        # Initialize the grid visualizer if we have an occupancy grid
        if self.grid_visualizer:
            self.grid_visualizer.robot_path = [self.robot_path_x, self.robot_path_y]
            self.grid_visualizer.initialize_display()
    
    def _update_frame(self, frame):
        """
        Update function for animation
        
        Args:
            frame (int): Current frame index
            
        Returns:
            tuple: Updated plot elements
        """
        # Update the current frame index
        self.current_frame_index = frame
        
        parsed_data = self.parsed_data_list[frame]
        
        # Convert current scan to Cartesian coordinates with configured orientation
        x_points, y_points = scan_converter.convert_scans_to_cartesian(
            parsed_data['scan_ranges'], 
            self.angle_min, 
            self.angle_max, 
            parsed_data['pose'],
            **self.config
        )
        
        # Update LiDAR points
        self.scatter.set_offsets(np.column_stack((x_points, y_points)))
        
        # Get transformed robot pose for visualization
        robot_x, robot_y = parsed_data['pose']['x'], parsed_data['pose']['y']
        if self.config['flip_x']:
            robot_x = -robot_x
        if self.config['flip_y']:
            robot_y = -robot_y
        
        # Update robot position
        self.robot_pos.set_offsets([[robot_x, robot_y]])
        
        # Update robot path - build incrementally for animation
        if frame == 0:
            # Reset path for the first frame
            self.path_line.set_data([robot_x], [robot_y])
        else:
            # Get current path data
            path_x, path_y = self.path_line.get_data()
            
            # Append new position
            new_x = list(path_x) + [robot_x]
            new_y = list(path_y) + [robot_y]
            
            # Update path
            self.path_line.set_data(new_x, new_y)
        
        # Update text information
        elapsed_time = self.time_diffs[frame]
        timestamp_str = f"Time: {elapsed_time:.3f}s"
        self.timestamp_text.set_text(timestamp_str)
        
        # Show robot ID
        robot_id_str = f"Robot ID: {parsed_data['robot_id']}"
        self.robot_id_text.set_text(robot_id_str)
        
        # Show original pose values
        pose_str = f"Pose: x={parsed_data['pose']['x']:.3f}, y={parsed_data['pose']['y']:.3f}, θ={parsed_data['pose']['theta']:.3f}"
        self.pose_text.set_text(pose_str)
        
        # Update robot orientation arrow
        if self.arrow:
            self.arrow.remove()
        
        # Apply orientation transformation
        theta = parsed_data['pose']['theta']
        if self.config['flip_theta']:
            theta = -theta
        
        arrow_length = 0.5
        dx = arrow_length * math.cos(theta)
        dy = arrow_length * math.sin(theta)
        
        if self.config['flip_x']:
            dx = -dx
        if self.config['flip_y']:
            dy = -dy
            
        self.arrow = self.ax1.arrow(robot_x, robot_y, dx, dy, 
                                    head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        # Update occupancy grid if enabled
        if self.occupancy_grid and self.grid_visualizer:
            # Update the grid with current scan
            self.occupancy_grid.update_grid(robot_x, robot_y, x_points, y_points)
            
            # Update the grid visualizer
            self.grid_visualizer.update_grid((robot_x, robot_y))
        
        # Return updated elements
        if self.grid_visualizer:
            # Complex return including grid elements
            return (self.scatter, self.robot_pos, self.path_line, self.timestamp_text, 
                   self.robot_id_text, self.pose_text, self.settings_text, self.arrow)
        else:
            return (self.scatter, self.robot_pos, self.path_line, self.timestamp_text, 
                   self.robot_id_text, self.pose_text, self.settings_text, self.arrow)
    
    def animate(self, interval=50, save_path=None):
        """
        Start the animation
        
        Args:
            interval (int): Interval between frames in milliseconds
            save_path (str): Path to save the final grid if enabled
            
        Returns:
            matplotlib.animation.FuncAnimation: Animation object
        """
        if not self.parsed_data_list:
            print("No data to animate.")
            return None
        
        # Initialize the plot
        self._initialize_plot()
        
        # Create the animation
        self.animation = FuncAnimation(
            self.fig, 
            self._update_frame, 
            frames=len(self.parsed_data_list),
            interval=interval, 
            blit=False
        )
        
        return self.animation
    
    def show(self, interval=50, save_path=None):
        """
        Show the animation and start the interactive loop
        
        Args:
            interval (int): Interval between frames in milliseconds
            save_path (str): Path to save the final grid if enabled
        """
        self.animate(interval, save_path)
        
        # Adjust layout and display the window
        plt.tight_layout()
        plt.show()
        
        # Save the final grid if requested
        if save_path and self.occupancy_grid:
            # Get final robot path
            path_coords = list(zip(self.robot_path_x, self.robot_path_y))
            start_pos = path_coords[0] if path_coords else None
            final_pos = path_coords[-1] if path_coords else None
            
            # Generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_path, f"occupancy_grid_{timestamp}")
            
            # Save grid with metadata
            self.occupancy_grid.save_to_file(filename, format='all', include_metadata=True)
            
            # Also save as image
            file_utils.save_grid_as_image(
                self.occupancy_grid.get_grid(),
                filename,
                resolution=self.occupancy_grid.resolution,
                width=self.occupancy_grid.width,
                height=self.occupancy_grid.height,
                robot_path=path_coords,
                start_position=start_pos,
                current_position=final_pos
            )
            
            print(f"\nFinal occupancy grid map saved to {filename}")