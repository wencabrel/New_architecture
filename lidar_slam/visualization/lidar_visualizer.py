"""
LiDAR Visualizer Module

This module provides visualization tools for LiDAR scan data, including
real-time animation of scans and robot movement.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
from utils import file_utils

class LiDARVisualizer:
    """Class for visualizing LiDAR scan data with animation and occupancy grid"""
    
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
        
        # For storing displayed path that grows incrementally
        self.displayed_path_x = []
        self.displayed_path_y = []
        
        # Calculate time differences between frames
        if self.parsed_data_list:
            timestamps = [data['timestamp'] for data in self.parsed_data_list]
            self.start_time = timestamps[0]
            self.time_diffs = [t - self.start_time for t in timestamps]
        
        # IMPORTANT: Force Matplotlib to use a single figure
        plt.close('all')  # Close any existing figures to avoid confusion
        
        # Create SINGLE figure for both visualizations
        self.fig = plt.figure(figsize=self.figure_size)
        self.fig.suptitle('LiDAR SLAM Visualization', fontsize=16)
        
        # Create subplots using gridspec for better control
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        
        # LiDAR scan subplot
        self.ax1 = plt.subplot(gs[0])
        self.ax1.set_title('2D LiDAR Scan Visualization')
        self.ax1.set_xlabel('X (meters)')
        self.ax1.set_ylabel('Y (meters)')
        self.ax1.grid(True)
        
        # Occupancy grid subplot
        if occupancy_grid:
            self.ax2 = plt.subplot(gs[1])
            self.ax2.set_title('Occupancy Grid Map')
            self.ax2.set_xlabel('X (meters)')
            self.ax2.set_ylabel('Y (meters)')
            self.ax2.grid(True)
        else:
            self.ax2 = None
        
        # Initialize plot elements for LiDAR scan
        self.scatter = None
        self.robot_pos = None
        self.path_line = None
        self.arrow = None
        self.timestamp_text = None
        self.robot_id_text = None
        self.pose_text = None
        self.settings_text = None
        
        # Initialize plot elements for occupancy grid
        if occupancy_grid:
            # Custom colormap for occupancy grid
            self.grid_cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
            self.grid_bounds = [0, 0.4, 0.6, 1]
            self.grid_norm = colors.BoundaryNorm(self.grid_bounds, self.grid_cmap.N)
            
            # Grid image
            self.grid_img = None
            
            # Grid path elements
            self.grid_path_line = None
            self.grid_start_point = None
            self.grid_current_pos = None
            
            # Follow robot flag
            self.follow_robot = True
            
            # Original grid limits for reset
            self.grid_original_xlim = None
            self.grid_original_ylim = None
        
        # Buttons
        self.pause_button = None
        self.follow_button = None
        self.save_button = None
    
    def _initialize_plot(self):
        """Initialize all plot elements for both LiDAR scan and occupancy grid"""
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
        
        # Initialize LiDAR scan visualization elements
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
        
        # Set limits for LiDAR scan plot
        self.ax1.set_xlim(self.x_min, self.x_max)
        self.ax1.set_ylim(self.y_min, self.y_max)
        self.ax1.set_aspect('equal')
        self.ax1.legend(loc='upper right')
        
        # Initialize occupancy grid visualization if available
        if self.occupancy_grid and self.ax2:
            width = self.occupancy_grid.width
            height = self.occupancy_grid.height
            
            # Initialize grid image
            self.grid_img = self.ax2.imshow(
                self.occupancy_grid.get_grid(),
                cmap=self.grid_cmap,
                norm=self.grid_norm,
                origin='lower',
                extent=[-width/2, width/2, -height/2, height/2]
            )
            
            # Initialize grid path elements
            self.grid_path_line, = self.ax2.plot(
                [], [], 'r-', linewidth=2, label='Robot Path'
            )
            self.grid_start_point = self.ax2.scatter(
                [], [], c='green', s=100, marker='*', label='Start Position'
            )
            self.grid_current_pos = self.ax2.scatter(
                [], [], c='blue', s=100, marker='*', label='Current Position'
            )
            
            # Add metadata text
            self.ax2.text(
                0.02, 0.02,
                f"Resolution: {self.occupancy_grid.resolution:.3f}m/cell\n"
                f"Dimensions: {width:.1f}m × {height:.1f}m\n"
                f"Grid Size: {self.occupancy_grid.grid_width}×{self.occupancy_grid.grid_height} cells",
                transform=self.ax2.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                va='bottom'
            )
            
            # Add zoom information
            self.ax2.text(
                0.5, 0.01,
                "Left-click: Zoom in | Right-click: Zoom out | Middle-click: Reset zoom",
                transform=self.ax2.transAxes,
                va='bottom', ha='center',
                fontsize=8, color='blue',
                bbox=dict(facecolor='white', alpha=0.7)
            )
            
            # Set limits and legend
            self.ax2.set_aspect('equal')
            self.ax2.legend(loc='upper right')
            
            # Store original limits for reset
            self.grid_original_xlim = self.ax2.get_xlim()
            self.grid_original_ylim = self.ax2.get_ylim()
            
            # Add click handler for grid subplot
            self.fig.canvas.mpl_connect('button_press_event', self._on_grid_click)
        
        # Set up control buttons
        self._setup_buttons()
    
    def _setup_buttons(self):
        """Set up control buttons for the visualization"""
        # Make room for buttons
        plt.subplots_adjust(bottom=0.15)
        
        # Pause/Play button
        pause_button_ax = plt.axes([0.4, 0.05, 0.1, 0.04])
        self.pause_button = Button(pause_button_ax, 'Pause', color='lightcoral', hovercolor='0.9')
        self.pause_button.on_clicked(self.toggle_pause)
        
        # Only add follow and save buttons if we have an occupancy grid
        if self.occupancy_grid and self.ax2:
            # Follow Robot button
            follow_button_ax = plt.axes([0.55, 0.05, 0.1, 0.04])
            self.follow_button = Button(follow_button_ax, 'Following', 
                                      color='lightgoldenrodyellow', hovercolor='0.975')
            self.follow_button.on_clicked(self.toggle_follow)
            
            # Save Map button
            save_button_ax = plt.axes([0.7, 0.05, 0.1, 0.04])
            self.save_button = Button(save_button_ax, 'Save Map', 
                                     color='lightblue', hovercolor='0.8')
            self.save_button.on_clicked(self.save_current_map)
    
    def toggle_pause(self, event):
        """Toggle animation pause/play"""
        if self.animation and self.animation.event_source:
            if self.is_running:
                self.animation.event_source.stop()
                self.pause_button.label.set_text('Play')
            else:
                self.animation.event_source.start()
                self.pause_button.label.set_text('Pause')
            self.is_running = not self.is_running
    
    def toggle_follow(self, event):
        """Toggle robot following mode for occupancy grid"""
        if self.follow_button:
            self.follow_robot = not self.follow_robot
            self.follow_button.label.set_text('Following' if self.follow_robot else 'Not Following')
            
            # Center immediately if turning on following
            if self.follow_robot and self.displayed_path_x:
                self._center_on_robot()
                self.fig.canvas.draw_idle()
    
    def _center_on_robot(self):
        """Center the occupancy grid view on the current robot position"""
        if not self.displayed_path_x or not self.ax2:
            return
            
        # Get current position
        x = self.displayed_path_x[-1]
        y = self.displayed_path_y[-1]
        
        # Current axis limits
        xmin, xmax = self.ax2.get_xlim()
        ymin, ymax = self.ax2.get_ylim()
        width = xmax - xmin
        height = ymax - ymin
        
        # Center on robot position
        self.ax2.set_xlim(x - width/2, x + width/2)
        self.ax2.set_ylim(y - height/2, y + height/2)
    
    def _on_grid_click(self, event):
        """Handle mouse click events for zooming on occupancy grid"""
        # Only process clicks in the grid axis
        if not self.ax2 or event.inaxes != self.ax2:
            return
        
        # Get click coordinates
        x, y = event.xdata, event.ydata
        
        # Current axis limits
        xmin, xmax = self.ax2.get_xlim()
        ymin, ymax = self.ax2.get_ylim()
        width = xmax - xmin
        height = ymax - ymin
        
        # Left-click: Zoom in
        if event.button == 1:  # Left click
            # Zoom in by 50% around the clicked point
            new_width = width * 0.5
            new_height = height * 0.5
            self.ax2.set_xlim(x - new_width/2, x + new_width/2)
            self.ax2.set_ylim(y - new_height/2, y + new_height/2)
            
            # Turn off follow mode when manually zooming
            self.follow_robot = False
            if self.follow_button:
                self.follow_button.label.set_text('Not Following')
            
        # Right-click: Zoom out
        elif event.button == 3:  # Right click
            # Zoom out by 200%
            new_width = width * 2.0
            new_height = height * 2.0
            self.ax2.set_xlim(x - new_width/2, x + new_width/2)
            self.ax2.set_ylim(y - new_height/2, y + new_height/2)
            
        # Middle-click: Reset zoom
        elif event.button == 2:  # Middle click
            if self.grid_original_xlim and self.grid_original_ylim:
                self.ax2.set_xlim(self.grid_original_xlim)
                self.ax2.set_ylim(self.grid_original_ylim)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
    
    def save_current_map(self, event):
        """Save the current occupancy grid map to a file"""
        if not self.occupancy_grid:
            return
            
        # Generate a timestamp-based filename
        save_path = "maps"
        file_utils.ensure_directory_exists(save_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.join(save_path, f"occupancy_grid_{timestamp}")
        
        # Get path coordinates for saving
        path_coords = list(zip(self.displayed_path_x, self.displayed_path_y))
        start_pos = path_coords[0] if path_coords else None
        current_pos = path_coords[-1] if path_coords else None
        
        # Save as image
        grid = self.occupancy_grid.get_grid()
        file_utils.save_grid_as_image(
            grid, 
            base_filename, 
            resolution=self.occupancy_grid.resolution,
            width=self.occupancy_grid.width,
            height=self.occupancy_grid.height,
            robot_path=path_coords,
            start_position=start_pos,
            current_position=current_pos
        )
        
        # Also save raw grid data and metadata
        self.occupancy_grid.save_to_file(base_filename, format='all', include_metadata=True)
        
        print(f"\nOccupancy grid map saved to {base_filename} with current robot path and positions")
    
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
            # Reset paths for the first frame
            self.displayed_path_x = [robot_x]
            self.displayed_path_y = [robot_y]
        else:
            # Append new position
            self.displayed_path_x.append(robot_x)
            self.displayed_path_y.append(robot_y)
        
        # Update path lines
        self.path_line.set_data(self.displayed_path_x, self.displayed_path_y)
        
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
        if hasattr(self, 'arrow') and self.arrow:
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
        if self.occupancy_grid and self.ax2:
            # Update the grid with current scan
            self.occupancy_grid.update_grid(robot_x, robot_y, x_points, y_points)
            
            # Update the grid image
            self.grid_img.set_data(self.occupancy_grid.get_grid())
            
            # Update the grid path
            self.grid_path_line.set_data(self.displayed_path_x, self.displayed_path_y)
            
            # Set start position on first frame
            if frame == 0:
                self.grid_start_point.set_offsets([[robot_x, robot_y]])
            
            # Update current position
            self.grid_current_pos.set_offsets([[robot_x, robot_y]])
            
            # Follow robot if enabled
            if self.follow_robot:
                self._center_on_robot()
        
        # Return updated elements
        return (self.scatter, self.robot_pos, self.path_line, self.timestamp_text, 
               self.robot_id_text, self.pose_text, self.settings_text, self.arrow)
    
    def animate(self, interval=50):
        """
        Start the animation
        
        Args:
            interval (int): Interval between frames in milliseconds
            
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
        self.animate(interval)
        
        # Adjust layout and display the single window
        plt.tight_layout()
        plt.show()
        
        # Save the final grid if requested
        if save_path and self.occupancy_grid:
            # Get final robot path
            path_coords = list(zip(self.displayed_path_x, self.displayed_path_y))
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