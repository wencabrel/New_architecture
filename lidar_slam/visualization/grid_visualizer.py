"""
Grid Visualizer Module

This module provides visualization tools for occupancy grid maps, including
interactive display with zooming and saving capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import time
import os
import sys

# Add the parent directory to the path to import other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import file_utils

class OccupancyGridVisualizer:
    """Class for visualizing occupancy grid maps with interactive features"""
    
    def __init__(self, occupancy_grid, robot_path=None, figure_size=(12, 10)):
        """
        Initialize the visualizer with an occupancy grid
        
        Args:
            occupancy_grid: OccupancyGrid object from mapping.occupancy_grid
            robot_path (list): Optional list of (x, y) coordinates of robot path
            figure_size (tuple): Size of the figure in inches
        """
        self.occupancy_grid = occupancy_grid
        self.robot_path = robot_path or []
        self.figure_size = figure_size
        
        # Create figure and axis for standalone visualization
        # When used within LiDARVisualizer, these will be overridden
        self.standalone_mode = True  # Will be set to False by LiDARVisualizer
        self.fig = None
        self.ax = None
        
        # Custom colormap for occupancy grid
        self.cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
        self.bounds = [0, 0.4, 0.6, 1]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        
        # Initialize flags and variables
        self.zoom_factor = 0.5  # How much to zoom in/out (0.5 = 50% zoom)
        self.follow_robot = True  # Default to following robot
        self.current_robot_pos = (0, 0)
        self.is_initialized = False
        
        # Path elements
        self.path_line = None
        self.start_point = None
        self.current_pos = None
        
        # Initialize with empty path that will be built gradually
        self.displayed_path_x = []
        self.displayed_path_y = []
        
        # Text elements
        self.info_text = None
        self.metadata_text = None
        self.zoom_info_text = None
        
        # Grid image
        self.grid_img = None
        
        # Store original axis limits for reset
        self.original_xlim = None
        self.original_ylim = None
        
        # Buttons
        self.follow_button = None
        self.save_button = None
    
    def _setup_event_handlers(self):
        """Set up mouse event handlers for interactive features"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add buttons for controls - these need to be within the same figure
        plt.subplots_adjust(bottom=0.15)  # Make room for buttons
        
        # Follow Robot button
        self.follow_button_ax = plt.axes([0.85, 0.05, 0.1, 0.04])
        self.follow_button = Button(self.follow_button_ax, 'Following', 
                                   color='lightgoldenrodyellow', hovercolor='0.975')
        self.follow_button.on_clicked(self.toggle_follow)
        
        # Save Map button
        self.save_button_ax = plt.axes([0.70, 0.05, 0.1, 0.04])
        self.save_button = Button(self.save_button_ax, 'Save Map', 
                                 color='lightblue', hovercolor='0.8')
        self.save_button.on_clicked(self.save_current_map)
    
    def initialize_display(self):
        """Initialize the grid display and all plot elements"""
        if self.is_initialized:
            return
            
        # Create figure and axes if in standalone mode
        if self.standalone_mode:
            self.fig, self.ax = plt.subplots(figsize=self.figure_size)
        
        # Ensure we have a valid figure and axis
        if self.fig is None or self.ax is None:
            raise ValueError("Figure and axis must be set before initializing display")
            
        # Get grid dimensions
        width = self.occupancy_grid.width
        height = self.occupancy_grid.height
        
        # Plot the grid
        self.grid_img = self.ax.imshow(
            self.occupancy_grid.get_grid(), 
            cmap=self.cmap, 
            norm=self.norm, 
            origin='lower',
            extent=[-width/2, width/2, -height/2, height/2]
        )
        
        # Add grid lines
        self.ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Initialize path elements - start with empty path
        self.displayed_path_x = []
        self.displayed_path_y = []
        
        # Initialize with empty path that will be built over time
        self.path_line, = self.ax.plot(
            self.displayed_path_x, 
            self.displayed_path_y, 
            'r-', linewidth=2, label='Robot Path'
        )
        
        # Start point will be set when we get the first position
        self.start_point = self.ax.scatter(
            [], [], c='green', s=100, marker='*', label='Start Position'
        )
        
        # Current position will be updated during visualization
        self.current_pos = self.ax.scatter(
            [], [], c='blue', s=100, marker='*', label='Current Position'
        )
        
        # Add text elements
        metadata = self.occupancy_grid.get_metadata()
        self.metadata_text = self.ax.text(
            0.02, 0.02, 
            f"Resolution: {self.occupancy_grid.resolution:.3f}m/cell\n"
            f"Dimensions: {width:.1f}m × {height:.1f}m\n"
            f"Grid Size: {self.occupancy_grid.grid_width}×{self.occupancy_grid.grid_height} cells",
            transform=self.ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            va='bottom'
        )
        
        self.zoom_info_text = self.ax.text(
            0.5, 0.01, 
            "Left-click: Zoom in | Right-click: Zoom out | Middle-click: Reset zoom",
            transform=self.ax.transAxes,
            va='bottom', ha='center',
            fontsize=10, color='blue',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Set titles and labels
        self.ax.set_title('Occupancy Grid Map')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_aspect('equal')
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        # Store original axis limits for reset
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        
        # Setup event handlers and buttons if in standalone mode
        if self.standalone_mode:
            self._setup_event_handlers()
        
        # Mark as initialized
        self.is_initialized = True
    
    def update_grid(self, new_robot_pos=None):
        """
        Update the grid display
        
        Args:
            new_robot_pos (tuple): New robot position (x, y)
        """
        if not self.is_initialized:
            self.initialize_display()
        
        # Update grid image with latest data
        self.grid_img.set_data(self.occupancy_grid.get_grid())
        
        # Update robot position and path if provided
        if new_robot_pos is not None:
            self.current_robot_pos = new_robot_pos
            
            # Add position to displayed path
            self.displayed_path_x.append(new_robot_pos[0])
            self.displayed_path_y.append(new_robot_pos[1])
            
            # Update path line with current displayed path
            self.path_line.set_data(self.displayed_path_x, self.displayed_path_y)
            
            # Set start point on first update
            if len(self.displayed_path_x) == 1:
                self.start_point.set_offsets([new_robot_pos])
            
            # Update current position marker
            self.current_pos.set_offsets([new_robot_pos])
            
            # Follow robot if enabled
            if self.follow_robot:
                self._center_on_robot()
        
        # Redraw the plot
        self.fig.canvas.draw_idle()
    
    def _center_on_robot(self):
        """Center the view on the current robot position while maintaining zoom level"""
        if not self.current_robot_pos:
            return
            
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        width = xmax - xmin
        height = ymax - ymin
        
        x, y = self.current_robot_pos
        self.ax.set_xlim(x - width/2, x + width/2)
        self.ax.set_ylim(y - height/2, y + height/2)
    
    def on_click(self, event):
        """
        Handle mouse click events for zooming
        
        Args:
            event: MouseEvent object
        """
        # Only process clicks in the map axis
        if event.inaxes != self.ax:
            return
            
        # Get click coordinates
        x, y = event.xdata, event.ydata
        
        # Current axis limits
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        width = xmax - xmin
        height = ymax - ymin
        
        # Left-click: Zoom in
        if event.button == 1:  # Left click
            # Zoom in by zoom_factor around the clicked point
            new_width = width * self.zoom_factor
            new_height = height * self.zoom_factor
            self.ax.set_xlim(x - new_width/2, x + new_width/2)
            self.ax.set_ylim(y - new_height/2, y + new_height/2)
            self.follow_robot = False  # Turn off follow mode when manually zooming
            if self.follow_button:
                self.follow_button.label.set_text('Not Following')
            
        # Right-click: Zoom out
        elif event.button == 3:  # Right click
            # Zoom out by inverse of zoom_factor
            new_width = width / self.zoom_factor
            new_height = height / self.zoom_factor
            # Center on the clicked point
            self.ax.set_xlim(x - new_width/2, x + new_width/2)
            self.ax.set_ylim(y - new_height/2, y + new_height/2)
            
        # Middle-click: Reset zoom
        elif event.button == 2:  # Middle click
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
    
    def toggle_follow(self, event):
        """Toggle robot following mode"""
        self.follow_robot = not self.follow_robot
        self.follow_button.label.set_text('Following' if self.follow_robot else 'Not Following')
        
        if self.follow_robot and self.current_robot_pos:
            self._center_on_robot()
            self.fig.canvas.draw_idle()
    
    def save_current_map(self, event):
        """Save the current map to a file"""
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
    
    def show(self):
        """Display the visualization and start the interactive loop"""
        if not self.is_initialized:
            self.initialize_display()
        
        if self.standalone_mode:
            plt.tight_layout()
            plt.show()