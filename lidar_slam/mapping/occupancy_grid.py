"""
Enhanced Occupancy Grid Module

This module provides an enhanced occupancy grid implementation with improved
probabilistic updates and uncertainty handling for LiDAR SLAM.
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_agg import FigureCanvasAgg
import math
import pickle

class OccupancyGrid:
    """Class for maintaining an enhanced probabilistic occupancy grid map"""
    
    def __init__(self, resolution=0.05, width=20.0, height=20.0):
        """
        Initialize an occupancy grid with given parameters
        
        Args:
            resolution (float): Grid resolution in meters per cell
            width (float): Width of the grid in meters
            height (float): Height of the grid in meters
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        
        # Calculate grid dimensions
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Ensure grid dimensions are odd for easy centering
        if self.grid_width % 2 == 0:
            self.grid_width += 1
        if self.grid_height % 2 == 0:
            self.grid_height += 1
        
        # Update actual width and height
        self.width = self.grid_width * resolution
        self.height = self.grid_height * resolution
        
        # Create the grid with initial probabilities
        # 0.5 represents unknown (log-odds = 0)
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5
        
        # For log-odds representation
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))
        
        # Sensor model parameters
        self.p_hit = 0.7      # Probability of hit if occupied
        self.p_miss = 0.3     # Probability of miss if occupied
        self.p_occ_prior = 0.5  # Prior probability of occupancy
        
        # Log-odds versions
        self.log_odds_hit = np.log(self.p_hit / (1 - self.p_hit))
        self.log_odds_miss = np.log(self.p_miss / (1 - self.p_miss))
        self.log_odds_prior = np.log(self.p_occ_prior / (1 - self.p_occ_prior))
        
        # Clamping thresholds for log-odds values
        self.log_odds_min = -10.0  # Minimum log-odds value
        self.log_odds_max = 10.0   # Maximum log-odds value
        
        # Metadata
        self.metadata = {
            'creation_time': time.time(),
            'resolution': resolution,
            'width': width,
            'height': height,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'updates': 0,
            'sensor_model': {
                'p_hit': self.p_hit,
                'p_miss': self.p_miss,
                'p_occ_prior': self.p_occ_prior
            }
        }
    
    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices
        
        Args:
            x (float): X coordinate in world frame (meters)
            y (float): Y coordinate in world frame (meters)
            
        Returns:
            tuple: (grid_x, grid_y) indices in the grid
        """
        grid_x = int((x + self.width / 2) / self.resolution)
        grid_y = int((y + self.height / 2) / self.resolution)
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices to world coordinates
        
        Args:
            grid_x (int): X index in the grid
            grid_y (int): Y index in the grid
            
        Returns:
            tuple: (x, y) coordinates in world frame (meters)
        """
        x = grid_x * self.resolution - self.width / 2
        y = grid_y * self.resolution - self.height / 2
        
        return x, y
    
    def is_inside_grid(self, grid_x, grid_y):
        """
        Check if the given grid indices are within the grid bounds
        
        Args:
            grid_x (int): X index in the grid
            grid_y (int): Y index in the grid
            
        Returns:
            bool: True if inside grid, False otherwise
        """
        return (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height)
    
    def get_grid(self):
        """
        Get the current probability grid
        
        Returns:
            np.ndarray: The occupancy grid with probabilities
        """
        return self.grid.copy()
    
    def get_binary_grid(self, threshold=0.5):
        """
        Get a binary version of the grid (occupied/free)
        
        Args:
            threshold (float): Probability threshold for considering a cell occupied
            
        Returns:
            np.ndarray: Binary grid (1 for occupied, 0 for free)
        """
        return (self.grid > threshold).astype(np.int8)
    
    def get_metadata(self):
        """
        Get grid metadata
        
        Returns:
            dict: Grid metadata
        """
        return self.metadata.copy()
    
    def update_grid(self, robot_x, robot_y, scan_x, scan_y, uncertainty=None):
        """
        Update the occupancy grid with a new scan
        
        Args:
            robot_x (float): Robot X position in world frame
            robot_y (float): Robot Y position in world frame
            scan_x (list): List of X coordinates of scan points in world frame
            scan_y (list): List of Y coordinates of scan points in world frame
            uncertainty (tuple): Optional (x_var, y_var, theta_var) pose uncertainty
        """
        if len(scan_x) != len(scan_y) or len(scan_x) == 0:
            return
        
        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        # Check if robot is inside grid
        if not self.is_inside_grid(robot_grid_x, robot_grid_y):
            print(f"Warning: Robot at ({robot_x:.2f}, {robot_y:.2f}) is outside grid boundaries.")
            return
        
        # Process scan rays using Bresenham's line algorithm
        for i in range(len(scan_x)):
            # Convert scan point to grid coordinates
            scan_grid_x, scan_grid_y = self.world_to_grid(scan_x[i], scan_y[i])
            
            # Update cells along the ray
            self._update_ray(robot_grid_x, robot_grid_y, scan_grid_x, scan_grid_y, uncertainty)
        
        # Increment update counter
        self.metadata['updates'] += 1
    
    def _update_ray(self, x0, y0, x1, y1, uncertainty=None):
        """
        Update cells along a ray using Bresenham's line algorithm
        
        Args:
            x0, y0 (int): Start point (robot) in grid coordinates
            x1, y1 (int): End point (scan) in grid coordinates
            uncertainty (tuple): Optional (x_var, y_var, theta_var) pose uncertainty
        """
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        # Variables to track the ray
        cells_visited = []
        hit_endpoint = False
        
        # Apply uncertainty adjustment if provided
        uncertainty_factor = 1.0
        if uncertainty is not None:
            x_var, y_var, theta_var = uncertainty
            # Calculate distance from robot to scan point
            distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2) * self.resolution
            
            # Higher uncertainty at longer distances
            angle_uncertainty = math.atan2(math.sqrt(x_var + y_var), distance)
            uncertainty_factor = max(0.1, 1.0 - min(0.9, angle_uncertainty + theta_var))
        
        while True:
            # Check if we're inside the grid
            if not self.is_inside_grid(x, y):
                break
            
            # Add current cell to visited cells
            cells_visited.append((x, y))
            
            # Check if we've reached the endpoint
            if x == x1 and y == y1:
                hit_endpoint = True
                break
            
            # Bresenham's line algorithm step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Now update the cells
        for i, (cell_x, cell_y) in enumerate(cells_visited):
            if self.is_inside_grid(cell_x, cell_y):
                if hit_endpoint and i == len(cells_visited) - 1:
                    # Last cell and we hit the endpoint - it's an obstacle
                    # Apply uncertainty-adjusted hit update
                    log_odds_update = self.log_odds_hit * uncertainty_factor
                    self.log_odds_grid[cell_y, cell_x] = min(
                        self.log_odds_max, 
                        self.log_odds_grid[cell_y, cell_x] + log_odds_update
                    )
                else:
                    # Cell along the ray - it's free space
                    # Apply uncertainty-adjusted miss update
                    log_odds_update = self.log_odds_miss * uncertainty_factor
                    self.log_odds_grid[cell_y, cell_x] = max(
                        self.log_odds_min, 
                        self.log_odds_grid[cell_y, cell_x] + log_odds_update
                    )
        
        # Convert log-odds back to probabilities
        self.grid = 1 - (1 / (1 + np.exp(self.log_odds_grid)))
    
    def update_grid_with_uncertainty(self, robot_x, robot_y, scan_x, scan_y, pose_covariance):
        """
        Update the occupancy grid with a new scan, considering pose uncertainty
        
        Args:
            robot_x (float): Robot X position in world frame
            robot_y (float): Robot Y position in world frame
            scan_x (list): List of X coordinates of scan points in world frame
            scan_y (list): List of Y coordinates of scan points in world frame
            pose_covariance (np.ndarray): 3x3 pose covariance matrix (x, y, theta)
        """
        if pose_covariance is None:
            # Fall back to standard update if no covariance provided
            self.update_grid(robot_x, robot_y, scan_x, scan_y)
            return
        
        # Extract variances from covariance matrix
        x_var = pose_covariance[0, 0]
        y_var = pose_covariance[1, 1]
        theta_var = pose_covariance[2, 2]
        
        # Update with uncertainty information
        self.update_grid(robot_x, robot_y, scan_x, scan_y, (x_var, y_var, theta_var))
    
    def mark_cell(self, x, y, occupied=True, log_odds_value=None):
        """
        Explicitly mark a cell as occupied or free
        
        Args:
            x (float): X coordinate in world frame
            y (float): Y coordinate in world frame
            occupied (bool): True to mark as occupied, False to mark as free
            log_odds_value (float): Optional explicit log-odds value to set
        """
        grid_x, grid_y = self.world_to_grid(x, y)
        
        if not self.is_inside_grid(grid_x, grid_y):
            return
        
        if log_odds_value is not None:
            # Set explicit log-odds value
            self.log_odds_grid[grid_y, grid_x] = np.clip(
                log_odds_value, self.log_odds_min, self.log_odds_max
            )
        else:
            # Set occupied or free
            self.log_odds_grid[grid_y, grid_x] = (
                self.log_odds_max if occupied else self.log_odds_min
            )
        
        # Update probability
        self.grid[grid_y, grid_x] = 1 - (1 / (1 + np.exp(self.log_odds_grid[grid_y, grid_x])))
    
    def reset_grid(self):
        """Reset the grid to initial unknown state"""
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))
        
        # Reset update counter
        self.metadata['updates'] = 0
    
    def save_to_file(self, filename, format='all', include_metadata=True):
        """
        Save the occupancy grid to file(s)
        
        Args:
            filename (str): Base filename without extension
            format (str): Format to save ('png', 'npy', 'csv', 'pickle', or 'all')
            include_metadata (bool): Whether to include metadata in saved files
        """
        # Save as NumPy array
        if format in ['npy', 'all']:
            np.save(f"{filename}.npy", self.grid)
            
            if include_metadata:
                # Save metadata separately
                metadata_filename = f"{filename}_metadata.pkl"
                with open(metadata_filename, 'wb') as f:
                    pickle.dump(self.metadata, f)
        
        # Save as CSV
        if format in ['csv', 'all']:
            np.savetxt(f"{filename}.csv", self.grid, delimiter=',')
            
            if include_metadata:
                # Save metadata as JSON
                import json
                metadata_filename = f"{filename}_metadata.json"
                with open(metadata_filename, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
        
        # Save as pickle (complete object)
        if format in ['pickle', 'all']:
            with open(f"{filename}.pkl", 'wb') as f:
                pickle.dump(self, f)
        
        # Save as PNG image
        if format in ['png', 'all']:
            self.save_as_image(f"{filename}.png", include_metadata)
    
    def save_as_image(self, filename, include_metadata=True):
        """
        Save the occupancy grid as an image
        
        Args:
            filename (str): Filename to save the image
            include_metadata (bool): Whether to include metadata in the image
        """
        # Create a figure with a specific size
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111)
        
        # Custom colormap for occupancy grid
        cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
        bounds = [0, 0.4, 0.6, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot the grid
        img = ax.imshow(
            self.grid, 
            cmap=cmap, 
            norm=norm, 
            interpolation='nearest',
            extent=[-self.width/2, self.width/2, -self.height/2, self.height/2]
        )
        
        # Add colorbar
        cbar = fig.colorbar(img, ax=ax, ticks=[0.2, 0.5, 0.8])
        cbar.ax.set_yticklabels(['Free', 'Unknown', 'Occupied'])
        
        # Add grid lines
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Occupancy Grid Map')
        
        # Add metadata text if requested
        if include_metadata:
            metadata_str = (
                f"Resolution: {self.resolution:.3f}m/cell\n"
                f"Dimensions: {self.width:.1f}m × {self.height:.1f}m\n"
                f"Grid Size: {self.grid_width}×{self.grid_height} cells\n"
                f"Updates: {self.metadata['updates']}"
            )
            ax.text(
                0.02, 0.02, metadata_str,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                va='bottom'
            )
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
    
    @staticmethod
    def load_from_file(filename):
        """
        Load an occupancy grid from file
        
        Args:
            filename (str): Filename to load from
            
        Returns:
            OccupancyGrid: Loaded occupancy grid
        """
        # Check file extension
        extension = os.path.splitext(filename)[1].lower()
        
        if extension == '.pkl':
            # Load from pickle
            with open(filename, 'rb') as f:
                return pickle.load(f)
        elif extension == '.npy':
            # Load grid from NumPy and metadata from pickle if available
            grid = np.load(filename)
            
            # Try to load metadata
            metadata_filename = filename.replace('.npy', '_metadata.pkl')
            metadata = None
            if os.path.exists(metadata_filename):
                try:
                    with open(metadata_filename, 'rb') as f:
                        metadata = pickle.load(f)
                except:
                    metadata = None
            
            # Create new grid with loaded data
            if metadata:
                # Create with original parameters
                grid_obj = OccupancyGrid(
                    resolution=metadata['resolution'],
                    width=metadata['width'],
                    height=metadata['height']
                )
                
                # Replace grid and metadata
                grid_obj.grid = grid
                grid_obj.log_odds_grid = np.log(grid / (1 - grid + 1e-10))  # Avoid div by zero
                grid_obj.metadata = metadata
            else:
                # Create with default parameters
                grid_obj = OccupancyGrid()
                
                # Replace grid
                grid_height, grid_width = grid.shape
                if grid_height == grid_obj.grid_height and grid_width == grid_obj.grid_width:
                    grid_obj.grid = grid
                    grid_obj.log_odds_grid = np.log(grid / (1 - grid + 1e-10))
                else:
                    print(f"Warning: Grid dimensions mismatch. Resizing loaded grid.")
                    # Resize grid to match
                    from scipy.ndimage import zoom
                    zoom_x = grid_obj.grid_width / grid_width
                    zoom_y = grid_obj.grid_height / grid_height
                    grid_obj.grid = zoom(grid, (zoom_y, zoom_x), order=0)
                    grid_obj.log_odds_grid = np.log(grid_obj.grid / (1 - grid_obj.grid + 1e-10))
            
            return grid_obj
        else:
            raise ValueError(f"Unsupported file format: {extension}")


# Example usage
if __name__ == "__main__":
    # Create a test grid
    grid = OccupancyGrid(resolution=0.1, width=10.0, height=10.0)
    
    # Add some obstacles
    for x in range(-4, 5):
        grid.mark_cell(x, -4, occupied=True)
        grid.mark_cell(x, 4, occupied=True)
    
    for y in range(-4, 5):
        grid.mark_cell(-4, y, occupied=True)
        grid.mark_cell(4, y, occupied=True)
    
    # Add a simulated scan
    robot_x, robot_y = 0, 0
    scan_x = [3, 3, 3, 3, 3]
    scan_y = [-3, -1.5, 0, 1.5, 3]
    
    # Update grid with scan
    grid.update_grid(robot_x, robot_y, scan_x, scan_y)
    
    # Save to file
    grid.save_to_file("test_grid", format='all')
    
    print("Test grid saved to test_grid.*")