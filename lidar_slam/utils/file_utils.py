"""
File Utilities Module

This module provides utility functions for file operations related to
LiDAR data processing and map storage.
"""

import os
import time
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, creating it if necessary
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        bool: True if directory exists or was created, False otherwise
    """
    if not directory_path:
        return True  # Current directory always exists
        
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
            return True
        except Exception as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False
    return True

def generate_timestamped_filename(prefix="", directory=""):
    """
    Generate a timestamped filename
    
    Args:
        prefix (str): Prefix for the filename
        directory (str): Directory for the file
        
    Returns:
        str: Full path with timestamped filename
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}" if prefix else f"file_{timestamp}"
    
    if directory:
        ensure_directory_exists(directory)
        return os.path.join(directory, filename)
    else:
        return filename

def save_grid_as_image(grid, filename, resolution=0.05, width=20, height=20, 
                      robot_path=None, start_position=None, current_position=None,
                      include_metadata=True, dpi=300):
    """
    Save an occupancy grid as an image
    
    Args:
        grid (numpy.ndarray): The occupancy grid to save
        filename (str): Base filename without extension
        resolution (float): Grid resolution in meters
        width (float): Grid width in meters
        height (float): Grid height in meters
        robot_path (list): List of (x,y) tuples for robot path
        start_position (tuple): (x,y) coordinates of start position
        current_position (tuple): (x,y) coordinates of current position
        include_metadata (bool): Whether to include metadata in the image
        dpi (int): DPI for the saved image
        
    Returns:
        str: Path to the saved image
    """
    img_filename = f"{filename}.png"
    
    try:
        # Create a figure for the image
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Custom colormap: white (unknown), black (occupied), light gray (free)
        cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
        bounds = [0, 0.4, 0.6, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # Plot the grid
        img = ax.imshow(grid, cmap=cmap, norm=norm, origin='lower',
                      extent=[-width/2, width/2, -height/2, height/2])
        
        # Add grid lines
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Plot robot path if provided
        if robot_path is not None and len(robot_path) > 0:
            path_x, path_y = zip(*robot_path)
            ax.plot(path_x, path_y, 'r-', linewidth=2, label='Robot Path')
        
        # Plot start position if provided
        if start_position is not None:
            ax.scatter(start_position[0], start_position[1], c='green', s=100, 
                      marker='*', label='Start Position')
        
        # Plot current position if provided
        if current_position is not None:
            ax.scatter(current_position[0], current_position[1], c='blue', s=100, 
                      marker='*', linewidth=1, label='Final Position')
        
        # Add title and labels
        ax.set_title('Occupancy Grid Map')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        
        # Add legend if any path or positions were added
        if robot_path is not None or start_position is not None or current_position is not None:
            ax.legend(loc='best')
        
        # Add metadata as text if requested
        if include_metadata:
            metadata_text = (
                f"Resolution: {resolution:.3f}m/cell\n"
                f"Dimensions: {width:.1f}m × {height:.1f}m\n"
                f"Grid Size: {grid.shape[1]}×{grid.shape[0]} cells"
            )
            plt.figtext(0.02, 0.02, metadata_text, wrap=True, fontsize=8)
        
        # Save the figure
        plt.savefig(img_filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved grid as image: {img_filename}")
        return img_filename
    
    except Exception as e:
        print(f"Error saving grid as image: {e}")
        return None

def list_data_files(directory, extension=".clf"):
    """
    List all data files in a directory with a specific extension
    
    Args:
        directory (str): Directory to search
        extension (str): File extension to filter by
        
    Returns:
        list: List of file paths
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
        
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    
    return sorted(files)