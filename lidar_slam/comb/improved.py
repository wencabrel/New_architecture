import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import os
import time
from scipy.ndimage import binary_dilation

class ImprovedOccupancyGrid:
    """Enhanced occupancy grid that prevents overwriting of strong occupancy evidence"""
    
    def __init__(self, resolution=0.05, initial_width=10, initial_height=10, 
                 expansion_factor=1.5, sensor_noise_variance=0.01):
        """
        Initialize an improved occupancy grid with persistent occupancy features
        """
        self.resolution = resolution
        self.expansion_factor = expansion_factor
        self.sensor_noise_variance = sensor_noise_variance
        
        # Calculate initial grid dimensions
        self.grid_width = int(initial_width / resolution)
        self.grid_height = int(initial_height / resolution)
        
        # Initialize grid with unknown values (0.5 represents unknown)
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5
        
        # Track actual world dimensions (initialized from the center)
        self.width = initial_width
        self.height = initial_height
        
        # Origin of the grid (center of the grid)
        self.origin_x = initial_width / 2
        self.origin_y = initial_height / 2
        
        # World boundaries
        self.x_min = -self.origin_x
        self.x_max = self.origin_x
        self.y_min = -self.origin_y
        self.y_max = self.origin_y
        
        # Log odds version of the grid (for Bayesian updates)
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))
        
        # IMPROVED: More conservative parameters for occupancy update
        self.log_odds_occupied = np.log(0.8/0.2)   # Higher confidence for occupied
        self.log_odds_free = np.log(0.2/0.8)       # Lower confidence for free (more conservative)
        
        # Maximum and minimum log odds to prevent overflow
        self.log_odds_min = np.log(0.01/0.99)  # ~-4.6
        self.log_odds_max = np.log(0.99/0.01)  # ~4.6
        
        # NEW: Occupancy strength grid to track confidence
        self.occupancy_strength = np.zeros((self.grid_height, self.grid_width))
        self.max_strength = 10.0  # Maximum strength value
        
        # NEW: Evidence counter grid to track observation count
        self.evidence_count = np.zeros((self.grid_height, self.grid_width))
        
        # NEW: Last observation grid to track when cells were last updated
        self.last_observation = np.zeros((self.grid_height, self.grid_width))
        self.observation_counter = 0
        
        # Parameters for persistent occupancy
        self.min_observations_for_persistence = 3  # Minimum observations to make a cell persistent
        self.persistence_threshold = 2.0  # Strength threshold for persistence
        self.free_override_strength = 1.5  # Minimum strength needed to override free evidence
        
        # Temporal filtering parameters
        self.temporal_consistency_grid = np.zeros((self.grid_height, self.grid_width))
        self.consistency_threshold = 3
        
        # Performance tracking
        self.update_count = 0
        self.last_resize_count = 0
        self.resize_frequency = 10
        
        # Statistics for map
        self.stats = {
            'free_cell_count': 0,
            'occupied_cell_count': 0,
            'unknown_cell_count': self.grid_width * self.grid_height,
            'updates': 0,
            'resizes': 0,
            'persistent_cells': 0
        }
        
        print("[ImprovedGrid] Initialized with persistent occupancy features")
        print(f"  - Conservative free evidence: {np.exp(self.log_odds_free):.3f}")
        print(f"  - Strong occupied evidence: {np.exp(self.log_odds_occupied):.3f}")
        print(f"  - Persistence after {self.min_observations_for_persistence} observations")
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        
        grid_x = np.floor((x_array + self.origin_x) / self.resolution).astype(int)
        grid_y = np.floor((y_array + self.origin_y) / self.resolution).astype(int)
        
        if np.isscalar(x) and np.isscalar(y):
            return int(grid_x.item()), int(grid_y.item())
        else:
            return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates (center of cell)"""
        grid_x_array = np.asarray(grid_x)
        grid_y_array = np.asarray(grid_y)
        
        x = grid_x_array * self.resolution + self.resolution/2 - self.origin_x
        y = grid_y_array * self.resolution + self.resolution/2 - self.origin_y
        
        if np.isscalar(grid_x) and np.isscalar(grid_y):
            return x.item(), y.item()
        else:
            return x, y
    
    def _expand_grid(self, direction):
        """Expand the occupancy grid in a specified direction"""
        self.stats['resizes'] += 1
        
        old_height, old_width = self.grid.shape
        
        if direction in ['left', 'right']:
            new_width = int(old_width * self.expansion_factor)
            new_height = old_height
            additional_width = new_width - old_width
        else:
            new_height = int(old_height * self.expansion_factor)
            new_width = old_width
            additional_height = new_height - old_height
        
        # Create new grids
        new_grid = np.ones((new_height, new_width)) * 0.5
        new_log_odds = np.zeros((new_height, new_width))
        new_consistency = np.zeros((new_height, new_width))
        new_strength = np.zeros((new_height, new_width))
        new_evidence_count = np.zeros((new_height, new_width))
        new_last_observation = np.zeros((new_height, new_width))
        
        # Calculate where to place the old grid
        if direction == 'left':
            x_offset = additional_width
            y_offset = 0
            self.x_min -= additional_width * self.resolution
        elif direction == 'right':
            x_offset = 0
            y_offset = 0
            self.x_max += additional_width * self.resolution
        elif direction == 'down':
            x_offset = 0
            y_offset = additional_height
            self.y_min -= additional_height * self.resolution
        else:  # 'up'
            x_offset = 0
            y_offset = 0
            self.y_max += additional_height * self.resolution
        
        # Copy old data to new grids
        new_grid[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.grid
        new_log_odds[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.log_odds_grid
        new_consistency[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.temporal_consistency_grid
        new_strength[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.occupancy_strength
        new_evidence_count[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.evidence_count
        new_last_observation[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.last_observation
        
        # Update origin if needed
        if direction == 'left':
            self.origin_x += additional_width * self.resolution
        elif direction == 'down':
            self.origin_y += additional_height * self.resolution
        
        # Update grid dimensions
        self.grid_width = new_width
        self.grid_height = new_height
        self.width = new_width * self.resolution
        self.height = new_height * self.resolution
        
        # Update grids
        self.grid = new_grid
        self.log_odds_grid = new_log_odds
        self.temporal_consistency_grid = new_consistency
        self.occupancy_strength = new_strength
        self.evidence_count = new_evidence_count
        self.last_observation = new_last_observation
        
        print(f"Grid expanded {direction}: New dimensions {new_width}x{new_height} cells")
    
    def check_and_expand_grid(self, world_x, world_y):
        """Check if points are outside current grid bounds and expand if necessary"""
        self.update_count += 1
        
        if self.update_count - self.last_resize_count < self.resize_frequency:
            return False
        
        self.last_resize_count = self.update_count
        
        x_array = np.asarray(world_x)
        y_array = np.asarray(world_y)
        
        min_x, max_x = np.min(x_array), np.max(x_array)
        min_y, max_y = np.min(y_array), np.max(y_array)
        
        padding = 1.0
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        expanded = False
        
        if min_x < self.x_min:
            self._expand_grid('left')
            expanded = True
        
        if max_x > self.x_max:
            self._expand_grid('right')
            expanded = True
        
        if min_y < self.y_min:
            self._expand_grid('down')
            expanded = True
        
        if max_y > self.y_max:
            self._expand_grid('up')
            expanded = True
        
        return expanded
    
    def update_grid_with_persistence(self, robot_x, robot_y, scan_x, scan_y):
        """
        Update the occupancy grid with persistent occupancy features
        This method prevents strong occupancy evidence from being overwritten
        """
        # Increment observation counter
        self.observation_counter += 1
        
        # Check if we need to expand the grid
        self.check_and_expand_grid(scan_x, scan_y)
        
        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        # Convert scan points to grid coordinates
        scan_grid_x, scan_grid_y = self.world_to_grid(scan_x, scan_y)
        
        # Create temporary grids to store updates
        temp_log_odds_grid = np.zeros_like(self.log_odds_grid)
        temp_strength_grid = np.zeros_like(self.occupancy_strength)
        temp_evidence_grid = np.zeros_like(self.evidence_count)
        
        # Process each scan point
        for i in range(len(scan_grid_x)):
            endpoint_x, endpoint_y = scan_grid_x[i], scan_grid_y[i]
            
            # Skip if endpoint is outside grid
            if (endpoint_x < 0 or endpoint_x >= self.grid_width or 
                endpoint_y < 0 or endpoint_y >= self.grid_height):
                continue
            
            # Calculate scan distance for uncertainty model
            scan_distance = np.sqrt((robot_x - scan_x[i])**2 + (robot_y - scan_y[i])**2)
            
            # Adjust occupancy update based on distance
            distance_factor = 1.0 + scan_distance * self.sensor_noise_variance
            occupied_update = self.log_odds_occupied / distance_factor
            
            # Get cells along ray using Bresenham line algorithm
            cells_x, cells_y = self._bresenham_line(robot_grid_x, robot_grid_y, endpoint_x, endpoint_y)
            
            # IMPROVED: Process free space with persistence checking
            for j in range(len(cells_x) - 1):  # Exclude the endpoint
                cell_x, cell_y = cells_x[j], cells_y[j]
                if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                    
                    # Check current occupancy strength
                    current_strength = self.occupancy_strength[cell_y, cell_x]
                    current_evidence = self.evidence_count[cell_y, cell_x]
                    
                    # Calculate free update
                    cell_distance = np.sqrt((cell_x - robot_grid_x)**2 + (cell_y - robot_grid_y)**2) * self.resolution
                    free_update = self.log_odds_free * (1.0 - cell_distance * self.sensor_noise_variance/2)
                    free_update = max(free_update, self.log_odds_free * 0.5)
                    
                    # IMPROVED: Apply persistence logic for free space updates
                    if current_evidence >= self.min_observations_for_persistence and current_strength >= self.persistence_threshold:
                        # This cell has strong occupancy evidence - reduce free update impact
                        persistence_factor = min(0.1, 1.0 / (current_strength + 1))
                        free_update *= persistence_factor
                        
                        # Only update if we have very strong new evidence
                        if abs(free_update) < abs(self.log_odds_free) * 0.1:
                            continue  # Skip this update to preserve occupancy
                    
                    # Apply the free space update
                    temp_log_odds_grid[cell_y, cell_x] += free_update
                    temp_evidence_grid[cell_y, cell_x] += 1
            
            # IMPROVED: Process occupied endpoint with strength tracking
            if (endpoint_x >= 0 and endpoint_x < self.grid_width and
                endpoint_y >= 0 and endpoint_y < self.grid_height):
                
                # Apply occupied update
                temp_log_odds_grid[endpoint_y, endpoint_x] += occupied_update
                
                # Increase strength for occupied observations
                strength_increase = min(1.0, 2.0 / distance_factor)  # Closer observations get more strength
                temp_strength_grid[endpoint_y, endpoint_x] += strength_increase
                temp_evidence_grid[endpoint_y, endpoint_x] += 1
                
                # Update last observation time
                self.last_observation[endpoint_y, endpoint_x] = self.observation_counter
        
        # Apply all updates to the main grids
        self.log_odds_grid += temp_log_odds_grid
        self.occupancy_strength += temp_strength_grid
        self.evidence_count += temp_evidence_grid
        
        # Clamp strength values
        self.occupancy_strength = np.clip(self.occupancy_strength, 0, self.max_strength)
        
        # Clamp log odds values to prevent numerical issues
        self.log_odds_grid = np.clip(self.log_odds_grid, self.log_odds_min, self.log_odds_max)
        
        # Convert log odds to probabilities
        self.grid = 1 - 1 / (1 + np.exp(self.log_odds_grid))
        
        # Update temporal consistency
        for i in range(len(scan_grid_x)):
            endpoint_x, endpoint_y = scan_grid_x[i], scan_grid_y[i]
            if (0 <= endpoint_x < self.grid_width and 0 <= endpoint_y < self.grid_height):
                self.temporal_consistency_grid[endpoint_y, endpoint_x] += 1
        
        # Decay temporal consistency grid
        self.temporal_consistency_grid *= 0.95
        
        # Update statistics
        self.stats['updates'] += 1
        self.stats['free_cell_count'] = np.sum(self.grid < 0.4)
        self.stats['occupied_cell_count'] = np.sum(self.grid > 0.6)
        self.stats['unknown_cell_count'] = self.grid_width * self.grid_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count'])
        self.stats['persistent_cells'] = np.sum(self.occupancy_strength >= self.persistence_threshold)
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """Compute cells on a line using Bresenham's algorithm"""
        x_cells = []
        y_cells = []
        
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        x, y = x0, y0
        while True:
            x_cells.append(x)
            y_cells.append(y)
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return x_cells, y_cells
    
    def get_grid_for_display(self, show_strength=False, show_evidence=False):
        """
        Get a copy of the grid suitable for display
        """
        if show_strength:
            # Show occupancy strength
            normalized_strength = self.occupancy_strength / self.max_strength
            return normalized_strength
        elif show_evidence:
            # Show evidence count
            max_evidence = np.max(self.evidence_count) if np.max(self.evidence_count) > 0 else 1
            normalized_evidence = self.evidence_count / max_evidence
            return normalized_evidence
        else:
            # Show normal occupancy probability
            return self.grid.copy()
    
    def get_persistence_grid(self):
        """Get a grid showing persistent cells"""
        persistence_grid = np.zeros_like(self.grid)
        persistence_mask = (self.occupancy_strength >= self.persistence_threshold) & (self.evidence_count >= self.min_observations_for_persistence)
        persistence_grid[persistence_mask] = 1.0
        return persistence_grid
    
    def save_to_file(self, filename, format='png', include_metadata=True, dpi=300, 
                    robot_path=None, start_position=None, current_position=None,
                    show_persistence=True):
        """Save the occupancy grid with persistence information"""
        saved_files = []
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"persistent_occupancy_grid_{timestamp}"
        
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                return saved_files
        
        # Save as image (PNG)
        if format.lower() == 'png' or format.lower() == 'all':
            img_filename = f"{filename}.png"
            try:
                fig, axes = plt.subplots(1, 3 if show_persistence else 2, figsize=(18 if show_persistence else 12, 6))
                
                # Plot 1: Normal occupancy grid
                ax1 = axes[0]
                cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
                bounds = [0, 0.4, 0.6, 1]
                norm = colors.BoundaryNorm(bounds, cmap.N)
                
                extent = [-self.origin_x, -self.origin_x + self.width,
                         -self.origin_y, -self.origin_y + self.height]
                
                im1 = ax1.imshow(self.grid, cmap=cmap, norm=norm, origin='lower', extent=extent)
                ax1.set_title('Occupancy Grid')
                ax1.set_xlabel('X (meters)')
                ax1.set_ylabel('Y (meters)')
                ax1.grid(True, alpha=0.3)
                
                # Plot robot path if provided
                if robot_path:
                    path_x, path_y = zip(*robot_path)
                    ax1.plot(path_x, path_y, 'r-', linewidth=2, label='Robot Path')
                
                if start_position:
                    ax1.scatter(start_position[0], start_position[1], c='green', s=100, marker='*', label='Start')
                
                if current_position:
                    ax1.scatter(current_position[0], current_position[1], c='blue', s=100, marker='*', label='End')
                
                ax1.legend()
                
                # Plot 2: Occupancy strength
                ax2 = axes[1]
                strength_display = self.get_grid_for_display(show_strength=True)
                im2 = ax2.imshow(strength_display, cmap='hot', origin='lower', extent=extent)
                ax2.set_title('Occupancy Strength')
                ax2.set_xlabel('X (meters)')
                ax2.set_ylabel('Y (meters)')
                ax2.grid(True, alpha=0.3)
                plt.colorbar(im2, ax=ax2, label='Strength')
                
                # Plot 3: Persistent cells (if enabled)
                if show_persistence:
                    ax3 = axes[2]
                    persistence_display = self.get_persistence_grid()
                    im3 = ax3.imshow(persistence_display, cmap='Reds', origin='lower', extent=extent)
                    ax3.set_title('Persistent Occupancy Cells')
                    ax3.set_xlabel('X (meters)')
                    ax3.set_ylabel('Y (meters)')
                    ax3.grid(True, alpha=0.3)
                    plt.colorbar(im3, ax=ax3, label='Persistent (1=Yes, 0=No)')
                
                # Add metadata
                if include_metadata:
                    metadata_text = (
                        f"Resolution: {self.resolution:.3f}m/cell\n"
                        f"Dimensions: {self.width:.1f}m × {self.height:.1f}m\n"
                        f"Occupied: {self.stats['occupied_cell_count']} cells\n"
                        f"Free: {self.stats['free_cell_count']} cells\n"
                        f"Persistent: {self.stats['persistent_cells']} cells\n"
                        f"Updates: {self.stats['updates']}"
                    )
                    plt.figtext(0.02, 0.02, metadata_text, wrap=True, fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7))
                
                plt.tight_layout()
                plt.savefig(img_filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                
                saved_files.append(img_filename)
                print(f"Saved persistent grid as image: {img_filename}")
            except Exception as e:
                print(f"Error saving grid as image: {e}")
        
        # Save additional data if requested
        if format.lower() == 'npy' or format.lower() == 'all':
            # Save multiple arrays
            arrays_to_save = [
                (f"{filename}_occupancy.npy", self.grid),
                (f"{filename}_log_odds.npy", self.log_odds_grid),
                (f"{filename}_strength.npy", self.occupancy_strength),
                (f"{filename}_evidence.npy", self.evidence_count),
                (f"{filename}_persistence.npy", self.get_persistence_grid())
            ]
            
            for npy_filename, array_data in arrays_to_save:
                try:
                    np.save(npy_filename, array_data)
                    saved_files.append(npy_filename)
                except Exception as e:
                    print(f"Error saving {npy_filename}: {e}")
            
            print(f"Saved persistent grid arrays to {len(arrays_to_save)} files")
        
        return saved_files
    
    # Backward compatibility methods
    def update_grid(self, robot_x, robot_y, scan_x, scan_y):
        """Compatibility method that calls the persistent update"""
        return self.update_grid_with_persistence(robot_x, robot_y, scan_x, scan_y)
    
    def update_grid_vectorized(self, robot_x, robot_y, scan_x, scan_y):
        """Compatibility method that calls the persistent update"""
        return self.update_grid_with_persistence(robot_x, robot_y, scan_x, scan_y)
    
    def update_grid_with_turn_handling(self, robot_x, robot_y, robot_theta, scan_x, scan_y, scan_ranges=None):
        """Enhanced turn handling with persistence"""
        # For now, just call the persistent update method
        # You could add turn-specific logic here if needed
        return self.update_grid_with_persistence(robot_x, robot_y, scan_x, scan_y)