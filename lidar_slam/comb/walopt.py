import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import os
import time
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening

class ThinWallOccupancyGrid:
    """Enhanced occupancy grid that produces thinner, more accurate walls"""
    
    def __init__(self, resolution=0.05, initial_width=10, initial_height=10, 
                 expansion_factor=1.5, sensor_noise_variance=0.01):
        """
        Initialize an occupancy grid optimized for thin walls
        """
        self.resolution = resolution
        self.expansion_factor = expansion_factor
        self.sensor_noise_variance = sensor_noise_variance
        
        # Calculate initial grid dimensions
        self.grid_width = int(initial_width / resolution)
        self.grid_height = int(initial_height / resolution)
        
        # Initialize grid with unknown values (0.5 represents unknown)
        self.grid = np.ones((self.grid_height, self.grid_width)) * 0.5
        
        # Track actual world dimensions
        self.width = initial_width
        self.height = initial_height
        self.origin_x = initial_width / 2
        self.origin_y = initial_height / 2
        self.x_min = -self.origin_x
        self.x_max = self.origin_x
        self.y_min = -self.origin_y
        self.y_max = self.origin_y
        
        # Log odds grid for Bayesian updates
        self.log_odds_grid = np.zeros((self.grid_height, self.grid_width))
        
        # OPTIMIZED: More balanced parameters for thin walls
        self.log_odds_occupied = np.log(0.75/0.25)   # Strong but not overwhelming occupied evidence
        self.log_odds_free = np.log(0.25/0.75)       # Balanced free evidence
        
        # Parameters for thin walls
        self.log_odds_min = np.log(0.01/0.99)
        self.log_odds_max = np.log(0.99/0.01)
        
        # NEW: Enhanced parameters for thin walls
        self.enable_endpoint_only_updates = True      # Only mark endpoints as occupied
        self.enable_probabilistic_ray_casting = True  # Use probabilistic ray casting
        self.enable_subpixel_accuracy = True          # Use subpixel accuracy for endpoints
        self.enable_motion_compensation = True        # Compensate for robot motion during scan
        
        # Ray casting parameters
        self.ray_decimation_factor = 1.0             # Factor to thin out rays (1.0 = all rays)
        self.endpoint_uncertainty_radius = 0.5        # Uncertainty radius around endpoints (in grid cells)
        self.free_space_decay = 0.8                  # How strongly to mark free space
        
        # Motion compensation
        self.previous_pose = None
        self.motion_threshold = 0.01  # meters - minimum motion to trigger compensation
        
        # Wall thinning post-processing
        self.enable_wall_thinning = True              # Enable morphological wall thinning
        self.thinning_frequency = 50                  # Apply thinning every N updates
        self.update_counter = 0
        
        # Performance tracking
        self.update_count = 0
        self.last_resize_count = 0
        self.resize_frequency = 10
        
        # Statistics
        self.stats = {
            'free_cell_count': 0,
            'occupied_cell_count': 0,
            'unknown_cell_count': self.grid_width * self.grid_height,
            'updates': 0,
            'resizes': 0,
            'wall_thinning_operations': 0
        }
        
        print("[ThinWallGrid] Initialized with thin wall optimization")
        print(f"  - Endpoint-only updates: {self.enable_endpoint_only_updates}")
        print(f"  - Subpixel accuracy: {self.enable_subpixel_accuracy}")
        print(f"  - Motion compensation: {self.enable_motion_compensation}")
        print(f"  - Wall thinning: {self.enable_wall_thinning}")
    
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
    
    def world_to_grid_subpixel(self, x, y):
        """Convert world coordinates to grid coordinates with subpixel accuracy"""
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        
        grid_x = (x_array + self.origin_x) / self.resolution
        grid_y = (y_array + self.origin_y) / self.resolution
        
        if np.isscalar(x) and np.isscalar(y):
            return float(grid_x.item()), float(grid_y.item())
        else:
            return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
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
        # Implementation similar to previous version
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
        
        # Calculate placement offset
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
        
        # Copy old data
        new_grid[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.grid
        new_log_odds[y_offset:y_offset+old_height, x_offset:x_offset+old_width] = self.log_odds_grid
        
        # Update origin if needed
        if direction == 'left':
            self.origin_x += additional_width * self.resolution
        elif direction == 'down':
            self.origin_y += additional_height * self.resolution
        
        # Update dimensions
        self.grid_width = new_width
        self.grid_height = new_height
        self.width = new_width * self.resolution
        self.height = new_height * self.resolution
        
        # Update grids
        self.grid = new_grid
        self.log_odds_grid = new_log_odds
        
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
    
    def update_grid_thin_walls(self, robot_x, robot_y, scan_x, scan_y, robot_theta=None):
        """
        Update the occupancy grid with optimizations for thin walls
        """
        # Check if we need to expand the grid
        self.check_and_expand_grid(scan_x, scan_y)
        
        # Motion compensation
        current_pose = (robot_x, robot_y, robot_theta if robot_theta else 0)
        if self.enable_motion_compensation and self.previous_pose is not None:
            # Check if robot moved significantly
            dx = robot_x - self.previous_pose[0]
            dy = robot_y - self.previous_pose[1]
            motion_distance = math.sqrt(dx*dx + dy*dy)
            
            if motion_distance > self.motion_threshold:
                # Apply motion compensation (simple interpolation)
                # For now, just use the current position
                pass
        
        self.previous_pose = current_pose
        
        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)
        
        # Create temporary grids for updates
        temp_log_odds_grid = np.zeros_like(self.log_odds_grid)
        
        # Process each scan point with optimizations for thin walls
        num_rays = len(scan_x)
        ray_step = max(1, int(1.0 / self.ray_decimation_factor))
        
        for i in range(0, num_rays, ray_step):
            scan_point_x, scan_point_y = scan_x[i], scan_y[i]
            
            if self.enable_subpixel_accuracy:
                # Use subpixel accuracy for endpoint
                endpoint_grid_x, endpoint_grid_y = self.world_to_grid_subpixel(scan_point_x, scan_point_y)
            else:
                endpoint_grid_x, endpoint_grid_y = self.world_to_grid(scan_point_x, scan_point_y)
            
            # Skip if endpoint is outside grid bounds
            if (endpoint_grid_x < 0 or endpoint_grid_x >= self.grid_width-1 or 
                endpoint_grid_y < 0 or endpoint_grid_y >= self.grid_height-1):
                continue
            
            # Calculate scan distance for uncertainty modeling
            scan_distance = math.sqrt((robot_x - scan_point_x)**2 + (robot_y - scan_point_y)**2)
            distance_factor = 1.0 + scan_distance * self.sensor_noise_variance
            
            if self.enable_endpoint_only_updates:
                # OPTIMIZATION 1: Only mark the endpoint as occupied, not the entire ray
                if self.enable_subpixel_accuracy:
                    # Distribute the occupancy evidence among neighboring cells
                    self._update_subpixel_occupancy(temp_log_odds_grid, endpoint_grid_x, endpoint_grid_y, 
                                                   self.log_odds_occupied / distance_factor)
                else:
                    # Standard integer grid update
                    endpoint_x_int, endpoint_y_int = int(endpoint_grid_x), int(endpoint_grid_y)
                    if (0 <= endpoint_x_int < self.grid_width and 0 <= endpoint_y_int < self.grid_height):
                        temp_log_odds_grid[endpoint_y_int, endpoint_x_int] += self.log_odds_occupied / distance_factor
            else:
                # Traditional ray casting with optimizations
                if self.enable_probabilistic_ray_casting:
                    # Use probabilistic ray casting
                    self._probabilistic_ray_casting(temp_log_odds_grid, robot_grid_x, robot_grid_y, 
                                                   endpoint_grid_x, endpoint_grid_y, distance_factor)
                else:
                    # Standard ray casting
                    self._standard_ray_casting(temp_log_odds_grid, robot_grid_x, robot_grid_y, 
                                              int(endpoint_grid_x), int(endpoint_grid_y), distance_factor)
        
        # Apply updates to main grid
        self.log_odds_grid += temp_log_odds_grid
        
        # Clamp values
        self.log_odds_grid = np.clip(self.log_odds_grid, self.log_odds_min, self.log_odds_max)
        
        # Convert to probabilities
        self.grid = 1 - 1 / (1 + np.exp(self.log_odds_grid))
        
        # Apply wall thinning periodically
        self.update_counter += 1
        if self.enable_wall_thinning and self.update_counter % self.thinning_frequency == 0:
            self._thin_walls()
        
        # Update statistics
        self.stats['updates'] += 1
        self.stats['free_cell_count'] = np.sum(self.grid < 0.4)
        self.stats['occupied_cell_count'] = np.sum(self.grid > 0.6)
        self.stats['unknown_cell_count'] = self.grid_width * self.grid_height - (
            self.stats['free_cell_count'] + self.stats['occupied_cell_count'])
    
    def _update_subpixel_occupancy(self, temp_grid, grid_x, grid_y, evidence):
        """Update occupancy with subpixel accuracy using bilinear interpolation"""
        # Get integer coordinates and fractional parts
        x_int = int(np.floor(grid_x))
        y_int = int(np.floor(grid_y))
        x_frac = grid_x - x_int
        y_frac = grid_y - y_int
        
        # Calculate weights for bilinear interpolation
        weights = [
            (1 - x_frac) * (1 - y_frac),  # Top-left
            x_frac * (1 - y_frac),        # Top-right
            (1 - x_frac) * y_frac,        # Bottom-left
            x_frac * y_frac               # Bottom-right
        ]
        
        # Update neighboring cells with weighted evidence
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        
        for offset, weight in zip(offsets, weights):
            cell_x = x_int + offset[0]
            cell_y = y_int + offset[1]
            
            if (0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height):
                temp_grid[cell_y, cell_x] += evidence * weight
    
    def _probabilistic_ray_casting(self, temp_grid, start_x, start_y, end_x, end_y, distance_factor):
        """Probabilistic ray casting that marks free space with uncertainty"""
        # Get cells along the ray
        cells_x, cells_y = self._bresenham_line(start_x, start_y, int(end_x), int(end_y))
        
        # Mark cells along ray as free, with decreasing confidence toward the end
        for j in range(len(cells_x) - 1):  # Exclude endpoint
            cell_x, cell_y = cells_x[j], cells_y[j]
            if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                # Calculate distance from robot
                cell_distance = math.sqrt((cell_x - start_x)**2 + (cell_y - start_y)**2) * self.resolution
                
                # Apply free space update with decay
                free_update = self.log_odds_free * self.free_space_decay
                free_update = free_update * (1.0 - cell_distance * self.sensor_noise_variance / 4)
                free_update = max(free_update, self.log_odds_free * 0.3)
                
                temp_grid[cell_y, cell_x] += free_update
        
        # Mark endpoint as occupied
        end_x_int, end_y_int = int(end_x), int(end_y)
        if 0 <= end_x_int < self.grid_width and 0 <= end_y_int < self.grid_height:
            occupied_update = self.log_odds_occupied / distance_factor
            temp_grid[end_y_int, end_x_int] += occupied_update
    
    def _standard_ray_casting(self, temp_grid, start_x, start_y, end_x, end_y, distance_factor):
        """Standard ray casting implementation"""
        # Get cells along the ray
        cells_x, cells_y = self._bresenham_line(start_x, start_y, end_x, end_y)
        
        # Mark free space
        for j in range(len(cells_x) - 1):
            cell_x, cell_y = cells_x[j], cells_y[j]
            if 0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height:
                temp_grid[cell_y, cell_x] += self.log_odds_free * self.free_space_decay
        
        # Mark endpoint as occupied
        if 0 <= end_x < self.grid_width and 0 <= end_y < self.grid_height:
            temp_grid[end_y, end_x] += self.log_odds_occupied / distance_factor
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm"""
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
    
    def _thin_walls(self):
        """Apply morphological operations to thin walls"""
        # Create binary occupancy grid
        binary_grid = (self.grid > 0.6).astype(np.uint8)
        
        # Apply morphological opening to remove noise and thin walls
        # Opening = erosion followed by dilation
        kernel = np.ones((3, 3), np.uint8)
        thinned_grid = binary_opening(binary_grid, structure=kernel)
        
        # Apply the thinning only to cells that were originally occupied
        # This prevents removing legitimate walls
        thinning_mask = binary_grid & ~thinned_grid
        
        # Reduce occupancy probability for thinned areas
        reduction_factor = 0.3  # Reduce by 70%
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if thinning_mask[y, x]:
                    # Reduce log odds for this cell
                    self.log_odds_grid[y, x] *= reduction_factor
        
        # Update probability grid
        self.grid = 1 - 1 / (1 + np.exp(self.log_odds_grid))
        
        self.stats['wall_thinning_operations'] += 1
    
    def get_grid_for_display(self, show_binary=False):
        """Get grid for display with options"""
        if show_binary:
            # Return binary version for clear wall visualization
            binary_grid = np.zeros_like(self.grid)
            binary_grid[self.grid > 0.6] = 1.0
            binary_grid[self.grid < 0.4] = 0.0
            binary_grid[(self.grid >= 0.4) & (self.grid <= 0.6)] = 0.5
            return binary_grid
        else:
            return self.grid.copy()
    
    def save_to_file(self, filename, format='png', include_metadata=True, dpi=300, 
                    robot_path=None, start_position=None, current_position=None,
                    show_comparison=True):
        """Save the thin wall occupancy grid"""
        saved_files = []
        
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"thin_wall_occupancy_grid_{timestamp}"
        
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save as image
        if format.lower() == 'png' or format.lower() == 'all':
            img_filename = f"{filename}.png"
            
            if show_comparison:
                # Show both probability and binary versions
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                
                # Plot 1: Probability grid
                cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
                bounds = [0, 0.4, 0.6, 1]
                norm = colors.BoundaryNorm(bounds, cmap.N)
                
                extent = [-self.origin_x, -self.origin_x + self.width,
                         -self.origin_y, -self.origin_y + self.height]
                
                im1 = ax1.imshow(self.grid, cmap=cmap, norm=norm, origin='lower', extent=extent)
                ax1.set_title('Probability Grid (Thin Walls)')
                ax1.set_xlabel('X (meters)')
                ax1.set_ylabel('Y (meters)')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Binary grid
                binary_display = self.get_grid_for_display(show_binary=True)
                im2 = ax2.imshow(binary_display, cmap=cmap, norm=norm, origin='lower', extent=extent)
                ax2.set_title('Binary Grid (Clear Walls)')
                ax2.set_xlabel('X (meters)')
                ax2.set_ylabel('Y (meters)')
                ax2.grid(True, alpha=0.3)
                
                # Add robot path if provided
                for ax in [ax1, ax2]:
                    if robot_path:
                        path_x, path_y = zip(*robot_path)
                        ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7)
                    if start_position:
                        ax.scatter(start_position[0], start_position[1], c='green', s=100, marker='*')
                    if current_position:
                        ax.scatter(current_position[0], current_position[1], c='blue', s=100, marker='*')
            else:
                # Show only the main grid
                fig, ax = plt.subplots(figsize=(10, 10))
                
                cmap = colors.ListedColormap(['white', 'lightgray', 'black'])
                bounds = [0, 0.4, 0.6, 1]
                norm = colors.BoundaryNorm(bounds, cmap.N)
                
                extent = [-self.origin_x, -self.origin_x + self.width,
                         -self.origin_y, -self.origin_y + self.height]
                
                im = ax.imshow(self.grid, cmap=cmap, norm=norm, origin='lower', extent=extent)
                ax.set_title('Thin Wall Occupancy Grid')
                ax.set_xlabel('X (meters)')
                ax.set_ylabel('Y (meters)')
                ax.grid(True, alpha=0.3)
                
                # Add robot path and positions
                if robot_path:
                    path_x, path_y = zip(*robot_path)
                    ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7, label='Robot Path')
                if start_position:
                    ax.scatter(start_position[0], start_position[1], c='green', s=100, marker='*', label='Start')
                if current_position:
                    ax.scatter(current_position[0], current_position[1], c='blue', s=100, marker='*', label='End')
                ax.legend()
            
            # Add metadata
            if include_metadata:
                metadata_text = (
                    f"Resolution: {self.resolution:.3f}m/cell\n"
                    f"Dimensions: {self.width:.1f}m × {self.height:.1f}m\n"
                    f"Occupied: {self.stats['occupied_cell_count']} cells\n"
                    f"Free: {self.stats['free_cell_count']} cells\n"
                    f"Updates: {self.stats['updates']}\n"
                    f"Wall thinning ops: {self.stats['wall_thinning_operations']}\n"
                    f"Endpoint-only: {self.enable_endpoint_only_updates}\n"
                    f"Subpixel accuracy: {self.enable_subpixel_accuracy}"
                )
                plt.figtext(0.02, 0.02, metadata_text, wrap=True, fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(img_filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            saved_files.append(img_filename)
            print(f"Saved thin wall grid as image: {img_filename}")
        
        return saved_files
    
    # Backward compatibility methods
    def update_grid(self, robot_x, robot_y, scan_x, scan_y):
        """Compatibility method"""
        return self.update_grid_thin_walls(robot_x, robot_y, scan_x, scan_y)
    
    def update_grid_vectorized(self, robot_x, robot_y, scan_x, scan_y):
        """Compatibility method"""
        return self.update_grid_thin_walls(robot_x, robot_y, scan_x, scan_y)
    
    def update_grid_with_turn_handling(self, robot_x, robot_y, robot_theta, scan_x, scan_y, scan_ranges=None):
        """Compatibility method with orientation"""
        return self.update_grid_thin_walls(robot_x, robot_y, scan_x, scan_y, robot_theta)