import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import math

class OccupancyGrid:
    def __init__(self, resolution=0.1, size=20):
        """
        Initialize Occupancy Grid
        resolution: size of each cell in meters
        size: size of the environment in meters
        """
        self.resolution = resolution
        self.size = size
        
        # Calculate grid dimensions
        self.grid_size = int(size / resolution)
        
        # Initialize grid with 0.5 (unknown) probability
        self.grid = np.ones((self.grid_size, self.grid_size)) * 0.5
        
        # Log odds ratio parameters
        self.l0 = 0  # Prior probability in log odds
        self.locc = 0.8  # Probability of occupation
        self.lfree = -0.4  # Probability of free space
        
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int((x + self.size/2) / self.resolution)
        grid_y = int((y + self.size/2) / self.resolution)
        return np.clip(grid_x, 0, self.grid_size-1), np.clip(grid_y, 0, self.grid_size-1)
    
    def update_cell(self, x, y, occupied):
        """Update single cell using log odds"""
        grid_x, grid_y = self.world_to_grid(x, y)
        
        # Current value in log odds
        l = np.log(self.grid[grid_x, grid_y] / (1 - self.grid[grid_x, grid_y]))
        
        # Update
        if occupied:
            l = l + self.locc
        else:
            l = l + self.lfree
            
        # Convert back to probability
        self.grid[grid_x, grid_y] = 1 - (1 / (1 + np.exp(l)))

    def bresenham_line(self, x0, y0, x1, y1):
        """Implementation of Bresenham's line algorithm for float coordinates"""
        # Convert to grid coordinates
        x0_grid, y0_grid = self.world_to_grid(x0, y0)
        x1_grid, y1_grid = self.world_to_grid(x1, y1)
        
        points = []
        dx = abs(x1_grid - x0_grid)
        dy = abs(y1_grid - y0_grid)
        
        x = x0_grid
        y = y0_grid
        
        step_x = 1 if x1_grid > x0_grid else -1
        step_y = 1 if y1_grid > y0_grid else -1
        
        if dx > dy:
            err = dx / 2
            while x != x1_grid:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += step_y
                    err += dx
                x += step_x
        else:
            err = dy / 2
            while y != y1_grid:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += step_x
                    err += dy
                y += step_y
                
        points.append((x, y))
        return points
    
    def update_line(self, x0, y0, x1, y1):
        """Update cells along a line"""
        points = self.bresenham_line(x0, y0, x1, y1)
        
        # Update all points except the last one as free
        for x, y in points[:-1]:
            self.grid[x, y] = max(0.3, self.grid[x, y] - 0.1)  # Mark as increasingly free
            
        # Update the last point as occupied
        if points:
            x, y = points[-1]
            self.grid[x, y] = min(0.7, self.grid[x, y] + 0.1)  # Mark as increasingly occupied

    def update_from_scan(self, robot_x, robot_y, ranges, angles):
        """Update grid using a laser scan"""
        for range_val, angle in zip(ranges, angles):
            if range_val < self.size:  # Ignore readings beyond map size
                # Calculate endpoint
                x = robot_x + range_val * np.cos(angle)
                y = robot_y + range_val * np.sin(angle)
                
                # Update cells along the beam
                self.update_line(robot_x, robot_y, x, y)

class LidarMapperWithGrid:
    def __init__(self, data_content, max_samples=3000):
        self.data_content = data_content
        self.max_samples = max_samples
        
        # Create occupancy grid
        self.grid = OccupancyGrid(resolution=0.1, size=20)
        
        # Create figure with three subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Setup polar plot (current scan)
        self.scan_line, = self.ax1.plot([], [], 'b.')
        self.robot_position_polar, = self.ax1.plot([], [], 'r^', markersize=10, label='Robot')
        self.ax1.set_title("Current LIDAR Scan")
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Setup cartesian plot (point cloud map)
        self.ax2.set_title("Point Cloud Map")
        self.ax2.grid(True)
        self.ax2.set_aspect('equal')
        self.robot_path, = self.ax2.plot([], [], 'r-', label='Robot Path', linewidth=2)
        
        # Create custom colormap for distance visualization
        colors = ['darkred', 'red', 'orange', 'yellow', 'green']
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        # Initialize storage for robot path and map points
        self.path_x = []
        self.path_y = []
        self.map_points = {
            'x': [],
            'y': [],
            'distances': []
        }
        
        # Initialize counter
        self.sample_count = 0

    def classify_points(self, distances):
        classifications = np.zeros_like(distances)
        for i, d in enumerate(distances):
            if d <= 2:
                classifications[i] = 0
            elif d <= 4:
                classifications[i] = 1
            elif d <= 6:
                classifications[i] = 2
            elif d <= 8:
                classifications[i] = 3
            else:
                classifications[i] = 4
        return classifications / 4

    def transform_to_cartesian(self, ranges, angles, robot_x, robot_y, robot_theta):
        local_x = ranges * np.cos(angles)
        local_y = ranges * np.sin(angles)
        
        cos_theta = np.cos(robot_theta)
        sin_theta = np.sin(robot_theta)
        
        global_x = local_x * cos_theta - local_y * sin_theta + robot_x
        global_y = local_x * sin_theta + local_y * cos_theta + robot_y
        
        return global_x, global_y

    def parse_flaser_line(self, line):
        parts = line.strip().split()
        if parts[0] == "FLASER":
            num_readings = int(parts[1])
            readings = [float(x) for x in parts[2:2+num_readings]]
            x = float(parts[-6])
            y = float(parts[-5])
            theta = float(parts[-4])
            return readings, (x, y, theta)
        return None, None

    def read_data(self):
        self.lines = [line for line in self.data_content.split('\n') 
                     if line.strip().startswith("FLASER")]
        self.lines = self.lines[:self.max_samples]
        self.current_line = 0
        print(f"Total FLASER samples to process: {len(self.lines)}")

    def update(self, frame):
        if self.current_line >= len(self.lines) or self.sample_count >= self.max_samples:
            return self.scan_line, self.robot_position_polar, self.robot_path

        readings, pose = self.parse_flaser_line(self.lines[self.current_line])
        if readings and pose:
            x, y, theta = pose
            
            # Update robot path
            self.path_x.append(x)
            self.path_y.append(y)
            
            # Update polar plot (current scan)
            angles = np.linspace(0, 2*np.pi, len(readings))
            colors = self.cmap(self.classify_points(readings))
            self.ax1.clear()
            self.ax1.scatter(angles, readings, c=colors, s=20)
            self.ax1.plot([theta], [0], 'r^', markersize=10, label='Robot')
            self.ax1.set_title(f"Current Scan - Sample {self.sample_count}/{self.max_samples}")
            self.ax1.grid(True)
            self.ax1.legend()
            
            # Transform and update map
            scan_x, scan_y = self.transform_to_cartesian(readings, angles, x, y, theta)
            self.map_points['x'].extend(scan_x)
            self.map_points['y'].extend(scan_y)
            self.map_points['distances'].extend(readings)
            
            # Update cartesian plot (point cloud map)
            self.ax2.clear()
            self.ax2.set_title("Point Cloud Map")
            
            colors = self.cmap(self.classify_points(self.map_points['distances']))
            self.ax2.scatter(self.map_points['x'], self.map_points['y'], 
                           c=colors, s=2, alpha=0.5)
            
            self.ax2.plot(self.path_x, self.path_y, 'r-', 
                         label='Robot Path', linewidth=2)
            self.ax2.plot(x, y, 'r^', markersize=10, label='Current Position')
            
            self.ax2.grid(True)
            self.ax2.legend()
            self.ax2.set_aspect('equal')
            
            # Auto-adjust map bounds
            margin = 2
            x_min, x_max = min(self.path_x), max(self.path_x)
            y_min, y_max = min(self.path_y), max(self.path_y)
            self.ax2.set_xlim(x_min - margin, x_max + margin)
            self.ax2.set_ylim(y_min - margin, y_max + margin)
            
            # Update occupancy grid
            self.grid.update_from_scan(x, y, readings, angles)
            
            # Update occupancy grid visualization
            self.ax3.clear()
            self.ax3.imshow(self.grid.grid.T, 
                          cmap='binary',
                          origin='lower',
                          extent=[-self.grid.size/2, self.grid.size/2, 
                                 -self.grid.size/2, self.grid.size/2])
            self.ax3.plot(x, y, 'r^', markersize=10, label='Robot')
            self.ax3.plot(self.path_x, self.path_y, 'r-', 
                         label='Robot Path', linewidth=2)
            self.ax3.set_title("Occupancy Grid Map")
            self.ax3.grid(True)
            self.ax3.legend()
            self.ax3.set_aspect('equal')

            self.sample_count += 1

        self.current_line += 1
        self.fig.tight_layout()
        return self.scan_line, self.robot_position_polar, self.robot_path

    def animate(self):
        self.read_data()
        anim = FuncAnimation(self.fig, self.update, frames=None,
                           interval=50, blit=False, cache_frame_data=False)
        plt.show()
# Usage
if __name__ == "__main__":
    # Read the data content
    with open('sample_data/mit-killian.clf', 'r') as file:
        data_content = file.read()
    
    # Create mapper with 3000 samples limit
    mapper = LidarMapperWithGrid(data_content, max_samples=3000)
    mapper.animate()