import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class OccupancyGrid:
    """
    Implements an occupancy grid map for robot mapping and localization.
    The grid represents the environment as a 2D grid where each cell contains
    the probability of that cell being occupied.
    """
    def __init__(self, mapXLength, mapYLength, initXY, unitGridSize, lidarFOV, 
                 numSamplesPerRev, lidarMaxRange, wallThickness):
        """
        Initialize occupancy grid with given parameters.
        
        Args:
            mapXLength (float): Initial map length in x direction (meters)
            mapYLength (float): Initial map length in y direction (meters)
            initXY (dict): Initial position {'x': x_pos, 'y': y_pos}
            unitGridSize (float): Size of each grid cell (meters)
            lidarFOV (float): LIDAR field of view (radians)
            numSamplesPerRev (int): Number of LIDAR samples per revolution
            lidarMaxRange (float): Maximum LIDAR range (meters)
            wallThickness (float): Assumed wall thickness (meters)
        """
        # Calculate number of grid cells in each dimension
        num_cells_x = int(mapXLength / unitGridSize)
        num_cells_y = int(mapYLength / unitGridSize)
        
        # Create grid coordinates centered at initial position
        x_coords = np.linspace(-num_cells_x * unitGridSize / 2, 
                             num_cells_x * unitGridSize / 2, 
                             num=num_cells_x + 1) + initXY['x']
        y_coords = np.linspace(-num_cells_y * unitGridSize / 2, 
                             num_cells_y * unitGridSize / 2, 
                             num=num_cells_y + 1) + initXY['y']
        
        # Create 2D grid of coordinates
        self.OccupancyGridX, self.OccupancyGridY = np.meshgrid(x_coords, y_coords)
        
        # Initialize occupancy counts
        # occupancyGridVisited: counts of cell being detected as occupied
        # occupancyGridTotal: total observations of cell
        self.occupancyGridVisited = np.ones((num_cells_x + 1, num_cells_y + 1))
        self.occupancyGridTotal = 2 * np.ones((num_cells_x + 1, num_cells_y + 1))
        
        # Store parameters
        self.unitGridSize = unitGridSize
        self.lidarFOV = lidarFOV
        self.lidarMaxRange = lidarMaxRange
        self.wallThickness = wallThickness
        
        # Store map boundaries
        self.mapXLim = [self.OccupancyGridX[0, 0], self.OccupancyGridX[0, -1]]
        self.mapYLim = [self.OccupancyGridY[0, 0], self.OccupancyGridY[-1, 0]]
        
        # Calculate angular parameters
        self.numSamplesPerRev = numSamplesPerRev
        self.angularStep = lidarFOV / numSamplesPerRev
        self.numSpokes = int(np.rint(2 * np.pi / self.angularStep))
        
        # Pre-compute spoke grid for ray tracing
        spoke_x_grid, spoke_y_grid, bearing_idx_grid, range_idx_grid = self.spokesGrid()
        rays_by_x, rays_by_y, rays_by_range = self.itemizeSpokesGrid(
            spoke_x_grid, spoke_y_grid, bearing_idx_grid, range_idx_grid)
        
        self.radByX = rays_by_x
        self.radByY = rays_by_y
        self.radByR = rays_by_range
        
        # Calculate starting spoke index for LIDAR scan
        # theta = 0 is x direction, spokes = 0 is y direction
        self.spokesStartIdx = int(((self.numSpokes / 2 - self.numSamplesPerRev) / 2) 
                                % self.numSpokes)

    def spokesGrid(self):
        """
        Create a grid of spoke directions for ray tracing.
        The grid represents possible ray directions from a central point.
        0th ray points south, increasing counter-clockwise.
        
        Returns:
            tuple: (x_grid, y_grid, bearing_idx_grid, range_idx_grid)
                  Grids containing ray tracing information
        """
        # Calculate grid dimensions based on max range
        num_half_cells = int(self.lidarMaxRange / self.unitGridSize)
        bearing_idx_grid = np.zeros((2 * num_half_cells + 1, 2 * num_half_cells + 1))
        
        # Create coordinate grids
        x_coords = np.linspace(-self.lidarMaxRange, self.lidarMaxRange, 
                             2 * num_half_cells + 1)
        y_coords = np.linspace(-self.lidarMaxRange, self.lidarMaxRange, 
                             2 * num_half_cells + 1)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Calculate bearing indices for right half of grid
        bearing_idx_grid[:, num_half_cells + 1:] = np.rint(
            (np.pi / 2 + np.arctan(y_grid[:, num_half_cells + 1:] / 
                                 x_grid[:, num_half_cells + 1:])) /
            np.pi / 2 * self.numSpokes - 0.5).astype(int)
        
        # Mirror for left half of grid
        bearing_idx_grid[:, 0:num_half_cells] = (
            np.fliplr(np.flipud(bearing_idx_grid))[:, 0:num_half_cells] + 
            int(self.numSpokes / 2))
        
        # Set vertical line (x = 0)
        bearing_idx_grid[num_half_cells + 1:, num_half_cells] = int(self.numSpokes / 2)
        
        # Calculate range grid
        range_idx_grid = np.sqrt(x_grid**2 + y_grid**2)
        
        return x_grid, y_grid, bearing_idx_grid, range_idx_grid

    def itemizeSpokesGrid(self, x_grid, y_grid, bearing_idx_grid, range_idx_grid):
        """
        Organize ray tracing grid into lists of rays by direction.
        Due to discretization, later theta could have up to 1 deg discretization error.
        
        Args:
            x_grid: Grid of x coordinates
            y_grid: Grid of y coordinates
            bearing_idx_grid: Grid of bearing indices
            range_idx_grid: Grid of range values
            
        Returns:
            tuple: (rays_by_x, rays_by_y, rays_by_range)
                  Lists of ray coordinates and ranges organized by direction
        """
        rays_by_x = []
        rays_by_y = []
        rays_by_range = []
        
        # Group grid points by bearing index
        for bearing_idx in range(self.numSpokes):
            # Find all points with current bearing
            points_at_bearing = np.argwhere(bearing_idx_grid == bearing_idx)
            
            # Store coordinates and ranges for these points
            rays_by_x.append(x_grid[points_at_bearing[:, 0], points_at_bearing[:, 1]])
            rays_by_y.append(y_grid[points_at_bearing[:, 0], points_at_bearing[:, 1]])
            rays_by_range.append(range_idx_grid[points_at_bearing[:, 0], 
                                              points_at_bearing[:, 1]])
            
        return rays_by_x, rays_by_y, rays_by_range

    def expandOccupancyGridHelper(self, position, axis):
        """
        Helper function to expand the occupancy grid in a given direction.
        
        Args:
            position (int): Position to insert new cells (0 for start, shape for end)
            axis (int): Axis along which to expand (0 for y, 1 for x)
        """
        grid_shape = self.occupancyGridVisited.shape
        
        if axis == 0:  # Expand in y direction
            insertion_size = int(grid_shape[0] / 5)
            insertion = np.ones((insertion_size, grid_shape[1]))
            
            if position == 0:  # Add to start
                y_coords = np.linspace(
                    self.mapYLim[0] - insertion_size * self.unitGridSize,
                    self.mapYLim[0],
                    num=insertion_size,
                    endpoint=False
                )
                x_coords = self.OccupancyGridX[0]
            else:  # Add to end
                y_coords = np.linspace(
                    self.mapYLim[1] + self.unitGridSize,
                    self.mapYLim[1] + insertion_size * self.unitGridSize,
                    num=insertion_size,
                    endpoint=False
                )
                x_coords = self.OccupancyGridX[0]
        else:  # Expand in x direction
            insertion_size = int(grid_shape[1] / 5)
            insertion = np.ones((grid_shape[0], insertion_size))
            
            if position == 0:  # Add to start
                x_coords = np.linspace(
                    self.mapXLim[0] - insertion_size * self.unitGridSize,
                    self.mapXLim[0],
                    num=insertion_size,
                    endpoint=False
                )
                y_coords = self.OccupancyGridY[:, 0]
            else:  # Add to end
                x_coords = np.linspace(
                    self.mapXLim[1] + self.unitGridSize,
                    self.mapXLim[1] + insertion_size * self.unitGridSize,
                    num=insertion_size,
                    endpoint=False
                )
                y_coords = self.OccupancyGridY[:, 0]
        
        # Update occupancy grids
        self.occupancyGridVisited = np.insert(self.occupancyGridVisited, 
                                            [position], insertion, axis=axis)
        self.occupancyGridTotal = np.insert(self.occupancyGridTotal, 
                                          [position], 2 * insertion, axis=axis)
        
        # Update coordinate grids
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        self.OccupancyGridX = np.insert(self.OccupancyGridX, [position], x_mesh, axis=axis)
        self.OccupancyGridY = np.insert(self.OccupancyGridY, [position], y_mesh, axis=axis)
        
        # Update map limits
        self.mapXLim[0] = self.OccupancyGridX[0, 0]
        self.mapXLim[1] = self.OccupancyGridX[0, -1]
        self.mapYLim[0] = self.OccupancyGridY[0, 0]
        self.mapYLim[1] = self.OccupancyGridY[-1, 0]

    def expandOccupancyGrid(self, expand_direction):
        """
        Expand the occupancy grid in specified direction.
        
        Args:
            expand_direction (int): Direction to expand
                1: expand left (negative x)
                2: expand right (positive x)
                3: expand down (negative y)
                4: expand up (positive y)
        """
        grid_shape = self.occupancyGridVisited.shape
        
        if expand_direction == 1:  # Expand left
            self.expandOccupancyGridHelper(0, 1)
        elif expand_direction == 2:  # Expand right
            self.expandOccupancyGridHelper(grid_shape[1], 1)
        elif expand_direction == 3:  # Expand down
            self.expandOccupancyGridHelper(0, 0)
        else:  # Expand up
            self.expandOccupancyGridHelper(grid_shape[0], 0)

    def convertRealXYToMapIdx(self, x, y):
        """
        Convert real-world coordinates to grid indices.
        
        Args:
            x (float or array): X coordinate(s) in real world
            y (float or array): Y coordinate(s) in real world
            
        Returns:
            tuple: (x_indices, y_indices) Grid indices corresponding to coordinates
        """
        x_idx = (np.rint((x - self.mapXLim[0]) / self.unitGridSize)).astype(int)
        y_idx = (np.rint((y - self.mapYLim[0]) / self.unitGridSize)).astype(int)
        return x_idx, y_idx

    def checkMapToExpand(self, x, y):
        """
        Check if map needs to be expanded to include given coordinates.
        
        Args:
            x (array): X coordinates to check
            y (array): Y coordinates to check
            
        Returns:
            int: Direction to expand (-1 if no expansion needed)
                1: expand left (negative x)
                2: expand right (positive x)
                3: expand down (negative y)
                4: expand up (positive y)
        """
        if any(x < self.mapXLim[0]):
            return 1  # Expand left
        elif any(x > self.mapXLim[1]):
            return 2  # Expand right
        elif any(y < self.mapYLim[0]):
            return 3  # Expand down
        elif any(y > self.mapYLim[1]):
            return 4  # Expand up
        else:
            return -1  # No expansion needed

    def checkAndExapndOG(self, x, y):
        """
        Check if points are inside grid and expand if necessary.
        
        Args:
            x (array): X coordinates to check
            y (array): Y coordinates to check
        """
        expand_direction = self.checkMapToExpand(x, y)
        while expand_direction != -1:
            self.expandOccupancyGrid(expand_direction)
            expand_direction = self.checkMapToExpand(x, y)

    def updateOccupancyGrid(self, reading, dTheta=0, update=True):
        """
        Update occupancy grid based on new sensor reading.
        
        Args:
            reading (dict): Sensor reading containing position and range data
            dTheta (float): Additional rotation to apply (default: 0)
            update (bool): Whether to update grid or just return points (default: True)
            
        Returns:
            If update is False, returns tuple of arrays:
            (empty_x, empty_y, occupied_x, occupied_y) representing coordinates
            of empty and occupied cells
        """
        # Extract reading data
        x, y = reading['x'], reading['y']
        theta = reading['theta'] + dTheta
        r_measure = np.asarray(reading['range'])
        
        # Calculate spoke offset based on rotation
        spokes_offset_idx = int(np.rint(theta / (2 * np.pi) * self.numSpokes))
        
        # Initialize lists for storing points
        empty_x_list, empty_y_list = [], []
        occupied_x_list, occupied_y_list = [], []
        
        # Process each LIDAR sample
        for sample_idx in range(self.numSamplesPerRev):
            # Calculate current spoke index
            spoke_idx = int(np.rint((self.spokesStartIdx + spokes_offset_idx + sample_idx) 
                                  % self.numSpokes))
            
            # Get ray coordinates for current spoke
            x_at_spoke = self.radByX[spoke_idx]
            y_at_spoke = self.radByY[spoke_idx]
            r_at_spoke = self.radByR[spoke_idx]
            
            # Find empty and occupied points along ray
            if r_measure[sample_idx] < self.lidarMaxRange:
                # Points before measurement are empty
                empty_idx = np.argwhere(r_at_spoke < r_measure[sample_idx] - 
                                      self.wallThickness / 2)
            else:
                empty_idx = []
            
            # Points at measurement distance are occupied
            occupied_idx = np.argwhere(
                (r_at_spoke > r_measure[sample_idx] - self.wallThickness / 2) & 
                (r_at_spoke < r_measure[sample_idx] + self.wallThickness / 2)
            )
            
            # Convert to grid coordinates
            empty_x, empty_y = self.convertRealXYToMapIdx(
                x + x_at_spoke[empty_idx], 
                y + y_at_spoke[empty_idx]
            )
            occupied_x, occupied_y = self.convertRealXYToMapIdx(
                x + x_at_spoke[occupied_idx],
                y + y_at_spoke[occupied_idx]
            )
            
            if update:
                # Expand grid if necessary and update occupancy counts
                self.checkAndExapndOG(x + x_at_spoke[occupied_idx],
                                    y + y_at_spoke[occupied_idx])
                
                if len(empty_idx) != 0:
                    self.occupancyGridTotal[empty_y, empty_x] += 1
                
                if len(occupied_idx) != 0:
                    self.occupancyGridVisited[occupied_y, occupied_x] += 2
                    self.occupancyGridTotal[occupied_y, occupied_x] += 2
            else:
                # Store points for return
                empty_x_list.extend(x + x_at_spoke[empty_idx])
                empty_y_list.extend(y + y_at_spoke[empty_idx])
                occupied_x_list.extend(x + x_at_spoke[occupied_idx])
                occupied_y_list.extend(y + y_at_spoke[occupied_idx])
        
        if not update:
            return (np.asarray(empty_x_list), np.asarray(empty_y_list),
                   np.asarray(occupied_x_list), np.asarray(occupied_y_list))

    def plotOccupancyGrid(self, x_range=None, y_range=None, plotThreshold=True):
        """
        Visualize the occupancy grid.
        
        Args:
            x_range (list): [min_x, max_x] range to plot (default: None, uses full range)
            y_range (list): [min_y, max_y] range to plot (default: None, uses full range)
            plotThreshold (bool): Whether to also plot thresholded version (default: True)
        """
        # Use full range if not specified
        if (x_range is None or 
            x_range[0] < self.mapXLim[0] or 
            x_range[1] > self.mapXLim[1]):
            x_range = self.mapXLim
            
        if (y_range is None or 
            y_range[0] < self.mapYLim[0] or 
            y_range[1] > self.mapYLim[1]):
            y_range = self.mapYLim
        
        # Calculate occupancy probabilities
        occupancy_map = self.occupancyGridVisited / self.occupancyGridTotal
        
        # Get grid indices for specified range
        x_idx, y_idx = self.convertRealXYToMapIdx(x_range, y_range)
        
        # Extract and process relevant portion of map
        occupancy_map = occupancy_map[y_idx[0]:y_idx[1], x_idx[0]:x_idx[1]]
        occupancy_map = np.flipud(1 - occupancy_map)  # Flip for visualization
        
        # Plot probability map
        plt.imshow(occupancy_map, cmap='gray', 
                  extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
        plt.show()
        
        if plotThreshold:
            # Plot thresholded version
            binary_map = occupancy_map >= 0.5
            plt.matshow(binary_map, cmap='gray',
                       extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
            plt.show()

def updateTrajectoryPlot(matched_reading, x_trajectory, y_trajectory, colors, count):
    """
    Update trajectory visualization with new position.
    
    Args:
        matched_reading (dict): Current sensor reading with position
        x_trajectory (list): List of x coordinates
        y_trajectory (list): List of y coordinates
        colors: Iterator of colors for visualization
        count (int): Current step count
    """
    x, y = matched_reading['x'], matched_reading['y']
    x_trajectory.append(x)
    y_trajectory.append(y)
    
    if count % 1 == 0:  # Plot every step
        plt.scatter(x, y, color=next(colors), s=35)

def main():
    """
    Main function to demonstrate occupancy grid mapping.
    """
    # Initialize parameters (in meters)
    init_map_x_length = 10
    init_map_y_length = 10
    unit_grid_size = 0.02
    lidar_fov = np.pi
    lidar_max_range = 10
    wall_thickness = 7 * unit_grid_size
    
    # Load sensor data
    json_file = "../DataSet/DataPreprocessed/intel-gfs"
    with open(json_file, 'r') as f:
        input_data = json.load(f)
        sensor_data = input_data['map']
    
    # Get number of samples per revolution from first reading
    num_samples_per_rev = len(sensor_data[list(sensor_data)[0]]['range'])
    
    # Get initial position
    init_xy = sensor_data[sorted(sensor_data.keys())[0]]
    
    # Initialize occupancy grid
    og = OccupancyGrid(init_map_x_length, init_map_y_length, init_xy,
                       unit_grid_size, lidar_fov, num_samples_per_rev,
                       lidar_max_range, wall_thickness)
    
    # Process all readings
    count = 0
    plt.figure(figsize=(19.20, 19.20))
    x_trajectory, y_trajectory = [], []
    colors = iter(cm.rainbow(np.linspace(1, 0, len(sensor_data) + 1)))
    
    for key in sorted(sensor_data.keys()):
        count += 1
        og.updateOccupancyGrid(sensor_data[key])
        updateTrajectoryPlot(sensor_data[key], x_trajectory, y_trajectory, 
                           colors, count)
    
    # Highlight start and end points
    plt.scatter(x_trajectory[0], y_trajectory[0], color='r', s=500)
    plt.scatter(x_trajectory[-1], y_trajectory[-1], color=next(colors), s=500)
    plt.plot(x_trajectory, y_trajectory)
    og.plotOccupancyGrid([-12, 20], [-23.5, 7])
    # og.plotOccupancyGrid()

if __name__ == '__main__':
    main()