import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Utils.OccupancyGrid import OccupancyGrid
from scipy.ndimage import gaussian_filter
import math

class ScanMatcher:
    """
    Implements scan matching for LIDAR data against an occupancy grid map.
    Uses a multi-resolution approach with coarse-to-fine matching strategy.
    """
    def __init__(self, og, searchRadius, searchHalfRad, scanSigmaInNumGrid, 
                 moveRSigma, maxMoveDeviation, turnSigma, missMatchProbAtCoarse, coarseFactor):
        """
        Initialize scan matcher with given parameters.
        
        Args:
            og (OccupancyGrid): Reference occupancy grid map
            searchRadius (float): Maximum search radius in meters
            searchHalfRad (float): Maximum rotation search range in radians
            scanSigmaInNumGrid (float): Standard deviation for scan correlation in grid cells
            moveRSigma (float): Standard deviation for movement in meters
            maxMoveDeviation (float): Maximum allowed movement deviation in meters
            turnSigma (float): Standard deviation for rotation in radians
            missMatchProbAtCoarse (float): Probability threshold for mismatches in coarse search
            coarseFactor (int): Factor for coarse-to-fine resolution reduction
        """
        self.searchRadius = searchRadius
        self.searchHalfRad = searchHalfRad
        self.og = og
        self.scanSigmaInNumGrid = scanSigmaInNumGrid
        self.coarseFactor = coarseFactor
        self.moveRSigma = moveRSigma
        self.turnSigma = turnSigma
        self.missMatchProbAtCoarse = missMatchProbAtCoarse
        self.maxMoveDeviation = maxMoveDeviation

    def frameSearchSpace(self, estimated_x, estimated_y, unit_length, sigma, mismatch_prob):
        """
        Create and initialize the search space for scan matching.
        
        Args:
            estimated_x (float): Estimated x position
            estimated_y (float): Estimated y position
            unit_length (float): Grid cell size for search space
            sigma (float): Standard deviation for Gaussian blurring
            mismatch_prob (float): Base probability for mismatches
            
        Returns:
            tuple: (x_range_list, y_range_list, probability_search_space)
                  Search space boundaries and probability distribution
        """
        # Calculate maximum scan radius including search area
        max_scan_radius = 1.1 * self.og.lidarMaxRange + self.searchRadius
        
        # Define search space boundaries
        x_range = [estimated_x - max_scan_radius, estimated_x + max_scan_radius]
        y_range = [estimated_y - max_scan_radius, estimated_y + max_scan_radius]
        
        # Calculate grid dimensions
        idx_end_x = int((x_range[1] - x_range[0]) / unit_length)
        idx_end_y = int((y_range[1] - y_range[0]) / unit_length)
        
        # Initialize search space with mismatch probability
        search_space = math.log(mismatch_prob) * np.ones((idx_end_y + 1, idx_end_x + 1))

        # Ensure occupancy grid covers search space
        self.og.checkAndExapndOG(x_range, y_range)
        
        # Convert range to grid indices
        x_range_idx, y_range_idx = self.og.convertRealXYToMapIdx(x_range, y_range)
        
        # Extract and process relevant portion of occupancy grid
        og_map = self.og.occupancyGridVisited[y_range_idx[0]: y_range_idx[1], 
                                             x_range_idx[0]: x_range_idx[1]] / \
                self.og.occupancyGridTotal[y_range_idx[0]: y_range_idx[1], 
                                         x_range_idx[0]: x_range_idx[1]]
        og_map = og_map > 0.5  # Convert to binary occupancy
        
        # Get coordinates for occupied cells
        og_x = self.og.OccupancyGridX[y_range_idx[0]: y_range_idx[1], 
                                     x_range_idx[0]: x_range_idx[1]]
        og_y = self.og.OccupancyGridY[y_range_idx[0]: y_range_idx[1], 
                                     x_range_idx[0]: x_range_idx[1]]
        og_x, og_y = og_x[og_map], og_y[og_map]
        
        # Convert occupied cells to search space indices
        og_idx = self.convertXYToSearchSpaceIdx(og_x, og_y, x_range[0], y_range[0], 
                                               unit_length)
        
        # Mark occupied cells in search space
        search_space[og_idx[1], og_idx[0]] = 0
        
        # Generate probability distribution
        prob_search_space = self.generateProbSearchSpace(search_space, sigma)
        
        return x_range, y_range, prob_search_space

    def generateProbSearchSpace(self, search_space, sigma):
        """
        Generate probability distribution for search space using Gaussian filtering.
        
        Args:
            search_space (numpy.ndarray): Initial search space
            sigma (float): Standard deviation for Gaussian filter
            
        Returns:
            numpy.ndarray: Probability distribution over search space
        """
        # Apply Gaussian filter
        prob_space = gaussian_filter(search_space, sigma=sigma)
        
        # Threshold probabilities
        prob_min = prob_space.min()
        prob_space[prob_space > 0.5 * prob_min] = 0
        
        return prob_space

    def matchScan(self, reading, est_moving_dist, est_moving_theta, count, matchMax=True):
        """
        Match current scan against occupancy grid using two-stage approach.
        
        Args:
            reading (dict): Current sensor reading
            est_moving_dist (float): Estimated movement distance
            est_moving_theta (float): Estimated movement angle
            count (int): Current time step
            matchMax (bool): Whether to use maximum likelihood (True) or sampling (False)
            
        Returns:
            tuple: (matched_reading, confidence)
                  Best match reading and confidence score
        """
        # Extract reading data
        estimated_x = reading['x']
        estimated_y = reading['y']
        estimated_theta = reading['theta']
        r_measure = np.asarray(reading['range'])
        
        if count == 1:
            return reading, 1
            
        # Coarse Search Stage
        coarse_search_step = self.coarseFactor * self.og.unitGridSize
        coarse_sigma = self.scanSigmaInNumGrid / self.coarseFactor
        
        x_range, y_range, prob_space = self.frameSearchSpace(
            estimated_x, estimated_y, coarse_search_step,
            coarse_sigma, self.missMatchProbAtCoarse
        )
        
        matched_px, matched_py, matched_reading, conv_total, coarse_confidence = \
            self.searchToMatch(
                prob_space, estimated_x, estimated_y, estimated_theta, r_measure,
                x_range, y_range, self.searchRadius, self.searchHalfRad,
                coarse_search_step, est_moving_dist, est_moving_theta,
                fineSearch=False, matchMax=matchMax
            )
            
        # Fine Search Stage
        fine_search_step = self.og.unitGridSize
        fine_sigma = self.scanSigmaInNumGrid
        fine_search_half_rad = self.searchHalfRad
        fine_mismatch_prob = self.missMatchProbAtCoarse ** (2 / self.coarseFactor)
        
        x_range, y_range, prob_space = self.frameSearchSpace(
            matched_reading['x'], matched_reading['y'],
            fine_search_step, fine_sigma, fine_mismatch_prob
        )
        
        matched_px, matched_py, matched_reading, conv_total, fine_confidence = \
            self.searchToMatch(
                prob_space, matched_reading['x'], matched_reading['y'],
                matched_reading['theta'], matched_reading['range'],
                x_range, y_range, coarse_search_step, fine_search_half_rad,
                fine_search_step, est_moving_dist, est_moving_theta,
                fineSearch=True, matchMax=True
            )
            
        return matched_reading, coarse_confidence

    def covertMeasureToXY(self, estimated_x, estimated_y, estimated_theta, r_measure):
        """
        Convert polar LIDAR measurements to Cartesian coordinates.
        
        Args:
            estimated_x (float): Estimated x position
            estimated_y (float): Estimated y position
            estimated_theta (float): Estimated orientation
            r_measure (numpy.ndarray): Range measurements
            
        Returns:
            tuple: (px, py) Arrays of x and y coordinates
        """
        # Calculate measurement angles
        angles = np.linspace(
            estimated_theta - self.og.lidarFOV / 2,
            estimated_theta + self.og.lidarFOV / 2,
            num=self.og.numSamplesPerRev
        )
        
        # Filter valid measurements
        valid_idx = r_measure < self.og.lidarMaxRange
        r_measure_valid = r_measure[valid_idx]
        angles = angles[valid_idx]
        
        # Convert to Cartesian coordinates
        px = estimated_x + np.cos(angles) * r_measure_valid
        py = estimated_y + np.sin(angles) * r_measure_valid
        
        return px, py

    def searchToMatch(self, prob_space, estimated_x, estimated_y, estimated_theta,
                     r_measure, x_range, y_range, search_radius, search_half_rad,
                     unit_length, est_moving_dist, est_moving_theta,
                     fineSearch=False, matchMax=True):
        """
        Search for best match by trying different positions and orientations.
        
        Args:
            prob_space (numpy.ndarray): Probability distribution over search space
            estimated_x, estimated_y (float): Estimated position
            estimated_theta (float): Estimated orientation
            r_measure (numpy.ndarray): Range measurements
            x_range, y_range (list): Search space boundaries
            search_radius (float): Maximum search radius
            search_half_rad (float): Maximum rotation search range
            unit_length (float): Grid cell size
            est_moving_dist (float): Estimated movement distance
            est_moving_theta (float): Estimated movement angle
            fineSearch (bool): Whether this is fine search stage
            matchMax (bool): Whether to use maximum likelihood or sampling
            
        Returns:
            tuple: (matched_px, matched_py, matched_reading, conv_total, confidence)
                  Matched points, reading, correlation results, and confidence
        """
        # Convert measurements to Cartesian coordinates
        px, py = self.covertMeasureToXY(estimated_x, estimated_y, estimated_theta, r_measure)
        
        # Calculate search grid
        num_cells_radius = int(search_radius / unit_length)
        x_moving_range = np.arange(-num_cells_radius, num_cells_radius + 1)
        y_moving_range = np.arange(-num_cells_radius, num_cells_radius + 1)
        x_grid, y_grid = np.meshgrid(x_moving_range, y_moving_range)
        
        # Initialize correlation weights
        if fineSearch:
            range_weight = np.zeros(x_grid.shape)
            theta_weight = np.zeros(x_grid.shape)
        else:
            # Calculate range correlation weight
            range_weight = -(1 / (2 * self.moveRSigma ** 2)) * \
                         (np.sqrt((x_grid * unit_length) ** 2 + 
                                (y_grid * unit_length) ** 2) - est_moving_dist) ** 2
            
            # Apply maximum deviation constraint
            range_deviation = np.abs(np.sqrt((x_grid * unit_length) ** 2 + 
                                          (y_grid * unit_length) ** 2) - est_moving_dist)
            range_weight[range_deviation > self.maxMoveDeviation] = -100
            
            # Calculate rotation correlation weight
            if est_moving_theta is not None:
                dist_grid = np.sqrt(np.square(x_grid) + np.square(y_grid))
                dist_grid[dist_grid == 0] = 0.0001
                theta_grid = np.arccos((x_grid * math.cos(est_moving_theta) + 
                                      y_grid * math.sin(est_moving_theta)) / dist_grid)
                theta_weight = -1 / (2 * self.turnSigma ** 2) * np.square(theta_grid)
            else:
                theta_weight = np.zeros(x_grid.shape)
        
        # Reshape search grids for broadcasting
        x_grid = x_grid.reshape((x_grid.shape[0], x_grid.shape[1], 1))
        y_grid = y_grid.reshape((y_grid.shape[0], y_grid.shape[1], 1))
        
        # Define rotation search range
        theta_range = np.arange(-search_half_rad, 
                              search_half_rad + self.og.angularStep, 
                              self.og.angularStep)
        
        # Initialize correlation results
        conv_total = np.zeros((len(theta_range), x_grid.shape[0], x_grid.shape[1]))
        
        # Search over all rotations
        for i, theta in enumerate(theta_range):
            # Rotate scan points
            rotated_px, rotated_py = self.rotate((estimated_x, estimated_y), 
                                               (px, py), theta)
            
            # Convert to search space indices
            rotated_px_idx, rotated_py_idx = self.convertXYToSearchSpaceIdx(
                rotated_px, rotated_py, x_range[0], y_range[0], unit_length
            )
            
            # Remove duplicate points
            unique_points = np.unique(np.column_stack((rotated_px_idx, rotated_py_idx)), 
                                   axis=0)
            rotated_px_idx, rotated_py_idx = unique_points[:, 0], unique_points[:, 1]
            
            # Reshape for broadcasting
            rotated_px_idx = rotated_px_idx.reshape(1, 1, -1)
            rotated_py_idx = rotated_py_idx.reshape(1, 1, -1)
            
            # Apply translation search
            translated_px_idx = rotated_px_idx + x_grid
            translated_py_idx = rotated_py_idx + y_grid
            
            # Calculate correlation
            conv_result = prob_space[translated_py_idx, translated_px_idx]
            conv_sum = np.sum(conv_result, axis=2)
            
            # Combine correlation with motion weights
            conv_sum = conv_sum + range_weight + theta_weight
            conv_total[i, :, :] = conv_sum

        # Select best match based on strategy
        if matchMax:
            # Maximum likelihood approach
            max_idx = np.unravel_index(conv_total.argmax(), conv_total.shape)
        else:
            # Probabilistic sampling approach
            conv_total_flat = np.reshape(conv_total, -1)
            conv_total_prob = np.exp(conv_total_flat) / np.exp(conv_total_flat).sum()
            max_idx = np.random.choice(np.arange(conv_total_flat.size), 1, 
                                     p=conv_total_prob)[0]
            max_idx = np.unravel_index(max_idx, conv_total.shape)

        # Calculate confidence score
        confidence = np.sum(np.exp(conv_total))

        # Convert best match to transformations
        dx = x_moving_range[max_idx[2]] * unit_length
        dy = y_moving_range[max_idx[1]] * unit_length
        dtheta = theta_range[max_idx[0]]

        # Create matched reading
        matched_reading = {
            "x": estimated_x + dx,
            "y": estimated_y + dy,
            "theta": estimated_theta + dtheta,
            "range": r_measure
        }

        # Transform scan points to matched position
        matched_px, matched_py = self.rotate((estimated_x, estimated_y), (px, py), dtheta)
        matched_px = matched_px + dx
        matched_py = matched_py + dy

        return matched_px, matched_py, matched_reading, conv_total, confidence

    def plotMatchOverlay(self, prob_space, matched_px, matched_py, matched_reading, 
                        x_range, y_range, unit_length):
        """
        Visualize scan matching results overlaid on probability space.
        
        Args:
            prob_space (numpy.ndarray): Probability distribution over search space
            matched_px, matched_py (numpy.ndarray): Matched scan points
            matched_reading (dict): Matched sensor reading
            x_range, y_range (list): Search space boundaries
            unit_length (float): Grid cell size
        """
        plt.figure(figsize=(19.20, 19.20))
        
        # Plot probability space
        plt.imshow(prob_space, origin='lower')
        
        # Convert and plot matched scan points
        px_idx, py_idx = self.convertXYToSearchSpaceIdx(
            matched_px, matched_py, x_range[0], y_range[0], unit_length
        )
        plt.scatter(px_idx, py_idx, c='r', s=5)
        
        # Convert and plot matched position
        pose_x_idx, pose_y_idx = self.convertXYToSearchSpaceIdx(
            matched_reading['x'], matched_reading['y'],
            x_range[0], y_range[0], unit_length
        )
        plt.scatter(pose_x_idx, pose_y_idx, color='blue', s=50)
        
        plt.show()

    def rotate(self, origin, point, angle):
        """
        Rotate point(s) around origin by given angle.
        
        Args:
            origin (tuple): (x, y) coordinates of rotation center
            point (tuple): (x, y) arrays of points to rotate
            angle (float): Rotation angle in radians (counterclockwise)
            
        Returns:
            tuple: (rotated_x, rotated_y) Rotated point coordinates
        """
        ox, oy = origin
        px, py = point
        
        # Apply rotation matrix
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        
        return qx, qy

    def convertXYToSearchSpaceIdx(self, px, py, begin_x, begin_y, unit_length):
        """
        Convert real-world coordinates to search space indices.
        
        Args:
            px, py (numpy.ndarray): Point coordinates
            begin_x, begin_y (float): Search space origin
            unit_length (float): Grid cell size
            
        Returns:
            tuple: (x_idx, y_idx) Grid indices
        """
        x_idx = ((px - begin_x) / unit_length).astype(int)
        y_idx = ((py - begin_y) / unit_length).astype(int)
        return x_idx, y_idx

def updateEstimatedPose(current_raw_reading, prev_matched_reading, prev_raw_reading, 
                       prev_raw_moving_theta, prev_matched_moving_theta):
    """
    Update estimated pose based on previous readings and movements.
    
    Args:
        current_raw_reading (dict): Current raw sensor reading
        prev_matched_reading (dict): Previous matched reading
        prev_raw_reading (dict): Previous raw reading
        prev_raw_moving_theta (float): Previous raw movement angle
        prev_matched_moving_theta (float): Previous matched movement angle
        
    Returns:
        tuple: (estimated_reading, est_moving_dist, est_moving_theta, raw_moving_theta)
    """
    # Calculate estimated orientation
    estimated_theta = (prev_matched_reading['theta'] + 
                      current_raw_reading['theta'] - 
                      prev_raw_reading['theta'])
    
    # Create estimated reading
    estimated_reading = {
        'x': prev_matched_reading['x'],
        'y': prev_matched_reading['y'],
        'theta': estimated_theta,
        'range': current_raw_reading['range']
    }
    
    # Calculate movement
    dx = current_raw_reading['x'] - prev_raw_reading['x']
    dy = current_raw_reading['y'] - prev_raw_reading['y']
    est_moving_dist = math.sqrt(dx**2 + dy**2)
    
    # Calculate movement angles for significant movements
    raw_moving_theta = None
    est_moving_theta = None
    
    if est_moving_dist > 0.3:
        raw_moving_theta = math.atan2(dy, dx)
        
        if prev_raw_moving_theta is not None:
            raw_turn_theta = raw_moving_theta - prev_raw_moving_theta
            est_moving_theta = prev_matched_moving_theta + raw_turn_theta
    
    return estimated_reading, est_moving_dist, est_moving_theta, raw_moving_theta

def updateTrajectory(matched_reading, x_trajectory, y_trajectory):
    """
    Update trajectory with new position.
    
    Args:
        matched_reading (dict): Current matched sensor reading
        x_trajectory (list): List of x coordinates
        y_trajectory (list): List of y coordinates
    """
    x_trajectory.append(matched_reading['x'])
    y_trajectory.append(matched_reading['y'])

def getMovingTheta(matched_reading, x_trajectory, y_trajectory):
    """
    Calculate movement angle from previous position.
    
    Args:
        matched_reading (dict): Current matched reading
        x_trajectory (list): List of x coordinates
        y_trajectory (list): List of y coordinates
        
    Returns:
        float: Movement angle in radians, or None if no movement
    """
    prev_x, prev_y = x_trajectory[-1], y_trajectory[-1]
    dx = matched_reading['x'] - prev_x
    dy = matched_reading['y'] - prev_y
    
    move_dist = math.sqrt(dx ** 2 + dy ** 2)
    if move_dist > 0:
        return math.atan2(dy, dx)
    return None

def processSensorData(sensor_data, og, sm):
    """
    Process sensor data through scan matcher.
    
    Args:
        sensor_data (dict): Dictionary of sensor readings
        og (OccupancyGrid): Occupancy grid map
        sm (ScanMatcher): Scan matcher instance
    """
    count = 0
    plt.figure(figsize=(19.20, 19.20))
    
    # Setup for trajectory visualization
    colors = iter(cm.rainbow(np.linspace(1, 0, len(sensor_data) + 1)))
    x_trajectory, y_trajectory = [], []
    
    for key in sorted(sensor_data.keys()):
        count += 1
        print(f"Processing reading {count}")
        
        if count == 1:
            # Initialize with first reading
            prev_raw_moving_theta = None
            prev_matched_moving_theta = None
            matched_reading = sensor_data[key]
            confidence = 1
        else:
            # Process subsequent readings
            current_raw_reading = sensor_data[key]
            
            # Update pose estimation
            estimated_reading, est_moving_dist, est_moving_theta, raw_moving_theta = \
                updateEstimatedPose(
                    current_raw_reading, prev_matched_reading,
                    prev_raw_reading, prev_raw_moving_theta,
                    prev_matched_moving_theta
                )
            
            # Perform scan matching
            matched_reading, confidence = sm.matchScan(
                estimated_reading, est_moving_dist,
                est_moving_theta, count
            )
            
            # Update movement tracking
            prev_raw_moving_theta = raw_moving_theta
            prev_matched_moving_theta = getMovingTheta(
                matched_reading, x_trajectory, y_trajectory
            )
        
        # Update map and trajectory
        og.updateOccupancyGrid(matched_reading)
        updateTrajectory(matched_reading, x_trajectory, y_trajectory)
        
        if count % 1 == 0:
            plt.scatter(x_trajectory[-1], y_trajectory[-1], 
                       color=next(colors), s=35)
        
        # Store readings for next iteration
        prev_matched_reading = matched_reading
        prev_raw_reading = sensor_data[key]
    
    # Plot final trajectory
    plt.scatter(x_trajectory[0], y_trajectory[0], color='r', s=500)
    plt.scatter(x_trajectory[-1], y_trajectory[-1], color=next(colors), s=500)
    plt.plot(x_trajectory, y_trajectory)
    
    # Plot final map
    og.plotOccupancyGrid([-13, 20], [-25, 7], plotThreshold=False)

def readJson(json_file):
    """
    Read sensor data from JSON file.
    
    Args:
        json_file (str): Path to JSON file
        
    Returns:
        dict: Sensor data from file
    """
    with open(json_file, 'r') as f:
        return json.load(f)['map']

def main():
    """
    Main function to demonstrate scan matching.
    Sets up parameters and processes sensor data.
    """
    # Initialize parameters
    init_map_x_length = 10  # meters
    init_map_y_length = 10  # meters
    unit_grid_size = 0.02  # meters
    lidar_fov = np.pi  # radians
    lidar_max_range = 10  # meters
    wall_thickness = 5 * unit_grid_size  # meters
    
    # Scan matching parameters
    search_radius = 1.4  # meters
    search_half_rad = 0.25  # radians
    scan_sigma = 2  # grid cells
    move_r_sigma = 0.1  # meters
    max_move_deviation = 0.25  # meters
    turn_sigma = 0.3  # radians
    mismatch_prob = 0.15
    coarse_factor = 5
    
    # Load sensor data
    sensor_data = readJson("../DataSet/DataPreprocessed/intel-gfs")
    num_samples_per_rev = len(sensor_data[list(sensor_data)[0]]['range'])
    init_xy = sensor_data[sorted(sensor_data.keys())[0]]
    
    # Initialize occupancy grid and scan matcher
    og = OccupancyGrid(
        init_map_x_length, init_map_y_length, init_xy,
        unit_grid_size, lidar_fov, num_samples_per_rev,
        lidar_max_range, wall_thickness
    )
    
    sm = ScanMatcher(
        og, search_radius, search_half_rad, scan_sigma,
        move_r_sigma, max_move_deviation, turn_sigma,
        mismatch_prob, coarse_factor
    )
    
    # Process sensor data
    processSensorData(sensor_data, og, sm)

if __name__ == '__main__':
    main()