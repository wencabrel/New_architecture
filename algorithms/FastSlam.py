import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Utils.OccupancyGrid import OccupancyGrid
from Utils.ScanMatcher_OGBased import ScanMatcher
import math
import copy

class ParticleFilter:
    """
    Implements a particle filter for FastSLAM, maintaining and updating multiple hypotheses (particles)
    of robot position and map state.
    """
    def __init__(self, num_particles, occupancy_grid_params, scan_matcher_params):
        """
        Initialize particle filter with specified number of particles and parameters.
        
        Args:
            num_particles: Number of particles to maintain in the filter
            occupancy_grid_params: Parameters for initializing occupancy grid for each particle
            scan_matcher_params: Parameters for scan matching in each particle
        """
        self.num_particles = num_particles
        self.particles = []
        self.initParticles(occupancy_grid_params, scan_matcher_params)
        self.step = 0
        self.prevMatchedReading = None
        self.prevRawReading = None
        self.particlesTrajectory = []

    def initParticles(self, occupancy_grid_params, scan_matcher_params):
        """
        Initialize all particles with identical starting conditions.
        Each particle gets its own occupancy grid and scan matcher.
        """
        for _ in range(self.num_particles):
            new_particle = Particle(occupancy_grid_params, scan_matcher_params)
            self.particles.append(new_particle)

    def updateParticles(self, sensor_reading, time_step):
        """
        Update all particles with new sensor reading.
        
        Args:
            sensor_reading: Current sensor measurement
            time_step: Current time step in the SLAM process
        """
        for particle in self.particles:
            particle.update(sensor_reading, time_step)

    def weightUnbalanced(self):
        """
        Check if particle weights are too unbalanced and resampling is needed.
        Uses a threshold based on particle variance to determine if resampling is necessary.
        
        Returns:
            bool: True if resampling is needed, False otherwise
        """
        self.normalizeWeights()
        weight_variance = 0
        expected_weight = 1.0 / self.num_particles
        
        # Calculate variance of particle weights
        for particle in self.particles:
            weight_variance += (particle.weight - expected_weight) ** 2
        
        # Threshold for resampling: if variance exceeds theoretical maximum balanced variance
        max_balanced_variance = ((self.num_particles - 1) / self.num_particles)**2 + \
                              (self.num_particles - 1.000000000000001) * (expected_weight)**2
        
        return weight_variance > max_balanced_variance

    def normalizeWeights(self):
        """
        Normalize particle weights so they sum to 1.0
        """
        total_weight = sum(particle.weight for particle in self.particles)
        for particle in self.particles:
            particle.weight = particle.weight / total_weight

    def resample(self):
        """
        Resample particles based on their weights using importance sampling.
        Particles with higher weights are more likely to be selected.
        """
        weights = np.array([particle.weight for particle in self.particles])
        temp_particles = [copy.deepcopy(particle) for particle in self.particles]
        
        # Select new particle set based on weights
        resampled_indices = np.random.choice(
            np.arange(self.num_particles), 
            self.num_particles, 
            p=weights
        )
        
        # Replace particles and reset weights
        for i, idx in enumerate(resampled_indices):
            self.particles[i] = copy.deepcopy(temp_particles[idx])
            self.particles[i].weight = 1.0 / self.num_particles

class Particle:
    """
    Represents a single particle in the FastSLAM particle filter.
    Each particle maintains its own hypothesis of the robot's trajectory and map.
    """
    def __init__(self, occupancy_grid_params, scan_matcher_params):
        """
        Initialize particle with its own occupancy grid and scan matcher.
        
        Args:
            occupancy_grid_params: List of parameters for occupancy grid initialization
            scan_matcher_params: List of parameters for scan matcher initialization
        """
        # Unpack occupancy grid parameters
        (init_map_x_length, init_map_y_length, init_xy, grid_size, 
         lidar_fov, lidar_max_range, samples_per_rev, wall_thickness) = occupancy_grid_params
        
        # Unpack scan matcher parameters
        (search_radius, search_half_rad, sigma_grid, move_sigma, 
         max_deviation, turn_sigma, mismatch_prob, coarse_factor) = scan_matcher_params
        
        # Initialize particle's map and scan matcher
        occupancy_grid = OccupancyGrid(
            init_map_x_length, init_map_y_length, init_xy, grid_size,
            lidar_fov, samples_per_rev, lidar_max_range, wall_thickness
        )
        scan_matcher = ScanMatcher(
            occupancy_grid, search_radius, search_half_rad, sigma_grid,
            move_sigma, max_deviation, turn_sigma, mismatch_prob, coarse_factor
        )
        
        self.og = occupancy_grid
        self.sm = scan_matcher
        self.xTrajectory = []
        self.yTrajectory = []
        self.weight = 1.0  # Initial particle weight
        
        # Movement tracking variables
        self.prevRawMovingTheta = None
        self.prevMatchedMovingTheta = None
        self.prevMatchedReading = None
        self.prevRawReading = None

    def updateEstimatedPose(self, current_raw_reading):
        """
        Estimate new pose based on raw sensor reading and previous state.
        
        Args:
            current_raw_reading: Current raw sensor measurement
            
        Returns:
            tuple: (estimated_reading, estimated_distance, estimated_moving_theta, raw_moving_theta)
        """
        # Calculate estimated orientation
        estimated_theta = (self.prevMatchedReading['theta'] + 
                         current_raw_reading['theta'] - 
                         self.prevRawReading['theta'])
        
        # Create estimated reading with previous position and new orientation
        estimated_reading = {
            'x': self.prevMatchedReading['x'],
            'y': self.prevMatchedReading['y'],
            'theta': estimated_theta,
            'range': current_raw_reading['range']
        }
        
        # Calculate movement vectors
        dx = current_raw_reading['x'] - self.prevRawReading['x']
        dy = current_raw_reading['y'] - self.prevRawReading['y']
        estimated_distance = math.sqrt(dx ** 2 + dy ** 2)
        
        # Calculate movement angles for larger movements
        raw_moving_theta = None
        estimated_moving_theta = None
        
        if estimated_distance > 0.3:  # Only calculate angles for significant movements
            raw_moving_theta = self._calculate_movement_angle(dx, dy, estimated_distance)
            estimated_moving_theta = self._update_movement_angle(raw_moving_theta)
            
        return (estimated_reading, estimated_distance, 
                estimated_moving_theta, raw_moving_theta)

    def _calculate_movement_angle(self, dx, dy, distance):
        """Helper method to calculate movement angle from displacement"""
        if dy > 0:
            return math.acos(dx / distance)
        return -math.acos(dx / distance)

    def _update_movement_angle(self, raw_moving_theta):
        """Helper method to update movement angle based on previous state"""
        if self.prevRawMovingTheta is not None:
            raw_turn = raw_moving_theta - self.prevRawMovingTheta
            return self.prevMatchedMovingTheta + raw_turn
        return None

    def getMovingTheta(self, matched_reading):
        """
        Calculate movement angle from last position to current matched position.
        
        Args:
            matched_reading: Current matched sensor reading
            
        Returns:
            float or None: Movement angle in radians, or None if no movement
        """
        if not self.xTrajectory:  # No previous position
            return None
            
        dx = matched_reading['x'] - self.xTrajectory[-1]
        dy = matched_reading['y'] - self.yTrajectory[-1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        
        if distance > 0:
            return self._calculate_movement_angle(dx, dy, distance)
        return None

    def update(self, reading, time_step):
        """
        Update particle state with new sensor reading.
        
        Args:
            reading: Current sensor reading
            time_step: Current time step
        """
        if time_step == 1:
            # First reading - initialize particle state
            matched_reading = reading
            confidence = 1
            self.prevRawMovingTheta = None
            self.prevMatchedMovingTheta = None
        else:
            # Process subsequent readings
            (estimated_reading, estimated_distance,
             estimated_moving_theta, raw_moving_theta) = self.updateEstimatedPose(reading)
            
            # Perform scan matching
            matched_reading, confidence = self.sm.matchScan(
                estimated_reading, estimated_distance,
                estimated_moving_theta, time_step, matchMax=False
            )
            
            # Update movement angles
            self.prevRawMovingTheta = raw_moving_theta
            self.prevMatchedMovingTheta = self.getMovingTheta(matched_reading)
        
        # Update particle state
        self.updateTrajectory(matched_reading)
        self.og.updateOccupancyGrid(matched_reading)
        self.prevMatchedReading = matched_reading
        self.prevRawReading = reading
        self.weight *= confidence

    def updateTrajectory(self, matched_reading):
        """
        Add new position to particle's trajectory history.
        
        Args:
            matched_reading: Current matched sensor reading
        """
        self.xTrajectory.append(matched_reading['x'])
        self.yTrajectory.append(matched_reading['y'])

    def plotParticle(self):
        """
        Visualize particle's trajectory and occupancy grid.
        """
        plt.figure(figsize=(19.20, 19.20))
        
        # Plot trajectory points with rainbow colors
        colors = iter(cm.rainbow(np.linspace(1, 0, len(self.xTrajectory) + 1)))
        
        # Highlight start point
        plt.scatter(self.xTrajectory[0], self.yTrajectory[0], 
                   color='r', s=500, label='Start')
        
        # Plot trajectory points
        for i in range(len(self.xTrajectory)):
            plt.scatter(self.xTrajectory[i], self.yTrajectory[i], 
                       color=next(colors), s=35)
        
        # Highlight end point
        plt.scatter(self.xTrajectory[-1], self.yTrajectory[-1], 
                   color=next(colors), s=500, label='End')
        
        # Plot trajectory line
        plt.plot(self.xTrajectory, self.yTrajectory)
        
        # Plot occupancy grid
        self.og.plotOccupancyGrid([-13, 20], [-25, 7], plotThreshold=False)

def processSensorData(particle_filter, sensor_data, plot_trajectory=True):
    """
    Process sensor data through particle filter and visualize results.
    
    Args:
        particle_filter: Initialized particle filter
        sensor_data: Dictionary of sensor readings
        plot_trajectory: Whether to plot trajectory (default: True)
    """
    time_step = 0
    plt.figure(figsize=(19.20, 19.20))
    
    # Process each sensor reading
    for key in sorted(sensor_data.keys()):
        time_step += 1
        print(f"Processing time step {time_step}")
        
        # Update particle filter
        particle_filter.updateParticles(sensor_data[key], time_step)
        
        # Resample if weights are unbalanced
        if particle_filter.weightUnbalanced():
            particle_filter.resample()
            print("Resampling particles")
        
        # Create visualization frame
        plt.figure(figsize=(19.20, 19.20))
        
        # Find and plot best particle
        best_particle = max(particle_filter.particles, 
                          key=lambda p: p.weight)
        plt.plot(best_particle.xTrajectory, best_particle.yTrajectory)
        
        # Plot occupancy grid
        x_range, y_range = [-13, 20], [-25, 7]
        occupancy_map = best_particle.og.occupancyGridVisited / best_particle.og.occupancyGridTotal
        x_idx, y_idx = best_particle.og.convertRealXYToMapIdx(x_range, y_range)
        
        # Process map for visualization
        occupancy_map = occupancy_map[y_idx[0]: y_idx[1], x_idx[0]: x_idx[1]]
        occupancy_map = np.flipud(1 - occupancy_map)
        
        # Display and save visualization
        plt.imshow(occupancy_map, cmap='gray', 
                  extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
        plt.savefig(f'../Output/{str(time_step).zfill(3)}.png')
        plt.close()
    
    # Final visualization of all particles
    for particle in particle_filter.particles:
        particle.plotParticle()
    
    # Plot best particle's final state
    best_particle = max(particle_filter.particles, 
                       key=lambda p: p.weight)
    best_particle.plotParticle()

def readJson(json_file_path):
    """
    Read sensor data from JSON file.
    
    Args:
        json_file_path: Path to JSON file
        
    Returns:
        dict: Sensor data from file
    """
    with open(json_file_path, 'r') as f:
        return json.load(f)['map']

def main():
    """
    Main function to run FastSLAM algorithm.
    Sets up parameters, initializes particle filter, and processes sensor data.
    """
    # Map and sensor parameters
    init_map_x_length = 50  # meters
    init_map_y_length = 50  # meters
    grid_size = 0.02  # meters per grid cell
    lidar_fov = np.pi  # field of view in radians
    lidar_max_range = 12  # meters
    wall_thickness = 5 * grid_size
    
    # Scan matching parameters
    # Scan matching parameters
    search_radius = 1.4  # meters
    search_half_rad = 0.25  # radians
    sigma_grid = 2  # grid cells
    move_sigma = 0.1  # meters
    max_deviation = 0.25  # meters
    turn_sigma = 0.3  # radians
    mismatch_prob = 0.15  # probability threshold for mismatches
    coarse_factor = 5  # factor for coarse-to-fine scan matching
    
    # Load sensor data from file
    sensor_data = readJson("../DataSet/DataPreprocessed/zjnu20_21_20-gfs")
    
    # Get number of samples per revolution from first reading
    samples_per_rev = len(sensor_data[list(sensor_data)[0]]['range'])
    
    # Get initial position from first reading
    init_xy = sensor_data[sorted(sensor_data.keys())[0]]
    
    # Number of particles for the filter
    num_particles = 10
    
    # Package parameters for initialization
    occupancy_grid_params = [
        init_map_x_length, init_map_y_length, init_xy, grid_size,
        lidar_fov, lidar_max_range, samples_per_rev, wall_thickness
    ]
    
    scan_matcher_params = [
        search_radius, search_half_rad, sigma_grid, move_sigma,
        max_deviation, turn_sigma, mismatch_prob, coarse_factor
    ]
    
    # Initialize particle filter
    particle_filter = ParticleFilter(num_particles, occupancy_grid_params, scan_matcher_params)
    
    # Process sensor data through particle filter
    processSensorData(particle_filter, sensor_data, plot_trajectory=True)

if __name__ == '__main__':
    main()