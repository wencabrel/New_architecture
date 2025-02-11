import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Utils.OccupancyGrid import OccupancyGrid
from Utils.ScanMatcher_OGBased import ScanMatcher
import json

class EKFSlamAdapter:
    """
    Implements EKF SLAM using existing OccupancyGrid for mapping and 
    ScanMatcher for measurement updates.
    """
    def __init__(self, og_params, sm_params):
        """
        Initialize EKF SLAM with occupancy grid and scan matcher.
        
        Args:
            og_params: Parameters for OccupancyGrid initialization
            sm_params: Parameters for ScanMatcher initialization
        """
        # Initialize map and scan matcher
        self.og = OccupancyGrid(*og_params)
        self.sm = ScanMatcher(self.og, *sm_params)
        
        # Initialize EKF state (x, y, theta)
        self.state_dim = 3
        self.mu = np.zeros((self.state_dim, 1))
        self.sigma = np.eye(self.state_dim) * 0.1
        
        # Store previous readings for pose prediction
        self.prev_matched_reading = None
        self.prev_raw_reading = None
        
        # Motion and measurement noise (can be tuned)
        self.R = np.diag([0.002, 0.002, 0.00003])  # Motion noise
        self.Q = np.diag([0.003, 0.003, 0.00003])  # Measurement noise

    def predict_step(self, current_reading):
        """
        EKF prediction step using odometry-like motion model.
        
        Args:
            current_reading: Current sensor reading
        """
        if self.prev_raw_reading is None:
            # First reading - initialize state
            self.mu[0] = current_reading['x']
            self.mu[1] = current_reading['y']
            self.mu[2] = current_reading['theta']
            return
            
        # Calculate motion from previous reading
        dx = current_reading['x'] - self.prev_raw_reading['x']
        dy = current_reading['y'] - self.prev_raw_reading['y']
        dtheta = current_reading['theta'] - self.prev_raw_reading['theta']
        
        # Ensure valid angle difference
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        
        # Motion model Jacobian
        G = np.eye(self.state_dim)
        G[0, 2] = -dx * np.sin(self.mu[2]) - dy * np.cos(self.mu[2])
        G[1, 2] = dx * np.cos(self.mu[2]) - dy * np.sin(self.mu[2])
        
        # Update state mean
        self.mu[0] += dx * np.cos(self.mu[2]) - dy * np.sin(self.mu[2])
        self.mu[1] += dx * np.sin(self.mu[2]) + dy * np.cos(self.mu[2])
        self.mu[2] = (self.mu[2] + dtheta + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
        
        # Update state covariance
        self.sigma = G @ self.sigma @ G.T + self.R
        
        # Ensure numerical stability
        self.sigma = (self.sigma + self.sigma.T) / 2  # Ensure symmetry
        min_variance = 1e-8
        self.sigma += np.eye(self.state_dim) * min_variance  # Add small variance

    def update_step(self, matched_reading, scan_confidence):
        """
        EKF update step using scan matcher results.
        
        Args:
            matched_reading: Result from scan matcher
            scan_confidence: Confidence score from scan matcher
        """
        try:
            # Ensure valid confidence value
            scan_confidence = max(scan_confidence, 1e-6)
            
            # Construct measurement
            z = np.array([[matched_reading['x']],
                         [matched_reading['y']],
                         [matched_reading['theta']]])
                         
            # Check for invalid measurements
            if not np.all(np.isfinite(z)):
                print("Warning: Invalid measurement detected, skipping update")
                return
                         
            # Measurement model is direct observation of state
            H = np.eye(self.state_dim)
            
            # Kalman gain with robust matrix inversion
            S = H @ self.sigma @ H.T + self.Q / scan_confidence
            S = (S + S.T) / 2  # Ensure symmetry
            
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                print("Warning: Matrix inversion failed, using pseudo-inverse")
                S_inv = np.linalg.pinv(S)
            
            K = self.sigma @ H.T @ S_inv
            
            # Update state mean and covariance
            innovation = z - self.mu
            innovation[2] = (innovation[2] + np.pi) % (2 * np.pi) - np.pi  # Angle wrapping
            
            self.mu = self.mu + K @ innovation
            self.mu[2] = (self.mu[2] + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
            
            # Joseph form of covariance update (more stable)
            I = np.eye(self.state_dim)
            self.sigma = (I - K @ H) @ self.sigma @ (I - K @ H).T + K @ self.Q @ K.T
            
            # Ensure numerical stability
            self.sigma = (self.sigma + self.sigma.T) / 2  # Ensure symmetry
            min_variance = 1e-8
            self.sigma += np.eye(self.state_dim) * min_variance  # Add small variance
            
        except Exception as e:
            print(f"Warning: Update step failed: {str(e)}")
            return

    def process_reading(self, current_reading, count):
        """
        Process a new sensor reading through EKF SLAM.
        """
        try:
            if count == 1:
                self.prev_matched_reading = current_reading
                self.prev_raw_reading = current_reading
                self.og.updateOccupancyGrid(current_reading)
                return current_reading
                
            self.predict_step(current_reading)
            
            if not np.all(np.isfinite(self.mu)):
                print("Warning: Invalid state detected, resetting to current reading")
                self.mu[0] = current_reading['x']
                self.mu[1] = current_reading['y']
                self.mu[2] = current_reading['theta']
                self.sigma = np.eye(self.state_dim) * 0.1
            
            estimated_reading = {
                'x': float(self.mu[0]),
                'y': float(self.mu[1]),
                'theta': float(self.mu[2]),
                'range': current_reading['range']
            }
            
            dx = current_reading['x'] - self.prev_raw_reading['x']
            dy = current_reading['y'] - self.prev_raw_reading['y']
            est_moving_dist = np.sqrt(dx**2 + dy**2)
            
            if est_moving_dist > 0.3:
                est_moving_theta = np.arctan2(dy, dx)
                if not np.isfinite(est_moving_theta):
                    est_moving_theta = None
            else:
                est_moving_theta = None
            
            matched_reading, confidence = self.sm.matchScan(
                estimated_reading, est_moving_dist,
                est_moving_theta, count, matchMax=True
            )
            
            # Check only scalar values for validity
            scalar_values = {k: v for k, v in matched_reading.items() if k != 'range'}
            if not np.all([np.isfinite(val) for val in scalar_values.values()]):
                print("Warning: Invalid scan matching result, using prediction only")
                matched_reading = estimated_reading
                confidence = 0.1
            
            self.update_step(matched_reading, confidence)
            
            final_reading = {
                'x': float(self.mu[0]),
                'y': float(self.mu[1]),
                'theta': float(self.mu[2]),
                'range': current_reading['range']
            }
            
            # Check only scalar values for validity
            scalar_values = {k: v for k, v in final_reading.items() if k != 'range'}
            if np.all([np.isfinite(val) for val in scalar_values.values()]):
                self.og.updateOccupancyGrid(final_reading)
                self.prev_matched_reading = final_reading
                self.prev_raw_reading = current_reading
            else:
                print("Warning: Invalid final reading, using current reading")
                final_reading = current_reading
            
            return final_reading
            
        except Exception as e:
            print(f"Error in process_reading: {str(e)}")
            return current_reading

def processSensorData(ekf_slam, sensor_data):
    """
    Process all sensor data through EKF SLAM.
    
    Args:
        ekf_slam: Initialized EKF SLAM instance
        sensor_data: Dictionary of sensor readings
    """
    count = 0
    plt.figure(figsize=(19.20, 19.20))
    
    # For trajectory visualization
    x_trajectory, y_trajectory = [], []
    colors = iter(cm.rainbow(np.linspace(1, 0, len(sensor_data) + 1)))
    
    # Process each reading
    for key in sorted(sensor_data.keys()):
        count += 1
        print(f"Processing reading {count}")
        
        # Process through EKF SLAM
        final_reading = ekf_slam.process_reading(sensor_data[key], count)
        
        # Update trajectory
        x_trajectory.append(final_reading['x'])
        y_trajectory.append(final_reading['y'])
        
        # Visualize current position
        if count % 1 == 0:
            plt.scatter(x_trajectory[-1], y_trajectory[-1], 
                       color=next(colors), s=35)
    
    # Final visualization
    plt.scatter(x_trajectory[0], y_trajectory[0], color='r', s=500)
    plt.scatter(x_trajectory[-1], y_trajectory[-1], color=next(colors), s=500)
    plt.plot(x_trajectory, y_trajectory)
    
    # Plot final map
    ekf_slam.og.plotOccupancyGrid([-13, 20], [-25, 7], plotThreshold=False)

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
    Main function to run EKF SLAM.
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
    
    # Package parameters
    og_params = [
        init_map_x_length, init_map_y_length, init_xy,
        unit_grid_size, lidar_fov, num_samples_per_rev,
        lidar_max_range, wall_thickness
    ]
    
    sm_params = [
        search_radius, search_half_rad, scan_sigma,
        move_r_sigma, max_move_deviation, turn_sigma,
        mismatch_prob, coarse_factor
    ]
    
    # Initialize and run EKF SLAM
    ekf_slam = EKFSlamAdapter(og_params, sm_params)
    processSensorData(ekf_slam, sensor_data)

if __name__ == '__main__':
    main()