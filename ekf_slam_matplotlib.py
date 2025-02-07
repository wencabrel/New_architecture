import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend explicitly
import matplotlib.pyplot as plt
# plt.style.use('seaborn')  # Optional: for better looking plots
from matplotlib.patches import Polygon, Ellipse, Arrow
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re
from scipy.spatial.transform import Rotation
from python_ugv_sim.utils.vehicles import DifferentialDrive
from matplotlib_environment import MatplotlibEnvironment

# <---------------DATASET HANDLING ---------------->
@dataclass
class OdomReading:
    x: float
    y: float
    theta: float
    v: float  # forward velocity
    omega: float  # angular velocity
    timestamp: float

@dataclass
class LaserReading:
    num_readings: int
    ranges: List[float]
    pose: Tuple[float, float, float]  # x, y, theta
    timestamp: float

class CLFDataLoader:
    def __init__(self, data_path: str, max_entries: int = 5000):
        """
        Initialize the CLF data loader
        
        Args:
            data_path (str): Path to the CLF dataset file
            max_entries (int): Maximum number of entries to load
        """
        self.data_path = data_path
        self.odom_readings: List[OdomReading] = []
        self.laser_readings: List[LaserReading] = []
        self._load_data(max_entries)
    
    def _parse_odom_line(self, parts: List[str]) -> OdomReading:
        """Parse an ODOM line from the CLF file"""
        return OdomReading(
            x=float(parts[1]),
            y=float(parts[2]),
            theta=float(parts[3]),
            v=float(parts[4]),
            omega=float(parts[5]),
            timestamp=float(parts[7])
        )
    
    def _parse_laser_line(self, parts: List[str]) -> LaserReading:
        """Parse a FLASER line from the CLF file"""
        num_readings = int(parts[1])
        ranges = [float(x) for x in parts[2:2+num_readings]]
        pose_idx = 2 + num_readings
        pose = (
            float(parts[pose_idx]),     # x
            float(parts[pose_idx + 1]), # y
            float(parts[pose_idx + 2])  # theta
        )
        timestamp = float(parts[pose_idx + 4])
        
        return LaserReading(
            num_readings=num_readings,
            ranges=ranges,
            pose=pose,
            timestamp=timestamp
        )
    
    def _load_data(self, max_entries: int):
        """Load and parse the CLF dataset"""
        try:
            with open(self.data_path, 'r') as f:
                odom_count = 0
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    reading_type = parts[0]
                    
                    if reading_type == 'ODOM':
                        self.odom_readings.append(self._parse_odom_line(parts))
                        odom_count += 1
                        if odom_count >= max_entries:
                            break
                    elif reading_type == 'FLASER':
                        self.laser_readings.append(self._parse_laser_line(parts))
            
            # Sort readings by timestamp
            self.odom_readings.sort(key=lambda x: x.timestamp)
            self.laser_readings.sort(key=lambda x: x.timestamp)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {self.data_path}")
        except ValueError as e:
            raise ValueError(f"Error parsing dataset: {e}")
    
    def get_pose_at_index(self, index: int) -> Tuple[float, float, float]:
        """Get pose at specific index from odometry readings"""
        if 0 <= index < len(self.odom_readings):
            reading = self.odom_readings[index]
            return (reading.x, reading.y, reading.theta)
        raise IndexError("Pose index out of range")
    
    def get_total_poses(self) -> int:
        """Get total number of odometry readings"""
        return len(self.odom_readings)
    
    def get_pose_state(self, index: int) -> np.ndarray:
        """Get pose state vector [x, y, theta] at index"""
        if 0 <= index < len(self.odom_readings):
            reading = self.odom_readings[index]
            return np.array([reading.x, reading.y, reading.theta])
        raise IndexError("Pose index out of range")
    
    def get_control_input(self, index: int) -> Optional[Tuple[float, float, float]]:
        """Get control input (v, omega) and time difference at index"""
        if index >= len(self.odom_readings) - 1:
            return None
        
        current = self.odom_readings[index]
        next_reading = self.odom_readings[index + 1]
        
        dt = next_reading.timestamp - current.timestamp
        return current.v, current.omega, dt
    
    def get_laser_reading_at_time(self, timestamp: float, 
                                tolerance: float = 0.1) -> Optional[LaserReading]:
        """Get laser reading closest to the given timestamp within tolerance"""
        for reading in self.laser_readings:
            if abs(reading.timestamp - timestamp) < tolerance:
                return reading
        return None

# <---------------EKF SLAM ---------------->
# EKF SLAM Parameters
n_state = 3  # Robot state dimension
robot_fov = 2  # Robot field of view

# Landmark parameters
landmarks = [
    (4, 4),
    (4, 8),
    (8, 8),
    (12, 8),
    (16, 8),
    (16, 4),
    (12, 4),
    (8, 4)
]
n_landmarks = len(landmarks)

# Noise parameters
R = np.diag([0.002, 0.002, 0.00003])  # Motion noise
Q = np.diag([0.003, 0.000003, 0.01])  # Measurement noise

# Initialize EKF variables
mu = np.zeros((n_state + 2 * n_landmarks, 1))
sigma = np.zeros((n_state + 2 * n_landmarks, n_state + 2 * n_landmarks))
mu[:] = np.nan
np.fill_diagonal(sigma, 100)

# Helper matrices
Fx = np.block([[np.eye(3), np.zeros((n_state, 2 * n_landmarks))]])

def sim_measurements(x, landmarks):
    """Simulate landmark measurements"""
    rx, ry, rtheta = x[0], x[1], x[2]
    zs = []
    
    for lidx, landmark in enumerate(landmarks):
        lx, ly = landmark
        dist = np.linalg.norm(np.array([lx-rx, ly-ry]))
        phi = np.arctan2(ly - ry, lx - rx) - rtheta
        phi = np.arctan2(np.sin(phi), np.cos(phi))
        
        if dist < robot_fov:
            signature = max(0, min(1, 1.0 - dist/robot_fov))
            zs.append((dist, phi, signature, lidx))
    return zs

def prediction_update(mu, sigma, u, dt):
    """EKF prediction step"""
    rx, ry, theta = mu[0], mu[1], mu[2]
    v, w = u[0], u[1]
    
    # Update state estimate
    state_model_mat = np.zeros((n_state, 1))
    if np.abs(w) > 0.01:
        state_model_mat[0] = -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)
        state_model_mat[1] = (v / w) * np.cos(theta) - (v / w) * np.cos(theta + w * dt)
    else:
        state_model_mat[0] = v * dt * np.cos(theta)
        state_model_mat[1] = v * dt * np.sin(theta)
    state_model_mat[2] = w * dt
    mu += np.transpose(Fx).dot(state_model_mat)
    
    # Update covariance
    state_jacobian_mat = np.zeros((n_state, n_state))
    if np.abs(w) > 0.01:
        state_jacobian_mat[0, 2] = -(v / w) * np.cos(theta) + (v / w) * np.cos(theta + w * dt)
        state_jacobian_mat[1, 2] = -(v / w) * np.sin(theta) + (v / w) * np.sin(theta + w * dt)
    else:
        state_jacobian_mat[0, 2] = -v * np.sin(theta) * dt
        state_jacobian_mat[1, 2] = v * np.cos(theta) * dt
    
    G = np.eye(sigma.shape[0]) + np.transpose(Fx).dot(state_jacobian_mat).dot(Fx)
    sigma = G.dot(sigma).dot(np.transpose(G)) + np.transpose(Fx).dot(R).dot(Fx)
    
    return mu, sigma

def measurement_update(mu, sigma, zs):
    """EKF measurement update step"""
    rx, ry, theta = mu[0, 0], mu[1, 0], mu[2, 0]
    delta_zs = [np.zeros((3, 1)) for _ in range(n_landmarks)]
    Ks = [np.zeros((mu.shape[0], 3)) for _ in range(n_landmarks)]
    Hs = [np.zeros((3, mu.shape[0])) for _ in range(n_landmarks)]
    
    for z in zs:
        dist, phi, signature, lidx = z
        mu_landmark = mu[n_state + lidx * 2 : n_state + lidx * 2 + 2]
        
        if np.isnan(mu_landmark[0]):
            mu_landmark[0] = rx + dist * np.cos(phi + theta)
            mu_landmark[1] = ry + dist * np.sin(phi + theta)
            mu[n_state + lidx * 2 : n_state + lidx * 2 + 2] = mu_landmark
        
        delta = mu_landmark - np.array([[rx], [ry]])
        q = np.linalg.norm(delta)**2
        
        dist_est = np.sqrt(q)
        phi_est = np.arctan2(delta[1, 0], delta[0, 0]) - theta
        phi_est = np.arctan2(np.sin(phi_est), np.cos(phi_est))
        
        z_est = np.array([[dist_est], [phi_est], [signature]])
        z_act = np.array([[dist], [phi], [signature]])
        
        delta_zs[lidx] = z_act - z_est
        
        Fxj = np.block([[Fx], [np.zeros((2, Fx.shape[1]))]])
        Fxj[n_state : n_state + 2, n_state + 2 * lidx : n_state + 2 * lidx + 2] = np.eye(2)
        
        H = np.array([
            [-delta[0, 0]/np.sqrt(q), -delta[1, 0]/np.sqrt(q), 0, delta[0, 0]/np.sqrt(q), delta[1, 0]/np.sqrt(q)],
            [delta[1, 0]/q, -delta[0, 0]/q, -1, -delta[1, 0]/q, delta[0, 0]/q],
            [0, 0, 0, 0, 0]
        ])
        H = H.dot(Fxj)
        Hs[lidx] = H
        Ks[lidx] = sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q))
    
    mu_offset = np.zeros(mu.shape)
    sigma_factor = np.eye(sigma.shape[0])
    for lidx in range(n_landmarks):
        mu_offset += Ks[lidx].dot(delta_zs[lidx])
        sigma_factor -= Ks[lidx].dot(Hs[lidx])
    
    mu += mu_offset
    sigma = sigma_factor.dot(sigma)
    
    return mu, sigma

class EKFSLAMVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.env = MatplotlibEnvironment()
        self.robot_path = []
        self.current_frame = 0
        
        # Initialize robot with first pose
        x_init = self.data_loader.get_pose_state(0)
        self.robot = DifferentialDrive(x_init)
        
        # Initialize state estimate
        global mu, sigma
        mu[0:3] = np.expand_dims(x_init, axis=1)
        sigma[0:3, 0:3] = 0.1 * np.eye(3)
        sigma[2, 2] = 0
        
        # Get and plot the full ground truth path
        self.ground_truth_path = []
        for i in range(self.data_loader.get_total_poses()):
            pose = self.data_loader.get_pose_state(i)
            self.ground_truth_path.append((pose[0], pose[1]))

    def init_animation(self):
        """Initialize the animation"""
        self.env.clear()
        return []

    def update(self, frame):
        """Animation update function"""
        self.current_frame = frame
        
        # Get control input from dataset
        control_data = self.data_loader.get_control_input(frame)
        if control_data is None:
            return []
        
        v, omega, dt = control_data
        u = np.array([v, omega])
        
        # Move robot
        self.robot.move_step(u, dt)
        
        # Get current odometry reading
        odom = self.data_loader.odom_readings[frame]
        
        # Get laser reading if available
        laser = self.data_loader.get_laser_reading_at_time(odom.timestamp)
        
        # Generate measurements
        if laser:
            zs = []
            for i, range_val in enumerate(laser.ranges):
                if range_val < robot_fov:
                    angle = -np.pi/2 + i * np.pi/180
                    zs.append((range_val, angle, range_val/robot_fov, i))
        else:
            zs = sim_measurements(self.robot.get_pose(), landmarks)
        
        # EKF SLAM updates
        global mu, sigma
        mu, sigma = prediction_update(mu, sigma, u, dt)
        mu, sigma = measurement_update(mu, sigma, zs)
        
        # Clear the previous frame
        self.env.ax.clear()
        self.env.setup_plot()
        
        # First plot the ground truth path
        if len(self.ground_truth_path) > 1:
            gt_path_array = np.array(self.ground_truth_path)
            self.env.ax.plot(gt_path_array[:, 0], gt_path_array[:, 1], 'g--', 
                            linewidth=2, alpha=0.5, label='Ground Truth Path')
        
        # Update and show robot path
        robot_pose = self.robot.get_pose()
        self.robot_path.append((robot_pose[0], robot_pose[1]))
        
        # Show robot
        self.env.show_robot(self.robot)
        
        # Then plot the robot's current path
        if len(self.robot_path) > 1:
            path_array = np.array(self.robot_path)
            self.env.ax.plot(path_array[:, 0], path_array[:, 1], 'r-', 
                            linewidth=2, alpha=0.8, label='Robot Path')
        
        # Show landmarks and measurements
        if not laser:  # Only show true landmarks for simulated measurements
            self.env.show_landmark_location(landmarks)
        self.env.show_measurements(self.robot.get_pose(), zs)
        
        # Show EKF estimates
        self.env.show_robot_estimate(mu, sigma)
        self.env.show_landmark_estimate(mu, sigma, n_landmarks, n_state)
        
        # Update plot bounds
        self.env.update_bounds(self.robot_path + self.ground_truth_path)  # Include both paths for bounds
        
        # Add status text
        status_text = f'Frame: {frame}/{self.data_loader.get_total_poses()}\n'
        status_text += f'Robot pose: ({robot_pose[0]:.2f}, {robot_pose[1]:.2f}, {robot_pose[2]:.2f})'
        self.env.ax.text(0.02, 0.98, status_text,
                        transform=self.env.ax.transAxes,
                        verticalalignment='top',
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7))
        
        # Add legend
        self.env.ax.legend(loc='upper right')
        
        # Return an empty list since we're not using blitting
        return []

    def animate(self, interval=50):
        """Start the animation"""
        plt.ion()  # Turn on interactive mode
    
        # Create a flag to track window status
        self.running = True
        
        def on_close(event):
            self.running = False
            # Save the final path plot
            plt.figure(figsize=(12, 8))
            path_array = np.array(self.robot_path)
            truth_path_array = np.array(self.ground_truth_path)
            plt.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=1, alpha=0.5, label='Robot Path')
            plt.plot(truth_path_array[:, 0], truth_path_array[:, 1], 'g--', linewidth=1, alpha=0.5, label='Ground Truth Path')
            plt.legend()
            plt.grid(True)
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.title('Robot Path')
            plt.axis('equal')
            plt.savefig('robot_path.jpg', bbox_inches='tight', dpi=300)
            plt.close('all')
        
        # Connect the close event
        self.env.fig.canvas.mpl_connect('close_event', on_close)
        anim = FuncAnimation(
            self.env.fig,
            self.update,
            init_func=self.init_animation,
            frames=self.data_loader.get_total_poses(),
            interval=interval,
            blit=False,  # Disable blitting
            repeat=False
        )
        plt.draw()
        plt.pause(0.001)  # Add a small pause to allow the window to appear
        try:
            plt.show(block=True)  # Block until window is closed
        except KeyboardInterrupt:
            plt.close('all')
        return anim

if __name__=='__main__':
    try:
        # Initialize the CLF data loader with limited entries
        data_loader = CLFDataLoader("sample_data/mit-killian.clf", max_entries=5200)
        
        # Create and run the visualizer
        visualizer = EKFSLAMVisualizer(data_loader)
        print("Starting animation...")
        anim = visualizer.animate(interval = 30)  # 50ms interval for smoother animation
        
    except Exception as e:
        print(f"Error occurred: {e}")
        plt.close('all')