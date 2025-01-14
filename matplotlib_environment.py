import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Arrow
from matplotlib.transforms import Affine2D
import matplotlib.animation as animation

class MatplotlibEnvironment:
    """Matplotlib-based environment for EKF SLAM visualization"""
    
    def __init__(self, meter_per_pixel=0.02):
        """Initialize the matplotlib environment"""
        self.METER_PER_PIXEL = meter_per_pixel
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        
        # Set colors
        self.colors = {
            'black': '#000000',
            'grey': '#464646',
            'dark_grey': '#141414',
            'blue': '#0000FF',
            'green': '#00FF00',
            'red': '#FF0000',
            'white': '#FFFFFF'
        }
        
        # Initialize the plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the plot parameters"""
        self.ax.grid(True)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('EKF SLAM Visualization')
        
    def clear(self):
        """Clear the current plot"""
        self.ax.clear()
        self.setup_plot()
        
    def show_robot(self, robot):
        """Display the robot on the plot"""
        corners = robot.get_corners()
        corners_array = np.array(corners)
        
        # Create polygon for robot body
        robot_poly = Polygon(corners_array, facecolor=self.colors['grey'],
                           edgecolor=self.colors['dark_grey'], alpha=0.7)
        self.ax.add_patch(robot_poly)
        
        # Add direction arrow
        pose = robot.get_pose()
        arrow_length = 0.5
        dx = arrow_length * np.cos(pose[2])
        dy = arrow_length * np.sin(pose[2])
        self.ax.arrow(pose[0], pose[1], dx, dy, 
                     head_width=0.2, head_length=0.2,
                     fc=self.colors['red'], ec=self.colors['red'])
    
    def show_uncertainty_ellipse(self, center, covariance, color='red', alpha=0.3):
        """Visualize an uncertainty ellipse"""
        eigenvals, eigenvecs = np.linalg.eig(covariance[:2, :2])
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        # Create ellipse patch
        ellip = Ellipse(xy=center[:2], width=2*np.sqrt(eigenvals[0]), 
                       height=2*np.sqrt(eigenvals[1]), angle=angle,
                       facecolor='none', edgecolor=color, alpha=alpha)
        self.ax.add_patch(ellip)
    
    def show_landmark_location(self, landmarks):
        """Display actual landmark locations"""
        landmarks_array = np.array(landmarks)
        self.ax.scatter(landmarks_array[:, 0], landmarks_array[:, 1],
                       c='cyan', s=100, alpha=0.6, label='True Landmarks')
    
    def show_measurements(self, robot_pose, measurements):
        """Display measurement lines and observed landmarks"""
        rx, ry, theta = robot_pose
        
        for z in measurements:
            dist, phi, signature, lidx = z
            # Calculate landmark position in global frame
            lx = rx + dist * np.cos(phi + theta)
            ly = ry + dist * np.sin(phi + theta)
            
            # Draw measurement line
            color_intensity = max(0.2, signature)
            self.ax.plot([rx, lx], [ry, ly], color='blue', alpha=color_intensity)
            self.ax.scatter(lx, ly, c='blue', s=50, alpha=color_intensity)
    
    def show_robot_estimate(self, mu, sigma):
        """Display robot position estimate with uncertainty"""
        robot_pos = mu[:2]
        robot_cov = sigma[:2, :2]
        self.show_uncertainty_ellipse(robot_pos, robot_cov, color='red')
    
    def show_landmark_estimate(self, mu, sigma, n_landmarks, n_state=3):
        """Display landmark position estimates with uncertainties"""
        for i in range(n_landmarks):
            idx = n_state + 2*i
            landmark_mu = mu[idx:idx+2]
            landmark_sigma = sigma[idx:idx+2, idx:idx+2]
            
            if not np.isnan(landmark_mu).any():
                self.show_uncertainty_ellipse(landmark_mu, landmark_sigma, 
                                           color='green', alpha=0.5)
    
    def update_bounds(self, positions):
        """Update plot bounds based on positions"""
        if len(positions) > 0:
            positions = np.array(positions)
            margin = 2.0  # meters
            self.ax.set_xlim(positions[:, 0].min() - margin, 
                           positions[:, 0].max() + margin)
            self.ax.set_ylim(positions[:, 1].min() - margin, 
                           positions[:, 1].max() + margin)
    
    def show(self):
        """Display the plot"""
        plt.show()
    
    def update(self):
        """Update the plot (for animation)"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()