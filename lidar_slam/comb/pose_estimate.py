import numpy as np

class PoseEstimate:
    """Class to represent a robot pose estimate with uncertainty"""
    
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """Initialize a pose estimate"""
        self.x = x
        self.y = y
        self.theta = theta
        
        # Covariance matrix for x, y, theta
        self.covariance = np.eye(3) * 0.01  # Default small uncertainty
        
    def to_dict(self):
        """Convert pose to dictionary format compatible with existing code"""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta
        }
    
    def from_dict(self, pose_dict):
        """Set pose from dictionary format"""
        self.x = pose_dict['x']
        self.y = pose_dict['y']
        self.theta = pose_dict['theta']
        return self
    
    def copy(self):
        """Create a copy of this pose estimate"""
        new_pose = PoseEstimate(self.x, self.y, self.theta)
        new_pose.covariance = self.covariance.copy()
        return new_pose