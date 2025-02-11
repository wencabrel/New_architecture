import json
import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

class LaserFeatureExtractor:
    def __init__(self, min_points_line=10, line_threshold=0.1, corner_angle_threshold=45):
        self.min_points_line = min_points_line
        self.line_threshold = line_threshold
        self.corner_angle_threshold = corner_angle_threshold

    def convert_scan_to_points(self, reading):
        """Convert laser scan reading to cartesian coordinates"""
        angles = np.linspace(-np.pi/2, np.pi/2, len(reading['range']))
        ranges = np.array(reading['range'])
        valid_idx = ranges < 10  # Filter out max range readings
        
        x = ranges[valid_idx] * np.cos(angles[valid_idx])
        y = ranges[valid_idx] * np.sin(angles[valid_idx])
        return np.column_stack((x, y))

    def extract_lines(self, points):
        """Extract line segments using RANSAC"""
        lines = []
        remaining_points = points.copy()
        
        while len(remaining_points) >= self.min_points_line:
            # Fit line using RANSAC
            ransac = RANSACRegressor(residual_threshold=self.line_threshold)
            
            try:
                X = remaining_points[:, 0].reshape(-1, 1)
                y = remaining_points[:, 1]
                ransac.fit(X, y)
                inliers = ransac.inlier_mask_
                
                if sum(inliers) >= self.min_points_line:
                    line_points = remaining_points[inliers]
                    lines.append(line_points)
                    remaining_points = remaining_points[~inliers]
                else:
                    break
            except:
                break
                
        return lines

    def detect_corners(self, lines):
        """Detect corners from intersecting lines"""
        corners = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i]
                line2 = lines[j]
                
                # Calculate line directions
                dir1 = np.polyfit(line1[:, 0], line1[:, 1], 1)
                dir2 = np.polyfit(line2[:, 0], line2[:, 1], 1)
                
                # Calculate angle between lines
                angle = np.abs(np.arctan((dir2[0] - dir1[0])/(1 + dir1[0]*dir2[0])))
                angle_deg = np.degrees(angle)
                
                if angle_deg >= self.corner_angle_threshold:
                    # Find intersection point
                    x_corner = (dir2[1] - dir1[1])/(dir1[0] - dir2[0])
                    y_corner = dir1[0] * x_corner + dir1[1]
                    corners.append((x_corner, y_corner, angle_deg))
        
        return corners

    def cluster_points(self, points, eps=0.3, min_samples=5):
        """Cluster scan points using DBSCAN"""
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        # Organize points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])
        
        return [np.array(cluster) for label, cluster in clusters.items() if label != -1]

    def plot_features(self, points, lines, corners, clusters):
        """Plot extracted features"""
        plt.figure(figsize=(12, 8))
        
        # Plot original points
        plt.scatter(points[:, 0], points[:, 1], c='gray', s=1, alpha=0.5, label='Scan Points')
        
        # Plot lines
        for line in lines:
            plt.plot(line[:, 0], line[:, 1], 'r-', linewidth=2)
        
        # Plot corners
        if corners:
            corner_points = np.array(corners)
            plt.scatter(corner_points[:, 0], corner_points[:, 1], c='green', s=100, marker='*', label='Corners')
        
        # Plot clusters with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
        for cluster, color in zip(clusters, colors):
            plt.scatter(cluster[:, 0], cluster[:, 1], c=[color], s=20, alpha=0.6)
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Extracted Features from Laser Scan')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.show()

    def process_scan(self, reading):
        """Process a single laser scan and extract all features"""
        # Convert scan to points
        points = self.convert_scan_to_points(reading)
        
        # Extract features
        lines = self.extract_lines(points)
        corners = self.detect_corners(lines)
        clusters = self.cluster_points(points)
        
        # Plot results
        self.plot_features(points, lines, corners, clusters)
        
        return {
            'points': points,
            'lines': lines,
            'corners': corners,
            'clusters': clusters
        }
    
def main():
    # Read sensor data
    jsonFile = "DataSet/DataPreprocessed/intel-gfs"
    with open(jsonFile, 'r') as f:
        input = json.load(f)
        sensor_data = input['map']
    
    # Create feature extractor
    feature_extractor = LaserFeatureExtractor(
        min_points_line=10,
        line_threshold=0.1,
        corner_angle_threshold=45
    )
    
    # Process first scan as example
    first_key = sorted(sensor_data.keys())[0]
    first_scan = sensor_data[first_key]
    
    # Extract and plot features
    features = feature_extractor.process_scan(first_scan)
    
    # Print summary of extracted features
    print(f"Found {len(features['lines'])} line segments")
    print(f"Found {len(features['corners'])} corners")
    print(f"Found {len(features['clusters'])} distinct point clusters")

if __name__ == '__main__':
    main()