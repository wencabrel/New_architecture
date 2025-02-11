import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN
import math

def extract_lines(scan_points, min_points=10, threshold=0.1):
    """Use Split-and-Merge algorithm for line extraction"""
    lines = []
    points = np.array(scan_points)
    # Implement split-and-merge algorithm
    # Return list of detected lines
    return lines

def detect_corners(scan_points, angle_threshold=30):
    """Detect corners using angle between consecutive points"""
    corners = []
    # Calculate angles between consecutive points
    # Mark points with sharp angles as corners
    return corners

def find_geometric_patterns(scan_points):
    """Identify common geometric patterns (rectangles, circles, etc.)"""
    patterns = {
        'rectangles': [],
        'circles': [],
        'triangles': []
    }
    # Implement pattern detection algorithms
    return patterns

def extract_stable_landmarks(raw_scan, corrected_scan):
    """Find stable landmarks that appear in both scans"""
    landmarks = []
    # Compare scans to find consistent features
    return landmarks

def cluster_scan_points(scan_points, eps=0.3, min_samples=5):
    """Cluster scan points using DBSCAN"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scan_points)
    return clustering.labels_

def main():
    # Load raw and corrected data
    with open("DataSet/DataPreprocessed/intel-gfs", 'r') as f:
        raw_data = json.load(f)['map']
    
    with open("DataSet/DataPreprocessed/intel_corrected-log", 'r') as f:
        corrected_data = json.load(f)['map']

    # Process both datasets
    for timestamp in sorted(raw_data.keys())[:1]:  # Process first scan for example
        raw_scan = raw_data[timestamp]
        corrected_scan = corrected_data[timestamp]

        # Convert polar to Cartesian coordinates
        angles = np.linspace(-np.pi/2, np.pi/2, len(raw_scan['range']))
        
        # Process raw scan
        raw_x = raw_scan['x'] + np.cos(angles) * raw_scan['range']
        raw_y = raw_scan['y'] + np.sin(angles) * raw_scan['range']
        raw_points = np.column_stack((raw_x, raw_y))

        # Process corrected scan
        corr_x = corrected_scan['x'] + np.cos(angles) * corrected_scan['range']
        corr_y = corrected_scan['y'] + np.sin(angles) * corrected_scan['range']
        corr_points = np.column_stack((corr_x, corr_y))

        # Feature extraction and comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

        # Plot original scans
        ax1.scatter(raw_x, raw_y, c='blue', label='Raw', s=1)
        ax1.scatter(corr_x, corr_y, c='red', label='Corrected', s=1)
        ax1.set_title('Original Scans')
        ax1.legend()

        # Plot clustered points
        raw_clusters = cluster_scan_points(raw_points)
        ax2.scatter(raw_x, raw_y, c=raw_clusters, cmap='viridis', s=1)
        ax2.set_title('Clustered Points (Raw)')

        # Plot detected lines
        raw_lines = extract_lines(raw_points)
        corr_lines = extract_lines(corr_points)
        ax3.scatter(raw_x, raw_y, c='gray', s=1)
        # Plot lines here
        ax3.set_title('Line Extraction')

        # Plot corners and landmarks
        raw_corners = detect_corners(raw_points)
        corr_corners = detect_corners(corr_points)
        ax4.scatter(raw_x, raw_y, c='gray', s=1)
        # Plot corners and landmarks here
        ax4.set_title('Corners and Landmarks')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()