import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import copy

class FeatureType(Enum):
    """Enumeration for different types of LiDAR features"""
    SHARP_EDGE = "sharp_edge"
    LESS_SHARP_EDGE = "less_sharp_edge"
    PLANAR = "planar"
    LESS_PLANAR = "less_planar"

@dataclass
class LiDARFeature:
    """
    Data class representing a single LiDAR feature point
    
    Attributes:
        point_local: [x, y] coordinates in robot's local frame
        point_world: [x, y] coordinates in world frame
        feature_type: Type of feature (edge or planar)
        curvature: Calculated curvature value
        strength: Feature strength/quality measure
        scan_index: Original index in the scan array
        sector: Scan sector this feature belongs to
        distance: Distance from robot to feature point
        angle: Angle of feature point relative to robot
    """
    point_local: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    point_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    feature_type: FeatureType = FeatureType.LESS_PLANAR
    curvature: float = 0.0
    strength: float = 0.0
    scan_index: int = -1
    sector: int = -1
    distance: float = 0.0
    angle: float = 0.0
    
    def copy(self):
        """Create a deep copy of the feature"""
        return LiDARFeature(
            point_local=self.point_local.copy(),
            point_world=self.point_world.copy(),
            feature_type=self.feature_type,
            curvature=self.curvature,
            strength=self.strength,
            scan_index=self.scan_index,
            sector=self.sector,
            distance=self.distance,
            angle=self.angle
        )

@dataclass
class FeatureSet:
    """
    Container for all features extracted from a single LiDAR scan
    
    Attributes:
        features: List of all extracted features
        sharp_edges: List of sharp edge features
        less_sharp_edges: List of less sharp edge features
        planar_features: List of planar features
        less_planar_features: List of less planar features
        scan_timestamp: Timestamp of the original scan
        robot_pose: Robot pose when scan was taken
        extraction_time: Time taken for feature extraction (ms)
        quality_metrics: Dictionary of quality assessment metrics
    """
    features: List[LiDARFeature] = field(default_factory=list)
    sharp_edges: List[LiDARFeature] = field(default_factory=list)
    less_sharp_edges: List[LiDARFeature] = field(default_factory=list)
    planar_features: List[LiDARFeature] = field(default_factory=list)
    less_planar_features: List[LiDARFeature] = field(default_factory=list)
    scan_timestamp: float = 0.0
    robot_pose: Any = None  # PoseEstimate object
    extraction_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[LiDARFeature]:
        """Get all features of a specific type"""
        if feature_type == FeatureType.SHARP_EDGE:
            return self.sharp_edges
        elif feature_type == FeatureType.LESS_SHARP_EDGE:
            return self.less_sharp_edges
        elif feature_type == FeatureType.PLANAR:
            return self.planar_features
        elif feature_type == FeatureType.LESS_PLANAR:
            return self.less_planar_features
        else:
            return []
    
    def get_feature_count_by_type(self) -> Dict[str, int]:
        """Get count of features by type"""
        return {
            'sharp_edges': len(self.sharp_edges),
            'less_sharp_edges': len(self.less_sharp_edges),
            'planar_features': len(self.planar_features),
            'less_planar_features': len(self.less_planar_features),
            'total': len(self.features)
        }
    
    def get_all_edge_features(self) -> List[LiDARFeature]:
        """Get all edge features (sharp and less sharp)"""
        return self.sharp_edges + self.less_sharp_edges
    
    def get_all_planar_features(self) -> List[LiDARFeature]:
        """Get all planar features (planar and less planar)"""
        return self.planar_features + self.less_planar_features

class CurvatureCalculator:
    """
    LOAM-style curvature calculation for LiDAR points
    """
    
    def __init__(self, window_size: int = 5, max_range: float = 11.9):
        """
        Initialize curvature calculator
        
        Args:
            window_size: Number of neighboring points to consider for curvature calculation
            max_range: Maximum valid range for LiDAR points
        """
        self.window_size = window_size
        self.max_range = max_range
        
    def calculate_curvature_array(self, scan_ranges: List[float], scan_angles: np.ndarray) -> np.ndarray:
        """
        Calculate curvature for all points in a scan
        
        Args:
            scan_ranges: List of range measurements
            scan_angles: Array of angles for each measurement
            
        Returns:
            Array of curvature values for each point
        """
        num_points = len(scan_ranges)
        curvatures = np.zeros(num_points)
        
        # Convert to numpy array for easier processing
        ranges = np.array(scan_ranges)
        
        # Calculate curvature for each point
        for i in range(num_points):
            curvatures[i] = self._calculate_point_curvature(ranges, i, num_points)
        
        return curvatures
    
    def _calculate_point_curvature(self, ranges: np.ndarray, center_idx: int, num_points: int) -> float:
        """
        Calculate curvature for a single point using LOAM approach
        
        Args:
            ranges: Array of range measurements
            center_idx: Index of the center point
            num_points: Total number of points in scan
            
        Returns:
            Curvature value for the center point
        """
        # Skip points at max range (invalid measurements)
        if ranges[center_idx] >= self.max_range:
            return 0.0
        
        # Define the window around the center point
        half_window = self.window_size // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(num_points - 1, center_idx + half_window)
        
        # Calculate sum of range differences
        range_sum = 0.0
        valid_count = 0
        
        for j in range(start_idx, end_idx + 1):
            if j != center_idx and ranges[j] < self.max_range:
                range_sum += abs(ranges[center_idx] - ranges[j])
                valid_count += 1
        
        # Avoid division by zero
        if valid_count == 0 or ranges[center_idx] == 0:
            return 0.0
        
        # LOAM curvature formula: c = |∑(r_i - r_j)| / (r_i * |S|)
        curvature = range_sum / (ranges[center_idx] * valid_count)
        
        return curvature
    
    def is_valid_for_feature_extraction(self, ranges: np.ndarray, center_idx: int, num_points: int) -> bool:
        """
        Check if a point is valid for feature extraction
        
        Args:
            ranges: Array of range measurements
            center_idx: Index of the point to check
            num_points: Total number of points
            
        Returns:
            True if point is valid for feature extraction
        """
        # Skip points at max range
        if ranges[center_idx] >= self.max_range:
            return False
        
        # Skip points too close to scan boundaries
        boundary_buffer = self.window_size // 2
        if center_idx < boundary_buffer or center_idx >= num_points - boundary_buffer:
            return False
        
        # Check if neighboring points are valid
        half_window = self.window_size // 2
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        valid_neighbors = 0
        for j in range(start_idx, end_idx + 1):
            if j != center_idx and ranges[j] < self.max_range:
                valid_neighbors += 1
        
        # Require at least half of the neighbors to be valid
        return valid_neighbors >= (self.window_size - 1) // 2

class FeatureClassifier:
    """
    Classifier for organizing LiDAR features into hierarchical categories
    """
    
    def __init__(self, num_sectors: int = 6, 
                 sharp_edge_threshold: float = 0.1, 
                 planar_threshold: float = 0.1,
                 max_sharp_edges_per_sector: int = 2,
                 max_less_sharp_per_sector: int = 20,
                 max_planar_per_sector: int = 4):
        """
        Initialize feature classifier
        
        Args:
            num_sectors: Number of sectors to divide the scan into
            sharp_edge_threshold: Curvature threshold for edge features
            planar_threshold: Curvature threshold for planar features
            max_sharp_edges_per_sector: Maximum sharp edge features per sector
            max_less_sharp_per_sector: Maximum less sharp features per sector
            max_planar_per_sector: Maximum planar features per sector
        """
        self.num_sectors = num_sectors
        self.sharp_edge_threshold = sharp_edge_threshold
        self.planar_threshold = planar_threshold
        self.max_sharp_edges_per_sector = max_sharp_edges_per_sector
        self.max_less_sharp_per_sector = max_less_sharp_per_sector
        self.max_planar_per_sector = max_planar_per_sector
        
    def classify_features(self, scan_x: List[float], scan_y: List[float], 
                         scan_ranges: List[float], curvatures: np.ndarray,
                         scan_angles: np.ndarray, robot_pose: Any) -> FeatureSet:
        """
        Classify scan points into different feature types
        
        Args:
            scan_x: X coordinates of scan points in world frame
            scan_y: Y coordinates of scan points in world frame
            scan_ranges: Range measurements
            curvatures: Calculated curvature values
            scan_angles: Angles for each scan point
            robot_pose: Current robot pose
            
        Returns:
            FeatureSet containing classified features
        """
        feature_set = FeatureSet()
        feature_set.robot_pose = robot_pose
        
        num_points = len(scan_x)
        points_per_sector = num_points // self.num_sectors
        
        # Process each sector separately
        for sector in range(self.num_sectors):
            start_idx = sector * points_per_sector
            end_idx = start_idx + points_per_sector
            if sector == self.num_sectors - 1:  # Last sector gets remaining points
                end_idx = num_points
            
            # Extract features from this sector
            sector_features = self._extract_sector_features(
                scan_x[start_idx:end_idx], 
                scan_y[start_idx:end_idx],
                scan_ranges[start_idx:end_idx],
                curvatures[start_idx:end_idx],
                scan_angles[start_idx:end_idx],
                start_idx, sector, robot_pose
            )
            
            # Add sector features to the main feature set
            feature_set.features.extend(sector_features['all'])
            feature_set.sharp_edges.extend(sector_features['sharp_edges'])
            feature_set.less_sharp_edges.extend(sector_features['less_sharp_edges'])
            feature_set.planar_features.extend(sector_features['planar'])
            feature_set.less_planar_features.extend(sector_features['less_planar'])
        
        return feature_set
    
    def _extract_sector_features(self, sector_x: List[float], sector_y: List[float],
                                sector_ranges: List[float], sector_curvatures: np.ndarray,
                                sector_angles: np.ndarray, sector_start_idx: int,
                                sector_num: int, robot_pose: Any) -> Dict[str, List[LiDARFeature]]:
        """
        Extract features from a single sector
        
        Args:
            sector_x, sector_y: Coordinates for this sector
            sector_ranges: Range measurements for this sector
            sector_curvatures: Curvature values for this sector
            sector_angles: Angles for this sector
            sector_start_idx: Starting index in the original scan
            sector_num: Sector number
            robot_pose: Current robot pose
            
        Returns:
            Dictionary containing lists of features by type
        """
        features = {
            'all': [],
            'sharp_edges': [],
            'less_sharp_edges': [],
            'planar': [],
            'less_planar': []
        }
        
        # Create candidate features with curvature values
        candidates = []
        for i, (x, y, range_val, curvature, angle) in enumerate(
            zip(sector_x, sector_y, sector_ranges, sector_curvatures, sector_angles)
        ):
            if range_val < 11.9:  # Skip invalid measurements
                feature = LiDARFeature(
                    point_world=np.array([x, y]),
                    point_local=self._world_to_local(np.array([x, y]), robot_pose),
                    curvature=curvature,
                    strength=curvature,
                    scan_index=sector_start_idx + i,
                    sector=sector_num,
                    distance=range_val,
                    angle=angle
                )
                candidates.append(feature)
        
        if not candidates:
            return features
        
        # Sort candidates by curvature for classification
        candidates.sort(key=lambda f: f.curvature, reverse=True)
        
        # Extract sharp edge features (highest curvature)
        edge_candidates = [f for f in candidates if f.curvature > self.sharp_edge_threshold]
        sharp_edges = edge_candidates[:self.max_sharp_edges_per_sector]
        for feature in sharp_edges:
            feature.feature_type = FeatureType.SHARP_EDGE
        features['sharp_edges'] = sharp_edges
        
        # Extract less sharp edge features
        remaining_edge_candidates = edge_candidates[len(sharp_edges):]
        less_sharp_edges = remaining_edge_candidates[:self.max_less_sharp_per_sector]
        for feature in less_sharp_edges:
            feature.feature_type = FeatureType.LESS_SHARP_EDGE
        features['less_sharp_edges'] = less_sharp_edges
        
        # Extract planar features (lowest curvature)
        candidates.sort(key=lambda f: f.curvature)  # Sort by ascending curvature
        planar_candidates = [f for f in candidates if f.curvature < self.planar_threshold]
        planar_features = planar_candidates[:self.max_planar_per_sector]
        for feature in planar_features:
            feature.feature_type = FeatureType.PLANAR
        features['planar'] = planar_features
        
        # Extract less planar features (remaining points, downsampled)
        used_indices = set()
        for feature_list in [sharp_edges, less_sharp_edges, planar_features]:
            for feature in feature_list:
                used_indices.add(feature.scan_index)
        
        remaining_candidates = [f for f in candidates if f.scan_index not in used_indices]
        # Downsample remaining candidates (take every 4th point)
        less_planar_features = remaining_candidates[::4]
        for feature in less_planar_features:
            feature.feature_type = FeatureType.LESS_PLANAR
        features['less_planar'] = less_planar_features
        
        # Combine all features
        features['all'] = sharp_edges + less_sharp_edges + planar_features + less_planar_features
        
        return features
    
    def _world_to_local(self, world_point: np.ndarray, robot_pose: Any) -> np.ndarray:
        """
        Transform a point from world frame to robot's local frame
        
        Args:
            world_point: Point in world frame [x, y]
            robot_pose: Robot pose (PoseEstimate object)
            
        Returns:
            Point in robot's local frame [x, y]
        """
        if robot_pose is None:
            return world_point.copy()
        
        # Extract pose components
        x, y, theta = robot_pose.x, robot_pose.y, robot_pose.theta
        
        # Create rotation matrix for inverse transform
        c = math.cos(theta)
        s = math.sin(theta)
        
        # Apply translation and rotation
        dx = world_point[0] - x
        dy = world_point[1] - y
        
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        
        return np.array([local_x, local_y])

class FeatureValidator:
    """
    Quality assessment and validation for extracted features
    """
    
    def __init__(self, min_features_per_scan: int = 5,
                 max_features_per_scan: int = 200,
                 min_distance_between_features: float = 0.1,
                 max_sector_imbalance: float = 0.7):
        """
        Initialize feature validator
        
        Args:
            min_features_per_scan: Minimum number of features required
            max_features_per_scan: Maximum number of features allowed
            min_distance_between_features: Minimum distance between features (meters)
            max_sector_imbalance: Maximum allowed imbalance in sector distribution
        """
        self.min_features_per_scan = min_features_per_scan
        self.max_features_per_scan = max_features_per_scan
        self.min_distance_between_features = min_distance_between_features
        self.max_sector_imbalance = max_sector_imbalance
        
    def validate_feature_set(self, feature_set: FeatureSet) -> Dict[str, float]:
        """
        Perform comprehensive quality assessment of a feature set
        
        Args:
            feature_set: FeatureSet to validate
            
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {}
        
        # Basic count metrics
        metrics['total_features'] = len(feature_set.features)
        metrics['edge_features'] = len(feature_set.get_all_edge_features())
        metrics['planar_features'] = len(feature_set.get_all_planar_features())
        
        # Feature type distribution
        counts = feature_set.get_feature_count_by_type()
        metrics.update(counts)
        
        # Quality assessments
        metrics['spatial_distribution_quality'] = self._assess_spatial_distribution(feature_set)
        metrics['sector_balance'] = self._assess_sector_balance(feature_set)
        metrics['feature_strength_avg'] = self._calculate_average_strength(feature_set)
        metrics['feature_strength_std'] = self._calculate_strength_std(feature_set)
        metrics['distance_distribution_quality'] = self._assess_distance_distribution(feature_set)
        
        # Overall quality score (0-1, higher is better)
        metrics['overall_quality'] = self._calculate_overall_quality(metrics)
        
        # Validation flags
        metrics['meets_minimum_count'] = metrics['total_features'] >= self.min_features_per_scan
        metrics['within_maximum_count'] = metrics['total_features'] <= self.max_features_per_scan
        metrics['good_sector_balance'] = metrics['sector_balance'] >= (1.0 - self.max_sector_imbalance)
        metrics['acceptable_quality'] = metrics['overall_quality'] >= 0.3
        
        # Store metrics in the feature set
        feature_set.quality_metrics = metrics
        
        return metrics
    
    def _assess_spatial_distribution(self, feature_set: FeatureSet) -> float:
        """
        Assess how well features are distributed spatially
        
        Returns:
            Quality score 0-1 (higher is better)
        """
        if len(feature_set.features) < 2:
            return 0.0
        
        # Calculate minimum distances between features
        min_distances = []
        features = feature_set.features
        
        for i in range(len(features)):
            min_dist = float('inf')
            for j in range(len(features)):
                if i != j:
                    dist = np.linalg.norm(features[i].point_world - features[j].point_world)
                    min_dist = min(min_dist, dist)
            if min_dist != float('inf'):
                min_distances.append(min_dist)
        
        if not min_distances:
            return 0.0
        
        # Calculate quality based on minimum distance distribution
        avg_min_distance = np.mean(min_distances)
        quality = min(1.0, avg_min_distance / self.min_distance_between_features)
        
        return quality
    
    def _assess_sector_balance(self, feature_set: FeatureSet) -> float:
        """
        Assess how balanced the feature distribution is across sectors
        
        Returns:
            Balance score 0-1 (higher is better)
        """
        if not feature_set.features:
            return 0.0
        
        # Count features per sector
        sector_counts = {}
        for feature in feature_set.features:
            sector = feature.sector
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        if len(sector_counts) <= 1:
            return 0.0
        
        # Calculate balance metric
        counts = list(sector_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 0.0
        
        # Coefficient of variation (lower is better, so invert)
        cv = std_count / mean_count
        balance = max(0.0, 1.0 - cv)
        
        return balance
    
    def _calculate_average_strength(self, feature_set: FeatureSet) -> float:
        """Calculate average feature strength (curvature)"""
        if not feature_set.features:
            return 0.0
        
        strengths = [feature.strength for feature in feature_set.features]
        return np.mean(strengths)
    
    def _calculate_strength_std(self, feature_set: FeatureSet) -> float:
        """Calculate standard deviation of feature strengths"""
        if not feature_set.features:
            return 0.0
        
        strengths = [feature.strength for feature in feature_set.features]
        return np.std(strengths)
    
    def _assess_distance_distribution(self, feature_set: FeatureSet) -> float:
        """
        Assess the distribution of features across different distances
        
        Returns:
            Quality score 0-1 (higher is better)
        """
        if not feature_set.features:
            return 0.0
        
        distances = [feature.distance for feature in feature_set.features]
        
        # Check for features at different distance ranges
        near_features = sum(1 for d in distances if d < 2.0)
        mid_features = sum(1 for d in distances if 2.0 <= d < 5.0)
        far_features = sum(1 for d in distances if d >= 5.0)
        
        total = len(distances)
        
        # Prefer balanced distribution across ranges
        near_ratio = near_features / total
        mid_ratio = mid_features / total
        far_ratio = far_features / total
        
        # Calculate distribution quality (penalize extreme imbalances)
        ratios = [near_ratio, mid_ratio, far_ratio]
        non_zero_ratios = [r for r in ratios if r > 0]
        
        if len(non_zero_ratios) < 2:
            return 0.5  # Poor distribution
        
        # Use entropy-like measure for distribution quality
        entropy = -sum(r * math.log(r + 1e-10) for r in non_zero_ratios)
        max_entropy = math.log(len(non_zero_ratios))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall quality score from individual metrics
        
        Args:
            metrics: Dictionary of quality metrics
            
        Returns:
            Overall quality score 0-1
        """
        # Weight different aspects of quality
        weights = {
            'spatial_distribution_quality': 0.3,
            'sector_balance': 0.2,
            'distance_distribution_quality': 0.2,
            'feature_count_quality': 0.3
        }
        
        # Feature count quality (normalized sigmoid)
        count = metrics['total_features']
        optimal_count = 50  # Target number of features
        count_quality = 1.0 / (1.0 + math.exp(-0.1 * (count - optimal_count/2)))
        
        # Combine weighted scores
        overall = (
            weights['spatial_distribution_quality'] * metrics['spatial_distribution_quality'] +
            weights['sector_balance'] * metrics['sector_balance'] +
            weights['distance_distribution_quality'] * metrics['distance_distribution_quality'] +
            weights['feature_count_quality'] * count_quality
        )
        
        return min(1.0, max(0.0, overall))

class FeatureVisualizer:
    """
    Visualization and debugging tools for feature extraction
    """
    
    def __init__(self, debug_level: int = 1):
        """
        Initialize feature visualizer
        
        Args:
            debug_level: Level of debug output (0=none, 1=basic, 2=detailed, 3=verbose)
        """
        self.debug_level = debug_level
        
    def visualize_features(self, feature_set: FeatureSet, scan_x: List[float], scan_y: List[float],
                          ax: plt.Axes = None, show_curvatures: bool = False,
                          show_sectors: bool = False) -> plt.Axes:
        """
        Visualize extracted features overlaid on the scan
        
        Args:
            feature_set: FeatureSet to visualize
            scan_x, scan_y: Original scan points
            ax: Matplotlib axis to plot on (None to create new)
            show_curvatures: Whether to show curvature values as colors
            show_sectors: Whether to show sector boundaries
            
        Returns:
            Matplotlib axis with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot original scan points
        ax.scatter(scan_x, scan_y, c='lightgray', s=1, alpha=0.5, label='Scan Points')
        
        # Define colors for different feature types
        feature_colors = {
            FeatureType.SHARP_EDGE: 'red',
            FeatureType.LESS_SHARP_EDGE: 'orange',
            FeatureType.PLANAR: 'blue',
            FeatureType.LESS_PLANAR: 'lightblue'
        }
        
        feature_sizes = {
            FeatureType.SHARP_EDGE: 50,
            FeatureType.LESS_SHARP_EDGE: 30,
            FeatureType.PLANAR: 40,
            FeatureType.LESS_PLANAR: 10
        }
        
        # Plot features by type
        for feature_type in FeatureType:
            features = feature_set.get_features_by_type(feature_type)
            if features:
                x_coords = [f.point_world[0] for f in features]
                y_coords = [f.point_world[1] for f in features]
                
                if show_curvatures:
                    curvatures = [f.curvature for f in features]
                    scatter = ax.scatter(x_coords, y_coords, 
                                       c=curvatures, cmap='viridis',
                                       s=feature_sizes[feature_type],
                                       alpha=0.8, label=feature_type.value.replace('_', ' ').title())
                    plt.colorbar(scatter, ax=ax, label='Curvature')
                else:
                    ax.scatter(x_coords, y_coords, 
                             c=feature_colors[feature_type],
                             s=feature_sizes[feature_type],
                             alpha=0.8, label=feature_type.value.replace('_', ' ').title())
        
        # Plot robot position if available
        if feature_set.robot_pose:
            ax.scatter(feature_set.robot_pose.x, feature_set.robot_pose.y, 
                      c='green', s=100, marker='*', label='Robot Position')
            
            # Draw orientation arrow
            arrow_length = 0.5
            dx = arrow_length * math.cos(feature_set.robot_pose.theta)
            dy = arrow_length * math.sin(feature_set.robot_pose.theta)
            ax.arrow(feature_set.robot_pose.x, feature_set.robot_pose.y, dx, dy,
                    head_width=0.1, head_length=0.1, fc='green', ec='green')
        
        # Show sector boundaries if requested
        if show_sectors and feature_set.features:
            self._draw_sector_boundaries(ax, feature_set)
        
        # Add title with metrics
        if feature_set.quality_metrics:
            metrics = feature_set.quality_metrics
            title = (f"Features: {metrics['total_features']} "
                    f"(E:{metrics['edge_features']}, P:{metrics['planar_features']}) "
                    f"Q:{metrics['overall_quality']:.2f}")
            ax.set_title(title)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        return ax
    
    def _draw_sector_boundaries(self, ax: plt.Axes, feature_set: FeatureSet):
        """Draw sector boundaries on the plot"""
        if not feature_set.robot_pose or not feature_set.features:
            return
        
        # Get unique sectors
        sectors = set(f.sector for f in feature_set.features)
        num_sectors = max(sectors) + 1 if sectors else 6
        
        robot_x, robot_y = feature_set.robot_pose.x, feature_set.robot_pose.y
        
        # Draw sector lines
        max_range = 12.0
        for i in range(num_sectors + 1):
            angle = -math.pi/2 + i * math.pi / num_sectors
            end_x = robot_x + max_range * math.cos(angle)
            end_y = robot_y + max_range * math.sin(angle)
            ax.plot([robot_x, end_x], [robot_y, end_y], 'k--', alpha=0.3, linewidth=0.5)
    
    def plot_quality_metrics(self, feature_sets: List[FeatureSet], 
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot quality metrics over time for multiple feature sets
        
        Args:
            feature_sets: List of FeatureSet objects to analyze
            figsize: Figure size for the plot
            
        Returns:
            Matplotlib figure with quality plots
        """
        if not feature_sets:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Feature Extraction Quality Metrics Over Time')
        
        # Extract metrics
        timestamps = [fs.scan_timestamp for fs in feature_sets]
        
        metrics_to_plot = [
            ('total_features', 'Total Features'),
            ('overall_quality', 'Overall Quality'),
            ('spatial_distribution_quality', 'Spatial Distribution'),
            ('sector_balance', 'Sector Balance'),
            ('feature_strength_avg', 'Average Feature Strength'),
            ('extraction_time', 'Extraction Time (ms)')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            values = []
            for fs in feature_sets:
                if metric in fs.quality_metrics:
                    values.append(fs.quality_metrics[metric])
                elif metric == 'extraction_time':
                    values.append(fs.extraction_time)
                else:
                    values.append(0)
            
            ax.plot(timestamps, values, 'b-', linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel('Timestamp')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_feature_summary(self, feature_set: FeatureSet):
        """Print a summary of the extracted features"""
        if self.debug_level < 1:
            return
        
        counts = feature_set.get_feature_count_by_type()
        metrics = feature_set.quality_metrics
        
        print(f"\n{'='*60}")
        print(f"FEATURE EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Features: {counts['total']}")
        print(f"  - Sharp Edges: {counts['sharp_edges']}")
        print(f"  - Less Sharp Edges: {counts['less_sharp_edges']}")
        print(f"  - Planar Features: {counts['planar_features']}")
        print(f"  - Less Planar Features: {counts['less_planar_features']}")
        
        if metrics:
            print(f"\nQuality Metrics:")
            print(f"  - Overall Quality: {metrics['overall_quality']:.3f}")
            print(f"  - Spatial Distribution: {metrics['spatial_distribution_quality']:.3f}")
            print(f"  - Sector Balance: {metrics['sector_balance']:.3f}")
            print(f"  - Average Strength: {metrics['feature_strength_avg']:.3f}")
            
        print(f"Extraction Time: {feature_set.extraction_time:.2f} ms")
        print(f"{'='*60}\n")

class FeatureExtractor:
    """
    Main feature extraction class that coordinates all feature extraction components
    """
    
    def __init__(self, debug_level: int = 1,
                 curvature_window_size: int = 5,
                 num_sectors: int = 6,
                 sharp_edge_threshold: float = 0.1,
                 planar_threshold: float = 0.1,
                 max_sharp_edges_per_sector: int = 2,
                 max_less_sharp_per_sector: int = 20,
                 max_planar_per_sector: int = 4):
        """
        Initialize the feature extraction system
        
        Args:
            debug_level: Level of debug output (0=none, 1=basic, 2=detailed, 3=verbose)
            curvature_window_size: Window size for curvature calculation
            num_sectors: Number of sectors to divide scan into
            sharp_edge_threshold: Curvature threshold for edge features
            planar_threshold: Curvature threshold for planar features
            max_sharp_edges_per_sector: Maximum sharp edge features per sector
            max_less_sharp_per_sector: Maximum less sharp features per sector
            max_planar_per_sector: Maximum planar features per sector
        """
        self.debug_level = debug_level
        
        # Initialize components
        self.curvature_calculator = CurvatureCalculator(
            window_size=curvature_window_size,
            max_range=11.9
        )
        
        self.feature_classifier = FeatureClassifier(
            num_sectors=num_sectors,
            sharp_edge_threshold=sharp_edge_threshold,
            planar_threshold=planar_threshold,
            max_sharp_edges_per_sector=max_sharp_edges_per_sector,
            max_less_sharp_per_sector=max_less_sharp_per_sector,
            max_planar_per_sector=max_planar_per_sector
        )
        
        self.feature_validator = FeatureValidator()
        
        self.feature_visualizer = FeatureVisualizer(debug_level=debug_level)
        
        # Feature extraction history
        self.feature_history = []
        self.extraction_stats = {
            'total_extractions': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'average_features_per_scan': 0.0
        }
        
        if self.debug_level > 0:
            print(f"[FeatureExtractor] Initialized with debug_level={debug_level}")
            print(f"[FeatureExtractor] Curvature window size: {curvature_window_size}")
            print(f"[FeatureExtractor] Number of sectors: {num_sectors}")
            print(f"[FeatureExtractor] Thresholds - Edge: {sharp_edge_threshold}, Planar: {planar_threshold}")
    
    def extract_features(self, scan_x: List[float], scan_y: List[float], 
                        scan_ranges: List[float], robot_pose: Any = None,
                        scan_timestamp: float = 0.0) -> FeatureSet:
        """
        Extract features from a LiDAR scan
        
        Args:
            scan_x: X coordinates of scan points in world frame
            scan_y: Y coordinates of scan points in world frame
            scan_ranges: Range measurements
            robot_pose: Current robot pose (PoseEstimate object)
            scan_timestamp: Timestamp of the scan
            
        Returns:
            FeatureSet containing extracted features
        """
        start_time = time.time()
        
        # Calculate scan angles
        num_points = len(scan_ranges)
        angle_min = -math.pi / 2
        angle_max = math.pi / 2
        scan_angles = np.linspace(angle_min, angle_max, num_points)
        
        # Calculate curvatures for all points
        curvatures = self.curvature_calculator.calculate_curvature_array(scan_ranges, scan_angles)
        
        # Classify features
        feature_set = self.feature_classifier.classify_features(
            scan_x, scan_y, scan_ranges, curvatures, scan_angles, robot_pose
        )
        
        # Set timestamp
        feature_set.scan_timestamp = scan_timestamp
        
        # Validate and assess quality
        self.feature_validator.validate_feature_set(feature_set)
        
        # Record extraction time
        extraction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        feature_set.extraction_time = extraction_time
        
        # Update statistics
        self._update_stats(feature_set)
        
        # Add to history
        self.feature_history.append(feature_set)
        
        # Print summary if debugging
        if self.debug_level >= 2:
            self.feature_visualizer.print_feature_summary(feature_set)
        
        return feature_set
    
    def _update_stats(self, feature_set: FeatureSet):
        """Update extraction statistics"""
        self.extraction_stats['total_extractions'] += 1
        self.extraction_stats['total_time'] += feature_set.extraction_time
        self.extraction_stats['average_time'] = (
            self.extraction_stats['total_time'] / self.extraction_stats['total_extractions']
        )
        
        total_features = sum(len(fs.features) for fs in self.feature_history)
        self.extraction_stats['average_features_per_scan'] = (
            total_features / self.extraction_stats['total_extractions']
        )
    
    def visualize_extraction(self, feature_set: FeatureSet, scan_x: List[float], scan_y: List[float],
                           show_curvatures: bool = False, show_sectors: bool = False) -> plt.Figure:
        """
        Create visualization of feature extraction results
        
        Args:
            feature_set: FeatureSet to visualize
            scan_x, scan_y: Original scan points
            show_curvatures: Whether to show curvature values
            show_sectors: Whether to show sector boundaries
            
        Returns:
            Matplotlib figure with visualization
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        self.feature_visualizer.visualize_features(
            feature_set, scan_x, scan_y, ax, show_curvatures, show_sectors
        )
        
        plt.tight_layout()
        return fig
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about feature extraction performance"""
        if not self.feature_history:
            return {}
        
        stats = self.extraction_stats.copy()
        
        # Quality statistics
        quality_scores = [fs.quality_metrics.get('overall_quality', 0) for fs in self.feature_history]
        stats['average_quality'] = np.mean(quality_scores)
        stats['quality_std'] = np.std(quality_scores)
        
        # Feature count statistics
        feature_counts = [len(fs.features) for fs in self.feature_history]
        stats['min_features'] = min(feature_counts)
        stats['max_features'] = max(feature_counts)
        stats['feature_count_std'] = np.std(feature_counts)
        
        # Timing statistics
        extraction_times = [fs.extraction_time for fs in self.feature_history]
        stats['min_extraction_time'] = min(extraction_times)
        stats['max_extraction_time'] = max(extraction_times)
        stats['extraction_time_std'] = np.std(extraction_times)
        
        return stats
    
    def reset_history(self):
        """Reset feature extraction history and statistics"""
        self.feature_history = []
        self.extraction_stats = {
            'total_extractions': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'average_features_per_scan': 0.0
        }
        
        if self.debug_level > 0:
            print("[FeatureExtractor] History and statistics reset")
    
    def save_feature_data(self, filename: str, format: str = 'json'):
        """
        Save extracted features to file
        
        Args:
            filename: Output filename
            format: Save format ('json' or 'csv')
        """
        if not self.feature_history:
            print("No feature data to save.")
            return
        
        if format.lower() == 'json':
            self._save_features_json(filename)
        elif format.lower() == 'csv':
            self._save_features_csv(filename)
        else:
            print(f"Unsupported format: {format}")
    
    def _save_features_json(self, filename: str):
        """Save features in JSON format"""
        import json
        
        # Convert feature data to serializable format
        data = {
            'extraction_stats': self.extraction_stats,
            'feature_sets': []
        }
        
        for fs in self.feature_history:
            fs_data = {
                'timestamp': fs.scan_timestamp,
                'extraction_time': fs.extraction_time,
                'quality_metrics': fs.quality_metrics,
                'features': []
            }
            
            for feature in fs.features:
                feature_data = {
                    'point_world': feature.point_world.tolist(),
                    'point_local': feature.point_local.tolist(),
                    'feature_type': feature.feature_type.value,
                    'curvature': feature.curvature,
                    'strength': feature.strength,
                    'distance': feature.distance,
                    'angle': feature.angle,
                    'sector': feature.sector
                }
                fs_data['features'].append(feature_data)
            
            data['feature_sets'].append(fs_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Feature data saved to {filename}")
    
    def _save_features_csv(self, filename: str):
        """Save features in CSV format"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'timestamp', 'feature_type', 'world_x', 'world_y', 
                'local_x', 'local_y', 'curvature', 'strength', 
                'distance', 'angle', 'sector'
            ])
            
            # Write feature data
            for fs in self.feature_history:
                for feature in fs.features:
                    writer.writerow([
                        fs.scan_timestamp,
                        feature.feature_type.value,
                        feature.point_world[0],
                        feature.point_world[1],
                        feature.point_local[0],
                        feature.point_local[1],
                        feature.curvature,
                        feature.strength,
                        feature.distance,
                        feature.angle,
                        feature.sector
                    ])
        
        print(f"Feature data saved to {filename}")
        
    def visualize_all_features(self) -> plt.Figure:
        """
        Create visualization showing all features from recent scans
        
        Returns:
            Matplotlib figure with visualization
        """
        if not self.feature_history:
            print("No feature history available for visualization")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Extraction Summary', fontsize=16)
        
        # Get recent feature sets (last 5 or all if less than 5)
        recent_features = self.feature_history[-5:]
        
        # Plot 1: Feature count over time
        ax1 = axes[0, 0]
        feature_counts = [len(fs.features) for fs in self.feature_history]
        ax1.plot(range(len(feature_counts)), feature_counts, 'b-', linewidth=2)
        ax1.set_title('Feature Count Over Time')
        ax1.set_xlabel('Scan Number')
        ax1.set_ylabel('Feature Count')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature types distribution
        ax2 = axes[0, 1]
        total_sharp = sum(len(fs.sharp_edges) for fs in recent_features)
        total_less_sharp = sum(len(fs.less_sharp_edges) for fs in recent_features)
        total_planar = sum(len(fs.planar_features) for fs in recent_features)
        total_less_planar = sum(len(fs.less_planar_features) for fs in recent_features)
        
        labels = ['Sharp Edges', 'Less Sharp', 'Planar', 'Less Planar']
        sizes = [total_sharp, total_less_sharp, total_planar, total_less_planar]
        colors = ['red', 'orange', 'blue', 'lightblue']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Feature Type Distribution')
        
        # Plot 3: Most recent scan features
        ax3 = axes[1, 0]
        if self.feature_history:
            latest_fs = self.feature_history[-1]
            if hasattr(latest_fs, 'robot_pose') and latest_fs.robot_pose:
                # Plot features by type
                for feature_type in [FeatureType.SHARP_EDGE, FeatureType.LESS_SHARP_EDGE, 
                                FeatureType.PLANAR, FeatureType.LESS_PLANAR]:
                    features = latest_fs.get_features_by_type(feature_type)
                    if features:
                        x_coords = [f.point_world[0] for f in features]
                        y_coords = [f.point_world[1] for f in features]
                        
                        colors_map = {
                            FeatureType.SHARP_EDGE: 'red',
                            FeatureType.LESS_SHARP_EDGE: 'orange', 
                            FeatureType.PLANAR: 'blue',
                            FeatureType.LESS_PLANAR: 'lightblue'
                        }
                        sizes_map = {
                            FeatureType.SHARP_EDGE: 50,
                            FeatureType.LESS_SHARP_EDGE: 30,
                            FeatureType.PLANAR: 40, 
                            FeatureType.LESS_PLANAR: 10
                        }
                        
                        ax3.scatter(x_coords, y_coords, 
                                c=colors_map[feature_type], 
                                s=sizes_map[feature_type],
                                label=feature_type.value, alpha=0.7)
                
                ax3.set_title('Latest Scan Features')
                ax3.set_xlabel('X (meters)')
                ax3.set_ylabel('Y (meters)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_aspect('equal')
        
        # Plot 4: Extraction statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create statistics text
        stats = self.get_extraction_statistics()
        stats_text = f"""EXTRACTION STATISTICS
        
    Total Extractions: {stats.get('total_extractions', 0)}
    Average Features/Scan: {stats.get('average_features_per_scan', 0):.1f}
    Average Time: {stats.get('average_time', 0):.2f} ms

    Quality Metrics:
    Average Quality: {stats.get('average_quality', 0):.3f}
    Min Features: {stats.get('min_features', 0)}
    Max Features: {stats.get('max_features', 0)}

    Timing:
    Min Time: {stats.get('min_extraction_time', 0):.2f} ms  
    Max Time: {stats.get('max_extraction_time', 0):.2f} ms
    """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
        
# =========================================================================================================    
# Testing functions for Feature Extractor module

import os
import sys
import argparse
from typing import List, Dict, Tuple

# Import your existing utility functions
# Assuming these are available in the same directory
try:
    from lidar_utility_functions import parse_lidar_data, convert_scans_to_cartesian, read_lidar_data_from_file
    from ScanMatcher import PoseEstimate
except ImportError:
    print("Warning: Could not import utility functions. Some tests may not work.")
    print("Make sure lidar_utility_functions.py and ScanMatcher.py are in the same directory.")

def test_feature_extraction_on_dataset(file_path: str, max_entries: int = 50, 
                                      debug_level: int = 2) -> Dict[str, Any]:
    """
    Test feature extraction on your existing LiDAR dataset
    
    Args:
        file_path: Path to your LiDAR data file
        max_entries: Maximum number of scans to process
        debug_level: Debug output level
        
    Returns:
        Dictionary containing test results and statistics
    """
    print(f"Testing Feature Extraction on Dataset: {file_path}")
    print(f"Max entries: {max_entries}, Debug level: {debug_level}")
    print("="*80)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return {}
    
    # Read LiDAR data
    try:
        parsed_data_list = read_lidar_data_from_file(file_path, max_entries)
    except Exception as e:
        print(f"Error reading data file: {e}")
        return {}
    
    if not parsed_data_list:
        print("No data was read from the file.")
        return {}
    
    print(f"Successfully loaded {len(parsed_data_list)} scans")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(debug_level=debug_level)
    
    # LiDAR scan parameters (from your existing setup)
    angle_min = -math.pi/2
    angle_max = math.pi/2
    
    # Process each scan
    results = {
        'successful_extractions': 0,
        'failed_extractions': 0,
        'total_features_extracted': 0,
        'average_extraction_time': 0.0,
        'quality_scores': [],
        'feature_counts': [],
        'extraction_times': [],
        'feature_sets': [],
        'test_scans': []
    }
    
    print(f"\nProcessing {len(parsed_data_list)} scans...")
    
    for i, scan_data in enumerate(parsed_data_list):
        try:
            # Convert scan to Cartesian coordinates (using your existing function)
            scan_x, scan_y = convert_scans_to_cartesian(
                scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
                flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
            )
            
            # Create pose estimate
            pose = PoseEstimate(
                scan_data['pose']['x'],
                scan_data['pose']['y'],
                scan_data['pose']['theta']
            )
            
            # Extract features
            feature_set = feature_extractor.extract_features(
                scan_x, scan_y, scan_data['scan_ranges'], 
                robot_pose=pose, scan_timestamp=scan_data['timestamp']
            )
            
            # Record results
            results['successful_extractions'] += 1
            results['total_features_extracted'] += len(feature_set.features)
            results['quality_scores'].append(feature_set.quality_metrics.get('overall_quality', 0))
            results['feature_counts'].append(len(feature_set.features))
            results['extraction_times'].append(feature_set.extraction_time)
            results['feature_sets'].append(feature_set)
            results['test_scans'].append({
                'scan_x': scan_x,
                'scan_y': scan_y,
                'scan_ranges': scan_data['scan_ranges'],
                'pose': pose,
                'index': i
            })
            
            # Progress indicator
            if (i + 1) % 10 == 0 or debug_level >= 2:
                print(f"  Processed scan {i+1}/{len(parsed_data_list)} - "
                      f"Features: {len(feature_set.features)}, "
                      f"Quality: {feature_set.quality_metrics.get('overall_quality', 0):.3f}, "
                      f"Time: {feature_set.extraction_time:.2f}ms")
            
        except Exception as e:
            print(f"  Error processing scan {i+1}: {e}")
            results['failed_extractions'] += 1
            continue
    
    # Calculate summary statistics
    if results['successful_extractions'] > 0:
        results['average_features_per_scan'] = results['total_features_extracted'] / results['successful_extractions']
        results['average_extraction_time'] = np.mean(results['extraction_times'])
        results['average_quality_score'] = np.mean(results['quality_scores'])
        results['min_extraction_time'] = min(results['extraction_times'])
        results['max_extraction_time'] = max(results['extraction_times'])
        results['std_extraction_time'] = np.std(results['extraction_times'])
        results['min_feature_count'] = min(results['feature_counts'])
        results['max_feature_count'] = max(results['feature_counts'])
        results['std_feature_count'] = np.std(results['feature_counts'])
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"FEATURE EXTRACTION TEST RESULTS")
    print(f"{'='*80}")
    print(f"Successful extractions: {results['successful_extractions']}")
    print(f"Failed extractions: {results['failed_extractions']}")
    print(f"Success rate: {results['successful_extractions']/(results['successful_extractions']+results['failed_extractions'])*100:.1f}%")
    
    if results['successful_extractions'] > 0:
        print(f"\nFeature Statistics:")
        print(f"  Total features extracted: {results['total_features_extracted']}")
        print(f"  Average features per scan: {results['average_features_per_scan']:.1f}")
        print(f"  Feature count range: {results['min_feature_count']} - {results['max_feature_count']}")
        print(f"  Feature count std: {results['std_feature_count']:.1f}")
        
        print(f"\nPerformance Statistics:")
        print(f"  Average extraction time: {results['average_extraction_time']:.2f} ms")
        print(f"  Extraction time range: {results['min_extraction_time']:.2f} - {results['max_extraction_time']:.2f} ms")
        print(f"  Extraction time std: {results['std_extraction_time']:.2f} ms")
        
        print(f"\nQuality Statistics:")
        print(f"  Average quality score: {results['average_quality_score']:.3f}")
        print(f"  Quality score range: {min(results['quality_scores']):.3f} - {max(results['quality_scores']):.3f}")
        print(f"  Quality score std: {np.std(results['quality_scores']):.3f}")
    
    print(f"{'='*80}")
    
    return results

def test_individual_components():
    """
    Test individual components of the feature extraction system
    """
    print("Testing Individual Components")
    print("="*50)
    
    # Test 1: Curvature Calculator
    print("\n1. Testing CurvatureCalculator...")
    curvature_calc = CurvatureCalculator(window_size=5)
    
    # Create synthetic scan data
    num_points = 180
    angles = np.linspace(-math.pi/2, math.pi/2, num_points)
    
    # Create a scan with some obvious features
    ranges = []
    for angle in angles:
        if abs(angle) < 0.1:  # Straight ahead - wall
            ranges.append(5.0)
        elif abs(angle - 0.5) < 0.05 or abs(angle + 0.5) < 0.05:  # Corners
            ranges.append(3.0)
        else:
            ranges.append(5.0 + 0.5 * math.sin(angle * 4))  # Wavy wall
    
    curvatures = curvature_calc.calculate_curvature_array(ranges, angles)
    print(f"  Calculated curvatures for {len(curvatures)} points")
    print(f"  Curvature range: {min(curvatures):.4f} - {max(curvatures):.4f}")
    print(f"  Average curvature: {np.mean(curvatures):.4f}")
    
    # Test 2: Feature Classifier
    print("\n2. Testing FeatureClassifier...")
    classifier = FeatureClassifier()
    
    # Convert ranges to x,y coordinates
    scan_x = [r * math.cos(a) for r, a in zip(ranges, angles)]
    scan_y = [r * math.sin(a) for r, a in zip(ranges, angles)]
    
    # Create dummy pose
    pose = PoseEstimate(0, 0, 0)
    
    feature_set = classifier.classify_features(
        scan_x, scan_y, ranges, curvatures, angles, pose
    )
    
    counts = feature_set.get_feature_count_by_type()
    print(f"  Extracted features:")
    for feature_type, count in counts.items():
        print(f"    {feature_type}: {count}")
    
    # Test 3: Feature Validator
    print("\n3. Testing FeatureValidator...")
    validator = FeatureValidator()
    
    metrics = validator.validate_feature_set(feature_set)
    print(f"  Quality metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"    {metric}: {value:.3f}")
        else:
            print(f"    {metric}: {value}")
    
    print("\nComponent testing completed successfully!")
    return True

def test_parameter_sensitivity(file_path: str, num_scans: int = 10):
    """
    Test sensitivity to different parameter settings
    
    Args:
        file_path: Path to LiDAR data file
        num_scans: Number of scans to test with
    """
    print("Testing Parameter Sensitivity")
    print("="*50)
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    # Load a few scans for testing
    parsed_data_list = read_lidar_data_from_file(file_path, num_scans)
    if not parsed_data_list:
        print("No data loaded.")
        return
    
    # Test different parameter combinations
    test_configs = [
        # Format: (window_size, num_sectors, sharp_threshold, planar_threshold, name)
        (3, 6, 0.05, 0.05, "Conservative"),
        (5, 6, 0.1, 0.1, "Default"),
        (7, 6, 0.15, 0.15, "Aggressive"),
        (5, 4, 0.1, 0.1, "Fewer Sectors"),
        (5, 8, 0.1, 0.1, "More Sectors"),
    ]
    
    results = {}
    
    # LiDAR parameters
    angle_min = -math.pi/2
    angle_max = math.pi/2
    
    for window_size, num_sectors, sharp_thresh, planar_thresh, config_name in test_configs:
        print(f"\nTesting configuration: {config_name}")
        print(f"  Window size: {window_size}, Sectors: {num_sectors}")
        print(f"  Thresholds - Sharp: {sharp_thresh}, Planar: {planar_thresh}")
        
        # Initialize feature extractor with test parameters
        feature_extractor = FeatureExtractor(
            debug_level=0,
            curvature_window_size=window_size,
            num_sectors=num_sectors,
            sharp_edge_threshold=sharp_thresh,
            planar_threshold=planar_thresh
        )
        
        config_results = {
            'feature_counts': [],
            'extraction_times': [],
            'quality_scores': []
        }
        
        # Process test scans
        for scan_data in parsed_data_list:
            try:
                # Convert scan
                scan_x, scan_y = convert_scans_to_cartesian(
                    scan_data['scan_ranges'], angle_min, angle_max, scan_data['pose'],
                    flip_x=False, flip_y=False, reverse_scan=True, flip_theta=False
                )
                
                pose = PoseEstimate(
                    scan_data['pose']['x'],
                    scan_data['pose']['y'],
                    scan_data['pose']['theta']
                )
                
                # Extract features
                feature_set = feature_extractor.extract_features(
                    scan_x, scan_y, scan_data['scan_ranges'], pose
                )
                
                config_results['feature_counts'].append(len(feature_set.features))
                config_results['extraction_times'].append(feature_set.extraction_time)
                config_results['quality_scores'].append(
                    feature_set.quality_metrics.get('overall_quality', 0)
                )
                
            except Exception as e:
                print(f"    Error processing scan: {e}")
                continue
        
        # Calculate statistics
        if config_results['feature_counts']:
            avg_features = np.mean(config_results['feature_counts'])
            avg_time = np.mean(config_results['extraction_times'])
            avg_quality = np.mean(config_results['quality_scores'])
            
            print(f"  Results:")
            print(f"    Average features: {avg_features:.1f}")
            print(f"    Average time: {avg_time:.2f} ms")
            print(f"    Average quality: {avg_quality:.3f}")
            
            results[config_name] = {
                'avg_features': avg_features,
                'avg_time': avg_time,
                'avg_quality': avg_quality,
                'std_features': np.std(config_results['feature_counts']),
                'std_time': np.std(config_results['extraction_times']),
                'std_quality': np.std(config_results['quality_scores'])
            }
    
    # Print comparison
    print(f"\n{'='*80}")
    print("PARAMETER SENSITIVITY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Config':<15} {'Features':<10} {'Time(ms)':<10} {'Quality':<10}")
    print(f"{'-'*45}")
    
    for config_name, stats in results.items():
        print(f"{config_name:<15} {stats['avg_features']:<10.1f} {stats['avg_time']:<10.2f} {stats['avg_quality']:<10.3f}")
    
    return results

def visualize_all_features_global(results: Dict[str, Any], output_dir: str = "feature_test_output"):
    """
    Create a comprehensive visualization showing ALL extracted features from ALL scans
    
    Args:
        results: Results from test_feature_extraction_on_dataset
        output_dir: Directory to save visualizations
    """
    if not results or not results.get('feature_sets'):
        print("No results to visualize.")
        return
    
    print("Creating global feature visualization...")
    
    feature_sets = results['feature_sets']
    test_scans = results['test_scans']
    
    # Collect all features and trajectory data
    all_features = {
        'sharp_edges': [],
        'less_sharp_edges': [],
        'planar_features': [],
        'less_planar_features': []
    }
    
    robot_trajectory_x = []
    robot_trajectory_y = []
    scan_timestamps = []
    
    print(f"  Collecting features from {len(feature_sets)} scans...")
    
    for i, (feature_set, scan_data) in enumerate(zip(feature_sets, test_scans)):
        # Collect robot trajectory
        robot_trajectory_x.append(scan_data['pose'].x)
        robot_trajectory_y.append(scan_data['pose'].y)
        scan_timestamps.append(feature_set.scan_timestamp)
        
        # Collect features by type
        for feature_type_name in all_features.keys():
            if feature_type_name == 'sharp_edges':
                features = feature_set.sharp_edges
            elif feature_type_name == 'less_sharp_edges':
                features = feature_set.less_sharp_edges
            elif feature_type_name == 'planar_features':
                features = feature_set.planar_features
            else:  # less_planar_features
                features = feature_set.less_planar_features
            
            for feature in features:
                # Add scan index and timestamp for temporal analysis
                feature_data = {
                    'x': feature.point_world[0],
                    'y': feature.point_world[1],
                    'curvature': feature.curvature,
                    'strength': feature.strength,
                    'distance': feature.distance,
                    'scan_index': i,
                    'timestamp': feature_set.scan_timestamp,
                    'robot_x': scan_data['pose'].x,
                    'robot_y': scan_data['pose'].y
                }
                all_features[feature_type_name].append(feature_data)
    
    # Create multiple comprehensive visualizations
    fig = plt.figure(figsize=(24, 16))
    
    # Define colors and sizes for feature types
    feature_colors = {
        'sharp_edges': 'red',
        'less_sharp_edges': 'orange', 
        'planar_features': 'blue',
        'less_planar_features': 'lightblue'
    }
    
    feature_sizes = {
        'sharp_edges': 30,
        'less_sharp_edges': 20,
        'planar_features': 25,
        'less_planar_features': 8
    }
    
    # 1. Main global view - all features with trajectory
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot robot trajectory
    ax1.plot(robot_trajectory_x, robot_trajectory_y, 'g-', linewidth=2, alpha=0.7, label='Robot Trajectory')
    
    # Plot all features by type
    total_features = 0
    for feature_type, features in all_features.items():
        if features:
            x_coords = [f['x'] for f in features]
            y_coords = [f['y'] for f in features]
            
            ax1.scatter(x_coords, y_coords,
                       c=feature_colors[feature_type],
                       s=feature_sizes[feature_type],
                       alpha=0.7,
                       label=f"{feature_type.replace('_', ' ').title()} ({len(features)})")
            total_features += len(features)
    
    # Mark start and end positions
    if robot_trajectory_x:
        ax1.scatter(robot_trajectory_x[0], robot_trajectory_y[0], 
                   c='green', s=150, marker='*', edgecolors='black', linewidth=2, label='Start')
        ax1.scatter(robot_trajectory_x[-1], robot_trajectory_y[-1], 
                   c='red', s=150, marker='*', edgecolors='black', linewidth=2, label='End')
    
    ax1.set_title(f'Global Feature Map\n({total_features} features from {len(feature_sets)} scans)')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_aspect('equal')
    
    # 2. Features colored by curvature strength
    ax2 = plt.subplot(2, 3, 2)
    
    # Plot trajectory
    ax2.plot(robot_trajectory_x, robot_trajectory_y, 'g-', linewidth=2, alpha=0.7, label='Robot Trajectory')
    
    # Collect all features for curvature visualization
    all_x = []
    all_y = []
    all_curvatures = []
    all_sizes = []
    
    for feature_type, features in all_features.items():
        for feature in features:
            all_x.append(feature['x'])
            all_y.append(feature['y'])
            all_curvatures.append(feature['curvature'])
            all_sizes.append(feature_sizes[feature_type])
    
    if all_x:
        scatter = ax2.scatter(all_x, all_y, c=all_curvatures, s=all_sizes,
                            cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax2, label='Curvature')
    
    ax2.set_title('Features by Curvature Strength')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Temporal evolution (features colored by time)
    ax3 = plt.subplot(2, 3, 3)
    
    # Plot trajectory
    ax3.plot(robot_trajectory_x, robot_trajectory_y, 'g-', linewidth=2, alpha=0.7, label='Robot Trajectory')
    
    # Collect features with timestamps
    all_timestamps = []
    for feature_type, features in all_features.items():
        for feature in features:
            all_timestamps.append(feature['timestamp'])
    
    if all_x and all_timestamps:
        # Normalize timestamps for color mapping
        min_time = min(all_timestamps)
        time_range = max(all_timestamps) - min_time
        normalized_times = [(t - min_time) / max(time_range, 1) for t in all_timestamps]
        
        scatter = ax3.scatter(all_x, all_y, c=normalized_times, s=all_sizes,
                            cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, ax=ax3, label='Time (normalized)')
    
    ax3.set_title('Feature Extraction Timeline')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Feature density heatmap
    ax4 = plt.subplot(2, 3, 4)
    
    if all_x:
        # Create 2D histogram for feature density
        hist, xedges, yedges = np.histogram2d(all_x, all_y, bins=50)
        
        # Plot heatmap
        im = ax4.imshow(hist.T, origin='lower', 
                       extent=[min(all_x), max(all_x), min(all_y), max(all_y)],
                       cmap='hot', alpha=0.7)
        plt.colorbar(im, ax=ax4, label='Feature Density')
        
        # Overlay trajectory
        ax4.plot(robot_trajectory_x, robot_trajectory_y, 'g-', linewidth=2, alpha=0.9)
    
    ax4.set_title('Feature Density Heatmap')
    ax4.set_xlabel('X (meters)')
    ax4.set_ylabel('Y (meters)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Feature distribution by distance from robot
    ax5 = plt.subplot(2, 3, 5)
    
    # Plot trajectory
    ax5.plot(robot_trajectory_x, robot_trajectory_y, 'g-', linewidth=2, alpha=0.7, label='Robot Trajectory')
    
    # Color features by distance from robot when detected
    all_distances = []
    for feature_type, features in all_features.items():
        for feature in features:
            all_distances.append(feature['distance'])
    
    if all_x and all_distances:
        scatter = ax5.scatter(all_x, all_y, c=all_distances, s=all_sizes,
                            cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, ax=ax5, label='Distance from Robot (m)')
    
    ax5.set_title('Features by Detection Distance')
    ax5.set_xlabel('X (meters)')
    ax5.set_ylabel('Y (meters)')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Statistics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create text summary
    stats_text = f"""GLOBAL FEATURE EXTRACTION SUMMARY
    
Total Scans Processed: {len(feature_sets)}
Total Features Extracted: {total_features}
Average Features per Scan: {total_features/len(feature_sets):.1f}

Feature Type Breakdown:
• Sharp Edges: {len(all_features['sharp_edges'])} ({len(all_features['sharp_edges'])/total_features*100:.1f}%)
• Less Sharp Edges: {len(all_features['less_sharp_edges'])} ({len(all_features['less_sharp_edges'])/total_features*100:.1f}%)
• Planar Features: {len(all_features['planar_features'])} ({len(all_features['planar_features'])/total_features*100:.1f}%)
• Less Planar Features: {len(all_features['less_planar_features'])} ({len(all_features['less_planar_features'])/total_features*100:.1f}%)

Trajectory Statistics:
• Path Length: {len(robot_trajectory_x)} poses
• Start: ({robot_trajectory_x[0]:.2f}, {robot_trajectory_y[0]:.2f})
• End: ({robot_trajectory_x[-1]:.2f}, {robot_trajectory_y[-1]:.2f})

Performance:
• Avg Extraction Time: {results.get('average_extraction_time', 0):.2f} ms
• Avg Quality Score: {results.get('average_quality_score', 0):.3f}
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_test_visualizations(results: Dict[str, Any], output_dir: str = "feature_test_output"):
    """
    Create comprehensive visualizations of test results
    
    Args:
        results: Results from test_feature_extraction_on_dataset
        output_dir: Directory to save visualizations
    """
    if not results or not results.get('feature_sets'):
        print("No results to visualize.")
        return
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Creating visualizations in {output_dir}...")
    
    feature_sets = results['feature_sets']
    test_scans = results['test_scans']
    
    # Initialize visualizer
    visualizer = FeatureVisualizer(debug_level=1)
    
    # 1. Create GLOBAL feature visualization showing ALL features
    print("  Creating global feature map visualization...")
    global_fig = visualize_all_features_global(results, output_dir)
    if global_fig:
        plt.figure(global_fig.number)
        plt.savefig(f'{output_dir}/global_feature_map.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Create individual feature extraction visualizations (first 5 scans)
    print("  Creating individual scan visualizations...")
    for i in range(min(5, len(feature_sets))):
        feature_set = feature_sets[i]
        scan_data = test_scans[i]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Left plot: Features with types
        visualizer.visualize_features(
            feature_set, scan_data['scan_x'], scan_data['scan_y'], 
            ax=axes[0], show_curvatures=False, show_sectors=True
        )
        axes[0].set_title(f'Scan {i+1}: Feature Classification')
        
        # Right plot: Features with curvature colors
        visualizer.visualize_features(
            feature_set, scan_data['scan_x'], scan_data['scan_y'], 
            ax=axes[1], show_curvatures=True, show_sectors=False
        )
        axes[1].set_title(f'Scan {i+1}: Curvature Visualization')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scan_{i+1}_features.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Create quality metrics over time
    print("  Creating quality metrics visualization...")
    fig = visualizer.plot_quality_metrics(feature_sets)
    if fig:
        plt.savefig(f'{output_dir}/quality_metrics_over_time.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Create performance summary plots
    print("  Creating performance summary...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Feature counts histogram
    axes[0, 0].hist(results['feature_counts'], bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Feature Count Distribution')
    axes[0, 0].set_xlabel('Number of Features')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Extraction times histogram
    axes[0, 1].hist(results['extraction_times'], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('Extraction Time Distribution')
    axes[0, 1].set_xlabel('Extraction Time (ms)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quality scores histogram
    axes[1, 0].hist(results['quality_scores'], bins=20, alpha=0.7, color='orange')
    axes[1, 0].set_title('Quality Score Distribution')
    axes[1, 0].set_xlabel('Quality Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature counts vs quality scatter
    axes[1, 1].scatter(results['feature_counts'], results['quality_scores'], alpha=0.6)
    axes[1, 1].set_title('Feature Count vs Quality Score')
    axes[1, 1].set_xlabel('Number of Features')
    axes[1, 1].set_ylabel('Quality Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Create feature type distribution analysis
    print("  Creating feature type analysis...")
    feature_type_counts = {
        'sharp_edges': [],
        'less_sharp_edges': [],
        'planar_features': [],
        'less_planar_features': []
    }
    
    for fs in feature_sets:
        counts = fs.get_feature_count_by_type()
        for ftype, count in feature_type_counts.items():
            count.append(counts.get(ftype, 0))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(feature_sets))
    width = 0.2
    
    ax.bar(x - 1.5*width, feature_type_counts['sharp_edges'], width, label='Sharp Edges', alpha=0.8)
    ax.bar(x - 0.5*width, feature_type_counts['less_sharp_edges'], width, label='Less Sharp Edges', alpha=0.8)
    ax.bar(x + 0.5*width, feature_type_counts['planar_features'], width, label='Planar Features', alpha=0.8)
    ax.bar(x + 1.5*width, feature_type_counts['less_planar_features'], width, label='Less Planar Features', alpha=0.8)
    
    ax.set_xlabel('Scan Number')
    ax.set_ylabel('Feature Count')
    ax.set_title('Feature Type Distribution Across Scans')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_type_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualizations saved to {output_dir}/")
    print(f"    - global_feature_map.png: Complete feature map from all scans")
    print(f"    - scan_X_features.png: Individual scan visualizations") 
    print(f"    - quality_metrics_over_time.png: Quality analysis")
    print(f"    - performance_summary.png: Performance distributions")
    print(f"    - feature_type_distribution.png: Feature type analysis")

def run_comprehensive_test(file_path: str, max_entries: int = 50, create_visualizations: bool = True):
    """
    Run comprehensive test suite for feature extraction
    
    Args:
        file_path: Path to LiDAR data file
        max_entries: Maximum number of scans to process
        create_visualizations: Whether to create visualization outputs
    """
    print("FEATURE EXTRACTION COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Test 1: Individual components
    print("\nPhase 1: Testing Individual Components")
    test_individual_components()
    
    # Test 2: Main dataset testing
    print(f"\nPhase 2: Testing on Dataset ({file_path})")
    results = test_feature_extraction_on_dataset(file_path, max_entries, debug_level=1)
    
    if not results:
        print("Dataset testing failed. Stopping comprehensive test.")
        return
    
    # Test 3: Parameter sensitivity
    print(f"\nPhase 3: Parameter Sensitivity Analysis")
    param_results = test_parameter_sensitivity(file_path, num_scans=min(10, max_entries))
    
    # Test 4: Create visualizations
    if create_visualizations:
        print(f"\nPhase 4: Creating Visualizations")
        print("  - Global feature map showing ALL features from ALL scans")
        print("  - Individual scan feature overlays") 
        print("  - Quality metrics over time")
        print("  - Performance distribution analysis")
        create_test_visualizations(results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST COMPLETED")
    print(f"{'='*80}")
    
    if results['successful_extractions'] > 0:
        print(f"✓ Successfully processed {results['successful_extractions']} scans")
        print(f"✓ Average {results['average_features_per_scan']:.1f} features per scan")
        print(f"✓ Average extraction time: {results['average_extraction_time']:.2f} ms")
        print(f"✓ Average quality score: {results['average_quality_score']:.3f}")
        
        # Performance assessment
        if results['average_extraction_time'] <= 15.0:
            print("✓ Performance: EXCELLENT (≤15ms per scan)")
        elif results['average_extraction_time'] <= 25.0:
            print("⚠ Performance: GOOD (15-25ms per scan)")
        else:
            print("⚠ Performance: NEEDS OPTIMIZATION (>25ms per scan)")
        
        # Feature count assessment
        if 10 <= results['average_features_per_scan'] <= 100:
            print("✓ Feature count: OPTIMAL (10-100 features per scan)")
        elif results['average_features_per_scan'] < 10:
            print("⚠ Feature count: LOW (<10 features per scan)")
        else:
            print("⚠ Feature count: HIGH (>100 features per scan)")
        
        # Quality assessment
        if results['average_quality_score'] >= 0.7:
            print("✓ Quality: EXCELLENT (≥0.7)")
        elif results['average_quality_score'] >= 0.5:
            print("✓ Quality: GOOD (0.5-0.7)")
        elif results['average_quality_score'] >= 0.3:
            print("⚠ Quality: ACCEPTABLE (0.3-0.5)")
        else:
            print("⚠ Quality: NEEDS IMPROVEMENT (<0.3)")
    
    print(f"{'='*80}")
    
    return results

def main():
    """
    Main function for standalone testing
    """
    parser = argparse.ArgumentParser(description='Feature Extraction Standalone Testing')
    
    parser.add_argument('--file', type=str, 
                       default="../dataset/raw_data/laser_data_synchronized_short_u_turn_fast_processed_reduced180.clf",
                       help='Path to the LiDAR data file')
    parser.add_argument('--max_entries', type=int, default=2000,
                       help='Maximum number of entries to process')
    parser.add_argument('--test_type', type=str, default='comprehensive',
                       choices=['components', 'dataset', 'parameters', 'comprehensive'],
                       help='Type of test to run')
    parser.add_argument('--debug_level', type=int, default=1, choices=[0, 1, 2, 3],
                       help='Debug output level')
    parser.add_argument('--visualizations', action='store_true', default=True,
                       help='Create visualization outputs')
    parser.add_argument('--output_dir', type=str, default='feature_test_output',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print("Feature Extraction Module - Standalone Testing")
    print(f"Data file: {args.file}")
    print(f"Max entries: {args.max_entries}")
    print(f"Test type: {args.test_type}")
    print(f"Debug level: {args.debug_level}")
    print()
    
    if args.test_type == 'components':
        test_individual_components()
    
    elif args.test_type == 'dataset':
        results = test_feature_extraction_on_dataset(args.file, args.max_entries, args.debug_level)
        if args.visualizations and results:
            create_test_visualizations(results, args.output_dir)
    
    elif args.test_type == 'parameters':
        test_parameter_sensitivity(args.file, min(20, args.max_entries))
    
    elif args.test_type == 'comprehensive':
        run_comprehensive_test(args.file, args.max_entries, args.visualizations)
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main()