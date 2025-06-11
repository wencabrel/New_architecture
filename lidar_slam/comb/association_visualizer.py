import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
import math
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import os

# Import existing components
try:
    from feature_extractor import FeatureSet, LiDARFeature, FeatureType
    from ScanMatcher import PoseEstimate
    from feature_association import FeatureDescriptor, AssociationScore, FeatureAssociationEngine
    from association_validator import ValidationResult, AssociationValidator
    from hybrid_pose_estimator import PoseEstimateWithConfidence, EnvironmentClassifier, HybridPoseEstimator
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    print("Warning: Some dependencies not available. Association visualization will have limited functionality.")
    DEPENDENCIES_AVAILABLE = False


@dataclass
class VisualizationConfig:
    """Configuration for association visualization"""
    
    # Figure settings
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 100
    
    # Color scheme
    current_feature_color: str = 'blue'
    previous_feature_color: str = 'red'
    association_line_color: str = 'green'
    validated_association_color: str = 'darkgreen'
    rejected_association_color: str = 'orange'
    
    # Feature visualization
    feature_sizes: Dict[str, int] = None
    feature_colors: Dict[str, str] = None
    
    # Association visualization
    show_all_associations: bool = True
    show_only_validated: bool = False
    association_line_alpha: float = 0.7
    association_line_width: float = 1.5
    
    # Quality visualization
    show_quality_metrics: bool = True
    show_confidence_bounds: bool = True
    
    # Performance settings
    max_features_to_plot: int = 200
    max_associations_to_plot: int = 100
    
    def __post_init__(self):
        if self.feature_sizes is None:
            self.feature_sizes = {
                'sharp_edge': 40,
                'less_sharp_edge': 25,
                'planar': 30,
                'less_planar': 15
            }
        
        if self.feature_colors is None:
            self.feature_colors = {
                'sharp_edge': 'red',
                'less_sharp_edge': 'orange',
                'planar': 'blue',
                'less_planar': 'lightblue'
            }


class AssociationVisualizer:
    """
    Comprehensive visualization and debugging tools for feature association
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None, 
                 save_plots: bool = False, output_dir: str = "association_debug"):
        """
        Initialize the association visualizer
        
        Args:
            config: Visualization configuration
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
        """
        self.config = config if config else VisualizationConfig()
        self.save_plots = save_plots
        self.output_dir = output_dir
        
        # Create output directory if saving plots
        if self.save_plots and not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
                print(f"[AssociationVisualizer] Created output directory: {self.output_dir}")
            except Exception as e:
                print(f"[AssociationVisualizer] Warning: Could not create output directory: {e}")
                self.save_plots = False
        
        # Storage for visualization data
        self.visualization_history = []
        self.performance_history = []
        
        print(f"[AssociationVisualizer] Initialized with save_plots={save_plots}")
    
    def visualize_associations(self, current_descriptors: List["FeatureDescriptor"],
                             previous_descriptors: List["FeatureDescriptor"],
                             associations: List["AssociationScore"],
                             validation_result: Optional["ValidationResult"] = None,
                             title: str = "Feature Associations") -> plt.Figure:
        """
        Visualize feature associations between two scans
        
        Args:
            current_descriptors: Current scan feature descriptors
            previous_descriptors: Previous scan feature descriptors
            associations: Feature associations
            validation_result: Validation result (optional)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not DEPENDENCIES_AVAILABLE:
            print("Warning: Dependencies not available for association visualization")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Limit features to plot for performance
        current_to_plot = current_descriptors[:self.config.max_features_to_plot]
        previous_to_plot = previous_descriptors[:self.config.max_features_to_plot]
        associations_to_plot = associations[:self.config.max_associations_to_plot]
        
        # Plot previous features
        self._plot_features(ax, previous_to_plot, 'previous', alpha=0.6)
        
        # Plot current features
        self._plot_features(ax, current_to_plot, 'current', alpha=0.8)
        
        # Plot associations
        self._plot_associations(ax, current_to_plot, previous_to_plot, 
                              associations_to_plot, validation_result)
        
        # Add robot positions if available
        self._add_robot_positions(ax, current_to_plot, previous_to_plot)
        
        # Add validation information
        if validation_result:
            self._add_validation_info(ax, validation_result)
        
        # Add association statistics
        self._add_association_statistics(ax, associations, validation_result)
        
        # Formatting
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f"{title}\n{len(associations)} associations, "
                    f"{len(current_to_plot)} current features, {len(previous_to_plot)} previous features")
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        # Save if requested
        if self.save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/associations_{timestamp}.png"
            try:
                plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
                print(f"[AssociationVisualizer] Saved association plot: {filename}")
            except Exception as e:
                print(f"[AssociationVisualizer] Warning: Could not save plot: {e}")
        
        return fig
    
    def _plot_features(self, ax: plt.Axes, descriptors: List["FeatureDescriptor"], 
                      scan_type: str, alpha: float = 0.8):
        """
        Plot features grouped by type
        
        Args:
            ax: Matplotlib axis
            descriptors: Feature descriptors to plot
            scan_type: 'current' or 'previous'
            alpha: Transparency level
        """
        # Group features by type
        feature_groups = {}
        for desc in descriptors:
            if desc.base_feature is None:
                continue
            
            feature_type = desc.base_feature.feature_type.value
            if feature_type not in feature_groups:
                feature_groups[feature_type] = []
            feature_groups[feature_type].append(desc)
        
        # Plot each feature type
        for feature_type, features in feature_groups.items():
            if not features:
                continue
            
            x_coords = [f.base_feature.point_world[0] for f in features]
            y_coords = [f.base_feature.point_world[1] for f in features]
            
            # Get color and size
            color = self.config.feature_colors.get(feature_type, 'gray')
            size = self.config.feature_sizes.get(feature_type, 20)
            
            # Adjust color for scan type
            if scan_type == 'current':
                color = color  # Keep original color
                marker = 'o'
                label = f"Current {feature_type.replace('_', ' ').title()} ({len(features)})"
            else:
                color = 'lightcoral' if 'edge' in feature_type else 'lightblue'
                marker = 's'  # Square for previous
                label = f"Previous {feature_type.replace('_', ' ').title()} ({len(features)})"
            
            ax.scatter(x_coords, y_coords, c=color, s=size, alpha=alpha, 
                      marker=marker, label=label, edgecolors='black', linewidth=0.5)
    
    def _plot_associations(self, ax: plt.Axes, current_descriptors: List["FeatureDescriptor"],
                          previous_descriptors: List["FeatureDescriptor"],
                          associations: List["AssociationScore"],
                          validation_result: Optional["ValidationResult"]):
        """
        Plot association lines between features
        
        Args:
            ax: Matplotlib axis
            current_descriptors: Current feature descriptors
            previous_descriptors: Previous feature descriptors
            associations: Feature associations
            validation_result: Validation result
        """
        if not associations:
            return
        
        # Determine which associations to show
        validated_indices = set()
        if validation_result and validation_result.inlier_associations:
            validated_indices = set(validation_result.inlier_associations)
        
        # Plot associations
        valid_lines = []
        invalid_lines = []
        
        for i, assoc in enumerate(associations):
            # Check bounds
            if (assoc.feature_idx1 >= len(current_descriptors) or 
                assoc.feature_idx2 >= len(previous_descriptors)):
                continue
            
            curr_desc = current_descriptors[assoc.feature_idx1]
            prev_desc = previous_descriptors[assoc.feature_idx2]
            
            if curr_desc.base_feature is None or prev_desc.base_feature is None:
                continue
            
            # Get positions
            curr_pos = curr_desc.base_feature.point_world
            prev_pos = prev_desc.base_feature.point_world
            
            # Create line
            line = [(prev_pos[0], prev_pos[1]), (curr_pos[0], curr_pos[1])]
            
            # Categorize line
            is_validated = (validation_result is None or  # Show all if no validation
                          i in validated_indices or
                          assoc.validated)
            
            if is_validated:
                valid_lines.append(line)
            else:
                invalid_lines.append(line)
        
        # Plot valid associations
        if valid_lines and (self.config.show_all_associations or self.config.show_only_validated):
            valid_collection = LineCollection(
                valid_lines, 
                colors=self.config.validated_association_color,
                linewidths=self.config.association_line_width,
                alpha=self.config.association_line_alpha,
                label=f'Validated Associations ({len(valid_lines)})'
            )
            ax.add_collection(valid_collection)
        
        # Plot invalid associations
        if invalid_lines and self.config.show_all_associations and not self.config.show_only_validated:
            invalid_collection = LineCollection(
                invalid_lines,
                colors=self.config.rejected_association_color,
                linewidths=self.config.association_line_width * 0.5,
                alpha=self.config.association_line_alpha * 0.5,
                linestyles='dashed',
                label=f'Rejected Associations ({len(invalid_lines)})'
            )
            ax.add_collection(invalid_collection)
    
    def _add_robot_positions(self, ax: plt.Axes, current_descriptors: List["FeatureDescriptor"],
                           previous_descriptors: List["FeatureDescriptor"]):
        """
        Add robot position markers if available
        
        Args:
            ax: Matplotlib axis
            current_descriptors: Current feature descriptors
            previous_descriptors: Previous feature descriptors
        """
        # Try to get robot positions from descriptors
        current_pose = None
        previous_pose = None
        
        if current_descriptors and hasattr(current_descriptors[0], 'base_feature'):
            # Estimate robot position (simplified - would use actual pose in practice)
            positions = [d.base_feature.point_world for d in current_descriptors if d.base_feature]
            if positions:
                current_pose = np.mean(positions, axis=0)
        
        if previous_descriptors and hasattr(previous_descriptors[0], 'base_feature'):
            positions = [d.base_feature.point_world for d in previous_descriptors if d.base_feature]
            if positions:
                previous_pose = np.mean(positions, axis=0)
        
        # Plot robot positions
        if current_pose is not None:
            ax.scatter(current_pose[0], current_pose[1], c='blue', s=150, 
                      marker='*', label='Current Robot', edgecolors='black', linewidth=2)
        
        if previous_pose is not None:
            ax.scatter(previous_pose[0], previous_pose[1], c='red', s=150, 
                      marker='*', label='Previous Robot', edgecolors='black', linewidth=2)
        
        # Draw arrow showing motion if both positions available
        if current_pose is not None and previous_pose is not None:
            motion_arrow = FancyArrowPatch(
                previous_pose, current_pose,
                arrowstyle='->', mutation_scale=20, color='purple',
                alpha=0.7, linewidth=2, label='Robot Motion'
            )
            ax.add_patch(motion_arrow)
    
    def _add_validation_info(self, ax: plt.Axes, validation_result: "ValidationResult"):
        """
        Add validation information to the plot
        
        Args:
            ax: Matplotlib axis
            validation_result: Validation result
        """
        # Add validation info text box
        validation_text = (
            f"Validation: {'PASSED' if validation_result.is_valid else 'FAILED'}\n"
            f"Confidence: {validation_result.confidence:.3f}\n"
            f"Inliers: {validation_result.inlier_count}/{validation_result.total_associations}\n"
            f"Geometric: {validation_result.geometric_consistency:.3f}\n"
            f"Temporal: {validation_result.temporal_consistency:.3f}"
        )
        
        # Color based on validation result
        box_color = 'lightgreen' if validation_result.is_valid else 'lightcoral'
        
        ax.text(0.02, 0.98, validation_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.8))
    
    def _add_association_statistics(self, ax: plt.Axes, associations: List["AssociationScore"],
                                  validation_result: Optional["ValidationResult"]):
        """
        Add association statistics to the plot
        
        Args:
            ax: Matplotlib axis
            associations: Feature associations
            validation_result: Validation result
        """
        if not associations:
            return
        
        # Calculate statistics
        scores = [a.score for a in associations]
        distances = [a.distance for a in associations]
        
        stats_text = (
            f"Association Statistics:\n"
            f"Count: {len(associations)}\n"
            f"Avg Score: {np.mean(scores):.3f}\n"
            f"Avg Distance: {np.mean(distances):.3f}m\n"
            f"Score Range: {min(scores):.3f} - {max(scores):.3f}"
        )
        
        if validation_result:
            stats_text += f"\nRANSAC Iterations: {validation_result.ransac_iterations}"
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                verticalalignment='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def visualize_association_quality_over_time(self, association_history: List[Dict[str, Any]],
                                              title: str = "Association Quality Over Time") -> plt.Figure:
        """
        Visualize association quality metrics over time
        
        Args:
            association_history: List of association data with timestamps
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not association_history:
            print("No association history available for visualization")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title)
        
        # Extract time series data
        timestamps = []
        association_counts = []
        average_scores = []
        validation_rates = []
        inlier_ratios = []
        
        for entry in association_history:
            timestamps.append(entry.get('timestamp', 0))
            association_counts.append(entry.get('association_count', 0))
            average_scores.append(entry.get('average_score', 0))
            validation_rates.append(entry.get('validation_success', 0))
            inlier_ratios.append(entry.get('inlier_ratio', 0))
        
        # Plot 1: Association count over time
        axes[0, 0].plot(timestamps, association_counts, 'b-', linewidth=2)
        axes[0, 0].set_title('Association Count Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Number of Associations')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average association score
        axes[0, 1].plot(timestamps, average_scores, 'g-', linewidth=2)
        axes[0, 1].set_title('Average Association Score')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Plot 3: Validation success rate
        axes[1, 0].plot(timestamps, validation_rates, 'r-', linewidth=2)
        axes[1, 0].set_title('Validation Success Rate')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: Inlier ratio
        axes[1, 1].plot(timestamps, inlier_ratios, 'm-', linewidth=2)
        axes[1, 1].set_title('RANSAC Inlier Ratio')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Inlier Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save if requested
        if self.save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/quality_over_time_{timestamp}.png"
            try:
                plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
                print(f"[AssociationVisualizer] Saved quality plot: {filename}")
            except Exception as e:
                print(f"[AssociationVisualizer] Warning: Could not save plot: {e}")
        
        return fig
    
    def visualize_environment_classification(self, classification_history: List[Dict[str, float]],
                                           title: str = "Environment Classification Over Time") -> plt.Figure:
        """
        Visualize environment classification over time
        
        Args:
            classification_history: List of environment classification dictionaries
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not classification_history:
            print("No environment classification history available")
            return None
        
        # Extract environment types
        all_env_types = set()
        for classification in classification_history:
            all_env_types.update(classification.keys())
        
        env_types = sorted(list(all_env_types))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title)
        
        # Plot 1: Stacked area chart of environment classifications
        time_indices = range(len(classification_history))
        
        # Prepare data for stacking
        env_data = {}
        for env_type in env_types:
            env_data[env_type] = [classification.get(env_type, 0) for classification in classification_history]
        
        # Create stacked area plot
        bottom = np.zeros(len(classification_history))
        colors = plt.cm.Set3(np.linspace(0, 1, len(env_types)))
        
        for i, env_type in enumerate(env_types):
            ax1.fill_between(time_indices, bottom, bottom + env_data[env_type], 
                           label=env_type.replace('_', ' ').title(), alpha=0.7, color=colors[i])
            bottom += env_data[env_type]
        
        ax1.set_title('Environment Classification Over Time')
        ax1.set_xlabel('Scan Number')
        ax1.set_ylabel('Classification Score')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Dominant environment over time
        dominant_environments = []
        for classification in classification_history:
            if classification:
                dominant = max(classification.items(), key=lambda x: x[1])[0]
                dominant_environments.append(dominant)
            else:
                dominant_environments.append('unknown')
        
        # Convert to numeric for plotting
        unique_envs = list(set(dominant_environments))
        env_to_num = {env: i for i, env in enumerate(unique_envs)}
        numeric_envs = [env_to_num[env] for env in dominant_environments]
        
        ax2.plot(time_indices, numeric_envs, 'o-', linewidth=2, markersize=4)
        ax2.set_title('Dominant Environment Type')
        ax2.set_xlabel('Scan Number')
        ax2.set_ylabel('Environment Type')
        ax2.set_yticks(range(len(unique_envs)))
        ax2.set_yticklabels([env.replace('_', ' ').title() for env in unique_envs])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if self.save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/environment_classification_{timestamp}.png"
            try:
                plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
                print(f"[AssociationVisualizer] Saved environment plot: {filename}")
            except Exception as e:
                print(f"[AssociationVisualizer] Warning: Could not save plot: {e}")
        
        return fig
    
    def visualize_hybrid_pose_performance(self, pose_history: List["PoseEstimateWithConfidence"],
                                        title: str = "Hybrid Pose Estimation Performance") -> plt.Figure:
        """
        Visualize hybrid pose estimation performance
        
        Args:
            pose_history: List of pose estimates with confidence
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if not pose_history:
            print("No pose history available for visualization")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(title)
        
        # Extract data
        timestamps = list(range(len(pose_history)))
        confidences = [p.confidence for p in pose_history]
        sources = [p.source.value for p in pose_history]
        feature_weights = [getattr(p, 'feature_weight', 0.5) for p in pose_history]
        processing_times = [p.processing_time for p in pose_history]
        position_uncertainties = [p.position_uncertainty for p in pose_history]
        
        # Plot 1: Confidence over time
        axes[0, 0].plot(timestamps, confidences, 'b-', linewidth=2)
        axes[0, 0].set_title('Pose Confidence Over Time')
        axes[0, 0].set_xlabel('Scan Number')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Pose source distribution
        unique_sources = list(set(sources))
        source_counts = [sources.count(source) for source in unique_sources]
        axes[0, 1].pie(source_counts, labels=[s.replace('_', ' ').title() for s in unique_sources], 
                      autopct='%1.1f%%')
        axes[0, 1].set_title('Pose Source Distribution')
        
        # Plot 3: Feature vs ICP weighting
        axes[0, 2].plot(timestamps, feature_weights, 'g-', linewidth=2, label='Feature Weight')
        axes[0, 2].plot(timestamps, [1-w for w in feature_weights], 'r-', linewidth=2, label='ICP Weight')
        axes[0, 2].set_title('Feature vs ICP Weighting')
        axes[0, 2].set_xlabel('Scan Number')
        axes[0, 2].set_ylabel('Weight')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
        
        # Plot 4: Processing time
        axes[1, 0].plot(timestamps, processing_times, 'purple', linewidth=2)
        axes[1, 0].set_title('Processing Time')
        axes[1, 0].set_xlabel('Scan Number')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Position uncertainty
        axes[1, 1].plot(timestamps, position_uncertainties, 'orange', linewidth=2)
        axes[1, 1].set_title('Position Uncertainty')
        axes[1, 1].set_xlabel('Scan Number')
        axes[1, 1].set_ylabel('Uncertainty (m)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Confidence vs uncertainty correlation
        axes[1, 2].scatter(confidences, position_uncertainties, alpha=0.6)
        axes[1, 2].set_title('Confidence vs Uncertainty')
        axes[1, 2].set_xlabel('Confidence')
        axes[1, 2].set_ylabel('Position Uncertainty (m)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if self.save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/hybrid_pose_performance_{timestamp}.png"
            try:
                plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
                print(f"[AssociationVisualizer] Saved pose performance plot: {filename}")
            except Exception as e:
                print(f"[AssociationVisualizer] Warning: Could not save plot: {e}")
        
        return fig
    
    def create_comprehensive_debug_plot(self, current_descriptors: List["FeatureDescriptor"],
                                      previous_descriptors: List["FeatureDescriptor"],
                                      associations: List["AssociationScore"],
                                      validation_result: "ValidationResult",
                                      feature_pose: "PoseEstimateWithConfidence",
                                      icp_pose: "PoseEstimateWithConfidence",
                                      hybrid_pose: "PoseEstimateWithConfidence",
                                      environment_classification: Dict[str, float]) -> plt.Figure:
        """
        Create a comprehensive debug plot showing all aspects of the association process
        
        Args:
            current_descriptors: Current scan descriptors
            previous_descriptors: Previous scan descriptors
            associations: Feature associations
            validation_result: Validation result
            feature_pose: Feature-based pose estimate
            icp_pose: ICP-based pose estimate
            hybrid_pose: Final hybrid pose estimate
            environment_classification: Environment classification
            
        Returns:
            Matplotlib figure with comprehensive debug information
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Main association plot (large subplot)
        ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        
        # Plot features and associations
        self._plot_features(ax_main, previous_descriptors, 'previous', alpha=0.6)
        self._plot_features(ax_main, current_descriptors, 'current', alpha=0.8)
        self._plot_associations(ax_main, current_descriptors, previous_descriptors, 
                              associations, validation_result)
        self._add_robot_positions(ax_main, current_descriptors, previous_descriptors)
        
        ax_main.set_aspect('equal')
        ax_main.grid(True, alpha=0.3)
        ax_main.set_xlabel('X (meters)')
        ax_main.set_ylabel('Y (meters)')
        ax_main.set_title(f'Feature Associations ({len(associations)} total)')
        ax_main.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        
        # Validation info subplot
        ax_val = plt.subplot2grid((3, 4), (0, 2))
        ax_val.axis('off')
        validation_text = (
            f"VALIDATION RESULTS\n"
            f"{'='*20}\n"
            f"Status: {'PASSED' if validation_result.is_valid else 'FAILED'}\n"
            f"Confidence: {validation_result.confidence:.3f}\n"
            f"Inliers: {validation_result.inlier_count}/{validation_result.total_associations}\n"
            f"Geometric: {validation_result.geometric_consistency:.3f}\n"
            f"Temporal: {validation_result.temporal_consistency:.3f}\n"
            f"RANSAC Iterations: {validation_result.ransac_iterations}\n"
            f"Method: {validation_result.validation_method}\n"
        )
        ax_val.text(0.05, 0.95, validation_text, transform=ax_val.transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='lightgreen' if validation_result.is_valid else 'lightcoral',
                            alpha=0.8))
        
        # Pose comparison subplot
        ax_pose = plt.subplot2grid((3, 4), (1, 2))
        ax_pose.axis('off')
        pose_text = (
            f"POSE ESTIMATES\n"
            f"{'='*15}\n"
            f"Feature-based:\n"
            f"  x: {feature_pose.pose.x:.3f}m\n"
            f"  y: {feature_pose.pose.y:.3f}m\n"
            f"  θ: {feature_pose.pose.theta:.3f}rad\n"
            f"  conf: {feature_pose.confidence:.3f}\n\n"
            f"ICP-based:\n"
            f"  x: {icp_pose.pose.x:.3f}m\n"
            f"  y: {icp_pose.pose.y:.3f}m\n"
            f"  θ: {icp_pose.pose.theta:.3f}rad\n"
            f"  conf: {icp_pose.confidence:.3f}\n\n"
            f"Hybrid:\n"
            f"  x: {hybrid_pose.pose.x:.3f}m\n"
            f"  y: {hybrid_pose.pose.y:.3f}m\n"
            f"  θ: {hybrid_pose.pose.theta:.3f}rad\n"
            f"  conf: {hybrid_pose.confidence:.3f}\n"
        )
        
        if hasattr(hybrid_pose, 'feature_weight'):
            pose_text += f"  feat_w: {hybrid_pose.feature_weight:.3f}\n"
        
        ax_pose.text(0.05, 0.95, pose_text, transform=ax_pose.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Environment classification subplot
        ax_env = plt.subplot2grid((3, 4), (0, 3))
        env_types = list(environment_classification.keys())
        env_scores = list(environment_classification.values())
        
        bars = ax_env.bar(range(len(env_types)), env_scores, alpha=0.7)
        ax_env.set_xticks(range(len(env_types)))
        ax_env.set_xticklabels([t.replace('_', '\n') for t in env_types], rotation=45, fontsize=8)
        ax_env.set_ylabel('Score')
        ax_env.set_title('Environment\nClassification')
        ax_env.set_ylim(0, 1)
        ax_env.grid(True, alpha=0.3)
        
        # Color bars by score
        for bar, score in zip(bars, env_scores):
            bar.set_color(plt.cm.viridis(score))
        
        # Association quality subplot
        ax_quality = plt.subplot2grid((3, 4), (1, 3))
        if associations:
            scores = [a.score for a in associations]
            distances = [a.distance for a in associations]
            
            ax_quality.scatter(distances, scores, alpha=0.6, s=30)
            ax_quality.set_xlabel('Distance (m)')
            ax_quality.set_ylabel('Association Score')
            ax_quality.set_title('Association\nQuality')
            ax_quality.grid(True, alpha=0.3)
        else:
            ax_quality.text(0.5, 0.5, 'No Associations', ha='center', va='center',
                           transform=ax_quality.transAxes)
            ax_quality.set_title('Association\nQuality')
        
        # Feature distribution subplot (bottom row)
        ax_dist = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        
        # Count features by type for both scans
        current_counts = {}
        previous_counts = {}
        
        for desc in current_descriptors:
            if desc.base_feature:
                ftype = desc.base_feature.feature_type.value
                current_counts[ftype] = current_counts.get(ftype, 0) + 1
        
        for desc in previous_descriptors:
            if desc.base_feature:
                ftype = desc.base_feature.feature_type.value
                previous_counts[ftype] = previous_counts.get(ftype, 0) + 1
        
        # Get all feature types
        all_types = set(list(current_counts.keys()) + list(previous_counts.keys()))
        all_types = sorted(list(all_types))
        
        x = np.arange(len(all_types))
        width = 0.35
        
        current_vals = [current_counts.get(t, 0) for t in all_types]
        previous_vals = [previous_counts.get(t, 0) for t in all_types]
        
        ax_dist.bar(x - width/2, current_vals, width, label='Current', alpha=0.7)
        ax_dist.bar(x + width/2, previous_vals, width, label='Previous', alpha=0.7)
        
        ax_dist.set_xlabel('Feature Type')
        ax_dist.set_ylabel('Count')
        ax_dist.set_title('Feature Distribution by Type')
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels([t.replace('_', '\n') for t in all_types])
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        # Performance metrics subplot
        ax_perf = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        ax_perf.axis('off')
        
        # Calculate performance metrics
        if associations:
            avg_score = np.mean([a.score for a in associations])
            avg_distance = np.mean([a.distance for a in associations])
            score_std = np.std([a.score for a in associations])
        else:
            avg_score = avg_distance = score_std = 0
        
        perf_text = (
            f"PERFORMANCE METRICS\n"
            f"{'='*25}\n"
            f"Total Associations: {len(associations)}\n"
            f"Average Score: {avg_score:.3f} ± {score_std:.3f}\n"
            f"Average Distance: {avg_distance:.3f}m\n"
            f"Validation Success: {validation_result.is_valid}\n"
            f"Processing Time: {hybrid_pose.processing_time:.2f}ms\n"
            f"Position Uncertainty: {hybrid_pose.position_uncertainty:.3f}m\n"
            f"Source: {hybrid_pose.source.value}\n"
        )
        
        ax_perf.text(0.05, 0.95, perf_text, transform=ax_perf.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if self.save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/comprehensive_debug_{timestamp}.png"
            try:
                plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
                print(f"[AssociationVisualizer] Saved comprehensive debug plot: {filename}")
            except Exception as e:
                print(f"[AssociationVisualizer] Warning: Could not save plot: {e}")
        
        return fig
    
    def save_association_data(self, filename: str, current_descriptors: List["FeatureDescriptor"],
                            previous_descriptors: List["FeatureDescriptor"],
                            associations: List["AssociationScore"],
                            validation_result: "ValidationResult"):
        """
        Save association data to file for later analysis
        
        Args:
            filename: Output filename
            current_descriptors: Current scan descriptors
            previous_descriptors: Previous scan descriptors
            associations: Feature associations
            validation_result: Validation result
        """
        import json
        
        # Prepare data for serialization
        data = {
            'timestamp': time.time(),
            'current_features': [],
            'previous_features': [],
            'associations': [],
            'validation': {}
        }
        
        # Current features
        for desc in current_descriptors:
            if desc.base_feature:
                data['current_features'].append({
                    'position': desc.base_feature.point_world.tolist(),
                    'type': desc.base_feature.feature_type.value,
                    'curvature': desc.base_feature.curvature,
                    'strength': desc.base_feature.strength,
                    'distance': desc.base_feature.distance
                })
        
        # Previous features
        for desc in previous_descriptors:
            if desc.base_feature:
                data['previous_features'].append({
                    'position': desc.base_feature.point_world.tolist(),
                    'type': desc.base_feature.feature_type.value,
                    'curvature': desc.base_feature.curvature,
                    'strength': desc.base_feature.strength,
                    'distance': desc.base_feature.distance
                })
        
        # Associations
        for assoc in associations:
            data['associations'].append({
                'current_idx': assoc.feature_idx1,
                'previous_idx': assoc.feature_idx2,
                'score': assoc.score,
                'distance': assoc.distance,
                'similarity': assoc.similarity,
                'confidence': assoc.confidence,
                'validated': assoc.validated
            })
        
        # Validation result
        data['validation'] = {
            'is_valid': validation_result.is_valid,
            'confidence': validation_result.confidence,
            'inlier_count': validation_result.inlier_count,
            'total_associations': validation_result.total_associations,
            'geometric_consistency': validation_result.geometric_consistency,
            'temporal_consistency': validation_result.temporal_consistency,
            'method': validation_result.validation_method,
            'ransac_iterations': validation_result.ransac_iterations
        }
        
        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[AssociationVisualizer] Saved association data: {filename}")
        except Exception as e:
            print(f"[AssociationVisualizer] Error saving association data: {e}")


# Utility functions for integration
def create_debug_visualizer(save_plots: bool = False, output_dir: str = "association_debug") -> AssociationVisualizer:
    """
    Create a configured association visualizer for debugging
    
    Args:
        save_plots: Whether to save plots
        output_dir: Output directory for plots
        
    Returns:
        AssociationVisualizer instance
    """
    config = VisualizationConfig()
    config.show_all_associations = True
    config.show_quality_metrics = True
    
    return AssociationVisualizer(config=config, save_plots=save_plots, output_dir=output_dir)


def quick_association_plot(current_features: FeatureSet, previous_features: FeatureSet,
                         associations: List["AssociationScore"],
                         validation_result: Optional["ValidationResult"] = None) -> plt.Figure:
    """
    Quick function to create an association visualization
    
    Args:
        current_features: Current feature set
        previous_features: Previous feature set
        associations: Feature associations
        validation_result: Validation result
        
    Returns:
        Matplotlib figure
    """
    if not DEPENDENCIES_AVAILABLE:
        print("Warning: Dependencies not available for quick association plot")
        return None
    
    # Create visualizer
    visualizer = AssociationVisualizer()
    
    # Create descriptors
    engine = FeatureAssociationEngine()
    current_descriptors = engine.create_descriptors(current_features, 1)
    previous_descriptors = engine.create_descriptors(previous_features, 0)
    
    # Create plot
    return visualizer.visualize_associations(
        current_descriptors, previous_descriptors, associations, validation_result
    )
