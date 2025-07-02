import numpy as np
import math


class CornerTracker:
    """
    Track corners across multiple LiDAR scans to build confidence over time
    """
    def __init__(self, matching_threshold=0.3, max_history=20):
        """
        Initialize the corner tracker
        
        Args:
            matching_threshold: Maximum distance (meters) to match corners between frames
            max_history: Maximum number of frames to keep unmatched corners
        """
        self.corners = []  # List of tracked corners
        self.matching_threshold = matching_threshold
        self.max_history = max_history
    
    def update(self, new_corners):
        """
        Update tracked corners with new detections
        
        Args:
            new_corners: List of newly detected corners
        """
        # If no existing corners, simply add all new ones
        if not self.corners:
            self.corners = [{'x': c['x'], 'y': c['y'], 
                           'angle': c['angle'],
                           'confidence': c['confidence'],
                           'observations': 1,
                           'last_seen': 0} for c in new_corners]
            return
        
        # Match new corners with existing ones
        matched = [False] * len(new_corners)
        
        for i, existing in enumerate(self.corners):
            existing['last_seen'] += 1
            
            # Find closest new corner
            closest_idx = -1
            closest_dist = float('inf')
            
            for j, new_corner in enumerate(new_corners):
                if matched[j]:
                    continue
                
                dist = np.sqrt((existing['x'] - new_corner['x'])**2 + 
                              (existing['y'] - new_corner['y'])**2)
                
                if dist < closest_dist and dist < self.matching_threshold:
                    closest_dist = dist
                    closest_idx = j
            
            # If found a match, update existing corner
            if closest_idx >= 0:
                matched[closest_idx] = True
                
                # Weighted average of positions (more weight to established corners)
                weight = min(existing['observations'] / 10.0, 0.9)
                existing['x'] = weight * existing['x'] + (1-weight) * new_corners[closest_idx]['x']
                existing['y'] = weight * existing['y'] + (1-weight) * new_corners[closest_idx]['y']
                
                # Update angle (weighted average)
                angle_diff = new_corners[closest_idx]['angle'] - existing['angle']
                # Handle angle wrap-around
                if angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                elif angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                existing['angle'] = existing['angle'] + (1-weight) * angle_diff
                
                # Update confidence and observations
                existing['confidence'] = max(existing['confidence'], 
                                          new_corners[closest_idx]['confidence'])
                existing['observations'] += 1
                existing['last_seen'] = 0
        
        # Add unmatched new corners
        for j, new_corner in enumerate(new_corners):
            if not matched[j]:
                self.corners.append({
                    'x': new_corner['x'],
                    'y': new_corner['y'],
                    'angle': new_corner['angle'],
                    'confidence': new_corner['confidence'],
                    'observations': 1,
                    'last_seen': 0
                })
        
        # Remove corners that haven't been seen in a while
        self.corners = [c for c in self.corners if c['last_seen'] < self.max_history]

def detect_corners(scan_x, scan_y, robot_x, robot_y, min_angle=0.5, min_points=3, min_segment_length=0.1):
    """
    Detect corners in LiDAR scan data using line segment fitting
    
    Args:
        scan_x, scan_y: Scan point coordinates
        robot_x, robot_y: Robot position
        min_angle: Minimum angle (radians) between line segments to detect corner
        min_points: Minimum number of points to fit a line segment
        min_segment_length: Minimum length (meters) of a line segment
        
    Returns:
        corners: List of corner positions and confidence values
    """
    corners = []
    
    # Need at least enough points for multiple line segments
    if len(scan_x) < min_points * 2:
        return corners
    
    # Convert to numpy arrays for easier manipulation
    points_x = np.array(scan_x)
    points_y = np.array(scan_y)
    
    # Sort points based on angle from robot center (for contiguous line detection)
    angles = np.arctan2(points_y - robot_y, points_x - robot_x)
    sorted_indices = np.argsort(angles)
    sorted_x = points_x[sorted_indices]
    sorted_y = points_y[sorted_indices]
    
    # RANSAC-like line segment extraction
    remaining_points = np.ones(len(sorted_x), dtype=bool)
    segments = []
    
    # Keep extracting line segments until no points remain
    while np.sum(remaining_points) >= min_points:
        # Get current set of available points
        current_x = sorted_x[remaining_points]
        current_y = sorted_y[remaining_points]
        
        if len(current_x) < min_points:
            break
            
        # Try to fit a line to a subset of points
        best_line = None
        best_inliers = None
        best_inlier_count = 0
        
        # Try multiple random seeds for RANSAC
        for _ in range(min(10, len(current_x) // 2)):
            # Pick two random points to define a line
            sample_indices = np.random.choice(len(current_x), 2, replace=False)
            p1 = (current_x[sample_indices[0]], current_y[sample_indices[0]])
            p2 = (current_x[sample_indices[1]], current_y[sample_indices[1]])
            
            # Skip if points are too close
            if np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < min_segment_length:
                continue
                
            # Calculate line equation: ax + by + c = 0
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0]*p1[1] - p1[0]*p2[1]
            
            # Normalize
            norm = np.sqrt(a*a + b*b)
            if norm < 1e-6:
                continue
                
            a /= norm
            b /= norm
            c /= norm
            
            # Find inliers (points close to the line)
            distances = np.abs(a*current_x + b*current_y + c)
            inliers = distances < 0.05  # 5cm threshold for inliers
            
            # If we found a better line, save it
            if np.sum(inliers) > best_inlier_count:
                best_line = (a, b, c)
                best_inliers = inliers
                best_inlier_count = np.sum(inliers)
        
        # If we found a good line segment
        if best_line is not None and best_inlier_count >= min_points:
            # Get segment endpoints
            inlier_x = current_x[best_inliers]
            inlier_y = current_y[best_inliers]
            
            # Calculate segment length
            min_x, max_x = np.min(inlier_x), np.max(inlier_x)
            min_y, max_y = np.min(inlier_y), np.max(inlier_y)
            segment_length = np.sqrt((max_x-min_x)**2 + (max_y-min_y)**2)
            
            # Only keep segments of sufficient length
            if segment_length >= min_segment_length:
                # Calculate angle of the line segment
                segment_angle = np.arctan2(-best_line[0], best_line[1])
                
                # Calculate center of the segment
                center_x = np.mean(inlier_x)
                center_y = np.mean(inlier_y)
                
                # Store the line segment
                segments.append({
                    'a': best_line[0],
                    'b': best_line[1],
                    'c': best_line[2],
                    'angle': segment_angle,
                    'center_x': center_x,
                    'center_y': center_y,
                    'length': segment_length,
                    'points': [(x, y) for x, y in zip(inlier_x, inlier_y)]
                })
                
                # Remove inliers from remaining points
                temp_indices = np.where(remaining_points)[0]
                remaining_indices = temp_indices[~best_inliers]
                remaining_points = np.zeros_like(remaining_points)
                remaining_points[remaining_indices] = True
            else:
                # If segment is too short, remove a few points and try again
                temp_indices = np.where(remaining_points)[0]
                remaining_indices = temp_indices[~np.random.choice([True, False], size=len(temp_indices), p=[0.3, 0.7])]
                remaining_points = np.zeros_like(remaining_points)
                remaining_points[remaining_indices] = True
        else:
            # If no good line was found, exit the loop
            break
    
    # Find corners at the intersection of line segments
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            seg1 = segments[i]
            seg2 = segments[j]
            
            # Calculate angle between line segments
            angle_diff = abs(seg1['angle'] - seg2['angle'])
            # Normalize to [0, π/2]
            angle_diff = min(angle_diff, np.pi - angle_diff)
            
            # Only consider segments with a sufficient angle between them (potential corners)
            if angle_diff > min_angle:
                # Check if the segments are close to each other
                # Calculate distance between segment centers
                center_dist = np.sqrt((seg1['center_x'] - seg2['center_x'])**2 + 
                                   (seg1['center_y'] - seg2['center_y'])**2)
                
                # Only consider segments that are close to each other
                if center_dist < max(seg1['length'], seg2['length']) * 0.5:
                    # Calculate the intersection point
                    a1, b1, c1 = seg1['a'], seg1['b'], seg1['c']
                    a2, b2, c2 = seg2['a'], seg2['b'], seg2['c']
                    
                    # Check if lines are not parallel
                    det = a1*b2 - a2*b1
                    if abs(det) > 1e-6:
                        # Calculate intersection
                        x = (b1*c2 - b2*c1) / det
                        y = (a2*c1 - a1*c2) / det
                        
                        # Calculate distance from robot for confidence weighting
                        distance = np.sqrt((x - robot_x)**2 + (y - robot_y)**2)
                        
                        # Calculate confidence based on angle and distance
                        angle_confidence = min(1.0, angle_diff / (np.pi/2))  # 90° is optimal
                        distance_confidence = 1.0 / (1.0 + distance * 0.1)   # Closer is better
                        
                        # Calculate confidence based on whether the corner matches a visible scan point
                        match_confidence = 0.5
                        for px, py in zip(scan_x, scan_y):
                            point_dist = np.sqrt((x - px)**2 + (y - py)**2)
                            if point_dist < 0.2:  # 20cm distance threshold
                                match_confidence = 1.0
                                break
                                
                        # Combined confidence score
                        confidence = angle_confidence * distance_confidence * match_confidence
                        
                        corners.append({
                            'x': x,
                            'y': y,
                            'angle': angle_diff,
                            'confidence': confidence,
                            'seg1': i,
                            'seg2': j
                        })
    
    return corners