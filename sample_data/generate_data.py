import numpy as np
import math
from datetime import datetime

def generate_circular_path():
    # Start with a base timestamp
    base_timestamp = 1031745824.958
    
    # Initial pose (x, y, theta) - start near center
    # Assuming image dimensions are roughly 800x600 pixels
    # Scale down to reasonable robot coordinates (e.g., 10x10 meters)
    x, y, theta = 5.0, 5.0, 0.0
    
    # Parameters for motion
    dt = 0.01  # Time step between measurements
    v = 0.2   # Forward velocity (m/s)
    radius = 2.0  # Radius of the circular path
    
    # Calculate angular velocity for a complete circle
    # Time to complete circle = 2*pi*radius / v
    # Angular velocity = 2*pi / time
    omega = v / radius  # Angular velocity for circular motion
    
    # Generate circular motion
    data_lines = []
    total_time = 2 * math.pi * radius / v  # Time to complete one circle
    num_steps = int(total_time / dt)
    
    for i in range(num_steps):
        # Calculate position on circle
        angle = omega * i * dt
        x = 5.0 + radius * math.cos(angle)  # Center at (5,5)
        y = 5.0 + radius * math.sin(angle)
        theta = angle + math.pi/2  # Robot faces tangent to circle
        
        # Add small random variations to make it more realistic
        x += np.random.normal(0, 0.02)
        y += np.random.normal(0, 0.02)
        current_v = v + np.random.normal(0, 0.01)
        
        # Current timestamp
        timestamp = base_timestamp + i * dt
        
        # Generate ODOM line
        odom_line = f"ODOM {x:.3f} {y:.3f} {theta:.6f} {current_v:.3f} {omega:.3f} 0 {timestamp:.3f} iB21 {timestamp-base_timestamp:.3f}"
        data_lines.append(odom_line)
        
        # Generate FLASER line with 180 measurements
        laser_ranges = []
        for angle_idx in range(180):
            # Generate realistic-looking laser readings
            base_range = 1.5 + 0.5 * math.cos(angle_idx * math.pi / 180)
            range_noise = np.random.normal(0, 0.05)
            laser_range = max(0.5, base_range + range_noise)
            laser_ranges.append(f"{laser_range:.3f}")
        
        # Construct FLASER line
        laser_line = f"FLASER 180 {' '.join(laser_ranges)} {x:.3f} {y:.3f} {theta:.6f} {x:.3f} {y:.3f} {theta:.6f} {timestamp+0.02:.3f} iB21 {timestamp-base_timestamp+0.02:.3f}"
        data_lines.append(laser_line)
    
    return data_lines

# Generate and save the sample data
sample_data = generate_circular_path()

# Save to file
with open('complex_path.clf', 'w') as f:
    for line in sample_data:
        f.write(line + '\n')

print("Circular path dataset generated and saved to 'complex_path.clf'")

# Print first few lines as an example
print("\nFirst few lines of the generated dataset:")
for line in sample_data[:4]:
    print(line)