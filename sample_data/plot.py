import numpy as np
import matplotlib.pyplot as plt

def plot_path_comparison(predicted_file_path, ground_truth_file_path):
    """
    Plot predicted path and ground truth path on the same plot for comparison.
    
    Args:
        predicted_file_path (str): Path to the predicted trajectory file
        ground_truth_file_path (str): Path to the ground truth CLF file
    """
    # Initialize empty lists for path data
    predicted_path = []
    ground_truth_path = []
    
    # Read predicted path data
    with open(predicted_file_path, 'r') as f:
        # Skip header if it exists
        first_line = f.readline()
        if not first_line.startswith('ODOM'):
            # File has header, start reading from next line
            for line in f:
                x, y = map(float, line.strip().split())
                predicted_path.append((x, y))
        else:
            # First line contains data
            x, y = map(float, first_line.strip().split()[1:3])  # Skip 'ODOM' and take next two values
            predicted_path.append((x, y))
            for line in f:
                x, y = map(float, line.strip().split()[1:3])
                predicted_path.append((x, y))
    
    # Read ground truth data
    with open(ground_truth_file_path, 'r') as f:
        for line in f:
            if line.startswith('ODOM'):
                parts = line.split()
                ground_truth_path.append((float(parts[1]), float(parts[2])))
    
    # Convert to numpy arrays for plotting
    predicted_path = np.array(predicted_path)
    ground_truth_path = np.array(ground_truth_path)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.plot(predicted_path[:, 0], predicted_path[:, 1], 'b-', label='Predicted Path', linewidth=2)
    plt.plot(ground_truth_path[:, 0], ground_truth_path[:, 1], 'r--', label='Ground Truth', linewidth=2)
    
    plt.title('Robot Path Comparison')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(True)
    plt.legend()
    
    # Make axes equal to preserve shape
    plt.axis('equal')
    
    return plt.gcf()  # Return the figure handle

# Example usage:
if __name__ == '__main__':
    # Create and show the plot
    fig = plot_path_comparison('../predicted_data/predicted_path.txt', '../robot_path_meters.clf')
    
    # Optional: Save the plot
    plt.savefig('path_comparison.png', dpi=300, bbox_inches='tight')
    
    # Display the plot
    plt.show()