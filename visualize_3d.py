import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def load_points(json_path):
    with open(json_path, 'r') as f:
        points = json.load(f)
    return np.array(points)

def create_table_plane():
    # Create a larger plane to represent the table
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    
    # Create grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                        np.linspace(y_min, y_max, 20))
    
    # Flat plane at z=0 to represent table surface
    zz = np.zeros_like(xx)
    
    return xx, yy, zz

def plot_hand_connections(ax, points, color):
    # Define hand keypoint connections
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (0, 5), (5, 9), (9, 13), (13, 17)
    ]
    
    # Draw lines between connected keypoints
    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            start_point = points[start_idx]
            end_point = points[end_idx]
            ax.plot([start_point[0], end_point[0]],
                   [start_point[1], end_point[1]],
                   [start_point[2], end_point[2]],
                   color=color, linewidth=2, alpha=0.8)

def main():
    # Load points from both JSON files
    left_points = load_points('data/output/xyz/left/000001.json')
    right_points = load_points('data/output/xyz/right/000001.json')
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot table plane
    xx, yy, zz = create_table_plane()
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')
    
    # Plot reference grid on the table
    for x in np.linspace(-2, 2, 5):
        ax.plot([x, x], [-2, 2], [0, 0], 'b-', alpha=0.2)
    for y in np.linspace(-2, 2, 5):
        ax.plot([-2, 2], [y, y], [0, 0], 'b-', alpha=0.2)
    
    # Plot points and connections
    ax.scatter(left_points[:, 0], left_points[:, 1], left_points[:, 2], 
              c='blue', marker='o', label='Left Hand', s=50)
    plot_hand_connections(ax, left_points, 'blue')
    
    ax.scatter(right_points[:, 0], right_points[:, 1], right_points[:, 2], 
              c='red', marker='o', label='Right Hand', s=50)
    plot_hand_connections(ax, right_points, 'red')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Hand Keypoints with Reference Table')
    ax.legend()
    
    # Add grid on walls for better 3D reference
    ax.grid(True)
    
    # Set consistent scale for all axes
    ax.set_box_aspect([1, 1, 1])
    
    # Adjust the view for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Show plot
    plt.show()

if __name__ == '__main__':
    main() 