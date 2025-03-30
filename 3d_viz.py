import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


HAND_STRUCTURE = {
    "thumb": {
        "connections": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "color": (0, 255, 0)  # Green (BGR for OpenCV)
    },
    "index": {
        "connections": [(0, 5), (5, 6), (6, 7), (7, 8)],
        "color": (0, 0, 255)  # Red
    },
    "middle": {
        "connections": [(0, 9), (9, 10), (10, 11), (11, 12)],
        "color": (0, 255, 255)  # Yellow
    },
    "ring": {
        "connections": [(0, 13), (13, 14), (14, 15), (15, 16)],
        "color": (255, 0, 0)  # Blue
    },
    "little": {
        "connections": [(0, 17), (17, 18), (18, 19), (19, 20)],
        "color": (255, 0, 255)  # Purple
    }
}

def tf_color(color):
    if color == (0, 255, 0):
        return 'g'
    if color == (0, 0, 255):
        return 'r'
    if color == (0, 255, 255):
        return 'y'
    if color == (255, 0, 0):
        return 'b'
    if color == (255, 0, 255):
        return 'm'
    return 'k'


def visualize_3d_pose(left_coord_xyz, right_coord_xyz, rgb_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(left_coord_xyz[:, 0], left_coord_xyz[:, 1], left_coord_xyz[:, 2], c='b', marker='o')
    ax.scatter(right_coord_xyz[:, 0], right_coord_xyz[:, 1], right_coord_xyz[:, 2], c='g', marker='o')
    
    for _, data in HAND_STRUCTURE.items():
        color = data["color"]
        for (i, j) in data["connections"]:
            ax.plot([left_coord_xyz[i, 0], left_coord_xyz[j, 0]],
                [left_coord_xyz[i, 1], left_coord_xyz[j, 1]],
                [left_coord_xyz[i, 2], left_coord_xyz[j, 2]], tf_color(color))
            
    for _, data in HAND_STRUCTURE.items():
        color = data["color"]
        for (i, j) in data["connections"]:
            ax.plot([right_coord_xyz[i, 0], right_coord_xyz[j, 0]],
                [right_coord_xyz[i, 1], right_coord_xyz[j, 1]],
                [right_coord_xyz[i, 2], right_coord_xyz[j, 2]], tf_color(color))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(f"./output/graphs/{rgb_filename}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    # right = np.array(json_load("./right_3d.json"))
    # left = np.array(json_load("./left_3d.json"))
    right = np.array(json_load("./xyz/right/000000.json"))
    left = np.array(json_load("./xyz/left/000000.json"))
    print(left.shape)
    visualize_3d_pose(left, right, "test_other")


