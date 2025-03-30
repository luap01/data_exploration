import json
import numpy as np
import cv2

with open("../HaMuCo/data/OR/calib/camera01.json", "r") as f:
    calib = json.load(f)

# Use color parameters for OpenPose images
color_params = calib["value0"]["color_parameters"]

# Build the intrinsic matrix (K)
K = np.array([
    [color_params["fov_x"], 0, color_params["c_x"]],
    [0, color_params["fov_y"], color_params["c_y"]],
    [0, 0, 1]
], dtype=np.float32)

# Build distortion coefficients vector.
# Here we assume the order: [k1, k2, p1, p2, k3]
# Based on your JSON:
radial = color_params["radial_distortion"]
tangential = color_params["tangential_distortion"]

distCoeffs = np.array([
    radial["m00"],       # k1
    radial["m10"],       # k2
    tangential["m00"],   # p1
    tangential["m10"],   # p2
    radial["m20"]        # k3
], dtype=np.float32)


# Assume openpose_keypoints is an (N, 2) numpy array of detected keypoints
# For example:
with open("./json/camera01_000000_keypoints.json", "r") as f:
    data = json.load(f)


left_openpose_keypoints = np.array(data['people'][0]['hand_right_keypoints_2d'])
right_openpose_keypoints = np.array(data['people'][0]['hand_right_keypoints_2d'])

# cv2.undistortPoints expects points in shape (N, 1, 2)
left_points = left_openpose_keypoints.reshape(-1, 1, 2)
right_points = right_openpose_keypoints.reshape(-1, 1, 2)

# Undistort points. Passing P=K returns them in pixel coordinates.
left_undistorted_points = cv2.undistortPoints(left_points, K, distCoeffs, P=K)
right_undistorted_points = cv2.undistortPoints(right_points, K, distCoeffs, P=K)

# Reshape back to (N, 2)
undistorted_points = left_undistorted_points.reshape(-1, 2)
print("Undistorted keypoints:\n", undistorted_points)
