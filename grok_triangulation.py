import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# --- File Paths (Adapt these to your local files) ---
CALIB_FILES = [
    "../HaMuCo/data/OR/calib/camera01.json",
    "../HaMuCo/data/OR/calib/camera02.json",
    "../HaMuCo/data/OR/calib/camera03.json",
    "../HaMuCo/data/OR/calib/camera04.json"
]

OPENPOSE_FILES = [
    "./json/camera01_000000_keypoints.json",
    "./json/camera02_000000_keypoints.json",
    "./json/camera03_000000_keypoints.json",
    "./json/camera04_000000_keypoints.json"
]

# --- Helper Functions ---
def load_calibration(file_path):
    with open(file_path, 'r') as f:
        calib = json.load(f)["value0"]
    
    # Intrinsics (assuming depth_parameters for OpenPose keypoints)
    intrinsics = calib["depth_parameters"]["intrinsics_matrix"]
    K = np.array([
        [intrinsics["m00"], intrinsics["m10"], intrinsics["m20"]],
        [intrinsics["m01"], intrinsics["m11"], intrinsics["m21"]],
        [intrinsics["m02"], intrinsics["m12"], intrinsics["m22"]]
    ], dtype=np.float32)

    # Distortion coefficients
    radial = calib["depth_parameters"]["radial_distortion"]
    tangential = calib["depth_parameters"]["tangential_distortion"]
    dist_coeffs = np.array([
        radial["m00"], radial["m10"], radial["m20"],  # k1, k2, k3
        tangential["m00"], tangential["m10"]         # p1, p2
    ], dtype=np.float32)

    # Extrinsics (camera pose in world coordinates)
    translation = calib["camera_pose"]["translation"]
    T = np.array([translation["m00"], translation["m10"], translation["m20"]], dtype=np.float32)
    rotation = calib["camera_pose"]["rotation"]
    quat = [rotation["x"], rotation["y"], rotation["z"], rotation["w"]]
    R = Rotation.from_quat(quat).as_matrix().astype(np.float32)

    # Projection matrix: P = K * [R | T]
    RT = np.hstack((R, T.reshape(3, 1)))
    P = K @ RT

    return K, dist_coeffs, P

def load_openpose(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    person = data["people"][0]  # Assuming one person per frame
    left_hand_2d = np.array(person["hand_left_keypoints_2d"], dtype=np.float32).reshape(-1, 3)[:, :2]  # [x, y, conf] -> [x, y]
    right_hand_2d = np.array(person["hand_right_keypoints_2d"], dtype=np.float32).reshape(-1, 3)[:, :2]
    return left_hand_2d, right_hand_2d

def undistort_points(points_2d, K, dist_coeffs):
    points_2d = points_2d.reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points_2d, K, dist_coeffs, P=K)
    return undistorted.reshape(-1, 2)

def triangulate_multi_view(projection_matrices, points_2d_list):
    num_views = len(projection_matrices)
    num_points = points_2d_list[0].shape[0]
    points_3d = []

    for i in range(num_points):
        A = []
        for j in range(num_views):
            x, y = points_2d_list[j][i]
            P = projection_matrices[j]
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        A = np.array(A)
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]  # Convert from homogeneous to 3D
        points_3d.append(X)
    
    return np.array(points_3d)

# --- Main Processing ---
def main():
    # Load calibration data for all cameras
    intrinsics_list = []
    dist_coeffs_list = []
    projection_matrices = []
    for calib_file in CALIB_FILES:
        K, dist_coeffs, P = load_calibration(calib_file)
        intrinsics_list.append(K)
        dist_coeffs_list.append(dist_coeffs)
        projection_matrices.append(P)

    # Load and undistort 2D keypoints from OpenPose
    left_hand_2d_views = []
    right_hand_2d_views = []
    for i, openpose_file in enumerate(OPENPOSE_FILES):
        left_2d, right_2d = load_openpose(openpose_file)
        left_2d_undistorted = undistort_points(left_2d, intrinsics_list[i], dist_coeffs_list[i])
        right_2d_undistorted = undistort_points(right_2d, intrinsics_list[i], dist_coeffs_list[i])
        left_hand_2d_views.append(left_2d_undistorted)
        right_hand_2d_views.append(right_2d_undistorted)

    # Triangulate 3D keypoints
    left_hand_3d = triangulate_multi_view(projection_matrices, left_hand_2d_views)
    right_hand_3d = triangulate_multi_view(projection_matrices, right_hand_2d_views)

    # Output results
    print("Left Hand 3D Keypoints (21 points, [X, Y, Z]):")
    for i, point in enumerate(left_hand_3d):
        print(f"Point {i}: {point}")

    print("\nRight Hand 3D Keypoints (21 points, [X, Y, Z]):")
    for i, point in enumerate(right_hand_3d):
        print(f"Point {i}: {point}")

    with open("right_3d.json", "w") as f:
        json.dump(right_hand_3d.tolist(), f, indent=4)
    
    with open("left_3d.json", "w") as f:
        json.dump(left_hand_3d.tolist(), f, indent=4)

    # Optionally save to file
    np.save("left_hand_3d.npy", left_hand_3d)
    np.save("right_hand_3d.npy", right_hand_3d)
    print("\n3D keypoints saved as 'left_hand_3d.npy' and 'right_hand_3d.npy'")

if __name__ == "__main__":
    main()