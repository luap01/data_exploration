import cv2
import numpy as np
import json
import os

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


def load_cam_params(path):
    K_list = json_load(path)
    color_params = K_list["value0"]["color_parameters"]
    fx = color_params["fov_x"]
    fy = color_params["fov_y"]
    cx = color_params["c_x"]
    cy = color_params["c_y"]
    intrinsics = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ], dtype=np.float32)

    camera_pose = K_list["value0"]["camera_pose"]
    t = camera_pose["translation"]
    r = camera_pose["rotation"]
    tx = t["m00"]
    ty = t["m10"]
    tz = t["m20"]
    qx = r["x"]
    qy = r["y"]
    qz = r["z"]
    qw = r["w"]

    # Convert quaternion to a 3x3 rotation matrix
    r00 = 1 - 2 * (qy**2 + qz**2)
    r01 = 2 * (qx*qy - qz*qw)
    r02 = 2 * (qx*qz + qy*qw)

    r10 = 2 * (qx*qy + qz*qw)
    r11 = 1 - 2 * (qx**2 + qz**2)
    r12 = 2 * (qy*qz - qx*qw)

    r20 = 2 * (qx*qz - qy*qw)
    r21 = 2 * (qy*qz + qx*qw)
    r22 = 1 - 2 * (qx**2 + qy**2)

    rotation_matrix = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])

    # Build the 4x4 extrinsics matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = [tx, ty, tz]

    radial = color_params["radial_distortion"]
    tangential = color_params["tangential_distortion"]

    distCoeffs = np.array([
    radial["m00"],       # k1
    radial["m10"],       # k2
    tangential["m00"],   # p1
    tangential["m10"],   # p2
    radial["m20"]        # k3
    ], dtype=np.float32)
    
    params = {}
    params['intrinsics'] = intrinsics
    params['extrinsics'] = extrinsic
    params['rotation'] = rotation_matrix
    params['distortion'] = distCoeffs
    return params


def triangulate_multiview(proj_matrices, points):
    """
    Triangulates a 3D point from multiple camera views using the DLT method.
    
    Parameters:
        proj_matrices (list): List of 3x4 projection matrices.
        points (list): List of corresponding 2D points [x, y] from each camera.
        
    Returns:
        3D point (x, y, z) as a numpy array.
    """
    A = []
    for P, pt in zip(proj_matrices, points):
        x, y = pt
        # For each camera, two equations are derived:
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    A = np.array(A)
    
    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[-1]  # Convert from homogeneous to Euclidean coordinates
    return X[:3]


def extract_hand_keypoints(json_data):
    people = json_data["people"]
    hand_keypoints = {}
    
    for person in people:
        # Extract left hand keypoints
        left_hand = []
        for i in range(0, len(person["hand_left_keypoints_2d"]), 3):
            x = person["hand_left_keypoints_2d"][i]
            y = person["hand_left_keypoints_2d"][i+1]
            confidence = person["hand_left_keypoints_2d"][i+2]
            # if confidence > 0.5:  # Filter by confidence
            left_hand.append([x, y, confidence])
        
        # Extract right hand keypoints
        right_hand = []
        for i in range(0, len(person["hand_right_keypoints_2d"]), 3):
            x = person["hand_right_keypoints_2d"][i]
            y = person["hand_right_keypoints_2d"][i+1]
            confidence = person["hand_right_keypoints_2d"][i+2]
            # if confidence > 0.5:  # Filter by confidence
            right_hand.append([x, y, confidence])
        
        hand_keypoints["left_hand"] = left_hand
        hand_keypoints["right_hand"] = right_hand
    
    return hand_keypoints


def undistort(keypoints, params):
    xy = []
    for i in range(0, len(keypoints), 3):
        xy.append(keypoints[i])
        xy.append(keypoints[i+1])

    points = np.array(xy).reshape(-1, 1, 2)
    print(points.shape)
    undistorted_points = cv2.undistortPoints(points, params['intrinsics'], params['distortion'], P=params['intrinsics'])
    return undistorted_points.reshape(-1, 2)

    
def triangulate_points_dlt(points_2d_cameras, camera_params_list):
    """
    Triangulate 3D points from multiple camera views using DLT.
    
    Args:
        points_2d_cameras: List of 2D points from each camera
        camera_params_list: List of camera parameters for each camera
    
    Returns:
        Array of 3D points
    """
    # Ensure we have the same number of points from each camera
    num_points = min(len(points) for points in points_2d_cameras)
    for points in points_2d_cameras:
        assert num_points == len(points)
    
    # Create projection matrices for each camera
    projection_matrices = []
    for camera_params in camera_params_list:
        # Create projection matrix
        P = camera_params["intrinsics"] @ camera_params['extrinsics'][:3, :]
        projection_matrices.append(P)
    
    # Triangulate each point
    points_3d = []
    for i in range(num_points):
        # Get the 2D coordinates of the point from each camera
        point_2d_views = []
        for cam_idx, points in enumerate(points_2d_cameras):
            if i < len(points):
                point_2d_views.append(points[i][:2])  # Only x,y coordinates
        
        # Only triangulate if we have the point from at least 2 cameras
        if len(point_2d_views) >= 2:
            # Build the DLT matrix
            A = np.zeros((2 * len(point_2d_views), 4))
            
            for j, point_2d in enumerate(point_2d_views):
                x, y = point_2d
                P = projection_matrices[j]
                
                # Fill the DLT matrix rows
                A[2*j]   = x * P[2] - P[0]
                A[2*j+1] = y * P[2] - P[1]
            
            # Solve the DLT system using SVD
            _, _, Vt = np.linalg.svd(A)
            point_3d_homogeneous = Vt[-1]
            
            # Convert from homogeneous coordinates to 3D
            point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
            points_3d.append(point_3d)
    
    return np.array(points_3d)


def save_3d_coord(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    cam_params_1 = load_cam_params("../HaMuCo/data/OR/calib/camera01.json")
    cam_params_2 = load_cam_params("../HaMuCo/data/OR/calib/camera02.json")
    cam_params_3 = load_cam_params("../HaMuCo/data/OR/calib/camera03.json")
    cam_params_4 = load_cam_params("../HaMuCo/data/OR/calib/camera04.json")

    num_files = 4426
    for idx in range(num_files):
        view_1 = json_load(f"./json/camera01_{str(idx).zfill(6)}_keypoints.json")
        view_2 = json_load(f"./json/camera02_{str(idx).zfill(6)}_keypoints.json")
        view_3 = json_load(f"./json/camera03_{str(idx).zfill(6)}_keypoints.json")
        view_4 = json_load(f"./json/camera04_{str(idx).zfill(6)}_keypoints.json")

        try:
            left_hand_keypoints = [extract_hand_keypoints(view_1)['left_hand'], extract_hand_keypoints(view_2)['left_hand'], extract_hand_keypoints(view_3)['left_hand'], extract_hand_keypoints(view_4)['left_hand']]  
            right_hand_keypoints = [extract_hand_keypoints(view_1)['right_hand'], extract_hand_keypoints(view_2)['right_hand'], extract_hand_keypoints(view_3)['right_hand'], extract_hand_keypoints(view_4)['right_hand']]

            undistorted_left_hand_keypoints = [undistort(keypoints, params) for keypoints, params in zip(left_hand_keypoints, [cam_params_1, cam_params_2, cam_params_3, cam_params_4])]
            undistorted_right_hand_keypoints = [undistort(keypoints, params) for keypoints, params in zip(right_hand_keypoints, [cam_params_1, cam_params_2, cam_params_3, cam_params_4])]
            left_data = triangulate_points_dlt(undistorted_left_hand_keypoints, [cam_params_1, cam_params_2, cam_params_3, cam_params_4])
            right_data = triangulate_points_dlt(undistorted_right_hand_keypoints, [cam_params_1, cam_params_2, cam_params_3, cam_params_4])
            save_3d_coord(left_data.tolist(), f"./xyz/left/{str(idx).zfill(6)}.json")
            save_3d_coord(right_data.tolist(), f"./xyz/right/{str(idx).zfill(6)}.json")
        except KeyError:
            print(f"File {idx} is not existent...")
        
        break
