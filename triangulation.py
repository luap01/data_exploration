import cv2
import numpy as np
import json
import os
from pathlib import Path

from utils.camera import load_cam_infos, load_projection_matrix
from utils.move import json_load, save_file
from utils.image import undistort_image
from utils.files import load_all_keypoints

# def load_cam_params(path):
#     K_list = json_load(path)
#     color_params = K_list["value0"]["color_parameters"]
#     fx = color_params["fov_x"]
#     fy = color_params["fov_y"]
#     cx = color_params["c_x"]
#     cy = color_params["c_y"]
#     intrinsics = np.array([
#         [fx, 0,  cx],
#         [0,  fy, cy],
#         [0,  0,   1]
#     ], dtype=np.float32)

#     camera_pose = K_list["value0"]["camera_pose"]
#     t = camera_pose["translation"]
#     r = camera_pose["rotation"]
#     tx = t["m00"]
#     ty = t["m10"]
#     tz = t["m20"]
#     qx = r["x"]
#     qy = r["y"]
#     qz = r["z"]
#     qw = r["w"]

#     # Convert quaternion to a 3x3 rotation matrix
#     r00 = 1 - 2 * (qy**2 + qz**2)
#     r01 = 2 * (qx*qy - qz*qw)
#     r02 = 2 * (qx*qz + qy*qw)

#     r10 = 2 * (qx*qy + qz*qw)
#     r11 = 1 - 2 * (qx**2 + qz**2)
#     r12 = 2 * (qy*qz - qx*qw)

#     r20 = 2 * (qx*qz - qy*qw)
#     r21 = 2 * (qy*qz + qx*qw)
#     r22 = 1 - 2 * (qx**2 + qy**2)

#     rotation_matrix = np.array([
#         [r00, r01, r02],
#         [r10, r11, r12],
#         [r20, r21, r22]
#     ])

#     # Build the 4x4 extrinsics matrix
#     extrinsic = np.eye(4)
#     extrinsic[:3, :3] = rotation_matrix
#     extrinsic[:3, 3] = [tx, ty, tz]

#     radial = color_params["radial_distortion"]
#     tangential = color_params["tangential_distortion"]

#     distCoeffs = np.array([
#     radial["m00"],       # k1
#     radial["m10"],       # k2
#     tangential["m00"],   # p1
#     tangential["m10"],   # p2
#     radial["m20"]        # k3
#     ], dtype=np.float32)
    
#     params = {}
#     params['intrinsics'] = intrinsics
#     params['extrinsics'] = extrinsic
#     params['rotation'] = rotation_matrix
#     params['distortion'] = distCoeffs
#     return params

def load_cam_params(path):
    return load_cam_infos(path)

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


def extract_hand_keypoints(json_data, including_confidence=False):
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
            left_hand.append([x, y, confidence]) if including_confidence else left_hand.append([x, y])
        
        # Extract right hand keypoints
        right_hand = []
        for i in range(0, len(person["hand_right_keypoints_2d"]), 3):
            x = person["hand_right_keypoints_2d"][i]
            y = person["hand_right_keypoints_2d"][i+1]
            confidence = person["hand_right_keypoints_2d"][i+2]
            # if confidence > 0.5:  # Filter by confidence
            right_hand.append([x, y, confidence]) if including_confidence else right_hand.append([x, y])
        
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

    
def triangulate_points_dlt(points_2d_cameras, camera_params_dict):
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
    for cam_idx in camera_params_dict.keys():
        # Create projection matrix
        camera_params = all_cam_params[cam_idx]
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



def triangulate_point(keypoints_2d, projection_matrices):
    A = []
    for i, (x, y) in enumerate(keypoints_2d):
        P = projection_matrices[f'camera0{i+1}']
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]    # Last row of V
    X = X / X[3]  # Homogeneous to Euclidean
    return X[:3]  # Return (X, Y, Z)


def validate_triangulation(point_3d, keypoints_2d, projection_matrices):
    reprojection_errors = []
    point_3d_homo = np.append(point_3d, 1)  # Convert to homogeneous coordinates [X, Y, Z, 1]

    for i, (x_orig, y_orig) in enumerate(keypoints_2d):
        P = projection_matrices[f'camera0{i+1}']
        
        # Project 3D point back to 2D
        point_2d_homo = P @ point_3d_homo  # [x', y', w']
        point_2d = point_2d_homo[:2] / point_2d_homo[2]  # Normalize by w' to get [x, y]
        x_reproj, y_reproj = point_2d
        
        # Compute Euclidean distance between original and reprojected points
        error = np.sqrt((x_orig - x_reproj)**2 + (y_orig - y_reproj)**2)
        reprojection_errors.append(error)
        
        print(f"Camera {i+1}: Original ({x_orig:.2f}, {y_orig:.2f}), "
              f"Reprojected ({x_reproj:.2f}, {y_reproj:.2f}), Error: {error:.2f} pixels")

    mean_error = np.mean(reprojection_errors)
    print(f"Mean reprojection error: {mean_error:.2f} pixels")
    return mean_error, reprojection_errors


if __name__ == "__main__":
    all_cam_params = load_cam_params(Path("../HaMuCo/data/OR"))

    num_files = 4426
    for idx in range(num_files):
        base_path = "./data/output/json"
        keypoints = load_all_keypoints(Path(base_path))

        projection_matrices = load_projection_matrix(all_cam_params)
        left_hand_keypoints = []
        right_hand_keypoints = []
        for cam_idx in keypoints.keys():
            keypoints_2d = extract_hand_keypoints(keypoints[cam_idx], including_confidence=False)
            left_hand_keypoints.append(np.array(keypoints_2d['left_hand']))
            right_hand_keypoints.append(np.array(keypoints_2d['right_hand']))

        left_triangulated_points = []
        right_triangulated_points = []
        for i in range(len(left_hand_keypoints[0])):
            l_kp_2d = [left_hand_keypoints[cam_idx][i] for cam_idx in range(len(projection_matrices))]
            r_kp_2d = [right_hand_keypoints[cam_idx][i] for cam_idx in range(len(projection_matrices))]

            left_triangulated_point = triangulate_point(l_kp_2d, projection_matrices)
            right_triangulated_point = triangulate_point(r_kp_2d, projection_matrices)

            print(left_triangulated_point)
            print(right_triangulated_point)
            left_triangulated_points.append(left_triangulated_point)
            right_triangulated_points.append(right_triangulated_point)

            l_mean_error, l_errors = validate_triangulation(left_triangulated_point, l_kp_2d, projection_matrices)
            r_mean_error, r_errors = validate_triangulation(right_triangulated_point, r_kp_2d, projection_matrices)

            print(l_mean_error)
            print(r_mean_error)


        save_base_path = "./data/output/xyz"
        save_file(np.array(left_triangulated_points).tolist(), f"{save_base_path}/left/{str(idx).zfill(6)}.json")
        save_file(np.array(right_triangulated_points).tolist(), f"{save_base_path}/right/{str(idx).zfill(6)}.json")
        
        break
        # view_1 = json_load(f"{base_path}/camera01_{str(idx).zfill(6)}_keypoints.json")
        # view_2 = json_load(f"{base_path}/camera02_{str(idx).zfill(6)}_keypoints.json")
        # view_3 = json_load(f"{base_path}/camera03_{str(idx).zfill(6)}_keypoints.json")
        # view_4 = json_load(f"{base_path}/camera04_{str(idx).zfill(6)}_keypoints.json")

        # try:
        #     left_hand_keypoints = [extract_hand_keypoints(view_1)['left_hand'], extract_hand_keypoints(view_2)['left_hand'], extract_hand_keypoints(view_3)['left_hand'], extract_hand_keypoints(view_4)['left_hand']]  
        #     right_hand_keypoints = [extract_hand_keypoints(view_1)['right_hand'], extract_hand_keypoints(view_2)['right_hand'], extract_hand_keypoints(view_3)['right_hand'], extract_hand_keypoints(view_4)['right_hand']]

        #     undistorted_left_hand_keypoints = left_hand_keypoints
        #     undistorted_right_hand_keypoints = right_hand_keypoints
        #     # undistorted_left_hand_keypoints = [undistort(keypoints, params) for keypoints, params in zip(left_hand_keypoints, [cam_params_1, cam_params_2, cam_params_3, cam_params_4])]
        #     # undistorted_right_hand_keypoints = [undistort(keypoints, params) for keypoints, params in zip(right_hand_keypoints, [cam_params_1, cam_params_2, cam_params_3, cam_params_4])]
        #     left_data = triangulate_points_dlt(undistorted_left_hand_keypoints, all_cam_params)
        #     right_data = triangulate_points_dlt(undistorted_right_hand_keypoints, all_cam_params)
        #     save_base_path = "./data/output/xyz"
        #     save_file(left_data.tolist(), f"{save_base_path}/left/{str(idx).zfill(6)}.json")
        #     save_file(right_data.tolist(), f"{save_base_path}/right/{str(idx).zfill(6)}.json")
        # except Exception as e:
        #     print(f"File {idx} is not existent...")
        #     print(repr(e))
        
        break
