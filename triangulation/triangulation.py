import cv2
import numpy as np
import json
import os
from pathlib import Path
import random

from utils.camera import load_cam_infos, load_projection_matrix, project_to_2d
from utils.created_2d_kps_file import json_load, save_file
from utils.image import undistort_image
from utils.files import load_all_keypoints


def load_cam_params(path, orbbec: bool = True, both: bool = False):
    return load_cam_infos(path, orbbec, both)

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


def distance(point_2d_observed, point_2d_projected):
    """
    Compute Euclidean distance between observed and projected 2D points.
    
    Args:
        point_2d_observed: Observed 2D point [x, y].
        point_2d_projected: Projected 2D point [x, y].
    
    Returns:
        Euclidean distance.
    """
    return np.sqrt(np.sum((point_2d_observed - point_2d_projected) ** 2))


def triangulate_point(keypoints_2d, projection_matrices):
    A = []
    for i, (x, y) in enumerate(keypoints_2d):
        # P = projection_matrices[f'camera0{i+1}']
        P = projection_matrices[i]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]    # Last row of V
    X = X / X[3]  # Homogeneous to Euclidean
    return X[:3]  # Return (X, Y, Z)



def ransac_triangulation(points_2d, projection_matrices, camera_params, max_iterations, threshold):
    """
    Perform RANSAC triangulation to estimate a 3D point robustly.
    
    Args:
        points_2d: List of 2D points [x, y] from multiple cameras.
        projection_matrices: List of 3x4 projection matrices.
        max_iterations: Number of RANSAC iterations.
        threshold: Reprojection error threshold in pixels.
    
    Returns:
        best_3d_point: Estimated 3D point [x, y, z].
        best_inliers: Indices of inlier views.
    """
    best_inliers = []
    best_3d_point = None
    best_responding_cam_idx = []
    
    for _ in range(max_iterations):
        # Randomly select two views
        indices = random.sample(range(len(points_2d)), 2)
        sampled_points = [points_2d[i] for i in indices]
        sampled_cameras = [projection_matrices[f'camera0{i+1}'] for i in indices]
        
        # Triangulate 3D point
        point_3d = triangulate_point(sampled_points, sampled_cameras)
        
        # Count inliers
        inliers = []
        responding_cam_idx = []
        for i in range(len(points_2d)):
            projected_2d = project_to_2d(point_3d, camera_params[f'camera0{i+1}']['intrinsics'], np.linalg.inv(camera_params[f'camera0{i+1}']['extrinsics']) if i < 4 else camera_params[f'camera0{i+1}']['extrinsics'])
            error = distance(projected_2d, points_2d[i])
            if error < threshold:
                inliers.append(i)
                responding_cam_idx.append(f'camera0{i+1}')
        
        # Update best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_3d_point = point_3d
            best_responding_cam_idx = responding_cam_idx

    # Refine using all inliers
    if best_inliers:
        refined_point = triangulate_point([points_2d[i] for i in best_inliers], [projection_matrices[cam_idx] for cam_idx in best_responding_cam_idx])
        if refined_point is not None:
            best_3d_point = refined_point
    
    return best_3d_point, best_inliers



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


def cv2_triangulation(points_2d, projection_matrices):
    all_cam_params = load_cam_params(Path("../HaMuCo/data/OR"), orbbec=True, both=False)
    
    base_path = "./data/output/openpose/json"
    keypoints = load_all_keypoints(Path(base_path))
    
    num_files = len(keypoints[list(keypoints.keys())[0]])

    for idx in range(num_files):
        projection_matrices = load_projection_matrix(all_cam_params)
        left_hand_keypoints = []
        right_hand_keypoints = []
        for cam_idx in all_cam_params.keys():
            keypoints_2d = extract_hand_keypoints(keypoints[cam_idx][idx], including_confidence=False)
            left_hand_keypoints.append(np.array(keypoints_2d['left_hand']))
            right_hand_keypoints.append(np.array(keypoints_2d['right_hand']))


        left_triangulated_points = cv2.triangulatePoints(projection_matrices['camera01'], projection_matrices['camera02'], left_hand_keypoints[0].T, left_hand_keypoints[1].T).T
        right_triangulated_points = cv2.triangulatePoints(projection_matrices['camera01'], projection_matrices['camera02'], right_hand_keypoints[0].T, right_hand_keypoints[1].T).T

        print(left_triangulated_points[:, :3].shape)
        for left_triangulated_point, right_triangulated_point, l_kp_2d, r_kp_2d, l_kp_2d_2, r_kp_2d_2 in zip(left_triangulated_points, right_triangulated_points, left_hand_keypoints[0], right_hand_keypoints[0], left_hand_keypoints[1], right_hand_keypoints[1]):
            left_2d_keypoints = np.array([l_kp_2d, l_kp_2d_2])
            right_2d_keypoints = np.array([r_kp_2d, r_kp_2d_2])

            l_mean_error, l_errors = validate_triangulation(left_triangulated_point, left_2d_keypoints, projection_matrices)
            r_mean_error, r_errors = validate_triangulation(right_triangulated_point, right_2d_keypoints, projection_matrices)
            print(l_mean_error)
            print(r_mean_error)


        save_base_path = "./data/output/xyz"
        save_file(np.array(left_triangulated_points[:, :3]).tolist(), f"{save_base_path}/left/{str(idx).zfill(6)}.json")
        save_file(np.array(right_triangulated_points[:, :3]).tolist(), f"{save_base_path}/right/{str(idx).zfill(6)}.json")


if __name__ == "__main__":
    all_cam_params = load_cam_params(Path("../HaMuCo/data/OR"), orbbec=True, both=False)

    base_path = "./data/output/openpose/json"
    keypoints = load_all_keypoints(Path(base_path))
    
    num_files = len(keypoints[list(keypoints.keys())[0]])
    print(num_files)

    for idx in range(num_files):
        # 22, 23, 44, 51, 52, 66, 74
        if idx > 16:
            break
        print(f"Processing file {idx}/{num_files}...")
        projection_matrices = load_projection_matrix(all_cam_params)
        left_hand_keypoints = []
        right_hand_keypoints = []
        for cam_idx in all_cam_params.keys():
            keypoints_2d = extract_hand_keypoints(keypoints[cam_idx][idx], including_confidence=False)
            left_hand_keypoints.append(np.array(keypoints_2d['left_hand']))
            right_hand_keypoints.append(np.array(keypoints_2d['right_hand']))


        left_triangulated_points = []
        right_triangulated_points = []
        for i in range(len(left_hand_keypoints[0])):
            l_kp_2d = [left_hand_keypoints[cam_idx][i] for cam_idx in range(len(projection_matrices))]
            r_kp_2d = [right_hand_keypoints[cam_idx][i] for cam_idx in range(len(projection_matrices))]


            left_triangulated_point, _ = ransac_triangulation(
                l_kp_2d,
                projection_matrices, 
                all_cam_params,
                max_iterations=10000, 
                threshold=100.0
            )
            
            right_triangulated_point, _ = ransac_triangulation(
                r_kp_2d, 
                projection_matrices,
                all_cam_params,
                max_iterations=10000,
                threshold=100.0
            )

            # cv2.triangulatePoints(projection_matrices['camera01'], projection_matrices['camera02'], l_kp_2d, r_kp_2d, )

            # left_triangulated_point = cv2.RANSAC.compute(l_kp_2d, projection_matrices['camera01'], projection_matrices['camera02'])
            # print(left_triangulated_point)
            # print(right_triangulated_point)
            left_triangulated_points.append(left_triangulated_point)
            right_triangulated_points.append(right_triangulated_point)

            # print(left_triangulated_point.shape)
            # print(right_triangulated_point.shape)
            try:
                l_mean_error, l_errors = validate_triangulation(left_triangulated_point, l_kp_2d, projection_matrices)
                r_mean_error, r_errors = validate_triangulation(right_triangulated_point, r_kp_2d, projection_matrices)
            except:
                print(f"No triangulation at {idx}: {i}")
            # print(l_mean_error)
            # print(r_mean_error)


        save_base_path = "./data/output/xyz"
        save_file(np.array(left_triangulated_points).tolist(), f"{save_base_path}/left/{str(idx).zfill(6)}.json")
        save_file(np.array(right_triangulated_points).tolist(), f"{save_base_path}/right/{str(idx).zfill(6)}.json")
        
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
    
