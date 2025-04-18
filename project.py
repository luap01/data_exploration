import cv2
import numpy as np
import json
import os
from pathlib import Path

from utils.camera import load_cam_infos, project_to_2d
from utils.files import json_load, load_all_images, load_all_xyz, load_all_keypoints
from utils.image import undistort_image
from scipy.spatial.transform import Rotation


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

HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)


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


def create_img(image, projected_points_2d, save_path):
    for _, data in HAND_STRUCTURE.items():
        color = data["color"]
        for start_idx, end_idx in data["connections"]:
            try:
                x1, y1 = int(projected_points_2d[start_idx][0]), int(projected_points_2d[start_idx][1])
                x2, y2 = int(projected_points_2d[end_idx][0]), int(projected_points_2d[end_idx][1])
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
            except Exception:
                continue

    # Draw points on the image
    for idx, point in enumerate(projected_points_2d):
        try:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)  # Green circles for projected points
        except Exception as e:
            print(f"Skipping point={idx}, point={point}, error={e}")

    # Save the image with points
    os.makedirs("/".join(str(save_path).split("/")[0:-1]), exist_ok=True)
    cv2.imwrite(save_path, image)


def visualize_openpose_keypoints(image_path, keypoints_left, keypoints_right, index):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")



    # Extract keypoints
    left_hand_keypoints = np.array(keypoints_left).reshape(-1, 3)
    right_hand_keypoints = np.array(keypoints_right).reshape(-1, 3) if keypoints_right is not None else np.zeros((0, 3))

    CONF_LVL = 0.00001
    # Draw hand keypoints
    for i, (x, y, conf) in enumerate(left_hand_keypoints):
        if conf > CONF_LVL:
            cv2.circle(img, (int(x), int(y)), 4, RED, -1)

    for i, (x, y, conf) in enumerate(right_hand_keypoints):
        if conf > CONF_LVL:
            cv2.circle(img, (int(x), int(y)), 4, BLUE, -1)

    # # Simple hand connections (connecting fingers to wrist)
    # for keypoints in [hand_left_keypoints, hand_right_keypoints]:
    #     if keypoints[0][2] > CONF_LVL:  # If wrist is detected
    #         wrist_x, wrist_y = int(keypoints[0][0]), int(keypoints[0][1])
    #         for i in range(1, len(keypoints), 4):  # Connect to finger tips
    #             if keypoints[i][2] > CONF_LVL:
    #                 finger_x, finger_y = int(keypoints[i][0]), int(keypoints[i][1])
    #                 cv2.line(img, (wrist_x, wrist_y), (finger_x, finger_y), GREEN, 1)

    
    # Draw connections for left hand
    for idx1, idx2 in HAND_CONNECTIONS:
        if (left_hand_keypoints[idx1][2] > CONF_LVL and left_hand_keypoints[idx2][2] > CONF_LVL):
            x1, y1 = int(left_hand_keypoints[idx1][0]), int(left_hand_keypoints[idx1][1])
            x2, y2 = int(left_hand_keypoints[idx2][0]), int(left_hand_keypoints[idx2][1])
            cv2.line(img, (x1, y1), (x2, y2), GREEN, 1)

    # Draw connections for right hand
    for idx1, idx2 in HAND_CONNECTIONS:
        if (right_hand_keypoints[idx1][2] > CONF_LVL and right_hand_keypoints[idx2][2] > CONF_LVL):
            x1, y1 = int(right_hand_keypoints[idx1][0]), int(right_hand_keypoints[idx1][1])
            x2, y2 = int(right_hand_keypoints[idx2][0]), int(right_hand_keypoints[idx2][1])
            cv2.line(img, (x1, y1), (x2, y2), YELLOW, 1)

    # Display and save the result
    cv2.imshow("OpenPose Visualization", img)
    cv2.waitKey(0)
    cv2.imwrite(image_path.replace("input", "output"), img)
    # cv2.destroyAllWindows()



def _project_3d_to_2d(xyz, intrinsics, extrinsics):
    """
    Project 3D points to 2D using camera intrinsics and extrinsics.
    Args:
        xyz: (N, 3) array of 3D points
        intrinsics: (3, 3) intrinsic matrix
        extrinsics: (4, 4) extrinsic matrix
    Returns:
        projected_2d: (N, 2) array of 2D pixel coordinates
    """
    # Convert to homogeneous coordinates (N, 3) -> (N, 4)
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    xyz_homo = np.hstack((xyz, ones))  # Shape: (21, 4)

    # Transform to camera coordinates: (4, 4) @ (21, 4).T -> (4, 21)
    cam_coords = extrinsics @ xyz_homo.T  # Shape: (4, 21)

    # Project to image plane: (3, 3) @ (3, 21) -> (3, 21)
    img_coords = intrinsics @ cam_coords[:3, :]  # Take only X_c, Y_c, Z_c, shape: (3, 21)

    # Normalize homogeneous coordinates
    img_coords = img_coords.T  # Shape: (21, 3) -> (u', v', w')
    projected_2d = img_coords[:, :2] / img_coords[:, 2:3]  # (u, v) = (u'/w', v'/w'), shape: (21, 2)

    return projected_2d


def project_3d_to_2d(xyz, K_camera, M_camera):
    """
    Projects 3D points (xyz) to 2D image plane using intrinsic (K) and extrinsic (M) matrices.
    """
    # Convert 3D points to homogeneous coordinates
    xyz_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))  # Shape: (N, 4)

    # Extract rotation (R) and translation (t) from M
    R = M_camera[:3, :3]  # First 3x3 part of M
    t = M_camera[:3, 3]   # Last column of M (translation)
    
    # Form the 3x4 extrinsic matrix
    extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))  # Shape: (3, 4)

    # Compute the full projection matrix
    projection_matrix = K_camera @ extrinsic_matrix  # Shape: (3, 4)
    
    # Apply projection
    projected_points = projection_matrix @ xyz_homogeneous.T  # Shape: (3, N)

    # Normalize to get pixel coordinates
    projected_points_2d = projected_points[:2, :] / projected_points[2, :]  # Normalize by depth

    return projected_points_2d.T  # Shape: (N, 2)


def project_3d_to_2d_alt(xyz, intrinsics, extrinsics):
    """Alternative projection method"""
    # Apply extrinsics (world to camera)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    
    # Transform points from world to camera coordinates
    xyz_cam = xyz @ R.T + t
    
    # Project to image plane
    x_img = xyz_cam[:, 0] / xyz_cam[:, 2] * intrinsics[0, 0] + intrinsics[0, 2]
    y_img = xyz_cam[:, 1] / xyz_cam[:, 2] * intrinsics[1, 1] + intrinsics[1, 2]
    
    return np.column_stack((x_img, y_img))


def update_intrinsics_from_fov(intrinsics, image_width, image_height):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    # If fx, fy are suspiciously small, assume they are FOV in degrees
    if fx < 10 or fy < 10:
        fov_x = np.radians(fx)  # Convert to radians if needed
        fov_y = np.radians(fy)
        fx = (image_width / 2) / np.tan(fov_x / 2)
        fy = (image_height / 2) / np.tan(fov_y / 2)
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
    return intrinsics


def invert_extrinsics(extrinsics):
    """Inverts extrinsic matrix from world-to-camera to camera-to-world or vice versa"""
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    
    R_inv = R.T
    t_inv = -R_inv @ t
    
    extrinsic_inv = np.eye(4)
    extrinsic_inv[:3, :3] = R_inv
    extrinsic_inv[:3, 3] = t_inv
    
    return extrinsic_inv


import cv2
import numpy as np

def project_3d_to_2d_distortion(xyz, K_camera, M_camera, distCoeffs):
    """
    Projects 3D points (xyz) to 2D image plane using intrinsic (K), extrinsic (M)
    and distortion coefficients.
    Uses cv2.projectPoints to account for radial and tangential distortion.
    
    Args:
        xyz: (N, 3) array of 3D points.
        K_camera: (3, 3) intrinsic matrix.
        M_camera: (4, 4) extrinsic matrix.
        distCoeffs: (1, 5) or (5,) array of distortion coefficients 
                    (typically [k1, k2, p1, p2, k3]).
    
    Returns:
        projected_points_2d: (N, 2) array of 2D pixel coordinates.
    """
    # Extract rotation and translation from the 4x4 extrinsic matrix
    R = M_camera[:3, :3]
    t = M_camera[:3, 3]
    
    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)
    
    # Use cv2.projectPoints which takes distortion into account
    image_points, _ = cv2.projectPoints(xyz, rvec, t, K_camera, distCoeffs)
    
    return image_points.reshape(-1, 2)




def test_run(kps, cam_params, colour, dir, img, idx):
    points_2d = []
    for i, point_3d in enumerate(kps):
            # Project 3D point to 2D image coordinates
            point_2d = project_to_2d(
                point_3d, cam_params["intrinsics"], np.linalg.inv(cam_params["extrinsics"])
            )

            # Save the 2D point
            points_2d.append((int(point_2d[0]), int(point_2d[1])))

            # Draw the point on the image
            cv2.circle(
                img,
                (int(point_2d[0]), int(point_2d[1])),
                10,
                colour,
                thickness=-1,
            )

    # cv2.imshow("OpenPose Visualization", img)
    # cv2.waitKey(0)
    cv2.imwrite(f"./output/images/test/cam{idx}_direction_{dir}.jpg", img)
    # cv2.destroyAllWindows()

    print("Projected 2D points for cam", idx, "in direction", dir, ":\n", points_2d)
    return points_2d


def tony_proj_function_test_run():
    all_cam_params = load_cam_infos(Path("../HaMuCo/data/OR"))
    for i in range(len(all_cam_params)):
        cam_idx = i + 1
        cam_params = all_cam_params[f'camera0{cam_idx}']

        x_kps = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        y_kps = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
        z_kps = np.array([[0, 0, -1], [0, 0, 0], [0, 0, 1]])

        img = cv2.imread(f"./input/cam{cam_idx}_000000.jpg")
        img = undistort_image(img, cam_params, 'color')
        test_run(x_kps, cam_params, (0, 0, 255), "x", img, cam_idx)
        img = cv2.imread(f"./input/cam{cam_idx}_000000.jpg")
        img = undistort_image(img, cam_params, 'color')
        test_run(y_kps, cam_params, (0, 255, 0), "y", img, cam_idx)
        img = cv2.imread(f"./input/cam{cam_idx}_000000.jpg")
        img = undistort_image(img, cam_params, 'color')
        test_run(z_kps, cam_params, (255, 0, 0), "z", img, cam_idx)


def assemble_2d_points(_3d_points, cam_params):
    points_2d = []
    for i, point_3d in enumerate(_3d_points):
        point_2d = project_to_2d(
            point_3d, cam_params["intrinsics"], np.linalg.inv(cam_params["extrinsics"])
        )
        points_2d.append((int(point_2d[0]), int(point_2d[1])))
    return points_2d

def visualize_2d_points(points_2d, img, dot_colour, line_colour):
   for i, (x, y) in enumerate(points_2d):
        cv2.circle(img, (int(x), int(y)), 4, dot_colour, -1)

   for idx1, idx2 in HAND_CONNECTIONS:
        x1, y1 = int(points_2d[idx1][0]), int(points_2d[idx1][1])
        x2, y2 = int(points_2d[idx2][0]), int(points_2d[idx2][1])
        cv2.line(img, (x1, y1), (x2, y2), line_colour, 1)


if __name__ == "__main__":
    # img = cv2.imread("./data/input/cam1_000000.jpg")
    # height, width = img.shape[:2]
    # print(f"Image size: {width}x{height}")


    # tony_proj_function_test_run()
    # keypoints = json_load("./openpose/json/camera01_000000_keypoints.json")
    # visualize_openpose_keypoints("./input/cam1_000000.jpg", keypoints['people'][0]['hand_left_keypoints_2d'], keypoints['people'][0]['hand_right_keypoints_2d'], 0)
    # keypoints = json_load("./openpose/json/camera02_000000_keypoints.json")
    # visualize_openpose_keypoints("./input/cam2_000000.jpg", keypoints['people'][0]['hand_left_keypoints_2d'], keypoints['people'][0]['hand_right_keypoints_2d'], 0)
    # keypoints = json_load("./openpose/json/camera03_000000_keypoints.json")
    # visualize_openpose_keypoints("./input/cam3_000000.jpg", keypoints['people'][0]['hand_left_keypoints_2d'], keypoints['people'][0]['hand_right_keypoints_2d'], 0)
    # keypoints = json_load("./openpose/json/camera04_000000_keypoints.json")
    # visualize_openpose_keypoints("./input/cam4_000000.jpg", keypoints['people'][0]['hand_left_keypoints_2d'], keypoints['people'][0]['hand_right_keypoints_2d'], 0)

    # openpose_left = np.array(keypoints['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)
    # openpose_right = np.array(keypoints['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)
    # print("OpenPose Left 2D sample:\n", openpose_left[:5])
    # visualize_openpose_keypoints("./000000.jpg", keypoints['people'][0]['hand_left_keypoints_2d'], keypoints['people'][0]['hand_right_keypoints_2d'], 0)

    input_base_pth = "./data/input/openpose/images/"
    keypoints_base_pth = "./data/output/openpose/json/"
    hamuco_base_pth = "../HaMuCo/data/OR"
    all_images = load_all_images(Path(input_base_pth))
    # all_xyz = load_all_xyz(Path(hamuco_base_pth + "/xyz/"))

    all_keypoints = load_all_keypoints(Path(keypoints_base_pth))
    all_cam_params = load_cam_infos(Path(hamuco_base_pth), both=True)
    iteration = len(os.listdir("./data/output/images/test/")) // 6
    i = 0
    for image in all_images.keys():
        if i == 0:
            i += iteration
            prev_cam = image.split("/")[-2]

        if image.split("/")[-2] != prev_cam:
            i = 0
            prev_cam = image.split("/")[-2]

        cam_idx = image.split("/")[-2]
        kpt_idx = image.replace(".jpg", ".json")

        if "camera05" not in image:
            continue
        
        if cam_idx == "camera05":
            image = image.replace("camera05", "camera01")
            cam_idx = "camera01"

        cam_params = all_cam_params[cam_idx]        
        img = all_images[image]
        img = undistort_image(img, cam_params, 'color', orbbec=False if "5" in cam_idx or "6" in cam_idx else True)

        # left_xyz = all_xyz["left"]
        # right_xyz = all_xyz["right"]

        # left_2d = assemble_2d_points(left_xyz, cam_params)
        # right_2d = assemble_2d_points(right_xyz, cam_params)
        
        left_2d = np.array(all_keypoints[kpt_idx]['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2]
        right_2d = np.array(all_keypoints[kpt_idx]['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]

        # Create 2D rotation matrix for 90 degree rotation
        theta = np.radians(-90)  # Convert -90 degrees to radians
        c, s = np.cos(theta), np.sin(theta)
        R_2d = np.array([[c, -s], [s, c]])
        
        # Apply rotation to points
        left_2d = (R_2d @ left_2d.T).T
        right_2d = (R_2d @ right_2d.T).T

        visualize_2d_points(left_2d, img, dot_colour=RED, line_colour=GREEN)
        visualize_2d_points(right_2d, img, dot_colour=BLUE, line_colour=YELLOW)
    
        cv2.imwrite(f"./data/output/images/proj/{i}_{cam_idx}_{kpt_idx.split('/')[-1].split('.')[0]}.jpg", img)

        i += 1
        
        
    # left_xyz = np.array(json_load("./xyz/left/000000.json"))
    # right_xyz = np.array(json_load("./xyz/right/000000.json"))
    # cam_params = load_cam_params("../HaMuCo/data/OR/calib/camera01.json")
    

    # left_2d = assemble_2d_points(left_xyz, cam_params)
    # right_2d = assemble_2d_points(right_xyz, cam_params)

    # print("Left 2D points:\n", left_2d[:5])
    # print("Right 2D points:\n", right_2d[:5])

    # visualize_2d_points(left_2d, img, dot_colour=RED, line_colour=GREEN)
    # visualize_2d_points(right_2d, img, dot_colour=BLUE, line_colour=YELLOW)
    # cv2.imshow("2D Points", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("./output/images/test/2d_points.jpg", img)
    # left_xyz = np.array(json_load("./xyz_left_000000.json"))
    # right_xyz = np.array(json_load("./xyz_right_000000.json"))

    # left_xyz = np.array(json_load("./left_3d.json"))
    # right_xyz = np.array(json_load("./right_3d.json"))

    # xyz_homogeneous = np.hstack((left_xyz[0:1], np.ones((1, 1))))
    # cam_coords = cam_params['extrinsics'] @ xyz_homogeneous.T
    # print("Camera Z:", cam_coords[2])

    # left_xyz[:, 2] = -left_xyz[:, 2]  # e.g., -1.89 → 1.89
    # right_xyz[:, 2] = -right_xyz[:, 2]
    # cam_params['extrinsics'][2, 3] -= 2.0  # Shift camera back
    # cam_params['intrinsics'][0, 2] = 946.35254  # Reset cx (undo previous +500)
    # cam_params['intrinsics'][1, 2] -= 500       # Shift v upward
    # cam_params['intrinsics'][0, 2] -= 400
    
    # print("Left XYZ sample:\n", left_xyz[:5])
    # print("Right XYZ sample:\n", right_xyz[:5])

    # left_xyz[:, 1] *= -1
    # right_xyz[:, 1] *= -1


    # R = cam_params['extrinsics'][:3, :3]
    # T = cam_params['extrinsics'][:3, 3]
    # print("Rotation:\n", R)
    # print("Translation:\n", T)
    # print("Determinant of R:", np.linalg.det(R))  # Should be ~1
    # print("Pre Transformation Intrinsics:\n", cam_params['intrinsics'])

    # cam_params['intrinsics'][0, 2] += 1000  # Adjust cx
    # cam_params['intrinsics'] = update_intrinsics_from_fov(cam_params['intrinsics'], width, height)
    
    # cam_params['extrinsics'] = invert_extrinsics(cam_params['extrinsics'])
    

    

    # Project 3D points to 2D
    # left_2d = project_3d_to_2d_distortion(left_xyz, cam_params['intrinsics'], cam_params['extrinsics'], cam_params['distortion'])
    # right_2d = project_3d_to_2d_distortion(right_xyz, cam_params['intrinsics'], cam_params['extrinsics'], cam_params['distortion'])
    # print("Left 2D points:\n", left_2d[:5])
    # print("Right 2D points:\n", right_2d[:5])


    
    # # Add a dummy confidence value to match OpenPose format (x, y, conf)
    # left_2d_conf = np.hstack((left_2d, np.ones((left_2d.shape[0], 1))))  # Shape: (21, 3)
    # right_2d_conf = np.hstack((right_2d, np.ones((right_2d.shape[0], 1))))  #

    # visualize_openpose_keypoints("./input/cam1_000000.jpg", left_2d_conf, right_2d_conf, 1)
