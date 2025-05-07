import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import json
import os
import argparse
from pathlib import Path
import time
import math

from openpose_impl import model
from openpose_impl import util
from openpose_impl.body import Body
from openpose_impl.hand import Hand

from utils.camera import load_cam_infos
from utils.image import undistort_image


def parse_args():
    parser = argparse.ArgumentParser(description='OpenPose detection')
    parser.add_argument('--save_images', default=False, action='store_true', help='Save rendered images with keypoints')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--cam_idx', type=int, default=5, help='Camera index')
    return parser.parse_args()


def enhance_image_with_blending(image, roi_points, brightness_factor=1.3):
    """
    Enhance image brightness outside ROI and blend with original ROI using Laplacian pyramid.
    
    Args:
        image: Input image
        roi_points: List of points defining ROI polygon
        brightness_factor: Factor to increase brightness outside ROI
    """
    # Create masks
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_points = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_points], 255)
    
    # Create enhanced version of the image
    enhanced = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    # Initialize output image
    result = np.zeros_like(image)
    
    # Number of pyramid levels
    levels = 4
    
    # Generate Gaussian pyramid for mask
    mask_pyramid = [mask.astype(float) / 255]
    for i in range(levels-1):
        mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))
    
    # Generate Laplacian pyramids for images
    orig_pyramid = [image.astype(float)]
    enhanced_pyramid = [enhanced.astype(float)]
    
    for i in range(levels-1):
        orig_pyramid.append(cv2.pyrDown(orig_pyramid[-1]))
        enhanced_pyramid.append(cv2.pyrDown(enhanced_pyramid[-1]))
    
    # Create Laplacian pyramids
    orig_laplacian = []
    enhanced_laplacian = []
    
    for i in range(levels-1):
        orig_size = (orig_pyramid[i].shape[1], orig_pyramid[i].shape[0])
        enhanced_size = (enhanced_pyramid[i].shape[1], enhanced_pyramid[i].shape[0])
        
        orig_up = cv2.pyrUp(orig_pyramid[i+1], dstsize=orig_size)
        enhanced_up = cv2.pyrUp(enhanced_pyramid[i+1], dstsize=enhanced_size)
        
        orig_laplacian.append(orig_pyramid[i] - orig_up)
        enhanced_laplacian.append(enhanced_pyramid[i] - enhanced_up)
    
    orig_laplacian.append(orig_pyramid[-1])
    enhanced_laplacian.append(enhanced_pyramid[-1])
    
    # Blend pyramids using mask
    blended_pyramid = []
    for orig_lap, enhanced_lap, mask_g in zip(orig_laplacian, enhanced_laplacian, mask_pyramid):
        blended = orig_lap * mask_g[..., np.newaxis] + enhanced_lap * (1 - mask_g[..., np.newaxis])
        blended_pyramid.append(blended)
    
    # Reconstruct image
    result = blended_pyramid[-1]
    for i in range(levels-2, -1, -1):
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result += blended_pyramid[i]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_image(image, alpha=1.5, beta=15):
    """Enhanced image preprocessing with multiple techniques"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Increase contrast
    enhanced_contrast = cv2.convertScaleAbs(enhanced_bgr, alpha=alpha, beta=beta)
    
    return enhanced_contrast


def transform_points(points, angle, image_shape):
    """Transform points back to original image orientation using OpenCV's rotation matrix"""
    if angle == 0 or len(points) == 0:
        return points

    height, width = image_shape[:2]
    center = (width / 2, height / 2)

    # Get the same rotation matrix that OpenCV uses
    if angle == 90:
        M = cv2.getRotationMatrix2D(center, -90, 1.0)  # Negative because we're rotating back
    elif angle == 180:
        M = cv2.getRotationMatrix2D(center, -180, 1.0)
    elif angle == 270:
        M = cv2.getRotationMatrix2D(center, -270, 1.0)
    else:
        return points

    # Convert points to homogeneous coordinates
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    # Transform points
    transformed_points = points_ones.dot(M.T)

    return transformed_points


def try_detect_hands(body_estimation, hand_estimation, image, save_enhanced=False, folder_name="enhanced", index=0):
    """Try to detect hands with different image enhancements and orientations"""
    # First try with original image
    candidate, subset = body_estimation(image)
    hands_list = util.handDetect(candidate, subset, image)
    
    if len(hands_list) == 2:
        return hands_list, candidate, subset, image, "original"
    
    # Try with enhanced image
    alpha = 0.5
    found = False
    while alpha < 3.1 and not found:
        beta = 0
        while beta < 30 and not found:
            enhanced_img = enhance_image(image, alpha=alpha, beta=beta)
            if save_enhanced:
                cv2.imwrite(f'{folder_name}/enhanced_{alpha}_{beta}_{index}.jpg', enhanced_img)
            
            # Try different rotations
            angles = [0, 90, 180, 270]
            for angle in angles:
                if angle == 0:
                    img_rotated = enhanced_img
                else:
                    if angle == 90:
                        img_rotated = cv2.rotate(enhanced_img, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        img_rotated = cv2.rotate(enhanced_img, cv2.ROTATE_180)
                    elif angle == 270:
                        img_rotated = cv2.rotate(enhanced_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                candidate_enhanced, subset_enhanced = body_estimation(img_rotated)
                hands_list_enhanced = util.handDetect(candidate_enhanced, subset_enhanced, img_rotated)
                
                if len(hands_list_enhanced) == 2:
                    # Transform points back if needed
                    if angle != 0:
                        height, width = image.shape[:2]
                        for hand in hands_list_enhanced:
                            if angle == 90:
                                hand[0], hand[1] = hand[1], width - hand[0]
                            elif angle == 180:
                                hand[0], hand[1] = width - hand[0], height - hand[1]
                            elif angle == 270:
                                hand[0], hand[1] = height - hand[1], hand[0]
                    
                    return hands_list_enhanced, candidate_enhanced, subset_enhanced, enhanced_img, "enhanced"
            
            beta += 10
        alpha += 0.5
    
    # If we found one hand, try blending
    if len(hands_list) == 1:
        x, y, w, is_left = hands_list[0]
        roi_points = np.array([
            [x - 20, y - 20],
            [x - 20, y + w + 20],
            [x + w + 20, y + w + 20],
            [x + w + 20, y - 20]
        ], dtype=np.int32)
        
        blended_img = enhance_image_with_blending(image, roi_points)
        if save_enhanced:
            cv2.imwrite(f'{folder_name}/blended_{index}.jpg', blended_img)
        
        candidate_blended, subset_blended = body_estimation(blended_img)
        hands_list_blended = util.handDetect(candidate_blended, subset_blended, blended_img)
        
        if len(hands_list_blended) == 2:
            return hands_list_blended, candidate_blended, subset_blended, blended_img, "blended"
    
    # Return best result (prefer more hands detected)
    return hands_list, candidate, subset, image, "original"


def main():
    args = parse_args()
    conf = args.conf
    cam_idx = args.cam_idx
    ORBBEC = True if cam_idx < 5 else False
    
    # Initialize paths
    input_base_path = f"./data/input/tony/Marshall/camera0{cam_idx}/images/"
    output_base_path = input_base_path.replace('input', 'output').replace('tony', 'tony/openpose/').replace('images', f'{conf:.2f}/images')
    keypoints_base_path = input_base_path.replace('images', 'keypoints').replace('input', 'output').replace('tony', 'tony/openpose').replace('keypoints', f'{conf:.2f}/keypoints')
    
    # Create output directories
    for dir_path in [output_base_path + '/success', output_base_path + '/partial', output_base_path + '/failure']:
        os.makedirs(dir_path, exist_ok=True)
    
    for dir_path in [keypoints_base_path + '/success', keypoints_base_path + '/partial', keypoints_base_path + '/failure']:
        os.makedirs(dir_path, exist_ok=True)
    
    os.makedirs(output_base_path + '/enhanced', exist_ok=True)
    
    # Initialize models
    BASE_PTH_MODELS = './openpose_impl/model/'
    body_estimation = Body(BASE_PTH_MODELS + 'body_pose_model.pth')
    hand_estimation = Hand(BASE_PTH_MODELS + 'hand_pose_model.pth')
    
    # Load camera parameters
    CALIB_DIR = './data/input/tony/'
    cam_infos = load_cam_infos(Path(CALIB_DIR), orbbec=ORBBEC)
    cam_params = cam_infos[f'camera0{cam_idx}']
    
    files = os.listdir(input_base_path)
    stats = {
        "success": 0,  # both hands
        "partial": 0,  # one hand
        "failure": 0,  # no hands
        "enhanced_success": 0,
        "blended_success": 0,
        "start_time": time.time()
    }
    
    for idx, file in enumerate(files):
        print(f"Processing {idx+1}/{len(files)}: {file}")
        image_path = os.path.join(input_base_path, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Undistort image
        if ORBBEC:
            image = undistort_image(image, cam_params, "color")
        else:
            image = cv2.undistort(
                image, 
                cam_params['intrinsics'], 
                np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0])
            )
        
        # Try detection with enhancements if needed
        hands_list, candidate, subset, processed_image, method_used = try_detect_hands(
            body_estimation, hand_estimation, image, 
            save_enhanced=True, folder_name=output_base_path + '/enhanced', 
            index=idx
        )
        
        # Initialize data structure for keypoints
        data = {
            "people": [{
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": []
            }]
        }
        
        # Process detected hands
        if hands_list:
            num_hands = len(hands_list)
            image_for_drawing = processed_image.copy()
            
            # Process each detected hand
            for hand_idx, (x, y, w, is_left) in enumerate(hands_list):
                # Extract and process hand region
                hand_roi = processed_image[y:y+w, x:x+w, :]
                peaks = hand_estimation(hand_roi)
                
                # Adjust peaks to image space
                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                
                # Convert peaks to keypoints
                hand_keypoints = []
                for i, keypoint in enumerate(peaks):
                    x_coord, y_coord = keypoint
                    hand_keypoints.extend([float(x_coord), float(y_coord), 1.0])
                
                # Store keypoints based on handedness
                if is_left:
                    data["people"][0]["hand_left_keypoints_2d"] = hand_keypoints
                else:
                    data["people"][0]["hand_right_keypoints_2d"] = hand_keypoints
                
                # Draw keypoints if requested
                if args.save_images:
                    image_for_drawing = util.draw_handpose(image_for_drawing, [peaks])
            
            # Determine success level and save
            if num_hands == 2:
                stats["success"] += 1
                if method_used != "original":
                    stats[f"{method_used}_success"] += 1
                category = "success"
            else:  # num_hands == 1
                stats["partial"] += 1
                category = "partial"
            
            # Save the annotated image and keypoints
            if args.save_images:
                output_image_path = os.path.join(output_base_path, category, file)
                cv2.imwrite(output_image_path, image_for_drawing)
            
            keypoints_file = os.path.join(keypoints_base_path, category, file.replace('.jpg', '.json'))
            with open(keypoints_file, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            stats["failure"] += 1
            # Save failed detection
            if args.save_images:
                output_image_path = os.path.join(output_base_path, 'failure', file)
                cv2.imwrite(output_image_path, processed_image)
            
            keypoints_file = os.path.join(keypoints_base_path, 'failure', file.replace('.jpg', '.json'))
            with open(keypoints_file, 'w') as f:
                json.dump(data, f, indent=4)
    
    # Calculate and print statistics
    stats["end_time"] = time.time()
    stats["total_time"] = stats["end_time"] - stats["start_time"]
    stats["total_images"] = len(files)
    stats["success_rate"] = stats["success"] / stats["total_images"] if stats["total_images"] > 0 else 0
    stats["partial_rate"] = stats["partial"] / stats["total_images"] if stats["total_images"] > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Time taken: {stats['total_time']:.2f} seconds for {stats['total_images']} images")
    print(f"Success (both hands): {stats['success']} images ({stats['success_rate']*100:.1f}%)")
    print(f"Partial (one hand): {stats['partial']} images ({stats['partial_rate']*100:.1f}%)")
    print(f"Enhanced image successes: {stats['enhanced_success']}")
    print(f"Blended image successes: {stats['blended_success']}")
    print(f"Failed (no hands): {stats['failure']} images")


if __name__ == "__main__":
    main()