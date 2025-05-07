import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import json
import os
import argparse
from pathlib import Path

from openpose_impl import model
from openpose_impl import util
from openpose_impl.body import Body
from openpose_impl.hand import Hand

from utils.camera import load_cam_infos
from utils.image import undistort_image


def parse_args():
    parser = argparse.ArgumentParser(description='OpenPose detection')
    parser.add_argument('--save_images', default=False, action='store_true', help='Save rendered images with keypoints')
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


def enhance_brightness(image, factor=1.3):
    """Simple brightness enhancement for the entire image"""
    return cv2.convertScaleAbs(image, alpha=factor, beta=25)


def enhance_image(image):
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
    alpha = 1.3  # Contrast control
    beta = 10    # Brightness control
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


def try_detect_hands(image, body_estimation, check_idx=0):
    """Try to detect hands with different image orientations"""
    best_result = {
        'hands_list': [],
        'candidate': None,
        'subset': None,
        'img_rotated': image,
        'angle': 0,
        'score': 0,
        'debug_info': {}  # Store debug information
    }
    
    angles = [0, 90, 180, 270]
    for angle in angles:
        if angle == 0:
            img_rotated = image
        else:
            # For 90 degree rotations, use cv2's built-in functions
            if angle == 90:
                img_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                img_rotated = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                img_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Try detection
        candidate, subset = body_estimation(img_rotated)
        hands_list = util.handDetect(candidate, subset, img_rotated)
        
        # Calculate detection score
        score = len(hands_list)
        if score > 0:
            if subset is not None and len(subset) > 0:
                score += np.mean(subset[subset > 0])
        
        # Store debug info
        best_result['debug_info'][angle] = {
            'hands_found': len(hands_list),
            'score': score
        }
        
        # Update best result if this is better
        if score > best_result['score']:
            best_result = {
                'hands_list': hands_list,
                'candidate': candidate,
                'subset': subset,
                'img_rotated': img_rotated,
                'angle': angle,
                'score': score,
                'debug_info': best_result['debug_info']
            }
        
        # If we found both hands, we can stop
        if len(hands_list) >= 2:
            break
    
    return (best_result['hands_list'], best_result['candidate'], 
            best_result['subset'], best_result['img_rotated'], 
            best_result['angle'])


def main():
    BASE_PTH_MODELS = './openpose_impl/model/'
    body_estimation = Body(BASE_PTH_MODELS + 'body_pose_model.pth')
    hand_estimation = Hand(BASE_PTH_MODELS + 'hand_pose_model.pth')

    CALIB_DIR = './data/input/tony/'
    cam_infos = load_cam_infos(Path(CALIB_DIR), orbbec=False)

    cam_idx = 5
    ORBBEC = True if cam_idx < 5 else False

    BASE_DIR = f'./data/input/tony/Marshall/camera0{cam_idx}/images/'

    os.makedirs(BASE_DIR.replace("input", "output"), exist_ok=True)
    os.makedirs(BASE_DIR.replace("input", "output").replace("images", "json"), exist_ok=True)

    args = parse_args()

    files = os.listdir(BASE_DIR)
    for idx, file in enumerate(files):
        cam_params = cam_infos[f'camera0{cam_idx}']
        file_path = BASE_DIR + file
        oriImg = cv2.imread(file_path)  # B,G,R order
        
        if ORBBEC:
            oriImg = undistort_image(oriImg, cam_params, "color")
        else:
            oriImg = cv2.undistort(oriImg, cam_params['intrinsics'], np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0]))

        # First try with original image
        hands_list, candidate, subset, processed_img, angle = try_detect_hands(oriImg, body_estimation)
        
        # If no hands detected or only one hand, try with enhanced image
        if len(hands_list) < 2:
            print(f"Less than two hands detected in original image {idx}/{len(files)}, trying with enhancement...")
            enhanced_img = enhance_image(oriImg)
            if args.save_images:
                cv2.imwrite(file_path.replace('input', 'output').replace('.jpg', '_enhanced.jpg'), enhanced_img)
            
            new_hands_list, new_candidate, new_subset, new_processed_img, new_angle = try_detect_hands(enhanced_img, body_estimation)
            
            # Use enhanced results if they're better
            if len(new_hands_list) > len(hands_list):
                hands_list, candidate, subset, processed_img, angle = new_hands_list, new_candidate, new_subset, new_processed_img, new_angle

        if len(hands_list) > 0:
            print(f"Detected {len(hands_list)} hands for {idx}/{len(files)} at angle {angle}")
            
            # Process detected hands
            all_hand_peaks = []
            data = {}
            data['people'] = []
            data['people'].append({})
            
            for i, (x, y, w, is_left) in enumerate(hands_list):
                # Extract hand region and process
                hand_roi = processed_img[y:y+w, x:x+w, :]
                peaks = hand_estimation(hand_roi)
                
                # First adjust peaks to the rotated image space
                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                
                # Transform back to original orientation if needed
                if angle != 0:
                    peaks = transform_points(peaks, angle, oriImg.shape)
                    # Debug print for coordinate transformation
                    print(f"Transforming points from {angle} degrees rotation:")
                    print(f"Image shape: {oriImg.shape}")
                    print(f"Original hand position: ({x}, {y})")
                    print(f"Transformed peaks shape: {peaks.shape}")
                    print(f"Sample transformed points: {peaks[:5]}")
                
                all_hand_peaks.append(peaks)
                
                hand_keypoints = []
                for i, keypoint in enumerate(peaks):
                    x, y = keypoint
                    hand_keypoints.append(int(round(x)))
                    hand_keypoints.append(int(round(y)))
                    hand_keypoints.append(1)
                data['people'][0]['hand_left_keypoints_2d' if is_left else 'hand_right_keypoints_2d'] = hand_keypoints

            with open(file_path.replace('input', 'output').replace('.jpg', '.json').replace('images', 'json'), "w") as json_file:
                json.dump(data, json_file, indent=4)

            if args.save_images:
                # Always use original image for visualization
                canvas = copy.deepcopy(oriImg)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
                cv2.imwrite(file_path.replace('input', 'output').replace('.jpg', '.png'), canvas)
        else:
            print(f"No hands detected for {idx}/{len(files)} even after enhancement")

if __name__ == "__main__":
    main()