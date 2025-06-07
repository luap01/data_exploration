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

def enhance_image_with_blending(image, roi_points, brightness_factor=1.5, levels=4):
    """
    Enhance image brightness outside ROI and blend with original ROI using Laplacian pyramid.
    
    Args:
        image: Input image
        roi_points: List of points defining ROI polygon
        brightness_factor: Factor to increase brightness outside ROI
        levels: Number of pyramid levels for blending
    """
    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_points = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_points], 255)
    
    # Create enhanced version of the image
    enhanced = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    # Initialize output image
    result = np.zeros_like(image)
    
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
        blended = orig_lap * (1 - mask_g[..., np.newaxis]) + enhanced_lap * mask_g[..., np.newaxis]
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
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def count_keypoints(peaks):
    """Count non-zero keypoints"""
    return np.sum(peaks[:, 0] > 0)

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

        # Step 1: Try to detect hands in original image
        candidate, subset = body_estimation(oriImg)
        hands_list = util.handDetect(candidate, subset, oriImg)
        current_img = oriImg

        if len(hands_list) < 2:
            all_hand_peaks = []
            for x, y, w, is_left in hands_list:
                peaks = hand_estimation(current_img[y:y+w, x:x+w, :])
                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                all_hand_peaks.append(peaks)
            
            if all(count_keypoints(peaks) == 21 for peaks in all_hand_peaks) and len(all_hand_peaks) > 0:  # Assuming 21 keypoints for a full hand
                print(f"All hands detected with full keypoints for {idx}/{len(files)}")
            else:
                print(f"Some keypoints missing in original image {idx}/{len(files)}, enhancing brightness...")
                # Step 2: Enhance brightness if not all keypoints are found
                current_img = enhance_brightness(current_img, factor=3.3)
                cv2.imwrite(file_path.replace('input', 'output').replace('.jpg', '_bright.jpg'), current_img)
                candidate, subset = body_estimation(current_img)
                hands_list = util.handDetect(candidate, subset, current_img)
                
                if len(hands_list) > 0:
                    all_hand_peaks = []
                    for x, y, w, is_left in hands_list:
                        peaks = hand_estimation(current_img[y:y+w, x:x+w, :])
                        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                        all_hand_peaks.append(peaks)
                    
                    if all(count_keypoints(peaks) == 21 for peaks in all_hand_peaks) and len(all_hand_peaks) > 0:
                        print(f"All hands detected with full keypoints after brightness for {idx}/{len(files)}")
                    else:
                        print(f"Still missing keypoints after brightness for {idx}/{len(files)}, applying blending...")
                        # Step 3: Apply blending if still missing keypoints
                        roi_points = []
                        padding = 100
                        for x, y, w, _ in hands_list:
                            roi_points.extend([
                                [max(0, x - padding), max(0, y - padding)],
                                [min(current_img.shape[1], x + w + padding), max(0, y - padding)],
                                [min(current_img.shape[1], x + w + padding), min(current_img.shape[0], y + w + padding)],
                                [max(0, x - padding), min(current_img.shape[0], y + w + padding)]
                            ])
                        
                        current_img = enhance_image_with_blending(current_img, np.array(roi_points))
                        cv2.imwrite(file_path.replace('input', 'output').replace('.jpg', '_blended.jpg'), current_img)
                        candidate, subset = body_estimation(current_img)
                        hands_list = util.handDetect(candidate, subset, current_img)
                        
                        if len(hands_list) > 0:
                            all_hand_peaks = []
                            for x, y, w, is_left in hands_list:
                                peaks = hand_estimation(current_img[y:y+w, x:x+w, :])
                                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                                all_hand_peaks.append(peaks)
                            
                            if all(count_keypoints(peaks) == 21 for peaks in all_hand_peaks):
                                print(f"All hands detected with full keypoints after blending for {idx}/{len(files)}")
                            else:
                                print(f"Still missing keypoints after blending for {idx}/{len(files)}")
                        else:
                            print(f"No hands detected after blending for {idx}/{len(files)}")
                else:
                    print(f"No hands detected after brightness enhancement for {idx}/{len(files)}")
        # Process detected hands and save results
        if len(hands_list) > 0:
            all_hand_peaks = []
            data = {}
            data['people'] = []
            data['people'].append({})
            
            for i, (x, y, w, is_left) in enumerate(hands_list):
                peaks = hand_estimation(current_img[y:y+w, x:x+w, :])
                peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
                all_hand_peaks.append(peaks)
                
                hand_keypoints = []
                for i, keypoint in enumerate(peaks):
                    x, y = keypoint
                    hand_keypoints.append(int(x))
                    hand_keypoints.append(int(y))
                    hand_keypoints.append(1)
                data['people'][0]['hand_left_keypoints_2d' if is_left else 'hand_right_keypoints_2d'] = hand_keypoints

            with open(file_path.replace('input', 'output').replace('.jpg', '.json').replace('images', 'json'), "w") as json_file:
                json.dump(data, json_file, indent=4)

            if args.save_images:
                canvas = util.draw_handpose(current_img, all_hand_peaks)
                cv2.imwrite(file_path.replace('input', 'output').replace('.jpg', '.png'), canvas)

        break

if __name__ == "__main__":
    main()