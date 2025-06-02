import cv2
import mediapipe as mp
import numpy as np
import os 
import time
import json
import math
from pathlib import Path
import argparse

from utils.camera import load_cam_infos
from utils.image import undistort_image

DEFAULT_CONF = 0.10

def parse_args():
    parser = argparse.ArgumentParser(description='OpenPose detection')
    parser.add_argument('--save_images', default=False, action='store_true', help='Save rendered images with keypoints')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF, help='Detection confidence threshold')
    parser.add_argument('--cam_idx', type=int, default=5, help='Camera index')
    return parser.parse_args()

def enhance_image_with_blending(image, roi_points, alpha_bright=1.2, beta_bright=10, alpha_dark=0.8, beta_dark=-10):
    """
    Enhance image brightness outside ROI and reduce it inside ROI using Laplacian pyramid.
    brightness_factor_outside > 1 increases brightness outside ROI
    brightness_factor_inside < 1 decreases brightness inside ROI
    """
    # Create masks
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_points = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_points], 255)
    
    # Apply Gaussian blur to mask for smoother transitions
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    
    # Create two enhanced versions of the image
    enhanced_bright = np.clip(image.astype(np.float32) * alpha_bright + beta_bright, 0, 255).astype(np.uint8)
    enhanced_dark = np.clip(image.astype(np.float32) * alpha_dark + beta_dark, 0, 255).astype(np.uint8)
    
    # Initialize output image
    result = np.zeros_like(image)
    
    # Number of pyramid levels - increased for smoother blending
    levels = 6
    
    # Generate Gaussian pyramid for mask
    mask_pyramid = [mask.astype(float) / 255]
    for i in range(levels-1):
        mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))
    
    # Generate Laplacian pyramids for images
    dark_pyramid = [enhanced_dark.astype(float)]
    bright_pyramid = [enhanced_bright.astype(float)]
    
    for i in range(levels-1):
        dark_pyramid.append(cv2.pyrDown(dark_pyramid[-1]))
        bright_pyramid.append(cv2.pyrDown(bright_pyramid[-1]))
    
    # Create Laplacian pyramids
    dark_laplacian = []
    bright_laplacian = []
    
    for i in range(levels-1):
        dark_size = (dark_pyramid[i].shape[1], dark_pyramid[i].shape[0])
        bright_size = (bright_pyramid[i].shape[1], bright_pyramid[i].shape[0])
        
        dark_up = cv2.pyrUp(dark_pyramid[i+1], dstsize=dark_size)
        bright_up = cv2.pyrUp(bright_pyramid[i+1], dstsize=bright_size)
        
        dark_laplacian.append(dark_pyramid[i] - dark_up)
        bright_laplacian.append(bright_pyramid[i] - bright_up)
    
    dark_laplacian.append(dark_pyramid[-1])
    bright_laplacian.append(bright_pyramid[-1])
    
    # Blend pyramids using mask
    blended_pyramid = []
    for dark_lap, bright_lap, mask_g in zip(dark_laplacian, bright_laplacian, mask_pyramid):
        # Inside ROI (mask=1) use darker image, outside (mask=0) use brighter image
        blended = dark_lap * mask_g[..., np.newaxis] + bright_lap * (1 - mask_g[..., np.newaxis])
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

def process_hand_landmarks(hand_landmarks, handedness, image_shape, angle=0):
    """Convert MediaPipe hand landmarks to a list of [x, y, confidence] format"""
    h, w = image_shape[:2]
    keypoints = []

    # If rotated 90 or 270 degrees, swap width and height
    if angle in [90, 270]:
        w, h = h, w
    
    for landmark in hand_landmarks.landmark:
        # Convert normalized coordinates to pixel coordinates
        x = landmark.x
        y = landmark.y

        x, y = simple_rotate_point(x, y, angle)

        x = x * w
        y = y * h

        # MediaPipe provides confidence per hand, not per keypoint
        confidence = 1.0
        keypoints.extend([float(x), float(y), confidence])
    
    return keypoints


def try_angles(hands, image):
    """Try different image enhancements and orientations"""
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
        
        results_enhanced = hands.process(img_rotated)
        if results_enhanced.multi_hand_landmarks:
            return results_enhanced, angle, img_rotated
    return None, 0, image


def rotate_point(x, y, angle_degrees, clockwise=True):
    angle_radians = math.radians(-angle_degrees if clockwise else angle_degrees)
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    return x_new, y_new

def simple_rotate_point(x, y, angle_degrees):
    if angle_degrees == 90:
        return -y, x
    elif angle_degrees == 180:
        return -x, -y
    elif angle_degrees == 270:
        return y, -x
    return x, y


def compute_bbox():
    pass

def get_roi_points(hand_landmarks, image_shape):
    """Get ROI points in original image coordinate system"""
    h, w = image_shape[:2]
    
    points = []
    for landmark in hand_landmarks.landmark:
        x = landmark.x * w
        y = landmark.y * h

        points.append((int(x), int(y)))
    
    x_coords, y_coords = zip(*points)
    roi_points = np.array([
        [min(x_coords) - 20, min(y_coords) - 20],
        [min(x_coords) - 20, max(y_coords) + 20],
        [max(x_coords) + 20, max(y_coords) + 20],
        [max(x_coords) + 20, min(y_coords) - 20]
    ], dtype=np.int32)
    return roi_points

def detect_hands(hands, image, save_enhanced=False, folder_name="enhanced", index=0):
    """Attempt hand detection with various image enhancements"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        return results, image, "original", 0
    
    # First enhancement: Basic image enhancement
    alpha = 1
    found = False
    detected_angle = 0
    detected_alpha = 0
    # alpha used to be <4.1 and beta <51
    while alpha < 4.1 and not found:
        detected_alpha = alpha
        beta = 0
        while beta < 51 and not found:
            # enhanced_img = enhance_image(image, alpha=alpha, beta=beta)
            enhanced_img = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            results_enhanced, angle, rotated_image = try_angles(hands, enhanced_rgb)
            if results_enhanced and results_enhanced.multi_hand_landmarks:
                found = True
                detected_angle = angle
                image = rotated_image
                print(f"Alpha: {alpha}, Beta: {beta}, Angle: {detected_angle}, Hand: {results_enhanced.multi_handedness[0].classification[0].label}")
                break
            beta += 10
        alpha += 0.3
    

    if results_enhanced and results_enhanced.multi_hand_landmarks and len(results_enhanced.multi_hand_landmarks) == 2:
        if save_enhanced:
            cv2.imwrite(f'{folder_name}/enhanced_success_{index}.jpg', enhanced_img)
        return results_enhanced, enhanced_img, "enhanced", detected_angle
    
    # Second enhancement: Try with ROI blending if we detected at least one hand
    if results_enhanced and results_enhanced.multi_hand_landmarks and len(results_enhanced.multi_hand_landmarks) == 1:
        # Get ROI from the detected hand in original coordinate system
        hand_landmarks = results_enhanced.multi_hand_landmarks[0]
        roi_points = get_roi_points(hand_landmarks, image.shape)  # No rotation for original detection

        alpha_bright = 1.0
        alpha_dark = 1.0
        while alpha_bright < 2.0 and alpha_dark > 0.0:
            beta_bright = 10
            beta_dark = -10
            while beta_bright < 101 and beta_dark > -101:
                blended_img = enhance_image_with_blending(image, roi_points, alpha_bright=alpha_bright, beta_bright=beta_bright, alpha_dark=alpha_dark, beta_dark=beta_dark)
                blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"test_blended/{alpha_bright}_{beta_bright}_{alpha_dark}_{beta_dark}.jpg", blended_rgb)
                results_blended = hands.process(blended_rgb)
                if results_blended.multi_hand_landmarks and len(results_blended.multi_hand_landmarks) == 1:
                    print(f"Blending: Alpha: {alpha}, Beta: {beta}, Angle: {detected_angle}, Hand: {results_blended.multi_handedness[0].classification[0].label}")
                if results_blended.multi_hand_landmarks and len(results_blended.multi_hand_landmarks) == 2:
                    if save_enhanced:
                        cv2.imwrite(f'{folder_name}/blended_success_{index}.jpg', blended_img)
                    return results_blended, blended_img, "blended", detected_angle


                beta_dark -= 10
                beta_bright += 10
            alpha_dark -= 0.3
            alpha_bright += 0.3

        
        alpha = 1.0
        while alpha > 0.0:
            beta = 0
            while beta > -51:
                enhanced_img = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                results_darker = hands.process(enhanced_rgb)
                if results_darker.multi_hand_landmarks and len(results_darker.multi_hand_landmarks) == 1:
                    print(f"Darker: Alpha: {alpha}, Beta: {beta}, Angle: {detected_angle}, Hand: {results_darker.multi_handedness[0].classification[0].label}")
                if results_darker.multi_hand_landmarks and len(results_darker.multi_hand_landmarks) == 2:
                    if save_enhanced:
                        cv2.imwrite(f'{folder_name}/blended_success_{index}.jpg', blended_img)
                    return results_blended, blended_img, "blended", detected_angle
                beta -= 5
            alpha -= 0.3
    
    # Return best result (prefer more hands detected)
    if results_blended and results_blended.multi_hand_landmarks and len(results_blended.multi_hand_landmarks) > len(results_enhanced.multi_hand_landmarks or []):
        return results_blended, blended_img, "blended", detect_anlge
    elif results_enhanced and results_enhanced.multi_hand_landmarks and len(results_enhanced.multi_hand_landmarks) > len(results.multi_hand_landmarks or []):
        return results_enhanced, enhanced_img, "enhanced", detected_angle
    return results, image, "original", detected_angle


def transform_landmarks(landmarks, angle, image_shape):
    """Transform landmarks to new angle considering image dimensions"""
    h, w = image_shape[:2]
    
    # If rotated 90 or 270 degrees, swap width and height for the rotated image
    if angle in [90, 270]:
        w, h = h, w
    
    for landmark in landmarks.landmark:
        # Convert from normalized to pixel coordinates in rotated image
        x = landmark.x * w
        y = landmark.y * h
        
        # Rotate point
        if angle == 90:
            x_new, y_new = y, w - x
        elif angle == 180:
            x_new, y_new = w - x, h - y
        elif angle == 270:
            x_new, y_new = h - y, x
        else:  # angle == 0
            x_new, y_new = x, y
        
        # Convert back to normalized coordinates in original image
        landmark.x = x_new / image_shape[1]  # original width
        landmark.y = y_new / image_shape[0]  # original height
    
    return landmarks

def main():
    args = parse_args()
    # Initialize MediaPipe Hands
    conf = args.conf
    cam_idx = args.cam_idx
    save = bool(args.save_images)
    ORBBEC = True if cam_idx < 5 else False
    input_base_path = f"./data/input/tony/Marshall/camera0{cam_idx}/images/"
    output_base_path = input_base_path.replace('input', 'output').replace('tony', 'tony/mediapipe').replace('images', f'{conf:.2f}/images')
    keypoints_base_path = input_base_path.replace('images', 'keypoints').replace('input', 'output').replace('tony', 'tony/mediapipe').replace('keypoints', f'{conf:.2f}/keypoints')

    # Create output directories
    for dir_path in [output_base_path + '/success', output_base_path + '/partial', output_base_path + '/failure']:
        os.makedirs(dir_path, exist_ok=True)

    for dir_path in [keypoints_base_path + '/success', keypoints_base_path + '/partial', keypoints_base_path + '/failure']:
        os.makedirs(dir_path, exist_ok=True)

    os.makedirs(output_base_path + '/enhanced', exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=conf,
        min_tracking_confidence=0.5
    )

    CALIB_DIR = './data/input/tony/'
    cam_infos = load_cam_infos(Path(CALIB_DIR), orbbec=ORBBEC)
    cam_params = cam_infos[f'camera0{cam_idx}']

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    files = sorted(os.listdir(input_base_path))
    stats = {
        "success": 0,  # both hands
        "partial": 0,  # one hand
        "failure": 0,  # no hands
        "enhanced_success": 0,
        "blended_success": 0,
        "start_time": time.time()
    }
    files = ['cropped_150.jpg']
    for idx, file in enumerate(files):
        # if file != "color_000507_camera01.jpg":
        #     continue
        # if idx < 506:
        #    continue
        
        if file != "cropped_150.jpg":
            continue
        print(file)
        if idx > 510:
            break
        image_path = os.path.join(input_base_path, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        if ORBBEC:
            image = undistort_image(image, cam_params, "color")
        else:
            image = cv2.undistort(
                image, 
                cam_params['intrinsics'], 
                np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0])
            )

        # Try detection with enhancements if needed
        results, processed_image, method_used, detected_angle = detect_hands(hands, image, save_enhanced=save, folder_name=f"{output_base_path}/enhanced", index=idx)
        
        # Initialize data structure for keypoints
        data = {
            "people": [{
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": []
            }]
        }
        
        # Process detected hands
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            image_for_drawing = image.copy()

            # Process each detected hand
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                
                retransformed_hand_landmarks = transform_landmarks(hand_landmarks, detected_angle, image.shape)
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    image_for_drawing,
                    retransformed_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Process landmarks into keypoints
                keypoints = []
                for landmark in retransformed_hand_landmarks.landmark:
                    keypoints.extend([float(landmark.x * image.shape[1]), float(landmark.y * image.shape[0]), 1.0])
                
                # Store keypoints based on handedness
                hand_type = handedness.classification[0].label.lower()
                if hand_type == "left":
                    data["people"][0]["hand_left_keypoints_2d"] = keypoints
                else:  # right
                    data["people"][0]["hand_right_keypoints_2d"] = keypoints
            
            # Determine success level and save accordingly
            if num_hands == 2:
                stats["success"] += 1
                if method_used != "original":
                    stats[f"{method_used}_success"] += 1
                category = "success"
            else:  # num_hands == 1
                stats["partial"] += 1
                category = "partial"
                
            # Save the annotated image and keypoints
            output_image_path = os.path.join(output_base_path, category, file)
            keypoints_file = os.path.join(keypoints_base_path, category, file.replace('.jpg', '.json'))
            
            cv2.imwrite(output_image_path, image_for_drawing)
            with open(keypoints_file, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            stats["failure"] += 1
            # Save failed detection
            output_image_path = os.path.join(output_base_path, 'failure', file)
            keypoints_file = os.path.join(keypoints_base_path, 'failure', file.replace('.jpg', '.json'))
            cv2.imwrite(output_image_path, processed_image)
            with open(keypoints_file, 'w') as f:
                json.dump(data, f, indent=4)

    # Release resources
    hands.close()

    # Calculate and print statistics
    stats["end_time"] = time.time()
    stats["total_time"] = stats["end_time"] - stats["start_time"]
    stats["total_images"] = len(files)
    stats["success_rate"] = stats["success"] / stats["total_images"] if stats["total_images"] > 0 else 0
    stats["partial_rate"] = stats["partial"] / stats["total_images"] if stats["total_images"] > 0 else 0

    print(f"Processing complete!")
    print(f"Time taken: {stats['total_time']:.2f} seconds for {stats['total_images']} images")
    print(f"Success (both hands): {stats['success']} images ({stats['success_rate']*100:.1f}%)")
    print(f"Partial (one hand): {stats['partial']} images ({stats['partial_rate']*100:.1f}%)")
    print(f"Enhanced image successes: {stats['enhanced_success']}")
    print(f"Blended image successes: {stats['blended_success']}")
    print(f"Failed (no hands): {stats['failure']} images")


if __name__ == "__main__":
    main()