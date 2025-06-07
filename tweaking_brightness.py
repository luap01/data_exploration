

import os
os.environ["TF_LOGGING_VERBOSITY"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", message="All log messages before absl::InitializeLog")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold(absl.logging.ERROR)

import cv2
from pathlib import Path
import numpy as np
import mediapipe as mp
import json

from utils.camera import load_cam_infos


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


def compute_bbox(roi, handside, shift):
    # Apply the same expansion logic for both hands
    half_shift = int(shift // 2)
    
    # Expand ROI in all directions
    roi[0, 0] -= half_shift        # top-left: left
    roi[0, 1] -= shift             # top-left: up
    roi[1, 0] -= half_shift        # bottom-left: left
    roi[1, 1] += shift             # bottom-left: down
    roi[2, 0] += half_shift        # bottom-right: right 
    roi[2, 1] += shift             # bottom-right: down
    roi[3, 0] += half_shift        # top-right: right
    roi[3, 1] -= shift             # top-right: up
    
    # Shift based on hand side
    if hand_side == "left":
        roi[:, 0] += shift  # Shift right for left hand
        roi[2, 0] += shift
        roi[3, 0] += shift
    else:
        roi[:, 0] -= shift  # Shift left for right hand
        roi[0, 0] -= shift
        roi[1, 0] -= shift
    return roi


def crop_to_bbox(image, bbox):
    """Crop image to bbox region, handling cases where bbox might go outside image boundaries"""
    # Get min and max coordinates
    x_coords = bbox[:, 0]
    y_coords = bbox[:, 1]
    
    # Get bbox boundaries
    x_min = max(0, int(min(x_coords)))
    y_min = max(0, int(min(y_coords)))
    x_max = min(image.shape[1], int(max(x_coords)))
    y_max = min(image.shape[0], int(max(y_coords)))
    
    # Crop the image
    return image[y_min:y_max, x_min:x_max], (x_min, y_min)


def crop_fixed_size(image, hand_landmarks, size=256, base_offset=(0, 0), current_image_size=None):
    """
    Crop a square region of specified size around the root joint (wrist).
    Handles boundary conditions and ensures the crop is within image bounds.
    
    Args:
        image: Input image to crop from
        hand_landmarks: MediaPipe hand landmarks (coordinates relative to current_image_size)
        size: Size of the square crop
        base_offset: (x_offset, y_offset) from previous crops to chain transformations
        current_image_size: (height, width) of the image where hand_landmarks were detected
    
    Returns:
        cropped_image, total_offset_from_original
    """
    h, w = image.shape[:2]
    
    # If current_image_size is provided, use it to convert normalized coordinates
    # Otherwise, assume hand_landmarks are relative to the input image
    if current_image_size is not None:
        curr_h, curr_w = current_image_size
    else:
        curr_h, curr_w = h, w
    
    # Get root joint (wrist) coordinates - landmark[0]
    # Convert normalized coordinates to pixel coordinates in the detected image space
    root_x_in_detected = int(hand_landmarks.landmark[0].x * curr_w)
    root_y_in_detected = int(hand_landmarks.landmark[0].y * curr_h)
    
    # Transform to original image coordinates by adding base_offset
    root_x = root_x_in_detected + base_offset[0]
    root_y = root_y_in_detected + base_offset[1]
    
    # Calculate crop boundaries
    half_size = size // 2
    x_min = max(0, root_x - half_size)
    y_min = max(0, root_y - half_size)
    x_max = min(w, root_x + half_size)
    y_max = min(h, root_y + half_size)
    
    # If crop would go out of bounds, adjust the crop region while maintaining size
    if x_min == 0:
        x_max = min(w, size)
    if y_min == 0:
        y_max = min(h, size)
    if x_max == w:
        x_min = max(0, w - size)
    if y_max == h:
        y_min = max(0, h - size)
    
    # Crop the image
    cropped = image[y_min:y_max, x_min:x_max]
    
    # Calculate total offset from original image (chain the transformations)
    x_offset = x_min
    y_offset = y_min
    
    # If the cropped image is smaller than desired size (happens at image boundaries)
    # pad it with zeros to maintain the desired size
    if cropped.shape[0] != size or cropped.shape[1] != size:
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        pad_y = (size - cropped.shape[0]) // 2
        pad_x = (size - cropped.shape[1]) // 2
        padded[pad_y:pad_y+cropped.shape[0], pad_x:pad_x+cropped.shape[1]] = cropped
        return padded, (x_offset, y_offset)
    
    return cropped, (x_offset, y_offset)

def get_keypoints(data, image, hand_landmarks, hand_type, coord_origin):
    # Process landmarks into keypoints
    keypoints = []

    crop_x_min, crop_y_min = coord_origin[-1]

    if len(coord_origin) > 1:
        captured_x_min, captured_y_min = coord_origin[-2][0], coord_origin[-2][1]
    else:
        captured_x_min, captured_y_min = 0, 0
    # x_orig, y_orig = 0, 0
    
    # for x,y in coord_origin:
    #     x_orig += x
    #     y_orig += y

    for landmark in hand_landmarks.landmark:
        keypoints.extend([float(landmark.x * image.shape[1]) - crop_x_min + captured_x_min, float(landmark.y * image.shape[0]) - crop_y_min + captured_y_min, 1.0])

    if hand_type == "left":
        data["people"][0]["hand_left_keypoints_2d"] = keypoints
        data["people"][0]["hand_left_shift"] = [crop_x_min, crop_y_min]
    else:  # right
        data["people"][0]["hand_right_keypoints_2d"] = keypoints
        data["people"][0]["hand_right_shift"] = [crop_x_min, crop_y_min]
    return data


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# img_path = "data/input/tony/Marshall/camera05/images/color_000507_camera01.jpg"
img_idx = "000510"
shift = 250
# for idx in range(101, 14018):
for idx in range(100, 500):
    img_idx = str.zfill(str(idx), 6)
    print(img_idx)
    img_path = f"data/input/tony/Marshall/camera05/images/color_{img_idx}_camera01.jpg"
    img = cv2.imread(img_path)
    cam_params = load_cam_infos(Path("./data/input/tony/"), orbbec=False)[f'camera05']

    if img is None:
        print(f"Failed to load image: {img_path}")
        continue

    img = cv2.undistort(
        img, 
        cam_params['intrinsics'], 
        np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0])
    )

    alpha = 0
    found = False
    while alpha < 4.1 and not found:
        beta = 0
        while beta < 51 and not found:
            enhanced_img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            results = hands.process(enhanced_rgb)
            if results and results.multi_hand_landmarks:
                # print(alpha, beta)
                found = True
            beta += 10
        alpha += 0.3

    # Initialize data structure for keypoints
    data = {
        "people": [{
            "hand_left_shift": [],
            "hand_left_keypoints_2d": [],
            "hand_right_shift": [],
            "hand_right_keypoints_2d": []
        }]
    }

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        blank_img, crop_origin = crop_fixed_size(img.copy(), results.multi_hand_landmarks[0], size=256)
        hand_side = results.multi_handedness[0].classification[0].label.lower()
        cv2.imwrite(f"test_bbox_comp/blanks/{img_idx}_cropped_256_{hand_side.lower()}_blank.jpg", blank_img)
        data = get_keypoints(data, img, results.multi_hand_landmarks[0], hand_side, [crop_origin])

        blank_img, crop_origin = crop_fixed_size(img.copy(), results.multi_hand_landmarks[1], size=256)
        hand_side = results.multi_handedness[1].classification[0].label.lower()
        cv2.imwrite(f"test_bbox_comp/blanks/{img_idx}_cropped_256_{hand_side.lower()}_blank.jpg", blank_img)
        data = get_keypoints(data, img, results.multi_hand_landmarks[1], hand_side, [crop_origin])


    elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        hand_side = results.multi_handedness[0].classification[0].label.lower()
        # print(hand_side)
        # print(len(results.multi_hand_landmarks))

        img_to_process = img.copy()
        # Crop 256x256 region around root joint
        blank_img, crop_origin = crop_fixed_size(img.copy(), results.multi_hand_landmarks[0], size=256)
        cv2.imwrite(f"test_bbox_comp/blanks/{img_idx}_cropped_256_{hand_side.lower()}_blank.jpg", blank_img)
        data = get_keypoints(data, img, results.multi_hand_landmarks[0], hand_side, [crop_origin])

        first_img = img.copy()
        for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            mp_drawing.draw_landmarks(
                first_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Crop 256x256 region around root joint
        cropped_img, _ = crop_fixed_size(first_img, results.multi_hand_landmarks[0], size=256)
        cv2.imwrite(f"test_bbox_comp/preds/{img_idx}_cropped_256_{hand_side.lower()}.jpg", cropped_img)
        
        # If you still want the original bbox visualization
        roi = get_roi_points(results.multi_hand_landmarks[0], img.shape)
        bbox = compute_bbox(roi.copy(), hand_side, shift)
        cv2.imwrite(f"test_bbox_comp/original/{img_idx}_test.jpg", img_to_process)

        # Draw bbox points and lines on the full image
        for x, y in bbox:
            # print(x,y)
            cv2.circle(first_img, (int(x), int(y)), 4, (0, 255, 255), -1)

        cv2.line(first_img, bbox[0], bbox[1], (255, 0, 255), 1)
        cv2.line(first_img, bbox[1], bbox[2], (255, 0, 255), 1)
        cv2.line(first_img, bbox[2], bbox[3], (255, 0, 255), 1)
        cv2.line(first_img, bbox[3], bbox[0], (255, 0, 255), 1)
        
        # Draw ROI points and lines on the full image
        for x,y in roi:
            cv2.circle(first_img, (int(x), int(y)), 4, (0, 255, 0), -1)

        cv2.line(first_img, roi[0], roi[1], (255, 0, 255), 1)
        cv2.line(first_img, roi[1], roi[2], (255, 0, 255), 1)
        cv2.line(first_img, roi[2], roi[3], (255, 0, 255), 1)
        cv2.line(first_img, roi[3], roi[0], (255, 0, 255), 1)

        # Save the full image with bbox visualization
        # cv2.imwrite(f"test_bbox_comp/shifted_roi/{img_idx}_test_{shift}.jpg", first_img)
        
        first_try = True
        retry = False
        fail = False
        while first_try or retry:
            cropped_img, bbox_origin = crop_to_bbox(img_to_process, bbox)
            # cv2.imshow("img", cropped_img)
            # cv2.waitKey(0)
            alpha = 0
            found = False
            while alpha < 4.1 and not found:
                beta = 0
                while beta < 51 and not found:
                    enhanced_img = np.clip(cropped_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                    results = hands.process(enhanced_rgb)
                    if results and results.multi_hand_landmarks:
                        # print(alpha, beta)
                        found = True
                        if retry:
                            retry = False
                            data["people"][0][f"hand_{hand_side}_keypoints_2d"] = data["people"][0][f"hand_{prev_detected_handside}_keypoints_2d"]
                            data["people"][0][f"hand_{hand_side}_shift"] = data["people"][0][f"hand_{prev_detected_handside}_shift"]
                            cv2.imwrite(f"test_bbox_comp/blanks/{img_idx}_cropped_256_{hand_side.lower()}_blank.jpg", blank_img)
                    beta += 10
                alpha += 0.3

            first_try = False
            try:
                length = len(results.multi_hand_landmarks)
                hand_side = results.multi_handedness[0].classification[0].label.lower()
            except Exception:
                if not retry:
                    retry = True
                    prev_detected_handside = hand_side
                    hand_side = "left" if prev_detected_handside == "right" else "right"
                    bbox = compute_bbox(roi.copy(), hand_side, shift)
                    retry_img = img.copy()
                    # Draw bbox points and lines on the full image
                    for x, y in bbox:
                        # print(x,y)
                        cv2.circle(retry_img, (int(x), int(y)), 4, (0, 255, 255), -1)

                    cv2.line(retry_img, bbox[0], bbox[1], (255, 0, 255), 1)
                    cv2.line(retry_img, bbox[1], bbox[2], (255, 0, 255), 1)
                    cv2.line(retry_img, bbox[2], bbox[3], (255, 0, 255), 1)
                    cv2.line(retry_img, bbox[3], bbox[0], (255, 0, 255), 1)
                    
                    # Draw ROI points and lines on the full image
                    for x,y in roi:
                        cv2.circle(retry_img, (int(x), int(y)), 4, (0, 255, 0), -1)

                    cv2.line(retry_img, roi[0], roi[1], (255, 0, 255), 1)
                    cv2.line(retry_img, roi[1], roi[2], (255, 0, 255), 1)
                    cv2.line(retry_img, roi[2], roi[3], (255, 0, 255), 1)
                    cv2.line(retry_img, roi[3], roi[0], (255, 0, 255), 1)

                    # Save the full image with bbox visualization
                    cv2.imwrite(f"test_bbox_comp/shifted_roi/{img_idx}_test_{shift}_retry.jpg", retry_img)
                else:
                    retry = False
                    print(f"Failed to detect second hand for {img_idx}")
                    cv2.imwrite(f"test_bbox_comp/failed/{img_idx}_test_{shift}_retry.jpg", retry_img)
                    cv2.imwrite(f"test_bbox_comp/failed/{img_idx}_test_{shift}.jpg", first_img)
                    fail = True

        if fail:
            continue
        
        # Save the full image with bbox visualization
        cv2.imwrite(f"test_bbox_comp/shifted_roi/{img_idx}_test_{shift}.jpg", first_img)

        # print(hand_side)
        roi = get_roi_points(results.multi_hand_landmarks[0], cropped_img.shape)
        bbox = compute_bbox(roi.copy(), hand_side, shift)

        blank_img, crop_origin = crop_fixed_size(img, results.multi_hand_landmarks[0], size=256, base_offset=bbox_origin, current_image_size=cropped_img.shape[:2])
        cv2.imwrite(f"test_bbox_comp/blanks/{img_idx}_cropped_256_{hand_side}_blank.jpg", blank_img)
        data = get_keypoints(data, cropped_img, results.multi_hand_landmarks[0], hand_side, [bbox_origin, crop_origin])

        for x, y in bbox:
            cv2.circle(cropped_img, (int(x), int(y)), 4, (0, 255, 255), -1)

        cv2.line(cropped_img, bbox[0], bbox[1], (255, 0, 255), 1)
        cv2.line(cropped_img, bbox[1], bbox[2], (255, 0, 255), 1)
        cv2.line(cropped_img, bbox[2], bbox[3], (255, 0, 255), 1)
        cv2.line(cropped_img, bbox[3], bbox[0], (255, 0, 255), 1)


        for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            mp_drawing.draw_landmarks(
                cropped_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
        # Crop 256x256 region around root joint
        cropped_img, _ = crop_fixed_size(cropped_img, results.multi_hand_landmarks[0], size=256)
        cv2.imwrite(f"test_bbox_comp/preds/{img_idx}_cropped_256_{hand_side.lower()}.jpg", cropped_img)


        # cv2.imshow("img", cropped_img)
        # cv2.waitKey(0)
        
        with open(f'test_bbox_comp/json/{img_idx}_test.json', 'w') as f:
            json.dump(data, f, indent=4)



