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
    if handside == "left":
        roi[:, 0] += shift
    else:
        half_shift = int(shift)
        # roi[:, 0] -= shift
        roi[0, :] -= half_shift
        roi[1, 0] -= half_shift
        roi[1, 1] += half_shift
        roi[2, :] += half_shift
        roi[3, 0] += half_shift
        roi[3, 1] -= half_shift
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
    return image[y_min:y_max, x_min:x_max]


def crop_fixed_size(image, hand_landmarks, size=256):
    """
    Crop a square region of specified size around the middle finger MCP joint (knuckle).
    Handles boundary conditions and ensures the crop is within image bounds.
    """
    h, w = image.shape[:2]
    
    # Get middle finger MCP joint (knuckle) coordinates - landmark index 9
    # root joint (wrist) is landmark[0]
    root_x = int(hand_landmarks.landmark[0].x * w)  # landmark[9] is middle finger MCP
    root_y = int(hand_landmarks.landmark[0].y * h)
    
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
    
    # If the cropped image is smaller than desired size (happens at image boundaries)
    # pad it with zeros to maintain the desired size
    if cropped.shape[0] != size or cropped.shape[1] != size:
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - cropped.shape[0]) // 2
        x_offset = (size - cropped.shape[1]) // 2
        padded[y_offset:y_offset+cropped.shape[0], x_offset:x_offset+cropped.shape[1]] = cropped
        return padded
    
    return cropped, (x_min, y_min)

def get_keypoints(data, image, hand_landmarks, hand_type, bbox_origin):
    # Process landmarks into keypoints
    keypoints = []
    x_min, y_min = bbox_origin
    for landmark in hand_landmarks.landmark:
        keypoints.extend([float(landmark.x * image.shape[1]) - x_min, float(landmark.y * image.shape[0]) - y_min, 1.0])

    if hand_type == "left":
        data["people"][0]["hand_left_keypoints_2d"] = keypoints
    else:  # right
        data["people"][0]["hand_right_keypoints_2d"] = keypoints
    
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


img_path = "data/input/tony/Marshall/camera05/images/color_000507_camera01.jpg"
img = cv2.imread(img_path)
cam_params = load_cam_infos(Path("./data/input/tony/"), orbbec=False)[f'camera05']

if img is None:
    print(f"Failed to load image: {img_path}")

img = cv2.undistort(
    img, 
    cam_params['intrinsics'], 
    np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0])
)

alpha = 1.6
beta = 20
enhanced_img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)

results = hands.process(enhanced_rgb)

# Initialize data structure for keypoints
data = {
    "people": [{
        "hand_left_keypoints_2d": [],
        "hand_right_keypoints_2d": []
    }]
}

if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
    hand_side = results.multi_handedness[0].classification[0].label.lower()
    print(hand_side)
    print(len(results.multi_hand_landmarks))

    img_to_process = img.copy()
    # Crop 256x256 region around root joint
    blank_img, crop_origin = crop_fixed_size(img, results.multi_hand_landmarks[0], size=256)
    cv2.imwrite(f"test_bbox/cropped_256_{hand_side.lower()}_blank.jpg", blank_img)
    data = get_keypoints(data, img, results.multi_hand_landmarks[0], hand_side, crop_origin)

    for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
        mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


    # Crop 256x256 region around root joint
    cropped_img, _ = crop_fixed_size(img, results.multi_hand_landmarks[0], size=256)
    cv2.imwrite(f"test_bbox/cropped_256_{hand_side.lower()}.jpg", cropped_img)
    
    # If you still want the original bbox visualization
    shift = 150
    roi = get_roi_points(results.multi_hand_landmarks[0], img.shape)
    bbox = compute_bbox(roi.copy(), hand_side, shift)

    # Draw bbox points and lines on the full image
    for x, y in bbox:
        print(x,y)
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)

    cv2.line(img, bbox[0], bbox[1], (255, 0, 255), 1)
    cv2.line(img, bbox[1], bbox[2], (255, 0, 255), 1)
    cv2.line(img, bbox[2], bbox[3], (255, 0, 255), 1)
    cv2.line(img, bbox[3], bbox[0], (255, 0, 255), 1)
    
    # Draw ROI points and lines on the full image
    for x,y in roi:
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

    cv2.line(img, roi[0], roi[1], (255, 0, 255), 1)
    cv2.line(img, roi[1], roi[2], (255, 0, 255), 1)
    cv2.line(img, roi[2], roi[3], (255, 0, 255), 1)
    cv2.line(img, roi[3], roi[0], (255, 0, 255), 1)

    # Save the full image with bbox visualization
    cv2.imwrite(f"test_bbox/test_{shift}.jpg", img)

    cropped_img = crop_to_bbox(img_to_process, bbox)
    cv2.imshow("img", cropped_img)
    cv2.waitKey(0)
    alpha = 1
    found = False
    while alpha < 4.1 and not found:
        beta = 0
        while beta < 51 and not found:
            enhanced_img = np.clip(cropped_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
            results = hands.process(enhanced_rgb)
            if results and results.multi_hand_landmarks:
                print(alpha, beta)
                found = True
            beta += 10
        alpha += 0.3

    
    print(len(results.multi_hand_landmarks))
    hand_side = results.multi_handedness[0].classification[0].label.lower()
    print(hand_side)
    roi = get_roi_points(results.multi_hand_landmarks[0], cropped_img.shape)
    bbox = compute_bbox(roi.copy(), hand_side, shift)
    

    blank_img, crop_origin = crop_fixed_size(cropped_img, results.multi_hand_landmarks[0], size=256)
    cv2.imwrite(f"test_bbox/cropped_256_{hand_side}_blank.jpg", blank_img)
    data = get_keypoints(data, cropped_img, results.multi_hand_landmarks[0], hand_side, crop_origin)
    
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
    cv2.imwrite(f"test_bbox/cropped_256_{hand_side.lower()}.jpg", cropped_img)


    cv2.imshow("img", cropped_img)
    cv2.waitKey(0)
    
    with open('test_bbox/test.json', 'w') as f:
        json.dump(data, f, indent=4)



