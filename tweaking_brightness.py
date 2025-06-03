import cv2
from pathlib import Path
import numpy as np
import mediapipe as mp


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

if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:

    hand_side = results.multi_handedness[0].classification[0].label
    print(hand_side)
    shift = 150
    roi = get_roi_points(results.multi_hand_landmarks[0], img.shape)
    bbox = compute_bbox(roi.copy(), hand_side, shift)
    

    print(img.shape)
    # # Draw bbox points and lines on the full image
    # for x, y in bbox:
    #     print(x,y)
    #     cv2.circle(img, (int(x), int(y)), 4, (0, 255, 255), -1)

    # cv2.line(img, bbox[0], bbox[1], (255, 0, 255), 1)
    # cv2.line(img, bbox[1], bbox[2], (255, 0, 255), 1)
    # cv2.line(img, bbox[2], bbox[3], (255, 0, 255), 1)
    # cv2.line(img, bbox[3], bbox[0], (255, 0, 255), 1)
    
    # # Draw ROI points and lines on the full image
    # for x,y in roi:
    #     cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

    # cv2.line(img, roi[0], roi[1], (255, 0, 255), 1)
    # cv2.line(img, roi[1], roi[2], (255, 0, 255), 1)
    # cv2.line(img, roi[2], roi[3], (255, 0, 255), 1)
    # cv2.line(img, roi[3], roi[0], (255, 0, 255), 1)

    # Save the full image with bbox visualization
    cv2.imwrite(f"test_bbox/test_{shift}.jpg", img)
    
    # Crop and save just the bbox region
    cropped_img = crop_to_bbox(img, bbox)
    cv2.imwrite(f"test_bbox/cropped_{shift}.jpg", cropped_img)
