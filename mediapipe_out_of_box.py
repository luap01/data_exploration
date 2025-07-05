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

DEFAULT_CONF = 0.4

def parse_args():
    parser = argparse.ArgumentParser(description='OpenPose detection')
    parser.add_argument('--save_images', default=False, action='store_true', help='Save rendered images with keypoints')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF, help='Detection confidence threshold')
    parser.add_argument('--cam_idx', type=int, default=5, help='Camera index')
    return parser.parse_args()


def main():
    args = parse_args()
    # Initialize MediaPipe Hands
    conf = args.conf
    cam_idx = args.cam_idx
    save = bool(args.save_images)
    ORBBEC = True if cam_idx < 5 else False
    input_base_path = f"./data/input/orbbec/camera0{cam_idx}/"
    output_base_path = input_base_path.replace('input', 'output').replace('tony', 'tony/mediapipe').replace(f'camera0{cam_idx}/', f'camera0{cam_idx}/{conf:2f}/images')
    keypoints_base_path = output_base_path.replace('images', 'keypoints')

    # Create output directories
    for dir_path in [output_base_path + '/success', output_base_path + '/partial', output_base_path + '/failure']:
        os.makedirs(dir_path, exist_ok=True)

    for dir_path in [keypoints_base_path + '/success', keypoints_base_path + '/partial', keypoints_base_path + '/failure']:
        os.makedirs(dir_path, exist_ok=True)

    # os.makedirs(output_base_path + '/enhanced', exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.5
    )

    CALIB_DIR = './data/input/'
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
    # files = os.listdir(input_base_path)
    for idx, file in enumerate(files):
        # if idx < 320:
        #     continue
        # if idx > 519:
        #     break
        # if idx % 250 == 0:
        #     print(idx)
        
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
        
        results = hands.process(image)
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
                
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    image_for_drawing,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Process landmarks into keypoints
                keypoints = []
                for landmark in hand_landmarks.landmark:
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
            cv2.imwrite(output_image_path, image)
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