import cv2
import json
import os
import numpy as np

def _json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)

img_idx = "000100"

# BASE_PATH = "test_bbox_larger_shift_into_opposite"
# BASE_PATH = "test_bbox"
BASE_PATH = "test_bbox_rotation_test"
for i in range(488, 492):
    img_idx = f"{i:06d}"

    if not os.path.exists(f"{BASE_PATH}/json/{img_idx}_test.json"):
        continue
    left = cv2.imread(f"{BASE_PATH}/blanks/{img_idx}_cropped_256_left_blank.jpg")
    right = cv2.imread(f"{BASE_PATH}/blanks/{img_idx}_cropped_256_right_blank.jpg")
    orig = cv2.imread(f"{BASE_PATH}/original/{img_idx}_test.jpg")

    # Check if images were loaded successfully
    if left is None or right is None or orig is None:
        print(f"Skipping {img_idx}: One or more images could not be loaded")
        continue

    kps = _json_load(f"{BASE_PATH}/json/{img_idx}_test.json")

    lkps = np.array(kps['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2]
    rkps = np.array(kps['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]
    l_shift = np.array(kps['people'][0]['hand_left_shift'])
    r_shift = np.array(kps['people'][0]['hand_right_shift'])

    def visualize_2d_points(points_2d, img, dot_colour, line_colour, offset):
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]

        x_shift, y_shift = offset[0], offset[1]
        for i, (x, y) in enumerate(points_2d):
            cv2.circle(img, (int(x) + x_shift, int(y) + y_shift), 4, dot_colour, -1)

        for idx1, idx2 in HAND_CONNECTIONS:
            x1, y1 = int(points_2d[idx1][0] + x_shift), int(points_2d[idx1][1] + y_shift)
            x2, y2 = int(points_2d[idx2][0] + x_shift), int(points_2d[idx2][1] + y_shift)
            cv2.line(img, (x1, y1), (x2, y2), line_colour, 1)

        return img


    l_img = visualize_2d_points(lkps, left, RED, GREEN, [0, 0])
    r_img = visualize_2d_points(rkps, right, BLUE, YELLOW, [0, 0])

    window_name = "img"
    cv2.imshow("img", l_img)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, -750)
    cv2.waitKey(0)
    cv2.imshow("img", r_img)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, -750)
    cv2.waitKey(0)


    orig = visualize_2d_points(lkps, orig, RED, GREEN, l_shift)
    orig = visualize_2d_points(rkps, orig, BLUE, YELLOW, r_shift)

    cv2.imshow("img", orig)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, -750)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.waitKey(0)