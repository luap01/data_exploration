import cv2
import json
import numpy as np

def _json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

left = cv2.imread("test_bbox/cropped_256_left_blank.jpg")
right = cv2.imread("test_bbox/cropped_256_right_blank.jpg")

kps = _json_load("test_bbox/test.json")

lkps = np.array(kps['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2]
rkps = np.array(kps['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]

def visualize_2d_points(points_2d, img, dot_colour, line_colour):
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    for i, (x, y) in enumerate(points_2d):
        cv2.circle(img, (int(x), int(y)), 4, dot_colour, -1)

    for idx1, idx2 in HAND_CONNECTIONS:
        x1, y1 = int(points_2d[idx1][0]), int(points_2d[idx1][1])
        x2, y2 = int(points_2d[idx2][0]), int(points_2d[idx2][1])
        cv2.line(img, (x1, y1), (x2, y2), line_colour, 1)

    return img


l_img = visualize_2d_points(lkps, left, (0, 0, 255), (255, 0, 0))
r_img = visualize_2d_points(rkps, right, (0, 0, 255), (255, 0, 0))

cv2.imshow("img", l_img)
cv2.waitKey(0)
cv2.imshow("img", r_img)
cv2.waitKey(0)