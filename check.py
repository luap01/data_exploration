import numpy as np

from utils.files import json_load


BASE_PATH_1 = "hand_detection/output/test/mediapipe/conf_0.50/camera05"
count = 0
for i in range(0, 20):
    img_idx = f"{i:06d}"

    try:
        kps_1 = json_load(f"{BASE_PATH_1}/json/{img_idx}_test.json")

        # Extract keypoints from both datasets
        lkps_1 = np.array(kps_1['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2]
        rkps_1 = np.array(kps_1['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]
        l_shift_1 = np.array(kps_1['people'][0]['hand_left_shift'])
        r_shift_1 = np.array(kps_1['people'][0]['hand_right_shift'])

        if l_shift_1[0] == 0 or l_shift_1[1] == 0 or r_shift_1[0] == 0 or r_shift_1[1] == 0:
            print(img_idx)
            count += 1
        elif len(lkps_1) < 21 or len(rkps_1) < 21:
            print(img_idx)
            count += 1

    except:
        print(img_idx)
        count += 1

print(count)
