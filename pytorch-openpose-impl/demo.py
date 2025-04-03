import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import json

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

for cam_idx in range(1, 5):
    # test_image = f'../data/input/cam{cam_idx}_000000.jpg'
    test_image = f'./images/1/task72688.png'
    oriImg = cv2.imread(test_image)  # B,G,R order

    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    data = {}
    data['people'] = []
    data['people'].append({})
    for i, (x, y, w, is_left) in enumerate(hands_list):
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)
        
        hand_keypoints = []
        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            hand_keypoints.append(int(x))
            hand_keypoints.append(int(y))
            hand_keypoints.append(1)
        data['people'][0]['hand_left_keypoints_2d' if is_left else 'hand_right_keypoints_2d'] = hand_keypoints

    with open(f"./output/{test_image.split('/')[-1].replace('.jpg', '').replace('cam', 'camera0')}_keypoints.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    canvas = util.draw_handpose(canvas, all_hand_peaks)

    # plt.imshow(canvas[:, :, [2, 1, 0]])
    # plt.axis('off')
    # plt.show()
