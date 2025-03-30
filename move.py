import numpy as np
import json
import os

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def save_file(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def build_arr(data):
    coords = []
    conf = []
    for idx in range(0, len(data), 3):
        coords.append(data[idx])
        coords.append(data[idx+1])
        conf.append(data[idx+2])
    return [coords, conf]

if __name__ == "__main__":
    files = os.listdir("./json")
    for file in files:
        try:
            data = json_load(f"./json/{file}")
            res_right = build_arr(data["people"][0]["hand_right_keypoints_2d"])
            res_left = build_arr(data["people"][0]["hand_left_keypoints_2d"])
            tar_pth_right = f"../HaMuCo/data/OR/rgb_2D_keypoints/right/{file.split('_')[0]}/{file.split('_')[1]}.json"
            tar_pth_left = f"../HaMuCo/data/OR/rgb_2D_keypoints/left/{file.split('_')[0]}/{file.split('_')[1]}.json"
            save_file(res_right, tar_pth_right)
            save_file(res_left, tar_pth_left)
        except IndexError:
            pass