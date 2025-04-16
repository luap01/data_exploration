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
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths relative to the script location
    base_path = os.path.join(script_dir, "..", "data", "output", "openpose", "json", "camera06")
    tar_base_path = os.path.join(script_dir, "..", "..", "HaMuCo", "data", "OR", "rgb_2D_keypoints")

    files = os.listdir(base_path)
    count = 0
    for file in files:
        try:
            data = json_load(os.path.join(base_path, file))

            if len(data["people"]) > 0:
                res_right = build_arr(data["people"][0]["hand_right_keypoints_2d"])
                res_left = build_arr(data["people"][0]["hand_left_keypoints_2d"])
            else:
                print(file)
                count += 1
                arr = [0] * 63
                res_right = build_arr(arr)
                res_left = build_arr(arr)
            
            # Create target paths relative to script location
            camera_name = os.path.basename(base_path)
            file_prefix = file.split('_')[0]
            
            tar_pth_right = os.path.join(tar_base_path, "right", camera_name, f"{file_prefix}.json")
            tar_pth_left = os.path.join(tar_base_path, "left", camera_name, f"{file_prefix}.json")
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(tar_pth_right), exist_ok=True)
            os.makedirs(os.path.dirname(tar_pth_left), exist_ok=True)
            
            save_file(res_right, tar_pth_right)
            save_file(res_left, tar_pth_left)
        except IndexError:
            pass
    
    print(count)