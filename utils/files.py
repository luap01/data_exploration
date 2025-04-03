import json
from pathlib import Path
import cv2


def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


def load_all_images(take_path: Path):
    images = {}
    i = 0
    while i < 5:
        img_path = take_path / f"cam{i+1}_000000.jpg"
        if Path(img_path).exists():
            images[f"camera0{i+1}"] = cv2.imread(img_path)
        i += 1
    return images


def load_all_xyz(take_path: Path):
    xyz = {}

    xyz_path = take_path / "left/000000.json"
    if Path(xyz_path).exists():
        xyz["left"] = json_load(xyz_path)

    xyz_path = take_path / "right/000000.json"
    if Path(xyz_path).exists():
        xyz["right"] = json_load(xyz_path)

    return xyz


def load_all_keypoints(take_path: Path):
    keypoints = {}
    for i in range(1, 5):
        keypoints_path = take_path / f"camera0{i}_000000_keypoints.json"
        if Path(keypoints_path).exists():
            keypoints[f"camera0{i}"] = json_load(keypoints_path)
    return keypoints
