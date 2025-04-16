import json
from pathlib import Path
import cv2
import os

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


def load_all_images(take_path: Path):
    images = {}
    dirs = os.listdir(take_path)
    dirs.sort()
    for dir in dirs:
        files = os.listdir(take_path / dir)
        files.sort()
        i = 0
        for file in files:
            img_path = take_path / dir / file
            key = dir + "/" + file
            images[key] = cv2.imread(str(img_path))
            i += 1
            if i > 10:
                break
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
    dirs = os.listdir(take_path)
    dirs.sort()
    for dir in dirs:
        files = os.listdir(take_path / dir)
        files.sort()
        i = 0
        for file in files:
            kpt_path = take_path / dir / file
            key = dir + "/" + file
            keypoints[key] = json_load(kpt_path)
            i += 1
            if i > 10:
                break
    return keypoints