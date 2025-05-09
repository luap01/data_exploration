import cv2
from pathlib import Path
import numpy as np

from utils.camera import load_cam_infos

img_path = "data/input/tony/Marshall/camera05/images/color_002191_camera01.jpg"
img = cv2.imread(img_path)
cam_params = load_cam_infos(Path("./data/input/tony/"), orbbec=False)[f'camera05']

if img is None:
    print(f"Failed to load image: {img_path}")

img = cv2.undistort(
    img, 
    cam_params['intrinsics'], 
    np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0])
)

alpha = 1.0
beta = -20

# new_img = cv2.convertScale(img, alpha=1.0, beta=50)
new_img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


cv2.imshow('Original', img)
cv2.imshow('Adjusted', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(f"test/test_{alpha}_{beta}.jpg", new_img)