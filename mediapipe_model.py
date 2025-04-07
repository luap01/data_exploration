import cv2
import mediapipe as mp
import numpy as np
import os 
import time
import math
import json

# Initialize MediaPipe Hands
conf = 0.33
input_base_path = "./data/input/openpose/images/camera05/"
output_base_path = input_base_path.replace('input', 'output')

os.makedirs(output_base_path, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=conf)
mp_drawing = mp.solutions.drawing_utils


files = os.listdir(input_base_path)
pos = 0
neg = 0
start_time = time.time()
vectors = {}
for file in files:
    # Load the image
    image = cv2.imread(f"{input_base_path}/{file}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    # Process the image
    results = hands.process(image_rgb)
    # Draw keypoints if hand is detected
    if results.multi_hand_landmarks:
        pos += 1
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw all hand landmarks and connections
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # # Get image dimensions
            # h, w, _ = image.shape
            
            # # Get palm center (landmark 9)
            # middle_finger_start = hand_landmarks.landmark[10]
            # palm_x = int(middle_finger_start.x * w)
            # palm_y = int(middle_finger_start.y * h)
            
            # # Get middle finger tip (landmark 12)
            # middle_finger_tip = hand_landmarks.landmark[12]
            # middle_x = int(middle_finger_tip.x * w)
            # middle_y = int(middle_finger_tip.y * h)

            # # Calculate unit vector for middle finger direction
            # dx = middle_x - palm_x
            # dy = middle_y - palm_y
            # magnitude = math.sqrt(dx**2 + dy**2)
            # norm_vector = (dx/magnitude, dy/magnitude)
            
            # vectors[file] = norm_vector
            # # Draw circles at these points
            # cv2.circle(image, (palm_x, palm_y), 10, (0, 255, 0), -1)  # Green circle for palm center
            # cv2.circle(image, (middle_x, middle_y), 10, (0, 0, 255), -1)  # Red circle for middle finger tip
            
            # # Optional: Draw line between palm center and middle finger tip
            # cv2.arrowedLine(image, (palm_x, palm_y), (middle_x, middle_y), (255, 0, 0), 2)
        
            
        cv2.imwrite(f'{output_base_path}/{str(file).split("/")[-1].split(".")[0]}.jpg', image)
    else:
        neg += 1
        # cv2.imwrite(f'{output_base_neg_path}/{str(file).split("/")[-1].split(".")[0]}.jpg', image)
    # Save or display the result

    cv2.imshow('Hand Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Release resources
hands.close()
end_time = time.time()
json.dump(vectors, open(f"./output/other/vectors{str(conf)}.json", "w"))
print(f"Time taken: {end_time - start_time} seconds for {len(files)} images")
print(f"Pos: {pos}, Neg: {neg}")
print(f"Pos: {pos/len(files)}, Neg: {neg/len(files)}")