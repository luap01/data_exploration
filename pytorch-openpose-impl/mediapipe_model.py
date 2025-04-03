import cv2
import mediapipe as mp
import numpy as np
import os 
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.01)
mp_drawing = mp.solutions.drawing_utils


files = os.listdir("./images/1/")
pos = 0
neg = 0
start_time = time.time()
for file in files:
    # Load the image
    image = cv2.imread(f"./images/1/{file}")  # Replace with your image path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(image_rgb)
    # Draw keypoints if hand is detected
    if results.multi_hand_landmarks:
        pos += 1
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Print the keypoints
            # for idx, landmark in enumerate(hand_landmarks.landmark):
            #     print(f'Keypoint {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})')
        cv2.imwrite(f'./output/seppi/pos0.01/{str(file).split("/")[-1].split(".")[0]}.jpg', image)
    else:
        neg += 1
        cv2.imwrite(f'./output/seppi/neg0.01/{str(file).split("/")[-1].split(".")[0]}.jpg', image)
    # Save or display the result

    # cv2.imshow('Hand Keypoints', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Release resources
hands.close()
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds for {len(files)} images")
print(f"Pos: {pos}, Neg: {neg}")
print(f"Pos: {pos/len(files)}, Neg: {neg/len(files)}")