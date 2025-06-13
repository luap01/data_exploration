import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sys
from pathlib import Path

# Add openpose_impl to path
openpose_path = Path(__file__).parent.parent / "openpose_impl"
if str(openpose_path) not in sys.path:
    sys.path.append(str(openpose_path))

from hand import Hand
from model import Model
from .hand_detector import HandDetector, HandDetection


class OpenPoseHandDetector(HandDetector):
    """OpenPose implementation of hand detector"""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5):
        self.model_path = model_path
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.model = None
        self.hand_detector = None
    
    def initialize(self):
        """Initialize OpenPose hand detector"""
        if self.model is None:
            self.model = Model()
            if self.model_path:
                self.model.load_model(self.model_path)
            self.hand_detector = Hand(self.model)
    
    def detect_hands(self, image: np.ndarray) -> Tuple[List[HandDetection], Optional[np.ndarray]]:
        """Detect hands using OpenPose"""
        if self.hand_detector is None:
            self.initialize()
        
        # Process image
        hand_keypoints, annotated_image = self.hand_detector.detect(image)
        
        detections = []
        if hand_keypoints is not None and len(hand_keypoints) > 0:
            # Process each detected hand
            for hand_idx in range(min(len(hand_keypoints), self.max_num_hands)):
                keypoints = hand_keypoints[hand_idx]
                
                # Convert to normalized coordinates
                h, w = image.shape[:2]
                normalized_keypoints = keypoints.copy()
                normalized_keypoints[:, 0] /= w
                normalized_keypoints[:, 1] /= h
                
                # Determine hand type (OpenPose might need additional logic here)
                # For now, we'll assume first hand is right and second is left
                hand_type = "right" if hand_idx == 0 else "left"
                
                # Calculate confidence as mean of keypoint confidences
                confidence = float(np.mean(keypoints[:, 2]))
                
                if confidence >= self.min_detection_confidence:
                    detection = HandDetection(
                        landmarks=normalized_keypoints,
                        hand_type=hand_type,
                        confidence=confidence
                    )
                    detections.append(detection)
        
        return detections, annotated_image
    
    def get_keypoints_data(self, detection: HandDetection, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Convert OpenPose detection to keypoints data"""
        h, w = image_shape
        
        # Convert normalized coordinates to pixel coordinates
        keypoints = []
        for x, y, c in detection.landmarks:
            px = float(x * w)
            py = float(y * h)
            keypoints.extend([px, py, c])
        
        return {
            f"hand_{detection.hand_type}_keypoints_2d": keypoints,
            f"hand_{detection.hand_type}_shift": [0, 0]  # No shift for initial detection
        }
    
    def release(self):
        """Release OpenPose resources"""
        if self.model is not None:
            self.model = None
            self.hand_detector = None 