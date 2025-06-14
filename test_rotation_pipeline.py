#!/usr/bin/env python3
"""
Test script to verify rotation functionality in hand detection pipeline
"""

import cv2
import numpy as np
from hand_detection_pipeline import HandDetectionPipeline, PipelineConfig


def create_test_image_with_rotated_hand():
    """Create a simple test image with a hand-like shape"""
    # Create a blank image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Draw a simple hand-like shape (rectangle for palm, lines for fingers)
    # This will be rotated to test the rotation detection
    cv2.rectangle(img, (150, 200), (250, 300), (255, 255, 255), -1)  # Palm
    cv2.rectangle(img, (170, 150), (180, 200), (255, 255, 255), -1)  # Finger 1
    cv2.rectangle(img, (190, 140), (200, 200), (255, 255, 255), -1)  # Finger 2
    cv2.rectangle(img, (210, 145), (220, 200), (255, 255, 255), -1)  # Finger 3
    cv2.rectangle(img, (230, 150), (240, 200), (255, 255, 255), -1)  # Finger 4
    cv2.rectangle(img, (130, 220), (150, 250), (255, 255, 255), -1)  # Thumb
    
    return img


def test_rotation_detection():
    """Test the rotation detection functionality"""
    print("Testing rotation detection in hand pipeline...")
    
    # Create test configuration
    config = PipelineConfig(
        input_path="test_images",
        output_path="test_rotation_output",
        crop_size=256,
        bbox_shift=100,
        min_detection_confidence=0.05  # Lower threshold for test
    )
    
    # Create pipeline
    try:
        pipeline = HandDetectionPipeline(config)
        print("✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        return False
    
    # Create test image
    original_img = create_test_image_with_rotated_hand()
    
    # Test different rotations
    angles = [0, 90, 180, 270]
    for angle in angles:
        print(f"\nTesting {angle}° rotation...")
        
        # Rotate the test image
        if angle == 90:
            test_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            test_img = cv2.rotate(original_img, cv2.ROTATE_180)
        elif angle == 270:
            test_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            test_img = original_img.copy()
        
        # Save test image
        cv2.imwrite(f"test_rotation_output/test_image_{angle}.jpg", test_img)
        
        # Test rotation detection
        try:
            results, detected_angle = pipeline.detect_hands_with_enhancement(test_img)
            if results and results.multi_hand_landmarks:
                print(f"  ✓ Hands detected with {detected_angle}° rotation adjustment")
                print(f"  Number of hands detected: {len(results.multi_hand_landmarks)}")
            else:
                print(f"  ✗ No hands detected")
        except Exception as e:
            print(f"  ✗ Error during detection: {e}")
    
    print("\nRotation detection test completed!")
    return True


def main():
    """Main test function"""
    print("Starting rotation pipeline tests...\n")
    
    # Create output directory
    import os
    os.makedirs("test_rotation_output", exist_ok=True)
    
    # Run tests
    test_rotation_detection()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main() 