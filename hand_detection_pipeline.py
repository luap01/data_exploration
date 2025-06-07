import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

from utils.camera import load_cam_infos


@dataclass
class PipelineConfig:
    """Configuration for the hand detection pipeline"""
    input_path: str = "data/input/tony/Marshall/camera05/images"
    output_path: str = "test_bbox"
    camera_name: str = "camera05"
    image_prefix: str = "color_"
    image_suffix: str = "_camera01.jpg"
    
    # Processing parameters
    crop_size: int = 256
    roi_padding: int = 20
    bbox_shift: int = 250
    
    # MediaPipe parameters
    max_num_hands: int = 2
    min_detection_confidence: float = 0.1
    min_tracking_confidence: float = 0.5
    
    # Brightness adjustment parameters
    max_alpha: float = 4.1
    alpha_step: float = 0.3
    max_beta: int = 51
    beta_step: int = 10
    
    # Threading parameters
    max_workers: int = 4
    batch_size: int = 100


class HandDetectionPipeline:
    """Clean pipeline for hand detection and cropping"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_logging()
        self.setup_output_dirs()
        self.load_camera_params()
        # Thread-safe counters
        self.processed_count = 0
        self.failed_count = 0
        self.lock = Lock()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_mediapipe_instance(self):
        """Create a new MediaPipe instance for thread safety"""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        return hands, mp.solutions.drawing_utils, mp.solutions.drawing_styles, mp_hands
    
    def setup_output_dirs(self):
        """Create output directories"""
        output_path = Path(self.config.output_path)
        self.dirs = {
            'blanks': output_path / 'blanks',
            'preds': output_path / 'preds',
            'original': output_path / 'original',
            'shifted_roi': output_path / 'shifted_roi',
            'failed': output_path / 'failed',
            'json': output_path / 'json'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_camera_params(self):
        """Load camera calibration parameters"""
        try:
            cam_infos = load_cam_infos(Path("./data/input/tony/"), orbbec=False)
            self.cam_params = cam_infos[self.config.camera_name]
        except Exception as e:
            self.logger.error(f"Failed to load camera parameters: {e}")
            raise
    
    def load_and_undistort_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load and undistort an image"""
        img = cv2.imread(img_path)
        if img is None:
            self.logger.warning(f"Failed to load image: {img_path}")
            return None
        
        # Apply undistortion
        distortion_coeffs = np.array([
            self.cam_params['radial_params'][0],
            self.cam_params['radial_params'][1],
            *self.cam_params['tangential_params'][:2],
            self.cam_params['radial_params'][2],
            0, 0, 0
        ])
        
        undistorted = cv2.undistort(img, self.cam_params['intrinsics'], distortion_coeffs)
        return undistorted
    
    def try_rotation_angles(self, image: np.ndarray, hands) -> Tuple[Optional[any], int, np.ndarray]:
        """Try different image rotations for hand detection"""
        angles = [0, 90, 180, 270]
        for angle in angles:
            if angle == 0:
                img_rotated = image
            else:
                # For 90 degree rotations, use cv2's built-in functions
                if angle == 90:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    img_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            results_rotated = hands.process(img_rotated)
            if results_rotated and results_rotated.multi_hand_landmarks:
                return results_rotated, angle, img_rotated
        return None, 0, image
    
    def transform_landmarks_back(self, landmarks, angle: int, rotated_image_shape: Tuple[int, int], 
                                original_image_shape: Tuple[int, int]):
        """Transform landmarks from rotated image back to original coordinate system"""
        h_rot, w_rot = rotated_image_shape[:2]
        h_orig, w_orig = original_image_shape[:2]
        
        for landmark in landmarks.landmark:
            # Convert from normalized to pixel coordinates in rotated image
            x = landmark.x * w_rot
            y = landmark.y * h_rot
            
            # Transform back to original orientation
            if angle == 90:
                x_new, y_new = y, w_rot - x
            elif angle == 180:
                x_new, y_new = w_rot - x, h_rot - y
            elif angle == 270:
                x_new, y_new = h_rot - y, x
            else:  # angle == 0
                x_new, y_new = x, y
            
            # Convert back to normalized coordinates in original image
            landmark.x = x_new / w_orig
            landmark.y = y_new / h_orig
        
        return landmarks
    
    def detect_hands_with_enhancement(self, image: np.ndarray, hands) -> Tuple[Optional[any], int]:
        """Detect hands with brightness/contrast enhancement and rotation if needed"""
        # Try original image first
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        if results and results.multi_hand_landmarks:
            return results, 0
        
        # Try with brightness/contrast adjustments
        for alpha in np.arange(0, self.config.max_alpha, self.config.alpha_step):
            for beta in range(0, self.config.max_beta, self.config.beta_step):
                enhanced = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                results = hands.process(enhanced_rgb)
                
                if results and results.multi_hand_landmarks:
                    return results, 0
                
                # If enhancement alone doesn't work, try with rotation
                results_rotated, angle, rotated_img = self.try_rotation_angles(enhanced_rgb, hands)
                if results_rotated and results_rotated.multi_hand_landmarks:
                    # Transform landmarks back to original coordinate system
                    for hand_landmarks in results_rotated.multi_hand_landmarks:
                        self.transform_landmarks_back(
                            hand_landmarks, angle, rotated_img.shape, image.shape
                        )
                    return results_rotated, angle
        
        # If enhancement fails, try rotation on original image
        results_rotated, angle, rotated_img = self.try_rotation_angles(rgb_image, hands)
        if results_rotated and results_rotated.multi_hand_landmarks:
            # Transform landmarks back to original coordinate system
            for hand_landmarks in results_rotated.multi_hand_landmarks:
                self.transform_landmarks_back(
                    hand_landmarks, angle, rotated_img.shape, image.shape
                )
            return results_rotated, angle
        
        return None, 0
    
    def get_roi_points(self, hand_landmarks, image_shape: Tuple[int, int]) -> np.ndarray:
        """Get ROI points around hand landmarks"""
        h, w = image_shape[:2]
        
        points = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
        
        x_coords, y_coords = zip(*points)
        padding = self.config.roi_padding
        
        roi_points = np.array([
            [min(x_coords) - padding, min(y_coords) - padding],
            [min(x_coords) - padding, max(y_coords) + padding],
            [max(x_coords) + padding, max(y_coords) + padding],
            [max(x_coords) + padding, min(y_coords) - padding]
        ], dtype=np.int32)
        
        return roi_points
    
    def compute_shifted_bbox(self, roi: np.ndarray, hand_side: str) -> np.ndarray:
        """Compute shifted bounding box for hand detection"""
        bbox = roi.copy()
        shift = self.config.bbox_shift
        half_shift = shift // 2
        
        # Expand ROI in all directions
        bbox[0, 0] -= half_shift        # top-left: left
        bbox[0, 1] -= shift             # top-left: up
        bbox[1, 0] -= half_shift        # bottom-left: left
        bbox[1, 1] += shift             # bottom-left: down
        bbox[2, 0] += half_shift        # bottom-right: right 
        bbox[2, 1] += shift             # bottom-right: down
        bbox[3, 0] += half_shift        # top-right: right
        bbox[3, 1] -= shift             # top-right: up
        
        # Shift based on hand side
        if hand_side == "left":
            bbox[:, 0] += shift  # Shift right for left hand
            bbox[2, 0] += shift
            bbox[3, 0] += shift
        else:
            bbox[:, 0] -= shift  # Shift left for right hand
            bbox[0, 0] -= shift
            bbox[1, 0] -= shift
        return bbox
    
    def crop_to_bbox(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop image to bounding box"""
        x_coords, y_coords = bbox[:, 0], bbox[:, 1]
        
        x_min = max(0, int(min(x_coords)))
        y_min = max(0, int(min(y_coords)))
        x_max = min(image.shape[1], int(max(x_coords)))
        y_max = min(image.shape[0], int(max(y_coords)))
        
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped, (x_min, y_min)
    
    def crop_fixed_size(self, image: np.ndarray, hand_landmarks, 
                       base_offset: Tuple[int, int] = (0, 0),
                       current_image_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop fixed size region around wrist joint"""
        h, w = image.shape[:2]
        size = self.config.crop_size
        
        # Determine coordinate system
        if current_image_size is not None:
            curr_h, curr_w = current_image_size
        else:
            curr_h, curr_w = h, w
        
        # Get wrist coordinates (landmark[0])
        root_x_detected = int(hand_landmarks.landmark[0].x * curr_w)
        root_y_detected = int(hand_landmarks.landmark[0].y * curr_h)
        
        # Transform to original image coordinates
        root_x = root_x_detected + base_offset[0]
        root_y = root_y_detected + base_offset[1]
        
        # Calculate crop boundaries
        half_size = size // 2
        x_min = max(0, root_x - half_size)
        y_min = max(0, root_y - half_size)
        x_max = min(w, root_x + half_size)
        y_max = min(h, root_y + half_size)
        
        # Adjust if out of bounds
        if x_min == 0:
            x_max = min(w, size)
        if y_min == 0:
            y_max = min(h, size)
        if x_max == w:
            x_min = max(0, w - size)
        if y_max == h:
            y_min = max(0, h - size)
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        # Pad if necessary
        if cropped.shape[0] != size or cropped.shape[1] != size:
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            pad_y = (size - cropped.shape[0]) // 2
            pad_x = (size - cropped.shape[1]) // 2
            padded[pad_y:pad_y+cropped.shape[0], pad_x:pad_x+cropped.shape[1]] = cropped
            return padded, (x_min, y_min)
        
        return cropped, (x_min, y_min)
    
    def get_keypoints_data(self, image: np.ndarray, hand_landmarks, hand_type: str,
                          coord_origins: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Extract keypoint data for JSON output"""
        keypoints = []
        
        # Calculate total offset
        x_min, y_min = coord_origins[-1]
        x_orig, y_orig = 0, 0
        if len(coord_origins) > 1:
            x_orig, y_orig = coord_origins[-2][0], coord_origins[-2][1] 

        # Extract keypoints
        for landmark in hand_landmarks.landmark:
            x = float(landmark.x * image.shape[1]) - x_min + x_orig
            y = float(landmark.y * image.shape[0]) - y_min + y_orig
            keypoints.extend([x, y, 1.0])
        
        return {
            f"hand_{hand_type}_keypoints_2d": keypoints,
            f"hand_{hand_type}_shift": [x_min, y_min]
        }
    
    def draw_landmarks_and_bbox(self, image: np.ndarray, hand_landmarks, bbox: np.ndarray, roi: np.ndarray, 
                               mp_drawing, mp_drawing_styles, mp_hands) -> np.ndarray:
        """Draw hand landmarks and bounding boxes on image"""
        result_img = image.copy()
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            result_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Draw bounding box
        for x, y in bbox:
            cv2.circle(result_img, (int(x), int(y)), 4, (0, 255, 255), -1)
        
        # Draw bbox lines
        for i in range(len(bbox)):
            cv2.line(result_img, tuple(bbox[i]), tuple(bbox[(i+1) % len(bbox)]), (255, 0, 255), 1)
        
        # Draw ROI
        for x, y in roi:
            cv2.circle(result_img, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        # Draw ROI lines
        for i in range(len(roi)):
            cv2.line(result_img, tuple(roi[i]), tuple(roi[(i+1) % len(roi)]), (255, 0, 255), 1)
        
        return result_img
    
    def process_single_hand(self, image: np.ndarray, results, img_idx: str, 
                           hands, mp_drawing, mp_drawing_styles, mp_hands) -> Optional[Dict[str, Any]]:
        """Process image with single hand detection"""
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_side = results.multi_handedness[0].classification[0].label.lower()
        
        self.logger.info(f"Processing {img_idx}: {hand_side} hand detected")
        
        # Initialize data structure
        data = {"people": [{"hand_left_shift": [], "hand_left_keypoints_2d": [], 
                           "hand_right_shift": [], "hand_right_keypoints_2d": []}]}
        
        # Initial crop from original image
        blank_img, crop_origin = self.crop_fixed_size(image.copy(), hand_landmarks)
        cv2.imwrite(self.dirs['blanks'] / f"{img_idx}_cropped_256_{hand_side}_blank.jpg", blank_img)
        
        keypoint_data = self.get_keypoints_data(image, hand_landmarks, hand_side, [crop_origin])
        data["people"][0].update(keypoint_data)
        
        # Create visualization with landmarks
        vis_img = image.copy()
        mp_drawing.draw_landmarks(
            vis_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        # Crop visualization
        cropped_vis, _ = self.crop_fixed_size(vis_img, hand_landmarks)
        cv2.imwrite(self.dirs['preds'] / f"{img_idx}_cropped_256_{hand_side}.jpg", cropped_vis)
        
        # Save original image
        cv2.imwrite(self.dirs['original'] / f"{img_idx}_test.jpg", image)
        
        # Try shifted bbox approach with retry mechanism
        roi = self.get_roi_points(hand_landmarks, image.shape)
        
        # Retry mechanism for hand misclassification
        first_try = True
        retry = False
        fail = False
        prev_detected_handside = hand_side
        current_hand_side = hand_side
        original_bbox = None  # Store original bbox for failure visualization
        
        while first_try or retry:
            try:
                bbox = self.compute_shifted_bbox(roi.copy(), current_hand_side)
                
                # Store original bbox for failure visualization
                if first_try:
                    original_bbox = bbox.copy()
                
                # Crop to shifted bbox
                cropped_img, bbox_origin = self.crop_to_bbox(image.copy(), bbox)
                
                # Detect hands in cropped region with enhancement
                cropped_results, angle = self.detect_hands_with_enhancement(cropped_img, hands)
                
                if angle != 0:
                    self.logger.info(f"Hands detected in {img_idx} using {angle}° rotation")
        
                found = False
                if cropped_results and cropped_results.multi_hand_landmarks:
                    found = True
                    if retry:
                        # If we're in retry mode and found something, copy the previous data
                        retry = False
                        data["people"][0][f"hand_{current_hand_side}_keypoints_2d"] = data["people"][0][f"hand_{prev_detected_handside}_keypoints_2d"]
                        data["people"][0][f"hand_{current_hand_side}_shift"] = data["people"][0][f"hand_{prev_detected_handside}_shift"]
                        cv2.imwrite(self.dirs['blanks'] / f"{img_idx}_cropped_256_{current_hand_side}_blank.jpg", blank_img)
                
                first_try = False
                
                if found:
                    # Process the detected hand
                    detected_landmarks = cropped_results.multi_hand_landmarks[0]
                    detected_hand_side = "right" if current_hand_side == "left" else "right"
                    
                    # Create final crop from original image using detected landmarks
                    final_crop, final_origin = self.crop_fixed_size(
                        image, detected_landmarks, 
                        base_offset=bbox_origin, 
                        current_image_size=cropped_img.shape[:2]
                    )
                    
                    # Update keypoints data
                    final_keypoint_data = self.get_keypoints_data(
                        cropped_img, detected_landmarks, detected_hand_side, 
                        [bbox_origin, final_origin]
                    )
                    data["people"][0].update(final_keypoint_data)
                    
                    # Save final crops
                    cv2.imwrite(self.dirs['blanks'] / f"{img_idx}_cropped_256_{detected_hand_side}_blank.jpg", final_crop)
                    
                    # Create visualization of cropped region
                    vis_cropped = self.draw_landmarks_and_bbox(
                        cropped_img, detected_landmarks, 
                        self.compute_shifted_bbox(
                            self.get_roi_points(detected_landmarks, cropped_img.shape), 
                            detected_hand_side
                        ), 
                        self.get_roi_points(detected_landmarks, cropped_img.shape),
                        mp_drawing, mp_drawing_styles, mp_hands
                    )
                    
                    final_vis_crop, _ = self.crop_fixed_size(vis_cropped, detected_landmarks)
                    cv2.imwrite(self.dirs['preds'] / f"{img_idx}_cropped_256_{detected_hand_side}.jpg", final_vis_crop)
                    
                    # Save shifted ROI visualization
                    full_vis = self.draw_landmarks_and_bbox(image, hand_landmarks, bbox, roi,
                                                           mp_drawing, mp_drawing_styles, mp_hands)
                    cv2.imwrite(self.dirs['shifted_roi'] / f"{img_idx}_test_{self.config.bbox_shift}.jpg", full_vis)
                    
                    break  # Success, exit the retry loop
                    
                else:
                    # No hand detected, try retry logic
                    if not retry:
                        retry = True
                        prev_detected_handside = current_hand_side
                        current_hand_side = "left" if prev_detected_handside == "right" else "right"
                        
                        self.logger.info(f"Retrying {img_idx} with {current_hand_side} hand classification")
                        
                        # Create retry visualization
                        retry_img = image.copy()
                        bbox_retry = self.compute_shifted_bbox(roi.copy(), current_hand_side)
                        
                        # Draw retry bbox visualization
                        retry_vis = self.draw_landmarks_and_bbox(retry_img, hand_landmarks, bbox_retry, roi,
                                                               mp_drawing, mp_drawing_styles, mp_hands)
                        cv2.imwrite(self.dirs['shifted_roi'] / f"{img_idx}_test_{self.config.bbox_shift}_retry.jpg", retry_vis)
                        
                    else:
                        # Both attempts failed
                        retry = False
                        fail = True
                        self.logger.warning(f"Failed to detect second hand for {img_idx}")
                        
                        # Save failure images
                        cv2.imwrite(self.dirs['failed'] / f"{img_idx}_test_{self.config.bbox_shift}_retry.jpg", retry_vis)
                        original_vis = self.draw_landmarks_and_bbox(image, hand_landmarks, original_bbox, roi,
                                                                   mp_drawing, mp_drawing_styles, mp_hands)
                        cv2.imwrite(self.dirs['failed'] / f"{img_idx}_test_{self.config.bbox_shift}.jpg", original_vis)
                        break
                        
            except Exception as e:
                self.logger.error(f"Error in retry loop for {img_idx}: {e}")
                if not retry:
                    retry = True
                    prev_detected_handside = current_hand_side
                    current_hand_side = "left" if prev_detected_handside == "right" else "right"
                else:
                    fail = True
                    break
        
        if fail:
            return None
        
        return data
    
    def process_double_hands(self, image: np.ndarray, results, img_idx: str,
                            hands, mp_drawing, mp_drawing_styles, mp_hands) -> Dict[str, Any]:
        """Process image with two hands detected"""
        self.logger.info(f"Processing {img_idx}: 2 hands detected")
        
        # Initialize data structure
        data = {"people": [{"hand_left_shift": [], "hand_left_keypoints_2d": [], 
                           "hand_right_shift": [], "hand_right_keypoints_2d": []}]}
        
        # Save original image
        cv2.imwrite(self.dirs['original'] / f"{img_idx}_test.jpg", image)

        # Process both hands
        for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_side = handedness.classification[0].label.lower()
            
            # Crop around each hand
            blank_img, crop_origin = self.crop_fixed_size(image.copy(), hand_landmarks)
            cv2.imwrite(self.dirs['blanks'] / f"{img_idx}_cropped_256_{hand_side}_blank.jpg", blank_img)
            
            # Get keypoints
            keypoint_data = self.get_keypoints_data(image, hand_landmarks, hand_side, [crop_origin])
            data["people"][0].update(keypoint_data)
        
        return data
    
    def process_image_thread_safe(self, img_idx: str) -> bool:
        """Thread-safe version of process_image"""
        # Create MediaPipe instances for this thread
        hands, mp_drawing, mp_drawing_styles, mp_hands = self.create_mediapipe_instance()
        
        # Construct image path
        img_path = f"{self.config.input_path}/{self.config.image_prefix}{img_idx}{self.config.image_suffix}"
        
        # Load and undistort image
        image = self.load_and_undistort_image(img_path)
        if image is None:
            return False
        
        # Detect hands
        results, angle = self.detect_hands_with_enhancement(image, hands)
        if not results or not results.multi_hand_landmarks:
            self.logger.info(f"No hands detected in {img_idx}")
            return False
        
        if angle != 0:
            self.logger.info(f"Hands detected in {img_idx} using {angle}° rotation")
        
        # Process based on number of hands
        if len(results.multi_hand_landmarks) == 2:
            data = self.process_double_hands(image, results, img_idx, hands, mp_drawing, mp_drawing_styles, mp_hands)
        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            data = self.process_single_hand(image, results, img_idx, hands, mp_drawing, mp_drawing_styles, mp_hands)
        else:
            return False
        
        if data is None:
            return False
        
        # Save JSON data
        json_path = self.dirs['json'] / f"{img_idx}_test.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return True
    
    def process_batch(self, indices: List[int]) -> Tuple[int, int]:
        """Process a batch of images and return (processed_count, failed_count)"""
        batch_processed = 0
        batch_failed = 0
        
        for idx in indices:
            img_idx = f"{idx:06d}"
            
            try:
                if self.process_image_thread_safe(img_idx):
                    batch_processed += 1
                else:
                    batch_failed += 1
            except Exception as e:
                self.logger.error(f"Error processing {img_idx}: {e}")
                batch_failed += 1
        
        # Thread-safe update of global counters
        with self.lock:
            self.processed_count += batch_processed
            self.failed_count += batch_failed
        
        return batch_processed, batch_failed
    
    def create_batches(self, start_idx: int, end_idx: int) -> List[List[int]]:
        """Create batches of image indices"""
        all_indices = list(range(start_idx, end_idx))
        batches = []
        
        for i in range(0, len(all_indices), self.config.batch_size):
            batch = all_indices[i:i + self.config.batch_size]
            batches.append(batch)
        
        return batches
    
    def run_multithreaded(self, start_idx: int, end_idx: int):
        """Run the pipeline with multithreading"""
        self.logger.info(f"Starting multithreaded pipeline for images {start_idx:06d} to {end_idx:06d}")
        self.logger.info(f"Configuration: {self.config.max_workers} workers, batch size {self.config.batch_size}")
        
        # Create batches
        batches = self.create_batches(start_idx, end_idx)
        self.logger.info(f"Created {len(batches)} batches")
        
        # Reset counters
        self.processed_count = 0
        self.failed_count = 0
        
        # Process batches with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(self.process_batch, batch): i for i, batch in enumerate(batches)}
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_processed, batch_failed = future.result()
                    self.logger.info(f"Batch {batch_idx + 1}/{len(batches)} completed: "
                                   f"processed={batch_processed}, failed={batch_failed}")
                except Exception as e:
                    self.logger.error(f"Batch {batch_idx + 1} failed with error: {e}")
        
        self.logger.info(f"Multithreaded pipeline completed. "
                        f"Total processed: {self.processed_count}, Total failed: {self.failed_count}")

    def process_image(self, img_idx: str) -> bool:
        """Process a single image (legacy method for backward compatibility)"""
        # Create MediaPipe instances
        hands, mp_drawing, mp_drawing_styles, mp_hands = self.create_mediapipe_instance()
        
        # Construct image path
        img_path = f"{self.config.input_path}/{self.config.image_prefix}{img_idx}{self.config.image_suffix}"
        
        # Load and undistort image
        image = self.load_and_undistort_image(img_path)
        if image is None:
            return False
        
        # Detect hands
        results, angle = self.detect_hands_with_enhancement(image, hands)
        if not results or not results.multi_hand_landmarks:
            self.logger.info(f"No hands detected in {img_idx}")
            return False
        
        if angle != 0:
            self.logger.info(f"Hands detected in {img_idx} using {angle}° rotation")
        
        # Process based on number of hands
        if len(results.multi_hand_landmarks) == 2:
            data = self.process_double_hands(image, results, img_idx, hands, mp_drawing, mp_drawing_styles, mp_hands)
        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            data = self.process_single_hand(image, results, img_idx, hands, mp_drawing, mp_drawing_styles, mp_hands)
        else:
            return False
        
        if data is None:
            return False
        
        # Save JSON data
        json_path = self.dirs['json'] / f"{img_idx}_test.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        return True

    def run(self, start_idx: int, end_idx: int):
        """Run the pipeline on a range of images (single-threaded)"""
        self.logger.info(f"Starting single-threaded pipeline for images {start_idx:06d} to {end_idx:06d}")
        
        processed_count = 0
        failed_count = 0
        
        for idx in range(start_idx, end_idx):
            img_idx = f"{idx:06d}"
            
            try:
                if self.process_image(img_idx):
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                self.logger.error(f"Error processing {img_idx}: {e}")
                failed_count += 1
        
        self.logger.info(f"Single-threaded pipeline completed. Processed: {processed_count}, Failed: {failed_count}")


def main():
    """Main execution function"""
    # Configure pipeline
    config = PipelineConfig(
        input_path="data/input/tony/Marshall/camera05/images",
        output_path="test_bbox_rotation_test_multithread",
        crop_size=256,
        bbox_shift=250,
        max_workers=4,  # Number of threads
        batch_size=75   # Images per batch
    )
    
    # Create pipeline
    pipeline = HandDetectionPipeline(config)

    start = time.time()
    
    # Use multithreaded version for better performance
    pipeline.run_multithreaded(start_idx=100, end_idx=500)
    
    # Alternative: use single-threaded version
    # pipeline.run(start_idx=100, end_idx=1000)
    
    end = time.time()
    
    elapsed_time = end - start
    pipeline.logger.info(f"Pipeline execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main() 