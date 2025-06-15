def detect_second_hand_or_retry(self, img_idx: int, image: np.ndarray, blank_img: np.ndarray, detection: HandDetection, roi: np.ndarray, data: Dict[str, List[Dict]]) -> None:
        """
        Attempt to detect the second hand with retry mechanism for misclassification.
        
        Strategy:
        1. Try to detect second hand in shifted ROI based on first hand's type
        2. If same hand detected (small shift diff), try with larger shift
        3. If no hand detected or still same hand, retry with opposite direction
        4. If hand detected in retry, handle potential misclassification
        """
        # Initial setup
        first_hand_type = detection.hand_type
        current_hand_side = first_hand_type
        original_bbox = None
        max_retries = 3
        retry_count = 0
        shift_multiplier = 1.0

        while retry_count < max_retries:
            try:
                # Compute and apply shifted bbox
                bbox = self.compute_shifted_bbox(roi.copy(), current_hand_side)
                if shift_multiplier > 1.0:
                    # Apply additional shift to try to get away from first hand
                    if current_hand_side == "left":
                        bbox[:, 0] += int(self.config.bbox_shift * (shift_multiplier - 1))  # Extra shift right
                    else:
                        bbox[:, 0] -= int(self.config.bbox_shift * (shift_multiplier - 1))  # Extra shift left

                # Store original bbox for visualization
                if retry_count == 0:
                    original_bbox = bbox.copy()

                # Crop and detect hands in shifted region
                cropped_img, bbox_origin = self.crop_to_bbox(image.copy(), bbox)
                cropped_results, angle = self.detect_hands_with_enhancement(cropped_img)

                if angle != 0:
                    self.logger.debug(f"Hands detected in {img_idx} using {angle}° rotation")

                if len(cropped_results) > 0:
                    # Process detected hand
                    detected_detection = cropped_results[0]
                    detected_hand_side = "right" if current_hand_side == "left" else "left"
                    detected_detection.hand_type = detected_hand_side

                    # Create final crop and update data
                    final_crop, final_origin = self.crop_fixed_size(
                        image, detected_detection,
                        base_offset=bbox_origin,
                        current_image_size=cropped_img.shape[:2]
                    )

                    # Update keypoints data
                    final_keypoint_data = self.detector.get_keypoints_data(
                        detected_detection, cropped_img.shape[:2],
                        [bbox_origin, final_origin]
                    )
                    
                    # Temporarily update data to check shift difference
                    temp_data = {"people": [data["people"][0].copy()]}
                    temp_data["people"][0].update(final_keypoint_data)
                    
                    diff = self.comp_shift_diff(temp_data)
                    
                    if diff > 100:  # Different hand detected
                        # Update actual data and save results
                        data["people"][0].update(final_keypoint_data)
                        
                        # Save final crops
                        cv2.imwrite(self.dirs['blanks'] / f"{img_idx}_cropped_256_{detected_hand_side}_blank.jpg", final_crop)
                        
                        # Create and save visualizations
                        vis_cropped = self.draw_landmarks_and_bbox(
                            cropped_img,
                            detected_detection,
                            bbox=self.compute_shifted_bbox(
                                self.get_roi_points(detected_detection, cropped_img.shape),
                                detected_hand_side
                            ),
                            roi=self.get_roi_points(detected_detection, cropped_img.shape)
                        )
                        
                        final_vis_crop, _ = self.crop_fixed_size(vis_cropped, detected_detection)
                        cv2.imwrite(self.dirs['preds'] / f"{img_idx}_cropped_256_{detected_hand_side}.jpg", final_vis_crop)
                        
                        # Save shifted ROI visualization
                        full_vis = self.draw_landmarks_and_bbox(image, detection, bbox, roi)
                        cv2.imwrite(self.dirs['shifted_roi'] / f"{img_idx}_test_{self.config.bbox_shift}.jpg", full_vis)
                        
                        # Handle potential misclassification
                        if retry_count > 0:
                            # Copy data for misclassified hand
                            data["people"][0][f"hand_{current_hand_side}_keypoints_2d"] = data["people"][0][f"hand_{first_hand_type}_keypoints_2d"]
                            data["people"][0][f"hand_{current_hand_side}_shift"] = data["people"][0][f"hand_{first_hand_type}_shift"]
                            cv2.imwrite(self.dirs['blanks'] / f"{img_idx}_cropped_256_{current_hand_side}_blank.jpg", blank_img)
                        
                        return  # Success!
                    else:
                        # Same hand detected, try with larger shift or opposite direction
                        if shift_multiplier == 1.0:
                            shift_multiplier = 1.5  # First try larger shift in same direction
                        else:
                            # Switch direction and reset shift multiplier
                            current_hand_side = "left" if current_hand_side == "right" else "right"
                            shift_multiplier = 1.0
                            
                        self.logger.info(f"Same hand detected for {img_idx}, retrying with {'larger shift' if shift_multiplier > 1 else 'opposite direction'}")
                else:
                    # No hand detected, try opposite direction
                    current_hand_side = "left" if current_hand_side == "right" else "right"
                    shift_multiplier = 1.0
                    self.logger.info(f"No hand detected for {img_idx}, retrying with opposite direction")

                # Create and save retry visualization
                retry_img = image.copy()
                bbox_retry = self.compute_shifted_bbox(roi.copy(), current_hand_side)
                retry_vis = self.draw_landmarks_and_bbox(retry_img, detection, bbox_retry, roi)
                cv2.imwrite(self.dirs['shifted_roi'] / f"{img_idx}_test_{self.config.bbox_shift}_retry_{retry_count}.jpg", retry_vis)

                retry_count += 1

            except Exception as e:
                self.logger.error(f"Error in retry loop for {img_idx}: {e}")
                retry_count += 1

        # All retries failed
        self.logger.warning(f"Failed to detect second hand for {img_idx} after {max_retries} attempts")
        
        # Save failure visualizations
        cv2.imwrite(self.dirs['failed'] / f"{img_idx}_test_{self.config.bbox_shift}_retry.jpg", retry_vis)
        original_vis = self.draw_landmarks_and_bbox(image, detection, original_bbox, roi)
        cv2.imwrite(self.dirs['failed'] / f"{img_idx}_test_{self.config.bbox_shift}.jpg", original_vis)
        
        return None