import cv2
import json
import os
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare keypoints between two datasets')
parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output with detailed statistics')
args = parser.parse_args()

def _json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)

img_idx = "000100"

# BASE_PATH = "test_bbox_larger_shift_into_opposite"
# BASE_PATH = "test_bbox"
BASE_PATH_1 = "test_bbox_rotation"
BASE_PATH_2 = "test_handdetection_pipeline"

for i in range(100, 500):
    img_idx = f"{i:06d}"

    if not os.path.exists(f"{BASE_PATH_1}/json/{img_idx}_test.json") or not os.path.exists(f"{BASE_PATH_2}/json/{img_idx}_test.json"):
        continue
    
    try:
        kps_1 = _json_load(f"{BASE_PATH_1}/json/{img_idx}_test.json")
        kps_2 = _json_load(f"{BASE_PATH_2}/json/{img_idx}_test.json")

        # Extract keypoints from both datasets
        lkps_1 = np.array(kps_1['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2]
        rkps_1 = np.array(kps_1['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]
        l_shift_1 = np.array(kps_1['people'][0]['hand_left_shift'])
        r_shift_1 = np.array(kps_1['people'][0]['hand_right_shift'])
        
        lkps_2 = np.array(kps_2['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2]
        rkps_2 = np.array(kps_2['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]
        l_shift_2 = np.array(kps_2['people'][0]['hand_left_shift'])
        r_shift_2 = np.array(kps_2['people'][0]['hand_right_shift'])
    except Exception as e:
        print(f"⚠️ Image {img_idx}: Error loading or parsing data - {str(e)}")
        continue


    def compare_arrays_within_percentage(arr1, arr2, threshold_percent=10, comparison_method='relative'):
        """
        Compare two arrays and check if the average difference is within a specified percentage range.
        
        Args:
            arr1, arr2: Arrays to compare (should have same shape)
            threshold_percent: Percentage threshold for average difference (default: 10%)
            comparison_method: 'relative' (relative to arr1) or 'absolute' (relative to max of both)
        
        Returns:
            dict: Contains comparison results and statistics
        """
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        
        if arr1.shape != arr2.shape:
            raise ValueError(f"Arrays must have same shape. Got {arr1.shape} and {arr2.shape}")
        
        # Avoid division by zero
        epsilon = 1e-8
        
        if comparison_method == 'relative':
            # Percentage difference relative to arr1
            percent_diff = np.abs((arr2 - arr1) / (arr1 + epsilon)) * 100
        elif comparison_method == 'absolute':
            # Percentage difference relative to the maximum absolute value
            max_vals = np.maximum(np.abs(arr1), np.abs(arr2))
            percent_diff = np.abs(arr2 - arr1) / (max_vals + epsilon) * 100
        else:
            raise ValueError("comparison_method must be 'relative' or 'absolute'")
        
        # Check which elements are within threshold (for individual statistics)
        within_threshold = percent_diff <= threshold_percent
        
        # Calculate statistics
        num_total = arr1.size
        num_within = np.sum(within_threshold)
        percent_within = (num_within / num_total) * 100
        
        max_diff = np.max(percent_diff)
        mean_diff = np.mean(percent_diff)
        
        # Check if average difference is within threshold (main comparison criterion)
        avg_within_threshold = mean_diff <= threshold_percent
        
        results = {
            'all_within_threshold': avg_within_threshold,  # Now based on average, not individual elements
            'percent_elements_within': percent_within,
            'num_elements_within': num_within,
            'total_elements': num_total,
            'max_percentage_diff': max_diff,
            'mean_percentage_diff': mean_diff,
            'percentage_differences': percent_diff,
            'within_threshold_mask': within_threshold
        }
        
        return results
    
    def print_comparison_summary(comparison_result, threshold_percent=10):
        """Print a summary of the array comparison results."""
        print(f"=== Array Comparison Summary (Threshold: {threshold_percent}%) ===")
        print(f"All elements within threshold: {comparison_result['all_within_threshold']}")
        print(f"Elements within threshold: {comparison_result['num_elements_within']}/{comparison_result['total_elements']} ({comparison_result['percent_elements_within']:.1f}%)")
        print(f"Maximum percentage difference: {comparison_result['max_percentage_diff']:.2f}%")
        print(f"Mean percentage difference: {comparison_result['mean_percentage_diff']:.2f}%")
        
        if not comparison_result['all_within_threshold']:
            outliers = np.where(~comparison_result['within_threshold_mask'])
            print(f"Outlier indices: {list(zip(*outliers))}")
    
    # Compare same keypoints between the two base paths
    try:
        differences_found = False
        diff_details = []
        
        # Compare left hand keypoints between datasets
        left_comparison = compare_arrays_within_percentage(lkps_1, lkps_2, threshold_percent=11, comparison_method='absolute')
        if not left_comparison['all_within_threshold']:
            differences_found = True
            diff_details.append(f"Left hand keypoints: {left_comparison['percent_elements_within']:.1f}% within threshold (max diff: {left_comparison['max_percentage_diff']:.2f}%)")
        
        # Compare right hand keypoints between datasets
        right_comparison = compare_arrays_within_percentage(rkps_1, rkps_2, threshold_percent=11, comparison_method='absolute')
        if not right_comparison['all_within_threshold']:
            differences_found = True
            diff_details.append(f"Right hand keypoints: {right_comparison['percent_elements_within']:.1f}% within threshold (max diff: {right_comparison['max_percentage_diff']:.2f}%)")
        
        # Compare left shift values between datasets
        left_shift_comparison = compare_arrays_within_percentage(l_shift_1, l_shift_2, threshold_percent=11, comparison_method='relative')
        if not left_shift_comparison['all_within_threshold']:
            differences_found = True
            diff_details.append(f"Left shift values: {left_shift_comparison['percent_elements_within']:.1f}% within threshold (max diff: {left_shift_comparison['max_percentage_diff']:.2f}%)")
        
        # Compare right shift values between datasets
        right_shift_comparison = compare_arrays_within_percentage(r_shift_1, r_shift_2, threshold_percent=11, comparison_method='relative')
        if not right_shift_comparison['all_within_threshold']:
            differences_found = True
            diff_details.append(f"Right shift values: {right_shift_comparison['percent_elements_within']:.1f}% within threshold (max diff: {right_shift_comparison['max_percentage_diff']:.2f}%)")
        
        # Output results
        if differences_found:
            if args.verbose:
                print(f"\n=== DIFFERENCES DETECTED for Image {img_idx} ===")
                # Show detailed comparisons in verbose mode
                if not left_comparison['all_within_threshold']:
                    print(f"Left hand keypoints comparison ({BASE_PATH_1} vs {BASE_PATH_2}):")
                    print_comparison_summary(left_comparison, threshold_percent=11)
                
                if not right_comparison['all_within_threshold']:
                    print(f"Right hand keypoints comparison ({BASE_PATH_1} vs {BASE_PATH_2}):")
                    print_comparison_summary(right_comparison, threshold_percent=11)
                
                if not left_shift_comparison['all_within_threshold']:
                    print(f"Left shift values comparison ({BASE_PATH_1} vs {BASE_PATH_2}):")
                    print_comparison_summary(left_shift_comparison, threshold_percent=11)
                
                if not right_shift_comparison['all_within_threshold']:
                    print(f"Right shift values comparison ({BASE_PATH_1} vs {BASE_PATH_2}):")
                    print_comparison_summary(right_shift_comparison, threshold_percent=11)
                
                print("=" * 50)
            else:
                # Minimal output by default
                print(f"❌ Image {img_idx}: Differences detected - {', '.join(diff_details)}")
        elif args.verbose:
            print(f"✅ Image {img_idx}: All keypoints within 11% threshold")
            
    except Exception as e:
        print(f"⚠️ Image {img_idx}: Error during comparison - {str(e)}")
        continue

    