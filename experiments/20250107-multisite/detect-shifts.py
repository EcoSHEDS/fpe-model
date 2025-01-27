import cv2
import numpy as np
import sys
from tqdm import tqdm

def detect_camera_shift_orb(img1_path, img2_path, threshold=0.7):
    # Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BF matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate average distance of top matches
    num_top_matches = min(50, len(matches))
    avg_distance = np.mean([m.distance for m in matches[:num_top_matches]])
    
    return avg_distance

def detect_camera_shift_sift(img1_path, img2_path):
    # Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # Define ROI mask - focus on buildings and stream banks
    height, width = img1.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    roi_points = np.array([[width//4, height//4], 
                          [3*width//4, height//4],
                          [3*width//4, 3*height//4],
                          [width//4, 3*height//4]])
    cv2.fillPoly(mask, [roi_points], 255)
    
    # Apply mask
    img1_masked = cv2.bitwise_and(img1, img1, mask=mask)
    img2_masked = cv2.bitwise_and(img2, img2, mask=mask)
    
    # Edge detection
    edges1 = cv2.Canny(img1_masked, 100, 200)
    edges2 = cv2.Canny(img2_masked, 100, 200)
    
    # Find homography between images
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(edges1, None)
    kp2, des2 = sift.detectAndCompute(edges2, None)
    
    if des1 is None or des2 is None:
        return float('inf')
        
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    return len(good_matches)

def analyze_sequence(image_paths):
    shifts = []
    for i in tqdm(range(len(image_paths)-1), desc="Analyzing image pairs"):
        dist = detect_camera_shift_sift(image_paths[i], image_paths[i+1])
        shifts.append(dist)
    return shifts

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect-shifts.py <image_list_file>")
        sys.exit(1)
        
    with open(sys.argv[1]) as f:
        image_paths = [line.strip() for line in f]
        
    shifts = analyze_sequence(image_paths)
    print(shifts)
