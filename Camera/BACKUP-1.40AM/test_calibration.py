import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np

# --- Configuration ---
CALIBRATION_FILE = "camera_calibration_data.npz"
FRAME_WIDTH = 1280 # Use the same resolution used for calibration
FRAME_HEIGHT = 720
# --- End Configuration ---

# Load the calibration data
try:
    with np.load(CALIBRATION_FILE) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        img_size = data['img_size'] # Optional: verify it matches current resolution
        print("Calibration data loaded successfully.", flush=True)
        print(f"Image size from calibration: {img_size}", flush=True)
        print("Camera Matrix (K):\n", camera_matrix, flush=True)
        print("Distortion Coefficients (D):\n", dist_coeffs, flush=True)
except FileNotFoundError:
    print(f"Error: Calibration file '{CALIBRATION_FILE}' not found.", flush=True)
    print("Please run the calibration script first.", flush=True)
    exit()
except Exception as e:
    print(f"Error loading calibration file: {e}", flush=True)
    exit()

print("\nStarting camera...", flush=True)
cap = cv2.VideoCapture(0) # Use 0 for default camera

if not cap.isOpened():
    print("Error: Could not open video device.", flush=True)
    exit()

# Set camera resolution (important to match calibration resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Verify the resolution was set correctly
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution set to: {actual_width}x{actual_height}", flush=True)
if img_size[0] != actual_width or img_size[1] != actual_height:
     print(f"Warning: Current camera resolution ({actual_width}x{actual_height}) differs from calibration resolution ({img_size[0]}x{img_size[1]}). Undistortion might be less accurate.", flush=True)


# --- Optional: Calculate Optimal New Camera Matrix for Cropping ---
# This can remove black borders introduced by undistortion, but might crop the image slightly
h, w = int(actual_height), int(actual_width)
# alpha=0 means all pixels retained, possibly with black areas
# alpha=1 means all 'bad' pixels are removed, potentially cropping significantly
# Values between 0 and 1 offer a balance. Let's try 0 for max view first.
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h))
# roi is the region of interest (x,y,w,h) containing valid pixels after undistortion and cropping (if alpha > 0)
x, y, w_roi, h_roi = roi
print(f"Region of Interest (ROI) for alpha=0: x={x}, y={y}, w={w_roi}, h={h_roi}", flush=True)
# You could use alpha=1 for maximum cropping:
# new_camera_matrix_cropped, roi_cropped = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), alpha=1, newImgSize=(w,h))


print("\nDisplaying original and undistorted views. Press 'q' to quit.", flush=True)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...", flush=True)
            break

        # --- Undistort the frame ---
        # Method 1: Using original camera matrix (might have black borders)
        # undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, None)

        # Method 2: Using the optimal new camera matrix (often preferred)
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # --- Optional: Crop using ROI from getOptimalNewCameraMatrix (useful if alpha > 0) ---
        # If you use alpha > 0 in getOptimalNewCameraMatrix and want to crop:
        # if w_roi > 0 and h_roi > 0: # Check if ROI is valid
        #    undistorted_frame = undistorted_frame[y:y+h_roi, x:x+w_roi]
        #    # Resize back if needed, or adjust downstream processing
        #    # undistorted_frame = cv2.resize(undistorted_frame, (w, h)) # Example resize
        # else: # Handle case where ROI is invalid (shouldn't happen with valid calibration)
        #    pass # Use the uncropped undistorted frame


        # --- Display side-by-side ---
        # Ensure both frames have the same dimensions if you cropped
        # (If not cropping, they will have the same dimensions)
        h_orig, w_orig = frame.shape[:2]
        h_undist, w_undist = undistorted_frame.shape[:2]

        # If shapes differ due to cropping, you might need resizing for simple stacking
        # Example: if cropped, resize undistorted_frame back to original size
        # if h_orig != h_undist or w_orig != w_undist:
        #    undistorted_frame = cv2.resize(undistorted_frame, (w_orig, h_orig))

        comparison_frame = np.hstack((frame, undistorted_frame)) # Stack horizontally

        # Add labels
        cv2.putText(comparison_frame, "Original (Distorted)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison_frame, "Undistorted", (w_orig + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Original vs Undistorted - Press Q to Quit', comparison_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("Releasing camera...", flush=True)
    cap.release()
    cv2.destroyAllWindows()
    print("Windows closed.", flush=True)
    print("Test script finished.", flush=True)