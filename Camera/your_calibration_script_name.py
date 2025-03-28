import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np

import time

# --- Configuration ---
# IMPORTANT: Change these values based on your checkerboard
CHECKERBOARD_DIMS = (6, 5)  # Number of inner corners (width-1, height-1)
# IMPORTANT: Measure one square edge on your physical printout VERY accurately (e.g., in mm)
SQUARE_SIZE_MM = 19.0       # <<<--- CHANGE THIS TO YOUR MEASURED VALUE
MIN_FRAMES_FOR_CALIBRATION = 22 # Number of good views needed
CALIBRATION_FILE = "camera_calibration_data.npz" # File to save results
FRAME_WIDTH = 1280 # Optional: Set camera resolution width
FRAME_HEIGHT = 720  # Optional: Set camera resolution height
# --- End Configuration ---

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (7,5,0)
# These are the 3D coordinates in the checkerboard's own coordinate system
objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM # Scale to real-world size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

print("Starting camera...")
cap = cv2.VideoCapture(0) # Use 0 for default camera, change if needed

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Optional: Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print(f"Camera resolution set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print("\n--- Calibration Procedure ---", flush=True)
print(f"Need {MIN_FRAMES_FOR_CALIBRATION} valid checkerboard views.", flush=True)
print("Show the checkerboard to the camera from various angles and distances.", flush=True)
print("Make sure the board is flat and well-lit.", flush=True)
print("Press 'q' to quit early (calibration will not be performed).", flush=True)

frames_captured = 0
last_capture_time = 0
capture_delay = 1.5 # Seconds delay between successful captures

try:
    while frames_captured < MIN_FRAMES_FOR_CALIBRATION:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # Use CALIB_CB_ADAPTIVE_THRESH for potentially better results in varying lighting
        # Use CALIB_CB_FAST_CHECK to quickly reject images with no checkerboard
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS,
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                        cv2.CALIB_CB_FAST_CHECK +
                                                        cv2.CALIB_CB_NORMALIZE_IMAGE)

        current_time = time.time()

        # If found, add object points, image points (after refining them)
        if ret_corners == True and (current_time - last_capture_time > capture_delay):
            frames_captured += 1
            last_capture_time = current_time

            # Refine corner locations for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_subpix)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame, CHECKERBOARD_DIMS, corners_subpix, ret_corners)
            status_text = f"Corners Found! Captured: {frames_captured}/{MIN_FRAMES_FOR_CALIBRATION}"
            print(status_text, flush=True)
            # Add a small pause to allow board movement
            # time.sleep(0.5) # Can be disruptive, using time check instead

        else:
            status_text = f"Looking... Captured: {frames_captured}/{MIN_FRAMES_FOR_CALIBRATION}"


        # Display the status and frame
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Calibration View - Press Q to Quit Early', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Calibration aborted by user.")
            frames_captured = -1 # Signal abortion
            break
        elif key == ord(' '): # Optional: manual capture trigger
             if ret_corners == True:
                 print("Manual capture triggered (if corners valid and delay passed)", flush=True)
             else:
                 print("Manual capture failed - no corners detected", flush=True)


finally:
    # Release resources AFTER the loop finishes or breaks
    print("Releasing camera...", flush=True)
    cap.release()
    cv2.destroyAllWindows()
    print("Windows closed.", flush=True)


# --- Perform Calibration (only if enough frames were captured) ---
if frames_captured >= MIN_FRAMES_FOR_CALIBRATION:
    print("\nStarting calibration calculation...", flush=True)
    if not objpoints or not imgpoints:
         print("Error: No points collected, cannot calibrate.", flush=True)
    elif len(objpoints) != len(imgpoints) or len(objpoints) < MIN_FRAMES_FOR_CALIBRATION:
         print(f"Error: Insufficient points ({len(objpoints)} collected, {MIN_FRAMES_FOR_CALIBRATION} needed). Cannot calibrate.")
    else:
        try:
            # Get the image size from the first valid grayscale image used
            h, w = gray.shape[:2] # Use the last successfully processed gray image size
            print(f"Calibrating using image size: {w}x{h}")

            # Perform calibration
            # K: Camera Matrix (Intrinsic)
            # D: Distortion Coefficients (k1, k2, p1, p2, k3)
            # rvecs: Rotation vectors for each view
            # tvecs: Translation vectors for each view
            ret_calib, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

            if ret_calib:
                print("\n--- Calibration Successful ---")
                print("Camera Matrix (K):")
                print(K)
                print("\nDistortion Coefficients (D):")
                print(D)

                # Calculate reprojection error (lower is better)
                mean_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error
                print(f"\nMean Reprojection Error: {mean_error / len(objpoints):.4f} pixels")
                print("Lower error indicates better calibration (typically < 1.0 is good).")


                # --- Save Calibration Data ---
                print(f"\nSaving calibration data to: {CALIBRATION_FILE}")
                np.savez(CALIBRATION_FILE, camera_matrix=K, dist_coeffs=D, rvecs=rvecs, tvecs=tvecs, ret=ret_calib, img_size=(w,h))
                print("Data saved successfully.")

                print("\n--- How to Use Later ---")
                print(f"Load data using: ")
                print("with np.load('camera_calibration_data.npz') as data:")
                print("  camera_matrix = data['camera_matrix']")
                print("  dist_coeffs = data['dist_coeffs']")
                print("  img_size = data['img_size']")
                print("\nThen use cv2.undistort(frame, camera_matrix, dist_coeffs) to correct images.")

            else:
                print("\n--- Calibration Failed ---")
                print("Check checkerboard detection, number of views, and square size.")

        except Exception as e:
            print(f"\nAn error occurred during calibration calculation: {e}")

elif frames_captured != -1: # Check if not aborted early
    print("\nCalibration not performed: Insufficient valid frames captured.")

print("\nScript finished.")