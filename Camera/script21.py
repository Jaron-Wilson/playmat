# script21.py
# MANUAL ALIGNMENT HELPER
# Projects ONE static dot. Debug view shows camera feed + detected dot position.
# GOAL: Adjust projector physically until detected dot (GREEN) matches TARGET coords.

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import tkinter as tk
import time

# --- Configuration ---
CAMERA_CALIBRATION_FILE = "camera_calibration_data.npz"
FRAME_WIDTH = 1280; FRAME_HEIGHT = 720
PROJECTOR_WINDOW_NAME = "PROJECTOR - Alignment Target"
DEBUG_WINDOW_NAME = "CAMERA - Adjust Projector"

# >>> Position where the white dot will be PROJECTED <<<
# >>> Try center first, adjust if needed <<<
TARGET_PROJECTOR_POS = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)

# --- Marker Appearance ---
PROJECTED_DOT_RADIUS = 20
PROJECTED_DOT_COLOR = (255, 255, 255) # White projected dot
DETECTED_DOT_COLOR = (0, 255, 0)      # Green marker for where camera sees it
DETECTED_DOT_RADIUS = 10

# --- Detection Tuning ---
# Adjust this threshold based on ambient light and projector brightness
# Lower value = detects dimmer spots, Higher value = requires brighter spots
THRESHOLD_VALUE = 200 # START HERE, ADJUST IF NEEDED (0-255)

# --- Camera/Display Flip Settings ---
FLIP_CAMERA_INPUT_VERTICALLY = True # Or False, based on camera mount
FLIP_CAMERA_INPUT_HORIZONTALLY = False # Should be False if handedness was correct

FLIP_DEBUG_VERTICALLY = False # Adjust if Monitor 1 view is wrong
# FLIP_OUTPUT_VERTICALLY = False # Not needed for static dot projection usually

UNDISTORT_ALPHA = 1.0 # Use alpha=1 for clearer central view during calibration
SHOW_DEBUG_WINDOW = True
DEBUG_WINDOW_WIDTH = 960; DEBUG_WINDOW_HEIGHT = 540
# --- End Configuration ---

# --- Load Camera Calibration ---
try:
    with np.load(CAMERA_CALIBRATION_FILE) as data:
        camera_matrix=data['camera_matrix']; dist_coeffs=data['dist_coeffs']
        img_size_calib=data['img_size']; print("Calibration data loaded.", flush=True)
except Exception as e: print(f"Error loading camera calib: {e}", flush=True); exit()

# --- Projector Fullscreen Setup ---
print("Setting up projector window...", flush=True)
try:
    root=tk.Tk(); root.withdraw(); screen_width=root.winfo_screenwidth(); root.destroy()
    monitor_x_offset=screen_width; monitor_y_offset=0
    cv2.namedWindow(PROJECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL); cv2.moveWindow(PROJECTOR_WINDOW_NAME, monitor_x_offset, monitor_y_offset)
    cv2.setWindowProperty(PROJECTOR_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Projector window configured.", flush=True)
except Exception as e: print(f"Error setting up projector: {e}. Normal window.", flush=True); cv2.namedWindow(PROJECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL)

# --- Initialize Camera ---
print("Starting camera...", flush=True); cap=cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: No video device.", flush=True); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
actual_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); actual_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}", flush=True)
if img_size_calib[0] != actual_width or img_size_calib[1] != actual_height: print(f"Warning: Resolution mismatch!", flush=True)

# --- Debug Window Setup ---
if SHOW_DEBUG_WINDOW:
    print("Setting up debug window (Monitor 1)...", flush=True); cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(DEBUG_WINDOW_NAME, 50, 50)
    try:
        debug_w=DEBUG_WINDOW_WIDTH if DEBUG_WINDOW_WIDTH > 0 else actual_width; debug_h=DEBUG_WINDOW_HEIGHT if DEBUG_WINDOW_HEIGHT > 0 else actual_height
        cv2.resizeWindow(DEBUG_WINDOW_NAME, debug_w, debug_h); print(f"Debug window resized to {debug_w}x{debug_h}.", flush=True)
    except Exception as e: print(f"Could not resize debug window: {e}", flush=True)
    print("Debug window (Monitor 1) enabled.", flush=True)

# --- Calculate Undistortion Maps ---
print(f"Calculating undistortion map with alpha={UNDISTORT_ALPHA}...", flush=True)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (actual_width, actual_height), alpha=UNDISTORT_ALPHA, newImgSize=(actual_width, actual_height))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (actual_width, actual_height), 5)
print("Undistortion maps created.", flush=True)

# --- Prepare Static Projection Canvas ---
projection_canvas = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)
cv2.circle(projection_canvas, TARGET_PROJECTOR_POS, PROJECTED_DOT_RADIUS, PROJECTED_DOT_COLOR, -1)
print(f"\nProjecting ONE target dot at {TARGET_PROJECTOR_POS}. Press 'q' to quit.")

try:
    while True:
        # --- Display Static Dot on Projector ---
        # (No vertical output flip needed as it's just one dot)
        cv2.imshow(PROJECTOR_WINDOW_NAME, projection_canvas)

        # --- Process Camera Feed ---
        ret, frame = cap.read()
        if not ret: print("Error: Can't receive frame.", flush=True); time.sleep(0.1); continue

        # Apply Input Flips
        if FLIP_CAMERA_INPUT_VERTICALLY: frame = cv2.flip(frame, 0)
        if FLIP_CAMERA_INPUT_HORIZONTALLY: frame = cv2.flip(frame, 1) # Should be False

        # 1. Undistort
        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR); h, w, _ = undistorted_frame.shape

        # 2. Prepare frame for detection
        gray_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

        # 3. Threshold to find the bright projected dot
        _, thresh_frame = cv2.threshold(gray_frame, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # 4. Find Contours
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_center = None
        if contours:
            # Assume the largest contour is our dot (or first one if only one expected)
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                # Calculate centroid
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_center = (cx, cy)

        # === DISPLAY DEBUG VIEW ===
        if SHOW_DEBUG_WINDOW:
            debug_display_frame = undistorted_frame.copy()

            # Draw instructions and target coordinates
            cv2.putText(debug_display_frame, "Adjust Projector until GREEN matches TARGET coords", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(debug_display_frame, f"TARGET Proj Coords: {TARGET_PROJECTOR_POS}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if detected_center:
                # Draw GREEN marker where the dot was detected
                cv2.circle(debug_display_frame, detected_center, DETECTED_DOT_RADIUS, DETECTED_DOT_COLOR, 2)
                cv2.line(debug_display_frame, (detected_center[0] - 15, detected_center[1]), (detected_center[0] + 15, detected_center[1]), DETECTED_DOT_COLOR, 1) # Crosshairs
                cv2.line(debug_display_frame, (detected_center[0], detected_center[1] - 15), (detected_center[0], detected_center[1] + 15), DETECTED_DOT_COLOR, 1) # Crosshairs
                cv2.putText(debug_display_frame, f"Detected Cam Coords: {detected_center}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DETECTED_DOT_COLOR, 2)
            else:
                cv2.putText(debug_display_frame, "Dot not detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            # Apply vertical flip to DEBUG view if needed
            if FLIP_DEBUG_VERTICALLY:
                 debug_display_frame = cv2.flip(debug_display_frame, 0)

            # Show the debug frame
            cv2.imshow(DEBUG_WINDOW_NAME, debug_display_frame)


        # Check for Quit Key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    print("Releasing resources...", flush=True); cap.release(); cv2.destroyAllWindows()
    print("Alignment helper finished.", flush=True)