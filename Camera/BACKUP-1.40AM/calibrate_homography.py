# calibrate_homography.py (Corrected - WITH FLIP FLAGS)
# Calculates the homography matrix mapping camera view points to projector points.

import os
# >>> FIX FOR POTENTIAL WEBCAM ISSUES ON WINDOWS <<<
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
# --------------------------------------------------
import cv2
import numpy as np
import tkinter as tk
import time # Keep time import

# --- Configuration ---
CAMERA_CALIBRATION_FILE = "camera_calibration_data.npz"
HOMOGRAPHY_FILE = "homography_matrix.npz" # Output file
FRAME_WIDTH = 1280; FRAME_HEIGHT = 720
PROJECTOR_WINDOW_NAME = "PROJECTOR - Click Dots"
CAMERA_WINDOW_NAME = "CAMERA - Click Dots Here"

# --- FIXES / SETTINGS ---
# >>> SET THESE FLAGS CONSISTENTLY WITH THE MAIN SCRIPT <<<
# Set True ONLY if Camera Hardware is mounted upside-down
FLIP_CAMERA_INPUT_VERTICALLY = False # <<< SET THIS TO FALSE FOR CURRENT TEST PLAN
# Set True ONLY if MediaPipe detects wrong handedness (Left as Right)
FLIP_CAMERA_INPUT_HORIZONTALLY = False # <<< KEEP FALSE
# -----------------------------------------------------

UNDISTORT_ALPHA = 1.0 # Keep consistent with main script (e.g., script24)

# Points to project
MARGIN = 150
projector_points_to_display = [(MARGIN, MARGIN), (FRAME_WIDTH-MARGIN, MARGIN),
                               (FRAME_WIDTH-MARGIN, FRAME_HEIGHT-MARGIN), (MARGIN, FRAME_HEIGHT-MARGIN),
                               (FRAME_WIDTH//2, FRAME_HEIGHT//2)]
NUM_POINTS_NEEDED = len(projector_points_to_display)
MARKER_RADIUS = 15; MARKER_COLOR = (255, 255, 255)
clicked_point = None
# --- End Configuration ---

# --- Mouse Callback Function ---
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"  Clicked at Camera Coordinates: {clicked_point}", flush=True)
# -----------------------------

# --- Load Camera Calibration ---
try:
    with np.load(CAMERA_CALIBRATION_FILE) as data:
        camera_matrix=data['camera_matrix']; dist_coeffs=data['dist_coeffs']
        print("Camera calibration data loaded.", flush=True)
except Exception as e: print(f"Error loading camera calib: {e}", flush=True); exit()

# --- Projector Fullscreen Setup ---
print("Setting up projector window...", flush=True)
try:
    root=tk.Tk(); root.withdraw(); screen_width=root.winfo_screenwidth(); root.destroy()
    monitor_x_offset=screen_width; monitor_y_offset=0
    cv2.namedWindow(PROJECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(PROJECTOR_WINDOW_NAME, monitor_x_offset, monitor_y_offset)
    cv2.setWindowProperty(PROJECTOR_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Projector window configured.", flush=True)
except Exception as e: print(f"Error setting up proj: {e}. Normal window.", flush=True); cv2.namedWindow(PROJECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL)

# --- Initialize Camera ---
print("Starting camera...", flush=True); cap=cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: No video device.", flush=True); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
actual_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); actual_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}", flush=True)
w_cam, h_cam = actual_width, actual_height

# --- Calculate Undistortion Maps ---
print(f"Calculating undistortion map with alpha={UNDISTORT_ALPHA}...", flush=True) # Added this print
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w_cam, h_cam), alpha=UNDISTORT_ALPHA, newImgSize=(w_cam, h_cam))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w_cam, h_cam), 5)
print("Undistortion maps created.", flush=True)

# --- Setup Camera Window and Callback ---
cv2.namedWindow(CAMERA_WINDOW_NAME); cv2.setMouseCallback(CAMERA_WINDOW_NAME, mouse_callback)
print("\n--- Homography Calibration ---"); print(f"Need clicks on {NUM_POINTS_NEEDED} dots.")

# --- Data Collection Loop ---
projector_points_collected = []; camera_points_collected = []; point_index = 0
while point_index < NUM_POINTS_NEEDED:
    proj_pt = projector_points_to_display[point_index]
    print(f"\nProjecting Point {point_index+1}/{NUM_POINTS_NEEDED} at Proj Coords: {proj_pt}")
    projection_canvas = np.zeros((h_cam, w_cam, 3), dtype=np.uint8)
    cv2.circle(projection_canvas, proj_pt, MARKER_RADIUS, MARKER_COLOR, -1)
    cv2.imshow(PROJECTOR_WINDOW_NAME, projection_canvas)
    key = cv2.waitKey(500) # Give time for display
    clicked_point = None
    print(">>> Click the dot center in 'CAMERA' window. (s=skip, q=quit).")
    while clicked_point is None:
        ret, frame = cap.read()
        if not ret: print("Error reading frame", flush=True); time.sleep(0.1); continue # Added delay

        # >>> APPLY INPUT FLIPS CONSISTENTLY <<<
        if FLIP_CAMERA_INPUT_VERTICALLY:
            frame = cv2.flip(frame, 0)
        if FLIP_CAMERA_INPUT_HORIZONTALLY:
            frame = cv2.flip(frame, 1)
        # --------------------------------------

        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        cv2.putText(undistorted_frame, f"Click dot {point_index+1}/{NUM_POINTS_NEEDED}. (s=skip, q=quit)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for i, pt in enumerate(camera_points_collected):
             cv2.circle(undistorted_frame, pt, 5, (0, 255, 0), -1)
             cv2.putText(undistorted_frame, str(i+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        cv2.imshow(CAMERA_WINDOW_NAME, undistorted_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Aborted.", flush=True); cap.release(); cv2.destroyAllWindows(); exit()
        elif key == ord('s'): print("Skipping.", flush=True); clicked_point = (-1, -1); break
        elif clicked_point is not None: break # Exit inner loop on click
    if clicked_point != (-1, -1) and clicked_point is not None: # If not skipped and clicked
        projector_points_collected.append(proj_pt)
        camera_points_collected.append(clicked_point)
        point_index += 1
    elif clicked_point == (-1,-1): # If skipped
        print("Retrying point."); time.sleep(0.5)

# --- Calculate Homography ---
if len(projector_points_collected) >= 4 and len(camera_points_collected) == len(projector_points_collected):
    print("\nCalculating Homography...", flush=True)
    np_camera_pts = np.array(camera_points_collected, dtype=np.float32)
    np_projector_pts = np.array(projector_points_collected, dtype=np.float32)
    try:
        # Use Camera points as source, Projector points as destination
        H, status = cv2.findHomography(np_camera_pts, np_projector_pts, cv2.RANSAC, 5.0)
        if H is not None:
            print("Homography OK!\nMatrix H (Cam->Proj):\n", H)
            try: np.savez(HOMOGRAPHY_FILE, homography=H); print(f"Saved: {HOMOGRAPHY_FILE}")
            except Exception as e: print(f"Error saving: {e}", flush=True)
        else: print("Homography Failed. Status:", status, flush=True)
    except Exception as e: print(f"Error during calculation: {e}", flush=True)
else: print("\nNeed >= 4 valid points. Homography not calculated.", flush=True)

# --- Cleanup ---
print("Calibration script finished.", flush=True); cap.release(); cv2.destroyAllWindows()