# calibrate_homography.py (Corrected - Grid + Crosses)
# Calculates homography using a 4x6 grid of projected CROSSES (+).

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import tkinter as tk
import time

# --- Configuration ---
CAMERA_CALIBRATION_FILE = "camera_calibration_data.npz"
# Output filename for this grid/cross calibration
HOMOGRAPHY_FILE = "homography_matrix_cross_grid.npz" # <<< New filename
FRAME_WIDTH = 1280; FRAME_HEIGHT = 720
PROJECTOR_WINDOW_NAME = "PROJECTOR - Click Cross Center"
CAMERA_WINDOW_NAME = "CAMERA - Click Cross Center Here"

# --- Grid Definition ---
GRID_COLS = 4
GRID_ROWS = 6
MARGIN = 150 # Pixels from edge

# Generate grid points
projector_points_to_display = []
# ... (Grid point generation logic remains the same as previous grid version) ...
if GRID_COLS > 1 and GRID_ROWS > 1:
    usable_w = FRAME_WIDTH - 2 * MARGIN; usable_h = FRAME_HEIGHT - 2 * MARGIN
    spacing_x = usable_w / (GRID_COLS - 1); spacing_y = usable_h / (GRID_ROWS - 1)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x = int(MARGIN + c * spacing_x); y = int(MARGIN + r * spacing_y)
            projector_points_to_display.append((x, y))
# (Include single row/col/point logic if needed, omitted for brevity)
elif GRID_COLS == 1 and GRID_ROWS > 1: # Special case: single column
    usable_h = FRAME_HEIGHT - 2 * MARGIN; spacing_y = usable_h / (GRID_ROWS - 1)
    x = FRAME_WIDTH // 2
    for r in range(GRID_ROWS): y = int(MARGIN + r * spacing_y); projector_points_to_display.append((x, y))
elif GRID_ROWS == 1 and GRID_COLS > 1: # Special case: single row
    usable_w = FRAME_WIDTH - 2 * MARGIN; spacing_x = usable_w / (GRID_COLS - 1)
    y = FRAME_HEIGHT // 2
    for c in range(GRID_COLS): x = int(MARGIN + c * spacing_x); projector_points_to_display.append((x, y))
else: projector_points_to_display.append((FRAME_WIDTH // 2, FRAME_HEIGHT // 2))


NUM_POINTS_NEEDED = len(projector_points_to_display)
if NUM_POINTS_NEEDED < 4: print("ERROR: Need >= 4 points."); exit()

# --- Marker Appearance ---
CROSS_HALF_LEN = 15 # Half the length of each cross arm (total length = 30)
CROSS_THICKNESS = 2
CROSS_COLOR = (255, 255, 255) # White projected cross

# --- Camera/Display Flip Settings ---
# Use settings consistent with the working main script (script24)
FLIP_CAMERA_INPUT_VERTICALLY = False # <<< Set based on successful settings from script24 test
FLIP_CAMERA_INPUT_HORIZONTALLY = False # <<< Should be False

UNDISTORT_ALPHA = 1.0 # Keep consistent

# --- Mouse callback variables ---
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
# ... (remains the same) ...
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
# ... (remains the same) ...
print("Starting camera...", flush=True); cap=cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: No video device.", flush=True); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
actual_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); actual_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}", flush=True)
w_cam, h_cam = actual_width, actual_height


# --- Calculate Undistortion Maps ---
print(f"Calculating undistortion map with alpha={UNDISTORT_ALPHA}...", flush=True)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w_cam, h_cam), alpha=UNDISTORT_ALPHA, newImgSize=(w_cam, h_cam))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w_cam, h_cam), 5)
print("Undistortion maps created.", flush=True)

# --- Setup Camera Window and Callback ---
cv2.namedWindow(CAMERA_WINDOW_NAME); cv2.setMouseCallback(CAMERA_WINDOW_NAME, mouse_callback)
print("\n--- Grid Homography Calibration (Crosses) ---"); print(f"Need clicks on {NUM_POINTS_NEEDED} crosses.")

# --- Data Collection Loop ---
projector_points_collected = []; camera_points_collected = []; point_index = 0
while point_index < NUM_POINTS_NEEDED:
    proj_pt = projector_points_to_display[point_index]
    print(f"\nProjecting Cross {point_index+1}/{NUM_POINTS_NEEDED} centered at Proj Coords: {proj_pt}")

    # Create canvas and draw CROSS marker
    projection_canvas = np.zeros((h_cam, w_cam, 3), dtype=np.uint8)
    # Draw Horizontal Line
    cv2.line(projection_canvas, (proj_pt[0] - CROSS_HALF_LEN, proj_pt[1]),
             (proj_pt[0] + CROSS_HALF_LEN, proj_pt[1]), CROSS_COLOR, CROSS_THICKNESS)
    # Draw Vertical Line
    cv2.line(projection_canvas, (proj_pt[0], proj_pt[1] - CROSS_HALF_LEN),
             (proj_pt[0], proj_pt[1] + CROSS_HALF_LEN), CROSS_COLOR, CROSS_THICKNESS)

    cv2.imshow(PROJECTOR_WINDOW_NAME, projection_canvas)
    key = cv2.waitKey(500)
    clicked_point = None
    print(f">>> Click cross INTERSECTION {point_index+1} in 'CAMERA' window. (s=skip, q=quit).")
    while clicked_point is None:
        ret, frame = cap.read()
        if not ret: print("Error reading frame", flush=True); time.sleep(0.1); continue

        # Apply Input Flips Consistently
        if FLIP_CAMERA_INPUT_VERTICALLY: frame = cv2.flip(frame, 0)
        if FLIP_CAMERA_INPUT_HORIZONTALLY: frame = cv2.flip(frame, 1)

        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        cv2.putText(undistorted_frame, f"Click cross center {point_index+1}/{NUM_POINTS_NEEDED}. (s=skip, q=quit)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Draw previously collected points
        for i, pt in enumerate(camera_points_collected):
             cv2.circle(undistorted_frame, pt, 5, (0, 255, 0), -1)
             # Optionally draw number
             # cv2.putText(undistorted_frame, str(i+1), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

        cv2.imshow(CAMERA_WINDOW_NAME, undistorted_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("Aborted.", flush=True); cap.release(); cv2.destroyAllWindows(); exit()
        elif key == ord('s'): print("Skipping.", flush=True); clicked_point = (-1, -1); break
        elif clicked_point is not None: break
    if clicked_point != (-1, -1) and clicked_point is not None:
        projector_points_collected.append(proj_pt)
        camera_points_collected.append(clicked_point)
        point_index += 1
    elif clicked_point == (-1,-1): print("Retrying point."); time.sleep(0.5)

# --- Calculate Homography ---
if len(projector_points_collected) >= 4 and len(camera_points_collected) == len(projector_points_collected):
    print(f"\nCalculating Homography using {len(projector_points_collected)} points...", flush=True)
    np_camera_pts = np.array(camera_points_collected, dtype=np.float32)
    np_projector_pts = np.array(projector_points_collected, dtype=np.float32)
    try:
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