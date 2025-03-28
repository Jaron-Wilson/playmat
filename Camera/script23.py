# script23.py
# TESTING INVERSE HOMOGRAPHY
# Loads the SAME homography matrix, but calculates and uses its INVERSE.

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk

# --- Configuration ---
CAMERA_CALIBRATION_FILE = "camera_calibration_data.npz"
HOMOGRAPHY_FILE = "homography_matrix.npz" # Load the existing one
FRAME_WIDTH = 1280; FRAME_HEIGHT = 720
MAX_HANDS = 2; DETECTION_CONFIDENCE = 0.6; TRACKING_CONFIDENCE = 0.6
PROJECTOR_WINDOW_NAME = "Projector Output"; DEBUG_WINDOW_NAME = "Camera Debug View (Main Monitor)"

# --- FIXES / SETTINGS ---
FLIP_CAMERA_INPUT_VERTICALLY = True # Or False, based on your camera mount
FLIP_CAMERA_INPUT_HORIZONTALLY = False # Should be False if handedness is correct
FLIP_DEBUG_VERTICALLY = False
FLIP_OUTPUT_VERTICALLY = False
UNDISTORT_ALPHA = 1.0 # Keep consistent with last calibration
SHOW_DEBUG_WINDOW = True
DEBUG_WINDOW_WIDTH = 960; DEBUG_WINDOW_HEIGHT = 540

# --- Styles ---
PROJECTED_DOT_COLOR=(0,0,255); PROJECTED_DOT_RADIUS=15
DEBUG_LANDMARK_COLOR=(0,255,0); DEBUG_DOT_RADIUS=10
# --- End Configuration ---

# --- Load Camera Calibration ---
try:
    with np.load(CAMERA_CALIBRATION_FILE) as data:
        camera_matrix=data['camera_matrix']; dist_coeffs=data['dist_coeffs']
        img_size_calib=data['img_size']; print("Calibration data loaded.", flush=True)
except Exception as e: print(f"Error loading camera calib: {e}", flush=True); exit()

# --- Load Homography Matrix ---
try:
    with np.load(HOMOGRAPHY_FILE) as data:
        H_matrix_loaded = data['homography'] # Load the original matrix
        if H_matrix_loaded is None or H_matrix_loaded.shape != (3, 3): raise ValueError("Invalid Matrix loaded")
    print(f"Homography loaded from '{HOMOGRAPHY_FILE}'.")

    # >>> CALCULATE THE INVERSE MATRIX <<<
    try:
        H_matrix_inverse = np.linalg.inv(H_matrix_loaded)
        print("Successfully calculated INVERSE homography matrix.")
        # Decide which matrix to USE for transformation
        # Let's USE THE INVERSE for this script
        H_matrix_to_use = H_matrix_inverse
        print("Using INVERSE matrix H^-1 (Proj -> Cam effectively) for transform.")
        # If you wanted to use the original: H_matrix_to_use = H_matrix_loaded
    except np.linalg.LinAlgError:
        print("ERROR: Could not calculate inverse of the loaded homography matrix. It might be singular.")
        print("This likely indicates a very bad calibration. Re-run calibrate_homography.py accurately!")
        exit()
    # -----------------------------------

except Exception as e: print(f"ERROR loading or processing '{HOMOGRAPHY_FILE}': {e}", flush=True); exit()

# --- Initialize MediaPipe Hands ---
try:
    mp_hands=mp.solutions.hands; hands=mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_HANDS, min_detection_confidence=DETECTION_CONFIDENCE, min_tracking_confidence=TRACKING_CONFIDENCE)
    mp_drawing=mp.solutions.drawing_utils; mp_drawing_styles=mp.solutions.drawing_styles
    print("MediaPipe Hands initialized.", flush=True)
except Exception as e: print(f"Error initializing MediaPipe: {e}", flush=True); exit()

# --- Projector Fullscreen Setup ---
# ... (Setup code remains the same) ...
print("Setting up projector window (Monitor 2)...", flush=True)
try:
    root=tk.Tk(); root.withdraw(); screen_width=root.winfo_screenwidth(); root.destroy()
    monitor_x_offset=screen_width; monitor_y_offset=0
    cv2.namedWindow(PROJECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL); cv2.moveWindow(PROJECTOR_WINDOW_NAME, monitor_x_offset, monitor_y_offset)
    cv2.setWindowProperty(PROJECTOR_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Projector window (Monitor 2) configured.", flush=True)
except Exception as e: print(f"Error setting up projector: {e}. Normal window.", flush=True); cv2.namedWindow(PROJECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL)

# --- Initialize Camera ---
# ... (Setup code remains the same) ...
print("Starting camera...", flush=True); cap=cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: No video device.", flush=True); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
actual_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); actual_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}", flush=True)

# --- Debug Window Setup ---
# ... (Setup code remains the same) ...
if SHOW_DEBUG_WINDOW:
    print("Setting up debug window (Monitor 1)...", flush=True); cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(DEBUG_WINDOW_NAME, 50, 50)
    try:
        debug_w=DEBUG_WINDOW_WIDTH if DEBUG_WINDOW_WIDTH > 0 else actual_width; debug_h=DEBUG_WINDOW_HEIGHT if DEBUG_WINDOW_HEIGHT > 0 else actual_height
        cv2.resizeWindow(DEBUG_WINDOW_NAME, debug_w, debug_h); print(f"Debug window resized to {debug_w}x{debug_h}.", flush=True)
    except Exception as e: print(f"Could not resize debug window: {e}", flush=True)
    print("Debug window (Monitor 1) enabled.", flush=True)

# --- Calculate Undistortion Maps ---
# ... (Setup code remains the same) ...
print(f"Calculating undistortion map with alpha={UNDISTORT_ALPHA}...", flush=True)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (actual_width, actual_height), alpha=UNDISTORT_ALPHA, newImgSize=(actual_width, actual_height))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (actual_width, actual_height), 5)
print("Undistortion maps created.", flush=True)


print("\nStarting main loop using INVERSE Homography. Press 'q' to quit.", flush=True)
prev_time = 0; hand_positions = []; projected_positions = []

try:
    while True:
        ret, frame = cap.read()
        if not ret: print("Error: Can't receive frame.", flush=True); break

        if FLIP_CAMERA_INPUT_VERTICALLY: frame = cv2.flip(frame, 0)
        if FLIP_CAMERA_INPUT_HORIZONTALLY: frame = cv2.flip(frame, 1)

        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR); h, w, _ = undistorted_frame.shape
        rgb_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB); rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame); rgb_frame.flags.writeable = True

        projection_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        undistorted_frame_for_debug = undistorted_frame.copy() if SHOW_DEBUG_WINDOW else None

        current_hand_positions=[]; current_projected_positions=[]; detected_handedness=[]
        if results.multi_hand_landmarks:
            if results.multi_handedness:
                for hand_info in results.multi_handedness: detected_handedness.append(hand_info.classification[0].label)

            for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                target_landmark = mp_hands.HandLandmark.INDEX_FINGER_TIP
                landmark_coords = hand_landmarks.landmark[target_landmark]
                cx = int(landmark_coords.x * w); cy = int(landmark_coords.y * h)

                if 0 <= cx < w and 0 <= cy < h:
                    current_hand_positions.append((cx, cy))
                    cam_pt = np.array([[[cx, cy]]], dtype=np.float32)

                    # >>> USE THE INVERSE MATRIX FOR TRANSFORMATION <<<
                    proj_pt_transformed = cv2.perspectiveTransform(cam_pt, H_matrix_to_use) # Using H_inverse here
                    # -------------------------------------------------

                    if proj_pt_transformed is not None and proj_pt_transformed.shape == (1,1,2):
                        proj_x = int(proj_pt_transformed[0,0,0]); proj_y = int(proj_pt_transformed[0,0,1])
                        current_projected_positions.append((proj_x, proj_y))
                        if 0 <= proj_x < w and 0 <= proj_y < h:
                            cv2.circle(projection_canvas, (proj_x, proj_y), PROJECTED_DOT_RADIUS, PROJECTED_DOT_COLOR, cv2.FILLED)

                    if SHOW_DEBUG_WINDOW and undistorted_frame_for_debug is not None:
                        mp_drawing.draw_landmarks(undistorted_frame_for_debug, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                        handedness_label = detected_handedness[hand_no] if hand_no < len(detected_handedness) else "N/A"
                        cv2.circle(undistorted_frame_for_debug, (cx, cy), DEBUG_DOT_RADIUS, DEBUG_LANDMARK_COLOR, cv2.FILLED)
                        cv2.putText(undistorted_frame_for_debug, f"H{hand_no}({handedness_label}):({cx},{cy})", (cx+15, cy-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, DEBUG_LANDMARK_COLOR, 2)

        hand_positions=current_hand_positions; projected_positions=current_projected_positions

        curr_time=time.time(); fps=1/(curr_time-prev_time) if (curr_time-prev_time)>0 else 0; prev_time=curr_time; fps_text=f"FPS: {int(fps)}"
        cv2.putText(projection_canvas, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if SHOW_DEBUG_WINDOW and undistorted_frame_for_debug is not None: cv2.putText(undistorted_frame_for_debug, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        frame_to_display_projector = projection_canvas
        if FLIP_OUTPUT_VERTICALLY: frame_to_display_projector = cv2.flip(frame_to_display_projector, 0)

        frame_to_display_debug = undistorted_frame_for_debug
        if SHOW_DEBUG_WINDOW and frame_to_display_debug is not None:
            if FLIP_DEBUG_VERTICALLY: frame_to_display_debug = cv2.flip(frame_to_display_debug, 0)

        cv2.imshow(PROJECTOR_WINDOW_NAME, frame_to_display_projector) # Monitor 2
        if SHOW_DEBUG_WINDOW and frame_to_display_debug is not None: cv2.imshow(DEBUG_WINDOW_NAME, frame_to_display_debug) # Monitor 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

finally:
    print("Releasing resources...", flush=True); cap.release(); hands.close(); cv2.destroyAllWindows()
    print("Windows closed.", flush=True); print("Script finished.", flush=True)