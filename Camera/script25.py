# script25.py
# Interactive Demo with Home Screen, Air Draw, and Sound Triggers
# Uses Homography for projection mapping.

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import mediapipe as mp
import time
import tkinter as tk
import pygame # For sound playback

# --- Constants & State ---
STATE_HOME = 0
STATE_DRAW = 1
STATE_SOUND = 2
current_state = STATE_HOME
DEBUG_DOT_RADIUS = 10

# --- Configuration ---
CAMERA_CALIBRATION_FILE = "camera_calibration_data.npz"
HOMOGRAPHY_FILE = "homography_matrix_cross_grid.npz" # <<< USE THE LATEST GRID/CROSS FILE
FRAME_WIDTH = 1280; FRAME_HEIGHT = 720
MAX_HANDS = 1 # <<< Start with 1 hand for simplicity, can increase later
DETECTION_CONFIDENCE = 0.7 # Slightly higher confidence might help stability
TRACKING_CONFIDENCE = 0.7
PROJECTOR_WINDOW_NAME = "Interactive Projection"; DEBUG_WINDOW_NAME = "Camera Debug View"

# --- Flip Settings (Set based on previous successful tests) ---
FLIP_CAMERA_INPUT_VERTICALLY = False
FLIP_CAMERA_INPUT_HORIZONTALLY = False
FLIP_DEBUG_VERTICALLY = False
FLIP_OUTPUT_VERTICALLY = False

UNDISTORT_ALPHA = 1.0 # Keep consistent with homography calibration
SHOW_DEBUG_WINDOW = True
DEBUG_WINDOW_WIDTH = 960; DEBUG_WINDOW_HEIGHT = 540

# --- Interaction Settings ---
DWELL_TIME_SEC = 1.0 # Seconds to hold over button to click
CURSOR_RADIUS = 15 # Size of the main interaction dot
CURSOR_COLOR = (0, 0, 255) # Red
HOVER_COLOR = (0, 255, 255) # Yellow hover
CLICK_COLOR = (0, 255, 0) # Green click feedback

# --- App Specific Settings ---
# Draw App
DRAW_COLOR = (255, 255, 255) # White lines
DRAW_THICKNESS = 5
draw_points = [] # List to store points for current drawing
is_drawing = False # Track if drawing is active (e.g., based on continuous presence)

# Sound App
SOUND_FOLDER = "sounds"
# Define sound trigger regions [ (x_min, y_min, x_max, y_max), sound_file ]
sound_regions = [
    ((100, 100, 400, 300), os.path.join(SOUND_FOLDER, "sound1.wav")),
    ((FRAME_WIDTH - 400, 100, FRAME_WIDTH - 100, 300), os.path.join(SOUND_FOLDER, "sound2.wav")),
    ((FRAME_WIDTH // 2 - 150, FRAME_HEIGHT - 300, FRAME_WIDTH // 2 + 150, FRAME_HEIGHT - 100), os.path.join(SOUND_FOLDER, "sound3.ogg"))
]
# --- End Configuration ---


# --- Helper Functions ---
def is_point_in_rect(point, rect):
    """Checks if point (x, y) is inside rect (x_min, y_min, x_max, y_max)."""
    if point is None or rect is None:
        return False
    x, y = point
    x_min, y_min, x_max, y_max = rect
    return x_min <= x < x_max and y_min <= y < y_max

# --- Initialization ---
# Pygame Mixer
try:
    pygame.mixer.init()
    print("Pygame mixer initialized.", flush=True)
    # Load sounds - Store as (pygame_sound_object, filename)
    loaded_sounds = []
    for rect, filename in sound_regions:
        try:
            sound = pygame.mixer.Sound(filename)
            loaded_sounds.append(sound)
            print(f"Loaded sound: {filename}", flush=True)
        except pygame.error as e:
            print(f"Warning: Could not load sound '{filename}': {e}", flush=True)
            loaded_sounds.append(None) # Add placeholder if sound fails
except Exception as e:
    print(f"Warning: Could not initialize pygame mixer: {e}. Sound app will not work.", flush=True)
    loaded_sounds = [None] * len(sound_regions)

# Load Camera Calibration
try:
    with np.load(CAMERA_CALIBRATION_FILE) as data:
        camera_matrix=data['camera_matrix']; dist_coeffs=data['dist_coeffs']
        img_size_calib=data['img_size']; print("Calibration data loaded.", flush=True)
except Exception as e: print(f"Error loading camera calib: {e}", flush=True); exit()

# Load Homography Matrix
try:
    with np.load(HOMOGRAPHY_FILE) as data: H_matrix_to_use = data['homography']
    if H_matrix_to_use is None or H_matrix_to_use.shape != (3, 3): raise ValueError("Invalid Matrix")
    print(f"Homography loaded from '{HOMOGRAPHY_FILE}'.", flush=True)
except Exception as e: print(f"ERROR loading '{HOMOGRAPHY_FILE}': {e}", flush=True); exit()

# Initialize MediaPipe Hands
try:
    mp_hands=mp.solutions.hands; hands=mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_HANDS, min_detection_confidence=DETECTION_CONFIDENCE, min_tracking_confidence=TRACKING_CONFIDENCE)
    mp_drawing=mp.solutions.drawing_utils; mp_drawing_styles=mp.solutions.drawing_styles
    print("MediaPipe Hands initialized.", flush=True)
except Exception as e: print(f"Error initializing MediaPipe: {e}", flush=True); exit()

# Projector & Debug Window Setup
# ... (Same setup code as script24) ...
print("Setting up windows...", flush=True)
try: # Projector
    root=tk.Tk(); root.withdraw(); screen_width=root.winfo_screenwidth(); root.destroy()
    monitor_x_offset=screen_width; monitor_y_offset=0
    cv2.namedWindow(PROJECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL); cv2.moveWindow(PROJECTOR_WINDOW_NAME, monitor_x_offset, monitor_y_offset)
    cv2.setWindowProperty(PROJECTOR_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
except Exception as e: print(f"Error setting up projector: {e}. Normal window.", flush=True); cv2.namedWindow(PROJECTOR_WINDOW_NAME, cv2.WINDOW_NORMAL)
if SHOW_DEBUG_WINDOW: # Debug
    cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL); cv2.moveWindow(DEBUG_WINDOW_NAME, 50, 50)
    try:
        debug_w=DEBUG_WINDOW_WIDTH if DEBUG_WINDOW_WIDTH>0 else FRAME_WIDTH; debug_h=DEBUG_WINDOW_HEIGHT if DEBUG_WINDOW_HEIGHT>0 else FRAME_HEIGHT
        cv2.resizeWindow(DEBUG_WINDOW_NAME, debug_w, debug_h)
    except Exception as e: print(f"Could not resize debug window: {e}", flush=True)
print("Windows configured.", flush=True)

# Initialize Camera
# ... (Same setup code as script24) ...
print("Starting camera...", flush=True); cap=cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: No video device.", flush=True); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
actual_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); actual_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}", flush=True)

# Calculate Undistortion Maps
print(f"Calculating undistortion map with alpha={UNDISTORT_ALPHA}...", flush=True)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (actual_width, actual_height), alpha=UNDISTORT_ALPHA, newImgSize=(actual_width, actual_height))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (actual_width, actual_height), 5)
print("Undistortion maps created.", flush=True)

# --- Button Definitions (Rectangles: x_min, y_min, x_max, y_max) ---
# Adjust positions and sizes as needed
BTN_WIDTH = 300; BTN_HEIGHT = 150; BTN_MARGIN = 50
BTN_Y_POS = FRAME_HEIGHT // 2 - BTN_HEIGHT // 2

home_buttons = {
    "draw": (BTN_MARGIN, BTN_Y_POS, BTN_MARGIN + BTN_WIDTH, BTN_Y_POS + BTN_HEIGHT),
    "sound": (FRAME_WIDTH - BTN_MARGIN - BTN_WIDTH, BTN_Y_POS, FRAME_WIDTH - BTN_MARGIN, BTN_Y_POS + BTN_HEIGHT)
}

# Generic Back Button (can be reused in apps)
BACK_BTN_RECT = (10, FRAME_HEIGHT - 60, 110, FRAME_HEIGHT - 10) # Bottom Left

# Draw App Buttons
DRAW_CLEAR_BTN_RECT = (FRAME_WIDTH - 120, 10, FRAME_WIDTH - 10, 60) # Top Right

# --- Dwell Time Tracking ---
hover_target = None # Which button/region is currently hovered
hover_start_time = 0

print("\nStarting main application loop. Press 'q' to quit.", flush=True)
prev_time = 0
cursor_pos = None # Store the main projected cursor position (proj_x, proj_y)

try:
    while True:
        ret, frame = cap.read()
        if not ret: print("Error: Can't receive frame.", flush=True); break

        # --- Input Processing ---
        if FLIP_CAMERA_INPUT_VERTICALLY: frame = cv2.flip(frame, 0)
        if FLIP_CAMERA_INPUT_HORIZONTALLY: frame = cv2.flip(frame, 1)
        undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR); h, w, _ = undistorted_frame.shape
        rgb_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB); rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame); rgb_frame.flags.writeable = True

        # --- State Variables for this Frame ---
        projection_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        debug_frame = undistorted_frame.copy() if SHOW_DEBUG_WINDOW else None
        cursor_pos = None # Reset cursor position each frame
        hand_detected = False
        currently_hovering = None # What is being hovered over THIS frame

        # --- Hand Detection and Transformation ---
        if results.multi_hand_landmarks:
            # Use the first detected hand (index 0)
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_detected = True

            # Get camera coords (cx, cy) for index finger tip
            target_landmark = mp_hands.HandLandmark.INDEX_FINGER_TIP
            landmark_coords = hand_landmarks.landmark[target_landmark]
            cx = int(landmark_coords.x * w); cy = int(landmark_coords.y * h)

            # Transform to projector coords (proj_x, proj_y)
            if 0 <= cx < w and 0 <= cy < h:
                cam_pt = np.array([[[cx, cy]]], dtype=np.float32)
                proj_pt_transformed = cv2.perspectiveTransform(cam_pt, H_matrix_to_use)
                if proj_pt_transformed is not None:
                    proj_x = int(proj_pt_transformed[0,0,0]); proj_y = int(proj_pt_transformed[0,0,1])
                    # Set cursor_pos IF it's within bounds
                    if 0 <= proj_x < w and 0 <= proj_y < h:
                        cursor_pos = (proj_x, proj_y)

            # Draw debug view (green dot at cx, cy)
            if SHOW_DEBUG_WINDOW and debug_frame is not None:
                 if 0 <= cx < w and 0 <= cy < h:
                    handedness_label = results.multi_handedness[0].classification[0].label if results.multi_handedness else "N/A"
                    cv2.circle(debug_frame, (cx, cy), DEBUG_DOT_RADIUS, (0,255,0), cv2.FILLED)
                    cv2.putText(debug_frame, f"{handedness_label}:({cx},{cy})", (cx+15, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


        # --- Update State & Handle Interactions ---
        clicked_target = None # Track if a click happened this frame

        # Check for hovers only if a hand/cursor is detected
        if cursor_pos:
            if current_state == STATE_HOME:
                for name, rect in home_buttons.items():
                    if is_point_in_rect(cursor_pos, rect):
                        currently_hovering = ("home", name)
                        break # Only hover one button at a time
            elif current_state == STATE_DRAW:
                if is_point_in_rect(cursor_pos, BACK_BTN_RECT):
                    currently_hovering = ("draw", "back")
                elif is_point_in_rect(cursor_pos, DRAW_CLEAR_BTN_RECT):
                    currently_hovering = ("draw", "clear")
                else:
                    # If not hovering UI, consider it drawing
                    is_drawing = True
                    draw_points.append(cursor_pos) # Add point to current line
            elif current_state == STATE_SOUND:
                sound_hover_found = False
                for i, (rect, filename) in enumerate(sound_regions):
                    if is_point_in_rect(cursor_pos, rect):
                        currently_hovering = ("sound_region", i)
                        sound_hover_found = True
                        # Play sound on ENTER (can change to on click later)
                        if loaded_sounds[i] is not None:
                             if not pygame.mixer.Channel(i).get_busy(): # Play only if not already playing on this channel
                                pygame.mixer.Channel(i).play(loaded_sounds[i])
                        break # Hover only one region
                if not sound_hover_found and is_point_in_rect(cursor_pos, BACK_BTN_RECT):
                     currently_hovering = ("sound", "back")

        # Reset drawing state if hand is lost in Draw mode
        if current_state == STATE_DRAW and not cursor_pos:
            is_drawing = False
            if len(draw_points) > 1: draw_points.append(None) # Add None to break line segment

        # Dwell Time Click Logic
        if currently_hovering:
            if hover_target == currently_hovering: # Still hovering the same target
                if time.time() - hover_start_time >= DWELL_TIME_SEC:
                    clicked_target = hover_target # CLICK!
                    hover_target = None # Reset hover after click
                    hover_start_time = 0 # Reset timer
            else: # Hovering a new target
                hover_target = currently_hovering
                hover_start_time = time.time()
        else: # Not hovering anything
            hover_target = None
            hover_start_time = 0

        # --- State Transitions Based on Clicks ---
        if clicked_target:
            click_state, click_name = clicked_target
            print(f"Clicked: {click_name}", flush=True)
            if click_state == "home":
                if click_name == "draw": current_state = STATE_DRAW; draw_points = [] # Clear drawing on enter
                elif click_name == "sound": current_state = STATE_SOUND
            elif click_state == "draw":
                if click_name == "back": current_state = STATE_HOME
                elif click_name == "clear": draw_points = [] # Clear drawing
            elif click_state == "sound":
                if click_name == "back": current_state = STATE_HOME
            # Add more transitions as needed
            clicked_target = None # Consume click


        # --- Drawing Logic (Projector Canvas) ---

        # Draw based on current state
        if current_state == STATE_HOME:
            # Draw Home Screen Buttons
            for name, rect in home_buttons.items():
                color = HOVER_COLOR if hover_target == ("home", name) else (200, 200, 200) # Gray default
                cv2.rectangle(projection_canvas, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
                text = name.upper()
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(projection_canvas, text, (rect[0] + (BTN_WIDTH - text_w)//2, rect[1] + (BTN_HEIGHT + text_h)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        elif current_state == STATE_DRAW:
            # Draw lines
            if len(draw_points) > 1:
                for i in range(1, len(draw_points)):
                    if draw_points[i-1] is not None and draw_points[i] is not None: # Check for breaks
                        cv2.line(projection_canvas, draw_points[i-1], draw_points[i], DRAW_COLOR, DRAW_THICKNESS)

            # Draw Back Button
            back_color = HOVER_COLOR if hover_target == ("draw", "back") else (200, 0, 0)
            cv2.rectangle(projection_canvas, (BACK_BTN_RECT[0], BACK_BTN_RECT[1]), (BACK_BTN_RECT[2], BACK_BTN_RECT[3]), back_color, -1)
            cv2.putText(projection_canvas, "BACK", (BACK_BTN_RECT[0]+15, BACK_BTN_RECT[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Draw Clear Button
            clear_color = HOVER_COLOR if hover_target == ("draw", "clear") else (0, 200, 0)
            cv2.rectangle(projection_canvas, (DRAW_CLEAR_BTN_RECT[0], DRAW_CLEAR_BTN_RECT[1]), (DRAW_CLEAR_BTN_RECT[2], DRAW_CLEAR_BTN_RECT[3]), clear_color, -1)
            cv2.putText(projection_canvas, "CLEAR", (DRAW_CLEAR_BTN_RECT[0]+15, DRAW_CLEAR_BTN_RECT[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        elif current_state == STATE_SOUND:
            # Draw Sound Regions
            for i, (rect, filename) in enumerate(sound_regions):
                 color = HOVER_COLOR if hover_target == ("sound_region", i) else (0, 0, 150) # Dark Red default
                 cv2.rectangle(projection_canvas, (rect[0], rect[1]), (rect[2], rect[3]), color, -1 if hover_target == ("sound_region", i) else 2) # Fill on hover
                 cv2.putText(projection_canvas, f"Sound {i+1}", (rect[0]+10, rect[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # Draw Back Button
            back_color = HOVER_COLOR if hover_target == ("sound", "back") else (200, 0, 0)
            cv2.rectangle(projection_canvas, (BACK_BTN_RECT[0], BACK_BTN_RECT[1]), (BACK_BTN_RECT[2], BACK_BTN_RECT[3]), back_color, -1)
            cv2.putText(projection_canvas, "BACK", (BACK_BTN_RECT[0]+15, BACK_BTN_RECT[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


        # Draw Cursor and Dwell Feedback
        if cursor_pos:
            final_cursor_color = CURSOR_COLOR
            if hover_target:
                final_cursor_color = HOVER_COLOR
                # Draw dwell progress bar (optional)
                dwell_progress = (time.time() - hover_start_time) / DWELL_TIME_SEC
                dwell_progress = min(max(dwell_progress, 0), 1) # Clamp between 0 and 1
                if dwell_progress > 0.1: # Only show if dwelling started noticeably
                    prog_w = int(dwell_progress * 40) # Max width 40 pixels
                    cv2.rectangle(projection_canvas, (cursor_pos[0] - 20, cursor_pos[1] + CURSOR_RADIUS + 5),
                                  (cursor_pos[0] - 20 + prog_w, cursor_pos[1] + CURSOR_RADIUS + 10),
                                  CLICK_COLOR, -1)

            cv2.circle(projection_canvas, cursor_pos, CURSOR_RADIUS, final_cursor_color, 2) # Draw outline


        # --- Display Outputs ---
        curr_time=time.time(); fps=1/(curr_time-prev_time) if (curr_time-prev_time)>0 else 0; prev_time=curr_time; fps_text=f"FPS: {int(fps)}"
        cv2.putText(projection_canvas, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # White FPS on projector
        if SHOW_DEBUG_WINDOW and debug_frame is not None: cv2.putText(debug_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green FPS on debug

        frame_to_display_projector = projection_canvas
        if FLIP_OUTPUT_VERTICALLY: frame_to_display_projector = cv2.flip(frame_to_display_projector, 0)

        frame_to_display_debug = debug_frame
        if SHOW_DEBUG_WINDOW and frame_to_display_debug is not None:
            if FLIP_DEBUG_VERTICALLY: frame_to_display_debug = cv2.flip(frame_to_display_debug, 0)

        cv2.imshow(PROJECTOR_WINDOW_NAME, frame_to_display_projector) # Monitor 2
        if SHOW_DEBUG_WINDOW and frame_to_display_debug is not None: cv2.imshow(DEBUG_WINDOW_NAME, frame_to_display_debug) # Monitor 1

        # Check for Quit Key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

finally:
    print("Releasing resources...", flush=True); cap.release(); hands.close(); cv2.destroyAllWindows()
    pygame.mixer.quit() # Quit pygame mixer
    print("Windows closed. Pygame mixer quit.", flush=True); print("Script finished.", flush=True)