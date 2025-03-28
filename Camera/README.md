# Interactive Projection System with Hand Tracking

This project explores creating interactive applications by projecting visuals onto a surface and using a camera to track hand movements, allowing users to interact directly with the projected elements. It utilizes Python with standard computer vision and media libraries.

**Demo Video:** (Optional: Add a GIF or link to a video of `script25.py` in action here!)

## Core Technologies

*   **Python 3:** The main programming language.
*   **OpenCV (`opencv-python`):** For camera access, image processing, undistortion, and homography calculations.
*   **MediaPipe (`mediapipe`):** For robust real-time hand landmark detection.
*   **NumPy (`numpy`):** For numerical operations, especially with coordinates and matrices.
*   **Tkinter (`tkinter`):** Built-in library used to get screen dimensions for multi-monitor setup.
*   **Pygame (`pygame`):** Used for easy sound playback in the demo script.

## Hardware Requirements

*   **Camera:** A standard webcam (Logitech C270 was used during development). Must be accessible by OpenCV.
*   **Projector:** Any standard projector connectable to the computer as a second display.
*   **Computer:** A reasonably modern computer capable of running Python and the required libraries in real-time.
*   **Projection Surface:** A flat surface (table, floor, wall) where the projector displays and interaction occurs.
*   **(Optional) 3D Printer:** Useful for printing calibration patterns (checkerboard) and potentially custom mounts for camera/projector stability.

## Core Concepts Implemented

*   **Camera Calibration:** Correcting lens distortion using a checkerboard pattern (`camera_calibration_data.npz`).
*   **Hand Tracking:** Detecting hand landmarks (fingertips, joints) using MediaPipe.
*   **Coordinate Undistortion:** Applying camera calibration data to incoming frames.
*   **Homography:** Calculating and applying a perspective transformation matrix (`homography_matrix*.npz`) to map points from the camera's view space to the projector's coordinate space, aligning the interaction.
*   **Multi-Monitor Display:** Displaying debug info on the primary monitor and the interactive projection fullscreen on the secondary monitor (projector).
*   **State Management:** Simple system to switch between different application modes (Home, Draw, Sound).
*   **Dwell Time Interaction:** Triggering actions by holding the tracked hand position (cursor) over a target area for a set duration.

## Setup & Installation

1.  **Clone Repository:** `git clone <your-repo-url>`
2.  **Install Libraries:**
    ```bash
    pip install opencv-python mediapipe numpy pygame
    ```
    *(Tkinter is usually included with Python)*
3.  **Physical Setup:**
    *   Connect the projector as a **second monitor** (extended display mode) to your computer. Note its position relative to the main monitor (e.g., right, left, above). The scripts often assume it's to the right - adjust `monitor_x_offset` in the code if needed.
    *   Mount the camera so it has a clear view of the projection surface.
    *   **IMPORTANT:** Ensure the camera and projector are **mounted stably** relative to each other. Any wobble or shift after calibration will ruin the alignment. A custom bracket is recommended over temporary solutions like tape.
4.  **Sound Files:** For `script25.py`, create a folder named `sounds` in the project directory and place required audio files (e.g., `.wav`, `.ogg`) inside it. Update the `sound_regions` list in the script if using different filenames.

## Calibration Procedure (Essential!)

You **MUST** run these calibration steps in order after setting up the hardware:

1.  **Camera Calibration (Checkerboard):**
    *   Print a checkerboard pattern (e.g., 9x7 squares, meaning 8x6 internal corners).
    *   **Accurately measure** the side length of one square (e.g., in mm).
    *   Modify and run the initial camera calibration script (e.g., `1_calibrate_camera.py` - based on the first calibration script we created). Update checkerboard dimensions (`CHECKERBOARD_DIMS`) and square size (`SQUARE_SIZE_MM`).
    *   Follow the script's instructions, showing the checkerboard at various angles, distances, and positions to the camera. Ensure good, even lighting and **manual focus**.
    *   This generates the `camera_calibration_data.npz` file. A low reprojection error (e.g., < 0.5) indicates a good calibration. Re-run if needed.
2.  **Homography Calibration (Projected Markers):**
    *   **Ensure Consistent Settings:** Make sure the `FLIP_CAMERA_INPUT_*` flags and `UNDISTORT_ALPHA` value in `calibrate_homography_grid.py` (or the specific calibration script used) **MATCH** the settings in the final application script (e.g., `script25.py`).
    *   Run the corrected `calibrate_homography_grid.py` (or similar, the one using crosses/grid).
    *   A window appears on the projector showing white crosses (+).
    *   A window appears on your main monitor showing the camera feed.
    *   **Accurately click the exact center (intersection) of each projected cross** as seen in the camera window. Precision is critical.
    *   This generates the `homography_matrix_cross_grid.npz` file (or similar, depending on the filename used).

## Running the Demo

After performing both calibrations successfully and ensuring the `.npz` files are present:

1.  Run the main application script:
    ```bash
    python script25.py
    ```
2.  The projector should display the interactive interface (starting with the Home screen).
3.  The main monitor should display the debug camera view (if `SHOW_DEBUG_WINDOW = True`).
4.  Use your hand (index fingertip is tracked by default) to interact:
    *   Move the projected red cursor over buttons/areas.
    *   Hold the cursor still over a target for ~1 second (dwell time) to "click" it.

## Script Descriptions (Development Journey)

This project evolved through several scripts. Here's a summary:

*   **`1_calibrate_camera.py` (Placeholder Name):** Initial script using OpenCV to detect a checkerboard pattern and calculate the camera's intrinsic matrix and distortion coefficients. Saves `camera_calibration_data.npz`. *Crucial first step.*
*   **`2_test_undistort.py` (Placeholder Name):** Loads `camera_calibration_data.npz` and displays the original vs. undistorted camera feed side-by-side to verify the camera calibration.
*   **`3_track_hands.py` (Placeholder Name):** Integrated MediaPipe Hands to detect and draw hand landmarks onto the undistorted camera feed.
*   **`4_projector_fullscreen.py` (Placeholder Name):** Configured a fullscreen window on the second monitor (projector) and started extracting basic hand coordinates (e.g., index fingertip). Displayed the camera feed with overlays on the projector (leading to the infinity mirror effect).
*   **`5_separate_canvas.py` (Placeholder Name):** Introduced the concept of a separate black `projection_canvas`. Detected hands from the camera feed but drew only minimal elements (like a red dot at the detected coordinates) onto the canvas, which was then displayed on the projector. This broke the infinity mirror loop. Included a separate debug window on the main monitor.
*   **`6_resizable_debug.py` (Placeholder Name):** Minor update to allow resizing the debug window on the main monitor.
*   **`7_revert_direct_projection.py` (Placeholder Name):** Temporarily reverted to projecting the full camera feed (with debug overlays) onto the projector to help visualize the raw misalignment during physical projector adjustment attempts. Reintroduced the infinity mirror effect for testing purposes.
*   **`script15.py`:** Added a `FLIP_DEBUG_VERTICALLY` flag to independently flip the debug view on Monitor 1 if it appeared upside down. Kept separate projection canvas.
*   **`script16.py`:** Introduced `FLIP_CAMERA_INPUT_VERTICALLY` flag to flip the raw camera input feed *before* any processing, aiming to fix orientation issues at the source. Output flips reset initially.
*   **`script17.py`:** Incorrectly attempted to fix a perceived left-right mirroring by adding `FLIP_PROJECTOR_OUTPUT_HORIZONTALLY`. Also added handedness detection display. Revealed the issue was likely mirrored *input* if handedness was wrong.
*   **`script18.py`:** Introduced `FLIP_CAMERA_INPUT_HORIZONTALLY` to flip the camera input horizontally, intended to correct wrong handedness detection by MediaPipe. Reset output horizontal flip.
*   **`script19.py`:** Reverted input horizontal flip after confirming handedness detection was actually correct. Focused back on geometric misalignment as the primary issue. Solidified the structure using the separate `projection_canvas`.
*   **`calibrate_homography.py`:** Introduced the software calibration method. This initial version projected dots and had the user click them in the camera view to collect point pairs for `cv2.findHomography`. Saved `homography_matrix.npz`. Included logic for loading camera calibration and undistorting the view used for clicking. *Corrected versions later included input flip flags.*
*   **`script20.py`:** First main application script to load the `homography_matrix.npz` and use `cv2.perspectiveTransform` to calculate the correct projector coordinates `(proj_x, proj_y)` from the camera coordinates `(cx, cy)`. Projected only the transformed red dot.
*   **`script21.py`:** (Two versions)
    *   Initially modified `script20` to test `UNDISTORT_ALPHA = 1.0` to try and fix edge warping/instability. **Required re-running `calibrate_homography.py`**.
    *   Also repurposed as a **Manual Alignment Helper**, projecting a single static dot and showing its detected position in the debug view to aid physical projector adjustment.
*   **`calibrate_homography_grid.py` (Placeholder Name, based on code):** Modified the homography calibration script to project a grid (e.g., 4x6) of points instead of just 5, aiming for a more accurate matrix calculation across the whole surface. *Corrected versions later included input flip flags and cross markers.*
*   **`script22.py`:** Main application script structure, updated to load the homography matrix generated by the grid calibration (initially still projecting just the red dot).
*   **`script23.py`:** Tested using the **inverse** of the loaded homography matrix (`np.linalg.inv(H)`) for the perspective transform to diagnose the "opposite movement" issue.
*   **`script24.py`:** Corrected main application script using the **original** homography matrix (from the grid calibration) after determining the input vertical flip setting was likely incorrect during previous calibrations. This version correctly mapped movement directions.
*   **`script25.py`:** The current **Demo Application**. Builds on `script24`. Loads the latest camera and homography calibrations. Implements a state machine for a Home screen, an Air Draw app, and a Sound Trigger app using dwell time for interaction. Uses the transformed coordinates for accurate cursor placement.

## Future Ideas

*   Implement more sophisticated gestures using MediaPipe's capabilities (e.g., pinch, fist, specific finger counts) for different actions.
*   Develop more mini-apps/games (e.g., simple target practice, virtual piano, physics sandbox).
*   Integrate tracking of 3D printed objects with ArUco markers alongside hand tracking.
*   Improve drawing app (colors, brush sizes, saving).
*   Explore non-linear mapping techniques if homography isn't accurate enough in specific setups (more complex).
*   Refine UI elements and visual feedback.

## Troubleshooting

*   **Bad Alignment/Warping:** Most likely caused by physical instability (camera/projector movement after calibration) or inaccurate clicks during homography calibration. **Solution:** Stabilize the mount, minimize projector keystone, re-run `calibrate_homography.py` with extreme precision. Also ensure `UNDISTORT_ALPHA` matches between calibration and main scripts. Re-run `1_calibrate_camera.py` if the base undistortion looks bad.
*   **Incorrect Orientation (Upside-down/Mirrored):**
    1.  Check Projector Menu settings (Projection Mode, Flip, Mirror).
    2.  Check Windows Display settings (Orientation).
    3.  Check Camera software/driver settings (Mirror video).
    4.  Adjust `FLIP_CAMERA_INPUT_*` flags in the Python scripts (recalibrating homography if input flips change).
    5.  Use `FLIP_DEBUG_VERTICALLY` / `FLIP_OUTPUT_VERTICALLY` flags only as a last resort for display issues.
*   **Hand Not Detected:** Check lighting. Increase distance between hand and camera slightly. Lower `DETECTION_CONFIDENCE` (but might increase false positives).
*   **Dot Not Detected (Alignment/Calibration Helper):** Adjust `THRESHOLD_VALUE` in `script21.py` or `calibrate_homography.py`. Ensure good contrast between projected dot/cross and surface. Check lighting.