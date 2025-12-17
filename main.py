import cv2
import mediapipe as mp
import numpy as np
import sys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Try to find available cameras
print("Checking for available cameras...")
available_cameras = []
for i in range(5):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        print(f"Camera {i} found!")
        available_cameras.append(i)
        temp_cap.release()

if not available_cameras:
    print("No cameras found!")
    sys.exit(1)

# Select camera (default to 1, fallback to first available)
if len(sys.argv) > 1:
    camera_index = int(sys.argv[1])
else:
    camera_index = 1 if 1 in available_cameras else available_cameras[0]

print(f"\nUsing camera {camera_index}")
print(f"To use a different camera, run: python main.py <camera_index>")
print(f"Available cameras: {available_cameras}\n")

# Open webcam
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

# Check if webcam opened successfully
if not cap.isOpened():
    print(f"Error: Could not open webcam at index {camera_index}.")
    sys.exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Webcam opened successfully on camera {camera_index}")
print("Starting webcam...")
print("Controls:")
print("  'v' or SPACE - Toggle video mode (Normal/Blurred/Hidden)")
print("  'q' - Quit\n")

# Create window
cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)

# Display mode: 0 = normal, 1 = blurred, 2 = hidden (black background)
display_mode = 0
mode_names = ["Normal Video", "Blurred Video", "Pose Only"]

try:
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera.")
                continue

            # Convert the BGR frame (OpenCV) to RGB (MediaPipe)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Run pose estimation
            results = pose.process(image_rgb)

            # Prepare display frame based on mode
            if display_mode == 0:  # Normal video
                display_frame = frame.copy()
            elif display_mode == 1:  # Blurred video
                display_frame = cv2.GaussianBlur(frame, (51, 51), 0)
            else:  # Hidden - black background
                display_frame = np.zeros_like(frame)

            # Draw pose landmarks on the display frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # Add mode indicator text
            cv2.putText(display_frame, f"Mode: {mode_names[display_mode]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'V' to change mode | 'Q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show the frame
            cv2.imshow('MediaPipe Pose', display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v') or key == 32:  # 'v' or spacebar
                display_mode = (display_mode + 1) % 3
                print(f"Display mode: {mode_names[display_mode]}")

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")
# Note: This code captures video from the webcam, performs pose estimation using MediaPipe,
# and displays the results in real-time without saving any output files.