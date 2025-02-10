import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam and face mesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen dimensions
screen_w, screen_h = pyautogui.size()

# Initialize previous positions
prev_x, prev_y = 0, 0

# Scrolling sensitivity factor
scroll_factor = 2  # Increase this value to make scrolling faster

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    output = face_mesh.process(rgb_frame)  # Process the frame to detect landmarks
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape  # Get the frame dimensions

    if landmark_points:
        landmarks = landmark_points[0].landmark
        # Use the landmark point between the eyes (id 1) to track head movement
        landmark = landmarks[1]

        # Get the position of the landmark
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)

        # Calculate movement difference for cursor control
        dx = x - prev_x
        dy = y - prev_y

        # Amplify the movement by a certain factor
        factor = 3  # Increase this value for more sensitivity
        screen_x = pyautogui.position().x + dx * factor
        screen_y = pyautogui.position().y + dy * factor

        # Update cursor position
        pyautogui.moveTo(screen_x, screen_y)

        # Update previous positions
        prev_x, prev_y = x, y

        # Draw the landmark on the frame for visualization
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            lx = int(landmark.x * frame_w)
            ly = int(landmark.y * frame_h)
            cv2.circle(frame, (lx, ly), 3, (0, 255, 255), -1)

        # Check for blink to perform a click
        if (left[0].y - left[1].y) < 0.01:
            pyautogui.click()
            pyautogui.sleep(1)

        # Use the nose tip (landmark 1) for scrolling
        nose_tip = landmarks[1]
        nose_y = int(nose_tip.y * frame_h)

        # Detect vertical head movement for scrolling
        if dy < -scroll_factor:
            pyautogui.scroll(100)  # Scroll up
        elif dy > scroll_factor:
            pyautogui.scroll(-100)  # Scroll down

    cv2.imshow('Eye Controlled Mouse with Scrolling', frame)
    cv2.waitKey(1)
