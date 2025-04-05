import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture video from the webcam
cam = cv2.Videocapture(0)

prev_positions = []
MAX_FRAMES = 5
GESTURE_DELAY = 0.8
gesture_start_time = None
SCROLL_SPEED = 30
SCROLL_THRESHOLD = 0.6
scrolling = None

# Check if a finger is up based on landmark y positions
def fingers_up(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers = []

    # Thumb: compare tip and MCP in x direction (since it's horizontal)
    fingers.append(landmarks[4].x > landmarks[3].x)  # Adjust based on left/right hand

    # Index, Middle, Ring, Pinky: tip below PIP in y direction (for upright hand)
    fingers.append(landmarks[8].y < landmarks[6].y)   # Index
    fingers.append(landmarks[12].y < landmarks[10].y) # Middle
    fingers.append(landmarks[16].y < landmarks[14].y) # Ring
    fingers.append(landmarks[20].y < landmarks[18].y) # Pinky

    return fingers

# Detect Swipe Gesture only if open palm
def detect_swipe():
    global gesture_start_time

    if len(prev_positions) < MAX_FRAMES:
        return

    x_movement = prev_positions[-1][0] - prev_positions[0][0]
    y_movement = prev_positions[-1][1] - prev_positions[0][1]

    if abs(y_movement) > 0.15:
        return
    elif abs(x_movement) > 0.15:
        if x_movement < 0:
            switch_application("left")
        else:
            switch_application("right")
        gesture_start_time = time.time()

def switch_application(direction):
    if direction == "right":
        pyautogui.keyDown("alt")
        pyautogui.press("tab")
        pyautogui.keyUp("alt")
    elif direction == "left":
        pyautogui.keyDown("alt")
        pyautogui.keyDown("shift")
        pyautogui.press("tab")
        pyautogui.keyUp("shift")
        pyautogui.keyUp("alt")

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    height, width, _ = frame.shape
    section_width = width // 3
    line1_x = section_width
    line2_x = section_width * 2

    # Draw vertical section lines
    cv2.line(frame, (line1_x, 0), (line1_x, height), (0, 255, 0), 2)
    cv2.line(frame, (line2_x, 0), (line2_x, height), (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_x_norm = hand_landmarks.landmark[8].x
            index_y_norm = hand_landmarks.landmark[8].y
            index_x_px = int(index_x_norm * width)

            if index_x_px < line1_x or index_x_px > line2_x:
                # Only process gestures in left/right zones
                if len(prev_positions) >= MAX_FRAMES:
                    prev_positions.pop(0)
                prev_positions.append((index_x_norm, index_y_norm))

                # Scroll detection
                if index_y_norm < 0.4:
                    scrolling = "up"
                elif index_y_norm > SCROLL_THRESHOLD:
                    scrolling = "down"
                else:
                    scrolling = None

                # Check open palm
                finger_states = fingers_up(hand_landmarks)
                is_open_palm = all(finger_states)

                if is_open_palm and (gesture_start_time is None or time.time() - gesture_start_time > GESTURE_DELAY):
                    detect_swipe()
            else:
                scrolling = None

    else:
        scrolling = None

    if scrolling == "up":
        pyautogui.scroll(SCROLL_SPEED)
    elif scrolling == "down":
        pyautogui.scroll(-SCROLL_SPEED)

    cv2.imshow("Gesture Control Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
