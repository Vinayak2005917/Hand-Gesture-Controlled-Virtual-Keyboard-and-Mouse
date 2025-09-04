import cv2
import mediapipe as mp
import streamlit as st
import time
import pyautogui
from utils import detect_gesture, mouse_controller, keyboard_disk, keyboard_qwerty

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit setup
st.title("Hand Tracking for Keyboard and Mouse Control")
st.write("This Product is still under development so stability is not guaranteed.")
stframe = st.empty()

# Camera
cap = cv2.VideoCapture(0)
cam_w, cam_h = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

# Screen info
screen_w, screen_h = pyautogui.size()

# Persistent state
state = {}
wait = 50
mode = "None"

# Timing
pTime = 0
tipIds = [4, 8, 12, 16, 20]
tolerance = 0.075
similarity_threshold = 0.75
    

st.sidebar.write("This is a real-time hand gesture controlled keyboard and mouse application.")
st.sidebar.markdown("""## Modes and controls

Mode selection (shown on start or after reset):
- Palm → Mouse mode
- Fist → Rotating Keyboard mode
- OK sign → QWERTY Keyboard mode
- Peace sign → Reset to None

Mouse mode (thumb vs. other fingertips):
- Thumb + Index pinch → Grab/drag cursor; move hand to move cursor
- Thumb + Middle pinch → Left click
- Thumb + Ring pinch → Right click
- Thumb + Pinky tip pinch → Scroll up
- Thumb + Pinky middle joint pinch → Scroll down

Rotating keyboard mode (circular letters):
- Thumb + Index pinch → Rotate counterclockwise
- Thumb + Pinky pinch → Rotate clockwise
- Thumb + Middle pinch → Select current highlighted (leftmost) letter and type it

QWERTY keyboard mode (on-screen keys):
- Hover the index fingertip over a key; push towards the camera (Z below threshold) and hold briefly to click
- Special keys: Backspace, Space, Enter
- Toggle case with the lower/UPPER key
- Typed text is shown on screen and sent to the active OS window via PyAutoGUI
    """)


user_input = st.text_input("Input Field to test keyboard", "", key="test_input")


while True:
    ret, img = cap.read()
    if not ret:
        st.write("⚠️ No camera feed detected")
        break
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Mode selection
    if mode == "None":
        cv2.putText(img, "Select mode:", (240, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 50, 50), 2)
        cv2.putText(img, "Palm = Mouse, Fist = Keyboard, Ok = rotating keyboard", (120, 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 50, 50), 2)

        palm_detected = detect_gesture(img, results, gesture_data_file="Gesture data/palm_data.txt", tolerance=tolerance, similarity_threshold=similarity_threshold)
        if palm_detected:
            mode = "Mouse"

        ok_sign_detected = detect_gesture(img, results, gesture_data_file="Gesture data/ok_symbol_data.txt",
                                           tolerance=tolerance, similarity_threshold=similarity_threshold)
        if ok_sign_detected:
            mode = "Rotating Keyboard"

        fist_detected = detect_gesture(img, results, gesture_data_file="Gesture data/fist_data.txt",
                                       tolerance=tolerance, similarity_threshold=similarity_threshold)
        if fist_detected:
            mode = "QWERTY Keyboard"
            cv2.putText(img, "Keyboard mode activated", (240, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 50, 50), 2)

    # Show current mode
    cv2.putText(img, f"Mode: {mode}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Reset to None if peace sign
    peace_sign_detected = detect_gesture(img, results, gesture_data_file="Gesture data/peace_sign_data.txt",
                                         tolerance=0.05, similarity_threshold=0.85)
    if peace_sign_detected:
        mode = "None"

    # --- Mouse Mode ---
    if mode == "Mouse":
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
        img, state = mouse_controller(img, results, fg=False, state=state,
                                      screen_w=screen_w, screen_h=screen_h,
                                      cam_w=cam_w, cam_h=cam_h)
        
    # --- QWERTY Keyboard Mode ---
    if mode == "QWERTY Keyboard" and wait <= 0:
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                # Pass Mediapipe landmarks list (with x, y, z) to keyboard_qwerty
                img, state = keyboard_qwerty(img, handLms.landmark, state, cz_threshold=-0.1, hold_frames=12)


    # --- Rotating Keyboard Mode ---
    if mode == "Rotating Keyboard" and wait <= 0:
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, c = img.shape
            landmarks = {id: (int(lm.x * w), int(lm.y * h))
                         for id, lm in enumerate(handLms.landmark)}
            img, state = keyboard_disk(img, landmarks, state, w // 2, h // 2, radius=100)

    elif mode in ("QWERTY Keyboard", "Rotating Keyboard") and wait > 0:
        cv2.putText(img, "Initializing Keyboard...", (10, 110),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
        wait -= 1

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show in Streamlit
    stframe.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()