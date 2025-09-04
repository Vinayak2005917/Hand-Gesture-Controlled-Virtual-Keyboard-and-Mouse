import cv2
import mediapipe as mp
import streamlit as st
import time
import pyautogui
from utils import detect_gesture, mouse_controller, keyboard_disk

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit setup
st.title("🖐 Hand Tracking with Gesture Modes (Streamlit + Mediapipe)")
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
        cv2.putText(img, "Palm = Mouse, Fist = Keyboard", (120, 55), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 50, 50), 2)

        palm_detected = detect_gesture(img, results, gesture_data_file="Gesture data/palm_data.txt", tolerance=tolerance, similarity_threshold=similarity_threshold)
        if palm_detected:
            mode = "Mouse"

        fist_detected = detect_gesture(img, results, gesture_data_file="Gesture data/fist_data.txt",
                                       tolerance=tolerance, similarity_threshold=similarity_threshold)
        if fist_detected:
            mode = "Keyboard"
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

    # --- Keyboard Mode ---
    if mode == "Keyboard" and wait <= 0:
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            h, w, c = img.shape
            landmarks = {id: (int(lm.x * w), int(lm.y * h))
                         for id, lm in enumerate(handLms.landmark)}
            img, state = keyboard_disk(img, landmarks, state, w // 2, h // 2, radius=100)

    elif mode == "Keyboard" and wait > 0:
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