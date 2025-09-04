import cv2 as cv
import mediapipe as mp
import streamlit as st
import numpy as np
import time
import pyautogui
import sys
import math

def hand_bounding_box(landmarks, img , color = (0, 255, 0)):
    h, w, _ = img.shape

    # Convert normalized landmark coords (0-1) → pixel coords
    xs = [int(lm.x * w) for lm in landmarks.landmark]
    ys = [int(lm.y * h) for lm in landmarks.landmark]

    # Find extremes
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Draw rectangle
    cv.rectangle(img, (min_x, min_y), (max_x, max_y), color, 2)

    return (min_x, min_y, max_x, max_y)  # return box if needed

def detect_gesture(img, results, gesture_data_file="palm_data.txt", make_box=False, tolerance=0.02, similarity_threshold=0.8):
    """
    Detects palm gesture using hand landmarks and reference palm data.

    Args:
        img: The current video frame.
        results: Mediapipe hands result object (results.multi_hand_landmarks).
        palm_data_file: Path to reference palm data file.
        tolerance: Allowed difference in distances.
        similarity_threshold: Fraction of matching distances required.

    Returns:
        img: Frame with detection text if palm detected.
        bool: True if palm detected, False otherwise.
    """
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands

    gesture_detected = False

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get landmarks (normalized x,y in range [0,1])
            landmarks = [(lm.x, lm.y) for lm in handLms.landmark]

            # Compute all pairwise distances
            hand_landmark_data = []
            for i in range(len(landmarks)):
                for j in range(i + 1, len(landmarks)):
                    dist = ((landmarks[i][0] - landmarks[j][0]) ** 2 +
                            (landmarks[i][1] - landmarks[j][1]) ** 2) ** 0.5
                    hand_landmark_data.append((i, j, dist))

            # Load reference palm data
            with open(gesture_data_file, "r") as f:
                palm_data = [tuple(map(float, line.strip().split(","))) for line in f]

            # Compare with tolerance
            match_count = 0
            total_count = len(palm_data)

            for ref in palm_data:
                ref_i, ref_j, ref_dist = int(ref[0]), int(ref[1]), ref[2]

                for live in hand_landmark_data:
                    live_i, live_j, live_dist = live
                    if live_i == ref_i and live_j == ref_j:
                        if abs(live_dist - ref_dist) <= tolerance:
                            match_count += 1
                        break

            # Palm detected if enough distances match
            if match_count / total_count > similarity_threshold:
                gesture_detected = True

    return gesture_detected

def mouse_controller(img, results,
                     fg=True,
                     screen_w=1920, screen_h=1080,
                     cam_w=640, cam_h=480,
                     smooth_factor=0.5,
                     click_cooldown=0.3,
                     scroll_cooldown=0.3,
                     state={}):
    """
    Mouse controller using hand gestures.

    Args:
        img: The video frame.
        results: Mediapipe hands result object.
        fg: Show fancy graphics if True.
        screen_w, screen_h: Monitor resolution.
        cam_w, cam_h: Camera capture resolution.
        smooth_factor: Smoothing factor for cursor movement.
        click_cooldown: Cooldown between clicks (seconds).
        scroll_cooldown: Cooldown between scrolls (seconds).
        state: Dictionary for maintaining mouse state across frames.

    Returns:
        img: Frame with gesture overlays.
        state: Updated mouse state dictionary.
    """

    # Initialize persistent state (so function can be stateless in caller)
    state.setdefault("mouse_grabbed", False)
    state.setdefault("prev_finger", None)
    state.setdefault("prev_dx", 0)
    state.setdefault("prev_dy", 0)
    state.setdefault("last_click_time", 0)
    state.setdefault("last_scroll_time", 0)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        h, w, c = img.shape
        landmarks = {id: (int(lm.x * w), int(lm.y * h))
                     for id, lm in enumerate(handLms.landmark)}

        if all(k in landmarks for k in [4, 8, 12, 16, 20, 18]):  # Thumb, tips + pinky middle
            cx4, cy4 = landmarks[4]   # Thumb tip
            cx8, cy8 = landmarks[8]   # Index tip
            cx12, cy12 = landmarks[12] # Middle tip
            cx16, cy16 = landmarks[16] # Ring tip
            cx20, cy20 = landmarks[20] # Pinky tip
            cx18, cy18 = landmarks[18] # Pinky middle

            # Distances
            dist4_8 = ((cx4 - cx8)**2 + (cy4 - cy8)**2)**0.5
            dist4_12 = ((cx4 - cx12)**2 + (cy4 - cy12)**2)**0.5
            dist4_16 = ((cx4 - cx16)**2 + (cy4 - cy16)**2)**0.5
            dist4_20 = ((cx4 - cx20)**2 + (cy4 - cy20)**2)**0.5
            dist4_18 = ((cx4 - cx18)**2 + (cy4 - cy18)**2)**0.5

            # Fancy graphics
            if fg:
                for (cx, cy) in [landmarks[i] for i in [4, 8, 12, 16, 20, 18]]:
                    cv.circle(img, (cx, cy), 5, (0, 0, 255), -1)

                cv.line(img, (cx4, cy4), (cx8, cy8), (0, 255, 255), 2)
                cv.line(img, (cx4, cy4), (cx12, cy12), (0, 255, 255), 2)
                cv.line(img, (cx4, cy4), (cx16, cy16), (0, 255, 255), 2)
                cv.line(img, (cx4, cy4), (cx20, cy20), (0, 255, 255), 2)
                cv.line(img, (cx4, cy4), (cx18, cy18), (0, 255, 255), 2)

            # Gesture logic
            try:
                # Grab → drag & move mouse
                if dist4_8 < 20:
                    cv.putText(img, 'Grab', (cx8 + 20, cy8 - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if not state["mouse_grabbed"]:
                        state["mouse_grabbed"] = True
                        state["prev_finger"] = (cx8, cy8)
                        state["prev_dx"], state["prev_dy"] = 0, 0
                    else:
                        dx = cx8 - state["prev_finger"][0]
                        dy = cy8 - state["prev_finger"][1]

                        # Scale to screen
                        dx = dx * (screen_w / cam_w)
                        dy = dy * (screen_h / cam_h)

                        # Smooth movement
                        dx = dx * smooth_factor + state["prev_dx"] * (1 - smooth_factor)
                        dy = dy * smooth_factor + state["prev_dy"] * (1 - smooth_factor)

                        pyautogui.move(dx, dy)

                        state["prev_finger"] = (cx8, cy8)
                        state["prev_dx"], state["prev_dy"] = dx, dy
                else:
                    # Release
                    state["mouse_grabbed"] = False
                    state["prev_finger"] = None
                    state["prev_dx"], state["prev_dy"] = 0, 0

                # Left click
                if dist4_12 < 20:
                    if time.time() - state["last_click_time"] > click_cooldown:
                        pyautogui.click(button='left')
                        cv.putText(img, 'Left Click', (cx8 + 20, cy8 - 20),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        state["last_click_time"] = time.time()

                # Right click
                if dist4_16 < 20:
                    if time.time() - state["last_click_time"] > click_cooldown:
                        pyautogui.click(button='right')
                        cv.putText(img, 'Right Click', (cx8 + 20, cy8 - 20),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        state["last_click_time"] = time.time()

                # Scroll up
                if dist4_20 < 20:
                    if time.time() - state["last_scroll_time"] > scroll_cooldown:
                        pyautogui.scroll(-50)
                        cv.putText(img, 'Scroll Up', (cx20 + 20, cy20 - 20),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        state["last_scroll_time"] = time.time()

                # Scroll down
                if dist4_18 < 20:
                    if time.time() - state["last_scroll_time"] > scroll_cooldown:
                        pyautogui.scroll(50)
                        cv.putText(img, 'Scroll Down', (cx18 + 20, cy18 - 20),
                                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        state["last_scroll_time"] = time.time()

            except Exception as e:
                print(f"Error: {e}")
                cv.destroyAllWindows()

    return img, state

def keyboard_disk(img, landmarks, state, cx, cy, radius=200):
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    n = len(letters)

    # --- Gesture check ---
    cx4, cy4 = landmarks[4]    # thumb tip
    cx8, cy8 = landmarks[8]    # index tip
    cx12, cy12 = landmarks[12] # middle tip
    cx20, cy20 = landmarks[20] # pinky tip

    dist4_8 = ((cx8 - cx4) ** 2 + (cy8 - cy4) ** 2) ** 0.5
    dist4_12 = ((cx12 - cx4) ** 2 + (cy12 - cy4) ** 2) ** 0.5
    dist4_20 = ((cx20 - cx4) ** 2 + (cy20 - cy4) ** 2) ** 0.5

    # initialize state if not present
    if "disk_angle" not in state:
        state["disk_angle"] = 0
    if "rotating" not in state:
        state["rotating"] = False
    if "direction" not in state:
        state["direction"] = 1   # 1 = clockwise, -1 = counter
    if "last_selected" not in state:
        state["last_selected"] = None  # to avoid spamming keystrokes

    # --- Detect gestures ---
    # index pinch → start rotation (counterclockwise)
    if dist4_8 < 30:
        state["direction"] = -1
        state["rotating"] = True
    # thumb + pinky pinch → reverse direction (clockwise)
    elif dist4_20 < 30:
        state["direction"] = 1
        state["rotating"] = True
    else:
        state["direction"] = 1
        state["rotating"] = False

    # update rotation
    if state["rotating"]:
        state["disk_angle"] += 0.05 * state["direction"]  # constant angular speed

    # --- Draw Disk ---
    positions = []
    cv.circle(img, (cx, cy), radius, (0, 0, 0), 2)

    for i, letter in enumerate(letters):
        angle = (2 * math.pi / n) * i + state["disk_angle"]
        tx = int(cx + (radius + 30) * math.cos(angle))
        ty = int(cy + (radius + 30) * math.sin(angle))
        positions.append((letter, tx, ty))
        cv.putText(img, letter, (tx - 10, ty + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # --- Find leftmost letter ---
    leftmost = min(positions, key=lambda p: p[1])  # p[1] = tx
    letter, lx, ly = leftmost

    # Draw box around leftmost letter
    cv.rectangle(img, (lx - 20, ly - 15), (lx + 20, ly + 20), (0, 0, 255), 2)

    # --- Detect selection (thumb + middle pinch) ---
    if dist4_12 < 30:  # pinch detected
        if state["last_selected"] != letter:  # prevent repeat spamming
            pyautogui.write(letter)
            state["last_selected"] = letter
    else:
        state["last_selected"] = None  # reset when pinch released

    return img, state

def keyboard_qwerty(img, landmarks, state, cz_threshold=-0.1, hold_frames=12):
    if "buttons" not in state:
        # --- Create keyboard buttons ---
        class Button:
            def __init__(self, x, y, w, h, label):
                self.x, self.y, self.w, self.h = x, y, w, h
                self.label = label
                self.color = (200, 200, 200)
                self.counter = 0
                self.active = False

            def draw(self, img):
                cv.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, -1)
                cv.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 0, 0), 2)
                cv.putText(img, self.label, (self.x + 10, self.y + self.h - 15),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            def is_inside(self, px, py):
                return self.x < px < self.x + self.w and self.y < py < self.y + self.h

        keys = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM") + ["lower"],
        ]
        keys_spl = [
            ["Backspace","    Space    ","Enter"]
        ]

        buttons = []
        start_x, start_y = 10, 20
        key_w, key_h = 60, 60
        gap = 10

        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                w = 130 if key == "lower" else 60
                x = start_x + col_idx * (key_w + gap)
                y = start_y + row_idx * (key_h + gap)
                buttons.append(Button(x, y, w, key_h, key))

        start_x, start_y = 10, 225
        key_w, key_h = 190, 50
        for row_idx, row in enumerate(keys_spl):
            for col_idx, key in enumerate(row):
                if key == "    Space    ":
                    key_w = 290
                    start_x -= 100
                elif key == "Enter":
                    key_w = 125
                    start_x += 330
                x = start_x + col_idx * (key_w + gap)
                y = start_y + row_idx * (key_h + gap)
                buttons.append(Button(x, y, key_w, key_h, key))

        state["buttons"] = buttons
        state["typed"] = ""

    buttons = state["buttons"]

    # Draw buttons and typed text
    for btn in buttons:
        btn.draw(img)
    cv.putText(img, state["typed"], (10, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    if landmarks is None:
        return img, state

    # Use index fingertip for interaction
    lm = landmarks[8]
    h, w, _ = img.shape
    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
    cv.circle(img, (cx, cy), 10, (0, 0, 255), -1)
    cv.putText(img, f"Z: {cz:.2f}", (50, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check button clicks with debounce
    for btn in buttons:
        if btn.is_inside(cx, cy) and cz < cz_threshold:
            btn.color = (0, 255, 0)
            btn.counter += 1
            if btn.counter == hold_frames:
                if btn.label == "Backspace":
                    pyautogui.press("backspace")
                    state["typed"] = state["typed"][:-1]
                elif btn.label == "Enter":
                    pyautogui.press("enter")
                    state["typed"] += "\n"
                elif btn.label == "    Space    ":
                    pyautogui.press(" ")
                    state["typed"] += " "
                elif btn.label == "lower":
                    for b in buttons:
                        if len(b.label) == 1:
                            b.label = b.label.lower()
                    btn.label = "UPPER"
                elif btn.label == "UPPER":
                    for b in buttons:
                        if len(b.label) == 1:
                            b.label = b.label.upper()
                    btn.label = "lower"
                else:
                    pyautogui.typewrite(btn.label.strip())
                    state["typed"] += btn.label.strip()
                btn.active = True
        else:
            btn.color = (200, 200, 200)
            btn.counter = 0
            btn.active = False

    return img, state

