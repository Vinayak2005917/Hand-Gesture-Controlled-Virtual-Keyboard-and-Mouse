# Hand-Gesture Controlled Virtual Keyboard and Mouse

A simple Streamlit app that uses OpenCV and MediaPipe hand tracking to control the mouse and type letters via gestures.

## Quick start

Requirements:
- Python 3.10
- Webcam
- Windows (tested)

Setup (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run Main.py
```

Grant camera permissions when prompted.

## How it works
- The camera feed is shown in the browser via Streamlit.
- MediaPipe detects hand landmarks; gestures switch modes and trigger actions.
- PyAutoGUI performs mouse movement/clicks and types selected letters.

Gesture data files required (relative paths):
- `Gesture data/palm_data.txt`
- `Gesture data/fist_data.txt`
- `Gesture data/peace_sign_data.txt`

## Modes and controls

Mode selection (shown on start or after reset):
- Palm → Mouse mode
- Fist → Keyboard mode
- Peace sign → Reset to None

Mouse mode (thumb vs. other fingertips):
- Thumb + Index pinch → Grab/drag cursor; move hand to move cursor
- Thumb + Middle pinch → Left click
- Thumb + Ring pinch → Right click
- Thumb + Pinky tip pinch → Scroll up
- Thumb + Pinky middle joint pinch → Scroll down

Keyboard mode (rotating disk of letters):
- Thumb + Index pinch → Rotate counterclockwise
- Thumb + Pinky pinch → Rotate clockwise
- Thumb + Middle pinch → Select current highlighted (leftmost) letter and type it

## Files
- `Main.py` — Streamlit app loop, mode handling, camera/UI.
- `utils.py` — Gesture detection, mouse controller, and circular keyboard logic.

## Notes
- Screen size is auto-detected via PyAutoGUI; multi-monitor setups may behave differently.
- For reliable clicking/typing, keep the browser window focused.
- If no camera is detected, the app shows a warning and stops.
