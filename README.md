# AI Virtual Mouse 🖱️🤖

Control your computer's mouse using hand gestures and your webcam! This project uses OpenCV and Google's MediaPipe to detect hand landmarks and translates them into smooth cursor movements, clicks, and scrolls.

## Features ✨
*   **Hand Tracking:** Real-time, low-latency hand detection using MediaPipe.
*   **Cursor Control:** Smooth, jitter-free cursor movement with adjustable sensitivity.
*   **Gestures:** Out-of-the-box support for clicking, double-clicking, dragging, and scrolling.
*   **Customizable:** Easy-to-edit configuration (`config/settings.yaml` and `config/gestures.json`) to adjust speed, active areas, and add new gestures.
*   **HUD Overlay:** Sleek on-screen display showing FPS, recognized gestures, and hand skeletons.

## Installation 🛠️

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-virtual-mouse.git
   cd ai-virtual-mouse
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

## Usage 🚀

Run the main application script:
```bash
python src/main.py
```

### Default Gestures

| Gesture Name | Hand Shape | Action |
| :--- | :--- | :--- |
| **Move Cursor** | 🖐 Open Hand | Moves the mouse pointer |
| **Left Click** | ✊ Closed Fist | Single Left Click |
| **Double Click** | ✌️ Peace Sign | Double Left Click |
| **Hold / Drag** | 🤏 Pinch (Thumb + Index) | Clicks and holds (for dragging) |
| **Scroll** | ☝️ Pointing (Index Only) | Joystick Scroll (Move hand up/down) |
| **Refresh Page** | 3 Fingers | Presses F5 |

## Configuration ⚙️
You can tweak the performance, camera resolution, and mouse sensitivity inside `config/settings.yaml`. 
To modify or create custom hand shapes for new shortcuts, edit `config/gestures.json`.

## License 📄
This project is licensed under the MIT License - see the LICENSE file for details.
