# Virtual Mouse - AI Hand Gesture Control

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-orange?style=for-the-badge)
![PyTest](https://img.shields.io/badge/Tests-32%20Passing-brightgreen?style=for-the-badge&logo=pytest)
![CI](https://github.com/Nitin31804/Virtual-Mouse/actions/workflows/pytest.yml/badge.svg)

> Control your computer mouse entirely with hand gestures — no physical mouse needed. Built with real-time Computer Vision and a Neural Network hand tracker.

---

## How It Works

```
Webcam Feed --> MediaPipe Neural Network --> Finger Coordinate Mapping --> System Mouse Control
```

| Gesture | Action |
|---|---|
| Index finger up | Move cursor |
| Index + Middle finger up | Click |
| Pinch gesture | Scroll |
| Closed fist | Drag |

---

## Key Features

- **Ghost Camera Protection** — Validates real frame capture, bypasses virtual drivers like OBS
- **Adaptive FPS Throttling** — Auto, High (60fps), Balanced (30fps), Battery (15fps) modes
- **Multi-Camera Fallback** — Automatically finds a working camera across indexes 0, 1, 2
- **32 Unit Tests** — Fully mocked PyTest suite, zero hardware needed to run tests
- **CI/CD Pipeline** — GitHub Actions runs all 32 tests on every push

---

## Tech Stack

| Layer | Technology |
|---|---|
| Computer Vision | OpenCV 4.10 |
| Hand Tracking AI | Google MediaPipe |
| Mouse Control | PyAutoGUI |
| Testing | PyTest |
| CI/CD | GitHub Actions |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Nitin31804/Virtual-Mouse.git
cd Virtual-Mouse

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Performance Modes

```python
# High performance (Gaming PC)
cam.set_performance_mode("high")    # 60 FPS inference

# Balanced (Standard laptop)
cam.set_performance_mode("balanced") # 30 FPS inference

# Battery saver (Travel mode)
cam.set_performance_mode("battery")  # 15 FPS inference

# Auto (Scales to your hardware)
cam.set_performance_mode("auto")     # Dynamic
```

---

## Architecture

```
src/
|-- camera/          # CameraManager with adaptive FPS throttling
|-- hand_detection/  # MediaPipe HandTracker wrapper
|-- gestures/        # GestureRecognizer state machine
|-- mouse/           # System mouse controller
|-- main.py          # Application entry point
tests/               # 32 fully mocked unit tests
```

