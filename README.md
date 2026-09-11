# AI Virtual Mouse

## Overview
A real-time Computer Vision application that translates hand gestures into physical mouse movements on your screen using mathematical facial/hand coordinate mapping.

## Tech Stack
- **Core Logic:** Python 3.12
- **Computer Vision:** OpenCV
- **Neural Network Tracking:** Google MediaPipe
- **Testing:** PyTest
- **CI/CD:** GitHub Actions

## Key Features
- **Hardware-Aware FPS Throttling:** Intelligently drops inference frames dynamically (Auto, High, Balanced, Battery modes) to prevent CPU overheating and battery drain.
- **Ghost Camera Protection:** Automatically falls back and validates hardware camera streams by confirming real frame captures, bypassing dummy drivers like OBS Virtual Camera.
- **Perfect Test Coverage:** A fully mocked PyTest suite simulating camera hardware threads, mathematically protecting against infinite loop timeouts. Automatically run on every GitHub push.

## Getting Started
```bash
pip install -r requirements.txt
python src/main.py
```

## Testing
Run the unit tests:
```bash
pytest tests/ -v
```
