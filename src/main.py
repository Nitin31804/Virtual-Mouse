#!/usr/bin/env python3
"""
AI Virtual Mouse - Main Application
Control your computer cursor with hand gestures.
Orchestrates Camera, Tracker, Gesture Engine, and UI.
"""

import cv2
import sys
import logging
import os
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.camera.camera_manager import CameraManager
from src.hand_detection.hand_tracker import HandTracker
from src.gestures.gesture_recognizer import GestureRecognizer
from src.cursor.cursor_controller import CursorController
from src.ui.overlay import OverlayDisplay
from src.utils.calibration import CalibrationManager
from src.utils.config_loader import ConfigLoader


class AIVirtualMouse:
    def __init__(self, config_path="config/settings.yaml"):
        """Initialize the AI Virtual Mouse application"""
        self.setup_logging()

        # Load configuration
        self.config = ConfigLoader.load_config(config_path)
        if not self.config:
            logging.critical("Failed to load configuration. Exiting.")
            sys.exit(1)

        # Initialize components containers
        self.camera = None
        self.hand_tracker = None
        self.gesture_recognizer = None
        self.cursor_controller = None
        self.overlay = None
        self.calibration = None

        # Application state
        self.running = False
        self.paused = False

        # Performance tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = 0

        logging.info("AI Virtual Mouse initialized")

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("ai_virtual_mouse.log"),
                logging.StreamHandler(),
            ],
        )

    def initialize_components(self):
        """Initialize all system components"""
        try:
            logging.info("Initializing components...")

            # 1. Camera
            self.camera = CameraManager(self.config["camera"])

            # 2. Tracker
            self.hand_tracker = HandTracker(self.config["hand_detection"])

            # 3. Gestures (Load JSON)
            gesture_path = "config/gestures.json"
            if not os.path.exists(gesture_path):
                raise FileNotFoundError(f"Gestures config not found at {gesture_path}")

            # Use ConfigLoader if available, or standard json load
            with open(gesture_path, "r") as f:
                gestures_config = json.load(f)

            self.gesture_recognizer = GestureRecognizer(
                gestures_config, self.config["gestures"]
            )

            # 4. Cursor Controller
            self.cursor_controller = CursorController(self.config["mouse"])

            # 5. UI Overlay
            self.overlay = OverlayDisplay(self.config["ui"])

            # 6. Calibration Manager
            self.calibration = CalibrationManager(self.config["calibration"])
            # Load previous calibration if available
            calib_data = self.calibration.load_calibration()
            if calib_data:
                # Update cursor controller with loaded calibration
                self.cursor_controller.calib_slope = list(calib_data.values())[
                    0:2
                ]  # Simplification, ensure data matches
                self.cursor_controller.is_calibrated = True

            logging.info("All components initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            return False

    def process_frame(self, frame):
        """
        Process a single video frame.
        Handles the logic switch between 'Calibration Mode' and 'Normal Mode'.
        """
        # Detect hands
        hands = self.hand_tracker.detect(frame)
        primary_hand = hands[0] if hands else None

        height, width = frame.shape[:2]

        # ---------------------------------------------------------
        # MODE 1: CALIBRATION
        # ---------------------------------------------------------
        if self.calibration.active:
            # Run non-blocking calibration logic
            instruction = self.calibration.process_frame(primary_hand, width, height)

            # Draw visual feedback
            draw_data = self.calibration.get_draw_info()

            if draw_data:
                # Convert normalized target (0.1) to pixel (128, 72)
                tx, ty = draw_data["target"]
                target_px = (int(tx * width), int(ty * height))

                # Draw the target circle and progress bar
                # Note: We reuse draw_calibration_info from overlay.py
                # You might need to adjust arguments based on your exact overlay implementation
                self.overlay.draw_calibration_info(
                    frame,
                    self.calibration.step_index + 1,
                    len(self.calibration.targets),
                    instruction,
                )

            # Draw skeleton for feedback
            if primary_hand:
                self.overlay.draw_landmarks(frame, primary_hand)

            return frame

        # ---------------------------------------------------------
        # MODE 2: NORMAL OPERATION
        # ---------------------------------------------------------

        # Always draw the "Active Region" box so user knows the boundaries
        self.overlay.draw_active_region(
            frame, margin_ratio=self.cursor_controller.frame_margin
        )

        if hands:
            for hand in hands:
                # 1. Recognize gesture
                gesture_name = self.gesture_recognizer.recognize(hand)

                # 2. Execute Action (if not paused)
                if not self.paused and gesture_name:
                    self.cursor_controller.execute_action(gesture_name, hand)

                    # Trigger visual click animation if needed
                    if "click" in gesture_name:
                        # Get finger tip for animation origin
                        tip = hand.get_finger_tip(8)  # Index tip
                        if tip:
                            px, py = tip.to_pixel(width, height)
                            self.overlay.trigger_click_animation(px, py)

                # 3. Draw UI Layers
                # Grouped for performance
                if self.config["ui"]["show_landmarks"]:
                    self.overlay.draw_landmarks(frame, hand)

                if self.config["ui"]["show_gesture_info"]:
                    self.overlay.draw_gesture_info(frame, gesture_name, hand)

                if self.config["ui"]["show_hand_box"]:
                    self.overlay.draw_hand_box(frame, hand)

        # Draw Click Animation (Ripple effect)
        self.overlay.draw_click_animation(frame)

        # Draw FPS
        if self.config["ui"]["show_fps"]:
            self.overlay.draw_fps(frame, self.fps)

        return frame

    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def handle_keyboard_input(self, key):
        """Handle keyboard shortcuts"""
        if key == ord("q") or key == 27:  # 'q' or ESC
            self.running = False
            logging.info("Quit command received")

        elif key == ord("p"):
            self.paused = not self.paused
            status = "paused" if self.paused else "resumed"
            logging.info(f"Application {status}")

        elif key == ord("c"):
            # Toggle Calibration Mode
            if self.calibration.active:
                self.calibration.stop()
            else:
                self.calibration.start()
            logging.info(f"Calibration active: {self.calibration.active}")

        elif key == ord("s"):
            # Save current settings
            ConfigLoader.save_config(self.config, "config/settings_backup.yaml")
            logging.info("Settings saved")

    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up resources...")
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        logging.info("Application shutdown complete")

    def run(self):
        """Main application loop"""
        logging.info("Starting AI Virtual Mouse application")

        if not self.initialize_components():
            logging.error("Failed to initialize components. Exiting...")
            return

        if not self.camera.open():
            logging.error("Failed to open camera. Exiting...")
            return

        self.running = True
        logging.info(
            "Application started. Press 'q' to quit, 'p' to pause, 'c' to calibrate"
        )

        try:
            while self.running:
                # 1. Capture
                ret, frame = self.camera.read()
                if not ret:
                    logging.warning("Failed to capture frame")
                    continue

                # 2. Process
                if self.paused:
                    self.overlay.draw_paused(frame)
                    processed_frame = frame
                else:
                    processed_frame = self.process_frame(frame)

                # 3. Calculate FPS
                self.calculate_fps()

                # 4. Display
                cv2.imshow("AI Virtual Mouse", processed_frame)

                # 5. Input
                key = cv2.waitKey(1) & 0xFF
                self.handle_keyboard_input(key)

        except KeyboardInterrupt:
            logging.info("Application interrupted by user")
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    """Main entry point"""
    app = AIVirtualMouse()
    app.run()


if __name__ == "__main__":
    main()
