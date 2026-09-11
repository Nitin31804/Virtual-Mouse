"""
Calibration Manager
Handles calibration of hand tracking to screen coordinates using a State Machine pattern.
Non-blocking implementation for real-time UI updates.
"""

import numpy as np
import json
import logging
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CalibrationPoint:
    screen_x: int
    screen_y: int
    hand_x: float
    hand_y: float


class CalibrationManager:
    def __init__(self, config: dict):
        self.config = config

        # State Tracking
        self.active = False
        self.step_index = 0
        self.samples_collected = 0
        self.step_start_time = 0
        self.last_sample_time = 0

        # Configuration
        self.samples_required = config.get("samples_required", 30)
        self.sample_delay = 0.05  # Seconds between samples

        # Define 5 Target Points (Normalized Screen Coords: 0.0-1.0)
        # Top-Left, Top-Right, Center, Bottom-Left, Bottom-Right
        self.targets = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]

        # Data Storage
        self.calibration_points: List[CalibrationPoint] = []
        self.calib_file = Path("data/calibration/calibration.pkl")
        self.calib_file.parent.mkdir(parents=True, exist_ok=True)

        # Output Results
        self.x_slope = 1.0
        self.x_intercept = 0.0
        self.y_slope = 1.0
        self.y_intercept = 0.0

        logging.info("Calibration Manager initialized")

    def start(self):
        """Begin calibration process"""
        self.active = True
        self.step_index = 0
        self.samples_collected = 0
        self.calibration_points = []
        self.step_start_time = time.time()
        logging.info("Calibration started")

    def stop(self):
        """Cancel/Stop calibration"""
        self.active = False
        logging.info("Calibration stopped")

    def process_frame(self, hand, screen_width: int, screen_height: int) -> str:
        """
        Process a single frame. Called by main.py loop.

        Returns:
            str: Instruction text to display on Overlay
        """
        if not self.active:
            return ""

        # Check if finished
        if self.step_index >= len(self.targets):
            self._finalize_calibration()
            self.active = False
            return "Calibration Complete!"

        # Get current target
        target_x_norm, target_y_norm = self.targets[self.step_index]

        # Logic: Wait 2 seconds before collecting samples for user to settle
        time_in_step = time.time() - self.step_start_time
        if time_in_step < 2.0:
            return f"Move to Target {self.step_index + 1}/{len(self.targets)}"

        # Logic: Hand Detection Check
        if not hand:
            return "Hand not detected!"

        # Logic: Sample Collection
        if time.time() - self.last_sample_time > self.sample_delay:
            # Use Palm Center for stable calibration
            if hasattr(hand, "calculate_palm_center"):
                hx, hy = hand.calculate_palm_center()
            else:
                # Fallback to wrist if palm calc missing
                lm = hand.landmarks[0]
                hx, hy = lm.x, lm.y

            # Convert Target to Pixel Coordinates
            sx = int(target_x_norm * screen_width)
            sy = int(target_y_norm * screen_height)

            self.calibration_points.append(CalibrationPoint(sx, sy, hx, hy))
            self.samples_collected += 1
            self.last_sample_time = time.time()

        # Check Step Completion
        if self.samples_collected >= self.samples_required:
            self.step_index += 1
            self.samples_collected = 0
            self.step_start_time = time.time()  # Reset timer for next step
            return "Hold..."

        # Progress Status
        return f"Collecting: {int((self.samples_collected/self.samples_required)*100)}%"

    def _finalize_calibration(self):
        """Calculate Linear Regression (y = mx + c)"""
        if len(self.calibration_points) < 5:
            logging.error("Not enough points for calibration")
            return

        # Extract Arrays
        screen_x = np.array([p.screen_x for p in self.calibration_points])
        screen_y = np.array([p.screen_y for p in self.calibration_points])
        hand_x = np.array([p.hand_x for p in self.calibration_points])
        hand_y = np.array([p.hand_y for p in self.calibration_points])

        # Linear Regression: Screen = Slope * Hand + Intercept
        # We assume independent X and Y axes
        self.x_slope, self.x_intercept = np.polyfit(hand_x, screen_x, 1)
        self.y_slope, self.y_intercept = np.polyfit(hand_y, screen_y, 1)

        # Save Data
        data = {
            "x_slope": self.x_slope,
            "x_intercept": self.x_intercept,
            "y_slope": self.y_slope,
            "y_intercept": self.y_intercept,
        }

        with open(self.calib_file, "wb") as f:
            pickle.dump(data, f)

        logging.info(f"Calibration Saved: X_Slope={self.x_slope:.2f}")

    def load_calibration(self) -> Optional[dict]:
        """Load saved calibration data"""
        if self.calib_file.exists():
            try:
                with open(self.calib_file, "rb") as f:
                    data = pickle.load(f)
                    self.x_slope = data["x_slope"]
                    self.x_intercept = data["x_intercept"]
                    self.y_slope = data["y_slope"]
                    self.y_intercept = data["y_intercept"]
                    return data
            except Exception as e:
                logging.error(f"Failed to load calibration: {e}")
        return None

    def get_draw_info(self):
        """Returns data needed by Overlay to draw target circles"""
        if not self.active or self.step_index >= len(self.targets):
            return None
        return {
            "target": self.targets[self.step_index],
            "progress": self.samples_collected / self.samples_required,
        }
