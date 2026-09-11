"""
Gesture Recognizer
Recognizes hand gestures from hand landmark data.
Matches geometric finger states against a loaded configuration.
"""

import numpy as np
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum, auto


# We keep standard types for internal logic, but allow dynamic ones
class ActionType(Enum):
    """Standard actions that gestures can map to"""

    MOVE = "move_cursor"
    CLICK_LEFT = "left_click"
    CLICK_RIGHT = "right_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    DRAG = "drag"
    NONE = "none"


@dataclass
class Gesture:
    """Gesture data class"""

    name: str
    finger_state: List[int]  # [thumb, index, middle, ring, pinky]
    confidence: float
    timestamp: float = 0.0
    duration: float = 0.0

    def __str__(self):
        return f"{self.name} ({self.confidence:.2f})"


class GestureHistory:
    """Maintains history of detected gestures for smoothing"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Gesture] = []
        self.current_gesture: Optional[Gesture] = None
        self.gesture_start_time: float = 0.0

    def add(self, gesture: Gesture):
        """Add gesture to history with temporal consistency check"""
        gesture.timestamp = time.time()

        # Check if this is a continuation of the current gesture
        if self.current_gesture and self.current_gesture.name == gesture.name:
            gesture.duration = gesture.timestamp - self.gesture_start_time
            self.current_gesture = gesture  # Update current
        else:
            # New gesture detected
            gesture.duration = 0.0
            self.current_gesture = gesture
            self.gesture_start_time = gesture.timestamp

        self.history.append(gesture)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_most_common(self, recent_count: int = 5) -> Optional[Gesture]:
        """Smoothing: Get the dominant gesture from recent history"""
        if not self.history:
            return None

        # Slice the last N items
        recent = self.history[-recent_count:]
        if not recent:
            return None

        # Count frequency of names
        counts = {}
        for g in recent:
            counts[g.name] = counts.get(g.name, 0) + 1

        # Find winner
        winner_name = max(counts, key=counts.get)

        # Return the most recent instance of the winner
        for g in reversed(recent):
            if g.name == winner_name:
                return g
        return None

    def clear(self):
        self.history.clear()
        self.current_gesture = None
        self.gesture_start_time = 0.0


class GestureRecognizer:
    def __init__(self, gestures_config: dict, settings: dict):
        """
        Initialize gesture recognizer
        """
        self.gestures_config = gestures_config
        self.settings = settings

        # Load definitions dynamically
        self.gesture_definitions = self._load_gesture_definitions()

        # History buffer
        self.history = GestureHistory(
            max_history=settings.get("gesture_buffer_size", 10)
        )

        # Timers
        self.last_gesture_time = 0
        self.cooldown_period = settings.get("cooldown_period", 0.5)
        self.min_confidence = settings.get("confidence_threshold", 0.8)

        logging.info("Gesture Recognizer initialized")

    def _load_gesture_definitions(self) -> Dict[str, dict]:
        """Load gestures from JSON config"""
        definitions = {}

        if "gestures" not in self.gestures_config:
            logging.error("No gestures found in configuration")
            return definitions

        for key, config in self.gestures_config["gestures"].items():
            # Robustly handle the config
            if "finger_state" not in config:
                continue

            # Use the key (e.g., 'left_click') as the name
            definitions[key] = {
                "finger_state": config["finger_state"],
                "priority": config.get("priority", 0),
                "parameters": config.get("parameters", {}),
            }

        return definitions

    def recognize(self, hand) -> Optional[str]:
        """
        Main pipeline: Hand -> Finger State -> Match -> Smooth -> Result
        """
        if hand is None:
            return None

        try:
            # 1. Get Physical State
            finger_state = hand.get_finger_state()

            # 2. Match against database
            matched_gesture = self._find_best_match(finger_state, hand)

            if matched_gesture:
                # 3. Add to history
                self.history.add(matched_gesture)

                # 4. Apply Smoothing (Debouncing)
                # We look at the last 3 frames. If they agree, we accept.
                smoothed_gesture = self.history.get_most_common(recent_count=3)

                if smoothed_gesture:
                    # Optional: Check Cooldowns here if needed
                    return smoothed_gesture.name

            return None

        except Exception as e:
            logging.error(f"Error in gesture recognition: {e}")
            return None

    def _find_best_match(self, current_state: List[int], hand) -> Optional[Gesture]:
        """Compare current fingers against all definitions"""
        best_match = None
        best_score = -1.0

        for name, definition in self.gesture_definitions.items():
            target_state = definition["finger_state"]

            # 1. Base Score: Exact Finger Match
            # Simple Hamming distance (how many fingers match?)
            matches = sum(1 for a, b in zip(current_state, target_state) if a == b)
            score = matches / 5.0

            # 2. Filter: High Threshold
            # If the score isn't perfect (1.0) or very close, skip it.
            # We don't want "almost a click" to trigger a click.
            if score < 0.8:
                continue

            # 3. Rule-Based Validation (Hardcoded Physics Checks)
            if not self._validate_physics(name, hand):
                score -= 0.3  # Penalize invalid physics

            # 4. Priority Check
            # If we have two matches (e.g. Neutral vs Move), pick higher priority
            if score > best_score:
                best_score = score
                best_match = Gesture(
                    name=name, finger_state=current_state, confidence=score
                )
            elif score == best_score:
                # Tie-breaker: Priority
                if (
                    definition["priority"]
                    > self.gesture_definitions[best_match.name]["priority"]
                ):
                    best_match = Gesture(
                        name=name, finger_state=current_state, confidence=score
                    )

        if best_match and best_match.confidence >= self.min_confidence:
            return best_match
        return None

    def _validate_physics(self, gesture_name: str, hand) -> bool:
        """
        Sanity checks that pure finger states miss.
        e.g., Is the hand flat? Is the thumb actually tucked?
        """
        try:
            # Get key landmarks
            wrist = hand.get_landmark_by_id(0)
            index_tip = hand.get_landmark_by_id(8)
            middle_tip = hand.get_landmark_by_id(12)

            if not (wrist and index_tip):
                return False

            # Rule 1: "Click" gestures require a stable hand
            if "click" in gesture_name:
                # Check Z-depth to ensure hand is not tilted wildly
                if abs(wrist.z - index_tip.z) > 0.3:
                    return False

            return True

        except Exception:
            return True  # Fail open if landmarks missing
