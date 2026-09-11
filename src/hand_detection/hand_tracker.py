"""
Hand Tracker
Uses MediaPipe to detect and track hand landmarks in real-time
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class Handedness(Enum):
    """Handedness enumeration"""

    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"


@dataclass
class HandLandmark:
    """Represents a single hand landmark"""

    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    z: float  # Normalized z coordinate
    visibility: float  # Visibility/confidence score

    def to_pixel(self, image_width: int, image_height: int) -> tuple:
        """Convert normalized coordinates to pixel coordinates"""
        return (int(self.x * image_width), int(self.y * image_height))


@dataclass
class Hand:
    """Represents a detected hand"""

    landmarks: List[HandLandmark]  # 21 landmarks
    handedness: Handedness
    bounding_box: tuple  # (x_min, y_min, x_max, y_max)
    center: tuple  # (x, y) center of hand
    confidence: float
    world_landmarks: Optional[List] = None  # 3D world coordinates

    def get_landmark_by_id(self, landmark_id: int) -> Optional[HandLandmark]:
        """Get landmark by MediaPipe landmark ID"""
        if 0 <= landmark_id < len(self.landmarks):
            return self.landmarks[landmark_id]
        return None

    def get_finger_tip(self, finger_index: int) -> Optional[HandLandmark]:
        """Get fingertip landmark for a specific finger"""
        # MediaPipe landmark indices for fingertips
        fingertip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        if 0 <= finger_index < len(fingertip_indices):
            return self.get_landmark_by_id(fingertip_indices[finger_index])
        return None

    def get_finger_state(self, threshold: float = 0.5) -> List[int]:
        """
        Get finger extended state (1 = extended, 0 = folded)

        Args:
            threshold: Height threshold for finger extension

        Returns:
            List of 5 integers representing thumb, index, middle, ring, pinky state
        """
        # Landmark indices for finger joints
        finger_joints = [
            [2, 3, 4],  # Thumb
            [6, 7, 8],  # Index
            [10, 11, 12],  # Middle
            [14, 15, 16],  # Ring
            [18, 19, 20],  # Pinky
        ]

        finger_states = []

        for joints in finger_joints:
            # Get landmarks for this finger
            landmarks = [self.get_landmark_by_id(i) for i in joints]

            if all(landmarks):
                # Calculate if fingertip is above the middle joint
                tip_y = landmarks[2].y  # Fingertip
                pip_y = landmarks[1].y  # PIP joint (middle joint)

                # Finger is extended if tip is above PIP joint (lower y value)
                is_extended = tip_y < pip_y - threshold * 0.01
                finger_states.append(1 if is_extended else 0)
            else:
                finger_states.append(0)

        return finger_states

    def calculate_palm_center(self) -> tuple:
        """Calculate palm center from wrist and metacarpal landmarks"""
        # Use wrist and metacarpal joints to find palm center
        wrist = self.get_landmark_by_id(0)  # Wrist
        mcp_indices = [1, 5, 9, 13, 17]  # MCP joints

        if wrist:
            mcp_points = [self.get_landmark_by_id(i) for i in mcp_indices]
            valid_points = [p for p in mcp_points if p]

            if valid_points:
                # Average of wrist and MCP joints
                avg_x = (wrist.x + sum(p.x for p in valid_points)) / (
                    len(valid_points) + 1
                )
                avg_y = (wrist.y + sum(p.y for p in valid_points)) / (
                    len(valid_points) + 1
                )
                return (avg_x, avg_y)

        return (0.5, 0.5)  # Default center


class HandTracker:
    def __init__(self, config: dict):
        """
        Initialize hand tracker with configuration

        Args:
            config: Hand detection configuration dictionary
        """
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=config.get("static_image_mode", False),
            max_num_hands=config.get("max_num_hands", 2),
            model_complexity=config.get("model_complexity", 1),
            min_detection_confidence=config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.5),
        )

        # Performance tracking
        self.frame_count = 0
        self.detection_time = 0
        self.average_detection_time = 0

        # Cache for hand tracking
        self.last_hands = []
        self.tracking_failure_count = 0

        logging.info("Hand Tracker initialized")

    def detect(self, image: np.ndarray) -> List[Hand]:
        """
        Detect hands in an image

        Args:
            image: Input BGR image

        Returns:
            List of detected Hand objects
        """
        if image is None or image.size == 0:
            return []

        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and find hands
            results = self.hands.process(image_rgb)

            detected_hands = []

            if results.multi_hand_landmarks:
                # Get image dimensions
                height, width = image.shape[:2]

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Process hand landmarks
                    hand = self._process_hand_landmarks(
                        hand_landmarks, handedness, width, height
                    )

                    if hand:
                        detected_hands.append(hand)

            # Update tracking cache
            self.last_hands = detected_hands

            # Reset failure count if hands detected
            if detected_hands:
                self.tracking_failure_count = 0
            else:
                self.tracking_failure_count += 1

            # Update performance metrics
            self.frame_count += 1

            return detected_hands

        except Exception as e:
            logging.error(f"Error in hand detection: {e}")
            return []

    def _process_hand_landmarks(
        self, landmarks, handedness, width, height
    ) -> Optional[Hand]:
        """Process MediaPipe hand landmarks into Hand object"""
        try:
            # Convert landmarks to our format
            hand_landmarks = []
            for lm in landmarks.landmark:
                hand_landmark = HandLandmark(
                    x=lm.x, y=lm.y, z=lm.z, visibility=getattr(lm, "visibility", 1.0)
                )
                hand_landmarks.append(hand_landmark)

            # Determine handedness
            hand_label = handedness.classification[0].label
            confidence = handedness.classification[0].score

            handedness_enum = Handedness.UNKNOWN
            if hand_label == "Left":
                handedness_enum = Handedness.LEFT
            elif hand_label == "Right":
                handedness_enum = Handedness.RIGHT

            # Calculate bounding box
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Add padding to bounding box
            padding = 0.05
            x_min = max(0, x_min - padding)
            x_max = min(1, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(1, y_max + padding)

            bounding_box = (x_min, y_min, x_max, y_max)

            # Calculate hand center
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            # Create Hand object
            hand = Hand(
                landmarks=hand_landmarks,
                handedness=handedness_enum,
                bounding_box=bounding_box,
                center=(center_x, center_y),
                confidence=confidence,
            )

            return hand

        except Exception as e:
            logging.error(f"Error processing hand landmarks: {e}")
            return None

    def draw_landmarks_on_image(self, image: np.ndarray, hand: Hand) -> np.ndarray:
        """
        Draw hand landmarks on image

        Args:
            image: Input image
            hand: Hand object to draw

        Returns:
            Image with landmarks drawn
        """
        if image is None or hand is None:
            return image

        # Create a copy of the image
        annotated_image = image.copy()
        height, width = image.shape[:2]

        # Draw connections
        connections = self.mp_hands.HAND_CONNECTIONS

        for connection in connections:
            start_idx, end_idx = connection

            start_lm = hand.get_landmark_by_id(start_idx)
            end_lm = hand.get_landmark_by_id(end_idx)

            if start_lm and end_lm:
                start_point = start_lm.to_pixel(width, height)
                end_point = end_lm.to_pixel(width, height)

                # Draw line
                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

        # Draw landmarks
        for i, landmark in enumerate(hand.landmarks):
            point = landmark.to_pixel(width, height)

            # Different colors for different landmark types
            if i == 0:  # Wrist
                color = (255, 0, 0)  # Blue
                radius = 8
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                color = (0, 0, 255)  # Red
                radius = 6
            else:  # Other landmarks
                color = (0, 255, 0)  # Green
                radius = 4

            cv2.circle(annotated_image, point, radius, color, -1)

        return annotated_image

    def get_hand_orientation(self, hand: Hand) -> float:
        """
        Calculate hand orientation angle

        Args:
            hand: Hand object

        Returns:
            Orientation angle in degrees
        """
        # Use wrist and middle finger MCP to determine orientation
        wrist = hand.get_landmark_by_id(0)
        middle_mcp = hand.get_landmark_by_id(9)

        if wrist and middle_mcp:
            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y
            angle = np.degrees(np.arctan2(dy, dx))
            return angle

        return 0.0

    def get_hand_size(self, hand: Hand) -> float:
        """
        Calculate relative hand size

        Args:
            hand: Hand object

        Returns:
            Relative hand size (0-1)
        """
        x_min, y_min, x_max, y_max = hand.bounding_box
        width = x_max - x_min
        height = y_max - y_min
        size = np.sqrt(width * width + height * height)

        return size

    def get_distance_between_hands(self, hand1: Hand, hand2: Hand) -> float:
        """
        Calculate distance between two hands

        Args:
            hand1: First hand
            hand2: Second hand

        Returns:
            Normalized distance
        """
        dx = hand1.center[0] - hand2.center[0]
        dy = hand1.center[1] - hand2.center[1]
        distance = np.sqrt(dx * dx + dy * dy)

        return distance

    def reset(self):
        """Reset tracker state"""
        self.last_hands = []
        self.tracking_failure_count = 0

    def release(self):
        """Release resources"""
        if self.hands:
            self.hands.close()
            logging.info("Hand Tracker released")

    def __del__(self):
        """Destructor"""
        self.release()
