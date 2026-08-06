"""
Overlay Display
Handles drawing UI elements, information, and visual feedback on the camera feed.
Visualizes the AI's "Brain" (Gestures) and "Eyes" (Landmarks).
"""

import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class ColorScheme:
    """UI color scheme (BGR Format for OpenCV)"""
    background: Tuple[int, int, int]
    foreground: Tuple[int, int, int]
    accent: Tuple[int, int, int]
    warning: Tuple[int, int, int]
    success: Tuple[int, int, int]
    
    @classmethod
    def from_name(cls, scheme_name: str) -> 'ColorScheme':
        """Create color scheme from name (BGR colors)"""
        schemes = {
            "dark": cls(
                background=(52, 44, 40),    # Dark Gray
                foreground=(220, 220, 220), # White-ish
                accent=(239, 175, 97),      # Light Blue
                warning=(123, 192, 229),    # Orange/Yellow
                success=(121, 195, 152)     # Light Green
            ),
            "neon": cls(
                background=(20, 20, 20),
                foreground=(255, 255, 255),
                accent=(255, 0, 255),       # Magenta
                warning=(0, 255, 255),      # Cyan
                success=(0, 255, 0)         # Lime
            )
        }
        return schemes.get(scheme_name, schemes["dark"])

class OverlayDisplay:
    def __init__(self, config: dict):
        self.config = config
        self.colors = ColorScheme.from_name(config.get('color_scheme', 'neon'))
        
        # Settings
        self.opacity = config.get('overlay_opacity', 0.6)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Toggles
        self.show_fps = config.get('show_fps', True)
        self.show_landmarks = config.get('show_landmarks', True)
        self.show_gesture_info = config.get('show_gesture_info', True)
        self.show_box = config.get('show_hand_box', True)
        
        # State
        self.click_anim_start = 0
        self.click_anim_pos = (0, 0)
        
        logging.info("Overlay Display initialized")

    def draw_active_region(self, image: np.ndarray, margin_ratio: float = 0.15) -> np.ndarray:
        """
        Draw the rectangle representing the active mouse control area.
        Crucial for user to know where the edges of the screen are.
        """
        h, w = image.shape[:2]
        
        x1 = int(w * margin_ratio)
        y1 = int(h * margin_ratio)
        x2 = int(w * (1 - margin_ratio))
        y2 = int(h * (1 - margin_ratio))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), self.colors.accent, 1)
        return image

    def draw_landmarks(self, image: np.ndarray, hand) -> np.ndarray:
        """Draw hand skeleton"""
        if not self.show_landmarks or hand is None:
            return image
            
        h, w = image.shape[:2]
        
        # Standard MediaPipe connections
        connections = [
            (0,1), (1,2), (2,3), (3,4),     # Thumb
            (0,5), (5,6), (6,7), (7,8),     # Index
            (0,9), (9,10), (10,11), (11,12),# Middle
            (0,13), (13,14), (14,15), (15,16),# Ring
            (0,17), (17,18), (18,19), (19,20) # Pinky
        ]
        
        # Draw Bones
        for start, end in connections:
            p1 = hand.get_landmark_by_id(start)
            p2 = hand.get_landmark_by_id(end)
            if p1 and p2:
                pt1 = p1.to_pixel(w, h)
                pt2 = p2.to_pixel(w, h)
                cv2.line(image, pt1, pt2, self.colors.accent, 2)
        
        # Draw Joints
        for i, lm in enumerate(hand.landmarks):
            pt = lm.to_pixel(w, h)
            # Tips are Red, Joints are Green
            color = (0, 0, 255) if i in [4, 8, 12, 16, 20] else (0, 255, 0)
            radius = 6 if i in [4, 8, 12, 16, 20] else 4
            cv2.circle(image, pt, radius, color, -1)
            
        return image

    def draw_hand_box(self, image: np.ndarray, hand) -> np.ndarray:
        """
        Draw bounding box around hand.
        (This was the missing method causing the crash)
        """
        if not self.show_box or hand is None:
            return image
        
        height, width = image.shape[:2]
        
        # Get bounding box coordinates from the Hand object
        # (x_min, y_min, x_max, y_max) are normalized 0.0-1.0
        x_min, y_min, x_max, y_max = hand.bounding_box
        
        # Convert to pixel coordinates
        x1 = int(x_min * width)
        y1 = int(y_min * height)
        x2 = int(x_max * width)
        y2 = int(y_max * height)
        
        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), self.colors.warning, 2)
        
        # Optional: Draw label background
        label = f"{hand.handedness.value} ({int(hand.confidence * 100)}%)"
        (w, h), _ = cv2.getTextSize(label, self.font, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), self.colors.warning, -1)
        cv2.putText(image, label, (x1, y1 - 5), self.font, 0.5, self.colors.background, 1)
        
        return image

    def draw_gesture_info(self, image: np.ndarray, gesture_name: str, hand) -> np.ndarray:
        """Draw semi-transparent info panel"""
        if not self.show_gesture_info:
            return image
            
        h, w = image.shape[:2]
        panel_w, panel_h = 250, 80
        x, y = w - panel_w - 20, 20
        
        # Transparent Background
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), 
                      self.colors.background, -1)
        cv2.addWeighted(overlay, self.opacity, image, 1 - self.opacity, 0, image)
        
        # Text
        display_name = gesture_name.replace('_', ' ').upper() if gesture_name else "NEUTRAL"
        color = self.colors.success if gesture_name else self.colors.foreground
        
        cv2.putText(image, "GESTURE:", (x + 10, y + 30), 
                    self.font, 0.6, self.colors.foreground, 1)
        cv2.putText(image, display_name, (x + 10, y + 60), 
                    self.font, 0.8, color, 2)
        
        # Confidence Bar
        if hand and hasattr(hand, 'confidence'):
            bar_w = int((panel_w - 20) * hand.confidence)
            cv2.rectangle(image, (x + 10, y + 70), (x + 10 + bar_w, y + 75), 
                          self.colors.accent, -1)
            
        return image

    def draw_fps(self, image: np.ndarray, fps: float) -> np.ndarray:
        if not self.show_fps:
            return image
            
        text = f"FPS: {int(fps)}"
        cv2.putText(image, text, (20, 40), self.font, 1, self.colors.success, 2)
        return image

    def trigger_click_animation(self, x: int, y: int):
        """Start the ripple effect"""
        self.click_anim_start = time.time()
        self.click_anim_pos = (x, y)

    def draw_click_animation(self, image: np.ndarray) -> np.ndarray:
        """Draw expanding ripple if active"""
        elapsed = time.time() - self.click_anim_start
        if elapsed < 0.3: # 300ms animation
            radius = int(50 * (elapsed / 0.3))
            alpha = 1 - (elapsed / 0.3)
            
            overlay = image.copy()
            cv2.circle(overlay, self.click_anim_pos, radius, (0, 255, 255), 2)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
        return image

    def draw_paused(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        overlay = image.copy()
        # Darken screen
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        text = "PAUSED - Press 'P' to Resume"
        ts = cv2.getTextSize(text, self.font, 1, 2)[0]
        tx = (w - ts[0]) // 2
        ty = (h + ts[1]) // 2
        cv2.putText(image, text, (tx, ty), self.font, 1, self.colors.foreground, 2)
        return image
    
    def draw_calibration_info(self, image: np.ndarray, step: int, total_steps: int,
                             instruction: str) -> np.ndarray:
        """
        Draw calibration information (Progress bar and target)
        """
        height, width = image.shape[:2]
        
        # 1. Draw Progress Bar
        progress_width = width - 100
        progress_height = 20
        progress_x = 50
        progress_y = height - 100
        
        # Background
        cv2.rectangle(image, (progress_x, progress_y),
                     (progress_x + progress_width, progress_y + progress_height),
                     (100, 100, 100), -1)
        
        # Fill
        progress = (step - 1) / total_steps if total_steps > 0 else 0
        fill_width = int(progress_width * progress)
        cv2.rectangle(image, (progress_x, progress_y),
                     (progress_x + fill_width, progress_y + progress_height),
                     self.colors.success, -1)
        
        # Text
        prog_text = f"Calibration: {step}/{total_steps}"
        cv2.putText(image, prog_text, (progress_x, progress_y - 10),
                   self.font, 0.7, self.colors.foreground, 2)
        
        # 2. Draw Instruction
        text_size = cv2.getTextSize(instruction, self.font, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2 - 50
        
        # Add shadow for readability
        cv2.putText(image, instruction, (text_x + 2, text_y + 2),
                   self.font, 1.0, (0, 0, 0), 2)
        cv2.putText(image, instruction, (text_x, text_y),
                   self.font, 1.0, self.colors.warning, 2)
        
        # 3. Draw Target Circle
        # Note: The caller (main.py) usually draws the specific target circle location,
        # but we can provide visual flair here if needed.
        
        return image