"""
Cursor Controller
Controls mouse cursor movement, dragging, scrolling, and actions.
"""

import pyautogui
import numpy as np
import logging
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

class MouseAction(Enum):
    MOVE = "move"
    DRAG = "drag"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    LEFT_CLICK = "left_click"
    DOUBLE_CLICK = "double_click"
    REFRESH = "refresh"
    NONE = "none"

@dataclass
class CursorState:
    x: int = 0
    y: int = 0
    action: MouseAction = MouseAction.NONE
    last_action_time: float = 0
    is_dragging: bool = False

class CursorController:
    def __init__(self, config: dict):
        self.config = config
        
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        self.screen_width, self.screen_height = pyautogui.size()
        self.state = CursorState()
        self.smoothed_position = np.array([0.0, 0.0])
        
        # Settings
        self.smoothing_factor = config.get('smoothing_factor', 0.25)
        self.scroll_speed = 60
        self.frame_margin = config.get('frame_margin', 0.25)  # Increased from 0.15 to make it easier to reach screen edges 
        
        # Calibration
        self.is_calibrated = False
        self.calib_slope = np.array([1.0, 1.0])
        self.calib_intercept = np.array([0.0, 0.0])
        
        self.lock = threading.Lock()
        logging.info("Cursor Controller initialized")
    
    def execute_action(self, gesture_name: str, hand) -> bool:
        with self.lock:
            try:
                method_name = f"_handle_{gesture_name}"
                if hasattr(self, method_name):
                    handler = getattr(self, method_name)
                    return handler(hand)
                return False
            except Exception as e:
                logging.error(f"Error executing {gesture_name}: {e}")
                return False

    def _handle_cursor_move(self, hand) -> bool:
        """FIST: Move Cursor"""
        if self.state.is_dragging:
            pyautogui.mouseUp()
            self.state.is_dragging = False
            logging.info("Dropped File")

        knuckle = hand.get_landmark_by_id(5)
        if knuckle:
            self._move_cursor_smooth(knuckle.x, knuckle.y)
            self.state.action = MouseAction.MOVE
            return True
        return False

    def _handle_drag_middle(self, hand) -> bool:
        """MIDDLE FINGER UP: Hold/Drag"""
        if not self.state.is_dragging:
            pyautogui.mouseDown()
            self.state.is_dragging = True
            
        knuckle = hand.get_landmark_by_id(5)
        if knuckle:
            self._move_cursor_smooth(knuckle.x, knuckle.y)
            self.state.action = MouseAction.DRAG
            return True
        return True

    def _handle_scroll_smart(self, hand):
        """
        OPEN HAND: Joystick Scroll.
        Top 30% of screen = Scroll UP
        Bottom 30% of screen = Scroll DOWN
        Middle = Stop
        """
        # Track the Index Knuckle (Center of hand)
        knuckle = hand.get_landmark_by_id(5)
        
        if knuckle:
            # y is normalized (0.0 is top, 1.0 is bottom)
            
            if knuckle.y < 0.35: 
                # Hand is in Top 35% -> Scroll UP
                pyautogui.scroll(self.scroll_speed)
                self.state.action = MouseAction.SCROLL_UP
                
            elif knuckle.y > 0.65:
                # Hand is in Bottom 35% -> Scroll DOWN
                pyautogui.scroll(-self.scroll_speed)
                self.state.action = MouseAction.SCROLL_DOWN
                
            else:
                # Hand is in the Middle -> Do Nothing (Neutral)
                self.state.action = MouseAction.NONE
                
        return True

    def _handle_refresh(self, hand):
        """3 Fingers -> F5"""
        current_time = time.time()
        if current_time - self.state.last_action_time > 2.0:
            pyautogui.press('f5')
            self.state.last_action_time = current_time
            self.state.action = MouseAction.REFRESH
        return True

    def _handle_left_click(self, hand):
        """Rock Sign -> Left Click"""
        if self.state.is_dragging:
            pyautogui.mouseUp()
            self.state.is_dragging = False
            
        pyautogui.click()
        self.state.action = MouseAction.LEFT_CLICK
        self._handle_cursor_move(hand)
        return True

    def _handle_double_click(self, hand):
        """Index Only -> Double Click"""
        current_time = time.time()
        if current_time - self.state.last_action_time > 1.0:
            pyautogui.doubleClick()
            self.state.last_action_time = current_time
            self.state.action = MouseAction.DOUBLE_CLICK
        
        self._handle_cursor_move(hand)
        return True

    def _move_cursor_smooth(self, target_x_norm, target_y_norm):
        screen_x, screen_y = self._convert_to_screen_coordinates(target_x_norm, target_y_norm)
        
        current_pos = np.array([screen_x, screen_y])
        if self.smoothed_position[0] == 0:
            self.smoothed_position = current_pos
            
        self.smoothed_position = (
            self.smoothing_factor * current_pos + 
            (1 - self.smoothing_factor) * self.smoothed_position
        )
        
        pyautogui.moveTo(int(self.smoothed_position[0]), int(self.smoothed_position[1]))
        self.state.x = int(self.smoothed_position[0])
        self.state.y = int(self.smoothed_position[1])

    def _convert_to_screen_coordinates(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        if self.is_calibrated:
            x_norm = x_norm * self.calib_slope[0] + self.calib_intercept[0]
            y_norm = y_norm * self.calib_slope[1] + self.calib_intercept[1]

        screen_x = np.interp(x_norm, [self.frame_margin, 1 - self.frame_margin], [0, self.screen_width])
        screen_y = np.interp(y_norm, [self.frame_margin, 1 - self.frame_margin], [0, self.screen_height])
        return screen_x, screen_y

    def calibrate(self, calibration_data: dict) -> bool:
        try:
            points = calibration_data.get('points', [])
            if len(points) < 2: return False
            
            hand_x = np.array([p['hand_x'] for p in points])
            hand_y = np.array([p['hand_y'] for p in points])
            target_x = np.array([p['screen_x'] / self.screen_width for p in points])
            target_y = np.array([p['screen_y'] / self.screen_height for p in points])
            
            slope_x, intercept_x = np.polyfit(hand_x, target_x, 1)
            slope_y, intercept_y = np.polyfit(hand_y, target_y, 1)
            
            self.calib_slope = np.array([slope_x, slope_y])
            self.calib_intercept = np.array([intercept_x, intercept_y])
            self.is_calibrated = True
            return True
        except Exception:
            return False