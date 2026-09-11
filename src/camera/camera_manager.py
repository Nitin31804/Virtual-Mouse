"""
Camera Manager
Handles camera initialization, frame capture, and camera settings.
Uses threading to prevent I/O blocking in the main application loop.
"""

import cv2
import logging
import time
from threading import Thread, Lock
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class CameraConfig:
    """Camera configuration data class"""

    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_horizontal: bool = True
    brightness: float = 0.5
    contrast: float = 0.5
    exposure: float = -1.0
    autofocus: bool = True
    inference_fps: int = 30 # User choice for inference limit
    performance_mode: str = "auto" # "high", "balanced", "battery", or "auto"


class CameraManager:
    def __init__(self, config: dict):
        """
        Initialize camera manager with configuration

        Args:
            config: Camera configuration dictionary
        """
        # Unpack dictionary to dataclass, ignoring extra keys safely
        valid_keys = CameraConfig.__annotations__.keys()
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        self.config = CameraConfig(**filtered_config)

        if "camera_index" in config and "device_id" not in config:
            self.config.device_id = self.config.camera_index
        elif "device_id" in config and "camera_index" not in config:
            self.config.camera_index = self.config.device_id

        self.cap = None
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.thread = None

        # Performance tracking
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0

        logging.info(f"Camera Manager initialized for device {self.config.device_id}")

    def open(self) -> bool:
        """
        Open camera connection and start capture thread

        Returns:
            bool: True if camera opened and started successfully
        """
        try:
            # 1. Open Camera with Fallback and Frame Validation
            devices_to_try = [self.config.device_id] + [i for i in range(3) if i != self.config.device_id]
            for idx in devices_to_try:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    # Validate it is a real camera, not a 'ghost' virtual driver
                    ret, _ = self.cap.read()
                    if ret:
                        self.config.device_id = idx
                        break
                    else:
                        self.cap.release()

            if not self.cap or not self.cap.isOpened():
                logging.error(f"Failed to open any camera device. Tried: {devices_to_try}")
                return False

            # 2. Apply Settings
            self._apply_camera_settings()

            # 3. Test Capture (Blocking)
            ret, _ = self.cap.read()
            if not ret:
                logging.error("Camera opened but failed to capture initial frame")
                self.cap.release()
                return False

            # 4. Start Background Thread
            self.running = True
            self.thread = Thread(target=self._capture_frames, daemon=True)
            self.thread.start()

            # 5. Wait for first threaded frame (Better than time.sleep)
            start_wait = time.time()
            while self.frame is None:
                if time.time() - start_wait > 2.0:
                    logging.error("Timeout waiting for camera thread to produce frames")
                    self.release()
                    return False
                time.sleep(0.01)

            logging.info(
                f"Camera started successfully: {self.config.width}x{self.config.height} @ {self.config.fps}FPS"
            )
            return True

        except Exception as e:
            logging.error(f"Error opening camera: {e}")
            self.release()
            return False

    def set_performance_mode(self, mode: str):
        """
        Dynamically adjusts the inference FPS cap based on user choice.
        'high' = 60 FPS (Gaming PCs)
        'balanced' = 30 FPS (Standard laptops)
        'battery' = 15 FPS (Low power devices)
        'auto' = Scales based on actual hardware capability
        """
        mode = mode.lower()
        if mode == "high":
            self.config.inference_fps = min(60, self.config.fps)
        elif mode == "balanced":
            self.config.inference_fps = min(30, self.config.fps)
        elif mode == "battery":
            self.config.inference_fps = min(15, self.config.fps)
        elif mode == "auto":
            # Auto mode caps inference to whatever the actual hardware is achieving
            self.config.inference_fps = self.fps if self.fps > 0 else 30
            
        self.config.performance_mode = mode
        logging.info(f"Performance mode set to: {mode} (Capped at {self.config.inference_fps} FPS)")

    def _apply_camera_settings(self):
        """Apply camera configuration settings safely"""
        if not self.cap:
            return

        try:
            # Resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Image adjustments (Hardware dependent)
            if self.config.brightness is not None:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)

            if self.config.contrast is not None:
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)

            # Exposure (-1 usually triggers auto, but backend specific)
            if self.config.exposure >= 0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)

            # Autofocus
            if self.config.autofocus:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        except Exception as e:
            logging.warning(f"Could not set some camera properties: {e}")

    def _capture_frames(self):
        """Background thread for continuous frame capture"""
        self.last_frame_time = time.time()

        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()

                if ret:
                    with self.lock:
                        self.frame = frame
                        self.frame_count += 1

                        # Calculate Camera Hardware FPS
                        current_time = time.time()
                        if current_time - self.last_frame_time >= 1.0:
                            self.fps = self.frame_count
                            self.frame_count = 0
                            self.last_frame_time = current_time
                else:
                    logging.warning("Camera stream interrupted")
                    time.sleep(0.1)

            except Exception as e:
                logging.error(f"Error in frame capture thread: {e}")
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[object]]:
        """
        Get the latest frame from the buffer, mathematically throttled by performance mode.
        """
        with self.lock:
            if self.frame is not None:
                # Dynamic Inference Throttling based on user config
                if self.config.performance_mode == "auto" and self.fps > 0:
                    target_fps = min(self.fps, 30) # Don't exceed 30 in auto to save battery
                else:
                    target_fps = self.config.inference_fps

                current_time = time.time()
                if not hasattr(self, 'last_inference_time'):
                    self.last_inference_time = 0
                
                # If we haven't waited long enough, skip inference!
                if (current_time - self.last_inference_time) < (1.0 / target_fps):
                    return False, None
                
                self.last_inference_time = current_time

                # IMPORTANT: We copy the frame to ensure thread safety
                # while the main loop processes it.
                frame = self.frame.copy()

                # Apply Mirroring Here (As per Architecture)
                if self.config.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                return True, frame

        return False, None

    def get_camera_info(self) -> dict:
        """Get diagnostic information"""
        return {
            "device_id": self.config.device_id,
            "resolution": f"{self.config.width}x{self.config.height}",
            "target_fps": self.config.fps,
            "actual_fps": self.fps,
            "is_running": self.running,
        }

    def release(self):
        """Release camera resources safely"""
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()

        logging.info("Camera resources released")

    def __del__(self):
        self.release()


if __name__ == "__main__":
    # Simple test to verify camera works standalone
    logging.basicConfig(level=logging.INFO)
    cam = CameraManager({"device_id": 0, "flip_horizontal": True})
    if cam.open():
        try:
            while True:
                ret, frame = cam.read()
                if ret:
                    cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
