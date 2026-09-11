"""
Tests for Camera Manager module
"""

import pytest
import cv2
import numpy as np
import threading
import time
from unittest.mock import Mock, patch
from src.camera.camera_manager import CameraManager, CameraConfig


class TestCameraConfig:
    """Test CameraConfig dataclass"""

    def test_default_values(self):
        config = CameraConfig()
        assert config.device_id == 0
        assert config.width == 1280
        assert config.fps == 30
        assert config.flip_horizontal is True

    def test_custom_values(self):
        config = CameraConfig(
            device_id=1, width=1920, height=1080, fps=60, flip_horizontal=False
        )
        assert config.device_id == 1
        assert config.width == 1920
        assert config.fps == 60
        assert config.flip_horizontal is False


class TestCameraManager:
    """Test CameraManager class"""

    @pytest.fixture
    def mock_camera_config(self):
        return {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "flip_horizontal": True,
            "brightness": 0.5,
            "contrast": 0.5,
        }

    @pytest.fixture
    def camera_manager(self, mock_camera_config):
        return CameraManager(mock_camera_config)

    @patch("cv2.VideoCapture")
    def test_successful_open(self, mock_video_capture, camera_manager):
        # Setup mock
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        # Return a black frame for the initial test read
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        def mock_start():
            camera_manager.frame = np.ones((10, 10, 3))
            camera_manager.running = True

        with patch("threading.Thread.start", side_effect=mock_start) as mock_thread_start:
            result = camera_manager.open()

            assert result is True
            assert camera_manager.cap == mock_cap
            assert camera_manager.running is True
            mock_thread_start.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_failed_open(self, mock_video_capture, camera_manager):
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        result = camera_manager.open()
        assert result is False

    def test_read_with_flip(self, camera_manager):
        """Test read with horizontal flip"""
        camera_manager.config.flip_horizontal = True

        # Create test frame: Left Half WHITE, Right Half BLACK
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame[:, :50] = 255

        # Manually set the frame buffer
        camera_manager.frame = test_frame

        success, frame = camera_manager.read()

        assert success is True
        # After horizontal flip:
        # Left half should be BLACK (0)
        # Right half should be WHITE (255)
        assert np.all(frame[:, :50] == 0)
        assert np.all(frame[:, 50:] == 255)

    @patch("cv2.VideoCapture")
    def test_release(self, mock_video_capture, camera_manager):
        """Test releasing camera resources"""
        # Setup mock to simulate open state
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        camera_manager.cap = mock_cap
        camera_manager.running = True

        # Simulate a running thread
        # We use a real thread that checks 'self.running' so we can verify release stops it
        def dummy_loop():
            while camera_manager.running:
                time.sleep(0.01)

        camera_manager.thread = threading.Thread(target=dummy_loop)
        camera_manager.thread.start()

        assert camera_manager.thread.is_alive() is True

        # Test release
        camera_manager.release()

        # Thread should be dead now
        assert camera_manager.running is False
        assert camera_manager.thread.is_alive() is False
        mock_cap.release.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
