"""
Tests for Hand Tracker module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.hand_detection.hand_tracker import (
    HandTracker, Hand, HandLandmark, Handedness
)

class TestHandLandmark:
    """Test HandLandmark dataclass"""
    
    def test_creation(self):
        """Test creating a hand landmark"""
        landmark = HandLandmark(x=0.5, y=0.5, z=0.1, visibility=0.9)
        
        assert landmark.x == 0.5
        assert landmark.y == 0.5
        assert landmark.z == 0.1
        assert landmark.visibility == 0.9
    
    def test_to_pixel(self):
        """Test converting normalized coordinates to pixel coordinates"""
        landmark = HandLandmark(x=0.25, y=0.75, z=0.0, visibility=1.0)
        
        pixel_coords = landmark.to_pixel(1920, 1080)
        
        assert pixel_coords == (480, 810)  # 1920*0.25 = 480, 1080*0.75 = 810
    
    def test_to_pixel_edge_cases(self):
        """Test pixel conversion with edge cases"""
        # Test minimum values
        landmark = HandLandmark(x=0.0, y=0.0, z=0.0, visibility=1.0)
        assert landmark.to_pixel(1920, 1080) == (0, 0)
        
        # Test maximum values
        landmark = HandLandmark(x=1.0, y=1.0, z=0.0, visibility=1.0)
        assert landmark.to_pixel(1920, 1080) == (1920, 1080)

class TestHand:
    """Test Hand dataclass"""
    
    @pytest.fixture
    def sample_landmarks(self):
        """Create sample hand landmarks"""
        landmarks = []
        for i in range(21):  # MediaPipe has 21 landmarks
            x = i * 0.05
            y = (i % 5) * 0.05
            landmarks.append(HandLandmark(x=x, y=y, z=0.0, visibility=1.0))
        return landmarks
    
    @pytest.fixture
    def sample_hand(self, sample_landmarks):
        """Create a sample hand"""
        return Hand(
            landmarks=sample_landmarks,
            handedness=Handedness.RIGHT,
            bounding_box=(0.1, 0.1, 0.9, 0.9),
            center=(0.5, 0.5),
            confidence=0.95
        )
    
    def test_creation(self, sample_hand):
        """Test creating a hand"""
        assert len(sample_hand.landmarks) == 21
        assert sample_hand.handedness == Handedness.RIGHT
        assert sample_hand.bounding_box == (0.1, 0.1, 0.9, 0.9)
        assert sample_hand.center == (0.5, 0.5)
        assert sample_hand.confidence == 0.95
    
    def test_get_landmark_by_id(self, sample_hand):
        """Test getting landmark by ID"""
        # Test valid ID
        landmark = sample_hand.get_landmark_by_id(0)
        assert landmark is not None
        assert landmark.x == 0.0  # First landmark
        
        # Test invalid ID
        landmark = sample_hand.get_landmark_by_id(100)
        assert landmark is None
    
    def test_get_finger_tip(self, sample_hand):
        """Test getting finger tip landmarks"""
        # MediaPipe fingertip indices: [4, 8, 12, 16, 20]
        finger_tips = [
            sample_hand.get_finger_tip(0),  # Thumb
            sample_hand.get_finger_tip(1),  # Index
            sample_hand.get_finger_tip(2),  # Middle
            sample_hand.get_finger_tip(3),  # Ring
            sample_hand.get_finger_tip(4),  # Pinky
        ]
        
        # All should exist
        assert all(tip is not None for tip in finger_tips)
        
        # Check indices
        assert finger_tips[0] == sample_hand.get_landmark_by_id(4)
        assert finger_tips[1] == sample_hand.get_landmark_by_id(8)
        
        # Test invalid finger index
        assert sample_hand.get_finger_tip(5) is None
    
    def test_calculate_palm_center(self, sample_hand):
        """Test calculating palm center"""
        center = sample_hand.calculate_palm_center()
        
        # Should return a tuple of floats
        assert isinstance(center, tuple)
        assert len(center) == 2
        assert all(isinstance(c, float) for c in center)
    
    def test_get_finger_state_extended(self):
        """Test finger state detection with extended fingers"""
        # Create hand with all fingers extended
        landmarks = []
        for i in range(21):
            # For extended fingers: fingertip y < PIP joint y (assuming hand is upright)
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                landmarks.append(HandLandmark(x=0.5, y=0.3, z=0.0, visibility=1.0))
            elif i in [3, 7, 11, 15, 19]:  # PIP joints
                landmarks.append(HandLandmark(x=0.5, y=0.4, z=0.0, visibility=1.0))
            else:
                landmarks.append(HandLandmark(x=0.5, y=0.5, z=0.0, visibility=1.0))
        
        hand = Hand(
            landmarks=landmarks,
            handedness=Handedness.RIGHT,
            bounding_box=(0.0, 0.0, 1.0, 1.0),
            center=(0.5, 0.5),
            confidence=0.9
        )
        
        finger_state = hand.get_finger_state(threshold=0.5)
        
        # All fingers should be extended (1)
        assert finger_state == [1, 1, 1, 1, 1]
    
    def test_get_finger_state_folded(self):
        """Test finger state detection with folded fingers"""
        # Create hand with all fingers folded
        landmarks = []
        for i in range(21):
            # For folded fingers: fingertip y > PIP joint y
            if i in [4, 8, 12, 16, 20]:  # Fingertips
                landmarks.append(HandLandmark(x=0.5, y=0.6, z=0.0, visibility=1.0))
            elif i in [3, 7, 11, 15, 19]:  # PIP joints
                landmarks.append(HandLandmark(x=0.5, y=0.4, z=0.0, visibility=1.0))
            else:
                landmarks.append(HandLandmark(x=0.5, y=0.5, z=0.0, visibility=1.0))
        
        hand = Hand(
            landmarks=landmarks,
            handedness=Handedness.RIGHT,
            bounding_box=(0.0, 0.0, 1.0, 1.0),
            center=(0.5, 0.5),
            confidence=0.9
        )
        
        finger_state = hand.get_finger_state(threshold=0.5)
        
        # All fingers should be folded (0)
        assert finger_state == [0, 0, 0, 0, 0]

class TestHandTracker:
    """Test HandTracker class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        return {
            'model_complexity': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'max_num_hands': 1,
            'static_image_mode': False
        }
    
    @pytest.fixture
    def hand_tracker(self, mock_config):
        """Create HandTracker instance with mocked mediapipe"""
        with patch('mediapipe.solutions.hands.Hands'):
            return HandTracker(mock_config)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        return np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    def test_initialization(self, hand_tracker, mock_config):
        """Test hand tracker initialization"""
        assert hand_tracker.config == mock_config
        assert hand_tracker.hands is not None
        assert hand_tracker.frame_count == 0
    
    def test_detect_empty_image(self, hand_tracker):
        """Test detection with empty image"""
        empty_image = np.array([])
        result = hand_tracker.detect(empty_image)
        
        assert result == []
    
    def test_get_hand_size(self, hand_tracker, sample_hand):
        """Test calculating hand size"""
        size = hand_tracker.get_hand_size(sample_hand)
        
        assert isinstance(size, float)
        # Fix: Diagonal of 1x1 bounding box is sqrt(2) approx 1.414
        assert 0 <= size <= 1.5
    
    def test_release(self, hand_tracker):
        """Test releasing resources"""
        # Mock the hands.close method
        hand_tracker.hands.close = Mock()
        
        hand_tracker.release()
        
        hand_tracker.hands.close.assert_called_once()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])