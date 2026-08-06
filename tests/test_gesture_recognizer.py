"""
Tests for Gesture Recognizer module
"""

import pytest
import time
from unittest.mock import Mock, patch
from src.gestures.gesture_recognizer import (
    GestureRecognizer, Gesture, GestureHistory
)

class TestGesture:
    """Test Gesture dataclass"""
    
    def test_creation(self):
        """Test creating a gesture"""
        # Note: 'type' field was removed in favor of dynamic string names
        gesture = Gesture(
            name="cursor_move",
            finger_state=[0, 1, 0, 0, 0],
            confidence=0.95,
            timestamp=time.time(),
            duration=1.5
        )
        
        assert gesture.name == "cursor_move"
        assert gesture.finger_state == [0, 1, 0, 0, 0]
        assert gesture.confidence == 0.95
        assert isinstance(gesture.timestamp, float)
        assert gesture.duration == 1.5
    
    def test_str_representation(self):
        """Test string representation of gesture"""
        gesture = Gesture(
            name="left_click",
            finger_state=[0, 1, 1, 0, 0],
            confidence=0.87,
            timestamp=time.time()
        )
        
        # Format defined in Gesture.__str__: f"{self.name} ({self.confidence:.2f})"
        expected_str = "left_click (0.87)"
        assert str(gesture) == expected_str

class TestGestureHistory:
    """Test GestureHistory class"""
    
    @pytest.fixture
    def gesture_history(self):
        return GestureHistory(max_history=5)
    
    @pytest.fixture
    def sample_gestures(self):
        return [
            Gesture(name="cursor_move", finger_state=[0, 1, 0, 0, 0], confidence=0.9),
            Gesture(name="left_click", finger_state=[0, 1, 1, 0, 0], confidence=0.85),
            Gesture(name="cursor_move", finger_state=[0, 1, 0, 0, 0], confidence=0.88),
        ]
    
    def test_initialization(self, gesture_history):
        assert gesture_history.max_history == 5
        assert gesture_history.history == []
        assert gesture_history.current_gesture is None
        assert gesture_history.gesture_start_time == 0.0
    
    def test_add_gesture(self, gesture_history, sample_gestures):
        """Test adding gestures to history"""
        for gesture in sample_gestures:
            gesture_history.add(gesture)
        
        assert len(gesture_history.history) == 3
        assert gesture_history.current_gesture.name == "cursor_move"
    
    def test_add_same_gesture_duration(self, gesture_history):
        """Test that adding the same gesture updates its duration"""
        gesture = Gesture(
            name="cursor_move",
            finger_state=[0, 1, 0, 0, 0],
            confidence=0.9
        )
        
        # First add
        gesture_history.add(gesture)
        start_time = gesture_history.gesture_start_time
        
        # Wait a bit (mocked or real)
        time.sleep(0.01)
        
        # Second add (same name)
        gesture2 = Gesture(
            name="cursor_move",
            finger_state=[0, 1, 0, 0, 0],
            confidence=0.95
        )
        gesture_history.add(gesture2)
        
        assert len(gesture_history.history) == 2
        assert gesture_history.current_gesture.duration > 0
        assert gesture_history.gesture_start_time == start_time
    
    def test_history_size_limit(self, gesture_history):
        """Test that history doesn't exceed max size"""
        for i in range(10):
            gesture = Gesture(
                name=f"gesture_{i}",
                finger_state=[0, 1, 0, 0, 0],
                confidence=0.9
            )
            gesture_history.add(gesture)
        
        assert len(gesture_history.history) == gesture_history.max_history
        # Should contain the last 5
        assert gesture_history.history[-1].name == "gesture_9"
    
    def test_get_most_common(self, gesture_history, sample_gestures):
        """Test smoothing logic"""
        for gesture in sample_gestures:
            gesture_history.add(gesture)
        
        # cursor_move appears twice, left_click once
        most_common = gesture_history.get_most_common(recent_count=3)
        assert most_common.name == "cursor_move"

    def test_clear(self, gesture_history, sample_gestures):
        gesture_history.add(sample_gestures[0])
        gesture_history.clear()
        
        assert gesture_history.history == []
        assert gesture_history.current_gesture is None

class TestGestureRecognizer:
    """Test GestureRecognizer class"""
    
    @pytest.fixture
    def mock_gestures_config(self):
        return {
            "gestures": {
                "cursor_move": {
                    "finger_state": [0, 1, 0, 0, 0],
                    "priority": 1
                },
                "left_click": {
                    "finger_state": [0, 1, 1, 0, 0],
                    "priority": 2
                }
            }
        }
    
    @pytest.fixture
    def mock_settings(self):
        return {
            "gesture_buffer_size": 10,
            "confidence_threshold": 0.7,
            "cooldown_period": 0.5
        }
    
    @pytest.fixture
    def gesture_recognizer(self, mock_gestures_config, mock_settings):
        return GestureRecognizer(mock_gestures_config, mock_settings)
    
    @pytest.fixture
    def mock_hand(self):
        hand = Mock()
        hand.get_finger_state.return_value = [0, 1, 0, 0, 0]
        
        # Create Dummy Landmarks for physics check (Z-depth)
        wrist = Mock(x=0.5, y=0.9, z=0.0)
        index_tip = Mock(x=0.5, y=0.5, z=0.0) # Flat hand (low z-diff)
        middle_tip = Mock(x=0.6, y=0.5, z=0.0)
        
        def get_lm(idx):
            if idx == 0: return wrist
            if idx == 8: return index_tip
            if idx == 12: return middle_tip
            return None
            
        hand.get_landmark_by_id.side_effect = get_lm
        return hand
    
    def test_initialization(self, gesture_recognizer):
        """Test config loading"""
        assert "cursor_move" in gesture_recognizer.gesture_definitions
        assert "left_click" in gesture_recognizer.gesture_definitions
        assert gesture_recognizer.min_confidence == 0.7

    def test_recognize_no_hand(self, gesture_recognizer):
        assert gesture_recognizer.recognize(None) is None

    def test_recognize_success(self, gesture_recognizer, mock_hand):
        """Test basic recognition"""
        # mock_hand has [0, 1, 0, 0, 0] which matches cursor_move
        result = gesture_recognizer.recognize(mock_hand)
        
        # It needs to pass smoothing (history buffer)
        # Add a few more consistent frames to trigger smoothing result
        gesture_recognizer.recognize(mock_hand)
        gesture_recognizer.recognize(mock_hand)
        result = gesture_recognizer.recognize(mock_hand)
        
        assert result == "cursor_move"

    def test_recognize_fail_score(self, gesture_recognizer, mock_hand):
        """Test recognition failure due to wrong finger state"""
        # Hand state [1, 1, 1, 1, 1] matches nothing in our mock config
        mock_hand.get_finger_state.return_value = [1, 1, 1, 1, 1]
        
        result = gesture_recognizer.recognize(mock_hand)
        assert result is None

    def test_validate_physics_fail(self, gesture_recognizer, mock_hand):
        """Test the physics validation (Z-depth check)"""
        # Simulate a tilted hand (Wrist Z far from Index Z)
        wrist = Mock(z=0.0)
        index_tip = Mock(z=0.5) # 0.5 diff > 0.3 threshold
        
        def get_lm_tilted(idx):
            if idx == 0: return wrist
            if idx == 8: return index_tip
            return None
        
        mock_hand.get_landmark_by_id.side_effect = get_lm_tilted
        mock_hand.get_finger_state.return_value = [0, 1, 1, 0, 0] # Matches left_click
        
        # We access the private method to test it specifically
        # (Though integration testing via recognize() is preferred, this confirms logic)
        is_valid = gesture_recognizer._validate_physics("left_click", mock_hand)
        assert is_valid is False

if __name__ == '__main__':
    pytest.main([__file__, '-v'])