"""
Configuration Loader
Handles loading, validation, and saving of configuration files.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any
import copy

class ConfigLoader:
    """Static class for configuration management"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logging.error(f"Configuration file not found: {config_path}")
            return ConfigLoader._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate and merge with default config
            default_config = ConfigLoader._get_default_config()
            config = ConfigLoader._merge_configs(default_config, config)
            
            return config
            
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return ConfigLoader._get_default_config()

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> bool:
        """Save configuration to YAML file"""
        try:
            path = Path(config_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False

    @staticmethod
    def _merge_configs(default: Dict, user: Dict) -> Dict:
        """Recursively merge user configuration with defaults"""
        result = copy.deepcopy(default)
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, list]:
        """Validate configuration values"""
        errors = []
        
        # Camera
        cam = config.get('camera', {})
        if cam.get('width', 0) <= 0 or cam.get('height', 0) <= 0:
            errors.append("Camera resolution must be positive")
            
        # UI
        ui = config.get('ui', {})
        # Updated to include 'neon'
        valid_schemes = ['dark', 'light', 'blue', 'green', 'neon'] 
        if ui.get('color_scheme') not in valid_schemes:
            errors.append(f"color_scheme must be one of {valid_schemes}")
            
        return (len(errors) == 0), errors

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'camera': {
                'device_id': 0, 'width': 1280, 'height': 720, 'fps': 30,
                'flip_horizontal': True, 'autofocus': True
            },
            'hand_detection': {
                'model_complexity': 1, 'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.5, 'max_num_hands': 1
            },
            'mouse': {
                'smoothing_factor': 0.25, 'scroll_sensitivity': 50,
                'enable_acceleration': True
            },
            'gestures': {
                'cooldown_period': 0.5, 'confidence_threshold': 0.8
            },
            'ui': {
                'show_fps': True, 'show_landmarks': True, 'show_gesture_info': True,
                'show_hand_box': True, 'color_scheme': 'neon'
            },
            'calibration': {
                'samples_required': 30, 'auto_calibrate': True, 'calibration_timeout': 10
            }
        }