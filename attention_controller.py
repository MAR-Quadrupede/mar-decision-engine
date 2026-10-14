#!/usr/bin/env python3
"""
M.A.R. — Active Perception & Attention Saccade Controller
Calculates gaze orientation angles and triggers target inspection.
"""

import time
from typing import Optional, Dict


class AttentionController:
    """Manages active perception orienting and motion blur gating."""

    def __init__(self, settle_time_sec: float = 0.3):
        self.settle_time_sec = settle_time_sec
        self.last_saccade_time = 0.0
        self.is_orienting = False

    def request_saccade(self, target_theta_deg: float) -> Dict:
        """Calculates gaze turn angle and timestamp."""
        self.last_saccade_time = time.time()
        self.is_orienting = True
        return {
            "theta_deg": target_theta_deg,
            "timestamp": self.last_saccade_time,
            "settle_until": self.last_saccade_time + self.settle_time_sec
        }

    def is_camera_settled(self) -> bool:
        """Returns True if the robot/camera has finished moving and vision can capture clean frames."""
        if not self.is_orienting:
            return True
        if time.time() >= (self.last_saccade_time + self.settle_time_sec):
            self.is_orienting = False
            return True
        return False
