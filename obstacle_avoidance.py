#!/usr/bin/env python3
"""
M.A.R. — Reactive Navigation & Obstacle Avoidance Decision Engine
Evaluates spatial risks across 3 visual sectors (Left, Center, Right), fuses
stereo vision with ultrasonic telemetry, and generates both M.A.R. protocol
and legacy commands for the Arduino motor controller.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    from ..config import NavigationConfig
    from ..detector.yolo_detector import DetectedObject
except (ImportError, ValueError):
    from config import NavigationConfig
    from detector.yolo_detector import DetectedObject


@dataclass
class NavigationDecision:
    """Represents a navigation decision ready for transmission to the microcontroller."""
    mar_command: str           # E.g. "CMD:FORWARD,50", "CMD:STOP", "CMD:TURN_L,30"
    legacy_char: str           # E.g. 'F', 'S', 'L', 'R'
    action_type: str           # 'FORWARD', 'STOP', 'TURN_LEFT', 'TURN_RIGHT', 'SLOW'
    description: str           # Human-readable explanation
    risk_level: int            # 0=Safe, 1=Caution, 2=Warning, 3=Danger, 4=Emergency
    risk_score: float          # 0.0 to 4.0
    most_dangerous_object: Optional[DetectedObject] = None


class ObstacleAvoidanceEngine:
    """
    Evaluates obstacles detected by YOLO and ultrasonic sensor to select safe robot actions.
    """

    def __init__(self, config: NavigationConfig):
        self.config = config

    def evaluate_object_risk(
        self,
        obj: DetectedObject,
        ultrasonic_distance_cm: float = -1.0
    ) -> float:
        """
        Computes a unified risk score (0.0 to 4.0) for an object based on:
        1. Visual area size in frame
        2. Stereo distance (if available)
        3. Ultrasonic distance (if object is central)
        """
        # Visual risk by bounding box ratio
        v_size = obj.relative_size
        if v_size >= self.config.visual_emergency_ratio:
            visual_risk = 4.0
        elif v_size >= self.config.visual_danger_ratio:
            visual_risk = 3.0
        elif v_size >= self.config.visual_warning_ratio:
            visual_risk = 2.0
        elif v_size >= self.config.visual_caution_ratio:
            visual_risk = 1.0
        else:
            visual_risk = 0.5

        # Distance risk by stereo depth or ultrasonic sensor
        dist_cm = None
        if obj.estimated_distance_cm is not None and obj.estimated_distance_cm > 0:
            dist_cm = obj.estimated_distance_cm
        elif obj.in_danger_zone and ultrasonic_distance_cm > 0:
            dist_cm = ultrasonic_distance_cm

        if dist_cm is not None:
            if dist_cm <= self.config.emergency_distance_cm:
                dist_risk = 4.0
            elif dist_cm <= self.config.danger_distance_cm:
                dist_risk = 3.0
            elif dist_cm <= self.config.warning_distance_cm:
                dist_risk = 2.0
            elif dist_cm <= self.config.safe_distance_cm:
                dist_risk = 1.0
            else:
                dist_risk = 0.0

            # Fuse with ultrasonic weighting
            fused_risk = (
                dist_risk * self.config.ultrasonic_weight +
                visual_risk * (1.0 - self.config.ultrasonic_weight)
            )
        else:
            fused_risk = visual_risk

        # Assign back to object
        obj.danger_level = int(round(fused_risk))
        danger_names = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "DANGER", 4: "EMERGENCY"}
        obj.danger_name = danger_names.get(obj.danger_level, "WARNING")

        return float(fused_risk)

    def decide(
        self,
        all_objects: List[DetectedObject],
        frame_width: int,
        ultrasonic_distance_cm: float = -1.0
    ) -> NavigationDecision:
        """
        Decides the best robot action based on the sector analysis of all obstacles.
        """
        # Hardware emergency safety override from ultrasonic sensor
        if 0 < ultrasonic_distance_cm <= self.config.emergency_distance_cm:
            return NavigationDecision(
                mar_command="CMD:STOP",
                legacy_char="S",
                action_type="STOP",
                description=f"EMERGENZA ULTRASUONI: Ostacolo a {ultrasonic_distance_cm:.1f} cm!",
                risk_level=4,
                risk_score=4.0,
                most_dangerous_object=None
            )

        if not all_objects:
            return NavigationDecision(
                mar_command="CMD:FORWARD,50",
                legacy_char="F",
                action_type="FORWARD",
                description="Percorso libero: avanzamento nominale",
                risk_level=0,
                risk_score=0.0,
                most_dangerous_object=None
            )

        # Danger zone boundaries
        zone_left = int(frame_width * (1.0 - self.config.danger_zone_width_ratio) / 2.0)
        zone_right = int(frame_width * (1.0 + self.config.danger_zone_width_ratio) / 2.0)

        # Classify objects into sectors and update danger levels
        danger_zone_objects: List[Tuple[DetectedObject, float]] = []
        left_sector_risk = 0.0
        right_sector_risk = 0.0

        for obj in all_objects:
            cx, cy = obj.center
            obj.in_danger_zone = (zone_left <= cx <= zone_right)
            risk = self.evaluate_object_risk(obj, ultrasonic_distance_cm)

            if obj.in_danger_zone:
                danger_zone_objects.append((obj, risk))

            # Left vs Right sector risk
            if cx < (frame_width // 2):
                left_sector_risk += risk
            else:
                right_sector_risk += risk

        if not danger_zone_objects:
            return NavigationDecision(
                mar_command="CMD:FORWARD,50",
                legacy_char="F",
                action_type="FORWARD",
                description="Nessun ostacolo nel corridoio centrale di avanzamento",
                risk_level=0,
                risk_score=0.0,
                most_dangerous_object=None
            )

        # Find the most dangerous central object
        danger_zone_objects.sort(key=lambda item: item[1], reverse=True)
        most_dangerous, max_risk = danger_zone_objects[0]

        # Decision matrix based on risk level
        if max_risk >= 3.5:
            # Imminent collision -> EMERGENCY STOP
            return NavigationDecision(
                mar_command="CMD:STOP",
                legacy_char="S",
                action_type="STOP",
                description=f"EMERGENZA: '{most_dangerous.label}' in rotta critica!",
                risk_level=4,
                risk_score=max_risk,
                most_dangerous_object=most_dangerous
            )

        elif max_risk >= 2.5:
            # Dangerous obstacle -> Evade away from the obstacle
            obj_cx = most_dangerous.center[0]
            mid_x = frame_width // 2

            if obj_cx <= mid_x:
                # Obstacle is on the left half -> Evade right
                return NavigationDecision(
                    mar_command="CMD:TURN_R,30",
                    legacy_char="R",
                    action_type="TURN_RIGHT",
                    description=f"EVASIONE DESTRA: Ostacolo '{most_dangerous.label}' a sinistra",
                    risk_level=3,
                    risk_score=max_risk,
                    most_dangerous_object=most_dangerous
                )
            else:
                # Obstacle is on the right half -> Evade left
                return NavigationDecision(
                    mar_command="CMD:TURN_L,30",
                    legacy_char="L",
                    action_type="TURN_LEFT",
                    description=f"EVASIONE SINISTRA: Ostacolo '{most_dangerous.label}' a destra",
                    risk_level=3,
                    risk_score=max_risk,
                    most_dangerous_object=most_dangerous
                )

        elif max_risk >= 1.5:
            # Warning -> Slow down
            return NavigationDecision(
                mar_command="CMD:FORWARD,25",
                legacy_char="F",
                action_type="SLOW",
                description=f"RALLENTA: '{most_dangerous.label}' rilevato a media distanza",
                risk_level=2,
                risk_score=max_risk,
                most_dangerous_object=most_dangerous
            )

        else:
            return NavigationDecision(
                mar_command="CMD:FORWARD,45",
                legacy_char="F",
                action_type="FORWARD",
                description="Attenzione: piccolo ostacolo periferico, prosegui",
                risk_level=1,
                risk_score=max_risk,
                most_dangerous_object=most_dangerous
            )
