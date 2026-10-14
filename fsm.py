#!/usr/bin/env python3
"""
M.A.R. — Finite State Machine (FSM) & Safety Engine
Manages robot behavioral states: IDLE, PATROL_FORWARD, INSPECTING_OBJECT,
AVOIDANCE, and EMERGENCY_STOP.
"""

from enum import Enum, auto
import time
from typing import Dict, Optional, Tuple


class RobotState(Enum):
    IDLE = auto()
    PATROL_FORWARD = auto()
    INSPECTING_OBJECT = auto()
    AVOIDANCE = auto()
    EMERGENCY_STOP = auto()


class DecisionFSM:
    """Deterministic behavioral state machine for M.A.R. quadrupede."""

    def __init__(self, forward_speed: int = 50, critical_distance_m: float = 0.30):
        self.state = RobotState.IDLE
        self.forward_speed = forward_speed
        self.critical_distance_m = critical_distance_m
        self.state_enter_time = time.time()
        self.target_theta: float = 0.0
        self.avoidance_direction: str = "LEFT"
        self.avoidance_start_time: float = 0.0

    def set_state(self, new_state: RobotState, reason: str = ""):
        if self.state != new_state:
            print(f"🔄 [FSM STATE CHANGE]: {self.state.name} ➔ {new_state.name} | Motivo: {reason}")
            self.state = new_state
            self.state_enter_time = time.time()

    def update(
        self,
        lidar_alert: Optional[Dict],
        vision_objects: Optional[Dict],
        ultrasonic_distance_cm: float = -1.0
    ) -> Tuple[str, str]:
        """
        Processes multi-sensor inputs and produces the next motion command.
        Returns: (CMD_STRING, REASON_DESCRIPTION)
        """
        now = time.time()

        # 1. Regola di Emergenza Prioritaria (Sensor fusion)
        is_critically_close = False
        if 0 < ultrasonic_distance_cm < (self.critical_distance_m * 100):
            is_critically_close = True
        if lidar_alert and lidar_alert.get("dist_m", 99) < self.critical_distance_m:
            is_critically_close = True
        if vision_objects:
            for obj in vision_objects.get("objects", []):
                dist = obj.get("distance_cm")
                if dist and dist < (self.critical_distance_m * 100):
                    is_critically_close = True
                    break

        if is_critically_close and self.state != RobotState.EMERGENCY_STOP:
            self.set_state(RobotState.EMERGENCY_STOP, f"Distanza critica superata (<{self.critical_distance_m}m)")

        # 2. Logica di transizione per ciascuno stato
        if self.state == RobotState.IDLE:
            return "CMD:STAND,180", "Robot in attesa (assetto eretto)"

        elif self.state == RobotState.EMERGENCY_STOP:
            return "CMD:STOP", "Arresto d'emergenza di sicurezza"

        elif self.state == RobotState.PATROL_FORWARD:
            if lidar_alert and lidar_alert.get("has_alert", False):
                self.target_theta = lidar_alert.get("theta", 0.0)
                self.set_state(RobotState.INSPECTING_OBJECT, f"LiDAR ha rilevato ostacolo a θ={self.target_theta}°")
                return f"CMD:TURN_R,{abs(int(self.target_theta))}" if self.target_theta > 0 else f"CMD:TURN_L,{abs(int(self.target_theta))}", "Puntamento saccadico verso ostacolo"

            return f"CMD:FORWARD,{self.forward_speed}", "Avanzamento di pattugliamento continuo"

        elif self.state == RobotState.INSPECTING_OBJECT:
            # Timeout ispezione: se dopo 1.5s non ci sono pericoli, riprendi
            if now - self.state_enter_time > 1.5:
                # Verifica se la visione ha confermato ostacoli vicini
                has_confirmed_obstacle = False
                if vision_objects and vision_objects.get("count", 0) > 0:
                    for obj in vision_objects.get("objects", []):
                        d = obj.get("distance_cm")
                        if d and d < 150:
                            has_confirmed_obstacle = True
                            break

                if has_confirmed_obstacle:
                    self.avoidance_direction = "LEFT" if self.target_theta >= 0 else "RIGHT"
                    self.avoidance_start_time = now
                    self.set_state(RobotState.AVOIDANCE, f"Ostacolo confermato dalla visione 3D. Schivamento verso {self.avoidance_direction}")
                else:
                    self.set_state(RobotState.PATROL_FORWARD, "Falso allarme / Area libera verificata da YOLO")

            return "CMD:STOP", "Ispezione visiva 3D mirata in corso (Saccade)"

        elif self.state == RobotState.AVOIDANCE:
            # Esegui manovra di svolta per 1.2 secondi poi riprendi
            if now - self.avoidance_start_time > 1.2:
                self.set_state(RobotState.PATROL_FORWARD, "Schivamento completato con successo")
                return f"CMD:FORWARD,{self.forward_speed}", "Ripresa marcia in avanti"

            turn_cmd = "CMD:TURN_L,40" if self.avoidance_direction == "LEFT" else "CMD:TURN_R,40"
            return turn_cmd, f"Manovra di schivamento attivo [{self.avoidance_direction}]"

        return "CMD:STOP", "Stato indefinito"
