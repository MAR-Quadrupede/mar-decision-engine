#!/usr/bin/env python3
"""
M.A.R. — Finite State Machine (FSM) & Multi-Sensor Safety Engine
Manages robot behavioral states:
  - IDLE: robot standing or awaiting start
  - PATROL_FORWARD: clear path forward motion (AVANTI / CMD:FORWARD)
  - INSPECTING_OBJECT: saccadic active perception towards anomaly
  - TERRAIN_HAZARD: low-profile ground obstacle detected by 3D vision (STOP_TERRA)
  - AVOIDANCE: dynamic steering towards clearer side (GIRA_DX / GIRA_SX)
  - EMERGENCY_STOP: immediate stop for critical obstacles < 30 cm (STOP)
"""

from enum import Enum, auto
import time
from typing import Dict, Optional, Tuple, Any


class RobotState(Enum):
    IDLE = auto()
    PATROL_FORWARD = auto()
    INSPECTING_OBJECT = auto()
    TERRAIN_HAZARD = auto()
    AVOIDANCE = auto()
    EMERGENCY_STOP = auto()


class DecisionFSM:
    """Deterministic multi-layer behavioral state machine for M.A.R. quadrupede."""

    def __init__(
        self,
        forward_speed: int = 50,
        critical_distance_cm: float = 30.0,
        ground_hazard_cm: float = 40.0,
        slow_warning_cm: float = 65.0
    ):
        self.state = RobotState.IDLE
        self.forward_speed = forward_speed
        self.critical_distance_cm = critical_distance_cm
        self.ground_hazard_cm = ground_hazard_cm
        self.slow_warning_cm = slow_warning_cm

        self.state_enter_time = time.time()
        self.target_theta: float = 0.0
        self.avoidance_direction: str = "RIGHT"
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
        ultrasonic_distance_cm: float = -1.0,
        ground_hazard_alert: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Processes multi-sensor inputs and produces the next motion command.
        Returns: (CMD_STRING, REASON_DESCRIPTION)
        """
        now = time.time()

        # 1. Estrazione metriche LiDAR (Fronte, DX, SX)
        lidar_front_cm = 999.0
        d_dx = 999.0
        d_sx = 999.0

        if lidar_alert:
            sec_mins = lidar_alert.get("sector_mins", {})
            # Read frontal sector in cm
            f_m = lidar_alert.get("dist_m")
            if f_m is not None:
                lidar_front_cm = f_m * 100.0

            # Read Lateral clearances
            val_dx = sec_mins.get("Destra") or sec_mins.get("Front-DX") or sec_mins.get("RIGHT")
            val_sx = sec_mins.get("Sinistra") or sec_mins.get("Front-SX") or sec_mins.get("LEFT")
            if val_dx is not None:
                d_dx = float(val_dx) * 100.0
            if val_sx is not None:
                d_sx = float(val_sx) * 100.0

        # 2. Estrazione metriche Visione 3D (Z_basso, Z_centro, Z_alto)
        z_basso_cm = 999.0
        z_centro_cm = 999.0
        has_ground_hazard = False

        if vision_objects:
            grid_3d = vision_objects.get("spatial_grid_3d", {})
            if "Basso_Centro" in grid_3d:
                z_basso_cm = float(grid_3d["Basso_Centro"][2])
            elif "z_basso_cm" in vision_objects:
                z_basso_cm = float(vision_objects["z_basso_cm"])

            if "Centro_Centro" in grid_3d:
                z_centro_cm = float(grid_3d["Centro_Centro"][2])
            elif "z_centro_cm" in vision_objects:
                z_centro_cm = float(vision_objects["z_centro_cm"])

            has_ground_hazard = bool(vision_objects.get("ground_hazard", False)) or (z_basso_cm < self.ground_hazard_cm)

        if ground_hazard_alert:
            has_ground_hazard = True
            if "z_basso_cm" in ground_hazard_alert:
                z_basso_cm = min(z_basso_cm, float(ground_hazard_alert["z_basso_cm"]))

        # 3. Calcolo Distanza Critica Frontale Unificata
        # min tra LiDAR fronte, Z basso, Z centro, e sensore ultrasuoni se valido
        candidati_fronte = [lidar_front_cm, z_basso_cm, z_centro_cm]
        if ultrasonic_distance_cm > 0:
            candidati_fronte.append(ultrasonic_distance_cm)

        dist_critica_fronte = min(candidati_fronte)

        # 4. Multi-Level Safety & Transition Hierarchy
        # Livello 1: Arresto Critico Immediato (< 30 cm)
        if dist_critica_fronte < self.critical_distance_cm:
            if self.state != RobotState.EMERGENCY_STOP:
                self.set_state(RobotState.EMERGENCY_STOP, f"Ostacolo critico frontale ({dist_critica_fronte:.0f} cm < {self.critical_distance_cm} cm)")
            return "CMD:STOP", f"[STOP EMERGENCY] Ostacolo a {dist_critica_fronte:.0f}cm"

        # Livello 2: Ostacolo a Terra / Dislivello non visibile dal LiDAR
        if has_ground_hazard and (z_basso_cm < self.ground_hazard_cm):
            if self.state != RobotState.TERRAIN_HAZARD:
                self.set_state(RobotState.TERRAIN_HAZARD, f"Dislivello o scalino a terra (Z_basso={z_basso_cm:.0f} cm)")
            return "CMD:STOP_TERRA", f"[OSTACOLO A TERRA] Z_basso: {z_basso_cm:.0f}cm"

        # Se eravamo in emergency o terrain hazard e la via è ora libera, torna a marcia
        if self.state in (RobotState.EMERGENCY_STOP, RobotState.TERRAIN_HAZARD):
            if dist_critica_fronte >= self.slow_warning_cm and not has_ground_hazard:
                self.set_state(RobotState.PATROL_FORWARD, "Via libera confermata dai sensori 3D/LiDAR")

        # Livello 3: Rallentamento e Schivamento Dinamico (< 65 cm)
        if dist_critica_fronte < self.slow_warning_cm:
            # Scegli la direzione con maggiore spazio libero (Destra vs Sinistra)
            if d_dx >= d_sx:
                direction = "RIGHT"
                turn_cmd = "CMD:TURN_R,45"
            else:
                direction = "LEFT"
                turn_cmd = "CMD:TURN_L,45"

            self.avoidance_direction = direction
            self.avoidance_start_time = now
            if self.state != RobotState.AVOIDANCE:
                self.set_state(RobotState.AVOIDANCE, f"Ostacolo a {dist_critica_fronte:.0f}cm. Svolta verso {direction} (DX={d_dx:.0f}cm, SX={d_sx:.0f}cm)")

            return turn_cmd, f"[DEVIAZIONE {direction}] DX:{d_dx:.0f}cm | SX:{d_sx:.0f}cm"

        # Livello 4: Puntamento saccadico LiDAR (se allarme angolare laterale)
        if lidar_alert and lidar_alert.get("has_alert", False) and abs(lidar_alert.get("theta", 0)) > 20:
            if self.state == RobotState.PATROL_FORWARD:
                self.target_theta = lidar_alert.get("theta", 0.0)
                self.set_state(RobotState.INSPECTING_OBJECT, f"LiDAR rileva ostacolo a θ={self.target_theta:.0f}°")
                turn_cmd = f"CMD:TURN_R,{abs(int(self.target_theta))}" if self.target_theta > 0 else f"CMD:TURN_L,{abs(int(self.target_theta))}"
                return turn_cmd, f"Puntamento saccadico verso θ={self.target_theta:.0f}°"

        if self.state == RobotState.INSPECTING_OBJECT:
            if now - self.state_enter_time > 1.2:
                self.set_state(RobotState.PATROL_FORWARD, "Ispezione saccadica conclusa")
            else:
                return "CMD:STOP", "Ispezione visiva 3D mirata"

        # Livello 5: Marcia regolare avanti libera (AVANTI)
        self.set_state(RobotState.PATROL_FORWARD, "Marcia regolare libera")
        return f"CMD:FORWARD,{self.forward_speed}", f"[MARCIA REGOLARE] Fronte: {dist_critica_fronte:.0f}cm"
