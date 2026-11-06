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

try:
    from safety_bands import RiskDecisionStabilizer, EXIT_CRITICAL_CM, EXIT_MODERATE_CM
    from sensor_fusion import FrontRangeFusion
except ImportError:
    from .safety_bands import RiskDecisionStabilizer, EXIT_CRITICAL_CM, EXIT_MODERATE_CM
    from .sensor_fusion import FrontRangeFusion


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
        critical_distance_cm: float = 30.0,   # CRITICAL STOP immediato (< 30 cm)
        high_distance_cm: float = 65.0,       # Deviazione / schivamento dinamico (< 65 cm)
        moderate_distance_cm: float = 120.0,  # Allerta periferica (< 120 cm)
        ground_hazard_cm: float = 40.0,       # Ostacolo a terra / dislivello (< 40 cm)
        slow_warning_cm: float = 65.0,        # Rallentamento / schivata (< 65 cm)
    ):
        self.state = RobotState.IDLE
        self.forward_speed = forward_speed
        self.critical_distance_cm = critical_distance_cm
        self.high_distance_cm = high_distance_cm
        self.moderate_distance_cm = moderate_distance_cm
        self.ground_hazard_cm = ground_hazard_cm
        self.slow_warning_cm = slow_warning_cm

        self.state_enter_time = time.time()
        self.target_theta: float = 0.0
        self.avoidance_direction: str = "RIGHT"
        self.avoidance_start_time: float = 0.0
        # Cooldown per evitare loop continui di saccadi visive
        self.last_saccade_end_time: float = 0.0
        self.saccade_cooldown_s: float = 6.0
        # Filtro debounce allarme Time-To-Collision
        self.ttc_alarm_count: int = 0
        # Operator / dashboard E-STOP stays latched until explicit resume
        self.operator_estop: bool = False
        # Fusione frontale ponderata (niente oscillazioni LOW↔CRITICAL)
        self.front_risk = RiskDecisionStabilizer(ema_alpha=0.25, escalate_s=0.35, deescalate_s=2.0)
        # Layered fusion stile Nav2 (US Range + LiDAR Obstacle + stereo corroboration)
        self.range_fusion = FrontRangeFusion(agree_ratio=0.35)
        self.last_fusion_meta: Dict[str, Any] = {}

    def set_state(self, new_state: RobotState, reason: str = ""):
        if self.state != new_state:
            print(f"🔄 [FSM STATE CHANGE]: {self.state.name} ➔ {new_state.name} | Motivo: {reason}")
            self.state = new_state
            self.state_enter_time = time.time()

    def trigger_operator_estop(self, reason: str = "E-Stop operatore"):
        """Latch emergency stop from dashboard / safety topic (does not auto-clear)."""
        self.operator_estop = True
        self.set_state(RobotState.EMERGENCY_STOP, reason)

    def clear_operator_estop(self, reason: str = "Resume operatore"):
        """Release latched operator E-STOP and return to patrol when path allows."""
        self.operator_estop = False
        if self.state == RobotState.EMERGENCY_STOP:
            self.set_state(RobotState.PATROL_FORWARD, reason)

    def update(
        self,
        lidar_alert: Optional[Dict],
        vision_objects: Optional[Dict],
        ultrasonic_distance_cm: float = -1.0,
        ground_hazard_alert: Optional[Dict] = None,
        imu_data: Optional[Dict] = None,
        motion_data: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Processes multi-sensor inputs (LiDAR 360°, Stereo 3D Vision, IMU Kalman, Ultrasonic)
        and produces the next deterministic motion command.
        Returns: (CMD_STRING, REASON_DESCRIPTION)
        """
        now = time.time()

        # Latched operator E-STOP: motors stay frozen until clear_operator_estop()
        if self.operator_estop:
            if self.state != RobotState.EMERGENCY_STOP:
                self.set_state(RobotState.EMERGENCY_STOP, "E-Stop operatore latched")
            return "CMD:STOP", "[E-STOP OPERATORE] In attesa di resume"

        # 0. Analisi Assetto e Sicurezza IMU (Rollio e Beccheggio)
        roll_robot = 0.0
        pitch_robot = 0.0
        if imu_data:
            t_imu = imu_data.get("timestamp", now)
            if now - t_imu < 1.0:
                roll_robot = float(imu_data.get("roll", 0.0))
                pitch_robot = float(imu_data.get("pitch", 0.0))

        if abs(roll_robot) > 40.0 or abs(pitch_robot) > 40.0:
            if self.state != RobotState.EMERGENCY_STOP:
                self.set_state(RobotState.EMERGENCY_STOP, f"Inclinazione critica (Roll:{roll_robot:.1f}°, Pitch:{pitch_robot:.1f}°)")
            return "CMD:STOP", f"[STOP INCLINAZIONE CRITICA] Roll:{roll_robot:.1f}° Pitch:{pitch_robot:.1f}°"

        # 1. Estrazione metriche LiDAR (Fronte, DX, SX)
        lidar_front_cm = 999.0
        d_dx = 999.0
        d_sx = 999.0

        if lidar_alert:
            t_lidar = lidar_alert.get("timestamp", now)
            if now - t_lidar < 1.0:
                sec_mins = lidar_alert.get("sector_mins", {})
                if not sec_mins and "sector_mins_en" in lidar_alert:
                    sec_mins = lidar_alert.get("sector_mins_en", {})

                # Distanza frontale da LiDAR
                f_m = lidar_alert.get("front_distance_m")
                if f_m is not None:
                    lidar_front_cm = float(f_m) * 100.0 if float(f_m) < 15.0 else float(f_m)
                elif lidar_alert.get("sector") in ("Fronte", "FRONT") and lidar_alert.get("dist_m") is not None:
                    dist_val = float(lidar_alert["dist_m"])
                    lidar_front_cm = dist_val * 100.0 if dist_val < 15.0 else dist_val
                elif sec_mins.get("Fronte") is not None or sec_mins.get("FRONT") is not None:
                    val_f = sec_mins.get("Fronte") if sec_mins.get("Fronte") is not None else sec_mins.get("FRONT")
                    lidar_front_cm = float(val_f) * 100.0 if float(val_f) < 15.0 else float(val_f)

                # Se un ostacolo imminente (< 30cm) invade i settori angolari frontali (Front-DX o Front-SX)
                for adj_key in ("Front-DX", "FRONT_RIGHT", "Front-SX", "FRONT_LEFT"):
                    if sec_mins.get(adj_key) is not None:
                        val_adj = float(sec_mins[adj_key])
                        cm_adj = val_adj * 100.0 if val_adj < 15.0 else val_adj
                        if cm_adj < self.critical_distance_cm:
                            lidar_front_cm = min(lidar_front_cm, cm_adj)

                # Spazi laterali per deviazione (Destra vs Sinistra)
                val_dx = (
                    sec_mins.get("Destra") if sec_mins.get("Destra") is not None else
                    sec_mins.get("Front-DX") if sec_mins.get("Front-DX") is not None else
                    sec_mins.get("RIGHT") if sec_mins.get("RIGHT") is not None else
                    sec_mins.get("FRONT_RIGHT")
                )
                val_sx = (
                    sec_mins.get("Sinistra") if sec_mins.get("Sinistra") is not None else
                    sec_mins.get("Front-SX") if sec_mins.get("Front-SX") is not None else
                    sec_mins.get("LEFT") if sec_mins.get("LEFT") is not None else
                    sec_mins.get("FRONT_LEFT")
                )

                if val_dx is not None:
                    d_dx = float(val_dx) * 100.0 if float(val_dx) < 15.0 else float(val_dx)
                if val_sx is not None:
                    d_sx = float(val_sx) * 100.0 if float(val_sx) < 15.0 else float(val_sx)

        # 2. Estrazione metriche Visione 3D (Z_basso, Z_centro, Z_alto)
        z_basso_cm = 999.0
        z_centro_cm = 999.0
        has_ground_hazard = False

        if vision_objects:
            t_vision = vision_objects.get("timestamp", now)
            if now - t_vision < 1.0:
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
            t_hazard = ground_hazard_alert.get("timestamp", now)
            if now - t_hazard < 1.0:
                has_ground_hazard = True
                if "z_basso_cm" in ground_hazard_alert:
                    z_basso_cm = min(z_basso_cm, float(ground_hazard_alert["z_basso_cm"]))

        # 3. Fusione frontale: LiDAR + stereo (target/z) + ultrasuoni
        vision_target_cm = 999.0
        if vision_objects:
            t_vision2 = vision_objects.get("timestamp", now)
            if now - t_vision2 < 1.0:
                for key in ("target_dist_cm", "obstacle_near_cm"):
                    if vision_objects.get(key) is not None:
                        try:
                            vision_target_cm = min(vision_target_cm, float(vision_objects[key]))
                        except Exception:
                            pass
                # risk_level CRITICAL from vision (FOV blocked / sheet) forces stop distance
                if str(vision_objects.get("risk_level", "")).upper() == "CRITICAL":
                    vision_target_cm = min(vision_target_cm, self.critical_distance_cm * 0.8)

        # Fusione layered ponderata con safety override
        stereo_for_fuse = vision_target_cm if vision_target_cm < 900.0 else (
            z_centro_cm if z_centro_cm < 900.0 else 999.0
        )
        motion_vz = float(motion_data.get("velocity_z_cms", 0.0)) if motion_data else 0.0
        ttc = float(motion_data.get("ttc_sec", 999.0)) if motion_data else 999.0

        dist_raw_fronte, fuse_meta = self.range_fusion.fuse(
            ultrasonic_cm=float(ultrasonic_distance_cm),
            lidar_front_cm=float(lidar_front_cm),
            stereo_cm=float(stereo_for_fuse),
            motion_vz_cms=motion_vz,
        )
        self.last_fusion_meta = fuse_meta

        # Preferisci risk già stabilizzato dalla vision se fresco e più severo
        vision_risk = ""
        if vision_objects:
            t_vr = vision_objects.get("timestamp", now)
            if now - t_vr < 1.0:
                vision_risk = str(vision_objects.get("risk_level", "") or "").upper()

        risk_level, dist_critica_fronte = self.front_risk.update(dist_raw_fronte, now=now)
        # Accordo con vision: se vision tiene CRITICAL/HIGH, accelera solo se confermato da ostacolo/hazard reale
        obs_cm = float(vision_objects.get("obstacle_near_cm", 999.0)) if vision_objects else 999.0
        has_vis_obs = bool(vision_objects.get("obstacle_found", False)) if vision_objects else False
        if vision_risk == "CRITICAL" and risk_level != "CRITICAL":
            if obs_cm < 45.0 or has_vis_obs or has_ground_hazard:
                risk_level, dist_critica_fronte = self.front_risk.update(
                    min(dist_critica_fronte, self.critical_distance_cm * 0.7), now=now
                )
        elif vision_risk == "HIGH" and risk_level in ("LOW", "MODERATE"):
            if (obs_cm < 75.0 and has_vis_obs) or has_ground_hazard:
                risk_level, dist_critica_fronte = self.front_risk.update(
                    min(dist_critica_fronte, self.high_distance_cm * 0.85), now=now
                )

        # 4. Gerarchia decisionale multi-livello con isteresi e persistenza delle decisioni
        MIN_HOLD_EMERGENCY_S = 2.0    # Minimo tempo di arresto prima di poter ripartire
        MIN_HOLD_AVOIDANCE_S = 2.0    # Minimo tempo per completare una manovra di sterzata
        MIN_HOLD_TERRAIN_S = 1.5      # Minimo tempo di stop per ostacolo a terra
        EXIT_CRITICAL_DIST_CM = 55.0  # Isteresi uscita da stop: deve vedere almeno 55cm liberi
        EXIT_AVOIDANCE_DIST_CM = 75.0 # Isteresi uscita da deviazione: deve vedere almeno 75cm liberi

        # Livello 0: Sicurezza Assoluta Ultrasuoni Diretto (< 30 cm)
        if 0.0 < ultrasonic_distance_cm < self.critical_distance_cm:
            self.set_state(
                RobotState.EMERGENCY_STOP,
                f"Ostacolo critico ultrasuoni ({ultrasonic_distance_cm:.0f}cm < {self.critical_distance_cm:.0f}cm)",
            )
            return "CMD:STOP", f"[STOP ULTRASUONI] Ostacolo a {ultrasonic_distance_cm:.0f}cm"

        # Livello 0b: Anticipazione predittiva per avvicinamento rapido reale (TTC < 1.2s, Vz < -25 cm/s)
        if motion_vz < -25.0 and ttc < 1.2 and dist_critica_fronte < self.slow_warning_cm:
            self.ttc_alarm_count += 1
            if self.ttc_alarm_count >= 2:
                self.set_state(
                    RobotState.EMERGENCY_STOP,
                    f"Rischio impatto imminente confermato (TTC: {ttc:.1f}s, Vz: {motion_vz:.0f}cm/s)",
                )
                return "CMD:STOP", f"[STOP IMPATTO IMMINENTE] TTC: {ttc:.1f}s (Vz: {motion_vz:.0f}cm/s)"
        else:
            self.ttc_alarm_count = 0

        # Livello 1: Collisione imminente (< 30 cm) -> Scatta IMMEDIATO
        if dist_critica_fronte < self.critical_distance_cm:
            self.set_state(
                RobotState.EMERGENCY_STOP,
                f"Ostacolo critico frontale ({dist_critica_fronte:.0f}cm < {self.critical_distance_cm:.0f}cm)",
            )
            return "CMD:STOP", f"[STOP EMERGENCY] Ostacolo a {dist_critica_fronte:.0f}cm"

        # Livello 2: Se siamo attualmente in EMERGENCY_STOP, mantieni l'arresto (hold time + isteresi)
        if self.state == RobotState.EMERGENCY_STOP:
            time_in_stop = now - self.state_enter_time
            if self.operator_estop:
                return "CMD:STOP", "[E-STOP OPERATORE BLOCCATO]"
            if time_in_stop < MIN_HOLD_EMERGENCY_S:
                return "CMD:STOP", f"[HOLD EMERGENCY STOP] {time_in_stop:.1f}s/{MIN_HOLD_EMERGENCY_S:.1f}s (fronte={dist_critica_fronte:.0f}cm)"
            # Hold time trascorso: per uscire, la strada deve essere chiaramente libera
            if dist_critica_fronte >= EXIT_CRITICAL_DIST_CM and not has_ground_hazard:
                self.set_state(RobotState.PATROL_FORWARD, f"Via libera confermata (fronte={dist_critica_fronte:.0f}cm)")
            else:
                return "CMD:STOP", f"[ATTESA VIA LIBERA] fronte={dist_critica_fronte:.0f}cm < {EXIT_CRITICAL_DIST_CM:.0f}cm"

        # Livello 3: Dislivello / gradino a terra (< 40 cm)
        if has_ground_hazard and (z_basso_cm < self.ground_hazard_cm):
            self.set_state(RobotState.TERRAIN_HAZARD, f"Dislivello a terra (Z_basso={z_basso_cm:.0f} cm)")
            return "CMD:STOP_TERRA", f"[OSTACOLO A TERRA] Z_basso: {z_basso_cm:.0f}cm"

        if self.state == RobotState.TERRAIN_HAZARD:
            time_in_th = now - self.state_enter_time
            if time_in_th < MIN_HOLD_TERRAIN_S or has_ground_hazard or z_basso_cm < 48.0:
                return "CMD:STOP_TERRA", f"[HOLD TERRAIN HAZARD] Z_basso={z_basso_cm:.0f}cm"
            self.set_state(RobotState.PATROL_FORWARD, "Terreno libero")

        # Livello 4: Se siamo in AVOIDANCE (deviazione in corso), PORTA AVANTI LA DECISIONE!
        if self.state == RobotState.AVOIDANCE:
            time_in_avoid = now - self.state_enter_time
            turn_cmd = "CMD:TURN_R,45" if self.avoidance_direction == "RIGHT" else "CMD:TURN_L,45"
            # Deve mantenere la sterzata per almeno MIN_HOLD_AVOIDANCE_S
            if time_in_avoid < MIN_HOLD_AVOIDANCE_S:
                return turn_cmd, f"[CONTINUA DEVIAZIONE {self.avoidance_direction}] {time_in_avoid:.1f}s/{MIN_HOLD_AVOIDANCE_S:.1f}s"
            # Se il tempo minimo è passato ed il fronte è ora libero
            if dist_critica_fronte >= EXIT_AVOIDANCE_DIST_CM:
                self.set_state(RobotState.PATROL_FORWARD, f"Deviazione conclusa con successo (fronte={dist_critica_fronte:.0f}cm)")
            else:
                # Fronte ancora occupato, continua la manovra
                return turn_cmd, f"[MANOVRA CONTINUA {self.avoidance_direction}] fronte={dist_critica_fronte:.0f}cm"

        # Livello 5: Ostacolo a media distanza (< 65 cm) -> Innesco nuova deviazione attiva
        if dist_critica_fronte < self.slow_warning_cm:
            if d_dx >= 900 and d_sx >= 900:
                direction = "RIGHT" if self.avoidance_direction != "RIGHT" else "LEFT"
            else:
                direction = "RIGHT" if d_dx >= d_sx else "LEFT"
            turn_cmd = "CMD:TURN_R,45" if direction == "RIGHT" else "CMD:TURN_L,45"
            self.avoidance_direction = direction
            self.set_state(RobotState.AVOIDANCE, f"Ostacolo a {dist_critica_fronte:.0f}cm -> Inizio deviazione {direction}")
            return turn_cmd, f"[DEVIAZIONE {direction}] DX:{d_dx:.0f}cm | SX:{d_sx:.0f}cm"

        # Livello 6: Puntamento saccadico LiDAR (solo se in pattugliamento libero e cooldown trascorso)
        sector = str(lidar_alert.get("sector", "") if lidar_alert else "")
        sector_en = str(lidar_alert.get("sector_en", "") if lidar_alert else "")
        frontal_sector = sector in ("Fronte", "Front-DX", "Front-SX") or sector_en in (
            "FRONT", "FRONT_RIGHT", "FRONT_LEFT"
        )
        if (
            self.state == RobotState.PATROL_FORWARD
            and lidar_alert
            and lidar_alert.get("has_alert", False)
            and frontal_sector
            and abs(float(lidar_alert.get("theta", 0) or 0)) > 20
            and (now - self.last_saccade_end_time > self.saccade_cooldown_s)
        ):
            self.target_theta = float(lidar_alert.get("theta", 0.0) or 0.0)
            self.set_state(RobotState.INSPECTING_OBJECT, f"LiDAR rileva ostacolo a θ={self.target_theta:.0f}°")
            turn_cmd = (
                f"CMD:TURN_R,{abs(int(self.target_theta))}"
                if self.target_theta > 0
                else f"CMD:TURN_L,{abs(int(self.target_theta))}"
            )
            return turn_cmd, f"Puntamento saccadico verso θ={self.target_theta:.0f}°"

        if self.state == RobotState.INSPECTING_OBJECT:
            if now - self.state_enter_time > 1.2:
                self.last_saccade_end_time = now
                self.set_state(RobotState.PATROL_FORWARD, "Ispezione saccadica conclusa")
            else:
                return "CMD:STOP", "Ispezione visiva 3D mirata"

        # Livello 7: Marcia regolare avanti / Controllo Pitch salita o discesa
        if self.state != RobotState.PATROL_FORWARD:
            self.set_state(RobotState.PATROL_FORWARD, "Marcia regolare libera")

        if pitch_robot > 15.0:
            return f"CMD:FORWARD,{self.forward_speed}", f"[PITCH ELEVATO: SALITA] Pitch: {pitch_robot:.1f}°"
        elif pitch_robot < -15.0:
            return "CMD:BACKWARD,30", f"[PITCH NEGATIVO: DISCESA] Pitch: {pitch_robot:.1f}°"
        else:
            return f"CMD:FORWARD,{self.forward_speed}", f"[MARCIA REGOLARE] Fronte: {dist_critica_fronte:.0f}cm"
