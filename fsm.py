#!/usr/bin/env python3
"""
M.A.R. — Safety FSM & Autonomous Behaviour

Stati:
  IDLE              fermo in attesa
  PATROL_FORWARD    marcia autonoma con velocità adattiva
  INSPECTING_OBJECT saccade: rotazione verso un'anomalia LiDAR laterale per inquadrarla
  AVOIDANCE         sterzata verso il settore migliore scelto dal Navigator
  ESCAPE            retromarcia controllata quando il fronte è bloccato / robot incastrato
  TERRAIN_HAZARD    dislivello / ostacolo basso a terra (stereo) → STOP_TERRA
  EMERGENCY_STOP    arresto: sensore affidabile < 40 cm, inclinazione critica, TTC, E-Stop operatore

Tutte le soglie di distanza provengono da safety_bands (40 / 70 / 120 cm) e sono le stesse
usate da mar-vision-service e dalla dashboard. L'ingresso è uno `Snapshot` del WorldModel,
quindi la FSM è deterministica e testabile iniettando `now`.
"""

from __future__ import annotations

import os
import time
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

from safety_bands import RiskDecisionStabilizer, classify_distance_cm, DIST_CRITICAL_CM, DIST_HIGH_CM, DIST_MODERATE_CM
from risk_model import RISK_CRITICAL, RISK_HIGH, RISK_MODERATE, risk_to_level
from navigator import Navigator, turn_command_for_angle
from world_model import Snapshot


class RobotState(Enum):
    IDLE = auto()
    PATROL_FORWARD = auto()
    INSPECTING_OBJECT = auto()
    AVOIDANCE = auto()
    ESCAPE = auto()
    TERRAIN_HAZARD = auto()
    EMERGENCY_STOP = auto()


_RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


class DecisionFSM:
    def __init__(
        self,
        forward_speed: int = 50,
        slow_speed: int = 25,
        critical_distance_cm: float = DIST_CRITICAL_CM,
        high_distance_cm: float = DIST_HIGH_CM,
        moderate_distance_cm: float = DIST_MODERATE_CM,
        ground_hazard_cm: float = 40.0,
        tilt_limit_deg: float = 40.0,
        autonomous: bool = True,
    ):
        self.state = RobotState.IDLE
        self.forward_speed = int(forward_speed)
        self.slow_speed = int(slow_speed)
        self.critical_distance_cm = float(critical_distance_cm)
        self.high_distance_cm = float(high_distance_cm)
        self.moderate_distance_cm = float(moderate_distance_cm)
        self.ground_hazard_cm = float(ground_hazard_cm)
        self.tilt_limit_deg = float(tilt_limit_deg)
        self.autonomous = bool(autonomous)

        self.state_enter_time = time.time()
        self.operator_estop = False
        self.target_theta = 0.0
        self.avoidance_sector: Optional[str] = None
        self.avoidance_cmd = "CMD:STOP"
        self.last_saccade_end_time = 0.0
        self.saccade_cooldown_s = 6.0
        self.ttc_alarm_count = 0
        self.escape_reason = ""

        self.navigator = Navigator()
        # Stabilizzatore sulla DISTANZA frontale fusa (isteresi 40/52, 70/85, 120/140)
        self.front_stabilizer = RiskDecisionStabilizer(
            ema_alpha=float(os.getenv("MAR_RISK_EMA_ALPHA", "0.30")),
            escalate_s=float(os.getenv("MAR_RISK_ESCALATE_S", "0.20")),
            deescalate_s=float(os.getenv("MAR_RISK_DEESCALATE_S", "1.2")),
        )
        # EMA asimmetrica sul RISCHIO frontale (sale subito, scende con calma)
        self.front_risk_ema: Optional[float] = None
        self.latest_risk_level = "LOW"
        self.latest_risk_value = 0.0
        self.last_decision: Dict[str, Any] = {}

        # Tempi minimi di permanenza (s)
        self.MIN_HOLD_EMERGENCY_S = 1.5
        self.MIN_HOLD_AVOIDANCE_S = 1.2
        self.MIN_HOLD_ESCAPE_S = 1.5
        self.MAX_ESCAPE_S = 3.0
        self.MIN_HOLD_TERRAIN_S = 1.5
        self.EXIT_CRITICAL_CM = self.critical_distance_cm + 15.0   # 55
        self.EXIT_AVOIDANCE_CM = self.high_distance_cm + 10.0      # 80

    # ---------------------------------------------------------------- helpers
    def set_state(self, new_state: RobotState, reason: str = "", now: Optional[float] = None):
        if self.state != new_state:
            print(f"🔄 [FSM] {self.state.name} ➔ {new_state.name} | {reason}")
            self.state = new_state
            self.state_enter_time = time.time() if now is None else now

    def trigger_operator_estop(self, reason: str = "E-Stop operatore"):
        self.operator_estop = True
        self.set_state(RobotState.EMERGENCY_STOP, reason)

    def clear_operator_estop(self, reason: str = "Resume operatore"):
        self.operator_estop = False
        if self.state == RobotState.EMERGENCY_STOP:
            self.set_state(RobotState.PATROL_FORWARD if self.autonomous else RobotState.IDLE, reason)

    def set_autonomous(self, enabled: bool, reason: str = ""):
        self.autonomous = bool(enabled)
        if not enabled and self.state not in (RobotState.EMERGENCY_STOP,):
            self.set_state(RobotState.IDLE, reason or "Autonomia disattivata")
        elif enabled and self.state == RobotState.IDLE and not self.operator_estop:
            self.set_state(RobotState.PATROL_FORWARD, reason or "Autonomia attivata")

    def _stabilized_risk(self, snap: Snapshot) -> Tuple[str, float, float]:
        """Ritorna (livello, distanza frontale stabilizzata, rischio frontale filtrato)."""
        _, _, dist_level, dist_smoothed = self.front_stabilizer.update(
            snap.gate_distance_cm(),
            fov_blocked=snap.fov_blocked,
            safety_alarm=snap.vision_safety_alarm,
            safety_reason="allarme lenti stereocamera",
            now=snap.now,
        )
        r = float(snap.front_risk)
        if self.front_risk_ema is None:
            self.front_risk_ema = r
        else:
            a = 0.70 if r > self.front_risk_ema else 0.20
            self.front_risk_ema = (1.0 - a) * self.front_risk_ema + a * r
        risk_level = risk_to_level(self.front_risk_ema)
        # Il livello finale è il più severo tra distanza stabilizzata e rischio fuso
        level = dist_level if _RISK_ORDER[dist_level] >= _RISK_ORDER[risk_level] else risk_level
        # Il settore Fronte con sensori AFFIDABILI vicini impone subito il suo livello (zero latenza)
        f = snap.sector_risks.get("Fronte")
        if f and f.min_reliable_cm is not None:
            direct = classify_distance_cm(f.min_reliable_cm)
            if _RISK_ORDER[direct] > _RISK_ORDER[level]:
                level = direct
        return level, float(dist_smoothed), float(self.front_risk_ema)

    # ------------------------------------------------------------------ update
    def update(self, snap: Snapshot) -> Tuple[str, str]:
        now = snap.now
        self.navigator.note_pose(snap.pose, now)

        risk_level, front_cm, front_risk = self._stabilized_risk(snap)
        self.latest_risk_level = risk_level
        self.latest_risk_value = front_risk

        cmd, reason = self._decide(snap, risk_level, front_cm, front_risk, now)

        if cmd.startswith("CMD:FORWARD"):
            self.navigator.on_forward(snap, now)
        else:
            self.navigator.on_not_forward()

        self.last_decision = {
            "cmd": cmd, "reason": reason, "state": self.state.name,
            "risk_level": risk_level, "front_risk": round(front_risk, 3), "front_cm": round(front_cm, 1),
            "nav_choice": self.navigator.last_choice,
            "nav_scores": {k: round(v, 3) for k, v in self.navigator.last_scores.items()},
        }
        return cmd, reason

    # ------------------------------------------------------------------ core
    def _decide(self, snap: Snapshot, risk_level: str, front_cm: float, front_risk: float, now: float) -> Tuple[str, str]:
        # 0) E-STOP operatore latched
        if self.operator_estop:
            self.set_state(RobotState.EMERGENCY_STOP, "E-Stop operatore latched", now)
            return "CMD:STOP", "[E-STOP OPERATORE] In attesa di resume"

        # 1) Inclinazione critica (IMU)
        roll = pitch = 0.0
        if snap.imu:
            try:
                roll = float(snap.imu.get("roll", 0.0))
                pitch = float(snap.imu.get("pitch", 0.0))
            except (TypeError, ValueError):
                roll = pitch = 0.0
        if abs(roll) > self.tilt_limit_deg or abs(pitch) > self.tilt_limit_deg:
            self.set_state(RobotState.EMERGENCY_STOP, f"Inclinazione critica roll={roll:.0f}° pitch={pitch:.0f}°", now)
            return "CMD:STOP", f"[STOP INCLINAZIONE] roll={roll:.1f}° pitch={pitch:.1f}°"

        if not self.autonomous:
            self.set_state(RobotState.IDLE, "Modalità manuale/idle", now)
            return "CMD:STOP", "[IDLE] Autonomia disattivata"

        # 2) Sensore AFFIDABILE (US / LiDAR) sotto soglia critica nel fronte → STOP immediato
        f = snap.sector_risks.get("Fronte")
        min_rel = f.min_reliable_cm if f else None
        if min_rel is not None and min_rel < self.critical_distance_cm:
            src = min(((v, k) for k, v in (f.inputs or {}).items() if k != "stereo"), default=(min_rel, "lidar/us"))[1]
            self.set_state(RobotState.EMERGENCY_STOP, f"Ostacolo critico {src} a {min_rel:.0f}cm", now)
            return "CMD:STOP", f"[STOP {src.upper()}] ostacolo a {min_rel:.0f}cm (< {self.critical_distance_cm:.0f})"

        # 2b) Diagonali frontali critiche (il corpo li lambisce)
        for k in ("Front-SX", "Front-DX"):
            s = snap.sector_risks.get(k)
            if s and s.min_reliable_cm is not None and s.min_reliable_cm < self.critical_distance_cm * 0.8:
                self.set_state(RobotState.EMERGENCY_STOP, f"Ostacolo critico {k} a {s.min_reliable_cm:.0f}cm", now)
                return "CMD:STOP", f"[STOP {k}] ostacolo a {s.min_reliable_cm:.0f}cm"

        # 3) Time-To-Collision (stereo motion tracker) — richiede 2 conferme
        vz = float(snap.motion.get("velocity_z_cms", 0.0) or 0.0)
        ttc = float(snap.motion.get("ttc_sec", 999.0) or 999.0)
        if vz < -25.0 and ttc < 1.2 and min(front_cm, snap.front_dist_cm) < self.high_distance_cm:
            self.ttc_alarm_count += 1
            if self.ttc_alarm_count >= 2:
                self.set_state(RobotState.EMERGENCY_STOP, f"Impatto imminente TTC={ttc:.1f}s", now)
                return "CMD:STOP", f"[STOP TTC] {ttc:.1f}s (vz={vz:.0f}cm/s)"
        else:
            self.ttc_alarm_count = 0

        # 4) Rischio/distanza frontale CRITICAL (fusione) → STOP
        if risk_level == "CRITICAL" or front_cm < self.critical_distance_cm:
            self.set_state(RobotState.EMERGENCY_STOP, f"Fronte CRITICAL (d={front_cm:.0f}cm R={front_risk:.2f})", now)
            return "CMD:STOP", f"[STOP CRITICAL] fronte {front_cm:.0f}cm R={front_risk:.2f}"

        # 4b) Attuatori assenti: le soglie di sicurezza sopra restano valide, ma non si
        #     comanda marcia, fuga o sterzata. Quando l'Arduino arriva si riparte da pattuglia,
        #     non da una manovra rimasta a metà.
        if snap.motion_connected is False:
            if self.state == RobotState.EMERGENCY_STOP:
                return "CMD:STOP", "[ATTESA ATTUATORI] stop di sicurezza, motion assente"
            if self.state not in (RobotState.PATROL_FORWARD, RobotState.IDLE):
                self.set_state(RobotState.PATROL_FORWARD, "attuatori assenti, manovra annullata", now)
            return "CMD:STOP", "[ATTESA ATTUATORI] controller motion non collegato"

        # 5) Permanenza / uscita da EMERGENCY_STOP
        if self.state == RobotState.EMERGENCY_STOP:
            t = now - self.state_enter_time
            if t < self.MIN_HOLD_EMERGENCY_S:
                return "CMD:STOP", f"[HOLD STOP] {t:.1f}/{self.MIN_HOLD_EMERGENCY_S:.1f}s fronte={front_cm:.0f}cm"
            if front_cm >= self.EXIT_CRITICAL_CM and not snap.ground_hazard and front_risk < RISK_HIGH:
                self.set_state(RobotState.PATROL_FORWARD, f"Via libera (fronte={front_cm:.0f}cm)", now)
            else:
                # fermi da un po' con il fronte ancora chiuso: prova una via di fuga
                if t > 4.0 and self.navigator.escape_possible(snap):
                    self.escape_reason = "fronte bloccato dopo stop"
                    self.set_state(RobotState.ESCAPE, self.escape_reason, now)
                    return "CMD:BACKWARD,30", f"[ESCAPE] {self.escape_reason}"
                if front_cm < self.EXIT_CRITICAL_CM:
                    why = f"fronte={front_cm:.0f}cm (< {self.EXIT_CRITICAL_CM:.0f})"
                elif snap.ground_hazard:
                    why = "ostacolo a terra"
                else:
                    objs = [o.get("class") for o in (snap.objects or [])][:2]
                    why = f"R={front_risk:.2f} ≥ {RISK_HIGH:.2f}" + (f" ({', '.join(str(o) for o in objs)})" if objs else "")
                return "CMD:STOP", f"[ATTESA VIA LIBERA] {why}"

        # 6) Dislivello / ostacolo a terra
        if snap.ground_hazard and snap.z_basso_cm < self.ground_hazard_cm:
            self.set_state(RobotState.TERRAIN_HAZARD, f"Ostacolo a terra Z_basso={snap.z_basso_cm:.0f}cm", now)
            return "CMD:STOP_TERRA", f"[OSTACOLO A TERRA] Z_basso={snap.z_basso_cm:.0f}cm"
        if self.state == RobotState.TERRAIN_HAZARD:
            t = now - self.state_enter_time
            if t < self.MIN_HOLD_TERRAIN_S or snap.ground_hazard or snap.z_basso_cm < self.ground_hazard_cm + 8.0:
                return "CMD:STOP_TERRA", f"[HOLD TERRAIN] Z_basso={snap.z_basso_cm:.0f}cm"
            self.set_state(RobotState.PATROL_FORWARD, "Terreno libero", now)

        # 7) ESCAPE in corso (retromarcia) → poi scegli una nuova direzione
        if self.state == RobotState.ESCAPE:
            t = now - self.state_enter_time
            rear_cm = snap.sector_dist("Retro", default=999.0)
            if rear_cm < self.critical_distance_cm + 10.0:
                self.set_state(RobotState.EMERGENCY_STOP, f"Retro bloccato ({rear_cm:.0f}cm)", now)
                return "CMD:STOP", f"[STOP RETRO] {rear_cm:.0f}cm"
            if t < self.MIN_HOLD_ESCAPE_S or (t < self.MAX_ESCAPE_S and front_cm < self.EXIT_AVOIDANCE_CM):
                return "CMD:BACKWARD,30", f"[ESCAPE] {self.escape_reason} ({t:.1f}s)"
            sector, ang = self.navigator.choose_heading(snap, now)
            if sector is None:
                self.set_state(RobotState.EMERGENCY_STOP, "Nessuna via di fuga", now)
                return "CMD:STOP", "[STOP] nessun settore percorribile"
            self.avoidance_sector, self.avoidance_cmd = sector, turn_command_for_angle(ang)
            self.set_state(RobotState.AVOIDANCE, f"Dopo escape → {sector}", now)
            return self.avoidance_cmd, f"[DEVIAZIONE {sector}] dopo retromarcia"

        # 8) AVOIDANCE in corso: completa la sterzata, poi riparti quando il fronte è libero
        if self.state == RobotState.AVOIDANCE:
            t = now - self.state_enter_time
            if t < self.MIN_HOLD_AVOIDANCE_S:
                return self.avoidance_cmd, f"[DEVIAZIONE {self.avoidance_sector}] {t:.1f}/{self.MIN_HOLD_AVOIDANCE_S:.1f}s"
            if front_cm >= self.EXIT_AVOIDANCE_CM and front_risk < RISK_MODERATE + 0.15:
                self.set_state(RobotState.PATROL_FORWARD, f"Deviazione completata (fronte={front_cm:.0f}cm)", now)
            elif t > 6.0:
                # sterzo da troppo senza aprire il fronte: ricalcola
                sector, ang = self.navigator.choose_heading(snap, now)
                if sector is None and self.navigator.escape_possible(snap):
                    self.escape_reason = "deviazione inefficace"
                    self.set_state(RobotState.ESCAPE, self.escape_reason, now)
                    return "CMD:BACKWARD,30", f"[ESCAPE] {self.escape_reason}"
                if sector is not None:
                    self.avoidance_sector, self.avoidance_cmd = sector, turn_command_for_angle(ang)
                    self.state_enter_time = now
                return self.avoidance_cmd, f"[DEVIAZIONE {self.avoidance_sector}] ricalcolo"
            else:
                return self.avoidance_cmd, f"[DEVIAZIONE {self.avoidance_sector}] fronte={front_cm:.0f}cm"

        # 9) Robot incastrato (FORWARD ma posa SLAM ferma)
        if self.state == RobotState.PATROL_FORWARD and self.navigator.is_stuck(snap, now):
            if self.navigator.escape_possible(snap):
                self.escape_reason = "robot incastrato (posa SLAM ferma)"
                self.set_state(RobotState.ESCAPE, self.escape_reason, now)
                return "CMD:BACKWARD,30", f"[ESCAPE] {self.escape_reason}"
            sector, ang = self.navigator.choose_heading(snap, now)
            if sector is not None and sector != "Fronte":
                self.avoidance_sector, self.avoidance_cmd = sector, turn_command_for_angle(ang)
                self.set_state(RobotState.AVOIDANCE, "Incastrato → nuova direzione", now)
                return self.avoidance_cmd, f"[DEVIAZIONE {sector}] robot incastrato"

        # 10) Fronte HIGH (ostacolo 40–70 cm o rischio alto) → nuova deviazione
        if risk_level == "HIGH" or front_cm < self.high_distance_cm or front_risk >= RISK_HIGH:
            sector, ang = self.navigator.choose_heading(snap, now)
            if sector is None:
                if self.navigator.escape_possible(snap):
                    self.escape_reason = "tutti i settori frontali bloccati"
                    self.set_state(RobotState.ESCAPE, self.escape_reason, now)
                    return "CMD:BACKWARD,30", f"[ESCAPE] {self.escape_reason}"
                self.set_state(RobotState.EMERGENCY_STOP, "Bloccato su tutti i lati", now)
                return "CMD:STOP", "[STOP] bloccato su tutti i lati"
            if sector == "Fronte":
                # il fronte resta il migliore (ostacolo laterale/diagonale): avanza piano
                self.set_state(RobotState.PATROL_FORWARD, "Fronte migliore, rallento", now)
                return f"CMD:FORWARD,{self.slow_speed}", f"[RALLENTA] fronte={front_cm:.0f}cm R={front_risk:.2f}"
            self.avoidance_sector, self.avoidance_cmd = sector, turn_command_for_angle(ang)
            self.set_state(RobotState.AVOIDANCE, f"Fronte {front_cm:.0f}cm R={front_risk:.2f} → {sector}", now)
            scores = self.navigator.last_scores
            return self.avoidance_cmd, f"[DEVIAZIONE {sector}] score={scores.get(sector, 0):.2f} fronte={front_cm:.0f}cm"

        # 11) Saccade verso anomalia LiDAR laterale (percezione attiva), solo in pattugliamento libero
        la = snap.lidar_alert or {}
        sector_alert = str(la.get("sector") or la.get("sector_en") or "")
        if (
            self.state == RobotState.PATROL_FORWARD
            and la.get("has_alert")
            and sector_alert in ("Front-DX", "Front-SX", "FRONT_RIGHT", "FRONT_LEFT")
            and (now - self.last_saccade_end_time) > self.saccade_cooldown_s
            and not snap.objects  # se YOLO vede già l'oggetto non serve ruotare
        ):
            try:
                theta = float(la.get("theta", 0.0) or 0.0)
            except (TypeError, ValueError):
                theta = 0.0
            if abs(theta) > 20.0:
                self.target_theta = theta
                self.set_state(RobotState.INSPECTING_OBJECT, f"Saccade verso θ={theta:.0f}°", now)
                return turn_command_for_angle(theta), f"[SACCADE] ispezione θ={theta:.0f}°"
        if self.state == RobotState.INSPECTING_OBJECT:
            if now - self.state_enter_time > 1.2:
                self.last_saccade_end_time = now
                self.set_state(RobotState.PATROL_FORWARD, "Ispezione conclusa", now)
            else:
                return "CMD:STOP", "[SACCADE] acquisizione stereo/YOLO"

        # 12) Marcia regolare con velocità adattiva (la distanza fusa modula la velocità anche se
        #     arriva dalla sola stereo: cautela sì, stop no)
        self.set_state(RobotState.PATROL_FORWARD, "Marcia autonoma", now)
        speed = self.navigator.cruise_speed(min(front_cm, snap.front_dist_cm), front_risk,
                                            base=self.forward_speed, slow=self.slow_speed)
        if snap.degraded:
            # solo stereo in linea: niente LiDAR/US a confermare → velocità ridotta
            speed = min(speed, self.slow_speed)
        if pitch > 15.0 and not snap.degraded:
            return f"CMD:FORWARD,{self.forward_speed}", f"[SALITA] pitch={pitch:.1f}° fronte={front_cm:.0f}cm"
        if pitch < -15.0:
            return f"CMD:FORWARD,{self.slow_speed}", f"[DISCESA] pitch={pitch:.1f}° fronte={front_cm:.0f}cm"
        tag = "MARCIA" if risk_level == "LOW" else "CAUTELA"
        return f"CMD:FORWARD,{speed}", f"[{tag}] fronte={front_cm:.0f}cm R={front_risk:.2f} v={speed}"
