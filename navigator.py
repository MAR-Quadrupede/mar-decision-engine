#!/usr/bin/env python3
"""
M.A.R. — Navigatore autonomo reattivo (VFH-lite sugli 8 settori + memoria spaziale).

Dato lo Snapshot del WorldModel (rischio e distanza fusa per settore, posa SLAM),
sceglie la direzione di marcia più promettente:

    score_s = clearance_s · (1 - R_s) · turn_cost_s · explore_s · hysteresis_s

  clearance_s : distanza fusa del settore normalizzata su 3 m
  (1 - R_s)   : sicurezza del settore secondo l'equazione di rischio
  turn_cost_s : preferisce rotazioni piccole (1.0 dritto → 0.65 dietro)
  explore_s   : 1.0 se la cella target (posa + 1.5 m lungo la direzione) non è stata visitata
                di recente, 0.6 se già visitata → il robot esplora la stanza invece di ripassare
  hysteresis  : +15% al lato dell'ultima sterzata (niente sinistra/destra a raffica)

Il navigatore gestisce anche:
  • velocità di crociera in funzione della distanza/rischio frontale
  • rilevamento "incastrato" (FORWARD comandato ma posa SLAM ferma)
  • manovra di fuga (retromarcia) quando tutti i settori utili sono bloccati
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from risk_model import SECTORS, SECTOR_ANGLE_DEG, RISK_HIGH, RISK_MODERATE, DIST_HIGH_CM

MAX_CLEAR_CM = 300.0
MIN_PASSABLE_CM = 60.0          # sotto questa distanza un settore non è percorribile
VISIT_CELL_M = 0.5
VISIT_MEMORY_S = 120.0
STUCK_AFTER_S = 4.0
STUCK_MIN_DISPLACEMENT_M = 0.06
MOTION_SEEN_MIN_M = 0.30        # il robot deve essersi mosso almeno così tanto prima di fidarsi dello stuck-check


def turn_command_for_angle(angle_deg: float) -> str:
    """Angolo relativo (positivo = destra) → comando firmware."""
    a = float(angle_deg)
    if abs(a) < 1.0:
        return "CMD:FORWARD,30"
    deg = int(round(min(180.0, abs(a))))
    return f"CMD:TURN_R,{deg}" if a > 0 else f"CMD:TURN_L,{deg}"


class Navigator:
    def __init__(self):
        self.visited: Deque[Tuple[int, int, float]] = deque(maxlen=600)
        self.last_turn_sign: int = 0          # +1 destra, -1 sinistra
        self.forward_since: Optional[float] = None
        self.forward_start_pose: Optional[Tuple[float, float]] = None
        self.total_motion_m: float = 0.0
        self._last_pose: Optional[Tuple[float, float]] = None
        self.last_scores: Dict[str, float] = {}
        self.last_choice: Optional[str] = None

    # ------------------------------------------------------------ pose memory
    def note_pose(self, pose: Optional[Dict], now: Optional[float] = None):
        if not pose:
            return
        now = time.time() if now is None else now
        try:
            x = float(pose.get("x_m", pose.get("x", 0.0)))
            y = float(pose.get("y_m", pose.get("y", 0.0)))
        except (TypeError, ValueError):
            return
        if self._last_pose is not None:
            step = math.hypot(x - self._last_pose[0], y - self._last_pose[1])
            # sotto 8 mm è jitter SLAM da fermo: non deve "armare" il rilevamento incastrato
            if step >= 0.008:
                self.total_motion_m += step
        self._last_pose = (x, y)
        cell = (int(math.floor(x / VISIT_CELL_M)), int(math.floor(y / VISIT_CELL_M)))
        if not self.visited or (self.visited[-1][0], self.visited[-1][1]) != cell:
            self.visited.append((cell[0], cell[1], now))

    def _visited_recently(self, x: float, y: float, now: float) -> bool:
        cell = (int(math.floor(x / VISIT_CELL_M)), int(math.floor(y / VISIT_CELL_M)))
        for cx, cy, ts in reversed(self.visited):
            if now - ts > VISIT_MEMORY_S:
                break
            if (cx, cy) == cell:
                return True
        return False

    # --------------------------------------------------------------- scoring
    def score_sectors(self, snap, now: Optional[float] = None, allow_rear: bool = False) -> Dict[str, float]:
        now = snap.now if now is None else now
        pose = snap.pose or {}
        try:
            px = float(pose.get("x_m", pose.get("x", 0.0)))
            py = float(pose.get("y_m", pose.get("y", 0.0)))
            pth = math.radians(float(pose.get("theta_deg", pose.get("theta", 0.0))))
            have_pose = snap.pose is not None
        except (TypeError, ValueError):
            px = py = pth = 0.0
            have_pose = False

        scores: Dict[str, float] = {}
        for s in SECTORS:
            ang = SECTOR_ANGLE_DEG[s]
            if not allow_rear and s in ("Retro", "Retro-DX", "Retro-SX"):
                scores[s] = 0.0
                continue
            d = snap.sector_dist(s, default=MAX_CLEAR_CM)  # settore senza dati = sconosciuto ma aperto
            r = snap.sector_risk(s)
            if d < MIN_PASSABLE_CM or r >= RISK_HIGH:
                scores[s] = 0.0
                continue
            clearance = min(d, MAX_CLEAR_CM) / MAX_CLEAR_CM
            safety = 1.0 - r
            turn_cost = 1.0 - 0.35 * (abs(ang) / 180.0)
            explore = 1.0
            if have_pose:
                # cella target 1.5 m avanti lungo la direzione candidata
                # (bearing orario positivo = destra → in frame matematico theta - ang)
                th = pth - math.radians(ang)
                tx = px + 1.5 * math.cos(th)
                ty = py + 1.5 * math.sin(th)
                if self._visited_recently(tx, ty, now):
                    explore = 0.6
            hyst = 1.0
            if self.last_turn_sign != 0 and ang != 0.0 and (1 if ang > 0 else -1) == self.last_turn_sign:
                hyst = 1.15
            scores[s] = clearance * safety * turn_cost * explore * hyst
        self.last_scores = scores
        return scores

    def choose_heading(self, snap, now: Optional[float] = None) -> Tuple[Optional[str], float]:
        """Ritorna (settore scelto, angolo relativo in gradi). (None, 0) se tutto bloccato."""
        scores = self.score_sectors(snap, now, allow_rear=False)
        best = max(scores.items(), key=lambda kv: kv[1])
        if best[1] <= 0.0:
            scores = self.score_sectors(snap, now, allow_rear=True)
            best = max(scores.items(), key=lambda kv: kv[1])
            if best[1] <= 0.0:
                self.last_choice = None
                return None, 0.0
        sector, _ = best
        ang = SECTOR_ANGLE_DEG[sector]
        if ang != 0.0:
            self.last_turn_sign = 1 if ang > 0 else -1
        self.last_choice = sector
        return sector, ang

    # ---------------------------------------------------------------- speed
    @staticmethod
    def cruise_speed(front_dist_cm: float, front_risk: float, base: int = 50, slow: int = 25) -> int:
        d = float(front_dist_cm)
        if d >= 150.0:
            v = float(base)
        elif d <= DIST_HIGH_CM:
            v = float(slow)
        else:
            v = slow + (base - slow) * (d - DIST_HIGH_CM) / (150.0 - DIST_HIGH_CM)
        v *= max(0.5, 1.0 - max(0.0, min(1.0, front_risk)))
        return int(max(slow if d < 150.0 else slow, min(base, round(v))))

    # ---------------------------------------------------------------- stuck
    def on_forward(self, snap, now: float):
        """Da chiamare ad ogni tick in cui si comanda FORWARD."""
        pose = snap.pose
        if self.forward_since is None:
            self.forward_since = now
            if pose:
                try:
                    self.forward_start_pose = (float(pose.get("x_m", 0.0)), float(pose.get("y_m", 0.0)))
                except (TypeError, ValueError):
                    self.forward_start_pose = None
            else:
                self.forward_start_pose = None

    def on_not_forward(self):
        self.forward_since = None
        self.forward_start_pose = None

    def is_stuck(self, snap, now: float) -> bool:
        """FORWARD da ≥ STUCK_AFTER_S ma la posa SLAM non si è spostata (solo se il robot si è già mosso)."""
        if self.forward_since is None or self.forward_start_pose is None or not snap.pose:
            return False
        if self.total_motion_m < MOTION_SEEN_MIN_M:
            return False
        if now - self.forward_since < STUCK_AFTER_S:
            return False
        try:
            x = float(snap.pose.get("x_m", 0.0))
            y = float(snap.pose.get("y_m", 0.0))
        except (TypeError, ValueError):
            return False
        moved = math.hypot(x - self.forward_start_pose[0], y - self.forward_start_pose[1])
        return moved < STUCK_MIN_DISPLACEMENT_M

    def escape_possible(self, snap) -> bool:
        return snap.sector_dist("Retro", default=MAX_CLEAR_CM) > MIN_PASSABLE_CM and \
            snap.sector_risk("Retro") < RISK_MODERATE + 0.3
