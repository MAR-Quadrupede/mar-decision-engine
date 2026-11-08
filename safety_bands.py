#!/usr/bin/env python3
"""
M.A.R. — Unified distance → risk / command bands + decision stabilizer.

Test matrix (hold a sheet / hand / box in FRONT of the stereo camera):

| Level      | Distance (front) | Expected command | Expected UI risk |
|------------|------------------|------------------|------------------|
| CRITICAL   |  < 40 cm         | STOP             | CRITICAL         |
| HIGH       |  40 – 70 cm      | FRENA            | HIGH             |
| MODERATE   |  70 – 120 cm     | AVANTI (lento)   | MODERATE         |
| LOW        |  > 120 cm        | AVANTI           | LOW              |

LiDAR/FSM use the same centimetre cuts so dashboard, vision and motors agree.

Decision stability (RiskDecisionStabilizer):
  • EMA on distance (no raw frame-to-frame jumps)
  • Hysteresis exit thresholds (clearer than enter → no boundary chatter)
  • Escalate fast after short confirmation (safety)
  • De-escalate one level at a time after longer dwell (pondered)
  • FOV-blocked CRITICAL needs consecutive frames (kills LOW↔CRITICAL flicker)
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

# Centimetri — taratura fisica robot (ispirata al codice definitivo canelidarstereoimudef2)
DIST_CRITICAL_CM = 40.0   # < 40 cm: arresto immediato di collisione
DIST_HIGH_CM = 70.0       # 40-70 cm: deviazione / schivamento dinamico
DIST_MODERATE_CM = 120.0  # 70-120 cm: allerta di prossimità

# Metri (dashboard / MQTT consumers)
DIST_CRITICAL_M = DIST_CRITICAL_CM / 100.0
DIST_HIGH_M = DIST_HIGH_CM / 100.0
DIST_MODERATE_M = DIST_MODERATE_CM / 100.0

# Ordine di severità (indice crescente = peggiore)
RISK_LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")
_RISK_IDX = {name: i for i, name in enumerate(RISK_LEVELS)}

# Isteresi in uscita: bisogna essere più lontani per scendere di livello
EXIT_CRITICAL_CM = DIST_CRITICAL_CM + 12.0   # 52 — leave CRITICAL
EXIT_HIGH_CM = DIST_HIGH_CM + 15.0           # 85 — leave HIGH
EXIT_MODERATE_CM = DIST_MODERATE_CM + 20.0   # 140 — leave MODERATE


def classify_distance_cm(dist_cm: float) -> str:
    """Instantaneous band (no hysteresis). Return CRITICAL | HIGH | MODERATE | LOW."""
    if dist_cm <= 0:
        return "CRITICAL"
    if dist_cm < DIST_CRITICAL_CM:
        return "CRITICAL"
    if dist_cm < DIST_HIGH_CM:
        return "HIGH"
    if dist_cm < DIST_MODERATE_CM:
        return "MODERATE"
    return "LOW"


def classify_distance_m(dist_m: float) -> str:
    return classify_distance_cm(float(dist_m) * 100.0)


def command_for_risk(level: str, dist_cm: float) -> Tuple[str, str, str]:
    """Map a held risk level → (command, reason, risk_level)."""
    level = str(level).upper()
    if level not in _RISK_IDX:
        level = classify_distance_cm(dist_cm)
    # Motivo coerente col livello tenuto (non contraddire Z vs soglia)
    if level == "CRITICAL":
        return "STOP", f"CRITICAL held | Z={dist_cm:.0f}cm (soglia <{DIST_CRITICAL_CM:.0f})", level
    if level == "HIGH":
        return "FRENA", f"HIGH held | Z={dist_cm:.0f}cm (banda {DIST_CRITICAL_CM:.0f}-{DIST_HIGH_CM:.0f})", level
    if level == "MODERATE":
        return "AVANTI", f"MODERATE held | Z={dist_cm:.0f}cm (cautela <{DIST_MODERATE_CM:.0f})", level
    return "AVANTI", f"LOW held | Z={dist_cm:.0f}cm strada libera", level


def command_for_distance_cm(dist_cm: float) -> Tuple[str, str, str]:
    """
    Instantaneous mapping (no stabilizer). Prefer RiskDecisionStabilizer.update()
    for published / motor decisions.
    """
    return command_for_risk(classify_distance_cm(dist_cm), dist_cm)


def _candidate_level_with_hysteresis(dist_cm: float, held: str) -> str:
    """
    Propose a level from smoothed distance, applying exit hysteresis so the
    held level does not chatter at the enter thresholds.
    """
    held = held if held in _RISK_IDX else "LOW"
    raw = classify_distance_cm(dist_cm)
    raw_i = _RISK_IDX[raw]
    held_i = _RISK_IDX[held]

    # Escalation (worse): use enter thresholds as-is
    if raw_i > held_i:
        return raw

    # Same or better raw: only de-escalate if past exit threshold for current hold
    if held == "CRITICAL":
        if dist_cm >= EXIT_CRITICAL_CM:
            return "HIGH" if dist_cm < EXIT_HIGH_CM else (
                "MODERATE" if dist_cm < EXIT_MODERATE_CM else "LOW"
            )
        return "CRITICAL"
    if held == "HIGH":
        if dist_cm >= EXIT_HIGH_CM:
            return "MODERATE" if dist_cm < EXIT_MODERATE_CM else "LOW"
        if dist_cm < DIST_CRITICAL_CM:
            return "CRITICAL"
        return "HIGH"
    if held == "MODERATE":
        if dist_cm >= EXIT_MODERATE_CM:
            return "LOW"
        if dist_cm < DIST_CRITICAL_CM:
            return "CRITICAL"
        if dist_cm < DIST_HIGH_CM:
            return "HIGH"
        return "MODERATE"
    # held LOW
    return raw


class RiskDecisionStabilizer:
    """
    Stateful, pondered risk decision for ~10–30 Hz vision / sensor loops.

    Escalate: short confirmation, can jump to CRITICAL for safety.
    De-escalate: longer dwell, one level at a time (CRITICAL→HIGH→MOD→LOW).
    """

    def __init__(
        self,
        ema_alpha: Optional[float] = None,
        escalate_s: Optional[float] = None,
        deescalate_s: Optional[float] = None,
        fov_on_frames: Optional[int] = None,
        fov_off_frames: Optional[int] = None,
    ):
        self.ema_alpha = float(
            ema_alpha if ema_alpha is not None else os.getenv("MAR_RISK_EMA_ALPHA", "0.35")
        )
        # Confirmation windows (seconds). Escalate faster than de-escalate.
        self.escalate_s = float(
            escalate_s if escalate_s is not None else os.getenv("MAR_RISK_ESCALATE_S", "0.35")
        )
        self.deescalate_s = float(
            deescalate_s if deescalate_s is not None else os.getenv("MAR_RISK_DEESCALATE_S", "0.6")
        )
        self.fov_on_frames = int(
            fov_on_frames if fov_on_frames is not None else os.getenv("MAR_FOV_ON_FRAMES", "5")
        )
        # Sblocca FOV latch con calma quando la scena torna sicuramente valida
        self.fov_off_frames = int(
            fov_off_frames if fov_off_frames is not None else os.getenv("MAR_FOV_OFF_FRAMES", "5")
        )

        self.dist_ema: Optional[float] = None
        self.held_level: str = "LOW"
        self.pending_level: Optional[str] = None
        self.pending_since: float = 0.0
        self.level_since: float = time.time()

        self._fov_true_streak = 0
        self._fov_false_streak = 0
        self.fov_latched = False

    def reset(self, level: str = "LOW", dist_cm: float = 999.0):
        self.dist_ema = float(dist_cm)
        self.held_level = level if level in _RISK_IDX else "LOW"
        self.pending_level = None
        self.pending_since = 0.0
        self.level_since = time.time()
        self._fov_true_streak = 0
        self._fov_false_streak = 0
        self.fov_latched = False

    def _update_fov_latch(self, fov_blocked: bool) -> bool:
        if fov_blocked:
            self._fov_true_streak += 1
            self._fov_false_streak = 0
            if self._fov_true_streak >= self.fov_on_frames:
                self.fov_latched = True
        else:
            self._fov_false_streak += 1
            self._fov_true_streak = 0
            if self._fov_false_streak >= self.fov_off_frames:
                self.fov_latched = False
        return self.fov_latched

    def update(
        self,
        dist_cm: float,
        *,
        fov_blocked: bool = False,
        safety_alarm: bool = False,
        safety_reason: str = "",
        now: Optional[float] = None,
    ) -> Tuple[str, str, str, float]:
        """
        Returns (command, reason, risk_level, dist_smoothed_cm).
        """
        now = time.time() if now is None else float(now)
        raw_dist = float(dist_cm) if dist_cm is not None else 999.0
        if raw_dist <= 0 or raw_dist != raw_dist:  # NaN guard
            raw_dist = 999.0

        # Filtro asimmetrico: avvicinamenti (pericolo) rapidi, allontanamenti lenti
        if self.dist_ema is None:
            self.dist_ema = raw_dist
        else:
            a_near = min(0.75, self.ema_alpha + 0.25)  # più reattivo verso il vicino
            a_far = 0.50 if raw_dist >= 150.0 else max(0.20, self.ema_alpha - 0.10)
            a = a_near if raw_dist < self.dist_ema else a_far
            self.dist_ema = (1.0 - a) * self.dist_ema + a * raw_dist

        fov = self._update_fov_latch(bool(fov_blocked))
        effective = self.dist_ema
        if fov:
            # Treat latched FOV block as near-critical distance (stable, not 1-frame)
            effective = min(effective, DIST_CRITICAL_CM * 0.75)

        if safety_alarm:
            self.held_level = "CRITICAL"
            self.pending_level = None
            self.level_since = now
            motivo = f"SICUREZZA: {safety_reason or 'allarme lenti'}"
            return "STOP", motivo, "CRITICAL", float(effective)

        candidate = _candidate_level_with_hysteresis(effective, self.held_level)
        held_i = _RISK_IDX[self.held_level]
        cand_i = _RISK_IDX[candidate]

        if cand_i == held_i:
            self.pending_level = None
        elif self.pending_level is None:
            self.pending_level = candidate
            self.pending_since = now
        elif cand_i > held_i:
            # Escalation in corso: se peggiora ancora, aggiorna target ma NON azzerare il timer
            if _RISK_IDX.get(self.pending_level, -1) < cand_i:
                self.pending_level = candidate
            # CRITICAL: conferma un filo più lunga del rumore stereo (~3 frame @12fps),
            # ma resta rapida per ostacoli veri. FOV latch / safety_alarm bypassano comunque.
            needed = self.escalate_s
            if candidate == "CRITICAL":
                needed = max(self.escalate_s, 0.28)
            if (now - self.pending_since) >= needed:
                self.held_level = candidate  # può saltare verso CRITICAL (sicurezza)
                self.level_since = now
                self.pending_level = None
        elif cand_i < held_i:
            # De-escalation: se via libera (>150cm), de-escalation rapida
            dwell_needed = 0.35 if effective >= 150.0 else self.deescalate_s
            if self.pending_level != candidate:
                self.pending_level = candidate
                self.pending_since = now
            elif (now - self.pending_since) >= dwell_needed:
                if effective >= 150.0:
                    self.held_level = "LOW"
                else:
                    self.held_level = RISK_LEVELS[held_i - 1]  # un gradino alla volta
                self.level_since = now
                self.pending_level = None

        cmd, motivo, level = command_for_risk(self.held_level, effective)
        if fov and level == "CRITICAL":
            motivo = f"CRITICAL: FOV ostruito (latched) | {motivo}"
        return cmd, motivo, level, float(effective)

    @property
    def held_for_s(self) -> float:
        return max(0.0, time.time() - self.level_since)
