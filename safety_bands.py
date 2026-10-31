#!/usr/bin/env python3
"""
M.A.R. — Distance bands + RiskDecisionStabilizer (mirror of mar-vision-service).
Keep thresholds in sync with vision.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

DIST_CRITICAL_CM = 40.0
DIST_HIGH_CM = 70.0
DIST_MODERATE_CM = 120.0

EXIT_CRITICAL_CM = DIST_CRITICAL_CM + 12.0
EXIT_HIGH_CM = DIST_HIGH_CM + 15.0
EXIT_MODERATE_CM = DIST_MODERATE_CM + 20.0

RISK_LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")
_RISK_IDX = {name: i for i, name in enumerate(RISK_LEVELS)}


def classify_distance_cm(dist_cm: float) -> str:
    if dist_cm <= 0:
        return "CRITICAL"
    if dist_cm < DIST_CRITICAL_CM:
        return "CRITICAL"
    if dist_cm < DIST_HIGH_CM:
        return "HIGH"
    if dist_cm < DIST_MODERATE_CM:
        return "MODERATE"
    return "LOW"


def _candidate_level_with_hysteresis(dist_cm: float, held: str) -> str:
    held = held if held in _RISK_IDX else "LOW"
    raw = classify_distance_cm(dist_cm)
    raw_i = _RISK_IDX[raw]
    held_i = _RISK_IDX[held]
    if raw_i > held_i:
        return raw
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
    return raw


class RiskDecisionStabilizer:
    def __init__(
        self,
        ema_alpha: float = 0.30,
        escalate_s: float = 0.25,
        deescalate_s: float = 1.0,
    ):
        self.ema_alpha = float(os.getenv("MAR_RISK_EMA_ALPHA", str(ema_alpha)))
        self.escalate_s = float(os.getenv("MAR_RISK_ESCALATE_S", str(escalate_s)))
        self.deescalate_s = float(os.getenv("MAR_RISK_DEESCALATE_S", str(deescalate_s)))
        self.dist_ema: Optional[float] = None
        self.held_level: str = "LOW"
        self.pending_level: Optional[str] = None
        self.pending_since: float = 0.0

    def update(self, dist_cm: float, now: Optional[float] = None) -> Tuple[str, float]:
        """Returns (held_risk_level, smoothed_dist_cm)."""
        now = time.time() if now is None else float(now)
        raw = float(dist_cm) if dist_cm is not None else 999.0
        if raw <= 0 or raw != raw:
            raw = 999.0
        if self.dist_ema is None:
            self.dist_ema = raw
        else:
            a_near = min(0.75, self.ema_alpha + 0.25)
            a_far = max(0.12, self.ema_alpha - 0.15)
            a = a_near if raw < self.dist_ema else a_far
            self.dist_ema = (1.0 - a) * self.dist_ema + a * raw

        candidate = _candidate_level_with_hysteresis(self.dist_ema, self.held_level)
        held_i = _RISK_IDX[self.held_level]
        cand_i = _RISK_IDX[candidate]

        if cand_i == held_i:
            self.pending_level = None
        elif self.pending_level is None:
            self.pending_level = candidate
            self.pending_since = now
        elif cand_i > held_i:
            if _RISK_IDX.get(self.pending_level, -1) < cand_i:
                self.pending_level = candidate
            needed = min(self.escalate_s, 0.15) if candidate == "CRITICAL" else self.escalate_s
            if (now - self.pending_since) >= needed:
                self.held_level = candidate
                self.pending_level = None
        elif cand_i < held_i:
            if self.pending_level != candidate:
                self.pending_level = candidate
                self.pending_since = now
            elif (now - self.pending_since) >= self.deescalate_s:
                self.held_level = RISK_LEVELS[held_i - 1]
                self.pending_level = None

        return self.held_level, float(self.dist_ema)
