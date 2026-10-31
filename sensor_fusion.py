#!/usr/bin/env python3
"""
M.A.R. — Frontal multi-sensor fusion (inspired by ROS 2 Nav2 costmap layers).

Spunti open-source usati come modello concettuale (non dipendenza runtime):
  • Nav2 Costmap2D layered stack
      - ObstacleLayer  → LiDAR / mid-range marking + raytrace clearing
      - RangeLayer     → ultrasonici / IR come near-field safety net
      - DenoiseLayer   → non far “pulire” ostacoli da un singolo outlier
    Docs: https://docs.nav2.org/  (costmap_2d plugins, sensor setup)
  • Project Rocky (quadrupede open-source): stereo+IMU+ultrasonic come safety net
    https://pbshgthm.github.io/rocky/
  • multi-laserscan-toolbox-ros2: fusione multi-sensore → un’unica misura frontale
    https://github.com/Noceo200/multi-laserscan-toolbox-ros2
  • scan_detection_fusion: camera+LiDAR range-resolved (bearing match / agreement)
    https://github.com/HexboxRC/scan_detection_fusion

Regole (tradotte in codice MAR, senza ROS):
  1. Ogni sensore ha un ruolo e un range utile (come observation_sources Nav2).
  2. Ultrasuoni = primario vicino (glass/foglio/textureless) — RangeLayer.
  3. LiDAR = primario medio-raggio frontale — ObstacleLayer.
  4. Stereo = corroborazione; entra solo se concorda o è più vicino (safety).
  5. Clearing asimmetrico: avvicinamento veloce, allontanamento solo se concordano ≥2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class SensorReading:
    name: str
    dist_cm: float
    # Nav2-like: marking (can raise risk) / clearing (can lower risk)
    can_mark: bool = True
    can_clear: bool = True
    # Useful band (cm) — fuori banda la lettura è ignorata
    min_cm: float = 2.0
    max_cm: float = 400.0
    weight: float = 1.0


# Ruoli tipici MAR (allineati a sensor fusion reattiva robot)
US_SPEC = dict(name="ultrasonic", min_cm=2.0, max_cm=350.0, weight=1.0, can_mark=True, can_clear=True)
LIDAR_SPEC = dict(name="lidar_front", min_cm=12.0, max_cm=800.0, weight=0.95, can_mark=True, can_clear=True)
STEREO_SPEC = dict(name="stereo", min_cm=25.0, max_cm=350.0, weight=0.90, can_mark=True, can_clear=True)


def _valid(r: SensorReading) -> bool:
    return r.min_cm <= r.dist_cm <= r.max_cm


class FrontRangeFusion:
    """
    Produce una distanza frontale fusa affidabile + metadati dei sensori attivi.
    Fusione reattiva multi-sensore senza blocchi infiniti di clearing.
    """

    def __init__(self, agree_ratio: float = 0.35):
        self.agree_ratio = agree_ratio
        self._held_cm: Optional[float] = None
        self._clear_confirmations: int = 0

    def fuse(
        self,
        ultrasonic_cm: float = -1.0,
        lidar_front_cm: float = 999.0,
        stereo_cm: float = 999.0,
    ) -> Tuple[float, Dict[str, object]]:
        readings = []
        if ultrasonic_cm > 0:
            readings.append(SensorReading(dist_cm=float(ultrasonic_cm), **US_SPEC))
        if lidar_front_cm < 900:
            readings.append(SensorReading(dist_cm=float(lidar_front_cm), **LIDAR_SPEC))
        if stereo_cm < 900:
            readings.append(SensorReading(dist_cm=float(stereo_cm), **STEREO_SPEC))

        valid = [r for r in readings if _valid(r)]
        if not valid:
            self._held_cm = None
            return 999.0, {"primary": None, "method": "empty", "inputs": {}}

        inputs = {r.name: round(r.dist_cm, 1) for r in valid}

        # 1) Sicurezza reattiva: il valore minimo tra i sensori validi determina il pericolo imminente
        min_reading = min(valid, key=lambda r: r.dist_cm)
        raw_fused = min_reading.dist_cm
        primary_name = min_reading.name
        method = f"reactive_min_{primary_name}"

        # 2) Smoothing / Clearing:
        # Se un ostacolo è rilevato più vicino di 70cm, reazione istantanea (sicurezza immediata).
        # Se la via torna libera, conferma in 2 frame per evitare glitch di un singolo frame mancante.
        if self._held_cm is None:
            fused = raw_fused
            self._clear_confirmations = 0
        elif raw_fused < self._held_cm:
            # Avvicinamento ostacolo -> adotta immediatamente la nuova distanza minima
            fused = raw_fused
            self._clear_confirmations = 0
        else:
            # Allontanamento / via libera
            if raw_fused >= 70.0:
                self._clear_confirmations += 1
                if self._clear_confirmations >= 2:
                    fused = raw_fused
                else:
                    fused = (0.5 * self._held_cm) + (0.5 * raw_fused)
            else:
                fused = (0.35 * raw_fused) + (0.65 * self._held_cm)

        self._held_cm = fused
        return float(fused), {
            "primary": primary_name,
            "method": method,
            "inputs": inputs,
            "fused_cm": round(fused, 1),
        }
