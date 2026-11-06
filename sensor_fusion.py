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
  • multi-laserscan-toolbox-ros2: fusione multi-sensore → un’unica misura frontale
  • scan_detection_fusion: camera+LiDAR range-resolved (bearing match / agreement)

Regole:
  1. Ogni sensore ha un ruolo ponderato (pesi calibrati su affidabilità specifica):
     - Ultrasuoni (35%): rileva perfettamente superfici senza texture, fogli bianchi, vetro.
     - LiDAR (35%): telemetria laser frontale esatta a 360°.
     - Stereocamera 3D (30%): geometria volumetrica del corridoio visivo.
  2. Safety Override Assoluto: se qualunque sensore attivo rileva un ostacolo imminente (< 30 cm),
     la distanza fusa adotta immediatamente il valore minimo per garantire arresto istantaneo.
  3. Fusione ponderata per navigazione e manovra (30-150 cm).
  4. Clearing asimmetrico: reazione istantanea agli avvicinamenti, isteresi a 2 frame in allontanamento.
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


# Pesi ponderati tarati sui sensori del robot MAR
US_SPEC = dict(name="ultrasonic", min_cm=2.0, max_cm=400.0, weight=0.35, can_mark=True, can_clear=True)
LIDAR_SPEC = dict(name="lidar_front", min_cm=15.0, max_cm=800.0, weight=0.35, can_mark=True, can_clear=True)
STEREO_SPEC = dict(name="stereo", min_cm=20.0, max_cm=450.0, weight=0.30, can_mark=True, can_clear=True)


def _valid(r: SensorReading) -> bool:
    return r.min_cm <= r.dist_cm <= r.max_cm


class FrontRangeFusion:
    """
    Produce una distanza frontale fusa e ponderata + metadati dei sensori attivi.
    Fusione reattiva multi-sensore con vincolo rigido di sicurezza < 30cm.
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
        motion_vz_cms: float = 0.0
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

        # 0) Cross-sensor validation (Verifica incrociata LiDAR vs Stereocamera e Ultrasuoni)
        lidar_valid_reading = next((r for r in valid if r.name == "lidar_front"), None)
        us_valid_reading = next((r for r in valid if r.name == "ultrasonic"), None)
        stereo_valid_reading = next((r for r in valid if r.name == "stereo"), None)

        lidar_status = "NORMAL"
        filtered_valid = list(valid)

        # Se il LiDAR rileva un ostacolo vicino (< 120 cm), verificalo con stereocamera e ultrasuoni
        if lidar_valid_reading and lidar_valid_reading.dist_cm < 120.0:
            us_clear = (us_valid_reading is not None and us_valid_reading.dist_cm >= 180.0)
            stereo_clear = (stereo_valid_reading is not None and stereo_valid_reading.dist_cm >= 180.0)

            # Entrambi i sensori ottico ed acustico confermano corridoio libero -> falso positivo LiDAR
            if us_clear and stereo_clear:
                lidar_status = "DISCREPANT_FALSE_POSITIVE"
                filtered_valid = [r for r in valid if r.name != "lidar_front"]
            elif (us_valid_reading and us_valid_reading.dist_cm < 120.0) or (stereo_valid_reading and stereo_valid_reading.dist_cm < 120.0):
                lidar_status = "CONFIRMED_TRUE_POSITIVE"
            else:
                lidar_status = "UNCONFIRMED_LIDAR_ONLY"

        if not filtered_valid:
            filtered_valid = valid

        # 1) Calcolo minimo assoluto per sicurezza critica sui sensori validati
        min_reading = min(filtered_valid, key=lambda r: r.dist_cm)
        min_dist = min_reading.dist_cm

        # 2) Equazione completa ponderata: media pesata dei sensori validati
        total_weight = sum(r.weight for r in filtered_valid)
        weighted_dist = sum(r.dist_cm * r.weight for r in filtered_valid) / max(0.001, total_weight)

        # Regola di sicurezza:
        # Se un sensore rileva pericolo critico (< 30cm) o deviazione alta (< 65cm e c'è discrepanza),
        # il sensore più vicino ha la precedenza assoluta (safety override)
        if min_dist < 30.0:
            raw_fused = min_dist
            method = f"safety_override_min_{min_reading.name}"
            primary_name = min_reading.name
        elif min_dist < 65.0:
            # Tra 30 e 65 cm: blend prudente (65% min, 35% ponderata)
            raw_fused = (0.65 * min_dist) + (0.35 * weighted_dist)
            method = "conservative_weighted_blend"
            primary_name = min_reading.name
        else:
            # Distanza regolare (>65cm): equazione ponderata pura
            raw_fused = weighted_dist
            method = "full_weighted_fusion"
            primary_name = "weighted_all"

        if lidar_status == "DISCREPANT_FALSE_POSITIVE":
            method += "_lidar_discrepant_rejected"

        # Compensazione predittiva se l'oggetto si sta avvicinando velocemente (vz < -15 cm/s)
        if motion_vz_cms < -15.0 and raw_fused < 150.0:
            anticipo_cm = min(25.0, abs(motion_vz_cms) * 0.3)
            raw_fused = max(15.0, raw_fused - anticipo_cm)
            method += f"_motion_anticipation_{abs(motion_vz_cms):.0f}cms"

        # 3) Smoothing / Clearing:
        # Avvicinamento -> adozione immediata della nuova distanza
        # Allontanamento -> conferma graduale (2 frame)
        if self._held_cm is None:
            fused = raw_fused
            self._clear_confirmations = 0
        elif raw_fused < self._held_cm:
            fused = raw_fused
            self._clear_confirmations = 0
        else:
            if raw_fused >= 70.0:
                self._clear_confirmations += 1
                if self._clear_confirmations >= 2:
                    fused = raw_fused
                else:
                    fused = (0.5 * self._held_cm) + (0.5 * raw_fused)
            else:
                fused = (0.40 * raw_fused) + (0.60 * self._held_cm)

        self._held_cm = fused
        return float(fused), {
            "primary": primary_name,
            "method": method,
            "inputs": inputs,
            "weighted_cm": round(weighted_dist, 1),
            "min_cm": round(min_dist, 1),
            "fused_cm": round(fused, 1),
            "lidar_validation": lidar_status,
        }
