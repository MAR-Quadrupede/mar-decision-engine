#!/usr/bin/env python3
"""
M.A.R. — World Model: percezione dello spazio attorno al robot.

Raccoglie i payload MQTT grezzi dei vari servizi, ne gestisce la freschezza e
li proietta sugli 8 settori del robot, poi applica l'equazione di rischio di
`risk_model.py` producendo una "mappa di rischio" a 360° + metriche frontali.

Sorgenti:
  mar/lidar/alert|telemetry|scan   → sector_mins (8 settori), front_distance_m, point_count
  mar/ultrasonic/range, hardware   → us_cm  (solo settore Fronte, ignorato se simulated=true)
  mar/vision/detections            → spatial_grid_3d (SX/Centro/DX → Front-SX/Fronte/Front-DX),
                                     target_dist_cm, objects[] (YOLO: class, conf, dist_cm, bearing_deg)
  mar/vision/command               → risk_level, fov_blocked, safety_alarm
  mar/vision/ground_hazard         → z_basso_cm
  mar/slam/pose                    → posa (x, y, theta) nella stanza
  mar/imu/telemetry                → roll / pitch / yaw
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from risk_model import (
    SECTORS, SECTOR_EN_TO_IT, FRONT_SECTORS,
    W_ULTRASONIC, W_LIDAR, W_STEREO,
    SensorDistance, SectorObject, SectorRisk,
    compute_sector_risk, frontal_risk, frontal_distance, bearing_to_sector, risk_to_level,
    DIST_CRITICAL_CM,
)

# Freschezza massima per sorgente (s)
FRESH_LIDAR_S = 1.0
FRESH_US_S = 1.0
STEREO_MEDIAN_N = 5
FRESH_VISION_S = 1.0
FRESH_OBJECTS_S = 1.5
FRESH_IMU_S = 1.0
FRESH_POSE_S = 2.0


def _to_cm(v: Any) -> Optional[float]:
    """Accetta metri (< 15) o centimetri; None se non numerico/invalid."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f <= 0:
        return None
    return f * 100.0 if f < 15.0 else f


@dataclass
class Snapshot:
    """Fotografia coerente dello stato del mondo ad un istante."""
    now: float
    sector_risks: Dict[str, SectorRisk]
    front_risk: float
    front_level: str
    front_dist_cm: float
    lidar_fresh: bool
    us_fresh: bool
    stereo_fresh: bool
    us_cm: Optional[float]
    lidar_front_cm: Optional[float]
    stereo_front_cm: Optional[float]
    ground_hazard: bool
    z_basso_cm: float
    fov_blocked: bool
    vision_safety_alarm: bool
    vision_risk: str
    motion: Dict[str, Any]
    objects: List[Dict[str, Any]]
    pose: Optional[Dict[str, Any]]
    imu: Optional[Dict[str, Any]]
    lidar_alert: Optional[Dict[str, Any]]
    point_count: int
    degraded: bool = False  # True = solo stereo in linea (nessun LiDAR/US fresco)
    sensors_online: Dict[str, bool] = field(default_factory=dict)

    @property
    def front_reliable_cm(self) -> Optional[float]:
        """Distanza frontale minima dai soli sensori AFFIDABILI (US/LiDAR) sul settore Fronte."""
        f = self.sector_risks.get("Fronte")
        return None if f is None else f.min_reliable_cm

    def gate_distance_cm(self) -> float:
        """
        Distanza usata dalla FSM per le soglie dure (40/70/120 cm):
          • sensori affidabili in linea → il loro minimo frontale (999 se non vedono nulla:
            la stereo entra solo tramite il rischio fuso, non può da sola far scattare le soglie);
          • modalità degradata (solo stereo) → la distanza fusa (la stereo è l'unico occhio).
        """
        rel = self.front_reliable_cm
        if rel is not None:
            return float(rel)
        if not self.degraded:
            return 999.0
        return float(self.front_dist_cm)

    def sector_dist(self, name: str, default: float = 999.0) -> float:
        s = self.sector_risks.get(name)
        if s is None or s.fused_dist_cm is None:
            return default
        return float(s.fused_dist_cm)

    def sector_risk(self, name: str) -> float:
        s = self.sector_risks.get(name)
        return float(s.risk) if s else 0.0

    def to_payload(self) -> Dict[str, Any]:
        return {
            "front_risk": round(self.front_risk, 3),
            "front_level": self.front_level,
            "front_dist_cm": round(self.front_dist_cm, 1) if self.front_dist_cm < 900 else None,
            "sectors": {k: v.to_dict() for k, v in self.sector_risks.items()},
            "sensors": {
                "lidar": self.lidar_fresh,
                "ultrasonic": self.us_fresh,
                "stereo": self.stereo_fresh,
                "imu": self.imu is not None,
                "slam": self.pose is not None,
            },
            "degraded": self.degraded,
            "gate_distance_cm": round(self.gate_distance_cm(), 1) if self.gate_distance_cm() < 900 else None,
            "inputs": {
                "ultrasonic_cm": round(self.us_cm, 1) if self.us_cm else None,
                "lidar_front_cm": round(self.lidar_front_cm, 1) if self.lidar_front_cm else None,
                "stereo_front_cm": round(self.stereo_front_cm, 1) if self.stereo_front_cm else None,
            },
            "weights": {"ultrasonic": W_ULTRASONIC, "lidar": W_LIDAR, "stereo": W_STEREO},
            "objects": self.objects,
            "ground_hazard": self.ground_hazard,
            "fov_blocked": self.fov_blocked,
            "vision_risk": self.vision_risk,
        }


class WorldModel:
    def __init__(self):
        self.lidar: Optional[Dict[str, Any]] = None
        self.lidar_ts: float = 0.0
        self.us_cm: float = -1.0
        self.us_ts: float = 0.0
        self.us_simulated: bool = False
        self.vision: Optional[Dict[str, Any]] = None
        self.vision_ts: float = 0.0
        self.vision_cmd: Optional[Dict[str, Any]] = None
        self.vision_cmd_ts: float = 0.0
        self.objects: List[Dict[str, Any]] = []
        self.objects_ts: float = 0.0
        self.ground_hazard: Optional[Dict[str, Any]] = None
        self.ground_hazard_ts: float = 0.0
        self.motion: Dict[str, Any] = {}
        self.motion_ts: float = 0.0
        self.pose: Optional[Dict[str, Any]] = None
        self.pose_ts: float = 0.0
        self.imu: Optional[Dict[str, Any]] = None
        self.imu_ts: float = 0.0
        self._stereo_near_streak: int = 0
        # mediana corta (5 frame ≈ 0.25 s a 20 Hz) sulle colonne stereo: uccide i falsi positivi
        # di singolo frame senza aggiungere latenza percepibile
        self._stereo_hist: Dict[str, deque] = {c: deque(maxlen=STEREO_MEDIAN_N) for c in ("SX", "Centro", "DX")}

    # ------------------------------------------------------------------ stereo median
    def _push_stereo_columns(self, data: Dict[str, Any]) -> None:
        grid = data.get("spatial_grid_3d") or {}
        tgt = _to_cm(data.get("target_dist_cm"))
        for col in ("SX", "Centro", "DX"):
            vals = []
            for row in ("Centro", "Basso"):  # "Alto" = soffitto/porte: non blocca il corpo
                cell = grid.get(f"{row}_{col}")
                if isinstance(cell, (list, tuple)) and len(cell) >= 3:
                    z = _to_cm(cell[2])
                    if z is not None and 20.0 < z < 450.0:
                        vals.append(z)
            if col == "Centro" and tgt is not None and tgt < 450.0:
                vals.append(tgt)
            # 999 = "nulla in vista" così la mediana può anche liberare il settore
            self._stereo_hist[col].append(min(vals) if vals else 999.0)

    def _stereo_col_cm(self, col: str) -> Optional[float]:
        h = self._stereo_hist.get(col)
        if not h:
            return None
        vals = sorted(h)
        med = vals[len(vals) // 2]
        return med if med < 450.0 else None

    # ------------------------------------------------------------------ ingest
    def ingest(self, topic: str, data: Dict[str, Any], now: Optional[float] = None) -> bool:
        """Ritorna True se il payload è rilevante per una nuova decisione."""
        now = time.time() if now is None else now
        if not isinstance(data, dict):
            return False

        if topic in ("mar/lidar/alert", "mar/lidar/telemetry"):
            if self.lidar is None:
                self.lidar = {}
            self.lidar.update(data)
            self.lidar_ts = now
            return True

        if topic == "mar/lidar/scan":
            if self.lidar is None:
                self.lidar = {}
            for k in ("sector_mins", "sector_mins_en", "front_distance_m", "point_count", "slam"):
                if k in data:
                    self.lidar[k] = data[k]
            self.lidar_ts = now
            if isinstance(data.get("slam"), dict) and self.pose is None:
                self.pose = {"x_m": data["slam"].get("x_m"), "y_m": data["slam"].get("y_m"),
                             "theta_deg": data["slam"].get("theta_deg")}
                self.pose_ts = now
            return True

        if topic in ("mar/ultrasonic/range", "mar/hardware/telemetry"):
            simulated = bool(data.get("simulated", False)) or str(data.get("source", "")).upper().startswith("SIM")
            for key in ("us_cm", "ultrasonic_cm", "ultrasonic", "us"):
                v = data.get(key)
                if v is None:
                    continue
                try:
                    cm = float(v)
                except (TypeError, ValueError):
                    continue
                if 2.0 <= cm <= 400.0:
                    if simulated:
                        # letture simulate: NON entrano nella fusione di sicurezza
                        self.us_simulated = True
                        return False
                    self.us_simulated = False
                    self.us_cm = cm
                    self.us_ts = now
                    return True
            # telemetria Arduino con roll/pitch (fallback se non c'è l'IMU I2C)
            if "roll" in data and "pitch" in data and self.imu is None:
                try:
                    self.imu = {"roll": float(data["roll"]), "pitch": float(data["pitch"]), "timestamp": now}
                    self.imu_ts = now
                except (TypeError, ValueError):
                    pass
            return False

        if topic == "mar/vision/detections":
            self.vision = data
            self.vision_ts = now
            self._push_stereo_columns(data)
            if isinstance(data.get("objects"), list):
                self.objects = data["objects"]
                self.objects_ts = now
            if isinstance(data.get("motion"), dict):
                self.motion = data["motion"]
                self.motion_ts = now
            return True

        if topic == "mar/vision/command":
            self.vision_cmd = data
            self.vision_cmd_ts = now
            if isinstance(data.get("motion"), dict):
                self.motion = data["motion"]
                self.motion_ts = now
            return True

        if topic == "mar/vision/motion":
            self.motion = data
            self.motion_ts = now
            return True

        if topic == "mar/vision/ground_hazard":
            self.ground_hazard = data
            self.ground_hazard_ts = now
            return True

        if topic == "mar/slam/pose":
            self.pose = data
            self.pose_ts = now
            return False

        if topic == "mar/imu/telemetry":
            self.imu = data
            self.imu_ts = now
            return False

        return False

    def set_imu(self, fused: Dict[str, Any], now: Optional[float] = None):
        self.imu = fused
        self.imu_ts = time.time() if now is None else now

    # --------------------------------------------------------------- snapshot
    def snapshot(self, now: Optional[float] = None) -> Snapshot:
        now = time.time() if now is None else now

        lidar_fresh = self.lidar is not None and (now - self.lidar_ts) <= FRESH_LIDAR_S
        us_fresh = self.us_cm > 0 and (now - self.us_ts) <= FRESH_US_S
        stereo_fresh = self.vision is not None and (now - self.vision_ts) <= FRESH_VISION_S
        cmd_fresh = self.vision_cmd is not None and (now - self.vision_cmd_ts) <= FRESH_VISION_S
        objs_fresh = (now - self.objects_ts) <= FRESH_OBJECTS_S and bool(self.objects)
        imu = self.imu if (self.imu is not None and (now - self.imu_ts) <= FRESH_IMU_S) else None
        pose = self.pose if (self.pose is not None and (now - self.pose_ts) <= FRESH_POSE_S) else None
        motion = self.motion if (now - self.motion_ts) <= FRESH_VISION_S else {}

        # --- LiDAR per settore
        lidar_sectors: Dict[str, float] = {}
        lidar_front_cm: Optional[float] = None
        point_count = 0
        if lidar_fresh:
            mins = self.lidar.get("sector_mins") or {}
            if not mins and self.lidar.get("sector_mins_en"):
                mins = {SECTOR_EN_TO_IT.get(k, k): v for k, v in self.lidar["sector_mins_en"].items()}
            for s in SECTORS:
                cm = _to_cm(mins.get(s))
                if cm is not None:
                    lidar_sectors[s] = cm
            lidar_front_cm = _to_cm(self.lidar.get("front_distance_m"))
            if lidar_front_cm is None:
                lidar_front_cm = lidar_sectors.get("Fronte")
            try:
                point_count = int(self.lidar.get("point_count") or 0)
            except (TypeError, ValueError):
                point_count = 0

        # --- Stereo: colonne SX / Centro / DX → settori frontali
        stereo_cols: Dict[str, float] = {}
        stereo_front_cm: Optional[float] = None
        ground_hazard = False
        z_basso = 999.0
        if stereo_fresh:
            zc = self._stereo_col_cm("Centro")
            zl = self._stereo_col_cm("SX")
            zr = self._stereo_col_cm("DX")
            if zc is not None:
                stereo_cols["Fronte"] = zc
                stereo_front_cm = zc
            if zl is not None:
                stereo_cols["Front-SX"] = zl
            if zr is not None:
                stereo_cols["Front-DX"] = zr
            zb = _to_cm(self.vision.get("z_basso_cm"))
            if zb is not None:
                z_basso = zb
            ground_hazard = bool(self.vision.get("ground_hazard", False))

        if self.ground_hazard and (now - self.ground_hazard_ts) <= FRESH_VISION_S:
            ground_hazard = True
            zb = _to_cm(self.ground_hazard.get("z_basso_cm"))
            if zb is not None:
                z_basso = min(z_basso, zb)

        fov_blocked = False
        vision_alarm = False
        vision_risk = ""
        if cmd_fresh:
            fov_blocked = bool(self.vision_cmd.get("fov_blocked", False))
            vision_alarm = bool(self.vision_cmd.get("safety_alarm", False))
            vision_risk = str(self.vision_cmd.get("risk_level", "") or "").upper()
        elif stereo_fresh:
            vision_risk = str(self.vision.get("risk_level", "") or "").upper()

        # --- Oggetti YOLO → settori
        sector_objects: Dict[str, List[SectorObject]] = {s: [] for s in SECTORS}
        objects_out: List[Dict[str, Any]] = []
        if objs_fresh:
            for o in self.objects:
                if not isinstance(o, dict):
                    continue
                label = str(o.get("class") or o.get("label") or "object")
                try:
                    conf = float(o.get("confidence", o.get("conf", 0.5)))
                except (TypeError, ValueError):
                    conf = 0.5
                bearing = o.get("bearing_deg")
                try:
                    bearing = float(bearing) if bearing is not None else 0.0
                except (TypeError, ValueError):
                    bearing = 0.0
                sector = str(o.get("sector") or bearing_to_sector(bearing))
                if sector not in sector_objects:
                    sector = bearing_to_sector(bearing)
                d = _to_cm(o.get("dist_cm", o.get("estimated_distance_cm")))
                src = "stereo" if d is not None else "none"
                sector_objects[sector].append(SectorObject(label, conf, d, bearing, src))
                objects_out.append({
                    "class": label, "confidence": round(conf, 2), "sector": sector,
                    "bearing_deg": round(bearing, 1),
                    "dist_cm": round(d, 1) if d is not None else None,
                    "bbox": o.get("bbox"),
                })

        # --- Equazione di rischio per ogni settore
        # Modalità degradata: nessun sensore affidabile (LiDAR/US) in linea → la stereo è l'unico
        # occhio del robot e il suo peso sale a 1.0 (la FSM limita comunque la velocità).
        degraded = not (lidar_fresh or us_fresh)
        w_stereo = W_LIDAR if degraded else W_STEREO
        sector_risks: Dict[str, SectorRisk] = {}
        for s in SECTORS:
            readings: List[SensorDistance] = []
            if s in lidar_sectors:
                readings.append(SensorDistance("lidar", lidar_sectors[s], W_LIDAR, True))
            if s == "Fronte" and us_fresh:
                readings.append(SensorDistance("ultrasonic", self.us_cm, W_ULTRASONIC, True))
            if s in stereo_cols:
                readings.append(SensorDistance("stereo", stereo_cols[s], w_stereo, False))
            sector_risks[s] = compute_sector_risk(s, readings, sector_objects[s])

        # FOV stereo ostruito confermato dal servizio visione (foglio/mano a contatto):
        # senza LiDAR/US vicini vale come HIGH (non da sola CRITICAL)
        if fov_blocked and "Fronte" in sector_risks:
            f = sector_risks["Fronte"]
            if f.risk < 0.6:
                f.risk = 0.6
                f.level = risk_to_level(f.risk)
                f.note += "+fov_blocked"

        front_risk, front_level = frontal_risk(sector_risks)
        front_dist = frontal_distance(sector_risks)

        return Snapshot(
            now=now,
            sector_risks=sector_risks,
            front_risk=front_risk,
            front_level=front_level,
            front_dist_cm=front_dist,
            lidar_fresh=lidar_fresh,
            us_fresh=us_fresh,
            stereo_fresh=stereo_fresh,
            us_cm=self.us_cm if us_fresh else None,
            lidar_front_cm=lidar_front_cm,
            stereo_front_cm=stereo_front_cm,
            ground_hazard=ground_hazard,
            z_basso_cm=z_basso,
            fov_blocked=fov_blocked,
            vision_safety_alarm=vision_alarm,
            vision_risk=vision_risk,
            motion=motion,
            objects=objects_out,
            pose=pose,
            imu=imu,
            lidar_alert=self.lidar if lidar_fresh else None,
            point_count=point_count,
            degraded=degraded,
            sensors_online={"lidar": lidar_fresh, "ultrasonic": us_fresh, "stereo": stereo_fresh,
                            "imu": imu is not None, "slam": pose is not None},
        )
