#!/usr/bin/env python3
"""
M.A.R. — Equazione di rischio multi-sensore (settore per settore).

Tutto il sistema decisionale ruota attorno a UNA equazione per settore s
(8 settori LiDAR: Fronte, Front-DX, Destra, Retro-DX, Retro, Retro-SX, Sinistra, Front-SX):

    R_s = 1 - (1 - P_s) * (1 - Q_s)                      (OR probabilistico)

    P_s = prossimità = 1 - Π_i ( 1 - w_i · p(d_i) )       per ogni sensore i che vede il settore
    Q_s = oggetti    = max_o  w_class(o) · conf(o) · p(d_o)   per gli oggetti YOLO nel settore

    p(d) = 1 (d ≤ 30cm) → 0.85 (40cm) → 0.60 (70cm) → 0.30 (80cm) → 0 (≥120cm), lineare a tratti.
           Sotto 80 cm c'è cautela vera; un muro o un sensore a ~1 m non tengono il robot in MODERATE.

Pesi w_i = quanto ci fidiamo della DISTANZA riportata dal sensore i:

    ultrasuoni  1.00   (solo Fronte, cono ±15°, perfetto su vetro/pareti lisce)
    LiDAR       1.00   (tutti gli 8 settori)
    stereo      0.60   (Front-SX / Fronte / Front-DX: tanti falsi positivi → vale meno)
                       → da sola la stereo arriva al massimo a 0.60 = HIGH, mai CRITICAL;
                       → se un oggetto YOLO è nello stesso settore la sua distanza vale 1.0
                         (conferma indipendente: la stereo "ha visto giusto").

Regola di sicurezza: un sensore AFFIDABILE (US/LiDAR) sotto D_CRITICAL (40cm) → P_s = 1 (override).

Gli oggetti YOLO contano a PIENO peso come presenza (w_class · conf): per la loro distanza si
preferisce il LiDAR del settore (affidabile) e si usa la stereo solo come fallback; senza alcuna
distanza vale una presenza conservativa p = 0.45 (le entità dinamiche hanno un minimo garantito).

d_fused_s (solo per soglie FSM e telemetria) = media pesata, con il minimo affidabile che non può
essere "diluito" oltre il 25% e con la stereo sola clampata a ≥ 40cm senza conferma oggetto.

Livelli (coerenti con safety_bands):  CRITICAL ≥ 0.85 | HIGH ≥ 0.60 | MODERATE ≥ 0.30 | LOW
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from safety_bands import DIST_CRITICAL_CM, DIST_HIGH_CM, DIST_MODERATE_CM
except ImportError:  # pragma: no cover
    from .safety_bands import DIST_CRITICAL_CM, DIST_HIGH_CM, DIST_MODERATE_CM

SECTORS: Tuple[str, ...] = (
    "Fronte", "Front-DX", "Destra", "Retro-DX",
    "Retro", "Retro-SX", "Sinistra", "Front-SX",
)
SECTOR_ANGLE_DEG: Dict[str, float] = {
    "Fronte": 0.0, "Front-DX": 45.0, "Destra": 90.0, "Retro-DX": 135.0,
    "Retro": 180.0, "Retro-SX": -135.0, "Sinistra": -90.0, "Front-SX": -45.0,
}
SECTOR_EN_TO_IT: Dict[str, str] = {
    "FRONT": "Fronte", "FRONT_RIGHT": "Front-DX", "RIGHT": "Destra", "REAR_RIGHT": "Retro-DX",
    "REAR": "Retro", "REAR_LEFT": "Retro-SX", "LEFT": "Sinistra", "FRONT_LEFT": "Front-SX",
}
FRONT_SECTORS: Tuple[str, ...] = ("Front-SX", "Fronte", "Front-DX")

# Banda della funzione di prossimità p(d)
D_NEAR_CM = 30.0
D_FAR_CM = 120.0

# Soglie rischio → livello
RISK_CRITICAL = 0.85
RISK_HIGH = 0.60
RISK_MODERATE = 0.30

# Pesi di affidabilità della DISTANZA per sensore
W_ULTRASONIC = 1.00
W_LIDAR = 1.00
W_STEREO = 0.60

# Peso di pericolosità per classe YOLO (COCO). Default 0.6 per classi sconosciute.
CLASS_WEIGHTS: Dict[str, float] = {
    # entità dinamiche: massima priorità
    "person": 1.00, "dog": 1.00, "cat": 1.00, "bird": 0.80, "horse": 1.00, "sheep": 0.90, "cow": 1.00,
    # ostacoli rigidi tipici indoor
    "chair": 0.85, "couch": 0.85, "bed": 0.85, "dining table": 0.85, "bench": 0.85, "toilet": 0.80,
    "tv": 0.75, "refrigerator": 0.85, "potted plant": 0.75, "suitcase": 0.80, "backpack": 0.70,
    "bicycle": 0.90, "motorcycle": 0.90, "car": 1.00, "truck": 1.00, "bus": 1.00,
    # oggetti piccoli / da tavolo: rischio limitato ma non nullo (possono essere a terra)
    "bottle": 0.50, "cup": 0.40, "book": 0.40, "laptop": 0.55, "keyboard": 0.40, "mouse": 0.30,
    "cell phone": 0.30, "remote": 0.30, "vase": 0.55, "bowl": 0.35, "clock": 0.25,
    "sports ball": 0.60, "handbag": 0.60, "umbrella": 0.60, "skateboard": 0.70,
}
DEFAULT_CLASS_WEIGHT = 0.60
DYNAMIC_CLASSES = {"person", "dog", "cat", "bird", "horse", "sheep", "cow", "bicycle", "motorcycle", "car"}


# p(d) lineare a tratti, ancorata alle bande di safety_bands così che
# livello(p(d)) == classify_distance_cm(d):  40cm→0.85  70cm→0.60  80cm→0.30  ≥120cm→0
_P_KNOTS: Tuple[Tuple[float, float], ...] = (
    (D_NEAR_CM, 1.0),
    (DIST_CRITICAL_CM, RISK_CRITICAL),
    (DIST_HIGH_CM, RISK_HIGH),
    (DIST_MODERATE_CM, RISK_MODERATE),
    (D_FAR_CM, 0.0),
)


def proximity(d_cm: Optional[float], near: float = D_NEAR_CM, far: float = D_FAR_CM) -> float:
    """p(d): 1.0 a contatto (≤ near), 0.0 oltre far, lineare a tratti sulle bande. None/invalid → 0."""
    if d_cm is None:
        return 0.0
    try:
        d = float(d_cm)
    except (TypeError, ValueError):
        return 0.0
    if d != d or d <= 0.0 or d >= 900.0:
        return 0.0
    if d <= near:
        return 1.0
    if d >= far:
        return 0.0
    for (d0, p0), (d1, p1) in zip(_P_KNOTS[:-1], _P_KNOTS[1:]):
        if d0 <= d <= d1:
            return p0 + (p1 - p0) * (d - d0) / (d1 - d0)
    return 0.0


def class_weight(label: str) -> float:
    return CLASS_WEIGHTS.get(str(label).lower().strip(), DEFAULT_CLASS_WEIGHT)


def risk_to_level(r: float) -> str:
    if r >= RISK_CRITICAL:
        return "CRITICAL"
    if r >= RISK_HIGH:
        return "HIGH"
    if r >= RISK_MODERATE:
        return "MODERATE"
    return "LOW"


def or_combine(*probs: float) -> float:
    """OR probabilistico di evidenze indipendenti: 1 - Π(1 - p)."""
    acc = 1.0
    for p in probs:
        p = min(1.0, max(0.0, float(p)))
        acc *= (1.0 - p)
    return 1.0 - acc


def bearing_to_sector(bearing_deg: float) -> str:
    """Bearing robot (0 = fronte, positivo = destra/orario) → settore 45°."""
    a = float(bearing_deg) % 360.0
    if a >= 337.5 or a < 22.5:
        return "Fronte"
    if a < 67.5:
        return "Front-DX"
    if a < 112.5:
        return "Destra"
    if a < 157.5:
        return "Retro-DX"
    if a < 202.5:
        return "Retro"
    if a < 247.5:
        return "Retro-SX"
    if a < 292.5:
        return "Sinistra"
    return "Front-SX"


@dataclass
class SensorDistance:
    source: str          # "ultrasonic" | "lidar" | "stereo"
    dist_cm: float
    weight: float
    reliable: bool       # True: può imporre l'override di sicurezza


@dataclass
class SectorObject:
    label: str
    confidence: float
    dist_cm: Optional[float]     # distanza stimata dell'oggetto (stereo o LiDAR del settore)
    bearing_deg: float = 0.0
    dist_source: str = "none"

    @property
    def weight(self) -> float:
        return class_weight(self.label)

    @property
    def dynamic(self) -> bool:
        return str(self.label).lower() in DYNAMIC_CLASSES


@dataclass
class SectorRisk:
    sector: str
    risk: float
    level: str
    proximity_risk: float
    object_risk: float
    fused_dist_cm: Optional[float]
    min_reliable_cm: Optional[float]
    inputs: Dict[str, float] = field(default_factory=dict)
    objects: List[Dict] = field(default_factory=list)
    stereo_only_near: bool = False
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            "risk": round(self.risk, 3),
            "level": self.level,
            "p": round(self.proximity_risk, 3),
            "q": round(self.object_risk, 3),
            "dist_cm": round(self.fused_dist_cm, 1) if self.fused_dist_cm is not None else None,
            "min_reliable_cm": round(self.min_reliable_cm, 1) if self.min_reliable_cm is not None else None,
            "inputs": {k: round(v, 1) for k, v in self.inputs.items()},
            "objects": self.objects,
            "note": self.note,
        }


def fuse_sector_distance(
    readings: List[SensorDistance],
    has_object_confirmation: bool = False,
) -> Tuple[Optional[float], Optional[float], bool, str]:
    """
    Fusione ponderata della distanza di un settore.
    Ritorna (d_fused, min_reliable, stereo_only_near, nota).
    """
    valid = [r for r in readings if r.dist_cm is not None and 0.0 < r.dist_cm < 900.0]
    if not valid:
        return None, None, False, "no_data"

    reliable = [r for r in valid if r.reliable]
    min_reliable = min((r.dist_cm for r in reliable), default=None)
    stereo = [r for r in valid if not r.reliable]
    min_stereo = min((r.dist_cm for r in stereo), default=None)

    # Regola 1: sensore affidabile sotto soglia critica → override immediato sul minimo
    if min_reliable is not None and min_reliable < DIST_CRITICAL_CM:
        return float(min_reliable), float(min_reliable), False, "safety_override_reliable"

    # Media ponderata
    tot_w = sum(r.weight for r in valid)
    d_w = sum(r.dist_cm * r.weight for r in valid) / max(1e-6, tot_w)

    # Il sensore affidabile più vicino non può essere "diluito" oltre la banda HIGH
    # (es. LiDAR 60cm + stereo 300cm non devono dare 150cm "liberi")
    if min_reliable is not None:
        d_w = min(d_w, min_reliable * 1.25)

    # Regola 2: SOLO la stereo vede qualcosa di critico → non può da sola dichiarare CRITICAL
    stereo_only_near = False
    note = "weighted"
    if min_stereo is not None and min_stereo < DIST_CRITICAL_CM and (
        min_reliable is None or min_reliable >= DIST_HIGH_CM
    ):
        if has_object_confirmation:
            d_w = min(d_w, min_stereo)
            note = "stereo_near_confirmed_by_object"
        else:
            d_w = min(d_w, max(DIST_CRITICAL_CM, min_stereo * 1.3))
            stereo_only_near = True
            note = "stereo_only_near_clamped"
    elif min_stereo is not None and min_reliable is None:
        note = "stereo_only"

    return float(d_w), (float(min_reliable) if min_reliable is not None else None), stereo_only_near, note


def proximity_term(readings: List[SensorDistance], stereo_weight_override: Optional[float] = None) -> float:
    """
    P_s = 1 - Π_i (1 - w_i · p(d_i)).  Un sensore affidabile sotto D_CRITICAL → 1.0.
    `stereo_weight_override` alza il peso della stereo (es. 1.0 quando un oggetto YOLO conferma).
    """
    survive = 1.0
    for r in readings:
        if r.dist_cm is None or not (0.0 < r.dist_cm < 900.0):
            continue
        if r.reliable and r.dist_cm < DIST_CRITICAL_CM:
            return 1.0
        w = r.weight
        if not r.reliable and stereo_weight_override is not None:
            w = stereo_weight_override
        survive *= 1.0 - max(0.0, min(1.0, w)) * proximity(r.dist_cm)
    return max(0.0, min(1.0, 1.0 - survive))


def compute_sector_risk(
    sector: str,
    readings: List[SensorDistance],
    objects: List[SectorObject],
) -> SectorRisk:
    """Applica l'equazione R_s = 1 - (1-P_s)(1-Q_s) a un settore."""
    has_obj = len(objects) > 0
    d_fused, min_rel, stereo_only_near, note = fuse_sector_distance(readings, has_object_confirmation=has_obj)

    # P_s: OR pesato delle prossimità dei singoli sensori (la stereo pesa 0.6, 1.0 se confermata da YOLO)
    p_s = proximity_term(readings, stereo_weight_override=(W_LIDAR if has_obj else None))

    # Q_s: presenza oggetti a pieno peso; distanza: LiDAR del settore se disponibile, altrimenti stereo
    q_s = 0.0
    obj_dicts: List[Dict] = []
    for o in objects:
        d_obj = o.dist_cm
        src = o.dist_source
        if min_rel is not None and (d_obj is None or d_obj >= 900.0 or src == "stereo"):
            # preferisci la distanza affidabile del settore; se la stereo dice più vicino, tieni il minimo
            d_obj = min_rel if d_obj is None else min(d_obj, min_rel)
            src = "lidar"
        p_obj = proximity(d_obj) if d_obj is not None else 0.0
        if d_obj is None:
            # oggetto presente ma senza distanza (stereo cieca e niente LiDAR): presenza conservativa
            p_obj = 0.45
            src = "presence_only"
        contrib = o.weight * min(1.0, max(0.0, o.confidence)) * p_obj
        # entità dinamiche (persone/animali): cautela extra anche lontane
        if o.dynamic:
            contrib = max(contrib, 0.35 * o.weight * o.confidence)
        q_s = max(q_s, contrib)
        obj_dicts.append({
            "class": o.label,
            "conf": round(float(o.confidence), 2),
            "dist_cm": round(float(d_obj), 1) if d_obj is not None else None,
            "dist_src": src,
            "bearing_deg": round(float(o.bearing_deg), 1),
            "w": round(o.weight, 2),
            "contrib": round(contrib, 3),
        })

    r_s = or_combine(p_s, q_s)

    inputs = {r.source: r.dist_cm for r in readings if r.dist_cm is not None and 0 < r.dist_cm < 900}
    return SectorRisk(
        sector=sector,
        risk=float(r_s),
        level=risk_to_level(r_s),
        proximity_risk=float(p_s),
        object_risk=float(q_s),
        fused_dist_cm=d_fused,
        min_reliable_cm=min_rel,
        inputs=inputs,
        objects=obj_dicts,
        stereo_only_near=stereo_only_near,
        note=note,
    )


def frontal_risk(sector_risks: Dict[str, SectorRisk]) -> Tuple[float, str]:
    """
    Rischio della marcia avanti. Il numero grande segue il muso (Fronte).

    Un angolo entra nel numero solo se è davvero vicino: sotto 40 cm pesa 0.7
    (il corpo lo lambisce), tra 40 e 70 cm pesa 0.44. Oltre, è la stanza e non
    tiene il robot in MODERATE.
    """
    f = sector_risks.get("Fronte")
    r = f.risk if f else 0.0
    for key in ("Front-SX", "Front-DX"):
        s = sector_risks.get(key)
        if s is None:
            continue
        d = s.min_reliable_cm
        if d is not None and d < DIST_CRITICAL_CM:
            r = max(r, 0.7 * s.risk)
        elif d is not None and d < DIST_HIGH_CM:
            r = max(r, 0.44 * s.risk)
    return float(r), risk_to_level(r)


def frontal_distance(sector_risks: Dict[str, SectorRisk]) -> float:
    """Distanza frontale effettiva (cm) da usare per le soglie della FSM (999 se nulla)."""
    f = sector_risks.get("Fronte")
    d = f.fused_dist_cm if (f and f.fused_dist_cm is not None) else 999.0
    # un ostacolo CRITICO nei settori diagonali vale anche per il fronte (ingombro del corpo)
    for k in ("Front-SX", "Front-DX"):
        s = sector_risks.get(k)
        if s and s.min_reliable_cm is not None and s.min_reliable_cm < DIST_CRITICAL_CM:
            d = min(d, s.min_reliable_cm)
    return float(d)


__all__ = [
    "SECTORS", "SECTOR_ANGLE_DEG", "SECTOR_EN_TO_IT", "FRONT_SECTORS",
    "W_ULTRASONIC", "W_LIDAR", "W_STEREO",
    "SensorDistance", "SectorObject", "SectorRisk",
    "proximity", "class_weight", "risk_to_level", "or_combine", "bearing_to_sector",
    "fuse_sector_distance", "compute_sector_risk", "frontal_risk", "frontal_distance",
    "DIST_CRITICAL_CM", "DIST_HIGH_CM", "DIST_MODERATE_CM",
]
