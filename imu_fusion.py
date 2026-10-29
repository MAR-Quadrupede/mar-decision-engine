#!/usr/bin/env python3
"""
M.A.R. — IMU Sensor Fusion (Kalman Quaternion Filter)
Consumes raw MPU-6050 samples from `mar/imu/raw` (published by mar-serial-bridge),
performs an automatic steady-state bias calibration, runs the quaternion Kalman
filter from `imu_filter.py` (porting of the legacy IMUKalmanFilter.py), and
exposes roll / pitch / yaw for the Safety FSM and the dashboard.
"""

from typing import Dict, List, Optional
import threading
import time

import numpy as np

from imu_filter import IMUKalmanFilterPython


class IMUFusion:
    """Streaming Kalman attitude estimator with self-calibration."""

    def __init__(
        self,
        calibration_samples: int = 100,
        calibration_timeout_s: float = 12.0,
        max_sample_gap_s: float = 0.5
    ):
        self.filter = IMUKalmanFilterPython()
        self.calibration_samples = max(0, int(calibration_samples))
        self.calibration_timeout_s = float(calibration_timeout_s)
        self.max_sample_gap_s = float(max_sample_gap_s)

        self.calibrated = self.calibration_samples == 0
        self._gyro_sum = np.zeros(3)
        self._acc_sum = np.zeros(3)
        self._valid_samples = 0
        self._samples_seen = 0
        self._calibration_deadline = time.time() + self.calibration_timeout_s

        self._last_sample_ts: Optional[float] = None
        self._last_telemetry: Optional[Dict] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ calibration
    def _accumulate_calibration(self, gyro: np.ndarray, acc: np.ndarray):
        """Collects still samples and applies the same validity gate as the legacy monolith."""
        g = self.filter.GRAVITY
        still = (
            abs(gyro[0]) < 0.1 and abs(gyro[1]) < 0.1 and abs(gyro[2]) < 0.1
            and abs(acc[0]) < 1.0 and abs(acc[1]) < 1.0 and abs(acc[2] - g) < 1.0
        )
        if still:
            self._gyro_sum += gyro
            self._acc_sum += acc
            self._valid_samples += 1

        finished = self._samples_seen >= self.calibration_samples
        timed_out = time.time() >= self._calibration_deadline

        if finished or timed_out:
            if self._valid_samples > 0:
                self.filter.gyro_offset = self._gyro_sum / self._valid_samples
                self.filter.accel_offset = self._acc_sum / self._valid_samples
                self.filter.accel_offset[2] -= self.filter.GRAVITY
                print(f"  ✓ [IMU] Calibrazione completata su {self._valid_samples} campioni validi "
                      f"(gyro_off={np.round(self.filter.gyro_offset, 4)}, "
                      f"acc_off={np.round(self.filter.accel_offset, 3)})")
            else:
                print("  ⚠️ [IMU] Nessun campione stazionario: calibrazione con offset a zero "
                      "(tieni il robot fermo all'avvio per una stima precisa).")
            self.calibrated = True

    # ----------------------------------------------------------------- stream
    def feed(self, payload: Dict) -> Optional[Dict]:
        """
        Processes one `mar/imu/raw` payload.
        Returns the updated attitude dict once the filter is calibrated, else None.
        """
        try:
            gyro_raw = payload.get("gyro")
            acc_raw = payload.get("acc")
            if not gyro_raw or not acc_raw or len(gyro_raw) != 3 or len(acc_raw) != 3:
                return None
            gyro_raw = np.asarray(gyro_raw, dtype=float)
            acc_raw = np.asarray(acc_raw, dtype=float)
            gyro_scale = float(payload.get("gyro_scale", self.filter.GYRO_SCALE))
            accel_scale = float(payload.get("accel_scale", self.filter.ACCEL_SCALE))
            sample_ts = float(payload.get("timestamp", time.time()))
        except (TypeError, ValueError):
            return None

        with self._lock:
            self.filter.GYRO_SCALE = gyro_scale
            self.filter.ACCEL_SCALE = accel_scale

            if not self.calibrated:
                gyro = (gyro_raw / gyro_scale) * np.pi / 180.0
                acc = (acc_raw / accel_scale) * self.filter.GRAVITY
                self._samples_seen += 1
                self._accumulate_calibration(gyro, acc)
                return None

            # Time delta from sample timestamps (robust to MQTT jitter / dropped frames)
            if self._last_sample_ts is None:
                dt = 0.01
            else:
                dt = sample_ts - self._last_sample_ts
                if dt <= 0.0 or dt > self.max_sample_gap_s:
                    dt = 0.01
            self._last_sample_ts = sample_ts

            self.filter.update(gyro_raw, acc_raw, dt)

            self._last_telemetry = {
                "timestamp": time.time(),
                "sample_timestamp": sample_ts,
                "roll": round(float(self.filter.get_roll()), 2),
                "pitch": round(float(self.filter.get_pitch()), 2),
                "yaw": round(float(self.filter.get_yaw()), 2),
                "accel_x": round(float(self.filter.get_accel_x()), 3),
                "accel_y": round(float(self.filter.get_accel_y()), 3),
                "accel_z": round(float(self.filter.get_accel_z()), 3),
                "calibrated": True,
            }
            return self._last_telemetry

    def latest(self, max_age_s: float = 1.0) -> Optional[Dict]:
        """Returns the most recent attitude if fresh enough for the FSM."""
        with self._lock:
            if self._last_telemetry is None:
                return None
            age = time.time() - self._last_telemetry["timestamp"]
            if age > max_age_s:
                return None
            return dict(self._last_telemetry)

    @property
    def offset_debug(self) -> Dict[str, List[float]]:
        with self._lock:
            return {
                "gyro_offset": [round(float(v), 5) for v in self.filter.gyro_offset],
                "accel_offset": [round(float(v), 4) for v in self.filter.accel_offset],
            }
