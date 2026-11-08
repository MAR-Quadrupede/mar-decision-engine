#!/usr/bin/env python3
"""
M.A.R. — mar-decision-engine
Fusione multi-sensore (LiDAR 360° + ultrasuoni + stereo 3D + YOLO + IMU + SLAM),
equazione di rischio per settore, FSM di sicurezza e navigazione autonoma.

Topic MQTT in ingresso:  mar/lidar/*, mar/ultrasonic/range, mar/hardware/telemetry,
                         mar/vision/*, mar/imu/raw|telemetry, mar/slam/pose, mar/safety/*
Topic MQTT in uscita:    mar/control/motion   (comandi per mar-serial-bridge)
                         mar/fsm/state        (stato + mappa di rischio per la dashboard, 10 Hz)
                         mar/imu/telemetry    (assetto Kalman)
                         mar/decision/health  (watchdog sensori, 1 Hz)
"""

import json
import os
import signal
import sys
import threading
import time
from typing import Optional

from fsm import DecisionFSM, RobotState
from imu_fusion import IMUFusion
from world_model import WorldModel

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
IMU_CALIBRATION_SAMPLES = int(os.getenv("IMU_CALIBRATION_SAMPLES", "100"))
IMU_CALIBRATION_TIMEOUT_S = float(os.getenv("IMU_CALIBRATION_TIMEOUT_S", "12"))
FORWARD_SPEED = int(os.getenv("MAR_FORWARD_SPEED", "50"))
SLOW_SPEED = int(os.getenv("MAR_SLOW_SPEED", "25"))
AUTONOMOUS_AT_BOOT = os.getenv("MAR_AUTONOMOUS", "1").strip().lower() in ("1", "true", "yes", "on")
LOOP_HZ = float(os.getenv("MAR_DECISION_HZ", "20"))
STATE_PUBLISH_HZ = float(os.getenv("MAR_STATE_HZ", "10"))
CMD_REFRESH_S = float(os.getenv("MAR_CMD_REFRESH_S", "0.5"))   # ri-invio periodico del comando attivo
MANUAL_TIMEOUT_S = float(os.getenv("MAR_MANUAL_TIMEOUT_S", "3.0"))  # dopo un comando manuale l'autonomia tace

world = WorldModel()
fsm = DecisionFSM(forward_speed=FORWARD_SPEED, slow_speed=SLOW_SPEED, autonomous=AUTONOMOUS_AT_BOOT)
imu_fusion = IMUFusion(calibration_samples=IMU_CALIBRATION_SAMPLES, calibration_timeout_s=IMU_CALIBRATION_TIMEOUT_S)

decision_trigger = threading.Event()
state_lock = threading.Lock()
running = True
mqtt_client = None
last_manual_cmd_ts = 0.0


def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    ok = (rc == 0) or (str(rc) == "Success")
    if ok:
        print(f"  ✓ MQTT connesso ({MQTT_BROKER}:{MQTT_PORT}) — sottoscrizione topic di sistema")
        client.subscribe([
            ("mar/slam/pose", 0),
            ("mar/lidar/alert", 0),
            ("mar/lidar/telemetry", 0),
            ("mar/lidar/scan", 0),
            ("mar/vision/detections", 0),
            ("mar/vision/command", 0),
            ("mar/vision/motion", 0),
            ("mar/vision/ground_hazard", 0),
            ("mar/imu/raw", 0),
            ("mar/imu/telemetry", 0),
            ("mar/hardware/telemetry", 0),
            ("mar/ultrasonic/range", 0),
            ("mar/control/motion", 0),
            ("mar/control/mode", 1),
            ("mar/safety/estop", 2),
            ("mar/safety/resume", 1),
        ])
    else:
        print(f"  ✗ MQTT on_connect fallito rc={rc}")


def on_mqtt_message(client, userdata, msg):
    global last_manual_cmd_ts
    topic = msg.topic
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return

    now = time.time()
    if topic == "mar/imu/raw":
        imu_fusion.feed(data)
        return
    if topic == "mar/safety/estop":
        print(f"🚨 [E-STOP] {data}")
        with state_lock:
            fsm.trigger_operator_estop(f"E-Stop da {data.get('source', 'operatore')}")
        decision_trigger.set()
        return
    if topic == "mar/safety/resume":
        print(f"✅ [RESUME] {data}")
        with state_lock:
            fsm.clear_operator_estop(f"Resume da {data.get('source', 'operatore')}")
        decision_trigger.set()
        return
    if topic == "mar/control/mode":
        mode = str(data.get("mode", "")).upper()
        with state_lock:
            if mode in ("AUTO", "AUTONOMOUS"):
                fsm.set_autonomous(True, "Modalità AUTONOMA da operatore")
            elif mode in ("MANUAL", "IDLE", "TELEOP"):
                fsm.set_autonomous(False, f"Modalità {mode} da operatore")
        decision_trigger.set()
        return
    if topic == "mar/control/motion":
        # comandi manuali (dashboard / LoRa) → l'autonomia si sospende per MANUAL_TIMEOUT_S
        src = str(data.get("source", "")).upper()
        state = str(data.get("state", "")).upper()
        if src in ("WEBSOCKET", "LORA_RADIO") or state == "MANUAL":
            last_manual_cmd_ts = now
        return

    with state_lock:
        relevant = world.ingest(topic, data, now)
    if relevant:
        decision_trigger.set()


def signal_handler(sig, frame):
    global running
    print("\n[DECISION-ENGINE] Arresto...")
    running = False


def _publish(topic: str, payload: dict, qos: int = 0):
    if mqtt_client is not None and mqtt_client.is_connected():
        try:
            mqtt_client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=qos)
        except Exception as e:  # pragma: no cover
            print(f"[MQTT] publish {topic} fallita: {e}")


def _cmd_to_dict(cmd: str, reason: str) -> dict:
    payload_cmd = cmd.replace("CMD:", "", 1)
    parts = payload_cmd.split(",", 1)
    return {
        "cmd": parts[0],
        "param": parts[1] if len(parts) > 1 else "",
        "reason": reason,
        "state": fsm.state.name,
        "source": "DECISION_ENGINE",
        "timestamp": time.time(),
    }


def main():
    global running, mqtt_client
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 64)
    print(" 🐾 M.A.R. — DECISION ENGINE · SENSOR FUSION · SAFETY FSM")
    print("=" * 64)
    print(f"  • loop {LOOP_HZ:.0f} Hz | stato → dashboard {STATE_PUBLISH_HZ:.0f} Hz | autonomia={'ON' if AUTONOMOUS_AT_BOOT else 'OFF'}")
    print("  • equazione rischio: R_s = 1-(1-P_s)(1-Q_s) | pesi distanza US=1.0 LiDAR=1.0 stereo=0.6")

    if mqtt is not None:
        try:
            mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="mar-decision-engine")
        except Exception:
            mqtt_client = mqtt.Client(client_id="mar-decision-engine")
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        for attempt in range(1, 31):
            try:
                mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
                mqtt_client.loop_start()
                break
            except Exception as e:
                print(f"  ⚠️ MQTT tentativo {attempt}/30 fallito ({e}); retry 2s")
                time.sleep(2)
        else:
            print("  ⚠️ MQTT non raggiungibile: esecuzione standalone (nessun comando ai motori).")
    else:
        print("  ⚠️ paho-mqtt non installato: esecuzione standalone.")

    if AUTONOMOUS_AT_BOOT:
        fsm.set_state(RobotState.PATROL_FORWARD, "Avvio sistema autonomo")

    loop_dt = 1.0 / max(1.0, LOOP_HZ)
    state_dt = 1.0 / max(1.0, STATE_PUBLISH_HZ)
    last_cmd = ""
    last_cmd_ts = 0.0
    last_state_pub = 0.0
    last_health_pub = 0.0
    last_log_state: Optional[str] = None

    print("\n[DECISION-ENGINE] ciclo decisionale attivo\n")
    while running:
        now = time.time()

        fused_imu = imu_fusion.latest(max_age_s=1.0)
        with state_lock:
            if fused_imu is not None:
                world.set_imu(fused_imu, now)
            snap = world.snapshot(now)
            cmd, reason = fsm.update(snap)
            decision = dict(fsm.last_decision)

        manual_recent = (now - last_manual_cmd_ts) < MANUAL_TIMEOUT_S
        safety_cmd = cmd.startswith("CMD:STOP")
        # In teleoperazione recente l'autonomia non muove il robot, ma gli STOP di sicurezza passano sempre
        if cmd != last_cmd or (now - last_cmd_ts) >= CMD_REFRESH_S:
            if cmd != last_cmd:
                print(f"🤖 [DECISION] {cmd} | {reason}")
            if not manual_recent or safety_cmd:
                _publish("mar/control/motion", _cmd_to_dict(cmd, reason))
            last_cmd = cmd
            last_cmd_ts = now

        if now - last_state_pub >= state_dt:
            last_state_pub = now
            world_payload = snap.to_payload()
            state_payload = {
                "state": fsm.state.name,
                "autonomous": fsm.autonomous,
                "manual_override": manual_recent,
                "active_command": cmd,
                "reason": reason,
                "risk_level": fsm.latest_risk_level,
                "risk_value": round(fsm.latest_risk_value, 3),
                "timestamp": now,
                # compatibilità dashboard (campi piatti)
                "ultrasonic_cm": round(snap.us_cm, 1) if snap.us_cm else None,
                "lidar_front_m": round(snap.lidar_front_cm / 100.0, 3) if snap.lidar_front_cm else None,
                "stereo_front_cm": round(snap.stereo_front_cm, 1) if snap.stereo_front_cm else None,
                "front_dist_cm": round(snap.front_dist_cm, 1) if snap.front_dist_cm < 900 else None,
                "point_count": snap.point_count,
                "slam_pose": snap.pose,
                "imu": {"roll": snap.imu.get("roll"), "pitch": snap.imu.get("pitch"), "yaw": snap.imu.get("yaw")} if snap.imu else None,
                # nuova fusione
                "fusion": world_payload,
                "navigation": {
                    "choice": decision.get("nav_choice"),
                    "scores": decision.get("nav_scores", {}),
                    "front_cm": decision.get("front_cm"),
                    "front_risk": decision.get("front_risk"),
                },
            }
            _publish("mar/fsm/state", state_payload)
            if fused_imu is not None:
                _publish("mar/imu/telemetry", fused_imu)

        if now - last_health_pub >= 1.0:
            last_health_pub = now
            _publish("mar/decision/health", {
                "timestamp": now,
                "sensors": snap.sensors_online,
                "state": fsm.state.name,
                "autonomous": fsm.autonomous,
                "imu_calibrated": bool(getattr(imu_fusion, "calibrated", False)),
                "us_simulated_seen": world.us_simulated,
            })
            if last_log_state != fsm.state.name:
                last_log_state = fsm.state.name
            missing = [k for k, v in snap.sensors_online.items() if not v]
            if missing and int(now) % 10 == 0:
                print(f"  ℹ️ sensori non freschi: {', '.join(missing)}")

        decision_trigger.wait(timeout=loop_dt)
        decision_trigger.clear()

    # uscita pulita: ferma i motori
    _publish("mar/control/motion", _cmd_to_dict("CMD:STOP", "shutdown decision engine"))
    if mqtt_client is not None:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        except Exception:
            pass
    sys.exit(0)


if __name__ == "__main__":
    main()
