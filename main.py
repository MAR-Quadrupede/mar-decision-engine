#!/usr/bin/env python3
"""
M.A.R. — mar-decision-engine Microservice Daemon
Consolidates multi-sensor perception (LiDAR 360° + Stereo 3D Vision),
runs the Finite State Machine (FSM), and emits prioritized motion commands.
"""

import os
import sys
import time
import json
import signal

from fsm import DecisionFSM, RobotState
from attention_controller import AttentionController
from imu_fusion import IMUFusion

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
IMU_CALIBRATION_SAMPLES = int(os.getenv("IMU_CALIBRATION_SAMPLES", "100"))
IMU_CALIBRATION_TIMEOUT_S = float(os.getenv("IMU_CALIBRATION_TIMEOUT_S", "12"))

fsm = DecisionFSM(
    forward_speed=50,
    critical_distance_cm=40.0,
    high_distance_cm=70.0,
    moderate_distance_cm=120.0,
    slow_warning_cm=70.0,
)
attention = AttentionController(settle_time_sec=0.4)
imu_fusion = IMUFusion(
    calibration_samples=IMU_CALIBRATION_SAMPLES,
    calibration_timeout_s=IMU_CALIBRATION_TIMEOUT_S
)

latest_lidar_alert = None
latest_vision_objects = None
latest_ground_hazard = None
latest_imu_data = None
latest_motion_data = None
latest_slam_pose = None
latest_ultrasonic_cm = -1.0
latest_ultrasonic_ts = 0.0
decision_trigger = threading.Event()
running = True
mqtt_client = None


def _ingest_ultrasonic(data: dict):
    global latest_ultrasonic_cm, latest_ultrasonic_ts
    for key in ("us_cm", "ultrasonic_cm", "ultrasonic", "us"):
        if key in data and data[key] is not None:
            try:
                val = float(data[key])
                if 2.0 <= val <= 400.0:
                    latest_ultrasonic_cm = val
                    latest_ultrasonic_ts = time.time()
                    decision_trigger.set()
                    return
            except Exception:
                pass


def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"  ✓ MQTT on_connect: Sottoscrizione ai topic di sistema (rc={rc})")
        client.subscribe([
            ("mar/slam/pose", 0),
            ("mar/slam/map", 0),
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
            ("mar/safety/estop", 2),
            ("mar/safety/resume", 1),
        ])
    else:
        print(f"  ✗ MQTT on_connect fallito con codice rc={rc}")


def on_mqtt_message(client, userdata, msg):
    global latest_lidar_alert, latest_vision_objects, latest_ground_hazard, latest_imu_data, latest_ultrasonic_cm, latest_motion_data, latest_slam_pose
    topic = msg.topic
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return

    if topic == "mar/slam/pose":
        latest_slam_pose = data
    elif topic in ("mar/lidar/alert", "mar/lidar/telemetry"):
        latest_lidar_alert = data
        decision_trigger.set()
    elif topic == "mar/lidar/scan":
        if latest_lidar_alert is None:
            latest_lidar_alert = data
        else:
            if "sector_mins" in data:
                latest_lidar_alert["sector_mins"] = data["sector_mins"]
            if "front_distance_m" in data:
                latest_lidar_alert["front_distance_m"] = data["front_distance_m"]
            if "point_count" in data:
                latest_lidar_alert["point_count"] = data["point_count"]
            latest_lidar_alert["timestamp"] = data.get("timestamp", time.time())
        decision_trigger.set()
    elif topic == "mar/vision/motion":
        latest_motion_data = data
        decision_trigger.set()
    elif topic == "mar/vision/detections":
        latest_vision_objects = data
        if "motion" in data:
            latest_motion_data = data["motion"]
        decision_trigger.set()
    elif topic == "mar/vision/command":
        # Merge command risk/distance into detections for FSM fusion
        if latest_vision_objects is None:
            latest_vision_objects = {}
        latest_vision_objects = {
            **latest_vision_objects,
            "target_dist_cm": data.get("target_dist_cm", latest_vision_objects.get("target_dist_cm")),
            "obstacle_near_cm": data.get("obstacle_near_cm", latest_vision_objects.get("obstacle_near_cm")),
            "risk_level": data.get("risk_level", latest_vision_objects.get("risk_level")),
            "timestamp": data.get("timestamp", time.time()),
        }
        decision_trigger.set()
    elif topic == "mar/vision/ground_hazard":
        latest_ground_hazard = data
        decision_trigger.set()
    elif topic == "mar/imu/raw":
        # Raw IMU samples from mar-serial-bridge -> Kalman quaternion fusion
        imu_fusion.feed(data)
    elif topic == "mar/imu/telemetry":
        latest_imu_data = data
    elif topic == "mar/ultrasonic/range":
        _ingest_ultrasonic(data)
    elif topic == "mar/hardware/telemetry":
        _ingest_ultrasonic(data)
        # Extract IMU roll/pitch from Arduino telemetry if available
        if "roll" in data and "pitch" in data:
            try:
                latest_imu_data = {
                    "roll": float(data["roll"]),
                    "pitch": float(data["pitch"]),
                    "timestamp": time.time()
                }
            except Exception:
                pass
    elif topic == "mar/safety/estop":
        print(f"🚨 [FSM E-STOP TRIGGERED]: {data}")
        fsm.trigger_operator_estop("E-Stop da operatore o sensore critico")
        decision_trigger.set()
    elif topic == "mar/safety/resume":
        print(f"✅ [FSM RESUME]: {data}")
        fsm.clear_operator_estop("Resume da operatore")
        decision_trigger.set()


def signal_handler(sig, frame):
    global running
    print("\n[DECISION-ENGINE] Arresto del motore decisionale...")
    running = False
    sys.exit(0)


def main():
    global running, mqtt_client, latest_imu_data
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print(" 🐾 M.A.R. — CORE DECISION ENGINE & SAFETY FSM")
    print("=" * 60)

    # Inizializza stato: avvia in PATROL_FORWARD dopo 1 secondo di setup
    fsm.set_state(RobotState.PATROL_FORWARD, "Avvio del sistema autonomo")

    if mqtt:
        mqtt_client = mqtt.Client(client_id="mar-decision-engine")
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        connected = False
        for attempt in range(1, 16):
            try:
                mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
                mqtt_client.loop_start()
                print(f"  ✓ Avviato loop MQTT verso ({MQTT_BROKER}:{MQTT_PORT})")
                connected = True
                break
            except Exception as e:
                print(f"  ⚠️ MQTT tentativo {attempt}/15 fallito ({e}). Retry in 2s...")
                time.sleep(2)
        if not connected:
            print("  ⚠️ MQTT non raggiungibile. Esecuzione standalone.")

    print("\n[DECISION-ENGINE] Ciclo decisionale attivo a 20 Hz...\n")

    last_cmd = ""

    while running:
        # Attitude from the quaternion Kalman filter (fresh samples from mar-serial-bridge)
        fused_imu = imu_fusion.latest(max_age_s=1.0)
        if fused_imu is not None:
            latest_imu_data = fused_imu

        # Ultrasuoni solo se freschi (< 1.5s)
        us_cm = latest_ultrasonic_cm
        if us_cm > 0 and (time.time() - latest_ultrasonic_ts) > 1.5:
            us_cm = -1.0

        # Esegui tick FSM
        cmd, reason = fsm.update(
            lidar_alert=latest_lidar_alert,
            vision_objects=latest_vision_objects,
            ultrasonic_distance_cm=us_cm,
            ground_hazard_alert=latest_ground_hazard,
            imu_data=latest_imu_data,
            motion_data=latest_motion_data
        )

        # Se il comando cambia o a intervalli periodici, invia a mar-serial-bridge
        if cmd != last_cmd:
            last_cmd = cmd
            print(f"🤖 [DECISION]: {cmd} | {reason}")
            if mqtt_client and mqtt_client.is_connected():
                payload_cmd = cmd.replace("CMD:", "")
                parts = payload_cmd.split(",", 1)
                cmd_dict = {
                    "cmd": parts[0],
                    "param": parts[1] if len(parts) > 1 else "",
                    "reason": reason,
                    "state": fsm.state.name,
                    "timestamp": time.time()
                }
                mqtt_client.publish("mar/control/motion", json.dumps(cmd_dict))

        # Pubblica stato FSM unificato per la Dashboard (tutti i sensori passati assieme)
        if mqtt_client and mqtt_client.is_connected():
            fusion_meta = getattr(fsm, "last_fusion_meta", {})
            lidar_front_m = latest_lidar_alert.get("front_distance_m") if latest_lidar_alert else None
            point_cnt = latest_lidar_alert.get("point_count") if latest_lidar_alert else 0

            state_payload = {
                "state": fsm.state.name,
                "active_command": cmd,
                "reason": reason,
                "risk_level": getattr(fsm, "latest_risk_level", getattr(fsm.front_risk, "held_level", "LOW")),
                "timestamp": time.time(),
                "fusion": fusion_meta,
                "ultrasonic_cm": round(float(us_cm), 1) if us_cm > 0 else None,
                "lidar_front_m": lidar_front_m,
                "point_count": point_cnt,
                "slam_pose": latest_slam_pose,
            }
            mqtt_client.publish("mar/fsm/state", json.dumps(state_payload))

            # Assetto filtrato (Kalman) per dashboard, LoRa e telemetria
            if fused_imu is not None:
                mqtt_client.publish("mar/imu/telemetry", json.dumps(fused_imu))

        # Attesa reattiva: si sveglia istantaneamente se arriva un nuovo pacchetto sensore (< 1ms)
        decision_trigger.wait(timeout=0.04)
        decision_trigger.clear()


if __name__ == "__main__":
    main()
