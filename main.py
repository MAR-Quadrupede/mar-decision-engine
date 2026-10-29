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

fsm = DecisionFSM(forward_speed=50, critical_distance_cm=35.0)
attention = AttentionController(settle_time_sec=0.4)
imu_fusion = IMUFusion(
    calibration_samples=IMU_CALIBRATION_SAMPLES,
    calibration_timeout_s=IMU_CALIBRATION_TIMEOUT_S
)

latest_lidar_alert = None
latest_vision_objects = None
latest_ground_hazard = None
latest_imu_data = None
latest_ultrasonic_cm = -1.0
running = True
mqtt_client = None


def on_mqtt_message(client, userdata, msg):
    global latest_lidar_alert, latest_vision_objects, latest_ground_hazard, latest_imu_data, latest_ultrasonic_cm
    topic = msg.topic
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return

    if topic in ("mar/lidar/alert", "mar/lidar/telemetry"):
        latest_lidar_alert = data
    elif topic == "mar/lidar/scan":
        if latest_lidar_alert is None:
            latest_lidar_alert = data
        else:
            if "sector_mins" in data:
                latest_lidar_alert["sector_mins"] = data["sector_mins"]
            if "front_distance_m" in data:
                latest_lidar_alert["front_distance_m"] = data["front_distance_m"]
    elif topic == "mar/vision/detections":
        latest_vision_objects = data
    elif topic == "mar/vision/ground_hazard":
        latest_ground_hazard = data
    elif topic == "mar/imu/raw":
        # Raw IMU samples from mar-serial-bridge -> Kalman quaternion fusion
        imu_fusion.feed(data)
    elif topic == "mar/imu/telemetry":
        latest_imu_data = data
    elif topic == "mar/hardware/telemetry":
        # Extract ultrasonic if present
        if "ultrasonic" in data:
            try:
                latest_ultrasonic_cm = float(data["ultrasonic"])
            except Exception:
                pass
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
        fsm.set_state(RobotState.EMERGENCY_STOP, "E-Stop da operatore o sensore critico")


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
        try:
            mqtt_client = mqtt.Client(client_id="mar-decision-engine")
            mqtt_client.on_message = on_mqtt_message
            mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            mqtt_client.subscribe([
                ("mar/lidar/alert", 0),
                ("mar/lidar/telemetry", 0),
                ("mar/lidar/scan", 0),
                ("mar/vision/detections", 0),
                ("mar/vision/ground_hazard", 0),
                ("mar/imu/raw", 0),
                ("mar/imu/telemetry", 0),
                ("mar/hardware/telemetry", 0),
                ("mar/safety/estop", 2)
            ])
            mqtt_client.loop_start()
            print(f"  ✓ Connesso a MQTT Broker ({MQTT_BROKER}:{MQTT_PORT})")
        except Exception as e:
            print(f"  ⚠️ MQTT non raggiungibile ({e}). Esecuzione standalone.")

    print("\n[DECISION-ENGINE] Ciclo decisionale attivo a 20 Hz...\n")

    last_cmd = ""

    while running:
        # Attitude from the quaternion Kalman filter (fresh samples from mar-serial-bridge)
        fused_imu = imu_fusion.latest(max_age_s=1.0)
        if fused_imu is not None:
            latest_imu_data = fused_imu

        # Esegui tick FSM
        cmd, reason = fsm.update(
            lidar_alert=latest_lidar_alert,
            vision_objects=latest_vision_objects,
            ultrasonic_distance_cm=latest_ultrasonic_cm,
            ground_hazard_alert=latest_ground_hazard,
            imu_data=latest_imu_data
        )

        # Se il comando cambia o a intervalli periodici, invia a mar-serial-bridge
        if cmd != last_cmd:
            last_cmd = cmd
            print(f"🤖 [DECISION]: {cmd} | {reason}")
            if mqtt_client and mqtt_client.is_connected():
                # Estrai tipo comando e parametro (es. "CMD:FORWARD,50" -> cmd="FORWARD", param="50")
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

        # Pubblica stato FSM per la Dashboard (5 Hz)
        if mqtt_client and mqtt_client.is_connected():
            state_payload = {
                "state": fsm.state.name,
                "active_command": cmd,
                "reason": reason,
                "timestamp": time.time()
            }
            mqtt_client.publish("mar/fsm/state", json.dumps(state_payload))

            # Assetto filtrato (Kalman) per dashboard, LoRa e telemetria
            if fused_imu is not None:
                mqtt_client.publish("mar/imu/telemetry", json.dumps(fused_imu))

        time.sleep(0.05)  # 20 Hz update loop


if __name__ == "__main__":
    main()
