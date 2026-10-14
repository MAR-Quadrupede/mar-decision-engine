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

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

fsm = DecisionFSM(forward_speed=50, critical_distance_m=0.35)
attention = AttentionController(settle_time_sec=0.4)

latest_lidar_alert = None
latest_vision_objects = None
latest_ultrasonic_cm = -1.0
running = True
mqtt_client = None


def on_mqtt_message(client, userdata, msg):
    global latest_lidar_alert, latest_vision_objects, latest_ultrasonic_cm
    topic = msg.topic
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return

    if topic == "mar/lidar/alert":
        latest_lidar_alert = data
    elif topic == "mar/vision/detections":
        latest_vision_objects = data
    elif topic == "mar/hardware/telemetry":
        # Extract ultrasonic if present
        if "ultrasonic" in data:
            try:
                latest_ultrasonic_cm = float(data["ultrasonic"])
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
    global running, mqtt_client
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
                ("mar/vision/detections", 0),
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
        # Esegui tick FSM
        cmd, reason = fsm.update(
            lidar_alert=latest_lidar_alert,
            vision_objects=latest_vision_objects,
            ultrasonic_distance_cm=latest_ultrasonic_cm
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

        time.sleep(0.05)  # 20 Hz update loop


if __name__ == "__main__":
    main()
