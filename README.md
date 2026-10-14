# 🧠 Microservizio: `mar-decision-engine`

**Core Brain, Finite State Machine (FSM) & Active Perception**  
*Parte dell'architettura a microservizi M.A.R. (Multi-legged Autonomous Robot)*

---

## 🎯 Ruolo nell'Architettura a Microservizi

`mar-decision-engine` è il cervello decisionale autonomo del robot. Non accede direttamente all'hardware o alla fotocamera, ma agisce da **nodo di orchestrazione e fusione sensoriale di alto livello**:
1. **Fusione Sensoriale Asincrona**:
   - Ascolta i trigger perimetrali da `mar-lidar-service` (`mar/lidar/alert`).
   - Ascolta le classi semantiche e le distanze metriche 3D da `mar-vision-service` (`mar/vision/detections`).
   - Monitora la telemetria degli ultrasuoni e della batteria da `mar-serial-bridge` (`mar/hardware/telemetry`).
2. **Macchina a Stati Finita (Safety FSM)**:
   - `IDLE`: quadrupede in attesa con assetto eretto.
   - `PATROL_FORWARD`: marcia autonoma in avanti con scansione di sicurezza attiva.
   - `INSPECTING_OBJECT`: allerta ricevuta dal LiDAR; scatto della saccade per inquadrare l'oggetto e attendere la classificazione YOLO.
   - `AVOIDANCE`: calcolo del percorso alternativo e manovra di svolta/schivamento.
   - `EMERGENCY_STOP`: arresto prioritario istantaneo per oggetti troppo vicini ($< 35\text{ cm}$) o comando operatore.
3. **Emissione Comandi di Movimento**:
   - Pubblica le direttive cinematiche su `mar/control/motion` destinate al microcontrollore.

---

## 📡 Interfacce e Topic MQTT

| Topic | Tipo | Formato Payload | Descrizione |
| :--- | :--- | :--- | :--- |
| `mar/lidar/alert` | In ingresso (Sub) | `{"theta": 35.0, "sector": "FRONT_RIGHT", "dist_m": 0.85}` | Trigger LiDAR per attivare la saccade |
| `mar/vision/detections` | In ingresso (Sub) | `{"count": 1, "objects": [{"class": "box", "distance_cm": 82}]}` | Ostacoli 3D riconosciuti da YOLO |
| `mar/hardware/telemetry` | In ingresso (Sub) | `{"ultrasonic": "28.5", "batt": "11.8"}` | Telemetria fisica da Arduino |
| `mar/safety/estop` | In ingresso (Sub) | `{"reason": "OPERATOR"}` | Segnale di stop prioritario |
| `mar/control/motion` | In uscita (Pub) | `{"cmd": "FORWARD", "param": "50", "reason": "PATROL"}` | Comando per `mar-serial-bridge` |
| `mar/fsm/state` | In uscita (Pub) | `{"state": "PATROL_FORWARD", "active_command": "CMD:FORWARD,50"}` | Stato per la `mar-dashboard` |

---

## 📂 Struttura della Cartella

```text
mar-decision-engine/
├── README.md                  # Questo documento
├── requirements.txt           # paho-mqtt, numpy
├── fsm.py                     # Macchina a stati finita con regole deterministiche
├── attention_controller.py    # Logica per puntamento attivo (Saccade) e motion-blur gating
├── obstacle_avoidance.py      # Algoritmi di punteggio rischio e clearance ostacoli
└── main.py                    # Demone del microservizio
```

---

## 🚀 Avvio del Microservizio

### 1. Installazione Dipendenze
```bash
pip install -r requirements.txt
```

### 2. Esecuzione
```bash
# Avvio standard con broker locale
python main.py

# Con broker remoto
MQTT_BROKER=192.168.1.50 python main.py
```
