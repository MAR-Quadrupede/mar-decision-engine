import time
import numpy as np


class IMUKalmanFilterPython:

  def __init__(self):
    self.BUFFER_SIZE = 10
    self.ACCEL_SCALE = 16384.0
    self.GYRO_SCALE = 131.0
    self.GRAVITY = 9.81

    # Stato del quaternione [q0, q1, q2, q3]
    self.state = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # Angoli di Eulero in radianti
    self.roll = 0.0
    self.pitch = 0.0
    self.yaw = 0.0

    # Vettori fisici
    self.accel = np.zeros(3)
    self.velocity = np.zeros(3)
    self.position = np.zeros(3)

    # Offset
    self.gyro_offset = np.zeros(3)
    self.accel_offset = np.zeros(3)

    # Buffer per il filtro passa-basso
    self.gyro_buffer = np.zeros((3, self.BUFFER_SIZE))
    self.acc_buffer = np.zeros((3, self.BUFFER_SIZE))
    self.buffer_index = 0

    # Matrici di Kalman
    self.P = np.eye(4, dtype=float)
    self.Q = np.eye(4, dtype=float) * 0.0001
    self.R = np.zeros((6, 6), dtype=float)
    np.fill_diagonal(self.R, [0.01, 0.01, 0.01, 0.5, 0.5, 0.5])
    self.K = np.zeros((4, 6), dtype=float)

  def calibrate(self, read_func, samples=200):
    """Esegue la calibrazione leggendo N campioni dalla funzione di lettura

    passata come argomento (read_func), azzerando i bias.
    """
    print(f"Avvio calibrazione IMU ({samples} campioni)... tieni il sensore fermo!")
    gyro_sum = np.zeros(3)
    acc_sum = np.zeros(3)
    valid_samples = 0

    for _ in range(samples):
      gyro_raw, acc_raw = read_func()
      gyro, acc = self.process_imu_data_raw(gyro_raw, acc_raw)

      if (
          abs(gyro[0]) < 0.1
          and abs(gyro[1]) < 0.1
          and abs(gyro[2]) < 0.1
          and abs(acc[0]) < 1.0
          and abs(acc[1]) < 1.0
          and abs(acc[2] - self.GRAVITY) < 1.0
      ):
        gyro_sum += gyro
        acc_sum += acc
        valid_samples += 1

      time.sleep(0.005)

    if valid_samples > 0:
      self.gyro_offset = gyro_sum / valid_samples
      self.accel_offset = acc_sum / valid_samples
      self.accel_offset[2] -= self.GRAVITY  # Correzione per la gravità
      print(f"Calibrazione completata! Campioni validi: {valid_samples}")
      print(f"Gyro Offsets: {self.gyro_offset}")
      print(f"Accel Offsets: {self.accel_offset}")
    else:
      print(
          "Attenzione: Nessun campione valido durante la calibrazione! Sensore"
          " in movimento."
      )

  def process_imu_data_raw(self, gyro_raw, acc_raw):
    gyro = (np.array(gyro_raw, dtype=float) / self.GYRO_SCALE) * np.pi / 180.0
    acc = (np.array(acc_raw, dtype=float) / self.ACCEL_SCALE) * self.GRAVITY
    return gyro, acc

  def update(self, gyro_raw, acc_raw, dt):
    # Processa e scala i dati grezzi applicando gli offset
    gyro = (
        np.array(gyro_raw, dtype=float) / self.GYRO_SCALE
    ) * np.pi / 180.0 - self.gyro_offset
    acc = (
        np.array(acc_raw, dtype=float) / self.ACCEL_SCALE
    ) * self.GRAVITY - self.accel_offset

    # Filtro passa-basso
    gyro, acc = self.apply_low_pass_filter(gyro, acc)

    # Kalman Predict & Update
    self.predict(gyro, dt)
    self.update_with_measurements(gyro, acc)
    self.normalize_quaternion()
    self.quaternion_to_euler()

    # Integrazione accelerazione -> velocità -> posizione
    for i in range(3):
      self.accel[i] = acc[i]
      self.velocity[i] += self.accel[i] * dt
      self.position[i] += self.velocity[i] * dt

      if np.isnan(self.velocity[i]) or abs(self.velocity[i]) > 1000.0:
        self.velocity[i] = 0.0
      if np.isnan(self.position[i]) or abs(self.position[i]) > 10000.0:
        self.position[i] = 0.0

  def apply_low_pass_filter(self, gyro, acc):
    for i in range(3):
      self.gyro_buffer[i, self.buffer_index] = gyro[i]
      self.acc_buffer[i, self.buffer_index] = acc[i]
    self.buffer_index = (self.buffer_index + 1) % self.BUFFER_SIZE

    return np.mean(self.gyro_buffer, axis=1), np.mean(self.acc_buffer, axis=1)

  def predict(self, gyro, dt):
    q0, q1, q2, q3 = self.state
    dq = (
        0.5
        * np.array([
            -q1 * gyro[0] - q2 * gyro[1] - q3 * gyro[2],
            q0 * gyro[0] + q3 * gyro[1] - q2 * gyro[2],
            -q3 * gyro[0] + q0 * gyro[1] + q1 * gyro[2],
            q2 * gyro[0] - q1 * gyro[1] + q0 * gyro[2],
        ])
        * dt
    )

    self.state += dq
    self.state = np.where(np.isnan(self.state), [1.0, 0.0, 0.0, 0.0], self.state)

    gx, gy, gz = gyro * dt / 2.0
    F = np.array(
        [[1.0, -gx, -gy, -gz], [gx, 1.0, gz, -gy], [gy, -gz, 1.0, gx], [gz, gy, -gx, 1.0]]
    )
    self.P = F @ self.P @ F.T + self.Q

  def update_with_measurements(self, gyro, acc):
    q0, q1, q2, q3 = self.state
    g = self.GRAVITY

    z_pred = np.array([
        gyro[0],
        gyro[1],
        gyro[2],
        2 * (q1 * q3 - q0 * q2) * g,
        2 * (q2 * q3 + q0 * q1) * g,
        (q0**2 - q1**2 - q2**2 + q3**2) * g,
    ])

    z = np.array([gyro[0], gyro[1], gyro[2], acc[0], acc[1], acc[2]])
    y = z - z_pred

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 2 * q3 * g, -2 * q0 * g, 2 * q1 * g],
        [2 * q1 * g, 2 * q0 * g, 2 * q3 * g, 2 * q2 * g],
        [2 * q0 * g, -2 * q1 * g, -2 * q2 * g, 2 * q3 * g],
    ])

    S = H @ self.P @ H.T + self.R
    S_inv = np.linalg.inv(S + np.eye(6) * 0.001)
    self.K = self.P @ H.T @ S_inv

    self.state += self.K @ y
    self.state = np.where(np.isnan(self.state), [1.0, 0.0, 0.0, 0.0], self.state)

    self.P = (np.eye(4) - self.K @ H) @ self.P

  def normalize_quaternion(self):
    norm = np.linalg.norm(self.state)
    if norm < 0.0001 or np.isnan(norm):
      self.state = np.array([1.0, 0.0, 0.0, 0.0])
    else:
      self.state /= norm

  def quaternion_to_euler(self):
    q0, q1, q2, q3 = self.state
    self.roll = np.arctan2(
        2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1**2 + q2**2)
    )
    self.pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    self.yaw = np.arctan2(
        2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2**2 + q3**2)
    )

    if np.isnan(self.roll):
      self.roll = 0.0
    if np.isnan(self.pitch):
      self.pitch = 0.0
    if np.isnan(self.yaw):
      self.yaw = 0.0

  # --- GETTER ---
  def get_roll(self):
    return np.degrees(self.roll)

  def get_pitch(self):
    return np.degrees(self.pitch)

  def get_yaw(self):
    return np.degrees(self.yaw)

  def get_accel_x(self):
    return self.accel[0]

  def get_accel_y(self):
    return self.accel[1]

  def get_accel_z(self):
    return self.accel[2]

  def get_velocity_x(self):
    return self.velocity[0]

  def get_velocity_y(self):
    return self.velocity[1]

  def get_velocity_z(self):
    return self.velocity[2]

  def get_position_x(self):
    return self.position[0]

  def get_position_y(self):
    return self.position[1]

  def get_position_z(self):
    return self.position[2]