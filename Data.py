import numpy as np
import pandas as pd

# Parameters
n_flights = 50
timesteps = 200

data = []

for flight in range(n_flights):
    normal_altitude = np.cumsum(np.random.normal(0, 0.1, timesteps)) + 10
    normal_velocity = np.random.normal(1, 0.05, timesteps)
    normal_yaw = np.random.normal(0, 1, timesteps)
    normal_pitch = np.random.normal(0, 1, timesteps)
    normal_battery = np.linspace(100, 80, timesteps)
    normal_gps = np.random.normal(0, 0.05, timesteps)

    # Inject anomalies randomly
    anomaly_idx = np.random.choice(timesteps, size=5, replace=False)
    normal_altitude[anomaly_idx] += np.random.normal(2, 0.5, size=5)
    normal_velocity[anomaly_idx] += np.random.normal(2, 0.5, size=5)
    normal_gps[anomaly_idx] += np.random.normal(1, 0.2, size=5)

    df = pd.DataFrame({
        'flight': flight,
        'timestep': np.arange(timesteps),
        'altitude': normal_altitude,
        'velocity': normal_velocity,
        'yaw': normal_yaw,
        'pitch': normal_pitch,
        'battery': normal_battery,
        'gps_drift': normal_gps,
    })

    # Label anomalies (for evaluation)
    df['anomaly'] = 0
    df.loc[anomaly_idx, 'anomaly'] = 1

    data.append(df)

dataset = pd.concat(data, ignore_index=True)
dataset.to_csv("drone_telemetry.csv", index=False)
print("Sample dataset created!")
