import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("drone_telemetry.csv")
features = ['altitude','velocity','yaw','pitch','battery','gps_drift']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features])

# Prepare sequences for LSTM
def create_sequences(data, seq_length=20):
    sequences = []
    for i in range(len(data)-seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 20
X = create_sequences(data_scaled, seq_length)

# LSTM Autoencoder
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, X.shape[2]), return_sequences=False),
    RepeatVector(seq_length),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train
model.fit(X, X, epochs=10, batch_size=32, validation_split=0.1)

model.save("drone_anomaly_lstm_model.h5")
print("Model saved successfully!")

# Predict
X_pred = model.predict(X)
mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))

# Threshold (example: mean + 2*std)
threshold = mse.mean() + 2*mse.std()
anomaly_labels = (mse > threshold).astype(int)

print("Threshold:", threshold)
print("Detected anomalies:", anomaly_labels.sum())
