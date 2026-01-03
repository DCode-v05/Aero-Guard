# Aero Guard (Drone Flight Anomaly Detection System)

## Project Description
**Aero Guard** is a cutting-edge anomaly detection system designed to ensure the safety and reliability of drone operations. By monitoring flight telemetry data in real-time, it identifies potential hardware malfunctions, environmental interferences, or erratic flight patterns. The system combines rule-based safety checks with advanced Machine Learning algorithms to provide a robust layer of protection for autonomous aerial vehicles.

---

## Project Details

### Problem Statement
With the rapid adoption of drones in logistics, agriculture, and surveillance, the risk of mid-flight failures has increased. Traditional monitoring often relies on manual observation or simple threshold checks, which may miss subtle signs of degradation (e.g., gradual battery drain inconsistent with usage, or minor GPS drift) that precede a crash. Aero Guard addresses this by providing an automated, intelligent monitoring solution that detects deviations from normal behavior before they escalate into critical failures.

### Key Features
- **Real-time Telemetry Monitoring:** continuous tracking of essential parameters including Altitude, Velocity, Battery, Yaw, Pitch, and GPS Drift.
- **Dual-Layer Anomaly Detection:**
  - **Statistical/Rule-Based:** Instant alerts for values exceeding defined safety thresholds.
  - **Machine Learning (Isolation Forest):** Unsupervised learning to detect complex, multi-dimensional outliers that simple rules might miss.
- **Deep Learning Capability:** Includes an LSTM Autoencoder implementation for analyzing temporal sequences and detecting long-term pattern deviations.
- **3D Flight Visualization:** Interactive 3D trajectory plots to visualize the spatial context of anomalies.
- **Smart Alert System:** visual status indicators (Normal 🟢, Warning 🟡, Critical 🔴) based on the density and severity of detected anomalies.

### Data Generation & Processing
- **Synthetic Data Engine (`Data.py`):** Generates realistic drone flight logs (`drone_telemetry.csv`) with normal operating noise and randomly injected anomalies for training and testing.
- **Preprocessing:** 
  - Feature scaling using `MinMaxScaler`.
  - Sequence generation for LSTM model training.

### Model Architecture
The project employs two distinct ML approaches:
1. **Isolation Forest (Scikit-learn):** Used in the advanced dashboard for efficient, real-time outlier detection across tabular data.
2. **LSTM Autoencoder (TensorFlow/Keras):** A deep learning model that learns the "normal" temporal sequences of flight data. Anomalies are detected when the reconstruction error (MSE) of a new sequence exceeds a dynamic threshold.

### Visualizations
- **Real-time Line Charts:** Auto-refreshing plots for all telemetry sensors.
- **3D Trajectory Plots:** Altitude vs. GPS Drift vs. Time.
- **Correlation Heatmaps:** To understand relationships between different flight parameters (e.g., Velocity vs. Battery drain).
- **Distribution Histograms:** Monitoring parameter shifts and outliers.

---

## Tech Stack
- **Language:** Python 3.x
- **Web Framework:** Streamlit
- **Data Analysis:** pandas, numpy
- **Machine Learning:** scikit-learn, TensorFlow/Keras
- **Visualization:** Plotly (Express & Graph Objects), Matplotlib, Seaborn
- **Utilities:** joblib

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/DCode-v05/Aero-Guard.git
cd Aero-Guard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data (Optional)
If you need fresh test data, run the data generation script:
```bash
python Data.py
```

### 4. Run the Application
**For the Standard Dashboard (Real-time Monitoring):**
```bash
streamlit run streamlit_dashboard.py
```

**For the Advanced Dashboard (ML-based Detection):**
```bash
streamlit run advanced_dashboard.py
```

---

## Project Structure
```
Aero Guard/
│
├── streamlit_dashboard.py    # Main real-time monitoring dashboard
├── advanced_dashboard.py     # Advanced analytics & ML detection dashboard
├── Data.py                   # Synthetic data generation script
├── Model Train.py            # LSTM Autoencoder training script
├── drone_telemetry.csv       # Dataset with flight logs
├── drone_anomaly_lstm_model.h5 # Pre-trained LSTM model
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com