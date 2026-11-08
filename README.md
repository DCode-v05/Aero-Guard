# Aero-Guard

Anomaly detection for drone flight telemetry, with three Streamlit dashboards on top. It checks altitude, velocity, yaw, pitch, battery and GPS drift, and flags readings that look wrong, either on their own or in combination.

This is a personal project to practise an end-to-end anomaly-detection pipeline. The data is synthetic: `Data.py` generates 50 flights of 200 timesteps (10,000 rows) and injects anomalies on purpose, so the dashboards have labels to score against. There is no real flight data and no validated accuracy figure. Treat it as a demo of the technique, not a safety system.

## How it works

**Data.** For each flight `Data.py` builds altitude as a random walk around 10 m, velocity around 1 m/s, small-noise yaw and pitch, battery draining from 100% to 80%, and low-amplitude GPS drift. Five anomalies per flight (about 2.5% of rows) are injected as sharp offsets to altitude, velocity and GPS drift and labelled `anomaly = 1`. The output is `drone_telemetry.csv`.

**Statistical checks.** The basic dashboard flags a point if any channel leaves a fixed safe range (for example altitude 8 to 15 m). The advanced and ML dashboards learn each channel's mean and standard deviation from the data and flag values outside mean +/- 2.5 sigma. The advanced dashboard needs at least two channels out of range before counting a point.

**Isolation Forest.** An `IsolationForest` (100 trees, `contamination=0.1`) is fit on the MinMax-scaled six features. It needs no labels and catches readings that are only suspicious in combination. The advanced dashboard's Combined method calls a point anomalous if the forest flags it or at least two channels are out of range.

**LSTM autoencoder.** `Model Train.py` trains a Keras model on sliding windows of 20 timesteps: a 64-unit LSTM encoder, `RepeatVector`, a 64-unit LSTM decoder and a `TimeDistributed(Dense)` output, for 10 epochs with Adam and MSE. Windows whose reconstruction error is above mean + 2 std count as anomalies. The weights are saved to `drone_anomaly_lstm_model.h5`. The dashboards do not load this model. They use the Isolation Forest and the statistical checks.

## Dashboards

- `streamlit_dashboard.py` (basic): auto-refreshing view with fixed-range flags, metric cards, six time-series plots, a Normal/Warning/Critical indicator, anomaly breakdowns, a correlation heatmap and a 3D flight path.
- `advanced_dashboard.py`: a `DroneAnomalyDetector` class and five tabs (overview, anomaly analysis, parameter monitoring, model performance, data explorer). You can switch the method between Combined, Isolation Forest, Statistical and Ground Truth. The performance tab computes a confusion matrix, precision, recall, F1 and accuracy against the injected labels. The explorer exports to CSV.
- `drone_dashboard.py` (ML): single-page view that trains the Isolation Forest on load and compares ML and statistical counts, with an anomaly-score distribution and a recent-anomalies table.
- `dashboard_launcher.py`: landing page that compares the three and shows the run command for each.

Status is Normal, Warning or Critical depending on the anomaly rate in the selected window. The live views refresh every few seconds to imitate a stream.

## Stack

Python, Streamlit (`streamlit-autorefresh`, `streamlit-option-menu`), pandas, NumPy, scikit-learn, TensorFlow/Keras, Plotly, Matplotlib, joblib.

## Setup

```bash
git clone https://github.com/DCode-v05/Aero-Guard.git
cd Aero-Guard
pip install -r requirements.txt
streamlit run dashboard_launcher.py
```

You can also run any dashboard directly: `streamlit run streamlit_dashboard.py`, `advanced_dashboard.py` or `drone_dashboard.py`.

The repo already includes `drone_telemetry.csv` and a trained `drone_anomaly_lstm_model.h5`. To regenerate or retrain, run these from the repo root, since the paths are relative:

```bash
python Data.py
python "Model Train.py"
```

## Files

```
Data.py                       synthetic telemetry generator
Model Train.py                trains and saves the LSTM autoencoder
streamlit_dashboard.py        basic dashboard
advanced_dashboard.py         advanced dashboard
drone_dashboard.py            ML dashboard
dashboard_launcher.py         landing page
drone_telemetry.csv           generated dataset
drone_anomaly_lstm_model.h5   trained LSTM weights
requirements.txt
```

## Contact

[Portfolio](https://www.denistan.me) · [LinkedIn](https://www.linkedin.com/in/denistanb) · [GitHub](https://github.com/DCode-v05) · denistanb05@gmail.com
