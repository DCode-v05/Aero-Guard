# Aero Guard

**Drone flight anomaly detection — watches telemetry for altitude, velocity, battery, orientation and GPS drift, and flags abnormal readings before they turn into a crash.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Overview

Aero Guard takes a stream of drone flight telemetry and decides, point by point, whether the drone is behaving normally or showing signs of trouble — a sudden altitude jump, a velocity spike, GPS drift, an odd battery curve. It pairs simple statistical threshold checks with an unsupervised machine-learning model (Isolation Forest) so it catches both obvious out-of-range values and the subtler multi-parameter outliers that a single threshold would miss. Everything is wrapped in interactive Streamlit dashboards that auto-refresh, plot each sensor over time, show a 3D flight path, and roll the findings up into a Normal / Warning / Critical status.

I built this as a personal ML project to practice an end-to-end anomaly-detection pipeline — data generation, preprocessing, two different modeling approaches, evaluation, and a usable front end. The telemetry is synthetic: a generator script produces 50 flights of 200 timesteps each (10,000 rows, 6 features) with anomalies injected on purpose, which gives the models something to find and gives the dashboards labeled ground truth to score against. There is no real-flight dataset and no validated accuracy figure — it is a working demo of the technique, not a tested safety system.

## Key Features

- **Six telemetry channels tracked:** altitude, velocity, yaw, pitch, battery, and GPS drift, each plotted over time and monitored independently.
- **Two-layer anomaly detection:**
  - *Rule / statistical layer* — flags values outside fixed safe ranges (basic dashboard) or outside mean ± 2.5 standard deviations learned from the data (advanced and ML dashboards).
  - *Machine-learning layer* — an Isolation Forest scores every reading as an outlier or not, catching combinations of values that look wrong together even when no single channel is out of range.
- **Combined verdict:** the advanced dashboard merges both layers — a point counts as anomalous if the Isolation Forest flags it OR at least two parameters are out of range.
- **LSTM autoencoder (separate training path):** a Keras sequence model that learns the "normal" temporal shape of flight data and flags windows whose reconstruction error exceeds a dynamic threshold. Trained and saved by its own script; see the note below on how it relates to the dashboards.
- **Three Streamlit dashboards** at different levels of depth, plus a launcher page that explains the difference and gives you the run command for each.
- **Status tiering:** every view rolls anomalies up into a Normal / Warning / Critical state based on the anomaly rate over the selected window.
- **3D flight-path view:** trajectory plotted as time vs GPS drift vs altitude, with anomaly points marked.
- **Model performance tab:** because the synthetic data carries injected ground-truth labels, the advanced dashboard computes a live confusion matrix, precision/recall/F1, and overall accuracy of the chosen detection method against those labels.
- **Exploration tools:** correlation heatmaps, distribution histograms with threshold lines, box plots, parameter importance, per-flight anomaly tables, and a filterable data explorer with CSV export.
- **Auto-refresh:** the live dashboards re-poll on a 2–3 second timer to imitate a streaming feed.

## How It Works

The project is built as a small pipeline: generate data → preprocess and scale → detect anomalies (statistical + ML) → visualize and score. Here is what each piece actually does.

### 1. Data generation (`Data.py`)

There is no real drone here, so the data is synthesized. For each of 50 flights, the script generates 200 timesteps of telemetry:

- altitude as a random walk around 10 m, velocity around 1 m/s, yaw/pitch as small-noise signals, battery as a linear drain from 100% to 80%, and GPS drift as low-amplitude noise.
- Five anomalies per flight are injected at random timesteps by adding sharp offsets to altitude, velocity, and GPS drift, and the affected rows are labeled `anomaly = 1`.

The result is written to `drone_telemetry.csv` — 10,000 rows, six features plus a flight id, timestep, and ground-truth label.

### 2. Preprocessing

Features are scaled with `MinMaxScaler` before any model sees them, so no single channel dominates by virtue of its units. For the LSTM path, the scaled series is also chopped into overlapping sliding windows of 20 timesteps.

### 3. Detection — statistical layer

Two flavors:

- The **basic dashboard** uses hard-coded safe ranges (e.g. altitude 8–15 m, velocity 0.5–2.0 m/s, GPS drift ±0.5) and marks a point anomalous if any channel falls outside its range.
- The **advanced** and **ML** dashboards learn each parameter's mean and standard deviation from the data and flag values outside mean ± 2.5σ. The advanced dashboard requires at least two parameters out of range before it counts a point as a statistical anomaly, which cuts down on noise.

### 4. Detection — Isolation Forest

The ML layer is an `IsolationForest` (100 trees, `contamination=0.1`, fixed random seed) fit on the scaled six-feature matrix. It returns a per-point prediction (anomaly / normal) and a raw anomaly score; the ML dashboard also normalizes that score into a 0–1 confidence value. Isolation Forest is a good fit here because it is unsupervised — it does not need the labels to find outliers — and it handles the multi-dimensional case where a reading is only suspicious in combination.

### 5. LSTM autoencoder (`Model Train.py`)

A separate Keras `Sequential` model: an LSTM encoder (64 units) compresses a 20-step window into a single vector, `RepeatVector` re-expands it, a second 64-unit LSTM decodes it, and a `TimeDistributed(Dense)` layer reconstructs the original window. It trains for 10 epochs with an Adam optimizer and MSE loss to reconstruct only normal-looking sequences. At inference, windows with a reconstruction error above `mean + 2·std` are called anomalies. The trained weights are saved to `drone_anomaly_lstm_model.h5`.

Worth being clear about: this LSTM is a standalone train-and-save artifact. The three live dashboards run on the Isolation Forest and statistical checks, not on the `.h5` model — the autoencoder is included as a second, deeper modeling approach you can train and inspect on its own, not as the engine behind the real-time UI.

### 6. Dashboards

- **`streamlit_dashboard.py` (basic)** — auto-refreshing real-time view. Fixed-range anomaly flags, current-value metric cards with deltas, per-flight selection, a Normal/Warning/Critical sidebar indicator, six time-series subplots, an anomaly pie + per-parameter bar chart, a correlation heatmap, and a 3D flight path.
- **`advanced_dashboard.py` (advanced)** — wraps detection in a `DroneAnomalyDetector` class and adds five tabs: overview, anomaly analysis (parameter breakdown, severity histogram, anomaly-correlation matrix, temporal anomaly rate), parameter monitoring (importance, learned thresholds, distributions with threshold lines, box plots), model performance (confusion matrix, precision/recall/F1, accuracy vs ground truth), and a data explorer with column filtering and CSV download. You can switch the detection method live between Combined, Isolation Forest, Statistical, and Ground Truth.
- **`drone_dashboard.py` (ML)** — a polished single-page view with a styled gradient UI. Auto-trains the Isolation Forest on load, shows ML vs statistical anomaly counts side by side plus an average confidence score, a 2×3 analysis grid, an anomaly-score distribution, per-parameter analysis with threshold lines, and a recent-anomalies table.
- **`dashboard_launcher.py`** — a small landing page that compares the dashboards feature-by-feature and prints the exact `streamlit run` command for each.

## Results / Highlights

This is validated on synthetic data only, so there is no reported real-world accuracy — but the configuration and scale are concrete:

- **Dataset:** 50 flights × 200 timesteps = 10,000 rows, 6 features, with 5 injected anomalies per flight (~2.5% anomaly rate) and ground-truth labels.
- **Isolation Forest:** 100 estimators, contamination 0.1, MinMax-scaled inputs.
- **LSTM autoencoder:** 64-unit encoder/decoder, sequence length 20, 10 training epochs, Adam + MSE, reconstruction threshold at mean + 2σ.
- **Built-in evaluation:** because the data is labeled, the advanced dashboard computes a confusion matrix and precision/recall/F1/accuracy live for whichever method you pick — so you can compare the combined, Isolation-Forest-only, and statistical-only detectors against the injected truth yourself.

## Tech Stack

- **Language:** Python 3.x
- **Frameworks / UI:** Streamlit, with `streamlit-autorefresh` and `streamlit-option-menu`
- **Data / ML:** pandas, NumPy, scikit-learn (Isolation Forest, MinMaxScaler, classification metrics), TensorFlow / Keras (LSTM autoencoder)
- **Visualization:** Plotly (Express + Graph Objects + subplots), Matplotlib, Seaborn
- **Utilities:** joblib

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

```bash
git clone https://github.com/DCode-v05/Aero-Guard.git
cd Aero-Guard
pip install -r requirements.txt
```

### Generate data (optional)

The repo already ships `drone_telemetry.csv`. To regenerate a fresh synthetic set:

```bash
python Data.py
```

### Train the LSTM (optional)

A pre-trained `drone_anomaly_lstm_model.h5` is included. To retrain it:

```bash
python "Model Train.py"
```

### Run a dashboard

```bash
# Launcher (compares the dashboards and gives you the run commands)
streamlit run dashboard_launcher.py

# Basic real-time monitoring
streamlit run streamlit_dashboard.py

# Advanced analytics + ML detection + performance metrics
streamlit run advanced_dashboard.py

# Polished single-page ML dashboard
streamlit run drone_dashboard.py
```

## Usage

Open any dashboard in the browser tab Streamlit launches. Pick a flight from the sidebar and, on the advanced/ML views, set a timestep range. The page loads the telemetry, runs detection, and shows the metric cards, time-series plots, and a Normal/Warning/Critical status for the selected window. On the advanced dashboard you can switch the detection method (Combined, Isolation Forest, Statistical, or Ground Truth) to see how each behaves, open the Model Performance tab to score the method against the injected labels, and use the data explorer to filter to anomalies only and export them to CSV. The basic and ML dashboards auto-refresh every couple of seconds to imitate a live feed.

## Project Structure

```
Aero-Guard/
├── Data.py                       # Synthetic telemetry generator (50 flights x 200 steps, injected anomalies)
├── Model Train.py                # Trains and saves the LSTM autoencoder
├── streamlit_dashboard.py        # Basic real-time monitoring dashboard (fixed-range checks)
├── advanced_dashboard.py         # Advanced dashboard: ML + statistical detection, 5 tabs, performance metrics
├── drone_dashboard.py            # Polished single-page Isolation Forest dashboard
├── dashboard_launcher.py         # Landing page comparing the dashboards + run commands
├── drone_telemetry.csv           # Generated dataset (10,000 rows, 6 features + labels)
├── drone_anomaly_lstm_model.h5   # Pre-trained LSTM autoencoder weights
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
