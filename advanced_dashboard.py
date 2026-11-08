import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DroneAnomalyDetector:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.is_fitted = False
        self.feature_columns = ['altitude', 'velocity', 'yaw', 'pitch', 'battery', 'gps_drift']
        self.thresholds = {}
        
    def fit(self, data):
        """Train the anomaly detection model"""
        features = data[self.feature_columns]
        
        scaled_features = self.scaler.fit_transform(features)
        
        self.isolation_forest.fit(scaled_features)
        
        for col in self.feature_columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            self.thresholds[col] = {
                'lower': mean_val - 2.5 * std_val,
                'upper': mean_val + 2.5 * std_val,
                'mean': mean_val,
                'std': std_val
            }
        
        self.is_fitted = True
        return self
    
    def predict_anomalies(self, data):
        """Predict anomalies using multiple methods"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        features = data[self.feature_columns]
        scaled_features = self.scaler.transform(features)
        
        isolation_predictions = self.isolation_forest.predict(scaled_features)
        isolation_scores = self.isolation_forest.score_samples(scaled_features)
        
        statistical_anomalies = np.zeros(len(data))
        threshold_details = {}
        
        for i, col in enumerate(self.feature_columns):
            col_anomalies = (
                (data[col] < self.thresholds[col]['lower']) |
                (data[col] > self.thresholds[col]['upper'])
            ).astype(int)
            statistical_anomalies += col_anomalies
            threshold_details[f'{col}_anomaly'] = col_anomalies.astype(bool)
        
        combined_anomalies = (
            (isolation_predictions == -1) |
            (statistical_anomalies >= 2)
        ).astype(int)
        
        results = data.copy()
        results['isolation_anomaly'] = (isolation_predictions == -1).astype(int)
        results['isolation_score'] = isolation_scores
        results['statistical_anomaly'] = (statistical_anomalies >= 2).astype(int)
        results['combined_anomaly'] = combined_anomalies
        results['anomaly_score'] = statistical_anomalies
        
        for col, anomaly_data in threshold_details.items():
            results[col] = anomaly_data.astype(int)
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance based on anomaly frequency"""
        if not self.is_fitted:
            return None
        
        importance = {}
        for col in self.feature_columns:
            importance[col] = 1.0 / (self.thresholds[col]['std'] + 1e-6)
        
        total_importance = sum(importance.values())
        for col in importance:
            importance[col] /= total_importance
            
        return importance

@st.cache_data
def load_and_process_data():
    """Load and preprocess the drone data"""
    try:
        df = pd.read_csv("drone_telemetry.csv")
        return df
    except FileNotFoundError:
        st.error("drone_telemetry.csv not found. Please run Data.py first to generate the dataset.")
        return None

@st.cache_resource
def train_anomaly_detector(df):
    """Train and cache the anomaly detector"""
    detector = DroneAnomalyDetector()
    detector.fit(df)
    return detector

def create_advanced_dashboard():
    st.set_page_config(
        page_title="Advanced Drone Anomaly Detection",
        page_icon="🚁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .big-font {
            font-size: 20px !important;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .alert-critical {
            background-color: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #d32f2f;
        }
        .alert-warning {
            background-color: #fff8e1;
            color: #f57c00;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #ff9800;
        }
        .alert-success {
            background-color: #e8f5e8;
            color: #2e7d2e;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #4caf50;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("# 🚁 Advanced Drone Flight Anomaly Detection System")
    st.markdown("### Real-time monitoring and intelligent anomaly detection for drone operations")
    
    df = load_and_process_data()
    if df is None:
        st.stop()
    
    with st.spinner("Training anomaly detection model..."):
        detector = train_anomaly_detector(df)
    
    st.sidebar.header("🎛️ Control Panel")
    
    flights = sorted(df['flight'].unique())
    selected_flights = st.sidebar.multiselect(
        "Select Flights to Analyze",
        flights,
        default=[flights[0]] if flights else []
    )
    
    if not selected_flights:
        st.warning("Please select at least one flight to analyze.")
        st.stop()
    
    filtered_df = df[df['flight'].isin(selected_flights)].copy()
    
    detection_method = st.sidebar.selectbox(
        "Anomaly Detection Method",
        ["Combined (Recommended)", "Isolation Forest", "Statistical Thresholds", "Ground Truth"]
    )
    
    max_timesteps = filtered_df['timestep'].max()
    time_range = st.sidebar.slider(
        "Time Range",
        0, int(max_timesteps),
        (0, min(100, int(max_timesteps))),
        step=1
    )
    
    time_filtered_df = filtered_df[
        (filtered_df['timestep'] >= time_range[0]) & 
        (filtered_df['timestep'] <= time_range[1])
    ].copy()
    
    with st.spinner("Detecting anomalies..."):
        results = detector.predict_anomalies(time_filtered_df)
    
    anomaly_col_map = {
        "Combined (Recommended)": "combined_anomaly",
        "Isolation Forest": "isolation_anomaly",
        "Statistical Thresholds": "statistical_anomaly",
        "Ground Truth": "anomaly"
    }
    anomaly_col = anomaly_col_map[detection_method]
    
    total_points = len(results)
    anomaly_count = results[anomaly_col].sum()
    anomaly_rate = (anomaly_count / total_points) * 100 if total_points > 0 else 0
    
    if anomaly_rate > 20:
        status = "🔴 CRITICAL"
        status_class = "alert-critical"
    elif anomaly_rate > 10:
        status = "🟡 WARNING" 
        status_class = "alert-warning"
    else:
        status = "🟢 NORMAL"
        status_class = "alert-success"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Data Points",
            f"{total_points:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Anomalies Detected", 
            f"{anomaly_count}",
            delta=f"{anomaly_rate:.1f}% of total"
        )
    
    with col3:
        avg_score = results['isolation_score'].mean() if 'isolation_score' in results.columns else 0
        st.metric(
            "Avg Anomaly Score",
            f"{avg_score:.3f}",
            delta=None
        )
    
    with col4:
        flights_with_anomalies = len(results[results[anomaly_col] > 0]['flight'].unique())
        st.metric(
            "Flights with Anomalies",
            f"{flights_with_anomalies}",
            delta=f"{len(selected_flights)} total"
        )
    
    st.markdown(f"""
    <div class="{status_class}">
        <strong>System Status: {status}</strong><br>
        {anomaly_count} anomalies detected out of {total_points} data points ({anomaly_rate:.1f}%)
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview Dashboard", 
        "🔍 Anomaly Analysis", 
        "📈 Parameter Monitoring",
        "🤖 Model Performance",
        "📋 Detailed Data"
    ])
    
    with tab1:
        st.subheader("Real-time Flight Monitoring Dashboard")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Altitude vs Time', 'Velocity vs Time', 
                'Yaw & Pitch', 'Battery Level', 
                'GPS Drift', 'Anomaly Detection'
            ],
            vertical_spacing=0.1
        )
        
        normal_mask = results[anomaly_col] == 0
        anomaly_mask = results[anomaly_col] == 1
        
        fig.add_trace(go.Scatter(
            x=results[normal_mask]['timestep'], 
            y=results[normal_mask]['altitude'],
            mode='lines+markers',
            name='Normal Altitude',
            line=dict(color='blue'),
            marker=dict(size=3)
        ), row=1, col=1)
        
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(
                x=results[anomaly_mask]['timestep'], 
                y=results[anomaly_mask]['altitude'],
                mode='markers',
                name='Anomaly Altitude',
                marker=dict(color='red', size=6, symbol='x')
            ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=results[normal_mask]['timestep'], 
            y=results[normal_mask]['velocity'],
            mode='lines+markers',
            name='Normal Velocity',
            line=dict(color='green'),
            marker=dict(size=3)
        ), row=1, col=2)
        
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(
                x=results[anomaly_mask]['timestep'], 
                y=results[anomaly_mask]['velocity'],
                mode='markers',
                name='Anomaly Velocity',
                marker=dict(color='red', size=6, symbol='x')
            ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=results['timestep'], y=results['yaw'],
            mode='lines', name='Yaw',
            line=dict(color='orange')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=results['timestep'], y=results['pitch'],
            mode='lines', name='Pitch',
            line=dict(color='purple')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=results['timestep'], y=results['battery'],
            mode='lines+markers', name='Battery',
            line=dict(color='red', width=2),
            marker=dict(size=3)
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=results['timestep'], y=results['gps_drift'],
            mode='lines+markers', name='GPS Drift',
            line=dict(color='brown'),
            marker=dict(size=3)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=results['timestep'], 
            y=results[anomaly_col],
            mode='markers',
            name='Anomaly Detection',
            marker=dict(
                color=results[anomaly_col], 
                colorscale=[[0, 'green'], [1, 'red']],
                size=6
            )
        ), row=3, col=2)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Flight Statistics")
            latest_flight = results[results['flight'] == selected_flights[-1]].iloc[-10:] if len(results) > 0 else pd.DataFrame()
            
            if len(latest_flight) > 0:
                stats_data = {
                    'Parameter': ['Altitude', 'Velocity', 'Yaw', 'Pitch', 'Battery', 'GPS Drift'],
                    'Current': [
                        f"{latest_flight['altitude'].iloc[-1]:.2f} m",
                        f"{latest_flight['velocity'].iloc[-1]:.2f} m/s", 
                        f"{latest_flight['yaw'].iloc[-1]:.2f}°",
                        f"{latest_flight['pitch'].iloc[-1]:.2f}°",
                        f"{latest_flight['battery'].iloc[-1]:.1f}%",
                        f"{latest_flight['gps_drift'].iloc[-1]:.3f}"
                    ],
                    'Trend': [
                        "↗️" if latest_flight['altitude'].diff().iloc[-1] > 0 else "↘️",
                        "↗️" if latest_flight['velocity'].diff().iloc[-1] > 0 else "↘️",
                        "↗️" if latest_flight['yaw'].diff().iloc[-1] > 0 else "↘️",
                        "↗️" if latest_flight['pitch'].diff().iloc[-1] > 0 else "↘️",
                        "↗️" if latest_flight['battery'].diff().iloc[-1] > 0 else "↘️",
                        "↗️" if latest_flight['gps_drift'].diff().iloc[-1] > 0 else "↘️"
                    ]
                }
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        with col2:
            st.subheader("Anomaly Distribution by Flight")
            flight_anomalies = results.groupby('flight')[anomaly_col].agg(['count', 'sum']).reset_index()
            flight_anomalies['anomaly_rate'] = (flight_anomalies['sum'] / flight_anomalies['count'] * 100).round(2)
            flight_anomalies.columns = ['Flight', 'Total Points', 'Anomalies', 'Anomaly Rate (%)']
            st.dataframe(flight_anomalies, use_container_width=True)
    
    with tab2:
        st.subheader("Detailed Anomaly Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            param_cols = [col for col in results.columns if col.endswith('_anomaly') and col != 'combined_anomaly']
            param_anomaly_counts = {}
            for col in param_cols:
                param_name = col.replace('_anomaly', '').replace('_', ' ').title()
                param_anomaly_counts[param_name] = results[col].sum()
            
            if param_anomaly_counts:
                fig_params = go.Figure(data=[go.Bar(
                    x=list(param_anomaly_counts.keys()),
                    y=list(param_anomaly_counts.values()),
                    marker_color=['red' if v > 0 else 'green' for v in param_anomaly_counts.values()]
                )])
                fig_params.update_layout(
                    title="Anomalies by Parameter Type",
                    xaxis_title="Parameter",
                    yaxis_title="Number of Anomalies"
                )
                st.plotly_chart(fig_params, use_container_width=True)
        
        with col2:
            if 'anomaly_score' in results.columns:
                fig_severity = px.histogram(
                    results[results[anomaly_col] == 1],
                    x='anomaly_score',
                    nbins=10,
                    title="Anomaly Severity Distribution",
                    labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_severity, use_container_width=True)
        
        st.subheader("Parameter Anomaly Correlations")
        anomaly_corr_cols = [col for col in results.columns if col.endswith('_anomaly')]
        if len(anomaly_corr_cols) > 1:
            corr_matrix = results[anomaly_corr_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Anomaly Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Temporal Anomaly Patterns")
        results['time_window'] = (results['timestep'] // 10) * 10
        temporal_anomalies = results.groupby('time_window')[anomaly_col].agg(['count', 'sum']).reset_index()
        temporal_anomalies['anomaly_rate'] = temporal_anomalies['sum'] / temporal_anomalies['count']
        
        fig_temporal = go.Figure()
        fig_temporal.add_trace(go.Scatter(
            x=temporal_anomalies['time_window'],
            y=temporal_anomalies['anomaly_rate'],
            mode='lines+markers',
            name='Anomaly Rate',
            line=dict(color='red', width=2)
        ))
        fig_temporal.update_layout(
            title="Anomaly Rate Over Time Windows",
            xaxis_title="Time Window",
            yaxis_title="Anomaly Rate"
        )
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    with tab3:
        st.subheader("Parameter Monitoring and Thresholds")
        
        importance = detector.get_feature_importance()
        if importance:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_importance = go.Figure(data=[go.Bar(
                    x=list(importance.keys()),
                    y=list(importance.values()),
                    marker_color='skyblue'
                )])
                fig_importance.update_layout(
                    title="Parameter Importance for Anomaly Detection",
                    xaxis_title="Parameter",
                    yaxis_title="Importance Score"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2:
                st.subheader("Statistical Thresholds")
                threshold_data = []
                for param, thresholds in detector.thresholds.items():
                    threshold_data.append({
                        'Parameter': param.title(),
                        'Lower Threshold': f"{thresholds['lower']:.3f}",
                        'Upper Threshold': f"{thresholds['upper']:.3f}",
                        'Mean': f"{thresholds['mean']:.3f}",
                        'Std Dev': f"{thresholds['std']:.3f}"
                    })
                st.dataframe(pd.DataFrame(threshold_data), use_container_width=True)
        
        st.subheader("Parameter Distributions with Anomaly Thresholds")
        
        selected_param = st.selectbox(
            "Select Parameter to Analyze",
            detector.feature_columns
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.histogram(
                results,
                x=selected_param,
                color=anomaly_col,
                nbins=30,
                title=f"Distribution of {selected_param.title()}",
                color_discrete_map={0: 'blue', 1: 'red'}
            )
            
            if selected_param in detector.thresholds:
                thresholds = detector.thresholds[selected_param]
                fig_dist.add_vline(
                    x=thresholds['lower'], 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text="Lower Threshold"
                )
                fig_dist.add_vline(
                    x=thresholds['upper'], 
                    line_dash="dash", 
                    line_color="orange",
                    annotation_text="Upper Threshold"
                )
                fig_dist.add_vline(
                    x=thresholds['mean'], 
                    line_dash="solid", 
                    line_color="green",
                    annotation_text="Mean"
                )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                results,
                y=selected_param,
                color=anomaly_col,
                title=f"Box Plot of {selected_param.title()}",
                color_discrete_map={0: 'blue', 1: 'red'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    with tab4:
        st.subheader("Model Performance Analysis")
        
        if 'anomaly' in results.columns and anomaly_col != 'anomaly':
            col1, col2 = st.columns(2)
            
            with col1:
                y_true = results['anomaly']
                y_pred = results[anomaly_col]
                
                cm = confusion_matrix(y_true, y_pred)
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual")
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                report = classification_report(y_true, y_pred, output_dict=True)
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Precision', 'Recall', 'F1-Score'],
                    'Normal': [
                        report['0']['precision'],
                        report['0']['recall'], 
                        report['0']['f1-score']
                    ],
                    'Anomaly': [
                        report['1']['precision'],
                        report['1']['recall'],
                        report['1']['f1-score']
                    ]
                })
                
                fig_metrics = px.bar(
                    metrics_df.melt(id_vars='Metric'),
                    x='Metric',
                    y='value',
                    color='variable',
                    barmode='group',
                    title="Classification Metrics",
                    labels={'value': 'Score', 'variable': 'Class'}
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
                
                accuracy = (y_true == y_pred).mean()
                st.metric("Overall Accuracy", f"{accuracy:.3f}")
        
        st.subheader("Anomaly Score Distribution")
        if 'isolation_score' in results.columns:
            fig_scores = px.histogram(
                results,
                x='isolation_score',
                color=anomaly_col,
                nbins=30,
                title="Isolation Forest Anomaly Scores",
                color_discrete_map={0: 'blue', 1: 'red'}
            )
            st.plotly_chart(fig_scores, use_container_width=True)
        
        st.subheader("Feature Scaling Visualization")
        original_features = results[detector.feature_columns]
        scaled_features = detector.scaler.transform(original_features)
        scaled_df = pd.DataFrame(scaled_features, columns=detector.feature_columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Feature Ranges**")
            st.dataframe(original_features.describe(), use_container_width=True)
        
        with col2:
            st.write("**Scaled Feature Ranges**")
            st.dataframe(scaled_df.describe(), use_container_width=True)
    
    with tab5:
        st.subheader("Detailed Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_only_anomalies = st.checkbox("Show Only Anomalies")
        
        with col2:
            selected_columns = st.multiselect(
                "Select Columns to Display",
                results.columns.tolist(),
                default=['flight', 'timestep', 'altitude', 'velocity', 'battery', anomaly_col]
            )
        
        with col3:
            export_data = st.button("Export Data")
        
        display_data = results.copy()
        if show_only_anomalies:
            display_data = display_data[display_data[anomaly_col] == 1]
        
        if selected_columns:
            display_data = display_data[selected_columns]
        
        st.dataframe(
            display_data,
            use_container_width=True,
            height=400
        )
        
        if export_data:
            csv = display_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"drone_anomaly_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(display_data))
        
        with col2:
            if show_only_anomalies:
                st.metric("Anomaly Records", len(display_data))
            else:
                st.metric("Anomaly Records", display_data[anomaly_col].sum() if anomaly_col in display_data.columns else 0)
        
        with col3:
            st.metric("Time Range", f"{display_data['timestep'].min():.0f} - {display_data['timestep'].max():.0f}" if 'timestep' in display_data.columns else "N/A")

if __name__ == "__main__":
    create_advanced_dashboard()