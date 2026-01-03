import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Drone Anomaly Detection System",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .status-normal {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #FF9800, #f57c00);
        color: white;
    }
    
    .status-critical {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    /* Hide streamlit branding */
    .stApp > header {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

class DroneAnomalyModel:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.feature_columns = ['altitude', 'velocity', 'yaw', 'pitch', 'battery', 'gps_drift']
        self.is_trained = False
        self.training_stats = {}
        
    def train_model(self, data):
        """Train the anomaly detection model"""
        try:
            features = data[self.feature_columns]
            
            # Store training statistics
            self.training_stats = {
                'mean': features.mean().to_dict(),
                'std': features.std().to_dict(),
                'min': features.min().to_dict(),
                'max': features.max().to_dict()
            }
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(scaled_features)
            self.is_trained = True
            
            return True
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def predict_anomalies(self, data):
        """Predict anomalies on new data"""
        if not self.is_trained:
            return None, "Model not trained yet!"
        
        try:
            features = data[self.feature_columns]
            scaled_features = self.scaler.transform(features)
            
            # Predict anomalies (-1 = anomaly, 1 = normal)
            predictions = self.model.predict(scaled_features)
            anomaly_scores = self.model.score_samples(scaled_features)
            
            # Convert predictions to binary (1 = anomaly, 0 = normal)
            anomalies = (predictions == -1).astype(int)
            
            # Calculate confidence scores (normalized)
            confidence = np.abs(anomaly_scores)
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)
            
            results = data.copy()
            results['predicted_anomaly'] = anomalies
            results['anomaly_score'] = anomaly_scores
            results['confidence'] = confidence
            
            # Add statistical anomalies for comparison
            stat_anomalies = np.zeros(len(data))
            for col in self.feature_columns:
                mean_val = self.training_stats['mean'][col]
                std_val = self.training_stats['std'][col]
                threshold_low = mean_val - 2.5 * std_val
                threshold_high = mean_val + 2.5 * std_val
                
                col_anomalies = (
                    (data[col] < threshold_low) | 
                    (data[col] > threshold_high)
                ).astype(int)
                stat_anomalies += col_anomalies
                results[f'{col}_anomaly'] = col_anomalies
            
            results['statistical_anomaly'] = (stat_anomalies >= 1).astype(int)
            
            return results, "Predictions completed successfully!"
            
        except Exception as e:
            return None, f"Prediction failed: {str(e)}"

@st.cache_data
def load_data():
    """Load drone telemetry data"""
    try:
        df = pd.read_csv("drone_telemetry.csv")
        return df
    except FileNotFoundError:
        st.error("❌ drone_telemetry.csv not found! Run Data.py first to generate sample data.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def initialize_model():
    """Initialize and return model instance"""
    return DroneAnomalyModel()

def create_dashboard():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🚁 Drone Flight Anomaly Detection System</div>
        <div class="main-subtitle">AI-Powered Real-time Monitoring & Anomaly Detection</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data automatically
    with st.spinner("🔄 Loading drone telemetry data..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Initialize model
    model = initialize_model()
    
    # Auto-train model on first run
    if not model.is_trained:
        with st.spinner("🤖 Training anomaly detection model..."):
            success, message = model.train_model(df)
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
                st.stop()
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Flight selection
        flights = sorted(df['flight'].unique())
        selected_flight = st.selectbox("🛩️ Select Flight", flights, index=0)
        
        # Time range
        flight_data = df[df['flight'] == selected_flight]
        max_time = int(flight_data['timestep'].max())
        time_range = st.slider("⏱️ Time Range", 0, max_time, (0, min(100, max_time)))
        
        # Auto-refresh toggle
        auto_refresh = st.toggle("🔄 Auto Refresh", value=True)
        
        if auto_refresh:
            st_autorefresh(interval=3000, key="refresh")
        
        # Model info
        st.markdown("### 🤖 Model Status")
        st.markdown(f"""
        <div class="info-box">
            <strong>Status:</strong> {'✅ Trained' if model.is_trained else '❌ Not Trained'}<br>
            <strong>Algorithm:</strong> Isolation Forest<br>
            <strong>Features:</strong> 6 parameters<br>
            <strong>Training Data:</strong> {len(df):,} points
        </div>
        """, unsafe_allow_html=True)
    
    # Filter data based on selection
    filtered_data = df[
        (df['flight'] == selected_flight) & 
        (df['timestep'] >= time_range[0]) & 
        (df['timestep'] <= time_range[1])
    ].copy()
    
    if len(filtered_data) == 0:
        st.warning("⚠️ No data found for selected filters.")
        st.stop()
    
    # Make predictions
    with st.spinner("🔮 Generating predictions..."):
        predictions, pred_message = model.predict_anomalies(filtered_data)
    
    if predictions is None:
        st.error(f"❌ {pred_message}")
        st.stop()
    
    # Calculate key metrics
    total_points = len(predictions)
    ml_anomalies = predictions['predicted_anomaly'].sum()
    stat_anomalies = predictions['statistical_anomaly'].sum()
    ml_anomaly_rate = (ml_anomalies / total_points * 100) if total_points > 0 else 0
    
    # Determine system status
    if ml_anomaly_rate > 15:
        status = "CRITICAL ALERT"
        status_class = "status-critical"
        status_icon = "🚨"
    elif ml_anomaly_rate > 5:
        status = "WARNING"
        status_class = "status-warning"
        status_icon = "⚠️"
    else:
        status = "NORMAL OPERATION"
        status_class = "status-normal"
        status_icon = "✅"
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">📊 Data Points</h3>
            <h2 style="margin: 0;">{:,}</h2>
            <p style="color: #666; margin: 0;">Flight {}</p>
        </div>
        """.format(total_points, selected_flight), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #e74c3c; margin-bottom: 0.5rem;">🤖 ML Predictions</h3>
            <h2 style="margin: 0; color: #e74c3c;">{}</h2>
            <p style="color: #666; margin: 0;">{:.1f}% anomalous</p>
        </div>
        """.format(ml_anomalies, ml_anomaly_rate), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #f39c12; margin-bottom: 0.5rem;">📈 Statistical</h3>
            <h2 style="margin: 0; color: #f39c12;">{}</h2>
            <p style="color: #666; margin: 0;">Threshold based</p>
        </div>
        """.format(stat_anomalies), unsafe_allow_html=True)
    
    with col4:
        avg_confidence = predictions['confidence'].mean()
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #9b59b6; margin-bottom: 0.5rem;">🎯 Confidence</h3>
            <h2 style="margin: 0; color: #9b59b6;">{:.1f}%</h2>
            <p style="color: #666; margin: 0;">Prediction accuracy</p>
        </div>
        """.format(avg_confidence * 100), unsafe_allow_html=True)
    

    # Create visualizations
    st.markdown("## 📈 Real-time Flight Analysis")
    
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            '🏔️ Altitude Monitoring', '⚡ Velocity Profile', '🔋 Battery Status',
            '🧭 Orientation (Yaw/Pitch)', '📡 GPS Drift', '🚨 Anomaly Detection'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Color scheme
    normal_color = '#2E86AB'
    anomaly_color = '#F24236'
    
    # Separate normal and anomaly points
    normal_data = predictions[predictions['predicted_anomaly'] == 0]
    anomaly_data = predictions[predictions['predicted_anomaly'] == 1]
    
    # Altitude
    fig.add_trace(go.Scatter(
        x=normal_data['timestep'], y=normal_data['altitude'],
        mode='lines+markers', name='Normal Altitude',
        line=dict(color=normal_color, width=2),
        marker=dict(size=4), showlegend=True
    ), row=1, col=1)
    
    if len(anomaly_data) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_data['timestep'], y=anomaly_data['altitude'],
            mode='markers', name='Anomaly Points',
            marker=dict(color=anomaly_color, size=8, symbol='x'),
            showlegend=False
        ), row=1, col=1)
    
    # Velocity
    fig.add_trace(go.Scatter(
        x=normal_data['timestep'], y=normal_data['velocity'],
        mode='lines+markers', name='Normal Velocity',
        line=dict(color='#A23B72', width=2),
        marker=dict(size=4), showlegend=False
    ), row=1, col=2)
    
    if len(anomaly_data) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_data['timestep'], y=anomaly_data['velocity'],
            mode='markers', name='Velocity Anomaly',
            marker=dict(color=anomaly_color, size=8, symbol='x'),
            showlegend=False
        ), row=1, col=2)
    
    # Battery
    fig.add_trace(go.Scatter(
        x=predictions['timestep'], y=predictions['battery'],
        mode='lines+markers', name='Battery Level',
        line=dict(color='#F18F01', width=3),
        marker=dict(size=4), showlegend=False
    ), row=1, col=3)
    
    # Yaw and Pitch
    fig.add_trace(go.Scatter(
        x=predictions['timestep'], y=predictions['yaw'],
        mode='lines', name='Yaw', line=dict(color='#C73E1D', width=2),
        showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=predictions['timestep'], y=predictions['pitch'],
        mode='lines', name='Pitch', line=dict(color='#592941', width=2),
        showlegend=False
    ), row=2, col=1)
    
    # GPS Drift
    fig.add_trace(go.Scatter(
        x=normal_data['timestep'], y=normal_data['gps_drift'],
        mode='lines+markers', name='Normal GPS',
        line=dict(color='#6A994E', width=2),
        marker=dict(size=4), showlegend=False
    ), row=2, col=2)
    
    if len(anomaly_data) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_data['timestep'], y=anomaly_data['gps_drift'],
            mode='markers', name='GPS Anomaly',
            marker=dict(color=anomaly_color, size=8, symbol='x'),
            showlegend=False
        ), row=2, col=2)
    
    # Anomaly detection timeline
    fig.add_trace(go.Scatter(
        x=predictions['timestep'], y=predictions['predicted_anomaly'],
        mode='markers', name='ML Predictions',
        marker=dict(
            color=predictions['predicted_anomaly'],
            colorscale=[[0, normal_color], [1, anomaly_color]],
            size=8, line=dict(width=1, color='white')
        ), showlegend=False
    ), row=2, col=3)
    
    fig.update_layout(
        height=600,
        title_text="🚁 Comprehensive Flight Analysis Dashboard",
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Time Step")
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Additional analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Anomaly Score Distribution")
        
        
        fig_hist = px.histogram(
            predictions, x='anomaly_score', 
            color='predicted_anomaly',
            nbins=30,
            title="Model Confidence Distribution",
            color_discrete_map={0: normal_color, 1: anomaly_color},
            labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Parameter Analysis")
        
        
        # Parameter selection for detailed view
        param = st.selectbox("Select Parameter", model.feature_columns, 
                           format_func=lambda x: x.replace('_', ' ').title())
        
        fig_param = go.Figure()
        
        # Normal points
        fig_param.add_trace(go.Scatter(
            x=normal_data['timestep'], 
            y=normal_data[param],
            mode='markers',
            name='Normal',
            marker=dict(color=normal_color, size=6, opacity=0.7)
        ))
        
        # Anomaly points
        if len(anomaly_data) > 0:
            fig_param.add_trace(go.Scatter(
                x=anomaly_data['timestep'], 
                y=anomaly_data[param],
                mode='markers',
                name='Anomalies',
                marker=dict(color=anomaly_color, size=10, symbol='x')
            ))
        
        # Add statistical thresholds
        if param in model.training_stats['mean']:
            mean_val = model.training_stats['mean'][param]
            std_val = model.training_stats['std'][param]
            
            fig_param.add_hline(
                y=mean_val + 2.5 * std_val,
                line_dash="dash", line_color="orange",
                annotation_text="Upper Threshold"
            )
            fig_param.add_hline(
                y=mean_val - 2.5 * std_val,
                line_dash="dash", line_color="orange",
                annotation_text="Lower Threshold"
            )
            fig_param.add_hline(
                y=mean_val,
                line_dash="solid", line_color="green",
                annotation_text="Mean"
            )
        
        fig_param.update_layout(
            title=f"{param.title()} Analysis",
            xaxis_title="Time Step",
            yaxis_title=param.title(),
            height=400
        )
        
        st.plotly_chart(fig_param, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary table
    st.markdown("### 📋 Detailed Anomaly Summary")
    
    
    # Show recent anomalies
    recent_anomalies = predictions[predictions['predicted_anomaly'] == 1].tail(10)
    
    if len(recent_anomalies) > 0:
        st.write("**Recent Anomalies Detected:**")
        display_cols = ['timestep', 'altitude', 'velocity', 'battery', 'gps_drift', 'confidence']
        st.dataframe(
            recent_anomalies[display_cols].round(3),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("✅ No anomalies detected in the selected time range!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🚁 Drone Anomaly Detection System | Last Updated: {current_time} | 
        Model: Isolation Forest | Data Points: {total_points:,}
    </div>
    """, unsafe_allow_html=True)

def main():
    create_dashboard()

if __name__ == "__main__":
    main()