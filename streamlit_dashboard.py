import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Drone Flight Anomaly Detection Dashboard",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    .alert-success {
        background-color: #d1edfa;
        border-color: #17a2b8;
        color: #0c5460;
    }
    .status-indicator {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
    }
    .status-normal { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-critical { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("drone_telemetry.csv")
        return df
    except FileNotFoundError:
        st.error("drone_telemetry.csv file not found. Please ensure the data file exists.")
        return None

# Simulate real-time data
def simulate_realtime_data(df, current_time):
    """Simulate real-time streaming by selecting data based on current time"""
    if df is None:
        return None
    
    # Calculate which data points to show based on time
    total_points = len(df)
    points_to_show = (current_time % 100) + 50  # Show 50-150 points cyclically
    return df.head(points_to_show)

# Anomaly detection function
def detect_anomalies_simple(df):
    """Simple anomaly detection based on statistical thresholds"""
    if df is None or len(df) == 0:
        return df
    
    df = df.copy()
    
    # Define normal ranges for each parameter
    normal_ranges = {
        'altitude': (8, 15),
        'velocity': (0.5, 2.0),
        'yaw': (-2, 2),
        'pitch': (-2, 2),
        'battery': (20, 100),
        'gps_drift': (-0.5, 0.5)
    }
    
    # Calculate anomalies based on ranges
    df['altitude_anomaly'] = ~df['altitude'].between(*normal_ranges['altitude'])
    df['velocity_anomaly'] = ~df['velocity'].between(*normal_ranges['velocity'])
    df['yaw_anomaly'] = ~df['yaw'].between(*normal_ranges['yaw'])
    df['pitch_anomaly'] = ~df['pitch'].between(*normal_ranges['pitch'])
    df['battery_anomaly'] = ~df['battery'].between(*normal_ranges['battery'])
    df['gps_anomaly'] = ~df['gps_drift'].between(*normal_ranges['gps_drift'])
    
    # Overall anomaly if any parameter is anomalous
    df['detected_anomaly'] = (df['altitude_anomaly'] | df['velocity_anomaly'] | 
                            df['yaw_anomaly'] | df['pitch_anomaly'] | 
                            df['battery_anomaly'] | df['gps_anomaly'])
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">🚁 Drone Flight Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Auto-refresh every 2 seconds
    count = st_autorefresh(interval=2000, limit=None, key="data_refresh")
    
    # Load and process data
    df = load_data()
    if df is None:
        st.stop()
    
    # Simulate real-time data
    current_data = simulate_realtime_data(df, count)
    if current_data is None or len(current_data) == 0:
        st.warning("No data available")
        st.stop()
    
    # Detect anomalies
    processed_data = detect_anomalies_simple(current_data)
    
    # Sidebar configuration
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Flight selection
    available_flights = sorted(processed_data['flight'].unique())
    selected_flight = st.sidebar.selectbox("Select Flight", available_flights)
    
    # Filter data for selected flight
    flight_data = processed_data[processed_data['flight'] == selected_flight].copy()
    
    # Real-time status
    current_time = datetime.now()
    st.sidebar.markdown("### 📡 System Status")
    st.sidebar.markdown(f"**Last Update:** {current_time.strftime('%H:%M:%S')}")
    st.sidebar.markdown(f"**Data Points:** {len(flight_data)}")
    st.sidebar.markdown(f"**Flight ID:** {selected_flight}")
    
    # Calculate current anomaly status
    recent_anomalies = flight_data.tail(10)['detected_anomaly'].sum()
    if recent_anomalies >= 5:
        status = "🔴 CRITICAL"
        status_class = "status-critical"
    elif recent_anomalies >= 2:
        status = "🟡 WARNING"
        status_class = "status-warning"
    else:
        status = "🟢 NORMAL"
        status_class = "status-normal"
    
    st.sidebar.markdown(f"""
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <div class="status-indicator {status_class}"></div>
        <strong>{status}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    # Current metrics
    if len(flight_data) > 0:
        latest = flight_data.iloc[-1]
        
        with col1:
            st.metric(
                label="🏔️ Altitude (m)",
                value=f"{latest['altitude']:.2f}",
                delta=f"{latest['altitude'] - flight_data.iloc[-2]['altitude'] if len(flight_data) > 1 else 0:.2f}"
            )
        
        with col2:
            st.metric(
                label="⚡ Velocity (m/s)",
                value=f"{latest['velocity']:.2f}",
                delta=f"{latest['velocity'] - flight_data.iloc[-2]['velocity'] if len(flight_data) > 1 else 0:.2f}"
            )
        
        with col3:
            st.metric(
                label="🔋 Battery (%)",
                value=f"{latest['battery']:.1f}",
                delta=f"{latest['battery'] - flight_data.iloc[-2]['battery'] if len(flight_data) > 1 else 0:.1f}"
            )
        
        with col4:
            st.metric(
                label="📡 GPS Drift",
                value=f"{latest['gps_drift']:.3f}",
                delta=f"{latest['gps_drift'] - flight_data.iloc[-2]['gps_drift'] if len(flight_data) > 1 else 0:.3f}"
            )
    
    # Alert section
    anomaly_count = flight_data['detected_anomaly'].sum()
    total_points = len(flight_data)
    
    if anomaly_count > total_points * 0.1:  # More than 10% anomalies
        st.markdown(f"""
        <div class="alert-box alert-danger">
            <strong>🚨 HIGH ANOMALY ALERT!</strong><br>
            Detected {anomaly_count} anomalies out of {total_points} data points ({(anomaly_count/total_points*100):.1f}%)
        </div>
        """, unsafe_allow_html=True)
    elif anomaly_count > 0:
        st.markdown(f"""
        <div class="alert-box alert-warning">
            <strong>⚠️ Anomalies Detected</strong><br>
            Found {anomaly_count} anomalous readings. Monitor closely.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-box alert-success">
            <strong>✅ All Systems Normal</strong><br>
            No anomalies detected in current flight data.
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-time Monitoring", "🎯 Anomaly Analysis", "📊 Parameter Details", "🗺️ Flight Path"])
    
    with tab1:
        st.subheader("Real-time Flight Parameters")
        
        # Create subplots for real-time monitoring
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Altitude Over Time', 'Velocity Over Time', 'Yaw & Pitch', 'Battery Level', 'GPS Drift', 'Anomaly Detection'],
            vertical_spacing=0.08
        )
        
        # Altitude
        fig.add_trace(go.Scatter(
            x=flight_data['timestep'], 
            y=flight_data['altitude'],
            mode='lines+markers',
            name='Altitude',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ), row=1, col=1)
        
        # Velocity
        fig.add_trace(go.Scatter(
            x=flight_data['timestep'], 
            y=flight_data['velocity'],
            mode='lines+markers',
            name='Velocity',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ), row=1, col=2)
        
        # Yaw and Pitch
        fig.add_trace(go.Scatter(
            x=flight_data['timestep'], 
            y=flight_data['yaw'],
            mode='lines',
            name='Yaw',
            line=dict(color='orange', width=2)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=flight_data['timestep'], 
            y=flight_data['pitch'],
            mode='lines',
            name='Pitch',
            line=dict(color='purple', width=2)
        ), row=2, col=1)
        
        # Battery
        fig.add_trace(go.Scatter(
            x=flight_data['timestep'], 
            y=flight_data['battery'],
            mode='lines+markers',
            name='Battery',
            line=dict(color='red', width=3),
            marker=dict(size=4)
        ), row=2, col=2)
        
        # GPS Drift
        fig.add_trace(go.Scatter(
            x=flight_data['timestep'], 
            y=flight_data['gps_drift'],
            mode='lines+markers',
            name='GPS Drift',
            line=dict(color='brown', width=2),
            marker=dict(size=4)
        ), row=3, col=1)
        
        # Anomaly detection
        anomaly_points = flight_data[flight_data['detected_anomaly']]
        fig.add_trace(go.Scatter(
            x=flight_data['timestep'], 
            y=flight_data['detected_anomaly'].astype(int),
            mode='markers',
            name='Normal',
            marker=dict(color='green', size=6, symbol='circle')
        ), row=3, col=2)
        
        if len(anomaly_points) > 0:
            fig.add_trace(go.Scatter(
                x=anomaly_points['timestep'], 
                y=anomaly_points['detected_anomaly'].astype(int),
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x')
            ), row=3, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="Real-time Flight Monitoring Dashboard")
        fig.update_xaxes(title_text="Time Step")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Anomaly Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly distribution pie chart
            normal_count = len(flight_data) - anomaly_count
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Normal', 'Anomaly'],
                values=[normal_count, anomaly_count],
                hole=0.4,
                marker_colors=['green', 'red']
            )])
            fig_pie.update_layout(title="Anomaly Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Parameter-wise anomaly count
            param_anomalies = {
                'Altitude': flight_data['altitude_anomaly'].sum(),
                'Velocity': flight_data['velocity_anomaly'].sum(),
                'Yaw': flight_data['yaw_anomaly'].sum(),
                'Pitch': flight_data['pitch_anomaly'].sum(),
                'Battery': flight_data['battery_anomaly'].sum(),
                'GPS Drift': flight_data['gps_anomaly'].sum()
            }
            
            fig_bar = go.Figure(data=[go.Bar(
                x=list(param_anomalies.keys()),
                y=list(param_anomalies.values()),
                marker_color=['red' if v > 0 else 'green' for v in param_anomalies.values()]
            )])
            fig_bar.update_layout(title="Anomalies by Parameter", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Anomaly timeline
        st.subheader("Anomaly Timeline")
        anomaly_timeline = flight_data[flight_data['detected_anomaly']].copy()
        
        if len(anomaly_timeline) > 0:
            fig_timeline = go.Figure()
            
            for param in ['altitude', 'velocity', 'yaw', 'pitch', 'battery', 'gps_drift']:
                anomaly_param = flight_data[flight_data[f'{param}_anomaly']]
                if len(anomaly_param) > 0:
                    fig_timeline.add_trace(go.Scatter(
                        x=anomaly_param['timestep'],
                        y=[param] * len(anomaly_param),
                        mode='markers',
                        name=f'{param.title()} Anomaly',
                        marker=dict(size=12, symbol='x')
                    ))
            
            fig_timeline.update_layout(
                title="Anomaly Timeline by Parameter",
                xaxis_title="Time Step",
                yaxis_title="Parameter",
                height=400
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No anomalies detected in current flight data.")
    
    with tab3:
        st.subheader("Detailed Parameter Analysis")
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        stats_df = flight_data[['altitude', 'velocity', 'yaw', 'pitch', 'battery', 'gps_drift']].describe()
        st.dataframe(stats_df, use_container_width=True)
        
        # Parameter correlation heatmap
        st.subheader("🔗 Parameter Correlations")
        corr_matrix = flight_data[['altitude', 'velocity', 'yaw', 'pitch', 'battery', 'gps_drift']].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Parameter Correlation Matrix"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Distribution plots
        st.subheader("📈 Parameter Distributions")
        selected_param = st.selectbox("Select Parameter for Distribution", 
                                    ['altitude', 'velocity', 'yaw', 'pitch', 'battery', 'gps_drift'])
        
        fig_hist = px.histogram(
            flight_data, 
            x=selected_param, 
            nbins=30,
            title=f"Distribution of {selected_param.title()}",
            color_discrete_sequence=['skyblue']
        )
        fig_hist.add_vline(
            x=flight_data[selected_param].mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text="Mean"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab4:
        st.subheader("Flight Path Visualization")
        
        # 3D flight path (using altitude as Z-axis and timestep as trajectory)
        fig_3d = go.Figure()
        
        # Normal points
        normal_points = flight_data[~flight_data['detected_anomaly']]
        anomaly_points = flight_data[flight_data['detected_anomaly']]
        
        if len(normal_points) > 0:
            fig_3d.add_trace(go.Scatter3d(
                x=normal_points['timestep'],
                y=normal_points['gps_drift'],
                z=normal_points['altitude'],
                mode='markers+lines',
                name='Normal Flight Path',
                marker=dict(size=4, color='blue', opacity=0.7),
                line=dict(color='blue', width=2)
            ))
        
        if len(anomaly_points) > 0:
            fig_3d.add_trace(go.Scatter3d(
                x=anomaly_points['timestep'],
                y=anomaly_points['gps_drift'],
                z=anomaly_points['altitude'],
                mode='markers',
                name='Anomaly Points',
                marker=dict(size=8, color='red', symbol='x')
            ))
        
        fig_3d.update_layout(
            title="3D Flight Path (Time vs GPS Drift vs Altitude)",
            scene=dict(
                xaxis_title="Time Step",
                yaxis_title="GPS Drift",
                zaxis_title="Altitude (m)"
            ),
            height=600
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Flight trajectory over time
        col1, col2 = st.columns(2)
        
        with col1:
            fig_alt_time = px.line(
                flight_data, 
                x='timestep', 
                y='altitude',
                title="Altitude vs Time",
                color_discrete_sequence=['blue']
            )
            if len(anomaly_points) > 0:
                fig_alt_time.add_scatter(
                    x=anomaly_points['timestep'],
                    y=anomaly_points['altitude'],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='Anomalies'
                )
            st.plotly_chart(fig_alt_time, use_container_width=True)
        
        with col2:
            fig_vel_time = px.line(
                flight_data, 
                x='timestep', 
                y='velocity',
                title="Velocity vs Time",
                color_discrete_sequence=['green']
            )
            if len(anomaly_points) > 0:
                fig_vel_time.add_scatter(
                    x=anomaly_points['timestep'],
                    y=anomaly_points['velocity'],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='x'),
                    name='Anomalies'
                )
            st.plotly_chart(fig_vel_time, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🚁 Drone Flight Anomaly Detection Dashboard | "
        f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "
        "Auto-refreshing every 2 seconds"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()