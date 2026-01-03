"""
Drone Flight Anomaly Detection Dashboard Launcher

This script helps you launch the appropriate dashboard for your needs:
1. Basic Dashboard - Simple real-time monitoring with basic anomaly detection
2. Advanced Dashboard - Comprehensive analysis with ML-based anomaly detection
"""

import streamlit as st
import subprocess
import sys
import os

def main():
    st.set_page_config(
        page_title="Dashboard Launcher",
        page_icon="🚁",
        layout="centered"
    )
    
    st.title("🚁 Drone Anomaly Detection Dashboard")
    st.markdown("### Choose your dashboard experience")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 Basic Dashboard
        - Real-time parameter monitoring
        - Simple statistical anomaly detection
        - Interactive visualizations
        - Auto-refreshing display
        - Perfect for live monitoring
        """)
        
        if st.button("Launch Basic Dashboard", type="primary"):
            st.info("Launching Basic Dashboard...")
            st.markdown("Run this command in your terminal:")
            st.code("streamlit run streamlit_dashboard.py", language="bash")
    
    with col2:
        st.markdown("""
        #### 🤖 Advanced Dashboard
        - Machine Learning anomaly detection
        - Multiple detection algorithms
        - Performance metrics
        - Detailed analysis tools
        - Model diagnostics
        """)
        
        if st.button("Launch Advanced Dashboard", type="primary"):
            st.info("Launching Advanced Dashboard...")
            st.markdown("Run this command in your terminal:")
            st.code("streamlit run advanced_dashboard.py", language="bash")
    
    st.markdown("---")
    st.markdown("### 🛠️ Setup Instructions")
    
    with st.expander("First Time Setup"):
        st.markdown("""
        1. **Install Dependencies:**
           ```bash
           pip install -r requirements.txt
           ```
        
        2. **Generate Sample Data (if needed):**
           ```bash
           python Data.py
           ```
        
        3. **Train Model (optional for advanced dashboard):**
           ```bash
           python "Model Train.py"
           ```
        
        4. **Launch Dashboard:**
           - Basic: `streamlit run streamlit_dashboard.py`
           - Advanced: `streamlit run advanced_dashboard.py`
        """)
    
    with st.expander("Features Comparison"):
        comparison_data = {
            "Feature": [
                "Real-time Monitoring",
                "Parameter Visualization",
                "Anomaly Detection",
                "Auto-refresh",
                "Interactive Charts", 
                "Machine Learning",
                "Performance Metrics",
                "Data Export",
                "Multiple Algorithms",
                "Model Diagnostics"
            ],
            "Basic Dashboard": [
                "✅", "✅", "✅ (Statistical)", "✅", "✅",
                "❌", "❌", "❌", "❌", "❌"
            ],
            "Advanced Dashboard": [
                "✅", "✅", "✅ (ML + Statistical)", "❌", "✅", 
                "✅", "✅", "✅", "✅", "✅"
            ]
        }
        
        import pandas as pd
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    st.markdown("---")
    st.markdown("**Note:** Make sure you have generated the drone_telemetry.csv file before running either dashboard.")

if __name__ == "__main__":
    main()