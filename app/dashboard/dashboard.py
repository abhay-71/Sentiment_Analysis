"""
Sentiment Analysis Dashboard

This module implements a Streamlit dashboard for visualizing
sentiment analysis of fire brigade incident reports.
Supports multiple sentiment models.
"""
import os
import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.dashboard.data_service import (
    get_incidents_from_db, get_sentiment_statistics,
    predict_sentiment, get_sentiment_over_time, get_recent_incidents,
    get_available_models, compare_models
)
from app.utils.config import DASHBOARD_TITLE, REFRESH_INTERVAL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard')

# Set page configuration
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for model selection
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "synthetic"

if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False

def main():
    """Main function to run the Streamlit dashboard."""
    # Page title
    st.title(DASHBOARD_TITLE)
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Add auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh dashboard", value=True)
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)", 
        min_value=10, 
        max_value=300, 
        value=REFRESH_INTERVAL
    )
    
    if auto_refresh:
        st.sidebar.info(f"Dashboard will refresh every {refresh_interval} seconds")
    
    # Time range selector
    time_range = st.sidebar.slider(
        "Time range (days)",
        min_value=1,
        max_value=90,
        value=30
    )
    
    # Display last update time
    st.sidebar.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Model selection
    st.sidebar.header("Model Selection")
    
    # Get available models
    available_models = get_available_models()
    
    # Dictionary to display friendly names
    model_display_names = {
        "synthetic": "Synthetic (Fire Brigade Domain)",
        "twitter": "Twitter (General Social Media)",
        "hybrid": "Hybrid (Combined Approach)",
        "default": "Default Model"
    }
    
    # Create dropdown options
    model_options = {model_display_names.get(model, model): model for model in available_models}
    
    selected_display_name = st.sidebar.selectbox(
        "Select Sentiment Model",
        options=list(model_options.keys()),
        index=0
    )
    
    # Update the selected model in session state
    st.session_state.selected_model = model_options[selected_display_name]
    
    # Add model descriptions
    if st.session_state.selected_model == "synthetic":
        st.sidebar.info("The synthetic model is trained specifically on fire brigade domain text and performs best on domain-specific content (88.9% accuracy).")
    elif st.session_state.selected_model == "twitter":
        st.sidebar.info("The Twitter model is trained on real-world social media data and may handle colloquial expressions better, but has lower accuracy on domain text (33.3%).")
    elif st.session_state.selected_model == "hybrid":
        st.sidebar.info("The hybrid model combines both approaches, weighing predictions based on domain specificity and confidence scores (77.8% accuracy on domain text).")
    
    # Sentiment prediction tool
    st.sidebar.header("Sentiment Prediction Tool")
    custom_text = st.sidebar.text_area("Enter text to analyze:")
    
    # Add option to compare models
    st.session_state.show_comparison = st.sidebar.checkbox("Compare all models", value=st.session_state.show_comparison)
    
    if st.sidebar.button("Analyze Sentiment"):
        if custom_text:
            with st.sidebar:
                with st.spinner("Predicting sentiment..."):
                    if st.session_state.show_comparison:
                        display_model_comparison(custom_text)
                    else:
                        result = predict_sentiment(custom_text, model_type=st.session_state.selected_model)
                        display_sentiment_result(result)
        else:
            st.sidebar.warning("Please enter text to analyze")
    
    # Main content
    # Create a 2x2 grid with key metrics and charts
    col1, col2 = st.columns(2)
    
    # First row: summary metrics
    with col1:
        display_summary_metrics()
    
    with col2:
        display_sentiment_distribution()
    
    # Second row: time series and recent incidents
    col3, col4 = st.columns(2)
    
    with col3:
        display_sentiment_over_time(days=time_range)
    
    with col4:
        display_recent_incidents()
    
    # Auto-refresh
    if auto_refresh:
        st.empty()
        # Add JavaScript for auto-refresh
        st.markdown(
            f"""
            <script>
                setTimeout(function(){{
                    window.location.reload();
                }}, {refresh_interval * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )

def display_sentiment_result(result):
    """Display sentiment prediction result."""
    # Extract values from result
    sentiment = result.get("sentiment", "neutral")
    confidence = result.get("confidence", 0.0) * 100
    model_type = result.get("model_type", "unknown")
    model_used = result.get("model_used", None)
    
    # Select color based on sentiment
    if sentiment == "positive":
        color = "green"
    elif sentiment == "negative":
        color = "red"
    else:
        color = "gray"
    
    st.markdown(f"**Sentiment:** <span style='color:{color}'>{sentiment.upper()}</span>", unsafe_allow_html=True)
    st.progress(confidence / 100)
    st.text(f"Confidence: {confidence:.1f}%")
    
    # Display model information
    if model_type:
        st.text(f"Model: {model_type}")
    
    # For hybrid model, show which sub-model was used
    if model_used:
        st.text(f"Decision from: {model_used}")

def display_model_comparison(text):
    """Display comparison of sentiment predictions from all models."""
    result = compare_models(text)
    
    if "error" in result:
        st.error(f"Error comparing models: {result['error']}")
        return
    
    # Display original text
    st.markdown(f"**Text:** {result.get('text', '')}")
    
    # Create columns for each model
    models_data = result.get("models", {})
    
    if not models_data:
        st.warning("No model results available for comparison.")
        return
    
    st.markdown("### Model Comparison")
    
    # Create a table for comparison
    comparison_data = []
    
    for model_name, prediction in models_data.items():
        sentiment = prediction.get("sentiment", "neutral")
        confidence = prediction.get("confidence", 0.0) * 100
        model_used = prediction.get("model_used", "N/A")
        
        comparison_data.append({
            "Model": model_name,
            "Sentiment": sentiment.upper(),
            "Confidence": f"{confidence:.1f}%",
            "Decision Model": model_used if model_used != "N/A" else ""
        })
    
    # Convert to DataFrame and display
    comparison_df = pd.DataFrame(comparison_data)
    
    # Use st.dataframe for more styling options
    st.dataframe(
        comparison_df.style.applymap(
            lambda x: "background-color: #E8F0FE" if isinstance(x, str) and x.lower() == "positive" else
                     ("background-color: #FFEBEE" if isinstance(x, str) and x.lower() == "negative" else ""),
            subset=["Sentiment"]
        ),
        use_container_width=True
    )

def display_summary_metrics():
    """Display summary metrics in a card."""
    st.subheader("Summary Metrics")
    
    # Get sentiment statistics
    stats = get_sentiment_statistics()
    
    # Create three columns for the metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        positive_pct = (stats.get('positive', 0) / max(stats.get('total', 1), 1)) * 100
        st.metric(
            label="Positive Sentiment",
            value=f"{stats.get('positive', 0)}",
            delta=f"{positive_pct:.1f}%"
        )
    
    with metric_col2:
        neutral_pct = (stats.get('neutral', 0) / max(stats.get('total', 1), 1)) * 100
        st.metric(
            label="Neutral Sentiment",
            value=f"{stats.get('neutral', 0)}",
            delta=f"{neutral_pct:.1f}%"
        )
    
    with metric_col3:
        negative_pct = (stats.get('negative', 0) / max(stats.get('total', 1), 1)) * 100
        st.metric(
            label="Negative Sentiment",
            value=f"{stats.get('negative', 0)}",
            delta=f"{negative_pct:.1f}%",
            delta_color="inverse"  # Inverse color (red is bad for negative)
        )

def display_sentiment_distribution():
    """Display sentiment distribution chart."""
    st.subheader("Sentiment Distribution")
    
    # Get sentiment statistics
    stats = get_sentiment_statistics()
    
    # Create data for pie chart
    labels = ['Positive', 'Neutral', 'Negative']
    values = [stats.get('positive', 0), stats.get('neutral', 0), stats.get('negative', 0)]
    colors = ['#4CAF50', '#9E9E9E', '#F44336']  # Green, Gray, Red
    
    # Only create chart if we have data
    if sum(values) > 0:
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            values, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0.05, 0.05),
            wedgeprops={'width': 0.6, 'edgecolor': 'w'}
        )
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        st.pyplot(fig)
    else:
        st.info("No sentiment data available yet. Run the data ingestion script to collect data.")

def display_sentiment_over_time(days=30):
    """Display sentiment trends over time."""
    st.subheader(f"Sentiment Trends (Last {days} Days)")
    
    # Get sentiment trends data
    df = get_sentiment_over_time(days=days)
    
    if df.empty:
        st.info("No sentiment trend data available for the selected time range.")
        return
    
    # Melt the DataFrame for easier charting
    df_melted = df.melt(
        id_vars=['date'],
        value_vars=['positive', 'neutral', 'negative'],
        var_name='sentiment',
        value_name='count'
    )
    
    # Create Altair chart
    chart = alt.Chart(df_melted).mark_area().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('count:Q', title='Number of Incidents', stack='normalize'),
        color=alt.Color(
            'sentiment:N', 
            scale=alt.Scale(
                domain=['positive', 'neutral', 'negative'],
                range=['#4CAF50', '#9E9E9E', '#F44336']
            ),
            legend=alt.Legend(title='Sentiment')
        ),
        tooltip=['date:T', 'sentiment:N', 'count:Q']
    ).properties(
        height=300
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)

def display_recent_incidents():
    """Display a table of recent incidents with sentiment."""
    st.subheader("Recent Incidents")
    
    # Get recent incidents
    df = get_recent_incidents(limit=10)
    
    if df.empty:
        st.info("No incident data available. Run the data ingestion script to collect data.")
        return
    
    # Display incidents in a styled table
    # Create a color map for sentiment labels
    sentiment_colors = {
        'positive': '#E3F2E3',  # Light green
        'neutral': '#F5F5F5',   # Light gray
        'negative': '#FADBD8',  # Light red
        'not analyzed': '#FFFFFF'  # White
    }
    
    # Create a DataFrame for display
    display_df = df[['incident_id', 'report', 'formatted_time', 'sentiment_label']]
    display_df = display_df.rename(columns={
        'incident_id': 'ID',
        'report': 'Description',
        'formatted_time': 'Time',
        'sentiment_label': 'Sentiment'
    })
    
    # Style and display the table
    st.dataframe(
        display_df.style.applymap(
            lambda x: f"background-color: {sentiment_colors.get(x.lower(), '#FFFFFF')}" 
            if isinstance(x, str) and x.lower() in sentiment_colors 
            else "",
            subset=['Sentiment']
        ),
        use_container_width=True
    )

if __name__ == "__main__":
    main() 