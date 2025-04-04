# Sentiment Analysis Dashboard

This directory contains the Streamlit dashboard for visualizing sentiment analysis of fire brigade incident reports.

## Dashboard Features

The dashboard provides the following visualizations and tools:

1. **Summary Metrics**:
   - Count and percentage of positive, neutral, and negative sentiment incidents
   - Visual indicators of sentiment distributions

2. **Sentiment Distribution**:
   - Pie chart visualization of incident sentiment distribution
   - Color-coded segments for easy interpretation

3. **Sentiment Trends**:
   - Time series chart showing sentiment trends over time
   - Adjustable time range for different analysis periods
   - Interactive elements for exploring specific dates

4. **Recent Incidents**:
   - List of recent incident reports with sentiment analysis
   - Color-coded for quick recognition of sentiment
   - Timestamp and full report text

5. **Sentiment Analysis Tool**:
   - Interactive tool for analyzing custom text
   - Real-time sentiment prediction
   - Confidence score visualization

## Running the Dashboard

To run the dashboard locally:

```bash
streamlit run app/dashboard/dashboard.py
```

The dashboard will be available at http://localhost:8501 by default.

## Auto-Refresh

The dashboard includes an auto-refresh feature that periodically updates the visualizations with the latest data. This can be toggled on/off in the sidebar, and the refresh interval can be adjusted.

## Data Sources

The dashboard pulls data from:

1. **Database**: For historical incident reports and sentiment analysis
2. **Model API**: For real-time sentiment prediction of custom text

## Customization

The dashboard appearance and settings can be customized by modifying:

- `app/utils/config.py`: Dashboard title and default refresh interval
- `app/dashboard/dashboard.py`: Layout, charts, and visualization settings 