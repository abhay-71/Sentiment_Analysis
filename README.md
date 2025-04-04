# Sentiment Analysis Application

A cost-effective sentiment analysis application for a fire brigade company to analyze incident reports and visualize sentiment trends.

## Project Overview

This application:
- Ingests incident reports from a (mock) API
- Analyzes sentiment (positive, neutral, negative)
- Visualizes sentiment data in an interactive dashboard
- Uses free-tier services for cost-effectiveness

## Features

- **Mock API**: Simulates a fire brigade incident reporting API
- **Sentiment Analysis Model**: ML model to classify text sentiment
- **Interactive Dashboard**: Visualization of sentiment trends and statistics
- **Database**: SQLite storage for incidents and analysis results
- **Deployment**: Ready for Render.com deployment

## Components

1. **Mock API**: Provides simulated incident data
2. **Data Ingestion**: Collects and stores incident reports
3. **Sentiment Model**: Analyzes text sentiment using NLP
4. **Model API**: Serves sentiment predictions
5. **Dashboard**: Visualizes trends and statistics

## Setup Instructions

### Quick Start

The easiest way to get started is to run the setup script and then the run script:

```bash
# Initialize the application (database, model, initial data)
python setup.py

# Run all components (API, model, dashboard)
python run.py
```

### Manual Setup

For more control, you can set up and run each component separately:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Sentiment_Analysis.git
   cd Sentiment_Analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Setup environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration.

5. Initialize the application:
   ```bash
   python setup.py
   ```

6. Start each component manually:
   ```bash
   # In separate terminals:
   python app/api/mock_api.py
   python app/api/model_api.py
   streamlit run app/dashboard/dashboard.py
   ```

## Documentation

Additional documentation is available in the `DOCS` directory:
- [API Documentation](app/api/README.md)
- [Model Documentation](app/models/README.md)
- [Dashboard Documentation](app/dashboard/README.md)
- [Deployment Guide](DOCS/render_deployment.md)

## Deployment

To deploy to Render.com, follow the instructions in the [Deployment Guide](DOCS/render_deployment.md).

## License

MIT

## Contributors

- Your Name <your.email@example.com>