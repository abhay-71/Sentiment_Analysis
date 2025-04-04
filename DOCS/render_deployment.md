# Deploying Sentiment Analysis Application to Render

This guide explains how to deploy the Sentiment Analysis application to Render, a cloud platform for web applications.

## Prerequisites

- A Render account (sign up at [render.com](https://render.com))
- Git repository with your Sentiment Analysis application code

## Deployment Overview

The application consists of three components that need to be deployed separately:

1. **Mock API**: Provides simulated incident data
2. **Model API**: Serves sentiment predictions
3. **Dashboard**: Visualizes the sentiment analysis

## Step 1: Prepare Your Repository

Ensure your repository includes these required files:

- `requirements.txt`: Lists all dependencies
- `render.yaml`: Defines the Render Blueprint (see below)

Create a `render.yaml` file in your root directory:

```yaml
services:
  # Mock API Service
  - name: sentiment-mock-api
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python app/api/mock_api.py
    envVars:
      - key: API_PORT
        value: 10000
      - key: PYTHON_VERSION
        value: 3.9.0

  # Model API Service
  - name: sentiment-model-api
    type: web
    env: python
    buildCommand: pip install -r requirements.txt && python app/models/train_model.py
    startCommand: python app/api/model_api.py
    envVars:
      - key: API_PORT
        value: 10001
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: API_HOST
        fromService:
          name: sentiment-mock-api
          type: web
          property: host

  # Dashboard Service
  - name: sentiment-dashboard
    type: web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/dashboard/dashboard.py
    envVars:
      - key: API_HOST
        fromService:
          name: sentiment-mock-api
          type: web
          property: host
      - key: MODEL_API_HOST
        fromService:
          name: sentiment-model-api
          type: web
          property: host
      - key: PYTHON_VERSION
        value: 3.9.0
```

## Step 2: Deploy Using Render Blueprint

1. Log in to your Render account
2. Go to the Dashboard and click "New"
3. Select "Blueprint"
4. Connect your Git repository
5. Render will detect the `render.yaml` file and set up the services
6. Click "Apply" to create and deploy all services

## Step 3: Configure Environment Variables

After deployment, you may need to update these environment variables in each service:

- `API_HOST`: URL of the Mock API service
- `API_PORT`: Port for each service (default: 10000, 10001, 8501)
- `MODEL_API_URL`: Full URL of the Model API service

## Step 4: Initialize the Database

The database needs to be initialized with data. There are two options:

### Option 1: SSH into the Model API service

1. In your Render dashboard, go to the Model API service
2. Click "Shell" to open a terminal
3. Run: `python app/data/data_ingestion.py --count 50`

### Option 2: Add to the build command

Modify the Model API build command in `render.yaml`:

```yaml
buildCommand: pip install -r requirements.txt && python app/models/train_model.py && python app/data/data_ingestion.py --count 50
```

## Step 5: Verify Deployment

1. Check that all three services are running (green status in Render dashboard)
2. Test the Mock API by visiting its URL with `/health` appended
3. Test the Model API by visiting its URL with `/health` appended
4. Visit the Dashboard URL to see if it's correctly displaying data

## Troubleshooting

- **Service Crashes**: Check the logs in the Render dashboard
- **Database Issues**: Run data ingestion again manually
- **Connection Errors**: Verify environment variables are correctly set
- **Dashboard Not Showing Data**: Check if the APIs are accessible from the dashboard

## Maintenance

- Render automatically redeploys services when you push changes to your repository
- You can manually restart services from the Render dashboard
- Set up Render's built-in monitoring for each service 