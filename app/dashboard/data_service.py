"""
Data Service for Dashboard

This module provides functions to fetch and process data for the dashboard.
"""
import os
import sys
import logging
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.data.database import get_incidents, get_sentiment_stats
from app.utils.config import MOCK_API_URL, MODEL_API_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_service')

def fetch_raw_incidents(count=20):
    """
    Fetch raw incidents from the API.
    
    Args:
        count (int): Number of incidents to fetch
        
    Returns:
        list: List of incident dictionaries
    """
    try:
        response = requests.get(f"{MOCK_API_URL}?count={count}")
        response.raise_for_status()
        incidents = response.json()
        logger.info(f"Fetched {len(incidents)} raw incidents from API")
        return incidents
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching incidents from API: {str(e)}")
        return []

def get_incidents_from_db(limit=100):
    """
    Get incidents from the database.
    
    Args:
        limit (int): Maximum number of incidents to retrieve
        
    Returns:
        pandas.DataFrame: DataFrame with incident data
    """
    try:
        incidents = get_incidents(limit=limit)
        df = pd.DataFrame(incidents)
        
        if not df.empty:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            # Map sentiment to string labels
            sentiment_map = {1: 'positive', 0: 'neutral', -1: 'negative', None: 'not analyzed'}
            df['sentiment_label'] = df['sentiment'].map(lambda x: sentiment_map.get(x, 'not analyzed'))
            
            logger.info(f"Retrieved {len(df)} incidents from database")
        else:
            logger.warning("No incidents found in database")
            
        return df
    
    except Exception as e:
        logger.error(f"Error getting incidents from database: {str(e)}")
        return pd.DataFrame()

def get_sentiment_statistics():
    """
    Get sentiment statistics from the database.
    
    Returns:
        dict: Sentiment statistics
    """
    try:
        stats = get_sentiment_stats()
        logger.info(f"Retrieved sentiment statistics: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error retrieving sentiment statistics: {str(e)}")
        return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}

def predict_sentiment(text):
    """
    Get sentiment prediction for a text using the model API.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Prediction result
    """
    try:
        response = requests.post(
            f"{MODEL_API_URL}",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received prediction for text: {result}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting prediction: {str(e)}")
        return {
            "sentiment": "neutral",
            "sentiment_value": 0,
            "confidence": 0.0,
            "text": text
        }

def get_sentiment_over_time(days=30):
    """
    Get sentiment trends over time.
    
    Args:
        days (int): Number of days to include
        
    Returns:
        pandas.DataFrame: DataFrame with daily sentiment counts
    """
    try:
        # Get incidents from database
        df = get_incidents_from_db(limit=1000)
        
        if df.empty:
            return pd.DataFrame()
        
        # Filter for date range
        start_date = datetime.now().date() - timedelta(days=days)
        df = df[df['date'] >= start_date]
        
        # Group by date and sentiment
        sentiment_over_time = df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
        
        # Pivot table for easier plotting
        pivot_table = sentiment_over_time.pivot(
            index='date', 
            columns='sentiment_label', 
            values='count'
        ).fillna(0)
        
        # Ensure all sentiment labels exist
        for label in ['positive', 'neutral', 'negative', 'not analyzed']:
            if label not in pivot_table.columns:
                pivot_table[label] = 0
        
        pivot_table = pivot_table.reset_index()
        logger.info(f"Generated sentiment trends over {days} days")
        return pivot_table
    
    except Exception as e:
        logger.error(f"Error generating sentiment trends: {str(e)}")
        return pd.DataFrame()

def get_recent_incidents(limit=10):
    """
    Get recent incidents with sentiment.
    
    Args:
        limit (int): Maximum number of incidents to retrieve
        
    Returns:
        pandas.DataFrame: DataFrame with recent incidents
    """
    try:
        df = get_incidents_from_db(limit=limit)
        
        if df.empty:
            return pd.DataFrame()
        
        # Sort by timestamp (most recent first)
        df = df.sort_values('timestamp', ascending=False).head(limit)
        
        # Format timestamp for display
        df['formatted_time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting recent incidents: {str(e)}")
        return pd.DataFrame() 