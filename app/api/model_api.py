"""
Sentiment Analysis Model API

This module provides a Flask API for serving sentiment analysis predictions.
"""
import os
import sys
import logging
from flask import Flask, request, jsonify
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import model and utilities
from app.models.predict import predict_sentiment, batch_predict, MODEL_LOADED
from app.data.database import save_sentiment
from app.utils.config import MODEL_API_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_api')

# Create Flask application
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON: Status of the API
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for predicting sentiment of a single text.
    
    Request Body:
        JSON with 'text' field
        
    Returns:
        JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        
        # Make prediction
        sentiment_value, sentiment_label, confidence = predict_sentiment(text)
        
        # Save to database if incident_id is provided
        if 'incident_id' in data:
            save_sentiment(data['incident_id'], sentiment_value, confidence)
        
        return jsonify({
            "sentiment": sentiment_label,
            "sentiment_value": int(sentiment_value),
            "confidence": float(confidence),
            "text": text
        })
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict_route():
    """
    Endpoint for predicting sentiment of multiple texts.
    
    Request Body:
        JSON with 'texts' field (list of strings)
        
    Returns:
        JSON with list of prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' field in request"}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({"error": "'texts' field should be a list"}), 400
        
        # Make predictions
        predictions = batch_predict(texts)
        
        # Save to database if incident_ids are provided
        if 'incident_ids' in data and isinstance(data['incident_ids'], list):
            incident_ids = data['incident_ids']
            if len(incident_ids) == len(predictions):
                for incident_id, (value, _, confidence) in zip(incident_ids, predictions):
                    save_sentiment(incident_id, value, confidence)
        
        # Format results
        results = [
            {
                "sentiment": label,
                "sentiment_value": int(value),
                "confidence": float(conf),
                "text": text
            }
            for (value, label, conf), text in zip(predictions, texts)
        ]
        
        return jsonify({
            "predictions": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict_sample', methods=['GET'])
def predict_sample():
    """
    Endpoint for testing prediction with a sample text.
    
    Returns:
        JSON with prediction results
    """
    sample_text = "Successfully rescued family from burning building with no injuries."
    
    # Make prediction
    sentiment_value, sentiment_label, confidence = predict_sentiment(sample_text)
    
    return jsonify({
        "sentiment": sentiment_label,
        "sentiment_value": int(sentiment_value),
        "confidence": float(confidence),
        "text": sample_text
    })

if __name__ == '__main__':
    port = int(os.environ.get('MODEL_API_PORT', MODEL_API_PORT))
    logger.info(f"Starting model API on port {port}")
    logger.info(f"Model loaded: {MODEL_LOADED}")
    app.run(host='0.0.0.0', port=port) 