"""
Sentiment Analysis Model API

This module provides a Flask API for serving sentiment analysis predictions.
Supports multiple model types: synthetic, twitter, and hybrid.
"""
import os
import sys
import logging
from flask import Flask, request, jsonify
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import model and utilities
from app.models.predict import predict_sentiment, batch_predict, get_available_models
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
        JSON: Status of the API and available models
    """
    available_models = get_available_models()
    
    return jsonify({
        "status": "healthy",
        "available_models": available_models
    })

@app.route('/models', methods=['GET'])
def list_models():
    """
    List available models endpoint.
    
    Returns:
        JSON: List of available sentiment models
    """
    available_models = get_available_models()
    
    return jsonify({
        "models": available_models,
        "count": len(available_models),
        "default": "synthetic" if "synthetic" in available_models else available_models[0] if available_models else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for predicting sentiment of a single text.
    
    Request Body:
        JSON with 'text' field and optional 'model_type' field
        
    Returns:
        JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        
        # Get model type if specified, default to synthetic if available
        available_models = get_available_models()
        if not available_models:
            return jsonify({"error": "No models available"}), 500
            
        if 'model_type' in data and data['model_type'] in available_models:
            model_type = data['model_type']
        else:
            model_type = "synthetic" if "synthetic" in available_models else available_models[0]
            
        logger.info(f"Using model type: {model_type}")
        
        # Make prediction
        sentiment_value, sentiment_label, confidence, model_used = predict_sentiment(text, model_type)
        
        # Save to database if incident_id is provided
        if 'incident_id' in data:
            save_sentiment(data['incident_id'], sentiment_value, confidence)
        
        response = {
            "sentiment": sentiment_label,
            "sentiment_value": int(sentiment_value),
            "confidence": float(confidence),
            "text": text,
            "model_type": model_type
        }
        
        # Add model_used for hybrid model
        if model_type == "hybrid" and model_used:
            response["model_used"] = model_used
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict_route():
    """
    Endpoint for predicting sentiment of multiple texts.
    
    Request Body:
        JSON with 'texts' field (list of strings) and optional 'model_type' field
        
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
        
        # Get model type if specified, default to synthetic if available
        available_models = get_available_models()
        if not available_models:
            return jsonify({"error": "No models available"}), 500
            
        if 'model_type' in data and data['model_type'] in available_models:
            model_type = data['model_type']
        else:
            model_type = "synthetic" if "synthetic" in available_models else available_models[0]
            
        logger.info(f"Using model type for batch prediction: {model_type}")
        
        # Make predictions
        predictions = batch_predict(texts, model_type)
        
        # Save to database if incident_ids are provided
        if 'incident_ids' in data and isinstance(data['incident_ids'], list):
            incident_ids = data['incident_ids']
            if len(incident_ids) == len(predictions):
                for incident_id, (value, _, confidence, _) in zip(incident_ids, predictions):
                    save_sentiment(incident_id, value, confidence)
        
        # Format results
        results = []
        for (value, label, conf, model_used), text in zip(predictions, texts):
            result = {
                "sentiment": label,
                "sentiment_value": int(value),
                "confidence": float(conf),
                "text": text
            }
            
            # Add model_used for hybrid model
            if model_type == "hybrid" and model_used:
                result["model_used"] = model_used
                
            results.append(result)
        
        return jsonify({
            "predictions": results,
            "count": len(results),
            "model_type": model_type
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/predict_sample', methods=['GET'])
def predict_sample():
    """
    Endpoint for testing prediction with a sample text.
    
    Query Parameters:
        model_type (optional): The model to use for prediction
    
    Returns:
        JSON with prediction results
    """
    sample_text = "Successfully rescued family from burning building with no injuries."
    
    # Get model type from query parameter if provided
    model_type = request.args.get('model_type', None)
    
    # Check if model type is valid
    available_models = get_available_models()
    if not available_models:
        return jsonify({"error": "No models available"}), 500
        
    if model_type not in available_models:
        model_type = "synthetic" if "synthetic" in available_models else available_models[0]
    
    # Make prediction
    sentiment_value, sentiment_label, confidence, model_used = predict_sentiment(sample_text, model_type)
    
    response = {
        "sentiment": sentiment_label,
        "sentiment_value": int(sentiment_value),
        "confidence": float(confidence),
        "text": sample_text,
        "model_type": model_type
    }
    
    # Add model_used for hybrid model
    if model_type == "hybrid" and model_used:
        response["model_used"] = model_used
    
    return jsonify(response)

@app.route('/compare', methods=['POST'])
def compare_models():
    """
    Endpoint for comparing predictions from multiple models on the same text.
    
    Request Body:
        JSON with 'text' field
        
    Returns:
        JSON with prediction results from all available models
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        available_models = get_available_models()
        
        if not available_models:
            return jsonify({"error": "No models available"}), 500
        
        # Get predictions from all models
        results = {}
        for model_type in available_models:
            sentiment_value, sentiment_label, confidence, model_used = predict_sentiment(text, model_type)
            
            result = {
                "sentiment": sentiment_label,
                "sentiment_value": int(sentiment_value),
                "confidence": float(confidence)
            }
            
            # Add model_used for hybrid model
            if model_type == "hybrid" and model_used:
                result["model_used"] = model_used
                
            results[model_type] = result
        
        return jsonify({
            "text": text,
            "models": results,
            "available_models": available_models
        })
        
    except Exception as e:
        logger.error(f"Error in compare models endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('MODEL_API_PORT', MODEL_API_PORT))
    logger.info(f"Starting model API on port {port}")
    
    # Log available models
    available_models = get_available_models()
    logger.info(f"Available models: {', '.join(available_models) if available_models else 'None'}")
    
    app.run(host='0.0.0.0', port=port) 