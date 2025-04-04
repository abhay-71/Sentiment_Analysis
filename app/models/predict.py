"""
Prediction Module for Sentiment Analysis

This module contains functions for loading the trained model
and making sentiment predictions on new incident reports.
"""
import os
import sys
import logging
import joblib
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.utils.text_preprocessing import preprocess_text, map_sentiment_value
from app.utils.config import MODEL_PATH, VECTORIZER_PATH, SENTIMENT_LABELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('predict')

# Load vectorizer and model
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    logger.info("Model and vectorizer loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.warning("Model not loaded. Run train_model.py first.")
    MODEL_LOADED = False

def predict_sentiment(text):
    """
    Predict sentiment for a given text.
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (sentiment_value, sentiment_label, confidence)
    """
    if not MODEL_LOADED:
        logger.error("Model not loaded. Cannot make predictions.")
        return 0, "neutral", 0.0
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Make prediction
        sentiment_value = model.predict(text_vectorized)[0]
        
        # Get confidence score (distance from decision boundary)
        confidence = np.abs(model.decision_function(text_vectorized)[0])
        if isinstance(confidence, np.ndarray):
            confidence = np.mean(confidence)
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, confidence / 2.0)
        
        # Map sentiment value to label
        sentiment_label = map_sentiment_value(sentiment_value)
        
        logger.info(f"Predicted sentiment: {sentiment_label} (confidence: {confidence:.2f})")
        return sentiment_value, sentiment_label, confidence
    
    except Exception as e:
        logger.error(f"Error predicting sentiment: {str(e)}")
        return 0, "neutral", 0.0

def batch_predict(texts):
    """
    Predict sentiment for a batch of texts.
    
    Args:
        texts (list): List of text strings
        
    Returns:
        list: List of prediction tuples (sentiment_value, sentiment_label, confidence)
    """
    if not MODEL_LOADED:
        logger.error("Model not loaded. Cannot make predictions.")
        return [(0, "neutral", 0.0)] * len(texts)
    
    try:
        # Preprocess texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Vectorize texts
        texts_vectorized = vectorizer.transform(processed_texts)
        
        # Make predictions
        sentiment_values = model.predict(texts_vectorized)
        
        # Get confidence scores
        confidences = np.abs(model.decision_function(texts_vectorized))
        if len(confidences.shape) > 1:
            confidences = np.mean(confidences, axis=1)
        
        # Normalize confidences
        confidences = np.minimum(1.0, confidences / 2.0)
        
        # Map sentiment values to labels
        results = [
            (value, map_sentiment_value(value), confidence)
            for value, confidence in zip(sentiment_values, confidences)
        ]
        
        logger.info(f"Predicted sentiment for {len(texts)} texts")
        return results
    
    except Exception as e:
        logger.error(f"Error batch predicting sentiment: {str(e)}")
        return [(0, "neutral", 0.0)] * len(texts)

if __name__ == "__main__":
    # Test prediction
    test_texts = [
        "Successfully rescued family from burning building with no injuries.",
        "Responded to fire alarm which was determined to be a false alarm.",
        "Multiple injuries reported due to building collapse during firefighting operation."
    ]
    
    for text in test_texts:
        value, label, conf = predict_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {label} ({value})")
        print(f"Confidence: {conf:.2f}")
        print() 