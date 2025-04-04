"""
Mock API for Fire Brigade Incident Reports

This module provides a Flask API that returns random incident reports
with realistic content. It simulates a real API that would be used
to fetch incidents from the fire brigade's system.
"""
import os
import sys
import logging
from flask import Flask, jsonify, request
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the mock data generator
from app.utils.mock_data_generator import generate_incidents
from app.utils.config import MOCK_API_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mock_api')

# Create Flask application
app = Flask(__name__)

@app.route('/get_incidents', methods=['GET'])
def get_incidents():
    """
    Endpoint to fetch incident reports.
    
    Query Parameters:
        count (int): Number of incidents to return (default: 5)
        
    Returns:
        JSON: List of incident reports
    """
    try:
        # Get count parameter with default value
        count = request.args.get('count', default=5, type=int)
        
        # Validate count parameter
        if count < 1:
            return jsonify({"error": "Count must be at least 1"}), 400
        if count > 100:
            return jsonify({"error": "Count cannot exceed 100"}), 400
        
        # Generate random incidents
        incidents = generate_incidents(count)
        
        # Add a small delay to simulate network latency
        time.sleep(0.1)
        
        logger.info(f"Generated {count} incidents")
        return jsonify(incidents)
    
    except Exception as e:
        logger.error(f"Error generating incidents: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        JSON: Status of the API
    """
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('MOCK_API_PORT', MOCK_API_PORT))
    logger.info(f"Starting mock API on port {port}")
    app.run(host='0.0.0.0', port=port) 