#!/bin/bash

# Start Sentiment Analysis Application
# This script starts all components of the Fire Brigade Sentiment Analysis application

# Function to print colored output
print_color() {
    COLOR=$1
    TEXT=$2
    case $COLOR in
        "green") printf "\033[0;32m%s\033[0m\n" "$TEXT" ;;
        "red") printf "\033[0;31m%s\033[0m\n" "$TEXT" ;;
        "yellow") printf "\033[0;33m%s\033[0m\n" "$TEXT" ;;
        "blue") printf "\033[0;34m%s\033[0m\n" "$TEXT" ;;
        *) echo "$TEXT" ;;
    esac
}

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_color "yellow" "Virtual environment not activated. Please activate it first."
    print_color "blue" "Trying to activate from ./venv..."
    if [ -f "./venv/bin/activate" ]; then
        source ./venv/bin/activate
        print_color "green" "Virtual environment activated."
    else
        print_color "red" "Could not find virtual environment. Please create and activate it first."
        exit 1
    fi
fi

# Ensure we're in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_color "yellow" "No .env file found. Creating default configuration..."
    echo "MOCK_API_HOST=http://localhost" > .env
    echo "MOCK_API_PORT=5001" >> .env
    echo "MODEL_API_HOST=http://localhost" >> .env
    echo "MODEL_API_PORT=5002" >> .env
    print_color "green" "Created default .env file."
fi

# Start all services
print_color "blue" "Starting Fire Brigade Sentiment Analysis Application..."

# Create data directory if it doesn't exist
mkdir -p app/data

# Check if database exists, create if not
if [ ! -f "app/data/incidents.db" ]; then
    print_color "yellow" "Database not found. Initializing database..."
    python -c "from app.data.database import init_db; init_db()"
    print_color "green" "Database initialized."
fi

# Start mock API in the background
print_color "blue" "Starting Mock API..."
python app/api/mock_api.py &
MOCK_PID=$!
print_color "green" "Mock API started with PID $MOCK_PID."

# Short pause to ensure mock API is up
sleep 2

# Start model API in the background
print_color "blue" "Starting Model API (with multi-model support)..."
python app/api/model_api.py &
MODEL_PID=$!
print_color "green" "Model API started with PID $MODEL_PID."

# Short pause to ensure model API is up
sleep 2

# Start Streamlit dashboard
print_color "blue" "Starting Streamlit Dashboard..."
streamlit run app/dashboard/dashboard.py &
DASHBOARD_PID=$!
print_color "green" "Dashboard started with PID $DASHBOARD_PID."

# Write PIDs to file for future cleanup
echo "$MOCK_PID $MODEL_PID $DASHBOARD_PID" > .app_pids

print_color "green" "=========================================================="
print_color "green" "Fire Brigade Sentiment Analysis Application is running!"
print_color "green" "----------------------------------------"
print_color "blue" "Mock API:     http://localhost:5001"
print_color "blue" "Model API:    http://localhost:5002"
print_color "blue" "Dashboard:    http://localhost:8501"
print_color "green" "=========================================================="
print_color "yellow" "Press Ctrl+C to stop all services."

# Cleanup function
cleanup() {
    print_color "yellow" "Stopping all services..."
    kill $MOCK_PID $MODEL_PID $DASHBOARD_PID 2>/dev/null
    rm -f .app_pids
    print_color "green" "All services stopped."
    exit 0
}

# Register cleanup function
trap cleanup INT

# Keep script running
wait 