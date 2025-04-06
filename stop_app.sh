#!/bin/bash

# Stop Sentiment Analysis Application
# This script stops all components of the Fire Brigade Sentiment Analysis application

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

# Check if .app_pids file exists
if [ -f ".app_pids" ]; then
    print_color "yellow" "Stopping Fire Brigade Sentiment Analysis Application..."
    PIDS=$(cat .app_pids)
    for PID in $PIDS; do
        if ps -p $PID > /dev/null; then
            print_color "blue" "Stopping process with PID $PID..."
            kill $PID
            print_color "green" "Stopped process with PID $PID."
        else
            print_color "yellow" "Process with PID $PID is not running."
        fi
    done
    rm -f .app_pids
    print_color "green" "All services stopped."
else
    # Try to find and kill any remaining processes
    print_color "yellow" "No PID file found. Attempting to find and stop processes..."
    
    # Find processes for mock API
    MOCK_PIDS=$(ps aux | grep "mock_api.py" | grep -v grep | awk '{print $2}')
    if [ -n "$MOCK_PIDS" ]; then
        for PID in $MOCK_PIDS; do
            print_color "blue" "Stopping Mock API with PID $PID..."
            kill $PID
        done
        print_color "green" "Mock API stopped."
    else
        print_color "yellow" "No Mock API process found."
    fi
    
    # Find processes for model API
    MODEL_PIDS=$(ps aux | grep "model_api.py" | grep -v grep | awk '{print $2}')
    if [ -n "$MODEL_PIDS" ]; then
        for PID in $MODEL_PIDS; do
            print_color "blue" "Stopping Model API with PID $PID..."
            kill $PID
        done
        print_color "green" "Model API stopped."
    else
        print_color "yellow" "No Model API process found."
    fi
    
    # Find processes for Streamlit dashboard
    STREAMLIT_PIDS=$(ps aux | grep "streamlit run" | grep "dashboard.py" | grep -v grep | awk '{print $2}')
    if [ -n "$STREAMLIT_PIDS" ]; then
        for PID in $STREAMLIT_PIDS; do
            print_color "blue" "Stopping Streamlit dashboard with PID $PID..."
            kill $PID
        done
        print_color "green" "Streamlit dashboard stopped."
    else
        print_color "yellow" "No Streamlit dashboard process found."
    fi
fi

print_color "green" "=========================================================="
print_color "green" "Fire Brigade Sentiment Analysis Application stopped."
print_color "green" "==========================================================" 