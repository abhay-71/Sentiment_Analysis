@echo off
REM Start Sentiment Analysis Application
REM This script starts all components of the Fire Brigade Sentiment Analysis application
REM Windows version

setlocal enabledelayedexpansion

REM Function to print colored output
REM Colors are limited in Windows cmd but we'll do our best
call :print_header "Starting Fire Brigade Sentiment Analysis Application..."

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    call :print_yellow "Virtual environment not activated. Trying to activate it..."
    if exist ".\venv\Scripts\activate.bat" (
        call .\venv\Scripts\activate.bat
        call :print_green "Virtual environment activated."
    ) else (
        call :print_red "Could not find virtual environment. Please create and activate it first."
        exit /b 1
    )
)

REM Ensure we're in the project root
cd /d "%~dp0"

REM Check if .env file exists
if not exist ".env" (
    call :print_yellow "No .env file found. Creating default configuration..."
    echo MOCK_API_HOST=http://localhost > .env
    echo MOCK_API_PORT=5001 >> .env
    echo MODEL_API_HOST=http://localhost >> .env
    echo MODEL_API_PORT=5002 >> .env
    call :print_green "Created default .env file."
)

REM Create data directory if it doesn't exist
if not exist "app\data" mkdir app\data
if not exist "app\logs" mkdir app\logs
if not exist "app\data\credentials" mkdir app\data\credentials

REM Check if database exists, create if not
if not exist "app\data\incidents.db" (
    call :print_yellow "Database not found. Initializing database..."
    python -c "from app.data.database import init_db; init_db()"
    call :print_green "Database initialized."
)

REM Initialize social media database
call :print_yellow "Initializing social media database..."
python -c "from app.social_media.database import init_db; init_db()"
call :print_green "Social media database initialized."

REM Check for models
call :print_blue "Checking for available sentiment models..."

REM Check for domain-aware model
if exist "app\models\domain_aware_sentiment_model.pkl" (
    call :print_green "Domain-aware sentiment model found."
) else (
    call :print_yellow "Domain-aware sentiment model not found. Some features may be limited."
)

REM Check for expanded model
if exist "app\models\expanded_sentiment_model.pkl" (
    if exist "app\models\expanded_vectorizer.pkl" (
        call :print_green "Expanded sentiment model found."
    ) else (
        call :print_yellow "Expanded sentiment model files incomplete. Run train_expanded_sentiment_model.py."
    )
) else (
    call :print_yellow "Expanded sentiment model not found. Run train_expanded_sentiment_model.py to create it."
)

REM Start mock API in a new window
call :print_blue "Starting Mock API..."
start "Mock API" cmd /c "python app\api\mock_api.py"
call :print_green "Mock API started."

REM Short pause to ensure mock API is up
timeout /t 2 /nobreak > nul

REM Start model API in a new window
call :print_blue "Starting Model API (with multi-model support)..."
start "Model API" cmd /c "python app\api\model_api.py"
call :print_green "Model API started."

REM Short pause to ensure model API is up
timeout /t 2 /nobreak > nul

REM Start social media batch processor in a new window
call :print_blue "Starting Social Media Batch Processor..."
start "Social Media Processor" cmd /c "python app\social_media\data_processor.py"
call :print_green "Social Media Batch Processor started."

REM Start social media authentication UI in a new window
call :print_blue "Starting Social Media Authentication UI..."
start "Social Media Auth" cmd /c "streamlit run app\social_media\social_media_auth.py --server.port 8503"
call :print_green "Social Media Authentication UI started."

REM Start social media data visualization in a new window
call :print_blue "Starting Social Media Data Visualization..."
start "Social Media Data" cmd /c "streamlit run app\social_media\social_media_data.py --server.port 8504"
call :print_green "Social Media Data Visualization started."

REM Start CSV data upload page in a new window
call :print_blue "Starting CSV Data Upload Page..."
start "CSV Upload" cmd /c "streamlit run app\social_media\csv_data_upload.py --server.port 8505"
call :print_green "CSV Data Upload Page started."

REM Start Streamlit dashboard in a new window
call :print_blue "Starting Streamlit Dashboard..."
start "Dashboard" cmd /c "streamlit run app\dashboard\dashboard.py"
call :print_green "Dashboard started."

REM No need to write a separate process file since we're using named windows

call :print_header "Fire Brigade Sentiment Analysis Application is running!"
echo.
call :print_blue "Mock API:                  http://localhost:5001"
call :print_blue "Model API:                 http://localhost:5002"
call :print_blue "Main Dashboard:            http://localhost:8501"
call :print_blue "Social Media Auth:         http://localhost:8503"
call :print_blue "Social Media Dashboard:    http://localhost:8504"
call :print_blue "CSV Data Upload:           http://localhost:8505"
echo.
call :print_yellow "Close the individual windows to stop the services."
echo.
call :print_header "Application is running. Don't close this window until you're done with the application."

REM Keep this window open
pause > nul
exit /b 0

REM ======= Functions =======

:print_header
echo.
echo ==========================================
echo %~1
echo ==========================================
echo.
goto :eof

:print_green
echo [SUCCESS] %~1
goto :eof

:print_red
echo [ERROR] %~1
goto :eof

:print_yellow
echo [WARNING] %~1
goto :eof

:print_blue
echo [INFO] %~1
goto :eof 