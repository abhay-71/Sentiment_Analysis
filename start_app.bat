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

REM Check if database exists, create if not
if not exist "app\data\incidents.db" (
    call :print_yellow "Database not found. Initializing database..."
    python -c "from app.data.database import init_db; init_db()"
    call :print_green "Database initialized."
)

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

REM Start mock API in the background
call :print_blue "Starting Mock API..."
start /b cmd /c python app\api\mock_api.py > app\logs\mock_api.log 2>&1
set "MOCK_PID=%ERRORLEVEL%"
call :print_green "Mock API started."

REM Short pause to ensure mock API is up
timeout /t 2 /nobreak > nul

REM Start model API in the background
call :print_blue "Starting Model API (with multi-model support)..."
start /b cmd /c python app\api\model_api.py > app\logs\model_api.log 2>&1
set "MODEL_PID=%ERRORLEVEL%"
call :print_green "Model API started."

REM Short pause to ensure model API is up
timeout /t 2 /nobreak > nul

REM Start Streamlit dashboard
call :print_blue "Starting Streamlit Dashboard..."
start /b cmd /c streamlit run app\dashboard\dashboard.py > app\logs\dashboard.log 2>&1
set "DASHBOARD_PID=%ERRORLEVEL%"
call :print_green "Dashboard started."

REM Create necessary directories for logs
if not exist "app\logs" mkdir app\logs

REM Write information for cleanup
echo @echo off > .app_processes.bat
echo taskkill /f /fi "WindowTitle eq mock_api.py*" >> .app_processes.bat
echo taskkill /f /fi "WindowTitle eq model_api.py*" >> .app_processes.bat
echo taskkill /f /fi "WindowTitle eq streamlit run*" >> .app_processes.bat
echo echo All services stopped. >> .app_processes.bat

call :print_header "Fire Brigade Sentiment Analysis Application is running!"
echo.
call :print_blue "Mock API:     http://localhost:5001"
call :print_blue "Model API:    http://localhost:5002"
call :print_blue "Dashboard:    http://localhost:8501"
echo.
call :print_yellow "Run stop_app.bat to stop all services."
echo.
call :print_header "Application is running in the background."

goto :eof

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