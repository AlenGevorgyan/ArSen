@echo off
title Smart Garden Manager
color 0A

echo ================================
echo    Smart Garden Manager
echo ================================
echo.
echo Starting the application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python from https://www.python.org
    echo.
    pause
    exit /b 1
)

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo Flask not found. Installing required packages...
    pip install flask flask-cors
    echo.
)

REM Start the Flask server and open browser
echo Starting Flask server...
echo.
start "" http://localhost:5000
python smart_garden_backend.py

pause