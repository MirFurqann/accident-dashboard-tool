@echo off
:: ================================
:: Accident Dashboard Tool - Setup
:: Author: Furqan Ul Islam
:: Platform: Windows
:: ================================

echo ========================================================
echo Accident Dashboard Tool - Setup Script (Windows)
echo ========================================================
echo.

:menu
echo Choose an option:
echo [1] Setup using Conda (environment.yml)
echo [2] Setup using pip (requirements.txt)
echo [3] Run Dash Application
echo [4] Run Preprocessing Script (Excel_Balance.py)
echo [0] Exit
echo.

set /p choice="Enter your choice: "

if "%choice%"=="1" (
    echo.
    echo Creating conda environment from environment.yml...
    conda env create -f environment.yml
    call conda activate accident-dashboard-env
    echo Done! Environment activated.
    pause
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo Creating virtual environment and installing via pip...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
    echo Virtual environment ready and packages installed.
    pause
    goto menu
)

if "%choice%"=="3" (
    echo.
    echo Starting the Dash web application...
    python accident_dashboard_app.py
    pause
    goto menu
)

if "%choice%"=="4" (
    echo.
    echo Running Excel preprocessing script (Excel_Balance.py)...
    python Excel_Balance.py
    pause
    goto menu
)

if "%choice%"=="0" (
    echo Exiting setup. Goodbye!
    exit
)

echo Invalid option. Please try again.
pause
goto menu
