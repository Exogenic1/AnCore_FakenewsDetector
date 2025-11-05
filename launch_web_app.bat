@echo off
REM AnCore Web Application Launcher
REM Simple double-click launcher for Windows users

echo ========================================
echo  AnCore - Fake News Detector Web App
echo ========================================
echo.
echo Starting the web application...
echo Please wait while the server loads...
echo.
echo The app will open automatically in your browser.
echo If not, open: http://localhost:8501
echo.
echo To stop the server, close this window or press Ctrl+C
echo.
echo ========================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ERROR: Streamlit is not installed!
    echo.
    echo Installing Streamlit now...
    pip install streamlit
    echo.
)

REM Run the web app
streamlit run web_app.py

pause
