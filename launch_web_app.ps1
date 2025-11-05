# AnCore Web Application Launcher (PowerShell)
# Double-click to launch the web interface

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host " AnCore - Fake News Detector Web App"    -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting the web application..." -ForegroundColor Green
Write-Host "Please wait while the server loads..." -ForegroundColor Yellow
Write-Host ""

# Check if streamlit is installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import streamlit" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit not found"
    }
    Write-Host "✓ Streamlit is installed" -ForegroundColor Green
} catch {
    Write-Host "✗ Streamlit is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing Streamlit now..." -ForegroundColor Yellow
    pip install streamlit
    Write-Host ""
}

# Check if model exists
if (Test-Path "output\models\best_model.pt") {
    Write-Host "✓ Model file found" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: Model file not found!" -ForegroundColor Yellow
    Write-Host "  Please train the model first: python ancore_main.py --mode train" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "The app will open in your browser at:"     -ForegroundColor White
Write-Host "http://localhost:8501"                      -ForegroundColor Green
Write-Host ""
Write-Host "To stop the server:"                        -ForegroundColor White
Write-Host "- Close this window, or"                    -ForegroundColor White
Write-Host "- Press Ctrl+C"                            -ForegroundColor White
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

# Launch the web app
streamlit run web_app.py

# Keep window open if there's an error
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
