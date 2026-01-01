# Run SEMIINTEL Streamlit Web Application

Write-Host "🔬 Starting SEMIINTEL Web Application..." -ForegroundColor Cyan
Write-Host ""

# Set working directory
Set-Location "d:\LinkedinProjects\SemiIntel"

# Check if Streamlit is installed
Write-Host "📦 Checking Streamlit installation..." -ForegroundColor Yellow
$streamlitCheck = & "C:\Users\William Anthony\Miniconda3\python.exe" -c "import streamlit; print('✓ Streamlit installed')" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $streamlitCheck -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 Launching web application..." -ForegroundColor Cyan
    Write-Host "   URL: http://localhost:8501" -ForegroundColor Yellow
    Write-Host "   Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    # Run Streamlit
    & "C:\Users\William Anthony\Miniconda3\python.exe" -m streamlit run app.py
} else {
    Write-Host "❌ Streamlit not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing Streamlit..." -ForegroundColor Yellow
    conda install -y streamlit -c conda-forge
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Streamlit installed successfully!" -ForegroundColor Green
        Write-Host "🚀 Launching web application..." -ForegroundColor Cyan
        & "C:\Users\William Anthony\Miniconda3\python.exe" -m streamlit run app.py
    } else {
        Write-Host "❌ Failed to install Streamlit" -ForegroundColor Red
        Write-Host "Please run: conda install -y streamlit -c conda-forge" -ForegroundColor Yellow
    }
}
