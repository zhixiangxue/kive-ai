# Install Kive Dependencies
# Run this script to install all required dependencies

Write-Host "Installing Kive Dependencies..." -ForegroundColor Green
Write-Host "=" * 60

# Activate virtual environment
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvPath
} else {
    Write-Host "Virtual environment not found at $venvPath" -ForegroundColor Red
    Write-Host "Please create it first: python -m venv .venv" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install core dependencies
Write-Host "`nInstalling core dependencies..." -ForegroundColor Yellow
pip install pydantic httpx fastapi uvicorn[standard] loguru llama-index-core python-dotenv

# Install Cognee and its dependencies
Write-Host "`nInstalling Cognee..." -ForegroundColor Yellow
pip install cognee

# Install embedding engines
Write-Host "`nInstalling FastEmbed (local embeddings)..." -ForegroundColor Yellow
pip install fastembed

# Install vector databases
Write-Host "`nInstalling LanceDB..." -ForegroundColor Yellow
pip install lancedb

# Optional: ChromaDB (if user wants to test)
# Write-Host "`nInstalling ChromaDB (optional)..." -ForegroundColor Yellow
# pip install chromadb

Write-Host "`n" + "=" * 60
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "=" * 60

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Edit examples\start_server.py and fill in your Bailain API key"
Write-Host "2. Run tests: python tests\test_cognee_integration.py"
Write-Host "3. Start server: python examples\start_server.py"
