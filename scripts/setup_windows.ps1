# One-step Windows setup. Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Creates .venv with Python 3.11/3.12, installs dependencies, checks Java, creates .env from the template.

$ErrorActionPreference = "Continue"   # native tools write warnings to stderr
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "== Checking Python ==" -ForegroundColor Cyan
$py = $null
foreach ($v in @("3.11", "3.12", "3.10")) {
    try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = $v; break } } catch {}
}
if (-not $py) {
    Write-Host "Python 3.10-3.12 not found. Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add to PATH')." -ForegroundColor Red
    exit 1
}
Write-Host "Using Python $py"

Write-Host "== Checking Java (needed by PySpark) ==" -ForegroundColor Cyan
$java = Get-Command java -ErrorAction SilentlyContinue
if (-not $java -and -not $env:JAVA_HOME) {
    Write-Host "Java not found. Install Eclipse Temurin JDK 17 from https://adoptium.net (tick 'Set JAVA_HOME'), reopen PowerShell and re-run." -ForegroundColor Red
    exit 1
}
& java -version

Write-Host "== Creating virtual environment (.venv) ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { & py "-$py" -m venv .venv }
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "Could not create .venv" -ForegroundColor Red; exit 1 }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed - see the messages above." -ForegroundColor Red; exit 1 }

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env - open it and paste your GEMINI_API_KEY." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete. Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m pytest                                  # offline tests (~2 min)"
Write-Host "  python run_pipeline.py --provider gemini --record # live Gemini run, saved for replay"
Write-Host "  python scripts\evaluate_extraction.py             # Gemini extraction accuracy"
