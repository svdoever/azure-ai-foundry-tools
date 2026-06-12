# ── Check az ──────────────────────────────────────────────────────────────────
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Error "Azure CLI ('az') is not installed.`nInstall it from: https://learn.microsoft.com/cli/azure/install-azure-cli"
    exit 1
}

# ── Check python ──────────────────────────────────────────────────────────────
$python = $null
foreach ($candidate in @('python', 'python3', 'py')) {
    if (Get-Command $candidate -ErrorAction SilentlyContinue) {
        $python = $candidate
        break
    }
}
if (-not $python) {
    Write-Error "Python is not installed.`nInstall it from: https://www.python.org/downloads/"
    exit 1
}

# ── Azure login ───────────────────────────────────────────────────────────────
$null = az account show 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in to Azure. Running 'az login'..."
    az login
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

# ── Install dependencies ──────────────────────────────────────────────────────
$scriptDir = $PSScriptRoot
Write-Host "Installing requirements..."
& $python -m pip install -q -r "$scriptDir\requirements.txt"
if ($LASTEXITCODE -ne 0) { exit 1 }

# ── Launch TUI ────────────────────────────────────────────────────────────────
& $python "$scriptDir\toolbox-tui.py"
