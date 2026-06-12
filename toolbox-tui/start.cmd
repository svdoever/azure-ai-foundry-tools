@echo off
setlocal

:: ── Check az ──────────────────────────────────────────────────────────────────
where az >nul 2>&1
if errorlevel 1 (
    echo ERROR: Azure CLI ^('az'^) is not installed.
    echo Install it from: https://learn.microsoft.com/cli/azure/install-azure-cli
    exit /b 1
)

:: ── Check python ──────────────────────────────────────────────────────────────
set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if not defined PYTHON (
    where python3 >nul 2>&1 && set PYTHON=python3
)
if not defined PYTHON (
    where py >nul 2>&1 && set PYTHON=py
)
if not defined PYTHON (
    echo ERROR: Python is not installed.
    echo Install it from: https://www.python.org/downloads/
    exit /b 1
)

:: ── Azure login ───────────────────────────────────────────────────────────────
az account show >nul 2>&1
if errorlevel 1 (
    echo Not logged in to Azure. Running "az login"...
    az login
    if errorlevel 1 exit /b 1
)

:: ── Install dependencies ──────────────────────────────────────────────────────
echo Installing requirements...
%PYTHON% -m pip install -q -r "%~dp0requirements.txt"
if errorlevel 1 exit /b 1

:: ── Launch TUI ────────────────────────────────────────────────────────────────
%PYTHON% "%~dp0toolbox-tui.py"
