@echo off
:: Delegate to start.ps1 so the TUI runs inside a PowerShell host that supports
:: ANSI escape codes and interactive terminal control (required by Textual).
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-toolbox-tui.ps1"
exit /b %ERRORLEVEL%
