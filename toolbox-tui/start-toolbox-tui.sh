#!/usr/bin/env bash
set -euo pipefail

# ── Check az ──────────────────────────────────────────────────────────────────
if ! command -v az &>/dev/null; then
    echo "ERROR: Azure CLI ('az') is not installed."
    echo "Install it from: https://learn.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

# ── Check python ──────────────────────────────────────────────────────────────
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python is not installed."
    echo "Install it from: https://www.python.org/downloads/"
    exit 1
fi

# ── Azure login ───────────────────────────────────────────────────────────────
if ! az account show &>/dev/null; then
    echo "Not logged in to Azure. Running 'az login'..."
    az login
fi

# ── Install dependencies ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing requirements..."
"$PYTHON" -m pip install -q -r "$SCRIPT_DIR/requirements.txt"

# ── Launch TUI ────────────────────────────────────────────────────────────────
"$PYTHON" "$SCRIPT_DIR/toolbox-tui.py"
