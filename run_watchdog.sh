#!/bin/bash

clear
echo "============================================================"
echo "   WATCHDOG AI - ONE-CLICK EXECUTION"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found! Please install Python 3.8+"
    exit 1
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment exists"
fi

# Activate venv
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[3/4] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet numpy pandas scikit-learn flask flask-cors pytest 2>/dev/null

# Run demo
echo "[4/4] Running Watchdog AI Demo..."
echo ""
echo "============================================================"
echo ""
python run.py

echo ""
echo "============================================================"
echo "Execution complete!"