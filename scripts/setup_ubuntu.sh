#!/usr/bin/env bash
# One-time setup of the STT console on Ubuntu (lab PC).
# Run from the project folder:   bash scripts/setup_ubuntu.sh
set -e
sudo apt-get update
sudo apt-get install -y python3-venv libportaudio2
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
echo ""
echo "Setup done. Start the console with:"
echo "  source .venv/bin/activate && python app.py"