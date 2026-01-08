#!/bin/bash
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Starting service on port 8002..."
nohup uvicorn outbound_service:app --host 0.0.0.0 --port 8002 --reload > service.log 2>&1 &
echo "Service started. Check service.log for logs."
echo "PID: $!"
