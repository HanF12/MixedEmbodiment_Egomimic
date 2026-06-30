#!/bin/bash

# Start all scripts in the background
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

python3 "${SCRIPT_DIR}/realsense_double_record.py" &
PID1=$!
python3 "${SCRIPT_DIR}/bird_record.py" &
PID2=$!
python3 "${SCRIPT_DIR}/store_joint.py" &
PID3=$!

# Function to kill all processes
cleanup() {
    echo "One script stopped. Killing others."
    kill $PID1 $PID2 $PID3 2>/dev/null # 2>/dev/null to suppress "No such process" errors
    exit 0
}

# Trap Ctrl+C (SIGINT) and any exit to call cleanup
trap cleanup SIGINT EXIT

# Wait for any child process to exit
wait -n
