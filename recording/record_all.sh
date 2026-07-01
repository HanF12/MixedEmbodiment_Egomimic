#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CLEANED_UP=0

stop_all() {
    (( CLEANED_UP )) && return
    CLEANED_UP=1



    echo "Stopping all recorders (waiting for .npy files to save)..."
    # SIGINT lets Python run finally blocks; avoid SIGTERM/SIGKILL until after a wait.
    kill -INT $PID1 $PID2 $PID3 2>/dev/null
    wait $PID1 $PID2 $PID3 2>/dev/null

    # Force-stop anything still running after graceful shutdown.
    kill $PID1 $PID2 $PID3 2>/dev/null
    wait $PID1 $PID2 $PID3 2>/dev/null
    echo "All recorders stopped."
}

trap stop_all SIGINT SIGTERM

python3 "${SCRIPT_DIR}/realsense_double_record.py" &
PID1=$!
python3 "${SCRIPT_DIR}/bird_record.py" -c 6 &
PID2=$!
python3 "${SCRIPT_DIR}/store_joint.py" &
PID3=$!

# If any recorder exits on its own (e.g. 'q' in a window), stop the rest gracefully too.
wait -n $PID1 $PID2 $PID3 2>/dev/null
stop_all
