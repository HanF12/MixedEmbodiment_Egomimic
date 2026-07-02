#!/bin/bash
# Mode 2: left robot arm + right human hand.
# Records: left wrist cam, bird RealSense, right-hand pose, left joint data.
# Output: recording/sessions/left_robot_right_hand/

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON="${PYTHON:-python3}"

if ! "${PYTHON}" -c "import evdev" 2>/dev/null; then
    echo "evdev not found for: $("${PYTHON}" -c 'import sys; print(sys.executable)')"
    echo "Install with: $("${PYTHON}" -c 'import sys; print(sys.executable)') -m pip install evdev"
    exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/record_pedal.py" \
    --mode left_robot_right_hand \
    --bird-camera 6 \
    --pedal-device auto \
    --pedal-key b \
    "$@"
