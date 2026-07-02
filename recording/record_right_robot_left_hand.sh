#!/bin/bash
# Mode 3: right robot arm + left human hand (opposite of mode 2).
# Records: right wrist cam, bird RealSense, left-hand pose, right joint data.
# Output: recording/sessions/right_robot_left_hand/

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON="${PYTHON:-python3}"

if ! "${PYTHON}" -c "import evdev" 2>/dev/null; then
    echo "evdev not found for: $("${PYTHON}" -c 'import sys; print(sys.executable)')"
    echo "Install with: $("${PYTHON}" -c 'import sys; print(sys.executable)') -m pip install evdev"
    exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/record_pedal.py" \
    --mode right_robot_left_hand \
    --bird-camera 6 \
    --pedal-device auto \
    --pedal-key b \
    "$@"
