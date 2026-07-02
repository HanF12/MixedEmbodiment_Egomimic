#!/bin/bash
# Mode 4: pure human bimanual hands — bird RealSense video + both hand poses.
# Output: recording/sessions/human_hands_bimanual/

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON="${PYTHON:-python3}"

if ! "${PYTHON}" -c "import evdev" 2>/dev/null; then
    echo "evdev not found for: $("${PYTHON}" -c 'import sys; print(sys.executable)')"
    echo "Install with: $("${PYTHON}" -c 'import sys; print(sys.executable)') -m pip install evdev"
    exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/record_pedal.py" \
    --mode human_hands_bimanual \
    --bird-camera 6 \
    --pedal-device auto \
    --pedal-key b \
    "$@"
