"""
Foot-pedal input for the recording pipeline.

PCSensor FootSwitch keyboard interface (...-event-kbd) is used for recording.
Run diagnose_pedal.py --all-interfaces to inspect mouse/ABS interfaces.
"""

from __future__ import annotations

import argparse
import os
import sys
import termios
import threading
import time
import tty

DEBOUNCE_SEC = 0.4

PCSENSOR_KBD = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"

PCSENSOR_BY_ID = [
    PCSENSOR_KBD,
    "/dev/input/by-id/usb-PCsensor_FootSwitch-event-mouse",
    "/dev/input/by-id/usb-PCsensor_FootSwitch-event-if01",
]

PEDAL_KEY_ALIASES = {
    "enter": "KEY_ENTER",
    "return": "KEY_ENTER",
    "space": "KEY_SPACE",
    "1": "KEY_1",
    "2": "KEY_2",
    "3": "KEY_3",
    "a": "KEY_A",
    "b": "KEY_B",
    "c": "KEY_C",
    "f13": "KEY_F13",
    "f14": "KEY_F14",
    "f15": "KEY_F15",
    "left": "BTN_LEFT",
    "middle": "BTN_MIDDLE",
    "right": "BTN_RIGHT",
    "btn_left": "BTN_LEFT",
    "btn_middle": "BTN_MIDDLE",
    "btn_right": "BTN_RIGHT",
    "any": "ANY",
    "*": "ANY",
}

_cancel_event: threading.Event | None = None
_active_devices: list = []


def normalize_pedal_key(name: str) -> str:
    return PEDAL_KEY_ALIASES.get(name.lower().strip(), name.upper())


def _is_pedal_device(dev) -> bool:
    name = dev.name.lower()
    return any(x in name for x in ("pcsensor", "footswitch", "foot switch", "usb pedal", "pedal"))


def _import_evdev():
    try:
        from evdev import InputDevice, ecodes, list_devices
        return InputDevice, ecodes, list_devices
    except ImportError:
        print(
            f"evdev is not installed for this Python:\n  {sys.executable}\n\n"
            f"Install it with:\n  {sys.executable} -m pip install evdev\n",
            file=sys.stderr,
        )
        raise


def cancel_pedal_wait() -> None:
    """Interrupt a blocking wait_for_pedal() call (e.g. on Ctrl+C)."""
    global _cancel_event
    if _cancel_event is not None:
        _cancel_event.set()
    for dev in _active_devices:
        try:
            dev.close()
        except OSError:
            pass


def list_input_devices() -> None:
    try:
        InputDevice, ecodes, list_devices = _import_evdev()
    except ImportError:
        return

    print("PCsensor symlinks:")
    for path in PCSENSOR_BY_ID:
        exists = os.path.exists(path)
        target = os.path.realpath(path) if exists else "missing"
        print(f"  {path} -> {target} {'OK' if exists else 'NOT FOUND'}")

    print("\nAll input devices:")
    for path in list_devices():
        dev = InputDevice(path)
        caps = dev.capabilities().get(ecodes.EV_KEY, [])
        hint = "  <-- foot pedal" if _is_pedal_device(dev) else ""
        print(f"  {path}\n    name={dev.name!r}  keys={len(caps)}{hint}")
        dev.close()


def list_pedal_paths(all_interfaces: bool = False) -> list[str]:
    """Pedal evdev paths. Default: keyboard interface only."""
    if not all_interfaces:
        path = find_pedal_device()
        return [path] if path else []

    paths: list[str] = []
    for path in PCSENSOR_BY_ID:
        if os.path.exists(path):
            paths.append(path)

    try:
        InputDevice, _, list_devices = _import_evdev()
        for path in list_devices():
            try:
                dev = InputDevice(path)
            except OSError:
                continue
            if _is_pedal_device(dev):
                paths.append(path)
            dev.close()
    except ImportError:
        pass

    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        try:
            real = os.path.realpath(path)
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        unique.append(path)
    return unique


def find_pedal_device() -> str | None:
    """Keyboard interface used for pedal key events."""
    if os.path.exists(PCSENSOR_KBD):
        return PCSENSOR_KBD

    try:
        InputDevice, _, list_devices = _import_evdev()
        for path in list_devices():
            try:
                dev = InputDevice(path)
            except OSError:
                continue
            if _is_pedal_device(dev) and "keyboard" in dev.name.lower():
                dev.close()
                return path
            dev.close()
    except ImportError:
        pass
    return None


def resolve_pedal_device(device_path: str | None) -> str | None:
    if device_path in (None, "", "auto"):
        return find_pedal_device()
    return device_path


def open_pedal_devices(device_path: str | None, all_interfaces: bool = False) -> list:
    InputDevice, _, _ = _import_evdev()

    if device_path not in (None, "", "auto"):
        return [InputDevice(device_path)]

    paths = list_pedal_paths(all_interfaces=all_interfaces)
    devices = []
    for path in paths:
        try:
            devices.append(InputDevice(path))
        except OSError as exc:
            print(f"Warning: could not open {path}: {exc}", flush=True)
    return devices


def _event_label(event) -> str:
    from evdev import ecodes

    if event.type == ecodes.EV_KEY:
        name = ecodes.KEY.get(event.code, f"CODE_{event.code}")
        action = {0: "up", 1: "down", 2: "hold"}.get(event.value, str(event.value))
        return f"KEY {name} {action}"
    if event.type == ecodes.EV_ABS:
        name = ecodes.ABS.get(event.code, f"ABS_{event.code}")
        return f"ABS {name} value={event.value}"
    if event.type == ecodes.EV_REL:
        name = ecodes.REL.get(event.code, f"REL_{event.code}")
        return f"REL {name} value={event.value}"
    return f"type={event.type} code={event.code} value={event.value}"


def _is_trigger_event(event, accept_any: bool, key_code: int | None) -> bool:
    from evdev import ecodes

    if event.type != ecodes.EV_KEY or event.value != 1:
        return False
    return accept_any or event.code == key_code


def resolve_key_code(target: str) -> int | None:
    from evdev import ecodes

    if hasattr(ecodes, target):
        return getattr(ecodes, target)
    return ecodes.keys.get(target)


def _wait_evdev(
    pedal_key: str,
    device_path: str | None,
    *,
    quiet: bool = False,
    all_interfaces: bool = False,
) -> bool:
    from evdev import ecodes

    global _cancel_event, _active_devices

    target = normalize_pedal_key(pedal_key)
    accept_any = target == "ANY"
    key_code = None if accept_any else resolve_key_code(target)
    if not accept_any and key_code is None:
        raise ValueError(f"Unknown pedal key {pedal_key!r} (resolved to {target})")

    devices = open_pedal_devices(device_path, all_interfaces=all_interfaces)
    if not devices:
        if not quiet:
            print("No foot pedal devices found.", flush=True)
        return False

    if not accept_any:
        devices = [
            d for d in devices if key_code in d.capabilities().get(ecodes.EV_KEY, [])
        ]
        if not devices:
            if not quiet:
                print(
                    f"No pedal interface has {target}. "
                    f"Try --pedal-key any or run: python3 diagnose_pedal.py",
                    flush=True,
                )
            return False

    cancel = threading.Event()
    _cancel_event = cancel
    _active_devices = devices

    pressed = threading.Event()
    stop = threading.Event()
    last_press = [0.0]
    lock = threading.Lock()

    def listen(dev) -> None:
        grabbed = False
        try:
            dev.grab()
            grabbed = True
            for event in dev.read_loop():
                if stop.is_set() or cancel.is_set():
                    break
                if not _is_trigger_event(event, accept_any, key_code):
                    continue
                now = time.monotonic()
                with lock:
                    if now - last_press[0] < DEBOUNCE_SEC:
                        continue
                    last_press[0] = now
                if not quiet:
                    print(f"Pedal: {_event_label(event)}", flush=True)
                pressed.set()
                stop.set()
                break
        except OSError:
            if not cancel.is_set() and not quiet:
                print(f"Pedal read error on {dev.path}", flush=True)
        finally:
            if grabbed:
                try:
                    dev.ungrab()
                except OSError:
                    pass
            try:
                dev.close()
            except OSError:
                pass

    threads = [threading.Thread(target=listen, args=(d,), daemon=True) for d in devices]
    for t in threads:
        t.start()

    try:
        while not pressed.is_set():
            if cancel.is_set():
                stop.set()
                raise KeyboardInterrupt
            if pressed.wait(0.1):
                break
    finally:
        stop.set()
        _cancel_event = None
        _active_devices = []
        for t in threads:
            t.join(timeout=1.0)

    return True


def _wait_pynput(pedal_key: str, quiet: bool = False) -> bool:
    from pynput import keyboard

    target = pedal_key.lower().strip()
    accept_any = target in ("any", "*")
    pressed = threading.Event()
    last_press = [0.0]

    def on_press(key) -> None:
        name = None
        if hasattr(key, "char") and key.char:
            name = key.char.lower()
        elif hasattr(key, "name") and key.name:
            name = key.name.lower()
        if accept_any or name == target or (target == "enter" and name == "enter"):
            now = time.monotonic()
            if now - last_press[0] >= DEBOUNCE_SEC:
                last_press[0] = now
                if accept_any and not quiet:
                    print(f"Pedal/key press: {name}", flush=True)
                pressed.set()
                return False

    with keyboard.Listener(on_press=on_press) as listener:
        while not pressed.is_set():
            if _cancel_event is not None and _cancel_event.is_set():
                listener.stop()
                return False
            if pressed.wait(0.1):
                break
        listener.stop()
    return True


def _wait_stdin(pedal_key: str) -> None:
    target = pedal_key.lower().strip()
    enter_targets = {"enter", "return"}
    accept_any = target in ("any", "*")

    print(
        "[stdin mode] Focus this terminal and press the pedal (or any key)...",
        flush=True,
    )
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            if _cancel_event is not None and _cancel_event.is_set():
                raise KeyboardInterrupt
            ch = sys.stdin.read(1)
            if accept_any:
                break
            if target in enter_targets and ch in ("\r", "\n"):
                break
            if target == "space" and ch == " ":
                break
            if len(ch) == 1 and ch.lower() == target:
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def wait_for_pedal(
    pedal_key: str = "b",
    device_path: str | None = "auto",
    *,
    quiet: bool = True,
    all_interfaces: bool = False,
) -> None:
    """Block until the foot pedal is pressed once. Raises KeyboardInterrupt if cancelled."""
    global _cancel_event
    _cancel_event = threading.Event()

    try:
        if _wait_evdev(pedal_key, device_path, quiet=quiet, all_interfaces=all_interfaces):
            return
    except ImportError:
        if not quiet:
            print("evdev not installed. Trying fallback...", flush=True)
    except (PermissionError, OSError) as exc:
        if not quiet:
            print(f"evdev unavailable ({exc}); trying fallback...", flush=True)

    if _cancel_event is not None and _cancel_event.is_set():
        raise KeyboardInterrupt

    try:
        if _wait_pynput(pedal_key, quiet=quiet):
            return
    except ImportError:
        pass

    if _cancel_event is not None and _cancel_event.is_set():
        raise KeyboardInterrupt

    _wait_stdin(pedal_key)


def parse_pedal_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pedal-key", default="b")
    parser.add_argument("--pedal-device", default="auto")
    parser.add_argument("--list-pedal-devices", action="store_true")
    return parser.parse_known_args()[0]


_open_pedal_devices = open_pedal_devices
_pick_evdev_devices = open_pedal_devices


def _should_grab(dev) -> bool:
    return _is_pedal_device(dev)
