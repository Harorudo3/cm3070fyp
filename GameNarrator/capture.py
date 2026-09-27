"""Finds the game window and returns the screen region to capture."""
import ctypes

import numpy as np
import win32gui
import win32process

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _process_name(hwnd):
    """Executable name for a window handle, lowercase."""
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        buf = ctypes.create_unicode_buffer(260)
        size = ctypes.c_uint(260)
        ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        ctypes.windll.kernel32.CloseHandle(handle)
        return buf.value.rsplit("\\", 1)[-1].lower() or None
    except OSError:
        return None


def find_window(match):
    """First visible window matching process name and/or title substring."""
    target_proc = match.get("process", "").lower()
    target_title = match.get("title_contains", "").lower()
    found = []

    def callback(hwnd, _):
        if found or not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        if target_title and target_title not in title.lower():
            return
        if target_proc and _process_name(hwnd) != target_proc:
            return
        found.append(hwnd)

    win32gui.EnumWindows(callback, None)
    return found[0] if found else None


def window_region(hwnd):
    """Client area in screen coordinates. Excludes the title bar and borders."""
    l, t, r, b = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (l, t))
    right, bottom = win32gui.ClientToScreen(hwnd, (r, b))
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


_warned = set()


def _apply_crop(region, crop):
    """Crop by fraction, not pixels, so it survives a resize.

    left and top default to centring the box.
    """
    if not crop:
        return region
    w = crop.get("width", 1.0)
    h = crop.get("height", 1.0)
    left = crop.get("left", (1.0 - w) / 2)
    top = crop.get("top", (1.0 - h) / 2)
    return {
        "left": region["left"] + int(region["width"] * left),
        "top": region["top"] + int(region["height"] * top),
        "width": int(region["width"] * w),
        "height": int(region["height"] * h),
    }


def resolve_region(sct, capture_cfg, game_name):
    """Region to grab this tick. Re-resolved each call so it follows the window."""
    if capture_cfg.get("mode") == "window":
        hwnd = find_window(capture_cfg.get("match", {}))
        if hwnd:
            rect = window_region(hwnd)
            if rect["width"] > 0 and rect["height"] > 0:
                return _apply_crop(rect, capture_cfg.get("crop"))
        if game_name not in _warned:
            print(f"  [capture] window not found for {game_name}, "
                  f"using monitor {capture_cfg.get('fallback_monitor', 1)}")
            _warned.add(game_name)
    monitor = sct.monitors[capture_cfg.get("fallback_monitor", 1)]
    return _apply_crop(monitor, capture_cfg.get("crop"))


def grab_frame(sct, region):
    """Capture the region as an RGB array."""
    shot = np.array(sct.grab(region))
    return shot[:, :, :3][:, :, ::-1]  # BGRA to RGB
