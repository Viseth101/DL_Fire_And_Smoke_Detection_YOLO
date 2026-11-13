"""Tkinter live video HUD for smoke and fire detection."""

import platform
import time

import cv2
import numpy as np
from PIL import Image
# 1. NATIVE TKINTER LIVE VIDEO PLAYER (CROSS-PLATFORM)
# =====================================================================
def draw_hud(img, hud, scale=1.0):
    """
    Draws the translucent status bar onto `img` (in place).
    `scale` sets the text size (1.0 = 720p-equivalent); the whole HUD shrinks
    automatically if a line would not fit the image width, so it is never clipped.
    `hud` is a list of (text, bgr_color, base_font_scale, base_thickness).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]
    base_pad, base_gap, base_top = 15, 9, 10

    widest = max(cv2.getTextSize(t, font, fs, th)[0][0] for (t, _, fs, th) in hud)
    if widest + 2 * base_pad > 0:
        scale = min(scale, w / (widest + 2 * base_pad))
    scale = max(scale, 0.3)

    pad = int(base_pad * scale)
    y = int(base_top * scale)
    positions = []
    for (text, _, fs, th) in hud:
        (_, text_h), baseline = cv2.getTextSize(text, font, fs * scale, max(1, round(th * scale)))
        y += text_h
        positions.append(y)
        y += baseline + int(base_gap * scale)
    bar_h = min(h, y)

    roi = img[0:bar_h, :]
    roi[:] = cv2.addWeighted(np.full_like(roi, 20), 0.65, roi, 0.35, 0)

    for (text, color, fs, th), ty in zip(hud, positions):
        cv2.putText(img, text, (pad, ty), font, fs * scale, color, max(1, round(th * scale)), cv2.LINE_AA)


def _enable_windows_dpi_awareness():
    """Without this, Windows bitmap-stretches the Tk window on high-DPI screens (blurry video)."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class LiveVideoWindow:
    """
    Opens a live GUI window using Tkinter + PIL on Windows, macOS, and WSL2.

    - Opens fitted to ~80% of the screen (or `display_size`), preserving aspect ratio.
    - Freely resizable; the video is rescaled and letterboxed to the window on every frame.
    - HUD is drawn after scaling, so text stays a constant, readable size.
    - Keys: Q quit, P pause, F / F11 toggle fullscreen, Esc leaves fullscreen (or quits),
            + / - playback speed, 0 reset to 1x, M toggle max speed (no frame pacing).
    """
    SCREEN_FILL = 0.80
    MIN_SIZE = (320, 180)
    SPEED_STEPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

    def __init__(self, title="Smoke & Fire Detection HUD", display_size=None, fullscreen=False, speed=1.0):
        self.available = False
        self.quit_requested = False
        self.paused = False
        self.fullscreen = False
        self.max_speed = speed <= 0          # speed 0 = run as fast as the model allows
        self.speed = speed if speed > 0 else 1.0
        self.display_size = display_size
        self.start_fullscreen = fullscreen
        self._sized = False
        self._target = None
        self._last = None
        self._rendered_size = None
        try:
            _enable_windows_dpi_awareness()
            import tkinter as tk
            from PIL import ImageTk
            self.tk = tk
            self.ImageTk = ImageTk

            self.root = tk.Tk()
            self.root.title(title)
            self.root.configure(bg="black")
            self.root.minsize(*self.MIN_SIZE)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            for key in ("<q>", "<Q>"):
                self.root.bind(key, lambda e: self._on_close())
            for key in ("<p>", "<P>"):
                self.root.bind(key, lambda e: self._toggle_pause())
            for key in ("<f>", "<F>", "<F11>"):
                self.root.bind(key, lambda e: self._toggle_fullscreen())
            self.root.bind("<Escape>", lambda e: self._on_escape())
            for key in ("<plus>", "<equal>", "<KP_Add>"):
                self.root.bind(key, lambda e: self._step_speed(+1))
            for key in ("<minus>", "<underscore>", "<KP_Subtract>"):
                self.root.bind(key, lambda e: self._step_speed(-1))
            self.root.bind("<Key-0>", lambda e: self._set_speed(1.0))
            for key in ("<m>", "<M>"):
                self.root.bind(key, lambda e: self._toggle_max_speed())

            self.label = tk.Label(self.root, bg="black", bd=0, highlightthickness=0, padx=0, pady=0)
            self.label.pack(fill=tk.BOTH, expand=True)

            # HUD text size in screen pixels: 1.0 at 96 DPI, larger on high-DPI displays
            self.ui_scale = min(3.0, max(1.0, self.root.winfo_fpixels("1i") / 96.0))
            self.available = True
        except Exception as e:
            print(f"\nLive GUI window unavailable ({e}). Continuing in headless recording mode.\n")
            self.available = False

    # ---------- key handlers ----------
    def _on_close(self):
        self.quit_requested = True

    def _toggle_pause(self):
        self.paused = not self.paused

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _on_escape(self):
        if self.fullscreen:
            self._toggle_fullscreen()
        else:
            self._on_close()

    def _set_speed(self, value):
        self.speed = value
        self.max_speed = False

    def _step_speed(self, direction):
        steps = self.SPEED_STEPS
        if direction > 0:
            nxt = next((s for s in steps if s > self.speed + 1e-6), steps[-1])
        else:
            nxt = next((s for s in reversed(steps) if s < self.speed - 1e-6), steps[0])
        self._set_speed(nxt)

    def _toggle_max_speed(self):
        self.max_speed = not self.max_speed

    def wait(self, seconds):
        """Sleep for frame pacing while keeping the window responsive."""
        end = time.perf_counter() + seconds
        while not self.quit_requested and not self.paused:
            remaining = end - time.perf_counter()
            if remaining <= 0:
                break
            self.root.update()
            time.sleep(min(0.004, remaining))

    # ---------- sizing ----------
    def _apply_initial_geometry(self, vid_w, vid_h):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        if self.display_size:
            win_w, win_h = self.display_size
        else:
            s = min(self.SCREEN_FILL * screen_w / vid_w, self.SCREEN_FILL * screen_h / vid_h)
            win_w, win_h = int(vid_w * s), int(vid_h * s)
        win_w = max(self.MIN_SIZE[0], win_w)
        win_h = max(self.MIN_SIZE[1], win_h)
        x = max(0, (screen_w - win_w) // 2)
        y = max(0, (screen_h - win_h) // 2)
        self._target = (win_w, win_h)
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        if self.start_fullscreen:
            self._toggle_fullscreen()
        self.root.update()
        self._sized = True

    def _canvas_size(self):
        w, h = self.label.winfo_width(), self.label.winfo_height()
        if w < 2 or h < 2:  # not mapped yet
            return self._target
        return w, h

    # ---------- drawing ----------
    def _render(self):
        frame_bgr, hud = self._last
        win_w, win_h = self._canvas_size()
        vid_h, vid_w = frame_bgr.shape[:2]

        s = min(win_w / vid_w, win_h / vid_h)
        new_w, new_h = max(1, int(vid_w * s)), max(1, int(vid_h * s))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        scaled = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interp)
        if hud:
            draw_hud(scaled, hud, self.ui_scale)

        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        x0, y0 = (win_w - new_w) // 2, (win_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = scaled

        imgtk = self.ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)
        self._rendered_size = (win_w, win_h)

    def show(self, frame_bgr, hud=None):
        if not self.available or self.quit_requested:
            return False
        try:
            if not self._sized:
                self._apply_initial_geometry(frame_bgr.shape[1], frame_bgr.shape[0])
            self._last = (frame_bgr, hud)
            self._render()
            self.root.update_idletasks()
            self.root.update()

            while self.paused and not self.quit_requested:
                self.root.update()
                if self._canvas_size() != self._rendered_size:
                    self._render()  # keep resizing/fullscreen responsive while paused
                time.sleep(0.05)
            return not self.quit_requested
        except Exception:
            self.quit_requested = True
            return False

    def close(self):
        if self.available:
            try:
                self.root.destroy()
            except Exception:
                pass
