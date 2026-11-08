import os
import sys
import glob
import time
import shutil
import platform
import argparse
import threading
import subprocess
from datetime import datetime
from collections import deque


# =====================================================================
# 0. AUTO-INSTALLER FOR PORTABILITY ON ANY COMPUTER
# =====================================================================
def ensure_dependencies():
    """
    Checks if required Python packages and GUI libraries are installed.
    If run on a new computer, automatically installs them before starting.
    """
    required_packages = {
        "cv2": "opencv-python",
        "ultralytics": "ultralytics",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime",
        "PIL": "Pillow",
        "requests": "requests",
    }

    missing_pip = []
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_pip.append(pip_name)

    if missing_pip:
        print(f"📦 Installing missing Python packages for portability: {', '.join(missing_pip)}...")
        # Use 'uv pip' if available in environment, otherwise standard 'pip'
        if shutil.which("uv"):
            cmd = ["uv", "pip", "install", *missing_pip]
        else:
            cmd = [sys.executable, "-m", "pip", "install", *missing_pip]
        subprocess.check_call(cmd)
        print("✅ Python dependencies installed successfully!")

    # Check Tkinter on Linux/WSL (Windows & macOS include tkinter with Python by default)
    if platform.system() == "Linux":
        try:
            import tkinter  # noqa: F401
        except ImportError:
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            tk_pkg = f"python{py_ver}-tk"
            print(f"⚙️ Linux detected without '{tk_pkg}'. Attempting auto-install via apt...")
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=False)
                subprocess.run(["sudo", "apt-get", "install", "-y", tk_pkg, "python3-tk"], check=False)
            except Exception as e:
                print(f"⚠️ Could not auto-install {tk_pkg}: {e}")


# Run dependency check immediately before third-party imports
ensure_dependencies()

# Prevent OpenCV from loading its broken Qt5 'xcb' plugin on Linux/WSL2
if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO


# =====================================================================
# PATH CONFIGURATION (100% Portable Relative Paths + Local Fallback)
# =====================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DEPLOY_DIR = os.path.join(PROJECT_DIR, "Video Deploy")
DEFAULT_VIDEO_PATH = os.path.join(VIDEO_DEPLOY_DIR, "Source", "sample_fire3.mp4")

# Primary Portable Model Path (inside Project/Video Deploy/Models/)
PORTABLE_MODEL_ONNX = os.path.join(VIDEO_DEPLOY_DIR, "Models", "yolo26s_best.onnx")
PORTABLE_MODEL_PT = os.path.join(VIDEO_DEPLOY_DIR, "Models", "yolo26s_best.pt")

# Fallback Local Training Path (if running on Seth's training machine before copying)
LOCAL_RESULTS_ROOT = "/home/Seth_Ubuntu/dl_subject/runs/detect"
FALLBACK_TRAIN_ONNX = os.path.join(
    LOCAL_RESULTS_ROOT, "Smoke_Fire_Detection", "yolo26s_tuned", "weights", "best.onnx"
)


# =====================================================================
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
    - Keys: Q quit, P pause, F / F11 toggle fullscreen, Esc leaves fullscreen (or quits).
    """
    SCREEN_FILL = 0.80
    MIN_SIZE = (320, 180)

    def __init__(self, title="Smoke & Fire Detection HUD", display_size=None, fullscreen=False):
        self.available = False
        self.quit_requested = False
        self.paused = False
        self.fullscreen = False
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

            self.label = tk.Label(self.root, bg="black", bd=0, highlightthickness=0, padx=0, pady=0)
            self.label.pack(fill=tk.BOTH, expand=True)

            # HUD text size in screen pixels: 1.0 at 96 DPI, larger on high-DPI displays
            self.ui_scale = min(3.0, max(1.0, self.root.winfo_fpixels("1i") / 96.0))
            self.available = True
        except Exception as e:
            print(f"\n⚠️ Live GUI window unavailable ({e}). Continuing in headless recording mode...\n")
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


# =====================================================================
# 2. PORTABLE MODEL PATH RESOLVER & AUTO-COPIER
# =====================================================================
def resolve_model_path(requested_path=None):
    """
    1. Checks if user passed a custom --model path.
    2. Checks portable 'Video Deploy/Models/yolo26s_best.onnx'.
    3. Checks local training folder and automatically copies weights into 'Video Deploy/Models/' so the project folder is immediately portable for other computers!
    """
    models_dir = os.path.join(VIDEO_DEPLOY_DIR, "Models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Explicit custom path
    if requested_path and os.path.exists(requested_path):
        return requested_path

    # 2. Check portable ONNX inside Video Deploy/Models/
    if os.path.exists(PORTABLE_MODEL_ONNX):
        return PORTABLE_MODEL_ONNX

    # 3. Check portable PT inside Video Deploy/Models/ and export to ONNX
    if os.path.exists(PORTABLE_MODEL_PT):
        print(f"⚙️ Exporting portable {PORTABLE_MODEL_PT} to ONNX...")
        exported = YOLO(PORTABLE_MODEL_PT).export(format="onnx", imgsz=640, dynamic=False, simplify=True)
        shutil.move(str(exported), PORTABLE_MODEL_ONNX)
        return PORTABLE_MODEL_ONNX

    # 4. Search local training directories and auto-copy to Video Deploy/Models/ for portability
    search_roots = [PROJECT_DIR, LOCAL_RESULTS_ROOT, "/home/Seth_Ubuntu/dl_subject"]
    for root in search_roots:
        if not os.path.exists(root):
            continue

        onnx_matches = glob.glob(os.path.join(root, "**", "*yolo26s_tuned*", "weights", "best.onnx"), recursive=True)
        if onnx_matches:
            shutil.copy2(onnx_matches[0], PORTABLE_MODEL_ONNX)
            print(f"📦 Copied ONNX weights into portable folder: {PORTABLE_MODEL_ONNX}")
            return PORTABLE_MODEL_ONNX

        pt_matches = glob.glob(os.path.join(root, "**", "*yolo26s_tuned*", "weights", "best.pt"), recursive=True)
        if pt_matches:
            shutil.copy2(pt_matches[0], PORTABLE_MODEL_PT)
            print(f"⚙️ Exporting {pt_matches[0]} to portable ONNX...")
            exported = YOLO(PORTABLE_MODEL_PT).export(format="onnx", imgsz=640, dynamic=False, simplify=True)
            if str(exported) != PORTABLE_MODEL_ONNX:
                shutil.move(str(exported), PORTABLE_MODEL_ONNX)
            return PORTABLE_MODEL_ONNX

    sys.exit(
        f"\n❌ ERROR: Model weights not found!\n"
        f"To run on another computer without retraining, place 'yolo26s_best.onnx' inside:\n"
        f"  👉 {models_dir}\n"
    )


# =====================================================================
# 3. ASYNCHRONOUS NOTIFICATION ENGINE (WITH COOLDOWN)
# =====================================================================
class AlertDispatcher:
    def __init__(self, cooldown_seconds=60, enable_alerts=False):
        self.cooldown_seconds = cooldown_seconds
        self.enable_alerts = enable_alerts
        self.last_alert_time = 0.0

        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

    def can_trigger(self):
        return (time.time() - self.last_alert_time) >= self.cooldown_seconds

    def cooldown_remaining(self):
        rem = self.cooldown_seconds - (time.time() - self.last_alert_time)
        return max(0.0, rem)

    def trigger_alert(self, hazard_types, frame, max_conf):
        if not self.can_trigger():
            return False

        self.last_alert_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hazard_str = " & ".join(hazard_types)
        message = (
            f"🚨 [HAZARD ALERT] {hazard_str} DETECTED!\n"
            f"• Time: {timestamp}\n"
            f"• Peak Confidence: {max_conf:.2f}\n"
            f"• Cooldown: {self.cooldown_seconds}s"
        )

        alerts_dir = os.path.join(VIDEO_DEPLOY_DIR, "Alerts_Log")
        os.makedirs(alerts_dir, exist_ok=True)
        snap_path = os.path.join(alerts_dir, f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(snap_path, frame)

        thread = threading.Thread(
            target=self._send_payload,
            args=(message, snap_path),
            daemon=True
        )
        thread.start()
        return True

    def _send_payload(self, message, image_path):
        print("\n" + "=" * 55)
        print(message)
        print(f"📸 Snapshot saved: {image_path}")
        print("=" * 55)

        if not self.enable_alerts:
            return

        if self.telegram_bot_token != "YOUR_BOT_TOKEN_HERE":
            try:
                url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto"
                with open(image_path, "rb") as img_file:
                    resp = requests.post(
                        url,
                        data={"chat_id": self.telegram_chat_id, "caption": message},
                        files={"photo": img_file},
                        timeout=10
                    )
                if resp.status_code == 200:
                    print("✅ Telegram notification delivered successfully!")
                else:
                    print(f"⚠️ Telegram API returned {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"⚠️ Failed to send Telegram alert: {e}")


# =====================================================================
# 4. MAIN REAL-TIME INFERENCE PIPELINE
# =====================================================================
def select_source_interactively():
    source_dir = os.path.join(VIDEO_DEPLOY_DIR, "Source")
    print("\n" + "=" * 55)
    print("🔥 SMOKE & FIRE REAL-TIME DETECTION SYSTEM")
    print("=" * 55)
    print("Select Input Source:")
    print("  [1] Video File (.mp4, .avi, .mkv)")
    print("  [2] Live USB / Integrated Webcam")
    print("  [3] IP Camera / Phone Stream (RTSP or HTTP URL)")
    choice = input("\nEnter option (1/2/3): ").strip()

    if choice == "1":
        vids = []
        if os.path.exists(source_dir):
            vids = sorted([f for f in os.listdir(source_dir) if f.lower().endswith((".mp4", ".avi", ".mkv"))])
            if vids:
                print(f"\n📂 Available videos in 'Video Deploy/Source':")
                for idx, v in enumerate(vids, 1):
                    print(f"   ({idx}) {v}")
        user_in = input("\nEnter video number, filename, or path [Press Enter for sample_fire3.mp4]: ").strip().strip("\"'")
        if not user_in:
            return DEFAULT_VIDEO_PATH
        if vids and user_in.isdigit():
            idx = int(user_in) - 1
            if 0 <= idx < len(vids):
                return os.path.join(source_dir, vids[idx])
        candidate_in_source = os.path.join(source_dir, user_in)
        if os.path.exists(candidate_in_source):
            return candidate_in_source
        return user_in
    elif choice == "2":
        cam_id = input("Enter webcam index [default: 0]: ").strip()
        return int(cam_id) if cam_id.isdigit() else 0
    elif choice == "3":
        url = input("Enter RTSP/HTTP stream URL (e.g., http://192.168.1.50:8080/video): ").strip()
        return url.strip("\"'")
    else:
        print("Invalid choice, defaulting to sample_fire3.mp4.")
        return DEFAULT_VIDEO_PATH

# =====================================================================
# DEFAULT DEPLOYMENT SETTINGS (Change your defaults here in ONE place!)
# =====================================================================
DEFAULT_CONF = 0.10          # Minimum confidence threshold (0.0 to 1.0)
DEFAULT_BUFFER = 5           # Rolling window size (frames)
DEFAULT_HITS = 4             # Required detections inside buffer to trigger
DEFAULT_COOLDOWN = 60        # Seconds between notifications
DEFAULT_SAVE_VIDEO = False   # Set True to auto-save video, False to disable

def run_deployment(
    model_path=None,
    source=None,
    conf_thres=DEFAULT_CONF,
    buffer_size=DEFAULT_BUFFER,
    required_hits=DEFAULT_HITS,
    cooldown=DEFAULT_COOLDOWN,
    save_output=DEFAULT_SAVE_VIDEO,
    enable_alerts=False,
    no_gui=False,
    display_size=None,
    fullscreen=False
):
    resolved_model_path = resolve_model_path(model_path)
    print(f"🚀 Loading model: {resolved_model_path}")
    model = YOLO(resolved_model_path, task="detect")

    if source is None:
        source = select_source_interactively()
    elif isinstance(source, str) and source.isdigit():
        source = int(source)
    elif isinstance(source, str):
        source = source.strip("\"'")
        alt_path = os.path.join(VIDEO_DEPLOY_DIR, "Source", source)
        if not os.path.exists(source) and os.path.exists(alt_path):
            source = alt_path

    print(f"🎥 Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"\n❌ Error: Unable to open video source '{source}'. Check path or camera connection.")

    gui = None if no_gui else LiveVideoWindow(
        "Smoke & Fire Detection HUD (Q: Quit | P: Pause | F/F11: Fullscreen)",
        display_size=display_size,
        fullscreen=fullscreen,
    )
    if gui and gui.available:
        print("✅ Live Popup Window started! Q = quit, P = pause, F/F11 = fullscreen, Esc = exit fullscreen.")
    else:
        save_output = True

    out_writer = None
    output_dir = os.path.join(VIDEO_DEPLOY_DIR, "Output")
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(source, str) and os.path.isfile(source):
        src_name = os.path.splitext(os.path.basename(source))[0]
        output_video_path = os.path.join(output_dir, f"{src_name}_detected.mp4")
    else:
        output_video_path = os.path.join(output_dir, "live_stream_detected.mp4")

    if save_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        print(f"💾 Recording annotated output video to: {output_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    smoke_buffer = deque(maxlen=buffer_size)
    fire_buffer = deque(maxlen=buffer_size)
    dispatcher = AlertDispatcher(cooldown_seconds=cooldown, enable_alerts=enable_alerts)

    colors = {0: (180, 180, 180), 1: (0, 0, 255)}  # BGR: Smoke=Gray, Fire=Red
    classes = {0: "SMOKE", 1: "FIRE"}

    prev_time = time.time()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n⏹️ End of stream reached.")
            break

        frame_idx += 1
        results = model.predict(frame, imgsz=640, conf=conf_thres, verbose=False)[0]

        current_smoke = False
        current_fire = False
        max_conf_in_frame = 0.0
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id == 0:
                current_smoke = True
            elif cls_id == 1:
                current_fire = True

            max_conf_in_frame = max(max_conf_in_frame, conf)
            detections.append((cls_id, conf, x1, y1, x2, y2))

        smoke_buffer.append(1 if current_smoke else 0)
        fire_buffer.append(1 if current_fire else 0)

        smoke_confirmed = sum(smoke_buffer) >= required_hits
        fire_confirmed = sum(fire_buffer) >= required_hits

        active_hazards = []
        if smoke_confirmed:
            active_hazards.append("SMOKE")
        if fire_confirmed:
            active_hazards.append("FIRE")

        # Box line/label size relative to source resolution (720p = 1.0) so boxes
        # stay visible after the frame is scaled down for display.
        box_s = max(1.0, frame.shape[0] / 720.0)
        box_thick = max(2, round(3 * box_s))
        box_font = 0.65 * box_s
        box_font_thick = max(1, round(2 * box_s))
        for (cls_id, conf, x1, y1, x2, y2) in detections:
            if (cls_id == 0 and smoke_confirmed) or (cls_id == 1 and fire_confirmed):
                color = colors[cls_id]
                label = f"{classes[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thick)
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, box_font, box_font_thick)
                lbl_h = th + base + int(6 * box_s)
                ly = y1 if y1 - lbl_h >= 0 else y1 + lbl_h  # flip inside box if it would leave the frame
                cv2.rectangle(frame, (x1, ly - lbl_h), (x1 + tw + int(6 * box_s), ly), color, -1)
                cv2.putText(frame, label, (x1 + int(3 * box_s), ly - base - int(2 * box_s)),
                            cv2.FONT_HERSHEY_SIMPLEX, box_font, (255, 255, 255), box_font_thick, cv2.LINE_AA)

        if active_hazards:
            dispatcher.trigger_alert(active_hazards, frame, max_conf_in_frame)

        # --- Build Real-Time HUD (drawn per-output at the right size) ---
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        if active_hazards:
            status_str = f"STATUS: HAZARD CONFIRMED ({' + '.join(active_hazards)})"
            status_col = (0, 0, 255)
        else:
            status_str = "STATUS: MONITORING (CLEAR)"
            status_col = (0, 255, 0)

        cd_rem = dispatcher.cooldown_remaining()
        cd_text = f"Alert Cooldown: {cd_rem:.0f}s" if cd_rem > 0 else "Alert System: READY"
        cd_col = (0, 165, 255) if cd_rem > 0 else (180, 255, 180)

        hud = [
            (status_str, status_col, 0.75, 2),
            (f"FPS: {fps:.1f} | Conf: {conf_thres:.2f} | Buffer: Smoke({sum(smoke_buffer)}/{buffer_size}) "
             f"Fire({sum(fire_buffer)}/{buffer_size})", (220, 220, 220), 0.52, 1),
            (cd_text, cd_col, 0.50, 1),
        ]

        if out_writer is not None:
            rec_frame = frame.copy()
            draw_hud(rec_frame, hud, scale=max(1.0, rec_frame.shape[0] / 720.0))
            out_writer.write(rec_frame)

        if gui and gui.available:
            keep_going = gui.show(frame, hud)
            if not keep_going:
                break
        else:
            if frame_idx % 20 == 0 or frame_idx == total_frames:
                prog = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
                print(f"⏳ Frame {prog} | Speed: {fps:.1f} FPS | {status_str}", end="\r")

    cap.release()
    if out_writer is not None:
        out_writer.release()
        print(f"\n✅ Annotated video saved to: {output_video_path}")
    if gui:
        gui.close()
    print("👋 Deployment session closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portable Smoke & Fire YOLO Deployment Script")
    parser.add_argument("--model", type=str, default=None, help="Optional custom path to ONNX or PT weights")
    parser.add_argument("--source", type=str, default=None, help="Video path, webcam index (0), or RTSP URL.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Minimum confidence threshold")
    parser.add_argument("--buffer", type=int, default=DEFAULT_BUFFER, help="Temporal rolling window size (frames)")
    parser.add_argument("--hits", type=int, default=DEFAULT_HITS, help="Required detections inside buffer to trigger alarm")
    parser.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN, help="Cooldown seconds between notification alerts")
    parser.add_argument("--save", action="store_true", default=DEFAULT_SAVE_VIDEO, help="Save processed output video")
    parser.add_argument("--enable-alerts", action="store_true", help="Enable live Telegram/Webhook API calls")
    parser.add_argument("--no-gui", action="store_true", help="Run headless without opening GUI window")

    def _parse_size(value):
        try:
            w, h = (int(v) for v in value.lower().split("x"))
            if w <= 0 or h <= 0:
                raise ValueError
            return (w, h)
        except ValueError:
            raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, e.g. 1280x720")

    parser.add_argument("--display-size", type=_parse_size, default=None,
                        help="Initial window size, e.g. 1280x720 (default: fit ~80%% of the screen)")
    parser.add_argument("--fullscreen", action="store_true", help="Start the live window in fullscreen")
    args = parser.parse_args()

    run_deployment(
        model_path=args.model,
        source=args.source,
        conf_thres=args.conf,
        buffer_size=args.buffer,
        required_hits=args.hits,
        cooldown=args.cooldown,
        save_output=args.save,
        enable_alerts=args.enable_alerts,
        no_gui=args.no_gui,
        display_size=args.display_size,
        fullscreen=args.fullscreen
    )
