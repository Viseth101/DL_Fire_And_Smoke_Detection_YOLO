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
class LiveVideoWindow:
    """
    Opens a live GUI window using Tkinter + PIL on Windows, macOS, and WSL2.
    """
    def __init__(self, title="Smoke & Fire Detection HUD"):
        self.available = False
        self.quit_requested = False
        self.paused = False
        try:
            import tkinter as tk
            from PIL import ImageTk
            self.tk = tk
            self.ImageTk = ImageTk

            self.root = tk.Tk()
            self.root.title(title)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.root.bind("<q>", lambda e: self._on_close())
            self.root.bind("<Q>", lambda e: self._on_close())
            self.root.bind("<Escape>", lambda e: self._on_close())
            self.root.bind("<p>", lambda e: self._toggle_pause())
            self.root.bind("<P>", lambda e: self._toggle_pause())

            self.label = tk.Label(self.root, bg="black")
            self.label.pack(fill=tk.BOTH, expand=True)
            self.available = True
        except Exception as e:
            print(f"\n⚠️ Live GUI window unavailable ({e}). Continuing in headless recording mode...\n")
            self.available = False

    def _on_close(self):
        self.quit_requested = True

    def _toggle_pause(self):
        self.paused = not self.paused

    def show(self, frame_bgr):
        if not self.available or self.quit_requested:
            return False
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = self.ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            self.root.update_idletasks()
            self.root.update()

            while self.paused and not self.quit_requested:
                self.root.update()
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
    3. Checks local training folder and automatically copies weights into 'Video Deploy/Models/'
       so the project folder is immediately portable for other computers!
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


def run_deployment(
    model_path=None,
    source=None,
    conf_thres=0.45,
    buffer_size=5,
    required_hits=4,
    cooldown=60,
    save_output=True,
    enable_alerts=False,
    no_gui=False
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

    gui = None if no_gui else LiveVideoWindow("Smoke & Fire Detection HUD (Press 'Q' to Quit, 'P' to Pause)")
    if gui and gui.available:
        print("✅ Live Popup Window started! Press 'Q' inside the window to quit, 'P' to pause.")
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

        for (cls_id, conf, x1, y1, x2, y2) in detections:
            if (cls_id == 0 and smoke_confirmed) or (cls_id == 1 and fire_confirmed):
                color = colors[cls_id]
                label = f"{classes[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (x1, y1 - 24), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if active_hazards:
            dispatcher.trigger_alert(active_hazards, frame, max_conf_in_frame)

        # --- Draw Real-Time HUD Overlay ---
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        if active_hazards:
            status_str = f"STATUS: HAZARD CONFIRMED ({' + '.join(active_hazards)})"
            status_col = (0, 0, 255)
        else:
            status_str = "STATUS: MONITORING (CLEAR)"
            status_col = (0, 255, 0)

        cv2.putText(frame, status_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_col, 2)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Conf: {conf_thres:.2f} | Buffer: Smoke({sum(smoke_buffer)}/{buffer_size}) Fire({sum(fire_buffer)}/{buffer_size})",
            (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1
        )

        cd_rem = dispatcher.cooldown_remaining()
        cd_text = f"Alert Cooldown: {cd_rem:.0f}s" if cd_rem > 0 else "Alert System: READY"
        cd_col = (0, 165, 255) if cd_rem > 0 else (180, 255, 180)
        cv2.putText(frame, cd_text, (15, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.50, cd_col, 1)

        if out_writer is not None:
            out_writer.write(frame)

        if gui and gui.available:
            keep_going = gui.show(frame)
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
    parser.add_argument("--conf", type=float, default=0.45, help="Minimum confidence threshold")
    parser.add_argument("--buffer", type=int, default=5, help="Temporal rolling window size (frames)")
    parser.add_argument("--hits", type=int, default=4, help="Required detections inside buffer to trigger alarm")
    parser.add_argument("--cooldown", type=int, default=60, help="Cooldown seconds between notification alerts")
    parser.add_argument("--no-save", action="store_true", help="Disable saving output video")
    parser.add_argument("--enable-alerts", action="store_true", help="Enable live Telegram/Webhook API calls")
    parser.add_argument("--no-gui", action="store_true", help="Run headless without opening GUI window")
    args = parser.parse_args()

    run_deployment(
        model_path=args.model,
        source=args.source,
        conf_thres=args.conf,
        buffer_size=args.buffer,
        required_hits=args.hits,
        cooldown=args.cooldown,
        save_output=not args.no_save,
        enable_alerts=args.enable_alerts,
        no_gui=args.no_gui
    )
