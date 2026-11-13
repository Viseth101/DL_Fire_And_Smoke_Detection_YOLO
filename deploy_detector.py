"""CLI entry point for portable smoke and fire detection."""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

from src.config import (
    DEFAULT_BUFFER,
    DEFAULT_CONF,
    DEFAULT_COOLDOWN,
    DEFAULT_HITS,
    DEFAULT_SAVE_VIDEO,
    DEFAULT_VIDEO_PATH,
    OUTPUT_VIDEOS_DIR,
    SAMPLE_VIDEOS_DIR,
)
from src.detector import (
    LatestFrameCapture,
    TemporalBuffer,
    YoloDetector,
    onnx_has_cuda,
    resolve_model_path,
    select_device,
)
from src.gui import LiveVideoWindow, draw_hud
from src.notifier import AlertDispatcher
# 4. MAIN REAL-TIME INFERENCE PIPELINE
# =====================================================================
def select_source_interactively():
    source_dir = SAMPLE_VIDEOS_DIR
    print("\n" + "=" * 55)
    print("SMOKE & FIRE REAL-TIME DETECTION SYSTEM")
    print("=" * 55)
    print("Select Input Source:")
    print("  [1] Video File (.mp4, .avi, .mkv)")
    print("  [2] Live USB / Integrated Webcam")
    print("  [3] IP Camera / Phone Stream (RTSP or HTTP URL)")
    choice = input("\nEnter option (1/2/3): ").strip()

    if choice == "1":
        vids = []
        if source_dir.exists():
            vids = sorted(
                f.name for f in source_dir.iterdir()
                if f.suffix.lower() in (".mp4", ".avi", ".mkv")
            )
            if vids:
                print(f"\nAvailable videos in '{SAMPLE_VIDEOS_DIR}':")
                for idx, v in enumerate(vids, 1):
                    print(f"   ({idx}) {v}")
        user_in = input("\nEnter video number, filename, or path [Press Enter for sample_fire3.mp4]: ").strip().strip("\"'")
        if not user_in:
            return DEFAULT_VIDEO_PATH
        if vids and user_in.isdigit():
            idx = int(user_in) - 1
            if 0 <= idx < len(vids):
                return source_dir / vids[idx]
        candidate_in_source = os.path.join(source_dir, user_in)
        if Path(candidate_in_source).exists():
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
    conf_thres=DEFAULT_CONF,
    buffer_size=DEFAULT_BUFFER,
    required_hits=DEFAULT_HITS,
    cooldown=DEFAULT_COOLDOWN,
    save_output=DEFAULT_SAVE_VIDEO,
    enable_alerts=False,
    no_gui=False,
    display_size=None,
    fullscreen=False,
    device="auto",
    speed=1.0,
    detector=None,
):
    # 1. Let the user pick the source first, before any heavy loading happens.
    if source is None:
        source = select_source_interactively()
    elif isinstance(source, str) and source.isdigit():
        source = int(source)
    elif isinstance(source, str):
        source = source.strip("\"'")
        alt_path = SAMPLE_VIDEOS_DIR / source
        if not os.path.exists(source) and alt_path.exists():
            source = alt_path

    source = str(source) if isinstance(source, Path) else source
    print(f"Opening video source: {source}")
    cap = LatestFrameCapture(source)
    if not cap.isOpened():
        sys.exit(f"\nError: Unable to open video source '{source}'. Check path or camera connection.")

    # 2. Load and warm the model only for the first session. Interactive mode
    # reuses this detector when the user selects another source.
    if detector is None:
        device, device_label = select_device(device)
        on_gpu = device != "cpu"
        resolved_model_path = resolve_model_path(model_path, prefer_pt=on_gpu)

        # ONNX on GPU needs onnxruntime-gpu; without it ultralytics would silently run on CPU.
        if on_gpu and resolved_model_path.suffix.lower() == ".onnx" and not onnx_has_cuda():
            print("Warning: ONNX model selected but onnxruntime has no CUDA provider, so it runs on CPU. "
                  "Use the .pt weights for GPU speed.")
            device, device_label, on_gpu = "cpu", "CPU", False

        print(f"Loading model: {resolved_model_path}")
        print(f"Inference device: {device_label}")
        detector = YoloDetector(resolved_model_path, device=device, confidence=conf_thres)

        if on_gpu:
            print("Preparing GPU (one-time, can take a few seconds)...")
            t0 = time.perf_counter()
            detector.warm_up()
            print(f"GPU ready in {time.perf_counter() - t0:.1f}s")
    else:
        device_label = detector.device_label
        on_gpu = detector.on_gpu

    gui = None if no_gui else LiveVideoWindow(
        "Smoke & Fire Detection HUD (Q: Quit | P: Pause | F/F11: Fullscreen | +/-: Speed | M: Max)",
        display_size=display_size,
        fullscreen=fullscreen,
        speed=speed,
    )
    if gui and gui.available:
        print("Live Popup Window started! Q = Quit, P = Pause, F/F11 = Fullscreen, Esc = exit fullscreen,")
        print("   +/- = playback speed, 0 = reset to 1x, M = toggle max speed.")
    else:
        save_output = True

    out_writer = None
    output_dir = OUTPUT_VIDEOS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source, str) and os.path.isfile(source):
        src_name = os.path.splitext(os.path.basename(source))[0]
        output_video_path = output_dir / f"{src_name}_detected.mp4"
    else:
        output_video_path = output_dir / "live_stream_detected.mp4"

    if save_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
        print(f"Recording annotated output video to: {output_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temporal_buffer = TemporalBuffer(buffer_size, required_hits)
    dispatcher = AlertDispatcher(cooldown_seconds=cooldown, enable_alerts=enable_alerts)

    colors = {0: (180, 180, 180), 1: (0, 0, 255)}  # BGR: Smoke=Gray, Fire=Red
    classes = {0: "SMOKE", 1: "FIRE"}

    # --- Playback pacing: video files play at their own FPS (x speed) when the model is faster.
    # Live webcams/streams are already real-time, so they are never paced.
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    is_file = isinstance(source, str) and os.path.isfile(source)
    pace_playback = bool(gui and gui.available and is_file and 1.0 <= src_fps <= 240.0)
    next_due = None

    prev_time = time.perf_counter()
    infer_fps = disp_fps = 0.0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of stream reached.")
            break

        frame_idx += 1
        t_infer = time.perf_counter()
        detections = detector.predict(frame)
        inst = 1.0 / max(time.perf_counter() - t_infer, 1e-5)
        infer_fps = inst if infer_fps == 0 else 0.9 * infer_fps + 0.1 * inst

        current_smoke = False
        current_fire = False
        max_conf_in_frame = 0.0
        for cls_id, conf, x1, y1, x2, y2 in detections:

            if cls_id == 0:
                current_smoke = True
            elif cls_id == 1:
                current_fire = True

            max_conf_in_frame = max(max_conf_in_frame, conf)
        smoke_confirmed, fire_confirmed = temporal_buffer.update(current_smoke, current_fire)

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
        curr_time = time.perf_counter()
        inst = 1.0 / max(curr_time - prev_time, 1e-5)
        disp_fps = inst if disp_fps == 0 else 0.9 * disp_fps + 0.1 * inst
        prev_time = curr_time

        if not pace_playback:
            speed_txt = "Playback: live" if not is_file else "Playback: max"
        elif gui.max_speed:
            speed_txt = "Playback: MAX (unpaced)"
        else:
            target = src_fps * gui.speed
            speed_txt = f"Playback: {gui.speed:g}x ({target:.0f} FPS)"
            if infer_fps < target * 0.95:
                speed_txt += " - model-limited"

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
            (f"Model: {infer_fps:.0f} FPS ({'GPU' if on_gpu else 'CPU'}) | Display: {disp_fps:.1f} FPS | {speed_txt}",
             (220, 220, 220), 0.52, 1),
            (f"Conf: {conf_thres:.2f} | Buffer: Smoke({sum(temporal_buffer.smoke)}/{buffer_size}) "
             f"Fire({sum(temporal_buffer.fire)}/{buffer_size})", (220, 220, 220), 0.52, 1),
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

            if pace_playback and not gui.max_speed:
                now = time.perf_counter()
                if next_due is None:
                    next_due = now
                next_due += 1.0 / (src_fps * gui.speed)
                delay = next_due - now
                if delay > 0:
                    gui.wait(delay)
                elif delay < -0.25:
                    # Model can't keep up (or we just resumed from pause): resync instead of bursting.
                    next_due = now
            else:
                next_due = None
        else:
            if frame_idx % 20 == 0 or frame_idx == total_frames:
                prog = f"{frame_idx}/{total_frames}" if total_frames > 0 else f"{frame_idx}"
                print(f"Frame {prog} | Model: {infer_fps:.1f} FPS ({device_label}) | {status_str}", end="\r")

    cap.release()
    if out_writer is not None:
        out_writer.release()
        print(f"\nAnnotated video saved to: {output_video_path}")
    if gui:
        gui.close()
    print("Deployment session closed cleanly.")
    return detector


def run_cli(args):
    """Run one-shot CLI mode or a persistent interactive source-selection session."""
    detector = None
    while True:
        detector = run_deployment(
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
            fullscreen=args.fullscreen,
            device=args.device,
            speed=args.speed,
            detector=detector,
        )

        if args.source is not None:
            return
        try:
            again = input("\nSelect another source? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if again in ("n", "no"):
            return


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
    parser.add_argument("--device", type=str, default="auto",
                        help="Inference device: auto (GPU if available), cpu, or a GPU index like 0")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Video playback speed multiplier (1.0 = source FPS). 0 = as fast as the model can run")
    args = parser.parse_args()

    run_cli(args)
