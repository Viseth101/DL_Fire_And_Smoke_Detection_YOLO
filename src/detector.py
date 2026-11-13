"""YOLO inference and temporal persistence filtering."""

from collections import deque
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from .config import MODEL_ONNX_PATH, MODEL_PT_PATH


class LatestFrameCapture:
    """Drop-in capture wrapper that drains live-stream buffers in a worker thread."""

    LIVE_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")

    def __init__(self, source):
        self.source = source
        self.is_live = self._is_live_source(source)
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.ret = False
        self.frame_id = 0
        self._last_read_id = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.new_frame_event = threading.Event()
        self._thread = None

        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.ret, self.frame = self.cap.read()
            if self.ret:
                self.frame_id = 1
                self.new_frame_event.set()
            print("Capture mode: Zero-Latency Live Stream Mode (Threaded Buffer Drain)")
            if self.cap.isOpened():
                self._thread = threading.Thread(
                    target=self._drain_live_stream,
                    name="latest-frame-capture",
                    daemon=True,
                )
                self._thread.start()
        else:
            print("Capture mode: Sequential File Mode")

    @classmethod
    def _is_live_source(cls, source) -> bool:
        return isinstance(source, int) or str(source).lower().startswith(cls.LIVE_SCHEMES)

    def _drain_live_stream(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.ret = True
                    self.frame = frame
                    self.frame_id += 1
                self.new_frame_event.set()
            else:
                with self._lock:
                    self.ret = False
                time.sleep(0.01)

    def read(self):
        if not self.is_live:
            return self.cap.read()

        deadline = time.monotonic() + 0.5
        while not self._stop_event.is_set():
            with self._lock:
                if self.ret and self.frame is not None and self.frame_id > self._last_read_id:
                    self._last_read_id = self.frame_id
                    return True, self.frame.copy()

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, None
            self.new_frame_event.wait(timeout=min(0.05, remaining))
            self.new_frame_event.clear()
        return False, None

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def release(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


class TemporalBuffer:
    """Track whether each hazard has appeared in the recent K frames."""

    def __init__(self, buffer_size: int, required_hits: int):
        if buffer_size < 1:
            raise ValueError("buffer_size must be at least 1")
        if not 1 <= required_hits <= buffer_size:
            raise ValueError("required_hits must be between 1 and buffer_size")
        self.required_hits = required_hits
        self.smoke = deque(maxlen=buffer_size)
        self.fire = deque(maxlen=buffer_size)

    def update(self, smoke_detected: bool, fire_detected: bool) -> tuple[bool, bool]:
        self.smoke.append(int(smoke_detected))
        self.fire.append(int(fire_detected))
        return (
            sum(self.smoke) >= self.required_hits,
            sum(self.fire) >= self.required_hits,
        )


class YoloDetector:
    """Load a YOLO model and return normalized detection tuples per frame."""

    def __init__(self, model_path: str | Path, device="auto", confidence: float = 0.10):
        self.model_path = Path(model_path)
        self.device, self.device_label = select_device(device)
        self.confidence = confidence
        self.model = YOLO(str(self.model_path), task="detect")
        self.on_gpu = self.device != "cpu"
        self.predict_kwargs = {
            "imgsz": 640,
            "conf": confidence,
            "verbose": False,
            "device": self.device,
        }
        if self.on_gpu and self.model_path.suffix.lower() == ".pt":
            self.predict_kwargs["quantize"] = 16

    def warm_up(self) -> None:
        """Run one prediction to initialize GPU kernels before processing video."""
        if self.on_gpu:
            self.model.predict(
                np.zeros((640, 640, 3), dtype=np.uint8),
                **self.predict_kwargs,
            )

    def predict(self, frame: np.ndarray) -> list[tuple[int, float, int, int, int, int]]:
        """Return (class, confidence, x1, y1, x2, y2) detections for one frame."""
        result = self.model.predict(frame, **self.predict_kwargs)[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((cls_id, confidence, x1, y1, x2, y2))
        return detections


def select_device(requested="auto"):
    """Return the requested CUDA device when available, otherwise CPU."""
    import torch

    request = str(requested).lower()
    if request == "cpu":
        return "cpu", "CPU"
    if torch.cuda.is_available():
        index = 0 if request in ("auto", "cuda", "gpu") else int(request)
        return index, f"GPU: {torch.cuda.get_device_name(index)}"
    if request != "auto":
        print(f"Requested device '{requested}' is unavailable; falling back to CPU.")
    return "cpu", "CPU"


def onnx_has_cuda() -> bool:
    """Return whether the installed ONNX Runtime exposes CUDA execution."""
    import onnxruntime as ort

    return "CUDAExecutionProvider" in ort.get_available_providers()


def resolve_model_path(requested_path=None, prefer_pt: bool = False) -> Path:
    """Resolve an explicit or repository-local model without machine-specific paths."""
    if requested_path:
        candidate = Path(requested_path).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Requested model does not exist: {candidate}")
    if prefer_pt and MODEL_PT_PATH.exists():
        return MODEL_PT_PATH
    if MODEL_ONNX_PATH.exists():
        return MODEL_ONNX_PATH
    if MODEL_PT_PATH.exists():
        return MODEL_PT_PATH
    raise FileNotFoundError(
        "No model found. Expected models/yolo26s_best.onnx or models/yolo26s_best.pt."
    )
