"""Portable configuration and path helpers for smoke and fire detection."""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
SAMPLE_VIDEOS_DIR = ROOT_DIR / "data" / "sample_videos"
OUTPUTS_DIR = ROOT_DIR / "data" / "outputs"
OUTPUT_VIDEOS_DIR = OUTPUTS_DIR / "videos"
ALERTS_DIR = OUTPUTS_DIR / "alerts_log"

MODEL_ONNX_PATH = MODELS_DIR / "yolo26s_best.onnx"
MODEL_PT_PATH = MODELS_DIR / "yolo26s_best.pt"
DEFAULT_VIDEO_PATH = SAMPLE_VIDEOS_DIR / "sample_fire3.mp4"

DEFAULT_CONF = 0.10
DEFAULT_BUFFER = 5
DEFAULT_HITS = 4
DEFAULT_COOLDOWN = 60
DEFAULT_SAVE_VIDEO = False

load_dotenv(ROOT_DIR / ".env")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_SNS_TOPIC_ARN = os.getenv("AWS_SNS_TOPIC_ARN")


def get_env(name: str, default: str | None = None) -> str | None:
    """Return an environment value after loading the repository .env file."""
    return os.getenv(name, default)
