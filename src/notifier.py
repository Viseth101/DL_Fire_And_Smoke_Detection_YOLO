"""Asynchronous alert delivery with Telegram and cloud/VLM extension hooks."""

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

from .config import ALERTS_DIR, get_env


class AlertDispatcher:
    """Persist confirmed hazards and dispatch notifications without blocking video."""

    def __init__(self, cooldown_seconds: int = 60, enable_alerts: bool = False):
        self.cooldown_seconds = cooldown_seconds
        self.enable_alerts = enable_alerts
        self.last_alert_time = 0.0
        self.telegram_bot_token = get_env("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = get_env("TELEGRAM_CHAT_ID")

    def can_trigger(self) -> bool:
        return time.time() - self.last_alert_time >= self.cooldown_seconds

    def cooldown_remaining(self) -> float:
        return max(0.0, self.cooldown_seconds - (time.time() - self.last_alert_time))

    def trigger_alert(self, hazard_types, frame, max_conf: float) -> bool:
        if not self.can_trigger():
            return False
        self.last_alert_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = (
            f"[HAZARD ALERT] {' & '.join(hazard_types)} DETECTED!\n"
            f"Time: {timestamp}\n"
            f"Peak Confidence: {max_conf:.2f}\n"
            f"Cooldown: {self.cooldown_seconds}s"
        )
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot = ALERTS_DIR / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        if not cv2.imwrite(str(snapshot), frame):
            raise RuntimeError(f"Could not write alert snapshot: {snapshot}")
        threading.Thread(
            target=self._send_payload,
            args=(message, snapshot),
            daemon=True,
        ).start()
        return True

    def _send_payload(self, message: str, image_path: Path) -> None:
        print(f"\n{message}\nSnapshot saved: {image_path}")
        if not self.enable_alerts:
            return
        if self.telegram_bot_token and self.telegram_chat_id:
            try:
                with image_path.open("rb") as image_file:
                    response = requests.post(
                        f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto",
                        data={"chat_id": self.telegram_chat_id, "caption": message},
                        files={"photo": image_file},
                        timeout=10,
                    )
                if response.status_code == 200:
                    print("Telegram notification delivered successfully.")
                else:
                    print(f"Telegram API returned {response.status_code}: {response.text}")
            except (OSError, requests.RequestException) as exc:
                print(f"Failed to send Telegram alert: {exc}")

        self._upload_to_s3(image_path)
        self._publish_sns_or_webhook(message)
        self._verify_with_vlm(image_path)

    def _upload_to_s3(self, image_path: Path) -> None:
        """Hook for an optional S3 upload; intentionally inactive until configured."""
        return None

    def _publish_sns_or_webhook(self, message: str) -> None:
        """Hook for optional SNS or generic webhook delivery."""
        return None

    def _verify_with_vlm(self, image_path: Path) -> None:
        """Hook for optional Gemini Flash or another VLM verification step."""
        return None
