"""Asynchronous Telegram, S3, and SNS incident delivery."""

import threading
import time
from datetime import datetime
from pathlib import Path

import boto3
import cv2
import requests
from botocore.exceptions import BotoCoreError, ClientError

from .config import (
    ALERTS_DIR,
    AWS_REGION,
    AWS_S3_BUCKET,
    AWS_SNS_TOPIC_ARN,
    get_env,
)


class AlertDispatcher:
    """Persist confirmed hazards and dispatch notifications without blocking video."""

    def __init__(self, cooldown_seconds: int = 60, enable_alerts: bool = False):
        self.cooldown_seconds = cooldown_seconds
        self.enable_alerts = enable_alerts
        self.last_alert_time = 0.0
        self.telegram_bot_token = get_env("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = get_env("TELEGRAM_CHAT_ID")
        self.aws_region = AWS_REGION
        self.aws_s3_bucket = AWS_S3_BUCKET
        self.aws_sns_topic_arn = AWS_SNS_TOPIC_ARN
        self._aws_session = None
        self._s3_client = None
        self._sns_client = None

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
            args=(message, snapshot, hazard_types, timestamp, max_conf),
            daemon=True,
        ).start()
        return True

    def _send_payload(
        self,
        message: str,
        image_path: Path,
        hazard_types,
        timestamp: str,
        max_conf: float,
    ) -> None:
        print(f"\n{message}\nSnapshot saved: {image_path}")
        if not self.enable_alerts:
            return

        s3_uri, presigned_url = self._upload_to_s3(image_path)
        telegram_caption = message
        if presigned_url and self.telegram_bot_token and self.telegram_chat_id:
            telegram_caption += f"\nS3 URL (24h): {presigned_url}"
        self._send_telegram(telegram_caption, image_path)
        self._publish_to_sns(
            hazard_types=hazard_types,
            timestamp=timestamp,
            max_conf=max_conf,
            s3_uri=s3_uri,
            presigned_url=presigned_url,
        )

    def _send_telegram(self, caption: str, image_path: Path) -> None:
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return
        try:
            with image_path.open("rb") as image_file:
                response = requests.post(
                    f"https://api.telegram.org/bot{self.telegram_bot_token}/sendPhoto",
                    data={"chat_id": self.telegram_chat_id, "caption": caption},
                    files={"photo": image_file},
                    timeout=10,
                )
            if response.status_code == 200:
                print("Telegram notification delivered successfully.")
            else:
                print(f"Telegram API returned {response.status_code}: {response.text}")
        except (OSError, requests.RequestException) as exc:
            print(f"Failed to send Telegram alert: {exc}")

    def _get_aws_session(self):
        if not self.enable_alerts:
            return None
        if self._aws_session is None:
            try:
                self._aws_session = boto3.session.Session(region_name=self.aws_region)
            except (BotoCoreError, ClientError) as exc:
                print(f"AWS session initialization failed: {exc}")
                return None
        return self._aws_session

    def _upload_to_s3(self, image_path: Path) -> tuple[str | None, str | None]:
        """Upload an incident snapshot and return its URI and 24-hour URL."""
        if not self.enable_alerts or not self.aws_s3_bucket:
            return None, None
        try:
            if self._s3_client is None:
                session = self._get_aws_session()
                if session is None:
                    return None, None
                self._s3_client = session.client("s3")

            date_prefix = datetime.now().strftime("%Y-%m-%d")
            object_key = f"incidents/{date_prefix}/{image_path.name}"
            self._s3_client.upload_file(
                str(image_path),
                self.aws_s3_bucket,
                object_key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )
            presigned_url = self._s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.aws_s3_bucket, "Key": object_key},
                ExpiresIn=86400,
            )
            s3_uri = f"s3://{self.aws_s3_bucket}/{object_key}"
            print(f"S3 snapshot uploaded: {s3_uri}")
            return s3_uri, presigned_url
        except (ClientError, BotoCoreError, OSError) as exc:
            print(f"S3 snapshot upload failed: {exc}")
            return None, None

    def _publish_to_sns(
        self,
        hazard_types,
        timestamp: str,
        max_conf: float,
        s3_uri: str | None,
        presigned_url: str | None,
    ) -> None:
        """Publish the incident metadata and S3 links to the configured SNS topic."""
        if not self.enable_alerts or not self.aws_sns_topic_arn:
            return
        try:
            if self._sns_client is None:
                session = self._get_aws_session()
                if session is None:
                    return
                self._sns_client = session.client("sns")

            links = [
                f"S3 Object: {s3_uri or 'unavailable'}",
                f"Presigned URL (24h): {presigned_url or 'unavailable'}",
            ]
            message = (
                f"Hazard types: {', '.join(hazard_types)}\n"
                f"Timestamp: {timestamp}\n"
                f"Peak confidence: {max_conf:.2f}\n"
                + "\n".join(links)
            )
            self._sns_client.publish(
                TopicArn=self.aws_sns_topic_arn,
                Subject="Smoke and Fire Incident Alert",
                Message=message,
            )
            print("SNS incident notification published successfully.")
        except (ClientError, BotoCoreError) as exc:
            print(f"SNS incident notification failed: {exc}")
