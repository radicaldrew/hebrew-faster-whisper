"""
Webhook callback with HMAC-SHA256 signing for Vero Transcription worker.
"""

import hashlib
import hmac
import json
import time

import requests


MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0


def send_webhook(url: str, payload: dict, secret: str = None):
    """
    POST results to a webhook URL with optional HMAC-SHA256 signing.

    Args:
        url: The webhook endpoint URL.
        payload: JSON-serializable dict to send.
        secret: Optional shared secret for HMAC signing.

    Retries up to 3 times with exponential backoff on failure.
    """
    body = json.dumps(payload, ensure_ascii=False)
    headers = {"Content-Type": "application/json"}

    if secret:
        signature = _compute_hmac(body, secret)
        headers["X-Webhook-Signature"] = signature

    backoff = INITIAL_BACKOFF_S
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, data=body, headers=headers, timeout=30)
            if response.status_code < 400:
                print(f"Webhook delivered successfully (attempt {attempt}).")
                return
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"Webhook attempt {attempt} failed: {last_error}")
        except requests.RequestException as e:
            last_error = str(e)
            print(f"Webhook attempt {attempt} error: {last_error}")

        if attempt < MAX_RETRIES:
            time.sleep(backoff)
            backoff *= 2

    print(f"Webhook delivery failed after {MAX_RETRIES} attempts. Last error: {last_error}")


def _compute_hmac(body: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature, returned as hex string."""
    return hmac.new(
        secret.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
