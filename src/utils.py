"""
Utility functions for audio download and temp file management.
"""

import os
import tempfile
from urllib.parse import urlparse

import requests


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}


def download_audio(url: str) -> str:
    """
    Download audio from a URL to a temporary file.

    Supports signed URLs (R2, S3) and common audio formats.

    Args:
        url: URL to the audio file.

    Returns:
        Path to the downloaded temporary file.

    Raises:
        ValueError: If the audio format is not supported.
        RuntimeError: If the download fails.
    """
    # Determine file extension from URL path (ignoring query params)
    parsed = urlparse(url)
    path = parsed.path
    ext = os.path.splitext(path)[1].lower()

    # Default to .wav if no extension detected (common with signed URLs)
    if not ext or ext not in SUPPORTED_EXTENSIONS:
        ext = ".wav"

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download audio from URL: {e}") from e

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.close()
    except Exception as e:
        tmp.close()
        cleanup_file(tmp.name)
        raise RuntimeError(f"Error writing audio to temp file: {e}") from e

    return tmp.name


def cleanup_file(path: str):
    """Remove a temporary file if it exists."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
