"""
Utility functions for audio download and temp file management.
"""

import os
import subprocess
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


def convert_to_wav(audio_path: str) -> str:
    """
    Decode audio to 16 kHz mono PCM WAV via ffmpeg.

    pyannote batches fixed 10s windows (160000 samples @ 16 kHz) and crashes
    with "Sizes of tensors must match except in dimension 0" when compressed
    containers (mp3/m4a/webm, especially VBR) declare a duration that differs
    from the actual decoded sample count. Decoding to plain PCM up front makes
    the declared and actual lengths agree.

    Returns:
        Path to a new temporary .wav file. Caller owns cleanup.

    Raises:
        RuntimeError: If ffmpeg fails to decode the input.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-vn",  # drop video streams (mp4/webm inputs)
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cleanup_file(tmp.name)
        stderr_tail = result.stderr.strip().splitlines()[-5:]
        raise RuntimeError(
            f"ffmpeg failed to decode audio ({result.returncode}): "
            + " | ".join(stderr_tail)
        )

    return tmp.name


def cleanup_file(path: str):
    """Remove a temporary file if it exists."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
