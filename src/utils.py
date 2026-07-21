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


def get_channel_count(audio_path: str) -> int:
    """Return the number of audio channels via ffprobe (0 if probing fails)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "csv=p=0",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def split_stereo_channels(audio_path: str):
    """
    Split a stereo recording into two 16 kHz mono PCM WAVs (left, right).

    Returns (left_path, right_path), or None when the input is not stereo —
    the caller should fall back to model-based diarization in that case.
    Caller owns cleanup of both files.
    """
    if get_channel_count(audio_path) < 2:
        return None

    left = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    left.close()
    right = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    right.close()

    # aformat forces a plain `mono` layout — channelsplit tags its outputs
    # FL/FR, which makes the WAV muxer emit WAVE_FORMAT_EXTENSIBLE headers
    # that downstream stdlib readers can't parse.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", audio_path,
        "-filter_complex",
        "[0:a]channelsplit=channel_layout=stereo[Lr][Rr];"
        "[Lr]aformat=channel_layouts=mono[L];"
        "[Rr]aformat=channel_layouts=mono[R]",
        "-map", "[L]", "-ar", "16000", "-c:a", "pcm_s16le", left.name,
        "-map", "[R]", "-ar", "16000", "-c:a", "pcm_s16le", right.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cleanup_file(left.name)
        cleanup_file(right.name)
        stderr_tail = result.stderr.strip().splitlines()[-5:]
        raise RuntimeError(
            f"ffmpeg failed to split stereo channels ({result.returncode}): "
            + " | ".join(stderr_tail)
        )

    return left.name, right.name


def channels_nearly_identical(left_path: str, right_path: str,
                              sample_seconds: int = 60) -> bool:
    """
    Detect dual-mono: a "stereo" file whose channels carry the same signal.

    Some PBXs export dual-mono stereo. Splitting such a file would transcribe
    the whole call twice (once per fake speaker), so the caller must fall back
    to model-based diarization when this returns True. Compares up to
    sample_seconds of both channels by normalized correlation.

    Samples are decoded via ffmpeg rather than the stdlib wave module — the
    latter rejects WAVE_FORMAT_EXTENSIBLE headers (format 0xFFFE), which some
    ffmpeg builds emit for channel-split output.
    """
    import numpy as np

    def read_samples(path):
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-t", str(sample_seconds),
            "-i", path,
            "-f", "s16le", "-ac", "1", "-ar", "16000", "-",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {path} for dual-mono check"
            )
        return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)

    a = read_samples(left_path)
    b = read_samples(right_path)
    n = min(len(a), len(b))
    if n == 0:
        return True
    a, b = a[:n], b[:n]

    # Silent channel (one-sided call leg) is NOT dual-mono
    if np.abs(a).max() < 1 or np.abs(b).max() < 1:
        return False

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return True
    correlation = float(np.dot(a, b) / denom)
    return correlation > 0.98


def cleanup_file(path: str):
    """Remove a temporary file if it exists."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
