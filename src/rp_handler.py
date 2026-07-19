"""
RunPod serverless handler for Hebrew transcription worker.
"""

import time

import runpod
from runpod.serverless.utils.rp_validator import validate

import models
import predict
from diarization import run_diarization, align_speakers, count_speakers
from rp_schema import INPUT_VALIDATIONS
from utils import download_audio, cleanup_file
from webhook import send_webhook

# Load models at startup (single model, fast load)
models.setup()


def handler(job):
    """
    Main handler for Hebrew transcription jobs.

    Input schema:
        audio_url (str, required): URL to audio file or signed URL.
        audio_url_2 (str, optional): Second audio file for dual-channel mode —
            each file is one party's side of the same call, recorded from the
            same start instant. Segments from file 1 are labeled SPEAKER_00 and
            file 2 SPEAKER_01, merged by timestamp. Pyannote diarization is
            skipped (channel separation already identifies the speakers).
        diarize (bool, optional): Enable speaker diarization. Default: false.
        word_timestamps (bool, optional): Return word-level timestamps. Default: false.
        webhook_url (str, optional): URL to POST results to when done.
        webhook_secret (str, optional): Shared secret for HMAC signing.
        job_id (str, optional): Caller's reference ID for correlation.
    """
    start_time = time.time()
    job_input = job["input"]

    # Validate input
    validation = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validation:
        return _handle_error(
            job_input, start_time, "VALIDATION_ERROR",
            str(validation["errors"])
        )
    job_input = validation["validated_input"]

    audio_url = job_input["audio_url"]
    audio_url_2 = job_input.get("audio_url_2")
    diarize = job_input["diarize"]
    word_timestamps = job_input["word_timestamps"]
    webhook_url = job_input.get("webhook_url")
    webhook_secret = job_input.get("webhook_secret")
    job_id = job_input.get("job_id")

    audio_path = None
    audio_path_2 = None
    try:
        # Download audio
        try:
            audio_path = download_audio(audio_url)
            if audio_url_2:
                audio_path_2 = download_audio(audio_url_2)
        except Exception as e:
            return _handle_error(
                job_input, start_time, "AUDIO_DOWNLOAD_FAILED",
                f"Could not download audio from URL: {e}"
            )

        num_speakers = None

        if audio_path_2:
            # Dual-channel: one file per speaker, so pyannote is unnecessary
            # even when diarize=true — the channel split is the diarization.
            try:
                result = predict.transcribe_dual(
                    audio_path, audio_path_2,
                    word_timestamps=word_timestamps,
                )
                num_speakers = 2
            except Exception as e:
                return _handle_error(
                    job_input, start_time, "TRANSCRIPTION_FAILED",
                    f"Transcription error: {e}"
                )
        else:
            # Run transcription
            try:
                result = predict.transcribe(
                    audio_path=audio_path,
                    word_timestamps=word_timestamps,
                    diarize=diarize,
                )
            except Exception as e:
                return _handle_error(
                    job_input, start_time, "TRANSCRIPTION_FAILED",
                    f"Transcription error: {e}"
                )

            # Run diarization if requested
            if diarize:
                try:
                    diarization = run_diarization(audio_path)
                    num_speakers = count_speakers(diarization)
                    result["segments"] = align_speakers(
                        result["segments"], diarization
                    )
                except Exception as e:
                    return _handle_error(
                        job_input, start_time, "DIARIZATION_FAILED",
                        f"Diarization error: {e}"
                    )

        # Build response
        response = {
            "segments": result["segments"],
            "language": result["language"],
            "duration": result["duration"],
        }
        if num_speakers is not None:
            response["num_speakers"] = num_speakers

        # Send webhook if configured
        if webhook_url:
            processing_time_ms = int((time.time() - start_time) * 1000)
            webhook_payload = {
                "job_id": job_id,
                "status": "completed",
                "result": response,
                "metadata": {
                    "model": "hebrew",
                    "diarize": diarize,
                    "processing_time_ms": processing_time_ms,
                },
            }
            send_webhook(webhook_url, webhook_payload, webhook_secret)

        return response

    finally:
        if audio_path:
            cleanup_file(audio_path)
        if audio_path_2:
            cleanup_file(audio_path_2)


def _handle_error(job_input, start_time, code, message):
    """Build error response and optionally send webhook.

    The returned `error` must be a STRING — RunPod's SDK only marks the job
    FAILED (and preserves the message in /status) for string errors; a nested
    dict gets silently dropped and the job reports COMPLETED with no output.
    """
    print(f"ERROR [{code}]: {message}", flush=True)

    error_response = {
        "error": f"{code}: {message}",
    }

    webhook_url = job_input.get("webhook_url")
    if webhook_url:
        processing_time_ms = int((time.time() - start_time) * 1000)
        webhook_payload = {
            "job_id": job_input.get("job_id"),
            "status": "failed",
            "error": {"code": code, "message": message},
            "metadata": {
                "model": "hebrew",
                "diarize": job_input.get("diarize", False),
                "processing_time_ms": processing_time_ms,
            },
        }
        send_webhook(
            webhook_url, webhook_payload,
            job_input.get("webhook_secret")
        )

    return error_response


runpod.serverless.start({"handler": handler})
