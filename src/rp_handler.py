"""
RunPod serverless handler for Hebrew transcription worker.
"""

import runpod

print(">>> Handler module loaded", flush=True)

_initialized = False


def _lazy_init():
    """Import heavy modules and load models on first request."""
    global _initialized
    if _initialized:
        return
    print(">>> Initializing worker...", flush=True)

    import models
    models.setup()

    _initialized = True
    print(">>> Worker ready", flush=True)


def handler(job):
    """
    Main handler for Hebrew transcription jobs.

    Input schema:
        audio_url (str, required): URL to audio file or signed URL.
        diarize (bool, optional): Enable speaker diarization. Default: false.
        word_timestamps (bool, optional): Return word-level timestamps. Default: false.
        webhook_url (str, optional): URL to POST results to when done.
        webhook_secret (str, optional): Shared secret for HMAC signing.
        job_id (str, optional): Caller's reference ID for correlation.
    """
    import time
    from runpod.serverless.utils.rp_validator import validate
    import predict
    from diarization import run_diarization, align_speakers, count_speakers
    from rp_schema import INPUT_VALIDATIONS
    from utils import download_audio, cleanup_file
    from webhook import send_webhook

    _lazy_init()
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
    diarize = job_input["diarize"]
    word_timestamps = job_input["word_timestamps"]
    webhook_url = job_input.get("webhook_url")
    webhook_secret = job_input.get("webhook_secret")
    job_id = job_input.get("job_id")

    audio_path = None
    try:
        # Download audio
        try:
            audio_path = download_audio(audio_url)
        except Exception as e:
            return _handle_error(
                job_input, start_time, "AUDIO_DOWNLOAD_FAILED",
                f"Could not download audio from URL: {e}"
            )

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
        num_speakers = None
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


def _handle_error(job_input, start_time, code, message):
    """Build error response and optionally send webhook."""
    print(f"ERROR [{code}]: {message}", flush=True)

    error_response = {
        "error": {
            "code": code,
            "message": message,
        }
    }

    webhook_url = job_input.get("webhook_url")
    if webhook_url:
        import time
        from webhook import send_webhook
        processing_time_ms = int((time.time() - start_time) * 1000)
        webhook_payload = {
            "job_id": job_input.get("job_id"),
            "status": "failed",
            "error": error_response["error"],
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
