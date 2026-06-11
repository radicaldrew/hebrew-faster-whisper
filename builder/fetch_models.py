import os
from huggingface_hub import snapshot_download, login
from faster_whisper import WhisperModel

CT2_MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"

# Empty string (unset build arg) must behave like no token
HF_TOKEN = os.environ.get("HF_TOKEN") or None

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("HuggingFace login successful")
    except Exception as e:
        # Bad/revoked token must not kill the build — public downloads still work
        print(f"WARNING: HuggingFace login failed ({e}); continuing anonymously")
        HF_TOKEN = None
else:
    print("WARNING: HF_TOKEN not set - downloads may be rate-limited")


def download_ct2_model():
    """Download CTranslate2 Hebrew whisper model (public repo; required)."""
    print(f"Downloading CT2 model: {CT2_MODEL_ID}...")
    model_path = snapshot_download(repo_id=CT2_MODEL_ID, token=HF_TOKEN)
    print(f"Model downloaded to: {model_path}")

    print("Verifying model loads correctly...")
    WhisperModel(model_path, device="cpu", compute_type="int8")
    print("Finished downloading Hebrew model.")


def download_pyannote_pipeline():
    """Pre-cache pyannote speaker-diarization-3.1 (gated; best-effort).

    If this is skipped or fails, models.py downloads it at worker start
    using the endpoint's runtime HF_TOKEN env var (~25MB, a few seconds).
    """
    if not HF_TOKEN:
        print("Skipping pyannote pre-cache (no HF_TOKEN); will download at runtime.")
        return
    try:
        from pyannote.audio import Pipeline
        print("Downloading pyannote/speaker-diarization-3.1 pipeline...")
        Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        print("Finished downloading pyannote pipeline.")
    except Exception as e:
        print(f"WARNING: pyannote pre-cache failed ({e}); will download at runtime.")


if __name__ == "__main__":
    download_ct2_model()  # required — fail the build if the public model won't fetch
    download_pyannote_pipeline()  # best-effort
    print("Model fetch complete.")
