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
    """Pre-cache pyannote speaker-diarization-3.1 (gated).

    Baking the model into the image is the point of passing HF_TOKEN at
    build time — it removes the runtime dependency on huggingface.co at
    worker cold start (a transient Hub outage there poisons the worker).
    So if a token was provided and the fetch fails, FAIL THE BUILD instead
    of silently shipping an image that must download at runtime.

    Only when no token is given at all do we skip: models.py then downloads
    at worker start using the endpoint's runtime HF_TOKEN env var.
    """
    if not HF_TOKEN:
        print("WARNING: Skipping pyannote pre-cache (no HF_TOKEN); "
              "diarization will depend on a runtime download at cold start.")
        return
    from pyannote.audio import Pipeline
    print("Downloading pyannote/speaker-diarization-3.1 pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    if pipeline is None:
        # from_pretrained returns None instead of raising on some failures
        raise RuntimeError(
            "pyannote pre-cache failed: Pipeline.from_pretrained returned None. "
            "Check that the HF_TOKEN has accepted the gated-repo terms for both "
            "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0."
        )
    print("Finished downloading pyannote pipeline.")


if __name__ == "__main__":
    download_ct2_model()  # required — fail the build if the public model won't fetch
    download_pyannote_pipeline()  # required when HF_TOKEN is set
    print("Model fetch complete.")
