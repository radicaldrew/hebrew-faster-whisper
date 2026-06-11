"""
Model loading for Hebrew transcription worker.

Loads ivrit-ai whisper CT2 model and pyannote diarization pipeline.
"""

import os
import torch
from faster_whisper import WhisperModel
from runpod.serverless.utils import rp_cuda

MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"

# Global model instances
whisper_model = None
diarization_pipeline = None
diarization_load_error = None


def load_whisper_model():
    """Load Hebrew whisper CT2 model into GPU memory."""
    global whisper_model

    device = "cuda" if rp_cuda.is_available() else "cpu"
    compute_type = "float16" if rp_cuda.is_available() else "int8"

    print(f"Loading Hebrew whisper model: {MODEL_ID}...", flush=True)
    whisper_model = WhisperModel(
        MODEL_ID,
        device=device,
        compute_type=compute_type,
    )
    print(f"Hebrew whisper model loaded on {device}.", flush=True)


def load_diarization_pipeline():
    """Load pyannote speaker-diarization-3.1 pipeline."""
    global diarization_pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set. Diarization will not be available.", flush=True)
        return

    try:
        from pyannote.audio import Pipeline

        print("Loading pyannote/speaker-diarization-3.1 pipeline...", flush=True)
        try:
            # pyannote 3.x kwarg
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        except TypeError:
            # pyannote 4.x renamed it to `token`
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
        if rp_cuda.is_available():
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
            print("Diarization pipeline loaded on CUDA.", flush=True)
        else:
            print("Diarization pipeline loaded on CPU.", flush=True)
    except Exception as e:
        global diarization_load_error
        diarization_load_error = f"{type(e).__name__}: {e}"
        print(f"ERROR: Failed to load diarization pipeline: {diarization_load_error}", flush=True)
        diarization_pipeline = None


def get_whisper_model():
    """Get the loaded whisper model."""
    if whisper_model is None:
        raise RuntimeError("Whisper model not loaded.")
    return whisper_model


def get_diarization_pipeline():
    """Get the loaded diarization pipeline."""
    if diarization_pipeline is None:
        detail = diarization_load_error or "HF_TOKEN not set"
        raise RuntimeError(
            f"Diarization pipeline not loaded ({detail}). Ensure HF_TOKEN is "
            "set and pyannote/speaker-diarization-3.1 access is granted."
        )
    return diarization_pipeline


def setup():
    """Initialize all models."""
    print("=== Loading models ===", flush=True)
    load_whisper_model()
    load_diarization_pipeline()
    print("=== All models loaded ===", flush=True)
