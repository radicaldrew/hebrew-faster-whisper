import os
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

CT2_MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"

HF_TOKEN = os.environ.get("HF_TOKEN")


def download_ct2_model():
    """Download CTranslate2 Hebrew whisper model by instantiating it."""
    print(f"Downloading CT2 model: {CT2_MODEL_ID}...")
    WhisperModel(CT2_MODEL_ID, device="cpu", compute_type="int8")
    print("Finished downloading Hebrew model.")


def download_pyannote_pipeline():
    """Download pyannote speaker-diarization-3.1 pipeline."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set, skipping pyannote pipeline download.")
        print("Diarization will not be available at runtime without pre-downloaded models.")
        return

    print("Downloading pyannote/speaker-diarization-3.1 pipeline...")
    Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    print("Finished downloading pyannote pipeline.")


if __name__ == "__main__":
    download_ct2_model()
    download_pyannote_pipeline()
    print("All models downloaded successfully.")
