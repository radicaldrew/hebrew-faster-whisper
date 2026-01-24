"""
Transcription logic for Hebrew whisper model.
"""

from models import get_whisper_model


def transcribe(audio_path: str, word_timestamps: bool = False,
               diarize: bool = False) -> dict:
    """
    Run Hebrew transcription on an audio file.

    Args:
        audio_path: Path to the audio file.
        word_timestamps: Whether to include word-level timestamps.
        diarize: If True, forces word_timestamps=True for alignment.

    Returns:
        Dict with 'segments', 'language', 'duration'.
    """
    model = get_whisper_model()

    # Force word timestamps when diarization is enabled (needed for alignment)
    if diarize:
        word_timestamps = True

    segments_gen, info = model.transcribe(
        audio_path,
        language="he",
        beam_size=5,
        word_timestamps=word_timestamps,
        vad_filter=True,
    )

    segments = []
    for seg in segments_gen:
        segment_data = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }

        if word_timestamps and seg.words:
            segment_data["words"] = [
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                }
                for w in seg.words
            ]

        segments.append(segment_data)

    return {
        "segments": segments,
        "language": info.language,
        "duration": round(info.duration, 2),
    }
