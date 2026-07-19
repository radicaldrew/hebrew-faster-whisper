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


DUAL_CHANNEL_SPEAKERS = ("SPEAKER_00", "SPEAKER_01")


def transcribe_dual(audio_path_a: str, audio_path_b: str,
                    word_timestamps: bool = False) -> dict:
    """
    Transcribe two single-speaker recordings of the same call and merge them.

    Each file is one party's side (e.g. PBX call legs). Both recordings must
    start at the same instant — timestamps are taken as-is and interleaved.
    The file itself identifies the speaker, so labels are exact and no
    diarization model is involved.

    Returns:
        Dict with 'segments' (speaker-tagged, sorted by start), 'language',
        and 'duration' (max of the two files — they cover the same call).
    """
    merged = []
    duration = 0.0
    language = None

    for path, speaker in ((audio_path_a, DUAL_CHANNEL_SPEAKERS[0]),
                          (audio_path_b, DUAL_CHANNEL_SPEAKERS[1])):
        result = transcribe(path, word_timestamps=word_timestamps)
        for seg in result["segments"]:
            seg["speaker"] = speaker
            for word in seg.get("words", []):
                word["speaker"] = speaker
            merged.append(seg)
        duration = max(duration, result["duration"])
        language = language or result["language"]

    merged.sort(key=lambda s: s["start"])

    return {
        "segments": merged,
        "language": language,
        "duration": duration,
    }
