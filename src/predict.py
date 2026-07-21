"""
Transcription logic for Hebrew whisper model.
"""

from models import get_whisper_model


def transcribe(audio_path: str, word_timestamps: bool = False,
               diarize: bool = False, vad_parameters: dict = None) -> dict:
    """
    Run Hebrew transcription on an audio file.

    Args:
        audio_path: Path to the audio file.
        word_timestamps: Whether to include word-level timestamps.
        diarize: If True, forces word_timestamps=True for alignment.
        vad_parameters: Optional VAD tuning overrides (faster-whisper dict).

    Returns:
        Dict with 'segments', 'language', 'duration'.
    """
    model = get_whisper_model()

    # Force word timestamps when diarization is enabled (needed for alignment)
    if diarize:
        word_timestamps = True

    extra = {}
    if vad_parameters:
        extra["vad_parameters"] = vad_parameters

    segments_gen, info = model.transcribe(
        audio_path,
        language="he",
        beam_size=5,
        word_timestamps=word_timestamps,
        vad_filter=True,
        **extra,
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

# A single-speaker channel is mostly silence while the other party talks, so
# the default VAD (threshold 0.5, 2s min-silence) both misses faint far-end
# speech and glues words from different utterances into one whisper segment
# whose restored timestamps then span tens of seconds.
# NOTE: faster-whisper 1.1.0 names the speech threshold `onset` (older
# releases called it `threshold` — passing that raises TypeError).
DUAL_VAD_PARAMETERS = {
    "onset": 0.35,
    "min_silence_duration_ms": 1000,
    "speech_pad_ms": 400,
}

# Words within one utterance are near-contiguous; a bigger intra-segment gap
# means the VAD spliced separate utterances together — split there so the
# timestamp-sorted merge interleaves turns correctly.
MAX_INTRA_SEGMENT_WORD_GAP = 1.5


def _split_segments_on_word_gaps(segments: list,
                                 max_gap: float = MAX_INTRA_SEGMENT_WORD_GAP) -> list:
    """Split segments wherever consecutive words are separated by silence the
    VAD removed, and tighten each segment's bounds to its actual word span."""
    out = []
    for seg in segments:
        words = seg.get("words")
        if not words:
            out.append(seg)
            continue

        runs = [[words[0]]]
        for prev, word in zip(words, words[1:]):
            if word["start"] - prev["end"] > max_gap:
                runs.append([word])
            else:
                runs[-1].append(word)

        if len(runs) == 1:
            seg["start"] = words[0]["start"]
            seg["end"] = words[-1]["end"]
            out.append(seg)
            continue

        for run in runs:
            out.append({
                "start": run[0]["start"],
                "end": run[-1]["end"],
                "text": " ".join(w["word"] for w in run),
                "words": run,
            })
    return out


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
        result = transcribe(path, word_timestamps=word_timestamps,
                            vad_parameters=DUAL_VAD_PARAMETERS)
        for seg in _split_segments_on_word_gaps(result["segments"]):
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
