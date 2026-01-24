"""
Speaker diarization using pyannote.audio.

Runs diarization on audio and aligns speaker labels with whisper segments/words.
"""

from models import get_diarization_pipeline


def run_diarization(audio_path: str):
    """
    Run speaker diarization on an audio file.

    Returns pyannote Annotation object with speaker turns.
    """
    pipeline = get_diarization_pipeline()
    diarization = pipeline(audio_path)
    return diarization


def align_speakers(segments: list, diarization) -> list:
    """
    Align speaker labels from diarization with transcription segments.

    For each word (using word timestamps), finds the overlapping diarization turn
    and assigns the speaker. Segment-level speaker is determined by majority vote
    from word speakers.

    Args:
        segments: List of segment dicts with 'words' containing word-level timestamps.
        diarization: pyannote Annotation object from run_diarization().

    Returns:
        Updated segments with 'speaker' field on both segments and words.
    """
    # Build a flat list of diarization turns for efficient lookup
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    for segment in segments:
        word_speakers = []

        if "words" in segment and segment["words"]:
            for word in segment["words"]:
                speaker = _find_speaker_for_timespan(
                    word["start"], word["end"], turns
                )
                word["speaker"] = speaker
                if speaker:
                    word_speakers.append(speaker)

            # Segment speaker = majority from word speakers
            segment["speaker"] = _majority_speaker(word_speakers)
        else:
            # No word timestamps - assign speaker based on segment midpoint
            speaker = _find_speaker_for_timespan(
                segment["start"], segment["end"], turns
            )
            segment["speaker"] = speaker

    return segments


def _find_speaker_for_timespan(start: float, end: float, turns: list) -> str:
    """Find the speaker with the most overlap for a given time span."""
    best_speaker = None
    best_overlap = 0.0

    for turn in turns:
        overlap_start = max(start, turn["start"])
        overlap_end = min(end, turn["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    return best_speaker


def _majority_speaker(speakers: list) -> str:
    """Return the most common speaker label from a list."""
    if not speakers:
        return None

    counts = {}
    for s in speakers:
        if s:
            counts[s] = counts.get(s, 0) + 1

    if not counts:
        return None

    return max(counts, key=counts.get)


def count_speakers(diarization) -> int:
    """Count the number of unique speakers in the diarization result."""
    speakers = set()
    for _, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
    return len(speakers)
