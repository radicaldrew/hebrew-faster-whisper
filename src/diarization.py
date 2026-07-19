"""
Speaker diarization using pyannote.audio.

Runs diarization on audio and aligns speaker labels with whisper segments/words.
"""

from models import get_diarization_pipeline
from utils import convert_to_wav, cleanup_file


def run_diarization(audio_path: str, num_speakers: int = None):
    """
    Run speaker diarization on an audio file.

    The input is first decoded to 16 kHz mono PCM WAV — pyannote crashes on
    compressed containers whose declared duration disagrees with the decoded
    sample count (see convert_to_wav).

    Args:
        audio_path: Path to the audio file.
        num_speakers: Exact speaker count when the caller knows it (e.g. 2 for
            phone calls). Constraining the clustering this way is the single
            biggest accuracy lever on narrowband telephony audio, where
            unconstrained clustering often collapses both parties into one
            speaker. None = let pyannote decide.

    Returns pyannote Annotation object with speaker turns.
    """
    pipeline = get_diarization_pipeline()
    wav_path = convert_to_wav(audio_path)
    try:
        if num_speakers:
            diarization = pipeline(wav_path, num_speakers=num_speakers)
        else:
            diarization = pipeline(wav_path)
    finally:
        cleanup_file(wav_path)
    return diarization


def align_speakers(segments: list, diarization) -> list:
    """
    Align speaker labels from diarization with transcription segments.

    Each word is assigned the diarization turn it overlaps most. Whisper
    segments regularly span a speaker turn in fast dialogue, so segments are
    SPLIT at word-level speaker changes rather than majority-voted — a vote
    silently reassigns the minority speaker's words to the other party.

    Args:
        segments: List of segment dicts with 'words' containing word-level timestamps.
        diarization: pyannote Annotation object from run_diarization().

    Returns:
        New list of single-speaker segments with 'speaker' on segments and words.
    """
    # Build a flat list of diarization turns for efficient lookup
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    aligned = []
    for segment in segments:
        if "words" in segment and segment["words"]:
            for word in segment["words"]:
                word["speaker"] = _find_speaker_for_timespan(
                    word["start"], word["end"], turns
                )
            aligned.extend(_split_segment_by_speaker(segment))
        else:
            # No word timestamps - assign speaker based on segment span
            segment["speaker"] = _find_speaker_for_timespan(
                segment["start"], segment["end"], turns
            )
            aligned.append(segment)

    return aligned


def _split_segment_by_speaker(segment: dict) -> list:
    """
    Split one whisper segment into runs of consecutive same-speaker words.

    Words the diarizer couldn't place (speaker None, common in turn-change
    gaps) inherit the running speaker rather than starting a new run. If the
    whole segment is one speaker, it is returned as-is (with 'speaker' set)
    to preserve whisper's original text/punctuation spacing.
    """
    words = segment["words"]

    runs = []
    current = None
    for word in words:
        speaker = word["speaker"]
        if current is None:
            current = {"speaker": speaker, "words": [word]}
        elif speaker is None or speaker == current["speaker"]:
            current["words"].append(word)
        elif current["speaker"] is None:
            # Run started with unplaced words - adopt the first real speaker
            current["speaker"] = speaker
            current["words"].append(word)
        else:
            runs.append(current)
            current = {"speaker": speaker, "words": [word]}
    if current is not None:
        runs.append(current)

    if len(runs) == 1:
        segment["speaker"] = runs[0]["speaker"]
        return [segment]

    split = []
    for run in runs:
        run_words = run["words"]
        # Backfill None-speaker words that led the run
        for word in run_words:
            if word["speaker"] is None:
                word["speaker"] = run["speaker"]
        split.append({
            "start": run_words[0]["start"],
            "end": run_words[-1]["end"],
            "text": " ".join(w["word"] for w in run_words),
            "speaker": run["speaker"],
            "words": run_words,
        })
    return split


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


def count_speakers(diarization) -> int:
    """Count the number of unique speakers in the diarization result."""
    speakers = set()
    for _, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
    return len(speakers)
