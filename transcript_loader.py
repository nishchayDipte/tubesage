"""
transcript_loader.py
Fetches and prepares YouTube video transcripts for downstream processing.
"""

from typing import List, Optional

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)


class TranscriptUnavailableError(Exception):
    """Raised when no transcript/captions can be retrieved for a video."""


def fetch_transcript(video_id: str, languages: Optional[List[str]] = None) -> str:
    """
    Fetch and flatten the transcript of a YouTube video into a single string.

    Args:
        video_id: The YouTube video ID only (e.g. "Gfr50f6ZBvo"), NOT the full URL.
        languages: Preferred caption languages, in priority order. Defaults to ["en"].

    Returns:
        The full transcript as plain text.

    Raises:
        TranscriptUnavailableError: If captions are disabled or unavailable
            in the requested languages.
    """
    languages = languages or ["en"]

    try:
        transcript_chunks = YouTubeTranscriptApi.get_transcript(
            video_id, languages=languages
        )
    except TranscriptsDisabled as exc:
        raise TranscriptUnavailableError(
            f"Captions are disabled for video '{video_id}'."
        ) from exc
    except NoTranscriptFound as exc:
        raise TranscriptUnavailableError(
            f"No transcript found for video '{video_id}' in languages {languages}."
        ) from exc

    return " ".join(chunk["text"] for chunk in transcript_chunks)
