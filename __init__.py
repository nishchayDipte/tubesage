"""TubeSage - Chat with any YouTube video using RAG (Retrieval-Augmented Generation)."""

from .rag_pipeline import YouTubeRAG
from .transcript_loader import fetch_transcript, TranscriptUnavailableError

__all__ = ["YouTubeRAG", "fetch_transcript", "TranscriptUnavailableError"]

__version__ = "1.0.0"
