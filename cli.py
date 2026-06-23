"""
cli.py
Interactive command-line interface for TubeSage.
"""

import argparse
import sys

from dotenv import load_dotenv

from .rag_pipeline import YouTubeRAG
from .transcript_loader import TranscriptUnavailableError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with any YouTube video using Retrieval-Augmented Generation."
    )
    parser.add_argument(
        "video_id", help="YouTube video ID, e.g. Gfr50f6ZBvo (NOT the full URL)"
    )
    parser.add_argument(
        "-q", "--question", help="Ask a single question and exit (skips interactive mode)."
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["en"],
        help="Preferred transcript languages, in priority order.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    rag = YouTubeRAG()
    print(f"Fetching transcript and building index for video '{args.video_id}'...")
    try:
        num_chunks = rag.build_index(args.video_id, languages=args.lang)
    except TranscriptUnavailableError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Indexed {num_chunks} chunks. Ready!\n")

    if args.question:
        print(rag.ask(args.question))
        return

    print("Ask questions about the video below.")
    print("Type 'summary' for a summary, or 'exit' / 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        if question.lower() == "summary":
            print(f"Bot: {rag.summarize()}\n")
            continue

        print(f"Bot: {rag.ask(question)}\n")


if __name__ == "__main__":
    main()
