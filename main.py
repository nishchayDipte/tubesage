"""
TubeSage entry point.

Usage:
    python main.py <video_id>
    python main.py <video_id> -q "What is this video about?"
"""

from tubesage.cli import main

if __name__ == "__main__":
    main()
