from pathlib import Path
from config import IMAGE_EXT, VIDEO_EXT, GIF_EXT


def scan_files(root: str):
    root = Path(root)
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT | GIF_EXT:
            yield p