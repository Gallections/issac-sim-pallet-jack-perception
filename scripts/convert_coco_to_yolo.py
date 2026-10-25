"""CLI entry point for COCO-to-YOLO label conversion."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from palletjack_detection.coco_to_yolo import main


if __name__ == "__main__":
    raise SystemExit(main())
