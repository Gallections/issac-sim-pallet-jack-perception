"""Prepare the LOCO subset as one-class pallet-jack COCO data."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from palletjack_detection.process_loco_coco import main


if __name__ == "__main__":
    raise SystemExit(main())
