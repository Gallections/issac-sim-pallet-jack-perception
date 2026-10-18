"""Convert one KITTI annotation file to YOLO format."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from palletjack_detection.kitti_to_yolo import main


if __name__ == "__main__":
    raise SystemExit(main())
