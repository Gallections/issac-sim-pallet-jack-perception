"""Run a trained detector on real images and save annotated copies."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from palletjack_detection.predict_real_images import main


if __name__ == "__main__":
    raise SystemExit(main())
