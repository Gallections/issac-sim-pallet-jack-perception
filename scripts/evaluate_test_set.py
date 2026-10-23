"""Evaluate a selected detector on the held-out synthetic test split."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from palletjack_detection.evaluate_test_set import main


if __name__ == "__main__":
    raise SystemExit(main())
