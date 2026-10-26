"""Build a local Experiment 2 ground-truth annotation gallery."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from palletjack_detection.annotation_audit import build_experiment_annotation_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "experiment-02",
        help="Prepared Experiment 2 dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "local" / "annotation_audit" / "experiment-02",
        help="Experiment 2 gallery output directory.",
    )
    parser.add_argument("--page-size", type=int, default=50)
    args = parser.parse_args()
    processed_root = args.processed_root if args.processed_root.is_absolute() else PROJECT_ROOT / args.processed_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    summary = build_experiment_annotation_audit(
        processed_root,
        output_dir,
        page_size=args.page_size,
    )
    print(f"Experiment 2 gallery: {output_dir.resolve() / 'index.html'}")
    print(f"Images: {summary.image_count}; ground-truth boxes: {summary.box_count}")
    for split, scenario, images, boxes, pages in summary.galleries:
        print(f"{split}/{scenario}: {images} images, {boxes} boxes, {pages} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
