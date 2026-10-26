from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from palletjack_detection.annotation_audit import build_experiment_annotation_audit


class ExperimentAnnotationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temp_dir.name)
        self.processed = self.root / "processed"
        self.output = self.root / "annotation_audit" / "experiment-02"
        for split in ("train", "validation", "test"):
            (self.processed / "splits").mkdir(parents=True, exist_ok=True)
            (self.processed / f"images/{split}/distractors_additional").mkdir(
                parents=True, exist_ok=True
            )
            (self.processed / f"labels/{split}/distractors_additional").mkdir(
                parents=True, exist_ok=True
            )
            (self.processed / f"images/{split}/distractors_additional/0.png").write_bytes(
                b"image"
            )
            (self.processed / f"labels/{split}/distractors_additional/0.txt").write_text(
                "0 0.5 0.5 0.4 0.2\n", encoding="utf-8"
            )
            with (self.processed / f"splits/{split}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as manifest_file:
                writer = csv.DictWriter(
                    manifest_file,
                    fieldnames=("split", "scenario", "processed_image_path", "yolo_annotation_path"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "split": split,
                        "scenario": "distractors_additional",
                        "processed_image_path": f"images/{split}/distractors_additional/0.png",
                        "yolo_annotation_path": f"labels/{split}/distractors_additional/0.txt",
                    }
                )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_builds_separate_experiment_gallery_with_ground_truth_boxes(self) -> None:
        summary = build_experiment_annotation_audit(self.processed, self.output)

        self.assertEqual(summary.image_count, 3)
        index = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertIn("Experiment 2 (EXP-002)", index)
        self.assertIn("train-distractors_additional/page-001.html", index)
        page = (self.output / "train-distractors_additional/page-001.html").read_text(
            encoding="utf-8"
        )
        self.assertIn("stored YOLO ground-truth boxes drawn in green", page)
        self.assertIn("width:40.000000%", page)
        self.assertIn("../../../processed/images/train/distractors_additional/0.png", page)
        self.assertFalse((self.output / "images").exists())

    def test_rejects_manifest_paths_that_escape_processed_root(self) -> None:
        manifest_path = self.processed / "splits/train.csv"
        text = manifest_path.read_text(encoding="utf-8")
        manifest_path.write_text(text.replace("images/train", "../outside"), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "must stay within the processed dataset"):
            build_experiment_annotation_audit(self.processed, self.output)


if __name__ == "__main__":
    unittest.main()
