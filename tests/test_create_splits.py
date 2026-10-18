from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.create_splits import (
    KittiExportValidationError,
    create_stratified_split_manifests,
)


class CreateSplitManifestsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.base = Path(self.temp_dir.name)
        self.dataset_root = self.base / "raw"
        self.output_root = self.base / "processed" / "splits"
        self.processed_root = self.base / "processed" / "yolo"
        for scenario in ("no_distractors", "distractors_warehouse", "distractors_additional"):
            rgb_dir = self.dataset_root / scenario / "Camera" / "rgb"
            label_dir = self.dataset_root / scenario / "Camera" / "object_detection"
            rgb_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for frame in range(10):
                Image.new("RGB", (32, 24), color=(frame, 0, 0)).save(rgb_dir / f"{frame}.png")
                label = "palletjack 0 0 0 10 4 20 14 0 0 0 0 0 0 0\n" if frame == 0 else ""
                (label_dir / f"{frame}.txt").write_text(label, encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def read_manifest(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_creates_80_10_10_manifest_counts_per_scenario(self) -> None:
        result = create_stratified_split_manifests(
            self.dataset_root, self.output_root, processed_root=self.processed_root, seed=42
        )

        self.assertEqual(result.split_counts["train"], {
            "distractors_additional": 8,
            "distractors_warehouse": 8,
            "no_distractors": 8,
        })
        self.assertEqual(result.split_counts["validation"], {
            "distractors_additional": 1,
            "distractors_warehouse": 1,
            "no_distractors": 1,
        })
        self.assertEqual(result.split_counts["test"], {
            "distractors_additional": 1,
            "distractors_warehouse": 1,
            "no_distractors": 1,
        })
        self.assertEqual(result.total_frames, 30)
        self.assertEqual(set(result.manifest_paths), {"train", "validation", "test"})
        self.assertFalse(self.processed_root.exists())

    def test_each_frame_appears_once_with_raw_and_processed_paths(self) -> None:
        create_stratified_split_manifests(
            self.dataset_root, self.output_root, processed_root=self.processed_root, seed=42
        )
        rows = [
            row
            for split in ("train", "validation", "test")
            for row in self.read_manifest(self.output_root / f"{split}.csv")
        ]

        self.assertEqual(len(rows), 30)
        self.assertEqual(len({(row["scenario"], row["frame_id"]) for row in rows}), 30)
        self.assertTrue(all((self.dataset_root / row["source_image_path"]).is_file() for row in rows))
        self.assertTrue(all((self.dataset_root / row["kitti_annotation_path"]).is_file() for row in rows))
        self.assertTrue(all(row["processed_image_path"].startswith("images/") for row in rows))
        self.assertTrue(all(row["processed_kitti_annotation_path"].startswith("kitti_labels/") for row in rows))
        self.assertTrue(all(row["yolo_annotation_path"].startswith("labels/") for row in rows))
        self.assertFalse(self.processed_root.exists())

    def test_same_seed_reproduces_identical_manifests(self) -> None:
        first = self.base / "first" / "splits"
        second = self.base / "second" / "splits"
        create_stratified_split_manifests(self.dataset_root, first, seed=237)
        create_stratified_split_manifests(self.dataset_root, second, seed=237)

        for split in ("train", "validation", "test"):
            self.assertEqual((first / f"{split}.csv").read_bytes(), (second / f"{split}.csv").read_bytes())

    def test_invalid_dataset_is_rejected_without_writing_manifests(self) -> None:
        label = self.dataset_root / "no_distractors" / "Camera" / "object_detection" / "0.txt"
        label.write_text("palletjack malformed\n", encoding="utf-8")

        with self.assertRaises(KittiExportValidationError):
            create_stratified_split_manifests(self.dataset_root, self.output_root)

        self.assertFalse(self.output_root.exists())

    def test_split_ratios_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            create_stratified_split_manifests(
                self.dataset_root,
                self.output_root,
                train_ratio=0.7,
                validation_ratio=0.1,
                test_ratio=0.1,
            )
        self.assertFalse(self.output_root.exists())


if __name__ == "__main__":
    unittest.main()
