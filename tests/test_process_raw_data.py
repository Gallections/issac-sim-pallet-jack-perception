from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.create_splits import create_stratified_split_manifests
from palletjack_detection.process_raw_data import process_raw_export_from_manifests


class ProcessRawDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.base = Path(self.temp_dir.name)
        self.raw_root = self.base / "raw"
        self.manifest_dir = self.base / "processed" / "splits"
        self.output_root = self.base / "processed" / "yolo"
        for scenario in ("scenario_a", "scenario_b", "scenario_c"):
            rgb_dir = self.raw_root / scenario / "Camera" / "rgb"
            label_dir = self.raw_root / scenario / "Camera" / "object_detection"
            depth_dir = self.raw_root / scenario / "Camera" / "depth"
            rgb_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            depth_dir.mkdir(parents=True)
            (depth_dir / "unused.bin").write_bytes(b"depth modality is not selected")
            for frame in range(10):
                Image.new("RGB", (32, 24), color=(frame, 0, 0)).save(rgb_dir / f"{frame}.png")
                label = "palletjack 0 0 0 10 4 20 14 0 0 0 0 0 0 0\n" if frame == 0 else ""
                (label_dir / f"{frame}.txt").write_text(label, encoding="utf-8")
        create_stratified_split_manifests(self.raw_root, self.manifest_dir, processed_root=self.output_root, seed=42)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_copies_selected_rgb_and_kitti_files_then_converts_labels(self) -> None:
        result = process_raw_export_from_manifests(self.raw_root, self.manifest_dir, self.output_root)

        self.assertEqual(result.copied_image_count, 30)
        self.assertEqual(result.copied_kitti_annotation_count, 30)
        self.assertEqual(result.converted_yolo_annotation_count, 30)
        with (self.manifest_dir / "train.csv").open(newline="", encoding="utf-8") as handle:
            train_rows = list(csv.DictReader(handle))
        positive = next(row for row in train_rows if row["frame_id"] == "0")
        image_path = self.output_root / positive["processed_image_path"]
        kitti_path = self.output_root / positive["processed_kitti_annotation_path"]
        yolo_path = self.output_root / positive["yolo_annotation_path"]
        self.assertTrue(image_path.is_file())
        self.assertTrue(kitti_path.is_file())
        self.assertEqual(kitti_path.read_text(encoding="utf-8"), "palletjack 0 0 0 10 4 20 14 0 0 0 0 0 0 0\n")
        self.assertEqual(yolo_path.read_text(encoding="utf-8"), "0 0.484375 0.395833 0.34375 0.458333\n")
        self.assertTrue(result.dataset_yaml.is_file())
        self.assertTrue((self.raw_root / positive["source_image_path"]).is_file())
        self.assertFalse((self.output_root / "depth").exists())

    def test_processes_blank_kitti_annotations_as_empty_yolo_labels(self) -> None:
        process_raw_export_from_manifests(self.raw_root, self.manifest_dir, self.output_root)
        with (self.manifest_dir / "train.csv").open(newline="", encoding="utf-8") as handle:
            train_rows = list(csv.DictReader(handle))
        negative = next(row for row in train_rows if row["frame_id"] == "1")

        self.assertEqual((self.output_root / negative["yolo_annotation_path"]).read_text(encoding="utf-8"), "")

    def test_refuses_to_overwrite_unmarked_processed_files(self) -> None:
        protected_file = self.output_root / "images" / "keep.txt"
        protected_file.parent.mkdir(parents=True)
        protected_file.write_text("user data", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            process_raw_export_from_manifests(self.raw_root, self.manifest_dir, self.output_root)

        self.assertEqual(protected_file.read_text(encoding="utf-8"), "user data")

    def test_rejects_manifest_paths_that_escape_processed_root_before_copying(self) -> None:
        train_manifest = self.manifest_dir / "train.csv"
        with train_manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        rows[0]["processed_image_path"] = "../outside.png"
        with train_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaises(ValueError):
            process_raw_export_from_manifests(self.raw_root, self.manifest_dir, self.output_root)

        self.assertFalse((self.output_root / ".generated_by_process_raw_data").exists())


if __name__ == "__main__":
    unittest.main()
