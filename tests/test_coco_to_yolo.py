from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.coco_to_yolo import convert_coco_dataset_to_yolo, convert_coco_to_yolo_labels


def coco_fixture() -> dict:
    return {
        "images": [
            {"id": 10, "path": "subset-1/positive.png", "file_name": "positive.png", "width": 100, "height": 50},
            {"id": 11, "path": "subset-1/negative.png", "file_name": "negative.png", "width": 100, "height": 50},
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 7, "bbox": [10, 5, 20, 10]},
            {"id": 2, "image_id": 10, "category_id": 7, "bbox": [40, 10, 30, 20]},
        ],
        "categories": [{"id": 7, "name": "pallet-jack"}],
    }


class ConvertCocoToYoloLabelsTests(unittest.TestCase):
    def test_maps_coco_xywh_to_zero_based_normalized_yolo_and_writes_negative_label(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            (images / "subset-1").mkdir(parents=True)
            Image.new("RGB", (100, 50)).save(images / "subset-1" / "positive.png")
            Image.new("RGB", (100, 50)).save(images / "subset-1" / "negative.png")
            labels = root / "labels"

            result = convert_coco_to_yolo_labels(coco_fixture(), images, labels)

            self.assertEqual(result.image_count, 2)
            self.assertEqual(result.annotation_count, 2)
            self.assertEqual(result.negative_image_count, 1)
            self.assertEqual(
                (labels / "subset-1" / "positive.txt").read_text(encoding="utf-8"),
                "0 0.2 0.2 0.2 0.2\n0 0.55 0.4 0.3 0.4\n",
            )
            self.assertEqual((labels / "subset-1" / "negative.txt").read_text(encoding="utf-8"), "")

    def test_rejects_bbox_outside_image_bounds(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary_directory:
            root = Path(temporary_directory)
            (root / "subset-1").mkdir()
            Image.new("RGB", (100, 50)).save(root / "subset-1" / "positive.png")
            Image.new("RGB", (100, 50)).save(root / "subset-1" / "negative.png")
            coco = coco_fixture()
            coco["annotations"][0]["bbox"] = [90, 5, 20, 10]

            with self.assertRaisesRegex(ValueError, "outside image bounds"):
                convert_coco_to_yolo_labels(coco, root, root.parent / "labels")

    def test_rejects_image_paths_that_escape_image_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "images"
            image_root.mkdir()
            coco = coco_fixture()
            coco["images"][0]["path"] = "../outside.png"

            with self.assertRaisesRegex(ValueError, "escapes the image root"):
                convert_coco_to_yolo_labels(coco, image_root, root / "labels")


class ConvertCocoDatasetToYoloTests(unittest.TestCase):
    def test_writes_dataset_yaml_and_conversion_report_without_changing_coco(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            image_root = root / "images" / "subset-1"
            image_root.mkdir(parents=True)
            Image.new("RGB", (100, 50)).save(image_root / "positive.png")
            Image.new("RGB", (100, 50)).save(image_root / "negative.png")
            annotation_path = root / "annotations.json"
            source = coco_fixture()
            original_json = json.dumps(source)
            annotation_path.write_text(original_json, encoding="utf-8")

            result = convert_coco_dataset_to_yolo(annotation_path, root)

            self.assertEqual(result.image_count, 2)
            self.assertTrue((root / "dataset.yaml").is_file())
            self.assertTrue((root / "conversion_report.json").is_file())
            dataset_config = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
            self.assertEqual(Path(dataset_config["path"]), root.resolve())
            self.assertEqual(dataset_config["test"], "images")
            self.assertEqual(annotation_path.read_text(encoding="utf-8"), original_json)
            self.assertEqual((root / "labels" / "subset-1" / "negative.txt").read_text(encoding="utf-8"), "")

    def test_refuses_to_replace_existing_conversion_without_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary_directory:
            root = Path(temporary_directory) / "dataset"
            image_root = root / "images" / "subset-1"
            image_root.mkdir(parents=True)
            Image.new("RGB", (100, 50)).save(image_root / "positive.png")
            Image.new("RGB", (100, 50)).save(image_root / "negative.png")
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps(coco_fixture()), encoding="utf-8")
            (root / "labels").mkdir()

            with self.assertRaises(FileExistsError):
                convert_coco_dataset_to_yolo(annotation_path, root)

            self.assertFalse((root / "dataset.yaml").exists())


if __name__ == "__main__":
    unittest.main()
