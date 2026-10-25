from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.process_loco_coco import (
    filter_coco_to_pallet_jack,
    prepare_loco_pallet_jack_dataset,
)


def make_coco_fixture() -> dict:
    return {
        "info": {"description": "fixture"},
        "images": [
            {"id": 10, "file_name": "jack.jpg", "path": "subset-1/jack.jpg", "width": 20, "height": 12},
            {"id": 11, "file_name": "negative.jpg", "path": "subset-1/negative.jpg", "width": 20, "height": 12},
        ],
        "categories": [
            {"id": 7, "name": "pallet", "supercategory": ""},
            {"id": 11, "name": "pallet_truck", "supercategory": "", "color": "#6ba8dc"},
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 10,
                "category_id": 11,
                "bbox": [2, 3, 8, 6],
                "segmentation": [[2, 3, 2, 9, 10, 9, 10, 3]],
                "area": 48,
                "iscrowd": False,
            },
            {"id": 102, "image_id": 10, "category_id": 7, "bbox": [0, 0, 4, 4]},
            {"id": 103, "image_id": 11, "category_id": 7, "bbox": [5, 2, 3, 3]},
        ],
    }


class FilterCocoToPalletJackTests(unittest.TestCase):
    def test_keeps_only_pallet_trucks_and_retains_negative_images(self) -> None:
        source = make_coco_fixture()

        filtered = filter_coco_to_pallet_jack(source)

        self.assertEqual(filtered["images"], source["images"])
        self.assertEqual(filtered["categories"], [{"id": 1, "name": "pallet-jack", "supercategory": "", "color": "#6ba8dc"}])
        self.assertEqual([annotation["id"] for annotation in filtered["annotations"]], [101])
        self.assertEqual(filtered["annotations"][0]["category_id"], 1)
        self.assertEqual(filtered["annotations"][0]["segmentation"], source["annotations"][0]["segmentation"])
        self.assertEqual(filtered["info"], source["info"])
        self.assertEqual(len(source["annotations"]), 3, "filtering must not mutate the raw source object")
        self.assertIn(11, {image["id"] for image in filtered["images"]})

    def test_rejects_source_without_pallet_truck_category(self) -> None:
        source = make_coco_fixture()
        source["categories"] = [category for category in source["categories"] if category["name"] != "pallet_truck"]

        with self.assertRaisesRegex(ValueError, "pallet_truck"):
            filter_coco_to_pallet_jack(source)

    def test_rejects_annotations_with_unknown_image_references(self) -> None:
        source = make_coco_fixture()
        source["annotations"][0]["image_id"] = 999

        with self.assertRaisesRegex(ValueError, "image_id"):
            filter_coco_to_pallet_jack(source)


class PrepareLocoPalletJackDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.base = Path(self.temporary.name)
        self.raw_root = self.base / "raw"
        self.raw_images = self.raw_root / "subset-1"
        self.raw_images.mkdir(parents=True)
        for name in ("jack.jpg", "negative.jpg", "not_in_coco.jpg"):
            Image.new("RGB", (20, 12), color="white").save(self.raw_images / name)
        self.annotation_path = self.raw_root / "loco-all-v1.json"
        self.annotation_path.write_text(json.dumps(make_coco_fixture()), encoding="utf-8")
        self.output_root = self.base / "processed" / "real-pallet-jack-data"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_copies_annotated_images_filters_coco_and_reports_unmatched_raw_images(self) -> None:
        result = prepare_loco_pallet_jack_dataset(self.annotation_path, self.raw_root, self.output_root)

        output_data = json.loads(result.annotation_path.read_text(encoding="utf-8"))
        report = json.loads(result.report_path.read_text(encoding="utf-8"))
        self.assertEqual(result.copied_image_count, 2)
        self.assertEqual(result.target_annotation_count, 1)
        self.assertEqual(result.negative_image_count, 1)
        self.assertEqual(result.unmatched_image_files, ("subset-1/not_in_coco.jpg",))
        self.assertTrue((self.output_root / "images" / "subset-1" / "jack.jpg").is_file())
        self.assertTrue((self.output_root / "images" / "subset-1" / "negative.jpg").is_file())
        self.assertFalse((self.output_root / "images" / "subset-1" / "not_in_coco.jpg").exists())
        self.assertEqual(len(output_data["images"]), 2)
        self.assertEqual(output_data["images"][1]["id"], 11)
        self.assertEqual(output_data["categories"], [{"id": 1, "name": "pallet-jack", "supercategory": "", "color": "#6ba8dc"}])
        self.assertEqual([annotation["category_id"] for annotation in output_data["annotations"]], [1])
        self.assertEqual(output_data["images"][0]["file_name"], "subset-1/jack.jpg")
        self.assertEqual(report["unmatched_image_files"], ["subset-1/not_in_coco.jpg"])
        self.assertEqual(json.loads(self.annotation_path.read_text(encoding="utf-8")), make_coco_fixture())

    def test_refuses_to_replace_existing_output_without_overwrite(self) -> None:
        self.output_root.mkdir(parents=True)
        protected = self.output_root / "keep.txt"
        protected.write_text("user content", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            prepare_loco_pallet_jack_dataset(self.annotation_path, self.raw_root, self.output_root)

        self.assertEqual(protected.read_text(encoding="utf-8"), "user content")

    def test_rejects_source_image_paths_that_escape_raw_root(self) -> None:
        source = make_coco_fixture()
        source["images"][0]["path"] = "../outside.jpg"
        self.annotation_path.write_text(json.dumps(source), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "escapes"):
            prepare_loco_pallet_jack_dataset(self.annotation_path, self.raw_root, self.output_root)

        self.assertFalse(self.output_root.exists())

    def test_fails_before_writing_when_a_coco_image_is_missing(self) -> None:
        (self.raw_images / "negative.jpg").unlink()

        with self.assertRaises(FileNotFoundError):
            prepare_loco_pallet_jack_dataset(self.annotation_path, self.raw_root, self.output_root)

        self.assertFalse(self.output_root.exists())


if __name__ == "__main__":
    unittest.main()
