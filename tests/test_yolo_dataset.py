from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.kitti_to_yolo import (
    convert_kitti_annotation_file_to_yolo,
    format_kitti_annotation_as_yolo,
)


class KittiToYoloConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp_dir.name)
        self.label = self.root / "frame.txt"
        self.label.write_text("palletjack 0 0 0 10 4 20 14 0 0 0 0 0 0 0\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_converts_inclusive_kitti_box_to_normalized_yolo(self) -> None:
        converted = format_kitti_annotation_as_yolo(self.label, image_width=32, image_height=24)

        self.assertEqual(converted, "0 0.484375 0.395833 0.34375 0.458333\n")

    def test_converts_exclusive_max_edges_when_configured(self) -> None:
        converted = format_kitti_annotation_as_yolo(
            self.label,
            image_width=32,
            image_height=24,
            inclusive_max_coordinates=False,
        )

        self.assertEqual(converted, "0 0.46875 0.375 0.3125 0.416667\n")

    def test_keeps_empty_annotation_as_an_empty_yolo_file(self) -> None:
        self.label.write_text("", encoding="utf-8")

        self.assertEqual(format_kitti_annotation_as_yolo(self.label, image_width=32, image_height=24), "")

    def test_converts_one_kitti_annotation_using_paired_image_dimensions(self) -> None:
        image = self.root / "frame.png"
        Image.new("RGB", (32, 24)).save(image)
        output = self.root / "yolo" / "frame.txt"

        result = convert_kitti_annotation_file_to_yolo(self.label, image, output)

        self.assertEqual(result, output)
        self.assertEqual(output.read_text(encoding="utf-8"), "0 0.484375 0.395833 0.34375 0.458333\n")

    def test_converts_blank_annotation_to_empty_output_file(self) -> None:
        self.label.write_text("", encoding="utf-8")
        image = self.root / "frame.png"
        Image.new("RGB", (32, 24)).save(image)
        output = self.root / "yolo.txt"

        convert_kitti_annotation_file_to_yolo(self.label, image, output)

        self.assertTrue(output.is_file())
        self.assertEqual(output.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
