from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.validate_dataset import main, validate_kitti_object_detection_export


VALID_LABEL = "palletjack 0.00 0 0.00 10 12 99 79 0 0 0 0 0 0 0\n"


class ValidateKittiExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp_dir.name)
        self.rgb = self.root / "scenario" / "Camera" / "rgb"
        self.labels = self.root / "scenario" / "Camera" / "object_detection"
        self.rgb.mkdir(parents=True)
        self.labels.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_image(self, stem: str, size: tuple[int, int] = (100, 80)) -> None:
        Image.new("RGB", size).save(self.rgb / f"{stem}.png")

    def add_label(self, stem: str, content: str) -> None:
        (self.labels / f"{stem}.txt").write_text(content, encoding="utf-8")

    def issue_messages(self, report) -> list[str]:
        return [issue.message for issue in report.issues]

    def test_valid_kitti_label_is_counted_using_actual_image_dimensions(self) -> None:
        self.add_image("0")
        self.add_label("0", VALID_LABEL)

        report = validate_kitti_object_detection_export(self.root)

        self.assertTrue(report.ok)
        self.assertEqual(report.scenarios[0].image_count, 1)
        self.assertEqual(report.scenarios[0].annotation_count, 1)
        self.assertEqual(report.scenarios[0].object_count, 1)
        self.assertEqual(report.issues, [])

    def test_reports_missing_image_and_annotation_pairs_and_wrong_field_count(self) -> None:
        self.add_image("no_label")
        self.add_label("no_image", "")
        self.add_image("bad")
        self.add_label("bad", "palletjack 0 0 0 1 2 101 20 0 0 0 0 0 0\n")

        report = validate_kitti_object_detection_export(self.root)
        messages = self.issue_messages(report)

        self.assertFalse(report.ok)
        self.assertTrue(any("no annotation" in message for message in messages))
        self.assertTrue(any("no matching image" in message for message in messages))
        self.assertTrue(any("expected 15 fields" in message for message in messages))

    def test_reports_unknown_classes_and_invalid_box_order(self) -> None:
        self.add_image("0")
        self.add_label("0", "forklift 0 0 0 20 20 10 30 0 0 0 0 0 0 0\n")

        messages = self.issue_messages(validate_kitti_object_detection_export(self.root))

        self.assertTrue(any("unknown class 'forklift'" in message for message in messages))
        self.assertTrue(any("box coordinates must satisfy" in message for message in messages))

    def test_empty_annotation_is_reported_as_valid_empty_frame(self) -> None:
        self.add_image("0")
        self.add_label("0", "  \n")

        report = validate_kitti_object_detection_export(self.root)

        self.assertTrue(report.ok)
        self.assertEqual(report.scenarios[0].empty_annotation_count, 1)
        self.assertEqual(report.scenarios[0].object_count, 0)

    def test_nonfinite_values_and_invalid_occlusion_are_rejected(self) -> None:
        self.add_image("nan")
        self.add_label("nan", "palletjack 0 0 0 1 2 NaN 20 0 0 0 0 0 0 0\n")
        self.add_image("occlusion")
        self.add_label("occlusion", "palletjack 0 4 0 1 2 10 20 0 0 0 0 0 0 0\n")

        report = validate_kitti_object_detection_export(self.root)
        messages = self.issue_messages(report)

        self.assertTrue(any("finite numbers" in message for message in messages))
        self.assertTrue(any("occlusion must be an integer" in message for message in messages))
        self.assertEqual(report.scenarios[0].object_count, 0)

    def test_out_of_bounds_box_is_checked_against_each_image_size(self) -> None:
        self.add_image("small", (12, 8))
        self.add_label("small", "palletjack 0 0 0 1 2 13 7 0 0 0 0 0 0 0\n")

        messages = self.issue_messages(validate_kitti_object_detection_export(self.root))

        self.assertTrue(any("exceeds image bounds 12x8" in message for message in messages))

    def test_reports_unreadable_image_and_missing_required_directories(self) -> None:
        (self.rgb / "broken.png").write_bytes(b"not an image")
        missing_root = self.root / "other_scenario"
        missing_root.mkdir()

        report = validate_kitti_object_detection_export(self.root)
        messages = self.issue_messages(report)

        self.assertTrue(any("image cannot be opened" in message for message in messages))
        self.assertTrue(any("required directory is missing" in message for message in messages))
        missing_issue = next(issue for issue in report.issues if "required directory" in issue.message)
        self.assertEqual(missing_issue.path.replace(chr(92), "/"), "Camera/rgb")

    def test_invalid_root_produces_a_reported_issue(self) -> None:
        report = validate_kitti_object_detection_export(self.root / "missing")

        self.assertFalse(report.ok)
        self.assertTrue(any("dataset root does not exist" in issue.message for issue in report.issues))

    def test_validates_only_requested_scenarios_when_raw_root_contains_other_datasets(self) -> None:
        self.add_image("0")
        self.add_label("0", VALID_LABEL)
        (self.root / "real-pallet-jack-data").mkdir()

        report = validate_kitti_object_detection_export(self.root, scenario_names=("scenario",))

        self.assertTrue(report.ok)
        self.assertEqual([summary.name for summary in report.scenarios], ["scenario"])

    def test_requested_scenario_names_must_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested scenario does not exist"):
            validate_kitti_object_detection_export(self.root, scenario_names=("missing",))

    def test_command_returns_failure_status_when_data_has_issues(self) -> None:
        self.add_image("unmatched")
        output = StringIO()

        with redirect_stdout(output):
            status = main(["--root", str(self.root)])

        self.assertEqual(status, 1)
        self.assertIn("no annotation file", output.getvalue())


if __name__ == "__main__":
    unittest.main()





