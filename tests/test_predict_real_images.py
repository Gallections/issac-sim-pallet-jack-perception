from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.predict_real_images import (
    find_prediction_images,
    main,
    predict_real_images,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PredictRealImagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "real_photos"
        (self.source / "warehouse").mkdir(parents=True)
        (self.source / "front.JPG").write_bytes(b"image")
        (self.source / "warehouse" / "side.png").write_bytes(b"image")
        (self.source / "notes.txt").write_text("ignore", encoding="utf-8")
        self.model_path = self.root / "best.pt"
        self.model_path.write_bytes(b"checkpoint")
        self.output = self.root / "annotated"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_finds_supported_images_recursively_and_in_stable_order(self) -> None:
        images = find_prediction_images(self.source)

        self.assertEqual(
            images,
            [self.source / "front.JPG", self.source / "warehouse" / "side.png"],
        )

    def test_accepts_one_supported_image(self) -> None:
        image = self.source / "front.JPG"

        self.assertEqual(find_prediction_images(image), [image])

    def test_rejects_source_folder_without_supported_images(self) -> None:
        empty_folder = self.root / "empty"
        empty_folder.mkdir()

        with self.assertRaisesRegex(ValueError, "no supported image files"):
            find_prediction_images(empty_folder)

    def test_rejects_output_folder_nested_inside_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be inside the source folder"):
            predict_real_images(
                source=self.source,
                model_path=self.model_path,
                output_dir=self.source / "annotated",
            )

    def test_predicts_and_saves_annotated_images_preserving_subfolders(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResult:
            def save(self, filename: str) -> str:
                destination = Path(filename)
                destination.write_bytes(b"annotated image")
                return str(destination)

        class FakeYOLO:
            def __init__(self, model: str) -> None:
                calls.append({"model": model})

            def predict(self, **kwargs: object) -> list[FakeResult]:
                calls.append(kwargs)
                return [FakeResult()]

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            outputs = predict_real_images(
                source=self.source,
                model_path=self.model_path,
                output_dir=self.output,
                confidence=0.4,
                imgsz=640,
                device="0",
            )

        expected_outputs = [
            self.output / "front.JPG",
            self.output / "warehouse" / "side.png",
        ]
        self.assertEqual(outputs, expected_outputs)
        self.assertTrue(all(path.is_file() for path in expected_outputs))
        self.assertEqual(calls[0]["model"], str(self.model_path))
        self.assertEqual(len(calls), 3)
        for call in calls[1:]:
            self.assertEqual(call["conf"], 0.4)
            self.assertEqual(call["imgsz"], 640)
            self.assertEqual(call["device"], "0")
            self.assertFalse(call["verbose"])

    def test_cli_reports_saved_image_count_and_output_folder(self) -> None:
        expected_outputs = [self.output / "front.JPG", self.output / "warehouse" / "side.png"]
        stdout = io.StringIO()
        with patch(
            "palletjack_detection.predict_real_images.predict_real_images",
            return_value=expected_outputs,
        ) as prediction_mock, contextlib.redirect_stdout(stdout):
            status = main(
                [
                    "--source",
                    str(self.source),
                    "--model",
                    str(self.model_path),
                    "--output",
                    str(self.output),
                ]
            )

        self.assertEqual(status, 0)
        self.assertIn("2 annotated image(s)", stdout.getvalue())
        self.assertIn(str(self.output), stdout.getvalue())
        prediction_mock.assert_called_once()

    def test_rejects_missing_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "model checkpoint does not exist"):
            predict_real_images(
                source=self.source,
                model_path=self.root / "missing.pt",
                output_dir=self.output,
            )

    def test_rejects_confidence_outside_zero_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence must be between 0 and 1"):
            predict_real_images(
                source=self.source,
                model_path=self.model_path,
                output_dir=self.output,
                confidence=1.1,
            )


if __name__ == "__main__":
    unittest.main()
