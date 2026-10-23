from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.evaluate_test_set import evaluate_held_out_test_set


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HeldOutTestSetEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.fixture_root = Path(self.temp_dir.name)
        self.checkpoint = self.fixture_root / "best.pt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.dataset_root = self.fixture_root / "processed"
        (self.dataset_root / "images" / "test").mkdir(parents=True)
        dataset_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "palletjack"},
        }
        self.dataset_yaml = self.dataset_root / "dataset.yaml"
        self.dataset_yaml.write_text(yaml.safe_dump(dataset_yaml), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_evaluates_test_split_and_saves_metrics_in_repo_local_output(self) -> None:
        metrics = types.SimpleNamespace(
            box=types.SimpleNamespace(mp=0.925, mr=0.809, map50=0.906, map=0.786)
        )
        model = Mock()
        model.val.return_value = metrics
        ultralytics_mock = types.ModuleType("ultralytics")
        ultralytics_mock.YOLO = Mock(return_value=model)
        observed_dataset: dict[str, object] = {}

        def inspect_validation_dataset(**kwargs: object) -> object:
            observed_dataset.update(yaml.safe_load(Path(str(kwargs["data"])).read_text(encoding="utf-8")))
            return metrics

        model.val.side_effect = inspect_validation_dataset
        with patch.dict(sys.modules, {"ultralytics": ultralytics_mock}):
            output_dir = evaluate_held_out_test_set(
                checkpoint=self.checkpoint,
                dataset_yaml=self.dataset_yaml,
                repository_root=self.fixture_root,
                project="runs/evaluations",
                name="baseline-test",
                imgsz=640,
                batch=16,
                workers=4,
                device="0",
            )

        self.assertEqual(output_dir, self.fixture_root / "runs" / "evaluations" / "baseline-test")
        self.assertEqual(observed_dataset["path"], str(self.dataset_root.resolve()))
        self.assertEqual(observed_dataset["test"], "images/test")
        model.val.assert_called_once_with(
            data=model.val.call_args.kwargs["data"],
            split="test",
            imgsz=640,
            batch=16,
            workers=4,
            device="0",
            plots=True,
            project=str((self.fixture_root / "runs" / "evaluations").resolve()),
            name="baseline-test",
            exist_ok=False,
        )
        summary = json.loads((output_dir / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["split"], "test")
        self.assertEqual(summary["metrics"], {
            "precision": 0.925,
            "recall": 0.809,
            "mAP50": 0.906,
            "mAP50-95": 0.786,
        })

    def test_refuses_to_overwrite_existing_evaluation(self) -> None:
        output_dir = self.fixture_root / "runs" / "evaluations" / "baseline-test"
        output_dir.mkdir(parents=True)
        marker = output_dir / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        ultralytics_mock = types.ModuleType("ultralytics")
        ultralytics_mock.YOLO = Mock()

        with patch.dict(sys.modules, {"ultralytics": ultralytics_mock}):
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                evaluate_held_out_test_set(
                    checkpoint=self.checkpoint,
                    dataset_yaml=self.dataset_yaml,
                    repository_root=self.fixture_root,
                    project="runs/evaluations",
                    name="baseline-test",
                )

        ultralytics_mock.YOLO.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_rejects_dataset_without_a_test_split_before_model_loading(self) -> None:
        dataset_values = yaml.safe_load(self.dataset_yaml.read_text(encoding="utf-8"))
        del dataset_values["test"]
        self.dataset_yaml.write_text(yaml.safe_dump(dataset_values), encoding="utf-8")
        ultralytics_mock = types.ModuleType("ultralytics")
        ultralytics_mock.YOLO = Mock()

        with patch.dict(sys.modules, {"ultralytics": ultralytics_mock}):
            with self.assertRaisesRegex(ValueError, "test image path"):
                evaluate_held_out_test_set(
                    checkpoint=self.checkpoint,
                    dataset_yaml=self.dataset_yaml,
                    repository_root=self.fixture_root,
                )

        ultralytics_mock.YOLO.assert_not_called()


if __name__ == "__main__":
    unittest.main()
