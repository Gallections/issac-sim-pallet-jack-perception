from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.train_yolo import main, run_ultralytics_training
from palletjack_detection.training_config import load_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainYoloTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.fixture_root = Path(self.temp_dir.name)
        self.dataset_root = self.fixture_root / "processed"
        self.dataset_root.mkdir()
        for split in ("train", "val", "test"):
            (self.dataset_root / "images" / split).mkdir(parents=True)
        self.dataset_yaml = self.dataset_root / "dataset.yaml"
        self.dataset_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {0: "palletjack"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self.config_path = self.fixture_root / "train.yaml"
        values = {
            "model": "yolo26n.pt",
            "dataset_yaml": self.dataset_yaml.relative_to(PROJECT_ROOT).as_posix(),
            "epochs": 3,
            "imgsz": 640,
            "batch": 8,
            "device": "cpu",
            "seed": 42,
            "workers": 0,
            "project": (self.fixture_root / "runs").relative_to(PROJECT_ROOT).as_posix(),
            "name": "test-run",
        }
        self.config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fine_tunes_configured_checkpoint_and_saves_reproducibility_artifacts(self) -> None:
        calls: dict[str, object] = {}

        class FakeYOLO:
            def __init__(self, model: str) -> None:
                calls["model"] = model
                self.trainer = None

            def train(self, **kwargs: object) -> None:
                calls["train_kwargs"] = kwargs
                output_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                output_dir.mkdir(parents=True)
                self.trainer = types.SimpleNamespace(save_dir=output_dir)
                resolved_dataset = yaml.safe_load(Path(str(kwargs["data"])).read_text(encoding="utf-8"))
                calls["resolved_dataset"] = resolved_dataset

        ultralytics_mock = types.ModuleType("ultralytics")
        ultralytics_mock.YOLO = FakeYOLO
        config = load_training_config(self.config_path)
        with patch.dict(sys.modules, {"ultralytics": ultralytics_mock}), patch(
            "palletjack_detection.train_yolo.version", side_effect=lambda name: f"{name}-test-version"
        ):
            run_dir = run_ultralytics_training(config)

        self.assertEqual(calls["model"], "yolo26n.pt")
        train_kwargs = calls["train_kwargs"]
        self.assertEqual(train_kwargs["epochs"], 3)
        self.assertEqual(train_kwargs["imgsz"], 640)
        self.assertEqual(train_kwargs["batch"], 8)
        self.assertEqual(train_kwargs["device"], "cpu")
        self.assertEqual(train_kwargs["seed"], 42)
        self.assertEqual(train_kwargs["workers"], 0)
        self.assertEqual(train_kwargs["project"], str(config.project))
        self.assertEqual(train_kwargs["name"], "test-run")
        self.assertFalse(train_kwargs["exist_ok"])
        self.assertTrue(Path(str(train_kwargs["data"])).is_absolute())
        self.assertEqual(calls["resolved_dataset"]["path"], str(self.dataset_root))
        self.assertEqual(run_dir, self.fixture_root / "runs" / "test-run")
        self.assertTrue((run_dir / "training_config.yaml").is_file())
        self.assertEqual(
            yaml.safe_load((run_dir / "dataset.yaml").read_text(encoding="utf-8"))["path"],
            str(self.dataset_root),
        )
        metadata = json.loads((run_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["ultralytics_version"], "ultralytics-test-version")
        self.assertEqual(metadata["torch_version"], "torch-test-version")

    def test_preflight_command_does_not_start_training(self) -> None:
        output = io.StringIO()
        with patch("palletjack_detection.train_yolo.run_ultralytics_training") as train_mock:
            with contextlib.redirect_stdout(output):
                status = main(["--config", str(self.config_path), "--validate-only"])

        self.assertEqual(status, 0)
        self.assertIn("Configuration is valid", output.getvalue())
        self.assertIn(str(self.dataset_yaml), output.getvalue())
        self.assertIn("test-run", output.getvalue())
        train_mock.assert_not_called()

    def test_cli_launches_training_and_reports_run_directory(self) -> None:
        expected_run = self.fixture_root / "runs" / "test-run"
        output = io.StringIO()
        with patch(
            "palletjack_detection.train_yolo.run_ultralytics_training", return_value=expected_run
        ) as train_mock:
            with contextlib.redirect_stdout(output):
                status = main(["--config", str(self.config_path)])

        self.assertEqual(status, 0)
        self.assertIn(str(expected_run), output.getvalue())
        train_mock.assert_called_once()
        self.assertEqual(train_mock.call_args.args[0].dataset_yaml, self.dataset_yaml)

    def test_invalid_config_fails_before_model_loading(self) -> None:
        invalid_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        invalid_config["epochs"] = 0
        self.config_path.write_text(yaml.safe_dump(invalid_config), encoding="utf-8")
        ultralytics_mock = types.ModuleType("ultralytics")
        ultralytics_mock.YOLO = unittest.mock.Mock()
        output = io.StringIO()

        with patch.dict(sys.modules, {"ultralytics": ultralytics_mock}):
            with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as error:
                main(["--config", str(self.config_path)])

        self.assertEqual(error.exception.code, 2)
        ultralytics_mock.YOLO.assert_not_called()
        self.assertIn("epochs", output.getvalue())

    def test_run_name_collision_preserves_existing_artifacts_and_skips_model_loading(self) -> None:
        run_dir = self.fixture_root / "runs" / "test-run"
        run_dir.mkdir(parents=True)
        marker = run_dir / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")
        ultralytics_mock = types.ModuleType("ultralytics")
        ultralytics_mock.YOLO = unittest.mock.Mock()
        output = io.StringIO()

        with patch.dict(sys.modules, {"ultralytics": ultralytics_mock}):
            with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as error:
                main(["--config", str(self.config_path)])

        self.assertEqual(error.exception.code, 2)
        ultralytics_mock.YOLO.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
