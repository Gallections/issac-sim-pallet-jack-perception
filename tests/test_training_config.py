from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from palletjack_detection.training_config import load_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.fixture_root = Path(self.temp_dir.name)
        self.dataset_root = self.fixture_root / "processed"
        self.dataset_root.mkdir()
        for split in ("train", "val", "test"):
            (self.dataset_root / "images" / split).mkdir(parents=True)
        dataset_config = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "palletjack"},
        }
        (self.dataset_root / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
        )
        self.config_path = self.fixture_root / "train.yaml"
        self.write_training_config()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_training_config(self, **overrides: object) -> None:
        values: dict[str, object] = {
            "model": "yolo26n.pt",
            "dataset_yaml": self.dataset_root.relative_to(PROJECT_ROOT).as_posix() + "/dataset.yaml",
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "device": None,
            "seed": 42,
            "workers": 4,
            "project": self.fixture_root.relative_to(PROJECT_ROOT).as_posix() + "/runs",
            "name": "baseline",
        }
        values.update(overrides)
        self.config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    def test_loads_config_and_resolves_paths_from_repository_root(self) -> None:
        config = load_training_config(self.config_path)

        self.assertEqual(config.model, "yolo26n.pt")
        self.assertEqual(config.dataset_yaml, self.dataset_root / "dataset.yaml")
        self.assertEqual(config.dataset_root, self.dataset_root)
        self.assertEqual(config.project, self.fixture_root / "runs")
        self.assertEqual(config.run_dir, self.fixture_root / "runs" / "baseline")
        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.imgsz, 640)
        self.assertEqual(config.batch, 16)
        self.assertIsNone(config.device)

    def test_relative_paths_do_not_depend_on_the_current_working_directory(self) -> None:
        original_directory = Path.cwd()
        try:
            os.chdir(self.fixture_root)
            config = load_training_config(self.config_path)
        finally:
            os.chdir(original_directory)

        self.assertEqual(config.dataset_yaml, self.dataset_root / "dataset.yaml")
        self.assertEqual(config.project, self.fixture_root / "runs")

    def test_rejects_missing_dataset_yaml_with_actionable_error(self) -> None:
        self.write_training_config(dataset_yaml="processed/missing.yaml")

        with self.assertRaisesRegex(ValueError, "dataset YAML.*does not exist"):
            load_training_config(self.config_path)

    def test_rejects_missing_train_or_validation_images(self) -> None:
        (self.dataset_root / "images" / "val").rmdir()
        with self.assertRaisesRegex(ValueError, "validation image path.*does not exist"):
            load_training_config(self.config_path)

    def test_rejects_invalid_training_numeric_values(self) -> None:
        for key, value in (("epochs", 0), ("imgsz", 0), ("batch", 0), ("seed", -1), ("workers", -1)):
            with self.subTest(key=key):
                self.write_training_config(**{key: value})
                with self.assertRaisesRegex(ValueError, key):
                    load_training_config(self.config_path)

    def test_allows_ultralytics_auto_batch_value(self) -> None:
        self.write_training_config(batch=-1)

        config = load_training_config(self.config_path)

        self.assertEqual(config.batch, -1)

    def test_rejects_device_indices_below_ultralytics_idle_device_value(self) -> None:
        self.write_training_config(device=-2)

        with self.assertRaisesRegex(ValueError, "device"):
            load_training_config(self.config_path)

    def test_rejects_missing_explicit_local_checkpoint(self) -> None:
        self.write_training_config(model="models/missing.pt")

        with self.assertRaisesRegex(ValueError, "local model checkpoint does not exist"):
            load_training_config(self.config_path)

    def test_rejects_run_names_that_escape_the_project_folder(self) -> None:
        self.write_training_config(name="../outside")

        with self.assertRaisesRegex(ValueError, "one directory name"):
            load_training_config(self.config_path)

    def test_rejects_existing_run_directory_without_changing_it(self) -> None:
        run_dir = self.fixture_root / "runs" / "baseline"
        run_dir.mkdir(parents=True)
        marker = run_dir / "keep.txt"
        marker.write_text("preserve", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "run directory already exists"):
            load_training_config(self.config_path)

        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
