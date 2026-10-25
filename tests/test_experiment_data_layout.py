from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from palletjack_detection import data_paths
from palletjack_detection.coco_to_yolo import DEFAULT_DATASET_ROOT as COCO_YOLO_ROOT
from palletjack_detection.create_splits import DEFAULT_PROCESSED_ROOT, DEFAULT_RAW_ROOT, DEFAULT_SPLITS_ROOT
from palletjack_detection.evaluate_test_set import DEFAULT_DATASET_YAML as TEST_DATASET_YAML
from palletjack_detection.process_loco_coco import DEFAULT_OUTPUT_ROOT as LOCO_PROCESSED_ROOT
from palletjack_detection.process_loco_coco import DEFAULT_RAW_ROOT as LOCO_RAW_ROOT
from palletjack_detection.process_raw_data import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_OUTPUT_ROOT as RAW_PROCESSING_OUTPUT_ROOT,
    DEFAULT_RAW_ROOT as RAW_PROCESSING_ROOT,
)
from palletjack_detection.validate_dataset import DEFAULT_DATASET_ROOT as VALIDATION_ROOT


class ExperimentDataLayoutTests(unittest.TestCase):
    def test_experiment_roots_are_nested_under_raw_and_processed(self) -> None:
        expected_raw = PROJECT_ROOT / "data" / "raw" / "experiment-01"
        expected_processed = PROJECT_ROOT / "data" / "processed" / "experiment-01"

        self.assertEqual(data_paths.DEFAULT_RAW_ROOT, expected_raw)
        self.assertEqual(data_paths.DEFAULT_PROCESSED_ROOT, expected_processed)
        self.assertEqual(data_paths.raw_experiment_root("experiment-02"), PROJECT_ROOT / "data" / "raw" / "experiment-02")
        self.assertEqual(
            data_paths.processed_experiment_root("experiment-02"),
            PROJECT_ROOT / "data" / "processed" / "experiment-02",
        )

    def test_experiment_path_helper_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "experiment_id"):
            data_paths.raw_experiment_root("../outside")

    def test_synthetic_pipeline_defaults_all_point_to_experiment_01(self) -> None:
        expected_raw = PROJECT_ROOT / "data" / "raw" / "experiment-01"
        expected_processed = PROJECT_ROOT / "data" / "processed" / "experiment-01"

        self.assertEqual(VALIDATION_ROOT, expected_raw)
        self.assertEqual(DEFAULT_RAW_ROOT, expected_raw)
        self.assertEqual(DEFAULT_PROCESSED_ROOT, expected_processed)
        self.assertEqual(DEFAULT_SPLITS_ROOT, expected_processed / "splits")
        self.assertEqual(RAW_PROCESSING_ROOT, expected_raw)
        self.assertEqual(DEFAULT_MANIFEST_DIR, expected_processed / "splits")
        self.assertEqual(RAW_PROCESSING_OUTPUT_ROOT, expected_processed)
        self.assertEqual(TEST_DATASET_YAML, expected_processed / "dataset.yaml")

    def test_real_loco_defaults_stay_inside_experiment_01(self) -> None:
        expected_raw = PROJECT_ROOT / "data" / "raw" / "experiment-01" / "real-pallet-jack-data"
        expected_processed = PROJECT_ROOT / "data" / "processed" / "experiment-01" / "real-pallet-jack-data"

        self.assertEqual(LOCO_RAW_ROOT, expected_raw)
        self.assertEqual(LOCO_PROCESSED_ROOT, expected_processed)
        self.assertEqual(COCO_YOLO_ROOT, expected_processed)

    def test_default_training_config_uses_experiment_01_dataset_yaml(self) -> None:
        values = yaml.safe_load((PROJECT_ROOT / "configs" / "train_yolo.yaml").read_text(encoding="utf-8"))

        self.assertEqual(values["dataset_yaml"], "data/processed/experiment-01/dataset.yaml")

    def test_experiment_02_training_config_uses_experiment_02_dataset_and_run_root(self) -> None:
        values = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "train_yolo_experiment-02.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(values["dataset_yaml"], "data/processed/experiment-02/dataset.yaml")
        self.assertEqual(values["project"], "runs/detect/experiment-02")


if __name__ == "__main__":
    unittest.main()
