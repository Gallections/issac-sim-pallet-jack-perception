from __future__ import annotations

from pathlib import Path
import unittest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(relative_path: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


class Experiment04ConfigTests(unittest.TestCase):
    def test_initialization_configs_share_one_dataset_and_training_recipe(self) -> None:
        pretrained = _read_yaml("configs/train_yolo_e4_pretrained.yaml")
        scratch = _read_yaml("configs/train_yolo_e4_scratch.yaml")

        self.assertEqual(pretrained["model"], "yolo26n.pt")
        self.assertEqual(scratch["model"], "yolo26n.yaml")
        self.assertEqual(pretrained["dataset_yaml"], "configs/experiment-04-dataset.yaml")
        self.assertEqual(scratch["dataset_yaml"], "configs/experiment-04-dataset.yaml")
        for key in ("epochs", "imgsz", "batch", "device", "seed", "workers", "project"):
            with self.subTest(key=key):
                self.assertEqual(pretrained[key], scratch[key])
        self.assertNotEqual(pretrained["name"], scratch["name"])

    def test_dataset_combines_only_target_compatible_experiment_splits(self) -> None:
        dataset = _read_yaml("configs/experiment-04-dataset.yaml")

        self.assertEqual(dataset["path"], "../data/processed/experiment-01")
        self.assertEqual(dataset["names"], {0: "palletjack"})
        self.assertEqual(dataset["train"], ["images/train", "../experiment-03/images/train"])
        self.assertEqual(dataset["val"], ["images/val", "../experiment-03/images/val"])
        self.assertEqual(dataset["test"], ["images/test", "../experiment-03/images/test"])


if __name__ == "__main__":
    unittest.main()
