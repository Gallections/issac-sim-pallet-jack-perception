from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "data_processing.ipynb"


class DataProcessingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.markdown = "\n".join(
            "".join(cell["source"])
            for cell in cls.cells
            if cell["cell_type"] == "markdown"
        )
        cls.code_cells = [
            "".join(cell["source"])
            for cell in cls.cells
            if cell["cell_type"] == "code"
        ]
        cls.code = "\n\n".join(cls.code_cells)

    def test_notebook_is_valid_unexecuted_jupyter_notebook(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(self.notebook["nbformat_minor"], 4)
        self.assertEqual(len({cell["id"] for cell in self.cells}), len(self.cells))
        for cell in self.cells:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))

    def test_walkthrough_explains_synthetic_and_real_data_workflows(self) -> None:
        for phrase in (
            "Synthetic data",
            "LOCO",
            "KITTI",
            "YOLO",
            "COCO",
            "negative example",
            "65",
            "train",
            "validation",
            "test",
        ):
            self.assertIn(phrase.lower(), self.markdown.lower())
        for function_name in (
            "validate_kitti_object_detection_export",
            "create_stratified_split_manifests",
            "process_raw_export_from_manifests",
            "filter_coco_to_pallet_jack",
            "prepare_loco_pallet_jack_dataset",
            "convert_coco_dataset_to_yolo",
        ):
            self.assertIn(function_name, self.code)

    def test_data_writing_requires_explicit_notebook_opt_in(self) -> None:
        tree = ast.parse(self.code)
        guarded_names = {"RUN_SYNTHETIC_PROCESSING", "RUN_REAL_DATA_PROCESSING", "RUN_REAL_YOLO_CONVERSION"}
        guarded_calls: dict[str, set[str]] = {name: set() for name in guarded_names}
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name):
                continue
            if node.test.id not in guarded_names:
                continue
            guarded_calls[node.test.id].update(
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            )

        self.assertIn("create_stratified_split_manifests", guarded_calls["RUN_SYNTHETIC_PROCESSING"])
        self.assertIn("process_raw_export_from_manifests", guarded_calls["RUN_SYNTHETIC_PROCESSING"])
        self.assertIn("prepare_loco_pallet_jack_dataset", guarded_calls["RUN_REAL_DATA_PROCESSING"])
        self.assertIn("RUN_SYNTHETIC_PROCESSING = False", self.code)
        self.assertIn("RUN_REAL_DATA_PROCESSING = False", self.code)
        self.assertIn("convert_coco_dataset_to_yolo", guarded_calls["RUN_REAL_YOLO_CONVERSION"])
        self.assertIn("RUN_REAL_YOLO_CONVERSION = False", self.code)

    def test_synthetic_paths_are_selected_by_experiment_and_loco_source_is_shared(self) -> None:
        self.assertIn('EXPERIMENT_ID = "experiment-01"', self.code)
        self.assertIn('"raw" / EXPERIMENT_ID', self.code)
        self.assertIn('"processed" / EXPERIMENT_ID', self.code)
        self.assertIn('"raw" / "experiment-01" / "real-pallet-jack-data"', self.code)


if __name__ == "__main__":
    unittest.main()
