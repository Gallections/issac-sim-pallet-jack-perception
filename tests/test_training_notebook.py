from __future__ import annotations

import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "explore_training_results.ipynb"


class TrainingResultsNotebookTests(unittest.TestCase):
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

    def test_notebook_is_valid_jupyter_notebook(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreaterEqual(self.notebook["nbformat_minor"], 4)
        self.assertEqual(len({cell["id"] for cell in self.cells}), len(self.cells))
        for cell in self.cells:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_notebook_documents_run_selection_and_saved_result_exploration(self) -> None:
        self.assertIn("Select a training run", self.markdown)
        self.assertIn("results.csv", self.code)
        self.assertIn("confusion_matrix.png", self.code)
        self.assertIn("val_batch", self.code)
        self.assertIn("metrics/mAP50-95(B)", self.code)
        self.assertIn("metrics/mAP50-95(B)", self.code)
        self.assertIn("Plot training losses and validation metrics", self.markdown)

    def test_missing_run_artifacts_show_recovery_guidance(self) -> None:
        self.assertIn("choose another run", self.code.lower())
        self.assertIn("results.csv was not found", self.code)
        self.assertIn("not found", self.code.lower())

    def test_analysis_cells_execute_with_saved_run_artifact_fixture(self) -> None:
        from PIL import Image as PILImage

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            fixture_root = Path(temporary_directory)
            (fixture_root / "configs").mkdir()
            run_dir = fixture_root / "runs" / "detect" / "palletjack-detection-yolo26n-seed42"
            run_dir.mkdir(parents=True)
            (run_dir / "results.csv").write_text(
                "epoch,train/box_loss,val/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
                "0,1.0,1.2,0.8,0.7,0.75,0.5\n"
                "1,0.8,1.0,0.9,0.8,0.85,0.65\n",
                encoding="utf-8",
            )
            for image_name in (
                "BoxPR_curve.png",
                "confusion_matrix.png",
                "confusion_matrix_normalized.png",
                "val_batch0_labels.jpg",
                "val_batch0_pred.jpg",
            ):
                PILImage.new("RGB", (2, 2), color="white").save(run_dir / image_name)

            namespace: dict[str, object] = {}
            code_cells = [source for source in self.code_cells]
            with patch("pathlib.Path.cwd", return_value=fixture_root), patch(
                "matplotlib.pyplot.show"
            ), contextlib.redirect_stdout(io.StringIO()) as output:
                exec(code_cells[0], namespace)
                exec(code_cells[1], namespace)
                exec(code_cells[2], namespace)

            self.assertIn("Epoch with best validation mAP50-95: 1 0.65", output.getvalue())

    def test_missing_results_csv_is_reported_without_stopping_notebook(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            fixture_root = Path(temporary_directory)
            (fixture_root / "configs").mkdir()
            (
                fixture_root
                / "runs"
                / "detect"
                / "palletjack-detection-yolo26n-seed42"
            ).mkdir(parents=True)
            namespace: dict[str, object] = {}
            with patch("pathlib.Path.cwd", return_value=fixture_root), contextlib.redirect_stdout(
                io.StringIO()
            ) as output:
                exec(self.code_cells[0], namespace)
                exec(self.code_cells[1], namespace)

            self.assertIn("results.csv was not found", output.getvalue())

    def test_real_image_inference_requires_an_explicit_opt_in(self) -> None:
        inference_cell = next(
            source for source in self.code_cells if "predict_real_images(" in source
        )
        tree = ast.parse(inference_cell)
        guarded = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "RUN_REAL_IMAGE_PREDICTION"
        ]
        self.assertEqual(len(guarded), 1)
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "predict_real_images"
                for node in ast.walk(guarded[0])
            )
        )

    def test_training_launch_requires_explicit_opt_in(self) -> None:
        training_cell = next(
            source for source in self.code_cells if "train_yolo.py" in source
        )
        tree = ast.parse(training_cell)
        guarded = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "START_TRAINING"
        ]
        self.assertEqual(len(guarded), 1)
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                for node in ast.walk(guarded[0])
            )
        )


if __name__ == "__main__":
    unittest.main()

