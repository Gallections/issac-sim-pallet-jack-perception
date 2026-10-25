"""Evaluate one selected detector checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from palletjack_detection.data_paths import DEFAULT_PROCESSED_ROOT, PROJECT_ROOT as DATA_PROJECT_ROOT


PROJECT_ROOT = DATA_PROJECT_ROOT
DEFAULT_CHECKPOINT = Path("runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt")
DEFAULT_DATASET_YAML = DEFAULT_PROCESSED_ROOT / "dataset.yaml"
DEFAULT_RUN_NAME = "palletjack-detection-yolo26n-seed42-test"


def _resolve_project_path(path: str | Path, repository_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = repository_root / resolved
    return resolved.resolve()


def _load_test_dataset_config(dataset_yaml: Path) -> tuple[dict[str, Any], Path]:
    if not dataset_yaml.is_file():
        raise ValueError(f"dataset YAML does not exist: {dataset_yaml}")
    try:
        values = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"dataset YAML is not valid YAML: {dataset_yaml}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"dataset YAML must contain a YAML mapping: {dataset_yaml}")
    test_value = values.get("test")
    test_paths = test_value if isinstance(test_value, list) else [test_value]
    if not test_paths or any(not isinstance(path, str) or not path.strip() for path in test_paths):
        raise ValueError(f"dataset YAML must define a test image path: {dataset_yaml}")

    configured_root = values.get("path", ".")
    if not isinstance(configured_root, str) or not configured_root.strip():
        raise ValueError(f"dataset YAML path must be a non-empty string: {dataset_yaml}")
    dataset_root = Path(configured_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = dataset_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()
    for test_path in test_paths:
        resolved_test_path = Path(test_path).expanduser()
        if not resolved_test_path.is_absolute():
            resolved_test_path = dataset_root / resolved_test_path
        if not resolved_test_path.exists():
            raise ValueError(f"test image path does not exist: {resolved_test_path.resolve()}")
    values["path"] = str(dataset_root)
    return values, dataset_root


def evaluate_held_out_test_set(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    dataset_yaml: str | Path = DEFAULT_DATASET_YAML,
    repository_root: str | Path = PROJECT_ROOT,
    project: str | Path = "runs/detect",
    name: str = DEFAULT_RUN_NAME,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 4,
    device: int | str | None = None,
) -> Path:
    """Run test-only validation, saving plots and a machine-readable metric summary."""
    repository_root = Path(repository_root).expanduser().resolve()
    checkpoint_path = _resolve_project_path(checkpoint, repository_root)
    dataset_yaml_path = _resolve_project_path(dataset_yaml, repository_root)
    project_path = _resolve_project_path(project, repository_root)
    if not checkpoint_path.is_file():
        raise ValueError(f"model checkpoint does not exist: {checkpoint_path}")
    if not isinstance(name, str) or not name.strip() or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("name must be one non-empty directory name without path separators")
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz < 1:
        raise ValueError("imgsz must be a positive integer")
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError("batch must be a positive integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    if device is not None and not isinstance(device, (int, str)):
        raise ValueError("device must be null, a device index, or a device string")

    dataset_values, dataset_root = _load_test_dataset_config(dataset_yaml_path)
    run_directory = project_path / name
    if run_directory.exists():
        raise FileExistsError(f"evaluation output directory already exists: {run_directory}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Ultralytics is required; install project dependencies from requirements.txt") from exc

    local_directory = repository_root / "local"
    local_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="palletjack-test-dataset-", dir=local_directory) as temp_dir:
        resolved_dataset_yaml = Path(temp_dir) / "dataset.yaml"
        resolved_dataset_yaml.write_text(yaml.safe_dump(dataset_values, sort_keys=False), encoding="utf-8")
        model = YOLO(str(checkpoint_path))
        metrics = model.val(
            data=str(resolved_dataset_yaml),
            split="test",
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            plots=True,
            project=str(project_path),
            name=name,
            exist_ok=False,
        )

    box_metrics = metrics.box
    summary = {
        "model_checkpoint": str(checkpoint_path),
        "dataset_yaml": str(dataset_yaml_path),
        "dataset_root": str(dataset_root),
        "split": "test",
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "metrics": {
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
            "mAP50": float(box_metrics.map50),
            "mAP50-95": float(box_metrics.map),
        },
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "test_metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Ultralytics detector on the held-out test split only."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT, help="trained .pt checkpoint")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_YAML, help="dataset YAML with a test split")
    parser.add_argument("--imgsz", type=int, default=640, help="inference input size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="validation batch size (default: 16)")
    parser.add_argument("--workers", type=int, default=4, help="data loader workers (default: 4)")
    parser.add_argument("--device", default=None, help="device such as 0 or cpu (default: automatic)")
    parser.add_argument("--project", default="runs/detect", help="output parent directory, relative to repository root")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="new evaluation folder name; existing folders are protected")
    args = parser.parse_args(argv)
    try:
        run_directory = evaluate_held_out_test_set(
            checkpoint=args.model,
            dataset_yaml=args.dataset,
            project=args.project,
            name=args.name,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    summary = json.loads((run_directory / "test_metrics.json").read_text(encoding="utf-8"))
    print(f"Held-out test metrics saved to: {run_directory}")
    for metric_name, value in summary["metrics"].items():
        print(f"{metric_name}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
