"""Fine-tune one configured Ultralytics YOLO object detector."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from palletjack_detection.training_config import (
    PROJECT_ROOT,
    TrainingConfig,
    load_training_config,
    validate_training_config,
)


def _read_resolved_dataset_config(config: TrainingConfig) -> dict[str, Any]:
    dataset_values = yaml.safe_load(config.dataset_yaml.read_text(encoding="utf-8"))
    dataset_values["path"] = str(config.dataset_root)
    return dataset_values


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _save_run_reproducibility_files(
    run_dir: Path,
    config: TrainingConfig,
    resolved_dataset_config: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_config.yaml").write_text(
        config.config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (run_dir / "dataset.yaml").write_text(
        yaml.safe_dump(resolved_dataset_config, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "python_version": sys.version,
        "ultralytics_version": _package_version("ultralytics"),
        "torch_version": _package_version("torch"),
        "run_directory": str(run_dir.resolve()),
    }
    (run_dir / "runtime_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def run_ultralytics_training(config: TrainingConfig) -> Path:
    """Run configured fine-tuning and save the resolved experiment inputs."""
    validate_training_config(config)
    # Import lazily so config preflight works without loading PyTorch or Ultralytics.
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed in this Python environment; install the optional "
            "dependencies from requirements.txt"
        ) from exc

    resolved_dataset_config = _read_resolved_dataset_config(config)
    temporary_root = config.repository_root / "local"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="palletjack-yolo-data-", dir=temporary_root
    ) as temporary_directory:
        training_dataset_yaml = Path(temporary_directory) / "dataset.yaml"
        training_dataset_yaml.write_text(
            yaml.safe_dump(resolved_dataset_config, sort_keys=False), encoding="utf-8"
        )
        model = YOLO(config.model)
        try:
            model.train(
                data=str(training_dataset_yaml),
                epochs=config.epochs,
                imgsz=config.imgsz,
                batch=config.batch,
                device=config.device,
                seed=config.seed,
                workers=config.workers,
                project=str(config.project),
                name=config.name,
                exist_ok=False,
                save=True,
                plots=True,
            )
        finally:
            trainer = getattr(model, "trainer", None)
            trainer_save_dir = getattr(trainer, "save_dir", None)
            run_dir = Path(trainer_save_dir).resolve() if trainer_save_dir else config.run_dir
            if run_dir.exists():
                _save_run_reproducibility_files(run_dir, config, resolved_dataset_config)

    return run_dir


def _print_training_config(config: TrainingConfig) -> None:
    device = "automatic" if config.device is None else config.device
    print("Configuration is valid.")
    print(f"Model: {config.model}")
    print(f"Dataset YAML: {config.dataset_yaml}")
    print(f"Dataset root: {config.dataset_root}")
    print(f"Epochs: {config.epochs}")
    print(f"Training input size: {config.imgsz}")
    print(f"Batch size: {config.batch}")
    print(f"Device: {device}")
    print(f"Random seed: {config.seed}")
    print(f"Run directory: {config.run_dir}")


def main(argv: list[str] | None = None) -> int:
    """Validate the training settings or launch one configured run."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained Ultralytics YOLO detector on the processed dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_yolo.yaml",
        help="training YAML configuration (default: configs/train_yolo.yaml)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check the configuration and dataset paths without loading a model or starting training",
    )
    args = parser.parse_args(argv)

    try:
        config = load_training_config(args.config)
        validate_training_config(config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    _print_training_config(config)
    if args.validate_only:
        print("Validation only: no model was loaded and no training was started.")
        return 0

    try:
        run_dir = run_ultralytics_training(config)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Training finished. Run artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
