"""Load and validate the settings for one YOLO object-detection run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_CONFIG_KEYS = (
    "model",
    "dataset_yaml",
    "epochs",
    "imgsz",
    "batch",
    "device",
    "seed",
    "workers",
    "project",
    "name",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Resolved, validated settings used to fine-tune one detector."""

    model: str
    dataset_yaml: Path
    dataset_root: Path
    epochs: int
    imgsz: int
    batch: int
    device: int | str | None
    seed: int
    workers: int
    project: Path
    name: str
    config_path: Path
    repository_root: Path

    @property
    def run_dir(self) -> Path:
        """Return the intended output folder for this configured run."""
        return self.project / self.name


def _resolve_project_path(value: str, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _resolve_model_identifier(value: str, repository_root: Path) -> str:
    """Resolve explicit local checkpoint paths but preserve Ultralytics model IDs."""
    model_path = Path(value).expanduser()
    looks_like_path = model_path.is_absolute() or "/" in value or "\\" in value
    if not looks_like_path:
        return value
    resolved_path = model_path if model_path.is_absolute() else repository_root / model_path
    resolved_path = resolved_path.resolve()
    if not resolved_path.is_file():
        raise ValueError(f"local model checkpoint does not exist: {resolved_path}")
    return str(resolved_path)


def _read_training_config_values(config_path: Path) -> dict[str, Any]:
    try:
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"training config is not valid YAML: {config_path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError(f"training config must contain a YAML mapping: {config_path}")
    missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in values]
    if missing:
        raise ValueError(f"training config is missing required keys: {', '.join(missing)}")
    return values


def _require_nonempty_string(values: dict[str, Any], key: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_integer(values: dict[str, Any], key: str, *, minimum: int) -> int:
    value = values[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer greater than or equal to {minimum}")
    return value


def _resolve_dataset_root(dataset_yaml: Path, dataset_values: dict[str, Any]) -> Path:
    configured_root = dataset_values.get("path", ".")
    if not isinstance(configured_root, str) or not configured_root.strip():
        raise ValueError(f"dataset YAML path must be a non-empty string: {dataset_yaml}")
    path = Path(configured_root).expanduser()
    if not path.is_absolute():
        path = dataset_yaml.parent / path
    return path.resolve()


def _validate_dataset_yaml(dataset_yaml: Path) -> Path:
    if not dataset_yaml.is_file():
        raise ValueError(f"dataset YAML does not exist: {dataset_yaml}")
    try:
        dataset_values = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"dataset YAML is not valid YAML: {dataset_yaml}: {exc}") from exc
    if not isinstance(dataset_values, dict):
        raise ValueError(f"dataset YAML must contain a YAML mapping: {dataset_yaml}")
    names = dataset_values.get("names")
    if not isinstance(names, (dict, list)) or not names:
        raise ValueError(f"dataset YAML must define a non-empty class names mapping: {dataset_yaml}")

    dataset_root = _resolve_dataset_root(dataset_yaml, dataset_values)
    for split_key, split_label in (("train", "training"), ("val", "validation")):
        split_value = dataset_values.get(split_key)
        split_values = split_value if isinstance(split_value, list) else [split_value]
        if not split_values or any(not isinstance(item, str) or not item.strip() for item in split_values):
            raise ValueError(f"dataset YAML must define a {split_label} image path: {dataset_yaml}")
        for item in split_values:
            split_path = Path(item).expanduser()
            if not split_path.is_absolute():
                split_path = dataset_root / split_path
            if not split_path.exists():
                raise ValueError(f"{split_label} image path does not exist: {split_path.resolve()}")
    return dataset_root


def load_training_config(
    config_path: str | Path,
    *,
    repository_root: str | Path = PROJECT_ROOT,
) -> TrainingConfig:
    """Read a YAML config, resolve its paths, and reject unusable runs early."""
    repository_root = Path(repository_root).expanduser().resolve()
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = repository_root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ValueError(f"training config does not exist: {config_path}")
    values = _read_training_config_values(config_path)

    model_value = _require_nonempty_string(values, "model")
    dataset_yaml_value = _require_nonempty_string(values, "dataset_yaml")
    project_value = _require_nonempty_string(values, "project")
    name = _require_nonempty_string(values, "name")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("name must be one directory name without path separators")

    epochs = _require_integer(values, "epochs", minimum=1)
    imgsz = _require_integer(values, "imgsz", minimum=1)
    batch = values["batch"]
    if not isinstance(batch, int) or isinstance(batch, bool) or (batch != -1 and batch < 1):
        raise ValueError("batch must be a positive integer or -1 for automatic batch sizing")
    seed = _require_integer(values, "seed", minimum=0)
    workers = _require_integer(values, "workers", minimum=0)
    device = values["device"]
    if isinstance(device, bool) or not (device is None or isinstance(device, (int, str))):
        raise ValueError("device must be null, a device index, or a device string")
    if isinstance(device, int) and device < -1:
        raise ValueError("device must be null, -1, a nonnegative device index, or a device string")
    if isinstance(device, str):
        device = device.strip()
        if not device:
            raise ValueError("device must be null, a device index, or a non-empty device string")

    dataset_yaml = _resolve_project_path(dataset_yaml_value, repository_root)
    dataset_root = _validate_dataset_yaml(dataset_yaml)
    project = _resolve_project_path(project_value, repository_root)
    run_dir = project / name
    if run_dir.exists():
        raise ValueError(f"run directory already exists; choose another name: {run_dir}")

    return TrainingConfig(
        model=_resolve_model_identifier(model_value, repository_root),
        dataset_yaml=dataset_yaml,
        dataset_root=dataset_root,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        workers=workers,
        project=project,
        name=name,
        config_path=config_path,
        repository_root=repository_root,
    )


def validate_training_config(config: TrainingConfig) -> None:
    """Recheck inputs immediately before training, including the intended run path."""
    _validate_dataset_yaml(config.dataset_yaml)
    if config.run_dir.exists():
        raise ValueError(f"run directory already exists; choose another name: {config.run_dir}")
