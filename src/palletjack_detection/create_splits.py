"""Assign validated source frames to reproducible train, validation, and test splits."""

from __future__ import annotations

import argparse
import csv
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

from palletjack_detection.validate_dataset import (
    IMAGE_EXTENSIONS,
    KittiExportValidationError,
    validate_kitti_object_detection_export,
)

SPLIT_NAMES = ("train", "validation", "test")
YOLO_SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_MANIFEST_FIELDS = (
    "split",
    "scenario",
    "frame_id",
    "source_image_path",
    "kitti_annotation_path",
    "processed_image_path",
    "processed_kitti_annotation_path",
    "yolo_annotation_path",
)


@dataclass
class SplitManifestResult:
    source_root: Path
    manifest_dir: Path
    processed_root: Path
    seed: int
    split_counts: dict[str, dict[str, int]]
    manifest_paths: dict[str, Path]

    @property
    def total_frames(self) -> int:
        return sum(sum(scenario_counts.values()) for scenario_counts in self.split_counts.values())


def _allocate_frame_counts(frame_count: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    if frame_count < len(SPLIT_NAMES):
        raise ValueError(f"each scenario needs at least {len(SPLIT_NAMES)} frames to populate every split")
    exact_counts = [frame_count * ratio for ratio in ratios]
    allocated_counts = [math.floor(count) for count in exact_counts]
    remainder_order = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(exact_counts[index] - allocated_counts[index]), index),
    )
    for index in remainder_order[: frame_count - sum(allocated_counts)]:
        allocated_counts[index] += 1
    for empty_index in (1, 2, 0):
        if allocated_counts[empty_index] == 0:
            donor_index = max(
                (index for index, count in enumerate(allocated_counts) if count > 1),
                key=allocated_counts.__getitem__,
            )
            allocated_counts[donor_index] -= 1
            allocated_counts[empty_index] += 1
    return dict(zip(SPLIT_NAMES, allocated_counts, strict=True))


def _validate_train_validation_test_ratios(ratios: tuple[float, float, float]) -> None:
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("train, validation, and test ratios must all be positive finite numbers")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train, validation, and test ratios must sum to 1.0")


def assign_scenario_stratified_split_rows(
    source_root: Path,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, int]]]:
    """Assign each source image once and record its future processed-data paths."""
    split_rows: dict[str, list[dict[str, str]]] = {split: [] for split in SPLIT_NAMES}
    split_counts: dict[str, dict[str, int]] = {split: {} for split in SPLIT_NAMES}
    scenario_dirs = sorted((path for path in source_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        rgb_dir = scenario_dir / "Camera" / "rgb"
        kitti_dir = scenario_dir / "Camera" / "object_detection"
        images = sorted(
            (path for path in rgb_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: (path.stem, path.suffix.lower()),
        )
        kitti_by_stem = {path.stem: path for path in kitti_dir.glob("*.txt")}
        split_counts_for_scenario = _allocate_frame_counts(len(images), ratios)
        scenario_random = random.Random(f"{seed}:{scenario}")
        shuffled_images = images.copy()
        scenario_random.shuffle(shuffled_images)
        next_image_index = 0

        for split_name in SPLIT_NAMES:
            selected_images = shuffled_images[
                next_image_index : next_image_index + split_counts_for_scenario[split_name]
            ]
            next_image_index += split_counts_for_scenario[split_name]
            split_counts[split_name][scenario] = len(selected_images)
            yolo_split = YOLO_SPLIT_NAMES[split_name]
            for image_path in selected_images:
                kitti_path = kitti_by_stem[image_path.stem]
                processed_stem = Path(yolo_split) / scenario / image_path.name
                processed_kitti_stem = Path(yolo_split) / scenario / f"{image_path.stem}.txt"
                split_rows[split_name].append(
                    {
                        "split": split_name,
                        "scenario": scenario,
                        "frame_id": image_path.stem,
                        "source_image_path": image_path.relative_to(source_root).as_posix(),
                        "kitti_annotation_path": kitti_path.relative_to(source_root).as_posix(),
                        "processed_image_path": (Path("images") / processed_stem).as_posix(),
                        "processed_kitti_annotation_path": (Path("kitti_labels") / processed_kitti_stem).as_posix(),
                        "yolo_annotation_path": (Path("labels") / processed_kitti_stem).as_posix(),
                    }
                )
    return split_rows, split_counts


def create_stratified_split_manifests(
    source_root: str | Path,
    manifest_dir: str | Path,
    *,
    processed_root: str | Path | None = None,
    seed: int = 42,
    train_ratio: float = DEFAULT_SPLIT_RATIOS[0],
    validation_ratio: float = DEFAULT_SPLIT_RATIOS[1],
    test_ratio: float = DEFAULT_SPLIT_RATIOS[2],
    overwrite: bool = False,
) -> SplitManifestResult:
    """Validate and split source frames; write manifests without copying data."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    _validate_train_validation_test_ratios(ratios)
    root = Path(source_root).expanduser().resolve()
    output_dir = Path(manifest_dir).expanduser().resolve()
    processed_path = Path(processed_root).expanduser().resolve() if processed_root else output_dir.parent
    validation_report = validate_kitti_object_detection_export(root)
    if not validation_report.ok:
        raise KittiExportValidationError(validation_report)

    split_rows, split_counts = assign_scenario_stratified_split_rows(root, ratios, seed)
    manifest_paths = {split: output_dir / f"{split}.csv" for split in SPLIT_NAMES}
    if not overwrite:
        existing_manifests = [path for path in manifest_paths.values() if path.exists()]
        if existing_manifests:
            raise FileExistsError(f"split manifest already exists: {existing_manifests[0]} (pass overwrite=True to regenerate)")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".split-manifests-", dir=output_dir) as temporary_dir:
        staged_manifests: dict[str, Path] = {}
        for split, rows in split_rows.items():
            rows.sort(key=lambda row: (row["scenario"], row["frame_id"]))
            staged_path = Path(temporary_dir) / f"{split}.csv"
            with staged_path.open("w", newline="", encoding="utf-8") as manifest_file:
                writer = csv.DictWriter(manifest_file, fieldnames=SPLIT_MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            staged_manifests[split] = staged_path
        for split, output_path in manifest_paths.items():
            staged_manifests[split].replace(output_path)

    return SplitManifestResult(
        source_root=root,
        manifest_dir=output_dir,
        processed_root=processed_path,
        seed=seed,
        split_counts=split_counts,
        manifest_paths=manifest_paths,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create deterministic train/validation/test split manifests.")
    parser.add_argument("--root", type=Path, default=Path("data/raw"), help="source dataset root")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/splits"), help="manifest directory")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"), help="future processed-data root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0])
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2])
    parser.add_argument("--overwrite", action="store_true", help="replace existing split manifests")
    args = parser.parse_args(argv)
    try:
        result = create_stratified_split_manifests(
            args.root,
            args.output_dir,
            processed_root=args.processed_root,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            overwrite=args.overwrite,
        )
    except KittiExportValidationError as exc:
        print(f"Dataset validation failed: {exc}")
        for issue in exc.report.issues[:50]:
            location = f"{issue.scenario}/{issue.path}" if issue.scenario else issue.path
            print(f"  {location}: {issue.message}")
        if len(exc.report.issues) > 50:
            print(f"  ... {len(exc.report.issues) - 50} additional issue(s) not shown")
        return 1
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"Could not create split manifests: {exc}")
        return 1

    print(f"Created split manifests for {result.total_frames} frames from {result.source_root}")
    print(f"Split seed: {result.seed}; processed output paths are relative to {result.processed_root}")
    for split in SPLIT_NAMES:
        scenario_counts = result.split_counts[split]
        print(f"{split}: {sum(scenario_counts.values())} frames -> {result.manifest_paths[split]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
