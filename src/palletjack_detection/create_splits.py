"""Create reproducible, scenario-stratified train/validation/test manifests."""

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
    KittiValidationReport,
    validate_kitti_object_detection_export,
)


SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
MANIFEST_FIELDS = ("split", "scenario", "frame_id", "image_path", "annotation_path")


class DatasetValidationError(ValueError):
    """Raised when the source export fails KITTI object-detection validation."""

    def __init__(self, report: KittiValidationReport) -> None:
        self.report = report
        super().__init__(f"source dataset has {len(report.issues)} validation issue(s)")


@dataclass
class SplitManifestResult:
    root: Path
    output_dir: Path
    seed: int
    split_counts: dict[str, dict[str, int]]
    manifest_paths: dict[str, Path]

    @property
    def total_frames(self) -> int:
        return sum(sum(counts.values()) for counts in self.split_counts.values())


def _allocate_split_counts(
    frame_count: int,
    ratios: tuple[float, float, float],
) -> dict[str, int]:
    if frame_count < len(SPLIT_NAMES):
        raise ValueError(f"each scenario needs at least {len(SPLIT_NAMES)} frames to populate every split")

    exact_counts = [frame_count * ratio for ratio in ratios]
    counts = [math.floor(count) for count in exact_counts]
    remainder_order = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(exact_counts[index] - counts[index]), index),
    )
    for index in remainder_order[: frame_count - sum(counts)]:
        counts[index] += 1

    # For small scenario groups, make sure all three requested splits get a frame.
    for empty_index in (1, 2, 0):
        if counts[empty_index] == 0:
            donor_index = max((index for index, count in enumerate(counts) if count > 1), key=counts.__getitem__)
            counts[donor_index] -= 1
            counts[empty_index] += 1

    return dict(zip(SPLIT_NAMES, counts, strict=True))


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("train, validation, and test ratios must all be positive finite numbers")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train, validation, and test ratios must sum to 1.0")


def create_split_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    train_ratio: float = DEFAULT_RATIOS[0],
    validation_ratio: float = DEFAULT_RATIOS[1],
    test_ratio: float = DEFAULT_RATIOS[2],
) -> SplitManifestResult:
    """Validate the source and write per-split CSV manifests without copying data.

    Image and annotation paths in each CSV are relative to ``dataset_root``.
    Each scenario is shuffled independently with the supplied seed, preserving
    its representation in every split.
    """
    ratios = (train_ratio, validation_ratio, test_ratio)
    _validate_ratios(ratios)
    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    validation_report = validate_kitti_object_detection_export(root)
    if not validation_report.ok:
        raise DatasetValidationError(validation_report)

    split_rows: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    split_counts: dict[str, dict[str, int]] = {name: {} for name in SPLIT_NAMES}

    for scenario_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        scenario = scenario_dir.name
        rgb_dir = scenario_dir / "Camera" / "rgb"
        labels_dir = scenario_dir / "Camera" / "object_detection"
        images = sorted(
            (path for path in rgb_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: (path.stem, path.suffix.lower()),
        )
        labels_by_stem = {path.stem: path for path in labels_dir.glob("*.txt")}
        counts = _allocate_split_counts(len(images), ratios)
        split_counts_by_name = {name: 0 for name in SPLIT_NAMES}
        scenario_random = random.Random(f"{seed}:{scenario}")
        shuffled_images = images.copy()
        scenario_random.shuffle(shuffled_images)

        cursor = 0
        for split in SPLIT_NAMES:
            selected = shuffled_images[cursor : cursor + counts[split]]
            cursor += counts[split]
            split_counts_by_name[split] = len(selected)
            for image_path in selected:
                label_path = labels_by_stem[image_path.stem]
                split_rows[split].append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "frame_id": image_path.stem,
                        "image_path": image_path.relative_to(root).as_posix(),
                        "annotation_path": label_path.relative_to(root).as_posix(),
                    }
                )
        for split, count in split_counts_by_name.items():
            split_counts[split][scenario] = count

    destination.mkdir(parents=True, exist_ok=True)
    manifest_paths = {name: destination / f"{name}.csv" for name in SPLIT_NAMES}
    with tempfile.TemporaryDirectory(prefix=".split-manifests-", dir=destination) as temporary_dir:
        temporary_path = Path(temporary_dir)
        for split, rows in split_rows.items():
            rows.sort(key=lambda row: (row["scenario"], row["frame_id"]))
            staged_manifest = temporary_path / f"{split}.csv"
            with staged_manifest.open("w", newline="", encoding="utf-8") as manifest_file:
                writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        for split, manifest_path in manifest_paths.items():
            (temporary_path / f"{split}.csv").replace(manifest_path)

    return SplitManifestResult(
        root=root,
        output_dir=destination,
        seed=seed,
        split_counts=split_counts,
        manifest_paths=manifest_paths,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create deterministic train/validation/test CSV manifests.")
    parser.add_argument("--root", type=Path, default=Path("data/raw"), help="dataset root (default: data/raw)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="manifest output directory (default: data/processed/splits)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    args = parser.parse_args(argv)

    try:
        result = create_split_manifests(
            args.root,
            args.output_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
        )
    except DatasetValidationError as exc:
        print(f"Dataset validation failed: {exc}")
        for issue in exc.report.issues[:50]:
            location = f"{issue.scenario}/{issue.path}" if issue.scenario else issue.path
            print(f"  {location}: {issue.message}")
        if len(exc.report.issues) > 50:
            print(f"  ... {len(exc.report.issues) - 50} additional issue(s) not shown")
        return 1
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Validated {result.total_frames} frames from {result.root}")
    print(f"Split seed: {result.seed}")
    for split in SPLIT_NAMES:
        scenario_counts = result.split_counts[split]
        total = sum(scenario_counts.values())
        print(f"{split}: {total} frames -> {result.manifest_paths[split]}")
        for scenario in sorted(scenario_counts):
            print(f"  {scenario}: {scenario_counts[scenario]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
