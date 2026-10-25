"""Read-only validation for Isaac Sim KITTI-style object detection exports."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {"palletjack"}
KITTI_FIELD_COUNT = 15


@dataclass
class KittiValidationIssue:
    scenario: str
    path: str
    message: str


@dataclass
class KittiScenarioSummary:
    name: str
    image_count: int = 0
    annotation_count: int = 0
    object_count: int = 0
    empty_annotation_count: int = 0


@dataclass
class KittiValidationReport:
    root: Path
    scenarios: list[KittiScenarioSummary] = field(default_factory=list)
    issues: list[KittiValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


class KittiExportValidationError(ValueError):
    """Raised when a workflow requires a valid export but validation finds issues."""

    def __init__(self, report: KittiValidationReport) -> None:
        self.report = report
        super().__init__(f"source dataset has {len(report.issues)} validation issue(s)")


def _add_issue(report: KittiValidationReport, scenario: str, path: Path | str, message: str) -> None:
    try:
        base = report.root / scenario if scenario else report.root
        display_path = str(Path(path).relative_to(base))
    except (ValueError, TypeError):
        display_path = str(path)
    report.issues.append(KittiValidationIssue(scenario, display_path, message))


def _validate_kitti_label_file(
    path: Path,
    scenario: str,
    report: KittiValidationReport,
    image_size: tuple[int, int] | None,
    summary: KittiScenarioSummary,
) -> None:
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        _add_issue(report, scenario, path, f"cannot read annotation: {exc}")
        return

    records = [(line_number, line.split()) for line_number, line in enumerate(contents.splitlines(), 1) if line.strip()]
    if not records:
        summary.empty_annotation_count += 1
        return

    for line_number, fields in records:
        location = f"line {line_number}: "
        if len(fields) != KITTI_FIELD_COUNT:
            _add_issue(report, scenario, path, f"{location}expected 15 fields, found {len(fields)}")
            continue

        class_name = fields[0]
        record_is_valid = True
        if class_name not in CLASS_NAMES:
            _add_issue(report, scenario, path, f"{location}unknown class {class_name!r}")
            record_is_valid = False

        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            _add_issue(report, scenario, path, f"{location}non-numeric KITTI field")
            continue
        if not all(math.isfinite(value) for value in values):
            _add_issue(report, scenario, path, f"{location}KITTI fields must be finite numbers")
            continue

        occlusion = values[1]
        if not occlusion.is_integer() or not 0 <= occlusion <= 3:
            _add_issue(report, scenario, path, f"{location}occlusion must be an integer from 0 through 3")
            record_is_valid = False

        x_min, y_min, x_max, y_max = values[3:7]
        if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
            _add_issue(report, scenario, path, f"{location}box coordinates must satisfy 0 <= min < max")
            continue
        if image_size is not None:
            width, height = image_size
            if x_max > width or y_max > height:
                _add_issue(
                    report,
                    scenario,
                    path,
                    f"{location}box ({x_min:g}, {y_min:g}, {x_max:g}, {y_max:g}) exceeds image bounds {width}x{height}",
                )
                continue
        if record_is_valid:
            summary.object_count += 1


def _select_scenario_directories(root_path: Path, scenario_names: Sequence[str] | None) -> list[Path]:
    available = {path.name: path for path in root_path.iterdir() if path.is_dir()}
    if scenario_names is None:
        return sorted(available.values(), key=lambda path: path.name)
    if not scenario_names:
        raise ValueError("scenario_names must contain at least one scenario")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("scenario_names must not contain duplicates")
    for name in scenario_names:
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"scenario name must be a direct child directory name: {name!r}")
        if name not in available:
            raise ValueError(f"requested scenario does not exist: {name}")
    return [available[name] for name in sorted(scenario_names)]


def validate_kitti_object_detection_export(
    root: str | Path,
    *,
    scenario_names: Sequence[str] | None = None,
) -> KittiValidationReport:
    """Check scenario image/label pairs and KITTI 2D boxes without modifying data."""
    root_path = Path(root).expanduser().resolve()
    report = KittiValidationReport(root=root_path)
    if not root_path.is_dir():
        _add_issue(report, "", root_path, "dataset root does not exist or is not a directory")
        return report

    scenario_dirs = _select_scenario_directories(root_path, scenario_names)
    if not scenario_dirs:
        _add_issue(report, "", root_path, "dataset root contains no scenario directories")
        return report

    for scenario_dir in scenario_dirs:
        scenario = scenario_dir.name
        summary = KittiScenarioSummary(name=scenario)
        report.scenarios.append(summary)
        camera_dir = scenario_dir / "Camera"
        rgb_dir = camera_dir / "rgb"
        label_dir = camera_dir / "object_detection"
        for required_dir in (rgb_dir, label_dir):
            if not required_dir.is_dir():
                _add_issue(report, scenario, required_dir, "required directory is missing")

        images = sorted(
            (path for path in rgb_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: (path.stem, path.suffix.lower()),
        ) if rgb_dir.is_dir() else []
        labels = sorted(label_dir.glob("*.txt"), key=lambda path: path.name) if label_dir.is_dir() else []
        summary.image_count = len(images)
        summary.annotation_count = len(labels)

        images_by_stem: dict[str, list[Path]] = {}
        for image_path in images:
            images_by_stem.setdefault(image_path.stem, []).append(image_path)
        labels_by_stem = {path.stem: path for path in labels}

        for stem, matches in images_by_stem.items():
            if len(matches) > 1:
                _add_issue(report, scenario, matches[0], f"multiple RGB images share frame ID {stem!r}")
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image_size = image.size
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                _add_issue(report, scenario, image_path, f"image cannot be opened: {exc}")
                image_size = None
            if image_path.stem not in labels_by_stem:
                _add_issue(report, scenario, image_path, "no annotation file with matching frame ID")
                continue
            _validate_kitti_label_file(labels_by_stem[image_path.stem], scenario, report, image_size, summary)

        for label_path in labels:
            if label_path.stem not in images_by_stem:
                _add_issue(report, scenario, label_path, "no matching image with this frame ID")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an Isaac Sim KITTI-style object-detection export.")
    parser.add_argument("--root", type=Path, default=Path("data/raw"), help="dataset root (default: data/raw)")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_names",
        help="validate only this scenario directory (repeat for multiple scenarios)",
    )
    parser.add_argument("--max-issues", type=int, default=50, help="maximum issue details to print (default: 50)")
    args = parser.parse_args(argv)
    if args.max_issues < 0:
        parser.error("--max-issues must be zero or greater")

    try:
        report = validate_kitti_object_detection_export(args.root, scenario_names=args.scenario_names)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Dataset: {report.root}")
    for scenario in report.scenarios:
        print(
            f"{scenario.name}: {scenario.image_count} images, {scenario.annotation_count} annotations, "
            f"{scenario.object_count} valid objects, {scenario.empty_annotation_count} empty annotations"
        )
    if report.issues:
        print(f"Issues: {len(report.issues)}")
        for issue in report.issues[: args.max_issues]:
            location = f"{issue.scenario}/{issue.path}" if issue.scenario else issue.path
            print(f"  {location}: {issue.message}")
        remaining = len(report.issues) - args.max_issues
        if remaining > 0:
            print(f"  ... {remaining} additional issue(s) not shown; increase --max-issues to see more")
        return 1
    print("Validation passed with no issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
