"""Convert KITTI 2D bounding-box annotations into YOLO detection labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from palletjack_detection.validate_dataset import CLASS_NAMES


def _format_yolo_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_kitti_annotation_as_yolo(
    kitti_annotation_path: str | Path,
    *,
    image_width: int,
    image_height: int,
    inclusive_max_coordinates: bool = True,
) -> str:
    """Return normalized YOLO rows converted from one KITTI annotation file."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    yolo_rows: list[str] = []
    for line in Path(kitti_annotation_path).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        class_id = sorted(CLASS_NAMES).index(fields[0])
        x_min, y_min, x_max, y_max = map(float, fields[4:8])
        max_edge_adjustment = 1.0 if inclusive_max_coordinates else 0.0
        box_width = x_max - x_min + max_edge_adjustment
        box_height = y_max - y_min + max_edge_adjustment
        center_x = (x_min + x_max + max_edge_adjustment) / 2.0
        center_y = (y_min + y_max + max_edge_adjustment) / 2.0
        normalized_box = (
            center_x / image_width,
            center_y / image_height,
            box_width / image_width,
            box_height / image_height,
        )
        yolo_rows.append(
            f"{class_id} " + " ".join(_format_yolo_number(value) for value in normalized_box)
        )
    return "\n".join(yolo_rows) + ("\n" if yolo_rows else "")


def convert_kitti_annotation_file_to_yolo(
    kitti_annotation_path: str | Path,
    image_path: str | Path,
    yolo_annotation_path: str | Path,
    *,
    inclusive_max_coordinates: bool = True,
) -> Path:
    """Convert one KITTI annotation to YOLO using the paired image dimensions."""
    with Image.open(image_path) as image:
        yolo_text = format_kitti_annotation_as_yolo(
            kitti_annotation_path,
            image_width=image.width,
            image_height=image.height,
            inclusive_max_coordinates=inclusive_max_coordinates,
        )
    destination = Path(yolo_annotation_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yolo_text, encoding="utf-8", newline="\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert one KITTI annotation file to YOLO format.")
    parser.add_argument("--kitti-label", type=Path, required=True, help="source KITTI annotation file")
    parser.add_argument("--image", type=Path, required=True, help="paired image used to read image dimensions")
    parser.add_argument("--output-label", type=Path, required=True, help="destination YOLO annotation file")
    parser.add_argument(
        "--exclusive-max-coordinates",
        action="store_true",
        help="treat KITTI x_max/y_max as exclusive edges; default treats them as inclusive pixels",
    )
    args = parser.parse_args(argv)
    try:
        result = convert_kitti_annotation_file_to_yolo(
            args.kitti_label,
            args.image,
            args.output_label,
            inclusive_max_coordinates=not args.exclusive_max_coordinates,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Converted {args.kitti_label} -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
