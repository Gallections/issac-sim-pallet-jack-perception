"""Build a static ground-truth gallery for a prepared experiment dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import html
import math
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class AnnotationAuditSummary:
    image_count: int
    box_count: int
    galleries: tuple[tuple[str, str, int, int, int], ...]


_CSS = """
:root{color-scheme:light;font:15px/1.45 Segoe UI,Arial,sans-serif;color:#18212b;background:#f2f5f7}
body{margin:0 auto;max-width:1800px;padding:22px}a{color:#075b9a}
.note{background:#fff8d8;border-left:4px solid #d6a500;padding:12px 16px;margin:18px 0}
.nav{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0 24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,650px),1fr));gap:18px}
.card{background:white;padding:12px;border-radius:8px;box-shadow:0 1px 5px #0002}
.frame{position:relative;width:100%}.frame img{display:block;width:100%;height:auto}
.box{position:absolute;border:3px solid #00e05a;box-sizing:border-box;pointer-events:none}
.box span{position:absolute;left:-3px;top:-22px;background:#00c84d;color:#071b0e;font-weight:700;font-size:13px;padding:1px 5px;white-space:nowrap}
.filename{font-weight:600;overflow-wrap:anywhere;margin:7px 0 2px}.empty{color:#7b3f00;font-style:italic;margin-top:8px}
table{border-collapse:collapse;background:white}th,td{padding:10px 14px;border:1px solid #ccd4da;text-align:left}
@media(max-width:700px){body{padding:12px}.grid{grid-template-columns:1fr}}
"""


def _resolve_manifest_path(processed_root: Path, relative_path: str) -> Path:
    candidate = (processed_root / relative_path).resolve()
    try:
        candidate.relative_to(processed_root.resolve())
    except ValueError as exc:
        raise ValueError("Manifest paths must stay within the processed dataset") from exc
    return candidate


def _read_yolo_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"Expected one-class YOLO labels in {label_path}:{line_number}")
        try:
            center_x, center_y, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO box in {label_path}:{line_number}") from exc
        values = (center_x, center_y, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite YOLO box in {label_path}:{line_number}")
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"Out-of-range YOLO box in {label_path}:{line_number}")
        boxes.append(values)
    return boxes


def _box_overlay(boxes: list[tuple[float, float, float, float]]) -> str:
    overlays = []
    for center_x, center_y, width, height in boxes:
        left = (center_x - width / 2) * 100
        top = (center_y - height / 2) * 100
        overlays.append(
            '<div class="box" style="left:{:.6f}%;top:{:.6f}%;width:{:.6f}%;height:{:.6f}%">'
            "<span>palletjack</span></div>".format(left, top, width * 100, height * 100)
        )
    return "".join(overlays)


def _gallery_page(
    *,
    gallery_title: str,
    cards: list[tuple[str, str, list[tuple[float, float, float, float]]]],
    page_number: int,
    page_count: int,
    page_dir: Path,
    index_path: Path,
) -> str:
    rendered_cards = []
    for filename, image_path, boxes in cards:
        image_href = Path(os.path.relpath(image_path, page_dir)).as_posix()
        rendered_cards.append(
            '<div class="card"><div class="frame"><img loading="lazy" src="{}" alt="{} with ground-truth boxes">{}'
            '</div><div class="filename">{}</div>{}</div>'.format(
                html.escape(image_href, quote=True),
                html.escape(filename, quote=True),
                _box_overlay(boxes),
                html.escape(filename),
                '<div class="empty">Negative image: no pallet-jack annotation.</div>' if not boxes else "",
            )
        )
    navigation = ['<a href="{}">Experiment 2 index</a>'.format(
        html.escape(Path(os.path.relpath(index_path, page_dir)).as_posix(), quote=True)
    )]
    if page_number > 1:
        navigation.append(f'<a href="page-{page_number - 1:03d}.html">← Previous page</a>')
    navigation.append(f"<span>Page {page_number} of {page_count}</span>")
    if page_number < page_count:
        navigation.append(f'<a href="page-{page_number + 1:03d}.html">Next page →</a>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(gallery_title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(gallery_title)}</h1>"
        "<p>Experiment 2 processed images with stored YOLO ground-truth boxes drawn in green. "
        "No detector inference is shown.</p>"
        '<div class="note"><b>Green boxes are dataset annotations, not predictions.</b> '
        "Inspect box placement and negative images for missing labels.</div>"
        f'<div class="nav">{" ".join(navigation)}</div>'
        f'<div class="grid">{"".join(rendered_cards)}</div>'
        f'<div class="nav">{" ".join(navigation)}</div></body></html>'
    )


def build_experiment_annotation_audit(
    processed_root: Path,
    output_dir: Path,
    *,
    page_size: int = 50,
) -> AnnotationAuditSummary:
    """Generate an Experiment 2-only static HTML audit from split manifests."""
    processed_root = Path(processed_root).resolve()
    output_dir = Path(output_dir).resolve()
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    grouped: dict[tuple[str, str], list[tuple[str, Path, list[tuple[float, float, float, float]]]]] = {}
    for split in ("train", "validation", "test"):
        manifest_path = processed_root / "splits" / f"{split}.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Required split manifest does not exist: {manifest_path}")
        with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
            reader = csv.DictReader(manifest_file)
            required_columns = {"split", "scenario", "processed_image_path", "yolo_annotation_path"}
            if not required_columns.issubset(reader.fieldnames or ()):
                raise ValueError(f"Split manifest is missing required columns: {manifest_path}")
            for row in reader:
                row_split = row["split"]
                scenario = row["scenario"]
                if row_split != split or not re.fullmatch(r"[A-Za-z0-9_-]+", scenario):
                    raise ValueError(f"Unexpected split or scenario in manifest: {manifest_path}")
                image_path = _resolve_manifest_path(processed_root, row["processed_image_path"])
                label_path = _resolve_manifest_path(processed_root, row["yolo_annotation_path"])
                if not image_path.is_file() or not label_path.is_file():
                    raise FileNotFoundError(f"Missing processed image or YOLO label: {image_path} / {label_path}")
                grouped.setdefault((split, scenario), []).append(
                    (image_path.name, image_path, _read_yolo_boxes(label_path))
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    total_images = 0
    total_boxes = 0
    for (split, scenario), records in sorted(grouped.items()):
        group_name = f"{split}-{scenario}"
        page_dir = output_dir / group_name
        page_dir.mkdir(parents=True, exist_ok=True)
        page_count = max(1, math.ceil(len(records) / page_size))
        box_count = sum(len(record[2]) for record in records)
        for page_index in range(page_count):
            page_records = records[page_index * page_size : (page_index + 1) * page_size]
            page_text = _gallery_page(
                gallery_title=f"Experiment 2 — {split.title()} — {scenario}",
                cards=[
                    (filename, image_path, boxes)
                    for filename, image_path, boxes in page_records
                ],
                page_number=page_index + 1,
                page_count=page_count,
                page_dir=page_dir,
                index_path=output_dir / "index.html",
            )
            (page_dir / f"page-{page_index + 1:03d}.html").write_text(page_text, encoding="utf-8")
        summary_rows.append((split, scenario, len(records), box_count, page_count))
        total_images += len(records)
        total_boxes += box_count

    rows = "".join(
        '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><a href="{}/page-001.html">Open</a></td></tr>'.format(
            split.title(), html.escape(scenario), images, boxes, pages, f"{split}-{scenario}"
        )
        for split, scenario, images, boxes, pages in summary_rows
    )
    index = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Experiment 2 annotation audit</title><style>'
        f"{_CSS}</style></head><body><h1>Experiment 2 (EXP-002) — Annotation audit</h1>"
        '<div class="nav"><a href="../index.html">All experiments</a></div>'
        f"<p>{total_images:,} processed images and {total_boxes:,} stored YOLO boxes across train, validation, and test.</p>"
        '<div class="note">This gallery is Experiment 2 only. It shows stored ground-truth labels in green; '
        "it does not run the detector or show predictions. Images are linked from the processed dataset and are not copied.</div>"
        "<table><thead><tr><th>Split</th><th>Scenario</th><th>Images</th><th>Boxes</th><th>Pages</th><th>Gallery</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    return AnnotationAuditSummary(total_images, total_boxes, tuple(summary_rows))
