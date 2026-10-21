"""Run a trained detector on unlabeled images and save annotated copies."""

from __future__ import annotations

import argparse
from pathlib import Path


_SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def find_prediction_images(source: str | Path) -> list[Path]:
    """Resolve one image or recursively collect supported images from a folder."""
    source_path = Path(source).expanduser().resolve()
    if source_path.is_file():
        if source_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"source file is not a supported image: {source_path}")
        return [source_path]
    if not source_path.is_dir():
        raise ValueError(f"source image or folder does not exist: {source_path}")

    images = sorted(
        (
            path.resolve()
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix().casefold(),
    )
    if not images:
        raise ValueError(f"source folder contains no supported image files: {source_path}")
    return images


def _validate_output_location(source: Path, images: list[Path], output_dir: Path) -> None:
    if source.is_dir():
        try:
            output_dir.relative_to(source)
        except ValueError:
            pass
        else:
            raise ValueError("output folder must not be inside the source folder")

    destinations = [
        output_dir / (image.relative_to(source) if source.is_dir() else image.name)
        for image in images
    ]
    if set(destinations).intersection(images):
        raise ValueError("output paths would overwrite source images; choose another output folder")


def predict_real_images(
    *,
    source: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    confidence: float = 0.25,
    imgsz: int = 640,
    device: int | str | None = None,
) -> list[Path]:
    """Save annotated predictions for unlabeled real-world images.

    The output folder mirrors a source folder's relative layout. Source images
    are never modified. Images with no detections are saved with no boxes.
    """
    source_path = Path(source).expanduser().resolve()
    images = find_prediction_images(source_path)
    checkpoint = Path(model_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"model checkpoint does not exist: {checkpoint}")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz < 1:
        raise ValueError("imgsz must be a positive integer")
    if device is not None and not isinstance(device, (int, str)):
        raise ValueError("device must be null, a device index, or a device string")

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    _validate_output_location(source_path, images, resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed in this Python environment; install dependencies from requirements.txt"
        ) from exc

    model = YOLO(str(checkpoint))
    output_images: list[Path] = []
    for image in images:
        predictions = model.predict(
            source=str(image),
            conf=confidence,
            imgsz=imgsz,
            device=device,
            verbose=False,
            save=False,
        )
        if not predictions:
            raise RuntimeError(f"model returned no prediction result for image: {image}")

        relative_path = image.relative_to(source_path) if source_path.is_dir() else Path(image.name)
        output_image = resolved_output_dir / relative_path
        output_image.parent.mkdir(parents=True, exist_ok=True)
        predictions[0].save(filename=str(output_image))
        output_images.append(output_image)

    return output_images


def main(argv: list[str] | None = None) -> int:
    """Parse options, predict on images, and report the output location."""
    parser = argparse.ArgumentParser(
        description="Save visualized YOLO predictions for one image or a folder of unlabeled images."
    )
    parser.add_argument("--source", type=Path, required=True, help="image file or folder of images")
    parser.add_argument("--model", type=Path, required=True, help="trained Ultralytics .pt checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/predictions/real-images"),
        help="folder for annotated copies (default: runs/predictions/real-images)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence (0 to 1)")
    parser.add_argument("--imgsz", type=int, default=640, help="model input size (default: 640)")
    parser.add_argument("--device", default=None, help="inference device, such as 0 or cpu (default: automatic)")
    args = parser.parse_args(argv)

    try:
        output_images = predict_real_images(
            source=args.source,
            model_path=args.model,
            output_dir=args.output,
            confidence=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    output_dir = Path(args.output).expanduser().resolve()
    print(f"Saved {len(output_images)} annotated image(s) to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
