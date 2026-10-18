# Isaac Sim Pallet Jack Perception

A project for generating synthetic pallet-jack perception data in NVIDIA Isaac Sim, preparing datasets, fine-tuning an object-detection model externally, and evaluating and integrating the model.

## Project status

Initial data-preparation workflow. A read-only validator, split-manifest creator, raw-data processor, and KITTI-to-YOLO converter are available; model training, evaluation, and simulator inference workflows are not implemented yet.

## Repository layout
- `simulation/` - Synthetic Data Generation setup resources for Issac Sim.
- `configs/` — dataset and experiment configuration.
- `data/raw/` — source datasets (kept out of Git).
- `data/processed/` — converted datasets and splits (kept out of Git).
- `docs/` — data, training, evaluation, and integration notes.
- `models/` — downloaded or trained model weights (kept out of Git).
- `runs/` — training outputs and experiment artifacts (kept out of Git).
- `scripts/` — command-line workflow entry points.
- `src/palletjack_detection/` — reusable project code.
- `tests/` — unit and regression tests.

## Planned workflow

1. Generate and export synthetic data from Isaac Sim.
2. Document and validate image/annotation pairs (read-only validator available).
3. Create reproducible train/validation/test split manifests.
4. Copy only RGB images and KITTI object-detection labels into `data/processed/`, then convert the copied labels to YOLO format.
5. Fine-tune a pretrained detector in an external Python environment.
6. Evaluate synthetic and real images separately.
7. Export and validate inference in Isaac Sim.

See `docs/` for the workflow notes. Dataset dimensions, split strategy, model choice, and hardware requirements will be recorded once confirmed; none are assumed in this scaffold.

## Validate the raw export

Install the runtime dependencies in the project's Python environment, then run from the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/validate_dataset.py
```

For importing the package directly from other locations, install the project in editable mode with `python -m pip install -e .`.

The validator reads from `data/raw/`, prints per-scenario counts and issues, and exits with status `0` when no issues are found or `1` when it finds issues. To validate another directory or show more issue details:

```powershell
python scripts/validate_dataset.py --root path/to/export --max-issues 200
```

It does not modify the dataset. See `docs/data.md` for the checks and current export notes.

## Create synthetic data splits

Create reproducible scenario-stratified split manifests:

```bash
python scripts/create_splits.py
```

The default is an 80/10/10 train/validation/test split with seed `42`. This step only assigns frames and writes CSV manifests under `data/processed/splits/`; it does not copy or convert data. The manifests record each raw image and KITTI label path, plus the intended processed paths. Use `--overwrite` to regenerate existing manifests. See `docs/data.md` for details.

## Process selected raw data

After creating the split manifests, copy the selected RGB images and KITTI labels into `data/processed/`, convert those copied labels, and create the YOLO dataset config:

```bash
python scripts/process_raw_data.py
```

The processor calls the converter function for each copied annotation. It preserves the KITTI labels under `data/processed/kitti_labels/`, writes YOLO labels under `data/processed/labels/`, and copies RGB images under `data/processed/images/`. Other raw modalities such as depth and segmentation are not copied. The processed image copy will use about the same storage as the selected raw RGB images.

The YOLO converter is also available as a focused one-annotation command:

```bash
python scripts/convert_kitti_to_yolo.py \
  --kitti-label path/to/frame.txt \
  --image path/to/frame.png \
  --output-label path/to/yolo/frame.txt
```

The reusable conversion functions live in `src/palletjack_detection/kitti_to_yolo.py`; they only convert KITTI bounding boxes into normalized YOLO label rows.
