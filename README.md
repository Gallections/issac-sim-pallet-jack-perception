# Isaac Sim Pallet Jack Perception

A project for generating synthetic pallet-jack perception data in NVIDIA Isaac Sim, preparing datasets, fine-tuning an object-detection model externally, and evaluating and integrating the model.

## Project status

Initial project setup. A read-only validator for the Isaac Sim KITTI-style export is available; conversion, training, evaluation, and simulator inference workflows are not implemented yet.

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
3. Convert annotations to the selected detector format and create leakage-aware splits.
4. Fine-tune a pretrained detector in an external Python environment.
5. Evaluate synthetic and real images separately.
6. Export and validate inference in Isaac Sim.

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

After validation, create reproducible scenario-stratified CSV manifests:

```bash
python scripts/create_splits.py
```

The default is an 80/10/10 train/validation/test split with seed `42`. It writes `train.csv`, `validation.csv`, and `test.csv` under `data/processed/splits/`. The manifests reference the original images and labels; the script does not copy or modify raw data. See `docs/data.md` for details and options.
