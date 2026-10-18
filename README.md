# Isaac Sim Pallet Jack Perception

A project for generating synthetic pallet-jack perception data in NVIDIA Isaac Sim, preparing datasets, fine-tuning an object-detection model externally, and evaluating and integrating the model.

## Project status

Initial project scaffold. Dataset conversion, training, evaluation, and simulator inference workflows are not implemented yet.

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
2. Document and validate image/annotation pairs.
3. Convert annotations to the selected detector format and create leakage-aware splits.
4. Fine-tune a pretrained detector in an external Python environment.
5. Evaluate synthetic and real images separately.
6. Export and validate inference in Isaac Sim.

See `docs/` for the workflow notes. Dataset dimensions, split strategy, model choice, and hardware requirements will be recorded once confirmed; none are assumed in this scaffold.
