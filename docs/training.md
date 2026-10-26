# Training

## First model training run

The completed first model run is **Experiment 1 (`EXP-001`)**. It fine-tuned `yolo26n.pt` on the 4,000-image synthetic training split using seed 42, 100 epochs, 640 input size, and batch size 16. The selected checkpoint is `runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt` (best validation mAP50–95 at epoch 89). The exact config and run artifacts are listed in the [Experiment 1 record](experiments/experiment-01.md). The LOCO images were not used for training.

This stage fine-tunes an Ultralytics YOLO detector on a prepared synthetic dataset. The default Experiment 1 configuration reads `data/processed/experiment-01/dataset.yaml`. Experiment 2 uses two separate configurations, both pointed at `data/processed/experiment-02/dataset.yaml`: [`train_yolo_e2_pretrained.yaml`](../configs/train_yolo_e2_pretrained.yaml) starts from `yolo26n.pt`, while [`train_yolo_e2_exp01-best.yaml`](../configs/train_yolo_e2_exp01-best.yaml) starts from Experiment 1's selected `best.pt`. Training uses the train split and checks validation metrics during training; it does not evaluate the held-out test set or real photographs.

Experiment 2's raw dataset preparation and both training runs are complete. Validation metrics were calculated each epoch and used to select each run's `best.pt`; the held-out test set was evaluated afterward in a separate command. The run configurations and results are documented in the [Experiment 2 record](experiments/experiment-02.md). The test run directories are `runs/detect/experiment-02/palletjack-detection-yolo26n-pretrained-seed42-experiment-02-test/` and `runs/detect/experiment-02/palletjack-detection-exp01-best-seed42-experiment-02-test/`.

These are the original preflight and training commands for the `yolo26n.pt`-initialized run:

```bash
python scripts/train_yolo.py --config configs/train_yolo_e2_pretrained.yaml --validate-only
python scripts/train_yolo.py --config configs/train_yolo_e2_pretrained.yaml
```

That run wrote under `runs/detect/experiment-02/palletjack-detection-yolo26n-pretrained-seed42/`. The second completed run used a separate config and run name:

```bash
python scripts/train_yolo.py --config configs/train_yolo_e2_exp01-best.yaml --validate-only
python scripts/train_yolo.py --config configs/train_yolo_e2_exp01-best.yaml
```

It wrote under `runs/detect/experiment-02/palletjack-detection-exp01-best-seed42/`. Both completed runs were evaluated on Experiment 2's held-out synthetic test split; results are reported in the [Experiment 2 record](experiments/experiment-02.md). Evaluating both checkpoints on Experiment 1's held-out synthetic test split remains a useful regression check. Report any LOCO comparison as a development diagnostic because those images motivated the asset variants.

## Training environment

Install PyTorch using the [official PyTorch selector](https://pytorch.org/get-started/locally/) for the machine's operating system and CPU/CUDA setup. The PyTorch install varies by hardware, so it is selected separately. Then install all project dependencies from the single requirements file:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If an existing virtual environment points to a Python installation that has been moved or removed, recreate that environment before installing training packages. Check the active interpreter with `which python` and `python --version` after activation.

The model checkpoint defaults to `yolo26n.pt`, a small pretrained detection model. Ultralytics downloads that checkpoint the first time an actual training run needs it. The official [YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26) and [training documentation](https://docs.ultralytics.com/modes/train) describe the supported models and options; these can change with future Ultralytics releases.

## Configure and preflight

Edit `configs/train_yolo.yaml` to set the pretrained checkpoint, dataset YAML, epochs, model input size, batch size, device, seed, worker count, and run name. Relative paths resolve from the repository root. `imgsz` controls model input resizing; it does not change the 960x544 source images.

Before training, validate the config and its dataset paths:

```bash
python scripts/train_yolo.py --config configs/train_yolo.yaml --validate-only
```

This command checks the dataset YAML, train/validation image paths, settings, and run destination without importing Ultralytics, downloading weights, or starting training. A missing dataset or an already-used run name is reported before the model is loaded. Choose a new `name` in the config for each run; existing run folders are never overwritten.

## Start training

After preflight succeeds, launch the run:

```bash
python scripts/train_yolo.py --config configs/train_yolo.yaml
```

For a short first run, temporarily set `epochs` to a small value, `workers` to `0`, and choose an unused run `name`. Adjust `batch` to fit the available hardware; `-1` enables Ultralytics automatic batch sizing. `device: null` leaves device selection to Ultralytics; use a value such as `cpu` or `0` to select a device explicitly.

For Experiment 1, Ultralytics wrote checkpoints, metrics, logs, and plots beneath the configured `project/name` directory (the Experiment 1 run folder `runs/detect/palletjack-detection-yolo26n-seed42/`). The run directory also receives `training_config.yaml`, a copy of the dataset YAML with its root resolved, and `runtime_metadata.json` with Python, Ultralytics, and PyTorch versions. These generated artifacts remain excluded from Git.

The Experiment 1 held-out synthetic test and LOCO real-image diagnostic were run separately after training. Their metrics and plots are labeled in [the Experiment 1 evaluation record](experiments/experiment-01.md). Prediction visualization and Isaac Sim inference remain separate workflows.

## Explore training results in Jupyter

The Experiment 1 notebook at `notebooks/explore_training_results.ipynb` lets you select a saved run and inspect its per-epoch losses and validation metrics, precision–recall curve, confusion matrices, and ground-truth-versus-prediction images. It does not use the held-out test split.

Install the notebook dependencies from the project's single requirements file if needed. Register the active virtual environment as a Jupyter kernel once, then start JupyterLab from the repository root:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name palletjack-detection --display-name "Python (Pallet Jack Detection)"
jupyter lab notebooks/explore_training_results.ipynb
```

In JupyterLab, select the `Python (Pallet Jack Detection)` kernel if it is not selected automatically. This keeps notebook code running in the environment where the project dependencies and trained-model support are installed.

Edit `RUN_NAME` in the first code cell to choose another directory under `runs/detect`. The optional real-image cell uses photos in `local/real-images` and calls the same prediction function as `scripts/predict_real_images.py`; set `RUN_REAL_IMAGE_PREDICTION = True` to opt in. The separate training cell is also off by default and only starts a run if you set `START_TRAINING = True` and execute that cell.
