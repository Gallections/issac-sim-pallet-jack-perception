# Training

This stage fine-tunes a pretrained Ultralytics YOLO detector on the prepared synthetic dataset. It assumes the data-preparation workflow has already created `data/processed/dataset.yaml` and the `images/train`, `images/val`, and `labels/` directories. Training uses the train split and checks validation metrics during training; it does not evaluate the held-out test set or real photographs.

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

Ultralytics writes checkpoints, metrics, logs, and plots beneath the configured `project/name` directory (by default, `runs/detect/palletjack-yolo26n-seed42/`). The run directory also receives `training_config.yaml`, a copy of the dataset YAML with its root resolved, and `runtime_metadata.json` with Python, Ultralytics, and PyTorch versions. These generated artifacts remain excluded from Git.

Training and validation results can later be explored in a results notebook. That notebook is a separate feature and has not been added. Held-out test evaluation, real-photo evaluation, and Isaac Sim inference are also separate later stages.
