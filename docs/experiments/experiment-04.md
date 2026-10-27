# Experiment 4 — Pretrained versus scratch initialization

**Status:** dataset and both training configurations are prepared and pass preflight. No training or evaluation has been started.

## Objective

Compare YOLO26n initialized from pretrained `yolo26n.pt` weights with the same YOLO26n architecture initialized randomly from `yolo26n.yaml`. Both runs use the same prepared images, split, and hyperparameters. This isolates initialization as the main planned difference without collecting another simulation dataset.

## Data and label policy

The dataset YAML at [`configs/experiment-04-dataset.yaml`](../../configs/experiment-04-dataset.yaml) references the existing processed Experiment 1 and Experiment 3 splits directly. It does not copy images or annotations.

| Split | Experiment 1 | Experiment 3 | Combined images |
|---|---:|---:|---:|
| Train | 4,000 | 4,608 | 8,608 |
| Validation | 500 | 577 | 1,077 |
| Test | 500 | 577 | 1,077 |

The combined test split has 2,880 labeled objects. Experiment 1 contributes the original manual pallet-jack scenes. Experiment 3 adds manual and electric pallet trucks and its target-free hard negatives.

Experiment 2 is excluded because its original labels mark forklift stackers as positive, while the current policy treats stackers as negatives. LOCO is also excluded from training: its broad `pallet_truck` labels may include stackers.

The combined test split was held out from both Experiment 4 training runs, so it supports a fair initialization comparison. However, these exact test images were evaluated in earlier experiments and are not a fresh, project-wide untouched benchmark. Report results with that limitation.

## Training comparison

Both configs use 100 epochs, `imgsz=640`, batch 16, seed 42, automatic device selection, and four workers. The only intended experimental differences are the model initialization and unique run name.

| Run | Config | Initialization | Run directory |
|---|---|---|---|
| Pretrained | [`train_yolo_e4_pretrained.yaml`](../../configs/train_yolo_e4_pretrained.yaml) | `yolo26n.pt` | `runs/detect/experiment-04/palletjack-detection-yolo26n-pretrained-seed42/` |
| Scratch | [`train_yolo_e4_scratch.yaml`](../../configs/train_yolo_e4_scratch.yaml) | `yolo26n.yaml` (random initialization) | `runs/detect/experiment-04/palletjack-detection-yolo26n-scratch-seed42/` |

Run each preflight, then launch the training commands manually:

```powershell
python scripts/train_yolo.py --config configs/train_yolo_e4_pretrained.yaml --validate-only
python scripts/train_yolo.py --config configs/train_yolo_e4_scratch.yaml --validate-only
```

After reviewing both preflight outputs, launch one run at a time:

```powershell
python scripts/train_yolo.py --config configs/train_yolo_e4_pretrained.yaml
python scripts/train_yolo.py --config configs/train_yolo_e4_scratch.yaml
```

No training command has been run as part of preparing Experiment 4.

## Evaluation

After both training runs complete, evaluate each selected `best.pt` on the same Experiment 4 test split:

```powershell
python scripts/evaluate_test_set.py `
  --model runs/detect/experiment-04/palletjack-detection-yolo26n-pretrained-seed42/weights/best.pt `
  --dataset configs/experiment-04-dataset.yaml `
  --project runs/detect/experiment-04 `
  --name palletjack-detection-yolo26n-pretrained-seed42-test `
  --imgsz 640 --batch 16 --workers 4 --device 0

python scripts/evaluate_test_set.py `
  --model runs/detect/experiment-04/palletjack-detection-yolo26n-scratch-seed42/weights/best.pt `
  --dataset configs/experiment-04-dataset.yaml `
  --project runs/detect/experiment-04 `
  --name palletjack-detection-yolo26n-scratch-seed42-test `
  --imgsz 640 --batch 16 --workers 4 --device 0
```

Keep the two test results side by side and report precision, recall, mAP50, and mAP50–95. Do not interpret the Experiment 4 comparison as proof of real-world performance; the LOCO label mismatch remains unresolved.
