# Experiment 3 — Current target scope and hard negatives

**Status:** dataset preparation, training from `yolo26n.pt`, and held-out synthetic test evaluation are complete. Raw collection cleanup and KITTI validation are documented below.

## Objective and label policy

Experiment 3 updates the synthetic data to match the current detection target and adds target-free examples intended to teach the detector to reject confusing warehouse equipment.

- **Positive class (`palletjack`):** manual pallet jacks and the powered electric pallet truck represented by `electric_pallet_jack.usdz`.
- **Out of scope:** forklift stackers, including `pallet_jack_advanced.usdz`, and other warehouse equipment. These are distractors and do not receive pallet-jack boxes.
- **Hard negatives:** scenes with the combined warehouse and additional distractor pools, including the stacker and local props, but without a pallet-truck target. Their object-detection labels are empty.

The local distractor assets are `pallet_jack_advanced.usdz`, `forklift.usdz`, `ladder.usdz`, `shopping_cart.usdz`, and `warehouse_rack.usdz`. The electric pallet-truck asset is stored under `simulation/assets/pallet_jacks/`; the distractors are stored under `simulation/assets/`. The USDZ files are machine-local and ignored by Git, so they must be supplied separately to reproduce generation.

This policy differs from Experiment 2, where the powered electric pallet truck and forklift stacker were both labeled positive. Experiment 2 remains a historical result under its original labels; its annotations are not silently rewritten here.

## Scene generation

The generator uses the Isaac Sim Simple Warehouse scene and captures independently randomized 960×544 RGB frames with KITTI-style object-detection labels. It varies camera pose and scene-object pose per capture. The three target-positive conditions follow the same nominal frame counts as earlier collection:

| Raw group | Nominal frames | Scene contents |
|---|---:|---|
| `distractors_warehouse` | 2,000 | Pallet-truck targets with warehouse distractors |
| `distractors_additional` | 2,000 | Pallet-truck targets with additional distractor assets |
| `no_distractors` | 1,000 | Pallet-truck targets without an added distractor group |
| `hard_negatives` | 1,000 | Combined distractor pool, no pallet-truck targets, empty object labels |
| **Total generated** | **6,000** | **5,000 target-positive and 1,000 target-free frames** |

Run `simulation/scripts/generate_data.bat` on Windows or `generate_data.sh` in an Isaac Sim shell to generate all four groups under `data/raw/experiment-03/`. When only the target-free group needs to be regenerated, use `generate_hard_negatives.bat` or `generate_hard_negatives.sh`; it writes only to `data/raw/experiment-03/hard_negatives/`. The launchers and simulator settings are documented in the [simulation guide](../../simulation/README.md).

## Raw image cleanup and validation

After generation, the RGB folders were scanned for wholly transparent-black or all-black frames. Each invalid RGB image and the object-detection KITTI file with the same frame ID were removed together. The cleanup removed 238 pairs: 63 from `distractors_additional`, 52 from `distractors_warehouse`, 92 from `hard_negatives`, and 31 from `no_distractors`. Of these, 237 were fully transparent-black frames; one hard-negative frame was entirely black with an opaque alpha channel and was removed as an invalid black capture as well.

The current retained inventory is:

| Raw group | RGB images | KITTI files | Valid 2D objects | Empty labels |
|---|---:|---:|---:|---:|
| `distractors_additional` | 1,937 | 1,937 | 4,918 | 285 |
| `distractors_warehouse` | 1,948 | 1,948 | 5,829 | 230 |
| `hard_negatives` | 908 | 908 | 0 | 908 |
| `no_distractors` | 969 | 969 | 3,869 | 59 |
| **Total** | **5,762** | **5,762** | **14,616** | **1,482** |

The image and label stems match in each group. Validation was run with:

```powershell
python scripts/validate_dataset.py --root data/raw/experiment-03 `
  --scenario distractors_additional `
  --scenario distractors_warehouse `
  --scenario hard_negatives `
  --scenario no_distractors
```

The initial validation found one malformed box in `distractors_warehouse/Camera/object_detection/1727.txt`: line 3 had `x_min == x_max == 0`, so it had zero width. The frame was reviewed; its horizontal extent could not be recovered reliably from the image, so only that invalid annotation row was removed. The image and its other three valid boxes were retained. Validation was rerun and passed with no issues. This structural check is not a full visual audit of all images and labels.

## Split, processing, training, and held-out test results

The raw collection was split by generation condition using seed `42` and an 80/10/10 ratio. The processed dataset is at `data/processed/experiment-03/`; it contains normalized YOLO labels while retaining the source KITTI annotations.

| Split | Images | Ground-truth boxes |
|---|---:|---:|
| Train | 4,608 | 11,653 |
| Validation | 577 | 1,510 |
| Test | 577 | 1,453 |

A YOLO26n detector initialized from `yolo26n.pt` was trained for 100 epochs at `imgsz=640`, batch size 16, seed `42`, with four workers on an NVIDIA GeForce RTX 5080 Laptop GPU. The best validation checkpoint was epoch 83, selected by mAP50–95 (0.64275). Its validation precision was 0.85457, recall 0.69656, and mAP50 0.78304.

The selected checkpoint was evaluated once on the held-out test split:

| Test metric | Result |
|---|---:|
| Precision | 0.8538 |
| Recall | 0.7032 |
| mAP50 | 0.7875 |
| mAP50–95 | 0.6480 |

Evaluation used 577 images and 1,453 labeled objects. The run used `imgsz=640`, batch size 16, four workers, and CUDA device `0`. The test run artifacts and plots are in `runs/detect/experiment-03/palletjack-detection-yolo26n-pretrained-seed42-experiment-03-test/`; the trained checkpoint is `runs/detect/experiment-03/palletjack-detection-yolo26n-pretrained-seed42-experiment-03/weights/best.pt`. Generated run artifacts are gitignored.

Reproduce the test evaluation from the repository root with:

```powershell
python scripts/evaluate_test_set.py `
  --model runs/detect/experiment-03/palletjack-detection-yolo26n-pretrained-seed42-experiment-03/weights/best.pt `
  --dataset data/processed/experiment-03/dataset.yaml `
  --project runs/detect/experiment-03 `
  --name palletjack-detection-yolo26n-pretrained-seed42-experiment-03-test `
  --imgsz 640 --batch 16 --workers 4 --device 0
```

The Experiment 3 test result is below the historical Experiment 1 and Experiment 2 test scores, but the experiments use different generated data and target-label policies. This is not a controlled comparison. The test set is synthetic; real-world generalization remains an open question, and the broad LOCO labels do not reliably separate in-scope pallet trucks from forklift stackers.
