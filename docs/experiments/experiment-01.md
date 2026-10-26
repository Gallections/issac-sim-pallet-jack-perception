# Experiment 1 — Isaac Sim synthetic pallet-jack detector

## Purpose and scope

**Experiment ID:** `EXP-001`

Experiment 1 establishes the first end-to-end baseline: generate automatically labeled pallet-jack scenes in Isaac Sim, train Ultralytics YOLO26n outside the simulator using synthetic data only, measure performance on a held-out synthetic test split, and run a separate diagnostic evaluation on LOCO real images. It does not fine-tune on LOCO.

The existing raw-data and run-folder names are retained for traceability. This document is the index tying them to Experiment 1. Raw data, processed data, checkpoints, and full Ultralytics runs are local and gitignored; the paths below describe the expected project layout.

## Assets

### Isaac Sim source data

Raw exports are under `data/raw/experiment-01/`:

| Raw export folder | Frames | Condition |
|---|---:|---|
| `data/raw/experiment-01/distractors_warehouse/` | 2,000 | Warehouse props |
| `data/raw/experiment-01/distractors_additional/` | 2,000 | Additional asset-group props |
| `data/raw/experiment-01/no_distractors/` | 1,000 | No added distractor group |
| **Total** | **5,000** | Three conditions, independently randomized frames |

Each source condition includes RGB, KITTI-style `object_detection`, depth, instance segmentation, semantic, and semantic-segmentation exports. The Experiment 1 detector uses RGB plus 2D object-detection boxes. Source exports and metadata are preserved; preparation does not edit `data/raw/experiment-01/`. Generation implementation and settings are in [`simulation/README.md`](../../simulation/README.md) and [`simulation/scripts/palletjack_sdg/standalone_palletjack_sdg.py`](../../simulation/scripts/palletjack_sdg/standalone_palletjack_sdg.py).

### Processed synthetic dataset

The generated split manifests are under `data/processed/experiment-01/splits/`. The processed model dataset is under `data/processed/experiment-01/`, with the original KITTI labels retained alongside converted YOLO labels:

| Split | Images | Use |
|---|---:|---|
| Train | 4,000 | Update model weights |
| Validation | 500 | Monitor training and select checkpoint |
| Test | 500 | Held-out synthetic evaluation |

The split uses seed `42` and 80/10/10 allocation within each source condition. Processing details are in [`docs/data.md`](../data.md).

### Training run and checkpoint

- **Training configuration:** [`configs/train_yolo.yaml`](../../configs/train_yolo.yaml)
- **Base checkpoint:** `yolo26n.pt` (pretrained; not committed)
- **Experiment 1 training run:** `runs/detect/palletjack-detection-yolo26n-seed42/`
- **Selected checkpoint:** `runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt`
- **Checkpoint selection:** validation mAP50–95; best at epoch 89
- **Settings:** 100 epochs, 640 input size, batch size 16, seed 42, 4 workers, automatic device selection
- **Recorded runtime:** Ultralytics 8.4.182, PyTorch 2.14.1+cu132, NVIDIA GeForce RTX 5080 Laptop GPU

The checkpoint is local and gitignored. Reproduce the training launch with `python scripts/train_yolo.py --config configs/train_yolo.yaml`.

## Results

### Validation and held-out synthetic test

| Split | Images | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Validation, best checkpoint at epoch 89 | 500 | 0.9313 | 0.8243 | 0.9098 | 0.7918 |
| Held-out synthetic test | 500 | 0.9248 | 0.8087 | 0.9059 | 0.7861 |

The validation and test figures are snapshots in `docs/assets/baseline/` and `docs/assets/test/`. Full result files remain in the corresponding local run folders. See [`docs/evaluation.md`](../evaluation.md) for plot interpretation.

### External real-image diagnostic

Experiment 1's synthetic-trained checkpoint was evaluated, without real-data fine-tuning, on the 1,075 COCO-listed images from LOCO subset-1. The filtered set contains 594 pallet-jack boxes on 403 positive images and 672 negative images. The 65 JPGs without COCO image records were excluded.

| External evaluation | Images | Boxes | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| LOCO subset-1 | 1,075 | 594 | 0.4113 | 0.1768 | 0.1430 | 0.0600 |

Run output: `runs/detect/palletjack-detection-yolo26n-seed42-real-loco-subset1/`. Checked-in plot snapshots are in `docs/assets/real-loco/`. This result indicates weak transfer on this selected real-image set. It is a diagnostic, not a broad deployment benchmark; LOCO annotations have not received a comprehensive visual audit. Keep it separate from the synthetic test result.

## Figure and artifact index

| Result | Tracked visualization | Full local run artifact |
|---|---|---|
| Training losses and validation metrics | `docs/assets/baseline/training-results.png` | `runs/detect/palletjack-detection-yolo26n-seed42/results.png` and `results.csv` |
| Validation precision–recall and confusion matrices | `docs/assets/baseline/` | `runs/detect/palletjack-detection-yolo26n-seed42/` |
| Held-out synthetic test plots and examples | `docs/assets/test/` | Canonical report with metrics: `runs/detect/palletjack-detection-yolo26n-seed42-test-report/`; earlier plot-only copy: `runs/detect/palletjack-detection-yolo26n-seed42-test/` |
| LOCO real-image plots and examples | `docs/assets/real-loco/` | `runs/detect/palletjack-detection-yolo26n-seed42-real-loco-subset1/` |
| Local ground-truth annotation gallery | `local/annotation_audit/index.html` | Local-only; includes all 4,000 synthetic training and 1,075 LOCO images |

The local gallery draws stored ground-truth boxes and is intended for label review. Ultralytics validation montages draw ground truth and predictions separately; do not confuse those prediction overlays with the annotation-audit gallery.

## Relationship to Experiment 2

Experiment 1 shows strong performance on its held-out synthetic frames and much weaker performance on the LOCO subset. LOCO includes pallet-truck appearances and scales that may differ from the Isaac Sim asset. The aggregate real score alone cannot distinguish visual domain shift, object-design mismatch, annotation issues, or other causes.

Experiment 2 now adds two local USDZ variants to the Isaac Sim asset mix while retaining this experiment's data, checkpoint, plots, and metrics as the baseline. Its [separate record](experiment-02.md) documents the asset and data changes, preparation, and planned comparison. Since LOCO informed this design change, keep its Experiment 2 evaluation diagnostic; use an independent real-image holdout for any final transfer claim.
