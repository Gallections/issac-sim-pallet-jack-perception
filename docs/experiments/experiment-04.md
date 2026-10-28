# Experiment 4 — Pretrained versus scratch initialization

**Status:** Both training runs are complete. Both models have been evaluated on the combined synthetic test split and LOCO subset-1.

## Objective

Compare YOLO26n initialized from pretrained `yolo26n.pt` weights with the same YOLO26n architecture initialized randomly from `yolo26n.yaml`. Both runs use the same prepared images, split, and hyperparameters. No new simulation data was collected for this experiment.

## Data and label policy

The dataset YAML at [`configs/experiment-04-dataset.yaml`](../../configs/experiment-04-dataset.yaml) references the existing processed Experiment 1 and Experiment 3 splits directly; it does not copy images or annotations.

| Split | Experiment 1 | Experiment 3 | Combined images |
|---|---:|---:|---:|
| Train | 4,000 | 4,608 | 8,608 |
| Validation | 500 | 577 | 1,077 |
| Test | 500 | 577 | 1,077 |

The combined test split has 2,880 labeled objects. Experiment 1 contributes the original manual pallet-jack scenes. Experiment 3 adds manual and electric pallet trucks and target-free hard-negative scenes. Experiment 2 is excluded because its original labels mark forklift stackers as positive, while the current policy treats stackers as negatives.

The combined test images were held out from both Experiment 4 training runs, enabling a fair comparison between initialization choices. However, these exact images were evaluated in earlier experiments; they are not a fresh, project-wide untouched benchmark.

## Training comparison

Both configs use 100 epochs, `imgsz=640`, batch 16, seed 42, automatic device selection, and four workers. The models were trained on CUDA device 0 (NVIDIA GeForce RTX 5080 Laptop GPU).

| Run | Initialization | Best epoch | Best validation mAP50–95 | Run directory |
|---|---|---:|---:|---|
| Pretrained | `yolo26n.pt` | 83 | 0.73848 | `runs/detect/experiment-04/palletjack-detection-yolo26n-pretrained-seed42/` |
| Scratch | `yolo26n.yaml` (random initialization) | 91 | 0.69388 | `runs/detect/experiment-04/palletjack-detection-yolo26n-scratch-seed42/` |

The pretrained initialization scored 0.04460 higher in best validation mAP50–95 in this single-seed comparison. This is an observed result for these runs, not a general claim that pretraining always wins.

Training configurations: [`pretrained`](../../configs/train_yolo_e4_pretrained.yaml) and [`scratch`](../../configs/train_yolo_e4_scratch.yaml).

## Evaluation results

Both checkpoints were evaluated with 640-pixel input, batch 16, four workers, and CUDA device 0.

### Combined synthetic test split

| Initialization | Images | Boxes | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained | 1,077 | 2,880 | 0.9090 | 0.7807 | 0.8672 | 0.7455 |
| Scratch | 1,077 | 2,880 | 0.8731 | 0.7432 | 0.8349 | 0.6919 |

On this held-out synthetic split, the pretrained run is higher by 0.0363 mAP50–95. Both results are strong on synthetic data; neither establishes performance on real warehouse imagery.

### LOCO subset-1 real-image diagnostic

| Initialization | Images | Retained LOCO boxes | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained | 1,075 | 594 | 0.5615 | 0.2350 | 0.2240 | 0.0848 |
| Scratch | 1,075 | 594 | 0.4630 | 0.1919 | 0.1635 | 0.0546 |

The pretrained run is higher on all four reported metrics in this LOCO evaluation. Absolute scores remain low, especially recall and mAP50–95. A visual spot-check of the annotation gallery suggests boxes generally surround material-handling equipment, without an obvious systematic coordinate or box-placement issue in the reviewed images. However, LOCO uses a broad `pallet_truck` category that may include forklift stackers, which are excluded from the project target. These scores measure agreement with LOCO's category and are not verified accuracy for the project's manual/electric pallet-truck definition. The visual review was a sample, not a full annotation audit. LOCO subset-1 is also an official validation subset, not an independent project test set.

The main local comparison page includes saved prediction/label montages for both Experiment 4 variants: [`local/annotation_audit/index.html`](../../local/annotation_audit/index.html). The displayed montages cover only the first three evaluation batches (48 images), not a manual review of the full LOCO subset.

## Evaluation artifacts

| Run | Results |
|---|---|
| Pretrained synthetic test | `runs/detect/experiment-04/palletjack-detection-yolo26n-pretrained-seed42-test/` |
| Scratch synthetic test | `runs/detect/experiment-04/palletjack-detection-yolo26n-scratch-seed42-test/` |
| Pretrained LOCO diagnostic | `runs/detect/experiment-04/palletjack-detection-yolo26n-pretrained-seed42-loco-subset1/` |
| Scratch LOCO diagnostic | `runs/detect/experiment-04/palletjack-detection-yolo26n-scratch-seed42-loco-subset1/` |

Each evaluation directory contains a `test_metrics.json`, plots, confusion matrices, and saved label/prediction montage images. Results and checkpoints are local generated artifacts and are not committed when excluded by `.gitignore`.

## Interpretation

Experiment 4 shows that the pretrained initialization outperformed the scratch initialization on both the shared synthetic test split and the LOCO diagnostic. The LOCO result remains limited by the broad label scope, possible annotation omissions, and repeated views of similar warehouse scenes. A human-reviewed real-image holdout that distinguishes in-scope manual/electric pallet trucks from stackers is needed before making a reliable real-world performance claim.
