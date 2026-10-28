# Evaluation

Report synthetic and real-image results separately. Record model checkpoint, dataset provenance, thresholds, metrics where labels are available, and representative errors.

For interactive exploration of a saved training run and optional real-image predictions, see `notebooks/explore_training_results.ipynb` and the launch instructions in `docs/training.md`.

## Experiment 1 results

All quantitative results below are from this first model run. See the [Experiment 1 record](experiments/experiment-01.md) for the complete data, checkpoint, and figure index.

### LOCO subset-1 real-image evaluation

The trained `best.pt` checkpoint was evaluated on the LOCO images with ground-truth boxes. Since the existing Ultralytics validation path reads YOLO labels, first convert the filtered COCO labels. This conversion does not change the images or `annotations.json`; it writes YOLO label files for the evaluator.

```bash
python scripts/convert_coco_to_yolo.py
python scripts/evaluate_test_set.py \
  --model runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt \
  --dataset data/processed/experiment-01/real-pallet-jack-data/dataset.yaml \
  --name palletjack-detection-yolo26n-seed42-real-loco-subset1 \
  --device 0 --imgsz 640 --batch 16 --workers 4
```

The conversion report contains 1,075 images: 403 with 594 boxes from LOCO's broad `pallet_truck` category and 672 with no box in that category. The 65 additional JPGs without COCO image records are excluded. A visual spot-check suggests the boxes generally surround material-handling equipment, but LOCO does not separate manual and electric pallet trucks from stackers. The annotations have not been comprehensively subtype-audited against the project's target, so these metrics remain broad-category diagnostics. Evaluation ran on CUDA device 0 (NVIDIA GeForce RTX 5080 Laptop GPU), using the Isaac Sim-trained YOLO26n checkpoint, 640 input size, batch 16, and 4 data workers. Metrics are written to `runs/detect/palletjack-detection-yolo26n-seed42-real-loco-subset1/test_metrics.json`; Ultralytics also writes precision/recall curves, confusion matrices, and sample label/prediction montages to that run folder.

| Dataset | Images | LOCO `pallet_truck` boxes | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LOCO subset-1 real evaluation | 1,075 | 594 | 0.4113 | 0.1768 | 0.1430 | 0.0600 |

These results show weak agreement between this synthetic-trained baseline and the selected LOCO labels. The scores do not by themselves determine whether the gap comes from visual domain shift, annotation conventions, class scope, or model behavior. A spot-check found generally plausible box placement, but the project has not completed a comprehensive annotation or subtype audit. LOCO subset-1 is an official validation subset; treat this as a diagnostic, not a broad deployment benchmark or untouched final test. Keep it separate from synthetic held-out metrics.

Selected small snapshots are checked into `docs/assets/real-loco/`; complete run artifacts remain in the ignored `runs/` directory.

#### Inspect predictions on real images

The LOCO subset-1 images and COCO annotations are staged locally under `data/raw/experiment-01/real-pallet-jack-data/`. The data source, license, and inventory are documented in [`docs/data.md`](data.md). The JSON covers 1,075 images; 65 additional JPGs lack COCO image records and should not be scored as negative examples until checked. The filtered COCO labels are retained as source ground truth, and a separate COCO-to-YOLO adapter makes them readable by the existing Ultralytics evaluator. The adapter preserves LOCO's broad `pallet_truck` scope; it does not provide verified labels for the project's manual-plus-electric target because stackers may be included.

Use the trained checkpoint to produce annotated copies of one real photograph or every supported image in a folder. Images do not need labels; this is for visual inspection and does not calculate accuracy metrics.

```bash
python scripts/predict_real_images.py \
  --source path/to/real_photos \
  --model runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt \
  --output runs/predictions/real-photos
```

The command accepts a single image or a folder (searched recursively). It writes annotated copies under the output folder, preserving the folder's subdirectories, and leaves source images unchanged. Images with no detections are saved without boxes. The default confidence threshold is `0.25`, the input size is `640`, and Ultralytics selects the device automatically. Override these with `--conf`, `--imgsz`, or `--device` (for example, `--device 0` or `--device cpu`).

Use images with and without pallet jacks, and review missed detections and false detections. A visual prediction on unlabeled photos is not a measured evaluation; quantitative real-image metrics require consistently annotated ground truth and documented image provenance. Keep those results separate from held-out synthetic test metrics.

### Held-out synthetic test set

The training process uses validation data to monitor training and select `best.pt`. The separate test-only command evaluates the selected checkpoint using the `test` path in `data/processed/experiment-01/dataset.yaml` by default. Use `--dataset data/processed/experiment-02/dataset.yaml` for Experiment 2:

```bash
python scripts/evaluate_test_set.py --device 0 --name palletjack-detection-yolo26n-seed42-test-reproduction
```

The command records the checkpoint and settings in `test_metrics.json`, creates PR/confusion plots and sample predictions, and writes output under `runs/detect/<name>/`. Choose a new name for each run; existing output folders are not overwritten. Leave out `--device 0` for automatic selection or pass `cpu` to force CPU evaluation.

#### Synthetic baseline metrics

The selected YOLO26n checkpoint (`best.pt`, epoch 89 by validation mAP50-95) was evaluated at input size 640, batch size 16, on CUDA device 0 (NVIDIA GeForce RTX 5080 Laptop GPU).

| Split | Images | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 500 | 0.9313 | 0.8243 | 0.9098 | 0.7918 |
| Held-out test | 500 | 0.9248 | 0.8087 | 0.9059 | 0.7861 |

The test split has 200 `distractors_additional`, 200 `distractors_warehouse`, and 100 `no_distractors` images, with 1,427 labeled pallet-jack instances total. One close-up example (`1006.png`) can look like a mislabeled warehouse prop because the pallet jack is cropped and occupies most of the frame; visual review confirms it is a pallet jack. The reported scores measure performance on this held-out synthetic split against its exported labels. They do not measure real-photo performance or establish robustness to real camera, lighting, or occlusion conditions.

Even with verified labels, the result would be limited to held-out synthetic frames from the same data-generation setup. It would not measure transfer to real photographs, establish robustness to real lighting/cameras/occlusion, or prove deployment readiness.

Once the label audit is complete and the test metrics are regenerated, avoid repeatedly consulting this test set to tune confidence thresholds or choose among many checkpoints. Use a new untouched test set or independently labeled real-image benchmark for subsequent model changes. Keep real-image results separate from synthetic results and report dataset source, label status, checkpoint, settings, and metrics together.

Experiment 1 held-out test visualization snapshots are in `docs/assets/test/`; full evaluation outputs remain under ignored `runs/`.

## Experiment 2 held-out synthetic test

Both Experiment 2 runs completed 100 epochs, with validation metrics calculated each epoch and `best.pt` selected by validation mAP50-95. The selected checkpoint from each run was evaluated on the same Experiment 2 test split: 482 images and 2,307 labeled pallet-jack instances. Evaluation used 640 input size, batch 16, four workers, and CUDA device 0 (NVIDIA GeForce RTX 5080 Laptop GPU).

| Initialization | Best validation epoch | Validation precision | Validation recall | Validation mAP50 | Validation mAP50-95 | Test precision | Test recall | Test mAP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolo26n.pt` | 84 | 0.9116 | 0.7809 | 0.8750 | 0.7242 | 0.9132 | 0.7847 | 0.8845 | 0.7380 |
| Experiment 1 `best.pt` | 82 | 0.9033 | 0.7849 | 0.8731 | 0.7264 | 0.9108 | 0.7850 | 0.8795 | 0.7369 |

The test mAP50-95 values differ by about 0.0011. This is a small difference from one test split and does not establish that either initialization is better. The Experiment 1 checkpoint initialization did not improve the Experiment 2 test score in this run. These metrics describe held-out synthetic frames from this generator and its labels; they do not measure transfer to real photographs. The LOCO images motivated the asset changes and remain a development diagnostic rather than an untouched benchmark for this experiment.

The evaluation outputs, including `test_metrics.json`, confusion matrices, precision-recall plots, and prediction montages, are preserved locally under:

- `runs/detect/experiment-02/palletjack-detection-yolo26n-pretrained-seed42-experiment-02-test/`
- `runs/detect/experiment-02/palletjack-detection-exp01-best-seed42-experiment-02-test/`

Small report snapshots are checked into `docs/assets/experiment-02/`. To reproduce either result, run `scripts/evaluate_test_set.py` with the corresponding `best.pt` checkpoint, `--dataset data/processed/experiment-02/dataset.yaml`, `--device 0`, `--imgsz 640`, `--batch 16`, and a new `--name`. The full metric summary stores absolute local paths, so use the documented repo-relative checkpoint and dataset locations when reproducing on another machine.

## LOCO subset-1 real-image diagnostics

All six trained checkpoints from Experiments 1–4 were evaluated on the same processed LOCO subset-1 COCO adapter. This is the official LOCO validation subset. It contains 1,075 recorded images, 594 boxes from the broad `pallet_truck` category, 403 images with at least one such box, and 672 images with no such annotation; 65 additional JPGs without COCO records were excluded. LOCO does not distinguish manual/electric pallet trucks from stackers, so these metrics are not verified current-target detection scores. The evaluations used 640 input size, batch 16, four workers, and CUDA device 0.

| Model | Images | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Experiment 1 baseline (previous evaluation) | 1,075 | 594 | 0.4113 | 0.1768 | 0.1430 | 0.0600 |
| Experiment 2, `yolo26n.pt` initialization | 1,075 | 594 | 0.3797 | 0.2828 | 0.2147 | 0.0830 |
| Experiment 2, Experiment 1 `best.pt` initialization | 1,075 | 594 | 0.3947 | 0.2525 | 0.2239 | 0.0850 |
| Experiment 3, `yolo26n.pt` initialization | 1,075 | 594 | 0.4218 | 0.2391 | 0.2015 | 0.0642 |
| Experiment 4, `yolo26n.pt` initialization | 1,075 | 594 | 0.5615 | 0.2350 | 0.2240 | 0.0848 |
| Experiment 4, random initialization | 1,075 | 594 | 0.4630 | 0.1919 | 0.1635 | 0.0546 |

The Experiment 4 pretrained model has the highest precision in this diagnostic (0.5615), but its recall remains low (0.2350). Its mAP50–95 of 0.0848 is close to the highest listed result (0.0850 for Experiment 2 initialized from Experiment 1). The Experiment 4 pretrained model scores higher than its scratch counterpart on all four metrics. Since LOCO's broad target boxes may include stackers, these results do not establish current-target accuracy or a deployment result. See the [Experiment 4 record](experiments/experiment-04.md) for the synthetic-test comparison and output paths.

The comparison is not an independent test of Experiment 2: this LOCO subset motivated the asset change, so model and data choices were informed by it. In addition, the COCO `pallet_truck` category has not been audited to separate in-scope manual/electric pallet trucks from stackers. This is a development diagnostic, not a general sim-to-real benchmark. See the [Experiment 2 record](experiments/experiment-02.md) for plotted results and details.

### Experiment 2 per-image error audit at a fixed threshold

The project generated an all-image comparison gallery using confidence ≥ 0.25 and one-to-one matches at IoU ≥ 0.50. This is a diagnostic operating point; it differs from Ultralytics' aggregate precision and recall, which use a selected confidence threshold.

| Initialization | Matched boxes | Missed boxes | Unmatched predictions | Positive images with misses | Positive images fully missed | Negative images with any prediction |
|---|---:|---:|---:|---:|---:|---:|
| `yolo26n.pt` | 224 | 370 | 849 | 286 / 403 | 209 / 403 | 269 / 672 |
| Experiment 1 `best.pt` | 174 | 420 | 424 | 318 / 403 | 255 / 403 | 185 / 672 |

The fresh-pretrained model catches more boxes at this confidence threshold, but makes substantially more unmatched predictions; roughly 40% of the images without a LOCO `pallet_truck` box trigger at least one detection. The Experiment 1-initialized model has fewer such detections but misses more labeled objects. Because the LOCO category may include stackers outside the current target, these counts are not verified current-target false positives or misses. In the local working copy, open `local/annotation_audit/experiment-02-loco-predictions/index.html` to inspect paired ground truth and predictions, or use `local/annotation_audit/experiment-02-loco-predictions/per_image_audit.csv` to sort cases by misses and false detections. These generated files are ignored by Git. The gallery supports human review; it does not prove the source annotations are correct.
