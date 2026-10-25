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

The conversion report contains 1,075 images: 403 with 594 pallet-jack instances and 672 with no pallet-jack box. The 65 additional JPGs without COCO image records are excluded. Evaluation ran on CUDA device 0 (NVIDIA GeForce RTX 5080 Laptop GPU), using the Isaac Sim-trained YOLO26n checkpoint, 640 input size, batch 16, and 4 data workers. Metrics are written to `runs/detect/palletjack-detection-yolo26n-seed42-real-loco-subset1/test_metrics.json`; Ultralytics also writes precision/recall curves, confusion matrices, and sample label/prediction montages to that run folder.

| Dataset | Images | Pallet-jack boxes | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LOCO subset-1 real evaluation | 1,075 | 594 | 0.4113 | 0.1768 | 0.1430 | 0.0600 |

These results show weak transfer from this synthetic-trained baseline to the selected LOCO images. The scores are evidence about this dataset and checkpoint; they do not by themselves determine whether the gap comes from visual domain shift, annotation conventions, label quality, or model operating behavior. LOCO subset-1 is an official validation subset, and the project has not completed a comprehensive annotation audit. Treat this as an external evaluation and diagnostic, not a broad deployment benchmark or an untouched final test for future tuning. Keep the result separate from synthetic held-out metrics.

Selected small snapshots are checked into `docs/assets/real-loco/`; complete run artifacts remain in the ignored `runs/` directory.

#### Inspect predictions on real images

The LOCO subset-1 images and COCO annotations are staged locally under `data/raw/experiment-01/real-pallet-jack-data/`. The data source, license, and inventory are documented in [`docs/data.md`](data.md). The JSON covers 1,075 images; 65 additional JPGs lack COCO image records and should not be scored as negative examples until checked. The filtered COCO labels are retained as source ground truth, and a separate COCO-to-YOLO adapter makes them readable by the existing Ultralytics evaluator. The set has been evaluated separately from synthetic results.

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
