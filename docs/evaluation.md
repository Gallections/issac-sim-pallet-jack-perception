# Evaluation

Report synthetic and real-image results separately. Record model checkpoint, dataset provenance, thresholds, metrics where labels are available, and representative errors.

For interactive exploration of a saved training run and optional real-image predictions, see `notebooks/01_explore_training_results.ipynb` and the launch instructions in `docs/training.md`.

## Inspect predictions on real images

The LOCO subset-1 images and COCO annotations are staged locally under `data/raw/real-pallet-jack-data/`. The data source, license, and inventory are documented in [`docs/data.md`](data.md). The JSON covers 1,075 images; 65 additional JPGs lack COCO image records and should not be scored as negative examples until checked. The set has not yet been converted into the project's one-class format or used to calculate real-image metrics. Keep this real dataset separate from synthetic results.

Use the trained checkpoint to produce annotated copies of one real photograph or every supported image in a folder. Images do not need labels; this is for visual inspection and does not calculate accuracy metrics.

```bash
python scripts/predict_real_images.py \
  --source path/to/real_photos \
  --model runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt \
  --output runs/predictions/real-photos
```

The command accepts a single image or a folder (searched recursively). It writes annotated copies under the output folder, preserving the folder's subdirectories, and leaves source images unchanged. Images with no detections are saved without boxes. The default confidence threshold is `0.25`, the input size is `640`, and Ultralytics selects the device automatically. Override these with `--conf`, `--imgsz`, or `--device` (for example, `--device 0` or `--device cpu`).

Use images with and without pallet jacks, and review missed detections and false detections. A visual prediction on unlabeled photos is not a measured evaluation; quantitative real-image metrics require consistently annotated ground truth and documented image provenance. Keep those results separate from held-out synthetic test metrics.

## Held-out synthetic test set

The training process uses validation data to monitor training and select `best.pt`. The separate test-only command evaluates the selected checkpoint using the `test` path in `data/processed/dataset.yaml`:

```bash
python scripts/evaluate_test_set.py --device 0 --name palletjack-detection-yolo26n-seed42-test-reproduction
```

The command records the checkpoint and settings in `test_metrics.json`, creates PR/confusion plots and sample predictions, and writes output under `runs/detect/<name>/`. Choose a new name for each run; existing output folders are not overwritten. Leave out `--device 0` for automatic selection or pass `cpu` to force CPU evaluation.

### Current synthetic baseline

The selected YOLO26n checkpoint (`best.pt`, epoch 89 by validation mAP50-95) was evaluated at input size 640, batch size 16, on CUDA device 0 (NVIDIA GeForce RTX 5080 Laptop GPU).

| Split | Images | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 500 | 0.9313 | 0.8243 | 0.9098 | 0.7918 |
| Held-out test | 500 | 0.9248 | 0.8087 | 0.9059 | 0.7861 |

The test split has 200 `distractors_additional`, 200 `distractors_warehouse`, and 100 `no_distractors` images, with 1,427 labeled pallet-jack instances total. One close-up example (`1006.png`) can look like a mislabeled warehouse prop because the pallet jack is cropped and occupies most of the frame; visual review confirms it is a pallet jack. The reported scores measure performance on this held-out synthetic split against its exported labels. They do not measure real-photo performance or establish robustness to real camera, lighting, or occlusion conditions.

Even with verified labels, the result would be limited to held-out synthetic frames from the same data-generation setup. It would not measure transfer to real photographs, establish robustness to real lighting/cameras/occlusion, or prove deployment readiness.

Once the label audit is complete and the test metrics are regenerated, avoid repeatedly consulting this test set to tune confidence thresholds or choose among many checkpoints. Use a new untouched test set or independently labeled real-image benchmark for subsequent model changes. Keep real-image results separate from synthetic results and report dataset source, label status, checkpoint, settings, and metrics together.

Tracked visualization snapshots are in `docs/assets/test/`; full evaluation outputs remain under ignored `runs/`.
