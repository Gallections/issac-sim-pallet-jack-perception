# Evaluation

Report synthetic and real-image results separately. Record model checkpoint, dataset provenance, thresholds, metrics where labels are available, and representative errors.

For interactive exploration of a saved training run and optional real-image predictions, see `notebooks/01_explore_training_results.ipynb` and the launch instructions in `docs/training.md`.

## Inspect predictions on real images

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

The training process uses the validation split to monitor training and select `best.pt`; it does not report final held-out test metrics. The test split should be evaluated separately after a checkpoint is selected. Test-set evaluation should not be used to tune confidence thresholds or make repeated model-selection decisions.
