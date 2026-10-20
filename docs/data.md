# Data workflow

This document records the current local Isaac Sim synthetic-data export and the path into object-detection training. The raw export is kept as source data; processing scripts write derived files under `data/processed/` and leave the raw files unchanged.

## Current synthetic datasets

The local export contains three scenario groups:

| Group | Frames | Description |
|---|---:|---|
| `no_distractors` | 1,000 | Pallet-jack scenes without the added distractors. |
| `distractors_warehouse` | 2,000 | Pallet-jack scenes with warehouse distractors. |
| `distractors_additional` | 2,000 | Pallet-jack scenes with additional distractors. |

Each group is under `data/raw/<group>/Camera/` and contains these numbered, zero-based output folders:

- `rgb/` — color images used as the detector input.
- `object_detection/` — KITTI-style text annotations converted to YOLO targets during processing.
- `depth/` — depth images.
- `instance_segmentation/` — instance segmentation images.
- `semantic/` — semantic output images.
- `semantic_segmentation/` — semantic segmentation images.

A local inventory found matching file indices in all six folders within each group: `0`–`999` for `no_distractors`, and `0`–`1999` for each distractor group. This is an initial structural check, not a full image/label validity check. The other camera outputs are retained in the raw export for possible future uses; the object-detection workflow uses RGB plus `object_detection` labels. Images are reported as 960×544; validation reads actual dimensions from each file.

Each group also has a `metadata.txt` identifying the writer as `KittiWriter`, version `0.0.2`, and recording a Replicator global seed. Preserve these metadata files with the raw data.

## Annotation notes

A sample `object_detection` line has this form:

```text
palletjack 0.00 0 0.00 808 0 959 141 0.00 0.00 0.00 0.00 0.00 0.00 0.00
```

It has the standard KITTI object-label field ordering: class, truncation, occlusion, observation angle, 2D box (`x_min y_min x_max y_max`), followed by 3D/location fields. The observed class name is `palletjack`. For 2D detection conversion, the class and 2D box are the relevant fields; confirm the writer's coordinate convention and image bounds in the validator before converting. Do not infer image dimensions from the example label.

## Git and local storage

Raw and processed data, training runs, and model weights are intended to stay out of Git. The root `.gitignore` currently excludes `data/raw/`, `data/processed/`, `runs/`, and `models/`. Keep small documentation and code in the repository, not copies of the generated dataset.

## Read-only validator

Install dependencies with `python -m pip install -r requirements.txt`, then run the validator from the repository root:

```powershell
python scripts/validate_dataset.py
```

The reusable API is named `validate_kitti_object_detection_export` in `src/palletjack_detection/validate_dataset.py` to make its supported annotation format and task explicit.

Use `--root <path>` for another export and `--max-issues <number>` to control how many issue details are printed. The command does not change files. It exits with status `0` when no issues are found and `1` when any issue is found. Empty annotation files are counted as empty frames and are not treated as errors; this preserves the possibility of legitimate negative examples.

The validator checks that required `Camera/rgb` and `Camera/object_detection` folders exist, pairs files by frame stem, opens and runs Pillow's file verification on every RGB image to get its actual dimensions, and checks label field count, finite numeric values, known class (`palletjack`), KITTI occlusion range, positive 2D box extent, and image bounds. Pillow verification checks file integrity without fully decoding all image pixels. The validator reports unreadable images, missing pairs, unknown classes, and malformed annotations. It currently recognizes common image extensions (PNG, JPEG, BMP, TIFF, and WebP).

This is a validator for the 2D object-detection fields, not a full semantic validation of KITTI's 3D dimensions and pose values. It checks that the remaining fields are present and numeric, but does not interpret them; the current export includes zero-filled 3D fields.

The validator permits an upper box edge equal to the image width or height. The writer's exact coordinate convention has not been confirmed, so conversion currently uses the inclusive-pixel convention as a documented working assumption. The converter and processor expose `--exclusive-max-coordinates` if evidence establishes that the writer uses exclusive upper edges.

## Preparation workflow and checks

The validator summarizes counts and issues separately for each scenario. The following validation and preparation steps have been run for the current export.

### Current validation result

The validator was run on the local export on 2026-10-18. It found no structural or 2D annotation issues:

| Group | Images | Annotation files | Valid 2D objects | Empty annotations |
|---|---:|---:|---:|---:|
| `distractors_additional` | 2,000 | 2,000 | 5,017 | 259 |
| `distractors_warehouse` | 2,000 | 2,000 | 5,826 | 190 |
| `no_distractors` | 1,000 | 1,000 | 3,242 | 65 |

Empty annotations were accepted and counted as negative frames. The clean report is evidence of file pairing and the checks above; it does not settle the coordinate-edge convention or validate KITTI 3D values.

## Synthetic train/validation/test split

The frames are independent scenes, so the first experiment uses a seeded random split within each of the three scenario groups. The default proportions are 80% training, 10% validation, and 10% test, with random seed `42`. For the current dataset this produces:

| Group | Train | Validation | Test |
|---|---:|---:|---:|
| `distractors_additional` | 1,600 | 200 | 200 |
| `distractors_warehouse` | 1,600 | 200 | 200 |
| `no_distractors` | 800 | 100 | 100 |
| **Total** | **4,000** | **500** | **500** |

Create the split manifests from the repository root after installing dependencies:

```bash
python scripts/create_splits.py
```

`scripts/create_splits.py` validates the raw export and writes `train.csv`, `validation.csv`, and `test.csv` under `data/processed/splits/`. It assigns each frame once, stratifies within scenario, and records raw source paths plus intended paths for processed RGB, copied KITTI labels, and converted YOLO labels. It does not copy files or convert annotations. With the same source and seed, it reproduces the same assignments.

For the current data, the default 80/10/10 split still gives 4,000 training, 500 validation, and 500 test frames. Since manifests already exist in this workspace, regenerate them with `--overwrite` after changing the split workflow:

```bash
python scripts/create_splits.py --overwrite
```

### Process the selected raw files

After split manifests are ready, copy the selected RGB images and KITTI labels to the processed tree and convert the copied labels:

```bash
python scripts/process_raw_data.py
```

The processor reads the manifests, validates the raw export, and copies only these files:

- RGB images to `data/processed/images/{train,val,test}/<scenario>/`.
- Original KITTI labels to `data/processed/kitti_labels/{train,val,test}/<scenario>/`.
- Converted YOLO labels to `data/processed/labels/{train,val,test}/<scenario>/`.

It calls `convert_kitti_annotation_file_to_yolo` from `src/palletjack_detection/kitti_to_yolo.py` for each copied image/label pair and writes `data/processed/dataset.yaml`. Depth, semantic, and instance-segmentation data remain only in `data/raw/`. The RGB copy requires additional disk space approximately equal to the selected images (about 3.7 GB for this export).

The YOLO converter module contains annotation conversion only. Its standalone command converts one label file:

```bash
python scripts/convert_kitti_to_yolo.py --kitti-label path/to/frame.txt --image path/to/frame.png --output-label path/to/yolo/frame.txt
```

If raw-data processing stops partway through, rerun `scripts/process_raw_data.py --overwrite` to replace only outputs marked as generated by that processor.

Use `--seed`, `--root`, `--output-dir`, and `--processed-root` to change the split seed, source root, manifest location, or recorded processed paths. The proportions can be changed with `--train-ratio`, `--validation-ratio`, and `--test-ratio`; they must be positive and sum to 1.0. Both the processor and the single-file converter accept `--exclusive-max-coordinates` if the writer convention is confirmed to use exclusive upper edges. The current default treats maximum coordinates as inclusive pixel indices; this remains a working assumption pending confirmation.

Real photographs will be collected and evaluated separately later; they are not part of these synthetic manifests.

Keep the scenario identity available in manifests or split metadata so evaluation can report results by condition. Split by independent scene/capture groups if that information is available; avoid assigning near-duplicate frames from the same capture across training and evaluation splits.

## Real-image evaluation data

Real pallet-jack photographs are not currently included. Near the evaluation stage, identify images with a clear source and reuse license/permission, document their provenance, and annotate them consistently. Keep real images in a separate evaluation set and report real-image results separately from synthetic validation/test results. Do not mix them into training unless that becomes an explicit, documented experiment.

## Details still to confirm

- Isaac Sim version, scene/source project, camera configuration, and exact SDG export settings.
- Dataset origin and applicable asset/data license or usage restrictions.
- Isaac Sim image dimensions are reported as 960×544; validation reads each image's actual dimensions.
- KITTI coordinate conventions, including whether maximum box coordinates are inclusive or exclusive.
- Whether frames are independent captures or sequences, and what grouping information is available for leakage-aware splits.
- Whether future experiments should change the current YOLO format or 80/10/10 split policy.
- Provenance and licensing for any real-image evaluation set.
