# Data workflow

The original Isaac Sim collections and LOCO diagnostic are documented in [Experiment 1 (`EXP-001`)](experiments/experiment-01.md). Experiment 2 (`EXP-002`) has its own raw and processed roots and two completed training runs, but its original labels treat stackers as positive. Experiment 3 (`EXP-003`) has a separate raw collection under `data/raw/experiment-03/`, the current manual/electric pallet-truck target scope, and a target-free hard-negative condition; its data were validated, split, processed, trained, and evaluated. Experiment 4 (`EXP-004`) combines the existing processed Experiment 1 and 3 splits to compare pretrained and scratch initialization without another simulation run. See the [Experiment 4 record](experiments/experiment-04.md) for the exact data composition and label-scope rationale.

This document records the current Isaac Sim synthetic-data export and the path into object-detection training. Experiments 1–3 retain isolated raw and processed folders. Experiment 4 is a lightweight combined view: `configs/experiment-04-dataset.yaml` points at existing Experiment 1 and 3 processed split folders, so it does not duplicate their large image files. The LOCO source images are retained once under Experiment 1 because they were first used for external evaluation; their broad labels remain diagnostic until equipment subtypes are reviewed.

## Experiment-scoped layout

```text
data/
├── raw/
│   ├── experiment-01/       # Original synthetic exports and LOCO source files used for Experiment 1
│   ├── experiment-02/       # New Isaac Sim exports for Experiment 2
│   └── experiment-03/       # New positive-scene groups and target-free hard negatives
└── processed/
    ├── experiment-01/       # Experiment 1 manifests, synthetic dataset, and real-evaluation adapters
    ├── experiment-02/       # Experiment 2 manifests and derived datasets
    └── experiment-03/       # Experiment 3 manifests and derived data
```

Experiment 4 does not create `data/raw/experiment-04/` or copy data into a new processed directory. Its tracked dataset definition combines the existing train, validation, and test paths from Experiments 1 and 3. Experiment 2 is omitted because stacker-positive labels conflict with the current policy.

The Experiment 3 raw collection was generated with four groups: `distractors_warehouse/` (2,000 nominal target-positive frames), `distractors_additional/` (2,000), `no_distractors/` (1,000), and `hard_negatives/` (1,000 target-free frames). After image cleanup, 5,762 paired RGB/annotation files remain. The hard-negative run loads the combined warehouse and additional distractor pools, including local props and the stacker, then randomizes their poses and the camera per frame without adding pallet-truck targets. One `generate_data.bat` or `generate_data.sh` run generates all four groups. If the three positive groups are already collected and only hard negatives need another run, use `generate_hard_negatives.bat` or `generate_hard_negatives.sh`. Both options write under `experiment-03` and leave the other experiment folders unchanged. Change the full launcher's `EXPERIMENT_ID` setting to create a separate folder for a future collection.

The validation, split, processing, training, and evaluation defaults continue to target Experiment 1. To work on another experiment, pass its raw and processed roots to the preparation commands and select its training config. This keeps existing baseline commands reproducible and prevents a new experiment from replacing prior data. Experiments 1 and 2 reuse the same processed LOCO adapter, but its broad `pallet_truck` annotations have not been audited to distinguish in-scope manual/electric pallet trucks from stackers. They are not verified labels for the current target.

## Current synthetic source dataset inventory

Experiment 1's local export contains three scenario groups:

| Group | Frames | Description |
|---|---:|---|
| `no_distractors` | 1,000 | Pallet-jack scenes without the added distractors. |
| `distractors_warehouse` | 2,000 | Pallet-jack scenes with warehouse distractors. |
| `distractors_additional` | 2,000 | Pallet-jack scenes with additional distractors. |

Each Experiment 1 group is under `data/raw/experiment-01/<group>/Camera/` and contains these numbered, zero-based output folders:

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

Raw and processed data, training runs, and model weights are intended to stay out of Git. The root `.gitignore` excludes experiment contents while retaining `.gitkeep` placeholders for the experiment directories. Keep small documentation and code in the repository, not copies of the generated dataset.

## Read-only validator

Install dependencies with `python -m pip install -r requirements.txt`, then run the validator from the repository root:

```powershell
python scripts/validate_dataset.py --root data/raw/experiment-01 `
  --scenario distractors_additional `
  --scenario distractors_warehouse `
  --scenario no_distractors
```

The reusable API is named `validate_kitti_object_detection_export` in `src/palletjack_detection/validate_dataset.py` to make its supported annotation format and task explicit.

Use `--root <path>` for another export, repeat `--scenario <directory>` to select scenario folders when the raw root contains multiple datasets, and use `--max-issues <number>` to control how many issue details are printed. The command does not change files. It exits with status `0` when no issues are found and `1` when any issue is found. Empty annotation files are counted as empty frames and are not treated as errors; this preserves the possibility of legitimate negative examples.

The validator checks that required `Camera/rgb` and `Camera/object_detection` folders exist, pairs files by frame stem, opens and runs Pillow's file verification on every RGB image to get its actual dimensions, and checks label field count, finite numeric values, known class (`palletjack`), KITTI occlusion range, positive 2D box extent, and image bounds. Pillow verification checks file integrity without fully decoding all image pixels. The validator reports unreadable images, missing pairs, unknown classes, and malformed annotations. It currently recognizes common image extensions (PNG, JPEG, BMP, TIFF, and WebP). **It does not inspect image content or verify that a `palletjack` box encloses a pallet jack.**

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

Empty annotations were accepted and counted as negative frames. The clean report is evidence of file pairing and the checks above; it does not establish visual class correctness, settle the coordinate-edge convention, or validate KITTI 3D values. A close-up sample that initially looked suspicious was reviewed and confirmed to show a pallet jack; this spot check is not a full visual audit. The local annotation-audit gallery can be used to review the stored boxes across splits.

## Experiment 2 prepared dataset

Experiment 2's local USDZ assets were added to the Isaac Sim generator while preserving the original NVIDIA asset. They were tagged as positive pallet-jack instances under the original interpretation; they are now identified as an electric pallet truck and a forklift stacker. The current target includes electric pallet trucks, but excludes stackers. Experiment 2 therefore has stackers incorrectly labeled positive and remains a historical dataset under its earlier labeling policy. Future target-positive generation includes the electric pallet truck; the stacker and four local assets (`forklift.usdz`, `ladder.usdz`, `shopping_cart.usdz`, and `warehouse_rack.usdz`) are distractors in both regular groups and in the target-free hard-negative pool. None receive a pallet-jack label. Local USDZ files are ignored by Git. The generated RGBs include randomized appearance variants and imported-material appearances. `RTSubframes` was raised to `10` after intermittent transparent-black frames were observed; despite this, 176 fully transparent black RGB frames and their paired KITTI files were removed from the raw export. See the [Experiment 2 record](experiments/experiment-02.md) for details.

The raw export was then curated and validated. Three zero-width KITTI boxes were removed from their label files while preserving the corresponding images and all other valid boxes. The current resulting counts are:

| Scenario | Images and labels | Valid objects | Empty annotations |
|---|---:|---:|---:|
| `distractors_additional` | 1,941 | 8,740 | 129 |
| `distractors_warehouse` | 1,936 | 9,587 | 118 |
| `no_distractors` | 947 | 5,017 | 54 |
| **Total** | **4,824** | **23,344** | **301** |

The prepared dataset has 3,859 training, 483 validation, and 482 test images using the same seed-42 80/10/10 per-scenario split policy. Validation and processing are complete, and both training configurations passed preflight. The two trained models were validated during training and evaluated on the held-out test split. See the [experiment record](experiments/experiment-02.md) for per-scenario split counts, validation scores, and test comparison.

## Experiment 3 raw collection and curation

Experiment 3 follows the revised label policy: manual and electric pallet trucks are positive `palletjack` targets; forklift stackers and other warehouse props are unlabeled distractors. It includes three positive-scene conditions and a separate negative-only condition. The hard-negative scenes contain the combined warehouse and additional distractor pools, randomized in pose and camera view with no pallet-truck target. Their KITTI label files are empty, as expected for negative images.

The nominal collection contained 6,000 frames. Fully transparent-black RGB images and their corresponding KITTI labels were removed; one additional entirely black RGB image was removed by the same invalid-frame cleanup. The resulting retained pairs are:

| Group | RGB/annotation pairs | Valid 2D objects | Empty labels |
|---|---:|---:|---:|
| `distractors_additional` | 1,937 | 4,918 | 285 |
| `distractors_warehouse` | 1,948 | 5,829 | 230 |
| `hard_negatives` | 908 | 0 | 908 |
| `no_distractors` | 969 | 3,869 | 59 |
| **Total** | **5,762** | **14,616** | **1,482** |

The cleanup removed 238 RGB/label pairs: 63 from `distractors_additional`, 52 from `distractors_warehouse`, 92 from `hard_negatives`, and 31 from `no_distractors`. Image and annotation stems match in every group. Raw KITTI validation initially found a zero-width box in `distractors_warehouse/Camera/object_detection/1727.txt`; the invalid annotation row was removed, preserving the image and its three valid boxes. Validation was rerun and passed with no issues. The retained data were then split, processed, trained, and evaluated on the held-out synthetic test split. See the [Experiment 3 record](experiments/experiment-03.md) for split counts and model results.

## Experiment 1 synthetic train/validation/test split

Experiment 1 frames are independent scenes, so its split uses a seeded random allocation within each of the three scenario groups. The default proportions are 80% training, 10% validation, and 10% test, with random seed `42`. Experiment 1 produced:

| Group | Train | Validation | Test |
|---|---:|---:|---:|
| `distractors_additional` | 1,600 | 200 | 200 |
| `distractors_warehouse` | 1,600 | 200 | 200 |
| `no_distractors` | 800 | 100 | 100 |
| **Total** | **4,000** | **500** | **500** |

Create the split manifests from the repository root after installing dependencies:

```bash
python scripts/create_splits.py \
  --scenario distractors_additional \
  --scenario distractors_warehouse \
  --scenario no_distractors
```

`scripts/create_splits.py` validates the selected scenario folders and writes `train.csv`, `validation.csv`, and `test.csv` under `data/processed/experiment-01/splits/` by default. The explicit selection keeps `real-pallet-jack-data` from being mistaken for an Isaac Sim scenario. It assigns each frame once, stratifies within scenario, and records raw source paths plus intended paths for processed RGB, copied KITTI labels, and converted YOLO labels. It does not copy files or convert annotations. With the same source and seed, it reproduces the same assignments.

For the current data, the default 80/10/10 split still gives 4,000 training, 500 validation, and 500 test frames. Since manifests already exist in this workspace, regenerate them with `--overwrite` after changing the split workflow:

```bash
python scripts/create_splits.py --overwrite \
  --scenario distractors_additional \
  --scenario distractors_warehouse \
  --scenario no_distractors
```

### Process the selected raw files

After split manifests are ready, copy the selected RGB images and KITTI labels to the processed tree and convert the copied labels:

```bash
python scripts/process_raw_data.py
```

The processor reads the manifests, validates only the scenario folders named by those manifests, and copies only these files:

- RGB images to `data/processed/experiment-01/images/{train,val,test}/<scenario>/`.
- Original KITTI labels to `data/processed/experiment-01/kitti_labels/{train,val,test}/<scenario>/`.
- Converted YOLO labels to `data/processed/experiment-01/labels/{train,val,test}/<scenario>/`.

It calls `convert_kitti_annotation_file_to_yolo` from `src/palletjack_detection/kitti_to_yolo.py` for each copied image/label pair and writes `data/processed/experiment-01/dataset.yaml`. Depth, semantic, and instance-segmentation data remain only in the matching raw experiment folder. The RGB copy requires additional disk space approximately equal to the selected images (about 3.7 GB for this export).

The YOLO converter module contains annotation conversion only. Its standalone command converts one label file:

```bash
python scripts/convert_kitti_to_yolo.py --kitti-label path/to/frame.txt --image path/to/frame.png --output-label path/to/yolo/frame.txt
```

If raw-data processing stops partway through, rerun `scripts/process_raw_data.py --overwrite` to replace only outputs marked as generated by that processor.

Use `--seed`, `--root`, `--output-dir`, and `--processed-root` to change the split seed, source root, manifest location, or recorded processed paths. The proportions can be changed with `--train-ratio`, `--validation-ratio`, and `--test-ratio`; they must be positive and sum to 1.0. Both the processor and the single-file converter accept `--exclusive-max-coordinates` if the writer convention is confirmed to use exclusive upper edges. The current default treats maximum coordinates as inclusive pixel indices; this remains a working assumption pending confirmation.

For Experiment 2, pass `data/raw/experiment-02` as the source root, `data/processed/experiment-02/splits` as the manifest folder, and `data/processed/experiment-02` as the processed root. The [main README](../README.md#use-experiment-2-data) gives the commands to reproduce preparation after deliberate regeneration. The prepared dataset was used for both completed Experiment 2 training runs and the held-out synthetic test evaluation; see the [Experiment 2 record](experiments/experiment-02.md).

LOCO real photos are prepared and evaluated through a separate COCO-to-YOLO path; they are intentionally excluded from these synthetic manifests.

Keep the scenario identity available in manifests or split metadata so evaluation can report results by condition. Split by independent scene/capture groups if that information is available; avoid assigning near-duplicate frames from the same capture across training and evaluation splits.

## Real-image evaluation data

### LOCO real-world pallet-truck images

The project uses the public **Logistics Objects in Context (LOCO)** dataset from the Technical University of Munich's Materials Handling, Material Flow, Logistics group. LOCO contains real logistics-scene images and manually annotated bounding boxes for forklifts, pallet trucks, pallets, small load carriers, and stillages. The official [LOCO repository](https://github.com/tum-fml/loco) reports 37,988 images across five logistics environments, of which 5,593 are annotated, and publishes the dataset under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Cite the dataset as Mayershofer, Holm, Molter, and Fottner, “LOCO: Logistics Objects in Context,” ICMLA 2020; see the [paper](https://mediatum.ub.tum.de/doc/1578845/document.pdf) and [repository](https://github.com/tum-fml/loco).

The local, Git-ignored copy is in `data/raw/experiment-01/real-pallet-jack-data/`. It contains only `subset-1` images plus `loco-all-v1.json`. The official repository identifies subset 1 as a validation subset and provides COCO annotations in `rgb/loco-sub1-v1-val.json` as well as a full annotation file. The local JSON currently retains only records whose filenames match subset 1; it has 1,075 image records and 24,763 annotations across the original five classes, including 594 `pallet_truck` instances across 403 images. Of the 1,075 image records, 672 have no `pallet_truck` annotation. These are negative only for LOCO's recorded category; they have not been audited for stacker objects or verified as negatives for the current target. The folder has 1,140 JPGs, so 65 local images have no image record in this JSON. Keep those source files, but do not score them as negatives until ground truth is confirmed for them.

The existing adapter filters LOCO's `pallet_truck` category and maps it to `pallet-jack`, retaining all source images and empty annotations for the 672 JSON-listed images without a `pallet_truck` box. A visual spot-check suggests the boxes generally surround material-handling equipment, but the current target includes manual and electric pallet trucks while excluding stackers, and LOCO does not split those subtypes in its broad category. The filtered labels are therefore preserved for historical reproduction but are not fully verified current-scope ground truth; subtype review is still needed before interpreting the metrics for the current target. The 65 JPGs missing from the COCO `images` array need separate ground-truth review before inclusion. Keep this real-image evaluation data separate from synthetic training and evaluation, and report its metrics separately. The COCO annotations remain preserved as the source labels.

## COCO-to-YOLO adapter for real-image evaluation

The filtered LOCO COCO file used for historical Experiment 1 evaluation is retained at `data/processed/experiment-01/real-pallet-jack-data/annotations.json`. Ultralytics detection validation expects normalized YOLO text labels, so the project converts the COCO `bbox` values (`x`, `y`, `width`, `height`) using each listed image's actual dimensions. The resulting line is `class_id center_x center_y width height`; the one class, `pallet-jack`, maps to ID `0`. Every COCO image record gets a label file, including an empty file for each of the 672 images without a `pallet_truck` annotation. This adapter leaves the COCO JSON and source images unchanged. The output class name does not imply those boxes have been audited to separate in-scope pallet trucks from stackers.

Run from the repository root after processing the filtered LOCO data:

```bash
python scripts/convert_coco_to_yolo.py
```

The command writes `labels/`, `dataset.yaml`, and `conversion_report.json` below `data/processed/experiment-01/real-pallet-jack-data/`. Existing generated conversion outputs are protected; use `--overwrite` only to regenerate those outputs. The YAML declares all images as the `test` split, with no train or validation split, to avoid using this set to train the synthetic baseline. The converter validates box extents against actual image dimensions and rejects path escapes or malformed boxes.

The current conversion report records 1,075 images, 594 boxes from LOCO's broad `pallet_truck` category across 403 images, and 672 images without a box in that category. Those counts are not verified current-scope positives and negatives until the truck boxes are reviewed for stackers. The additional 65 JPGs missing from the COCO image list remain excluded. Conversion to YOLO is an evaluation-format adapter, not a replacement for COCO ground truth.

## Details still to confirm

- Isaac Sim version, scene/source project, camera configuration, and exact SDG export settings.
- Provenance and license for any additional data sources beyond LOCO.
- Isaac Sim image dimensions are reported as 960×544; validation reads each image's actual dimensions.
- KITTI coordinate conventions, including whether maximum box coordinates are inclusive or exclusive.
- Whether frames are independent captures or sequences, and what grouping information is available for leakage-aware splits.
- Whether future experiments should change the current YOLO format or 80/10/10 split policy.
- Complete visual audit and broader independent real-world evaluation beyond the selected LOCO validation subset.
