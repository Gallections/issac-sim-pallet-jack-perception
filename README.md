# Isaac Sim Pallet-Jack Perception

An end-to-end research project for generating synthetic pallet-jack images in NVIDIA Isaac Sim, preparing them for object detection, fine-tuning an Ultralytics YOLO model outside the simulator, and eventually evaluating and deploying that detector.

The repository keeps the work in distinct stages: simulator-based synthetic data generation, reproducible dataset preparation, external model training, visual and quantitative evaluation, and a planned path back to Isaac Sim for inference.

## At a glance

- **Task:** detect the `palletjack` object and return its image bounding box.
- **Synthetic source data:** 5,000 independently randomized rendered frames from a shared Isaac Sim warehouse environment, across three conditions.
- **Dataset splits:** 4,000 train, 500 validation, and 500 held-out test images (80/10/10, seed 42).
- **Baseline:** fine-tuned Ultralytics YOLO26n, 100 epochs, 640-pixel model input, batch size 16.
- **Current state:** Experiment 1 is the completed baseline. Experiment 2 has two completed training runs, both evaluated on the same held-out synthetic test split. LOCO subset-1 was evaluated separately for Experiment 1.
- **Not yet complete:** a broader, independently audited real-image benchmark and Isaac Sim inference integration.

## Why this project matters

This project tests a practical sim-to-real research question: **can pallet-jack examples generated in Isaac Sim provide useful training data for an object detector, and how well does the resulting detector perform on unseen scenes?** Synthetic generation makes it possible to create varied, automatically annotated examples without manually labeling every frame. That can reduce the cost of building an initial perception dataset and makes controlled scenario variation possible.

Experiment 1 is the initial, reproducible synthetic-data baseline. Its model was fine-tuned outside Isaac Sim and evaluated on a held-out synthetic split. Experiment 2 changes the pallet-jack asset mix to address appearance and scale differences highlighted by the separate LOCO diagnostic. Both Experiment 2 training runs are complete. Their held-out synthetic test scores are nearly identical; the result does not show a meaningful advantage from initializing with Experiment 1's checkpoint. Experiment 1 and 2 results show performance on unseen samples from their respective synthetic data collections—not real-world accuracy, robustness to deployment conditions, or readiness for a robot.

The research contribution at this stage is the documented end-to-end method and its measurements: controlled synthetic scene variation, automatically generated labels, a reproducible data-preparation and split process, external fine-tuning, and held-out evaluation. Experiment 1 also measures synthetic-to-real transfer on the selected LOCO subset, but it does not quantify which randomization choices help most or show that synthetic training data can replace real data. Experiment 2 compares two initialization choices after adding asset variants; it currently shows similar held-out synthetic scores, while real-world transfer remains to be established.

## Research workflow

```mermaid
flowchart LR
    A[Isaac Sim base warehouse] --> B[Randomize one scene state per frame]
    B --> C[Replicator renders RGB and writes labels]
    C --> D[Raw scenario exports]
    D --> E[Validate raw KITTI annotations]
    E --> F[Create seeded split manifests]
    F --> G[Copy selected RGB and preserve KITTI labels]
    G --> H[Convert KITTI boxes to YOLO labels]
    H --> I[Fine-tune YOLO externally]
    I --> J[Explore training and validation artifacts]
    I --> K[Evaluate held-out synthetic test set]
    L[Filter public LOCO subset-1 COCO data] --> O[Convert COCO boxes to YOLO evaluation labels] --> N[Evaluate real images separately]
    I --> N
    I --> M[Integrate inference in Isaac Sim - planned]
```

Isaac Sim creates the source data; training and evaluation run separately in a standard Python environment with PyTorch and Ultralytics. This keeps model work repeatable without requiring the simulator to stay open.

## How the synthetic data is generated

The data-generation script starts from Isaac Sim's Simple Warehouse environment. For each requested frame, NVIDIA Replicator samples a new scene state, renders that state, and exports image data with labels. The training examples are therefore **independent randomized frames**, not a set of consecutive frames from one continuous camera recording. They share a base warehouse environment while camera, object, material, and lighting settings vary between frames.

The generator adds pallet-jack assets and tags them with the semantic class `palletjack`. It samples camera position and orientation, and varies pallet-jack placement and rotation. It also randomizes floor and wall materials and lighting. Depending on the run, it adds warehouse props, props drawn from additional asset groups, or no extra distractor group. “No distractors” means no additional randomized distractors are added; the base warehouse environment remains.

Replicator's KITTI writer exports the rendered RGB frames and KITTI-style object-detection annotations. The pallet-jack semantic tags are intended to let the writer create object labels automatically, avoiding manual box drawing for synthetic images. Automatic export does not by itself guarantee that each box encloses a pallet jack: visual semantic review is still necessary. The downstream detector currently uses RGB plus the 2D boxes only; other modalities present in the raw export are retained as source data but are outside this baseline's training input.

The current collection contains three generation conditions:

| Generation condition | Frames | How the condition is built |
| --- | ---: | --- |
| `distractors_warehouse` | 2,000 | Base warehouse plus randomized warehouse props such as signs, containers, and carts |
| `distractors_additional` | 2,000 | Base warehouse plus distractors selected from additional asset groups, including hospital and office props |
| `no_distractors` | 1,000 | Base warehouse and pallet jacks, without an added distractor group |

The current generator requests 960x544 renders. These are unannotated RGB examples from each condition. They illustrate scene appearance only; they are not examples of verified ground-truth boxes.

| Warehouse distractors | Additional distractors | No added distractors |
| --- | --- | --- |
| ![Isaac Sim synthetic RGB frame with warehouse distractors](docs/assets/sdg/distractors_warehouse.jpg) | ![Isaac Sim synthetic RGB frame with additional distractors](docs/assets/sdg/distractors_additional.jpg) | ![Isaac Sim synthetic RGB frame without added distractors](docs/assets/sdg/no_distractors.jpg) |

The walkthrough is shown inline as a clickable preview; select it to open the [scene-generation video](docs/assets/sdg/SDG_sample0.mov):

[![Play the Isaac Sim scene-generation walkthrough](docs/assets/sdg/walkthrough-poster.jpg)](docs/assets/sdg/SDG_sample0.mov)

GitHub README pages do not play repository-relative video files directly, so this preview links to the video file. The implementation is in [`simulation/scripts/palletjack_sdg/standalone_palletjack_sdg.py`](simulation/scripts/palletjack_sdg/standalone_palletjack_sdg.py); its Isaac Sim runtime requirements and generation settings are summarized in [`simulation/README.md`](simulation/README.md).

## From generated frames to a detector

The data flow preserves both the source annotations and the model-ready annotations. Each experiment has its own raw and processed roots, such as `data/raw/experiment-01/` and `data/processed/experiment-01/`. Raw Isaac Sim outputs are not edited by preparation steps. The validator checks image/annotation pairs and box geometry. A seeded split step assigns every frame to train, validation, or test; the preparation step then copies selected RGB images and original KITTI labels into the matching processed experiment folder, converts the boxes to normalized YOLO text labels, and writes that experiment's dataset YAML.

The pretrained YOLO26n detector is fine-tuned outside the simulator on the train split. Validation data guides checkpoint selection. The selected checkpoint is then evaluated on the held-out synthetic test split. LOCO subset-1 has been filtered to the pallet-truck class, mapped to `pallet-jack`, and converted from COCO to YOLO labels solely to use Ultralytics validation. Its real-image evaluation is reported separately from the synthetic held-out test. Dataset provenance and counts are documented in [`docs/data.md`](docs/data.md).

### Experimental design and what the metrics mean

Frames are split within each generation condition using a fixed seed, so every condition contributes to train, validation, and test. The training split updates the model weights; validation supports checkpoint selection; the held-out test split is reserved for the reported final synthetic score. Since frames are separate randomized states rather than adjacent video frames, this design reduces direct frame-to-frame leakage. It still tests generalization within the same simulator assets and base environment.

Precision reflects how many predicted boxes are correct at the metric's operating point; recall reflects how many labeled objects are detected. `mAP50` summarizes average precision at an intersection-over-union (IoU) threshold of 0.50. `mAP50–95` averages that measure over IoU thresholds from 0.50 through 0.95, so it rewards tighter localization as well as detection. For this one-class project, “mean” is over the pallet-jack class. These scores are useful for comparing detector behavior on this dataset, but they do not replace testing under real camera, lighting, background, and occlusion conditions.

## Experiment 1 results

This section reports Experiment 1 (`EXP-001`) only. Experiment 2 has its own [record](docs/experiments/experiment-02.md); it is not included in these metrics or figures.

The current baseline used `yolo26n.pt`, seed 42, 100 epochs, a 640 input size, and batch size 16. The selected checkpoint is `best.pt`, chosen using validation mAP50-95. In this run, its best validation mAP50-95 was **0.79183 at epoch 89**. At that epoch, validation precision was **0.93125**, recall **0.82429**, and mAP50 **0.90984**.

The model was evaluated on the held-out synthetic test split after checkpoint selection. Validation and test metrics are shown side by side below. The test set contains 500 images and 1,427 labeled pallet jacks. Metrics were computed with Ultralytics at 640 input size on an NVIDIA RTX 5080 Laptop GPU. The plots are checked-in snapshots so they render on GitHub; full run files and checkpoints remain local under the ignored `runs/` directory.

| Split | Images | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation, best checkpoint at epoch 89 | 500 | 0.9313 | 0.8243 | 0.9098 | 0.7918 |
| Held-out synthetic test | 500 | 0.9248 | 0.8087 | 0.9059 | 0.7861 |
| LOCO subset-1 real evaluation | 1,075 | 0.4113 | 0.1768 | 0.1430 | 0.0600 |

**Reading the test examples:** one close-up frame (`1006.png`) may initially look like a mislabeled warehouse prop because the pallet jack is cropped and fills most of the image. On review, it is a pallet jack; its large box reflects the close camera view. The automated validator checks annotation structure and box geometry, while visual review helps confirm object identity. The reported metrics measure performance against the exported synthetic labels and should be interpreted as synthetic-test results, not real-world accuracy.

The LOCO row is a separate real-image evaluation: 1,075 COCO-listed images, 594 pallet-jack boxes, 403 positive images, and 672 negative images. The 65 JPGs missing from the COCO image records were excluded. This is evidence of a substantial synthetic-to-real performance gap on this selected logistics subset, but it is not a broad real-world benchmark or evidence of deployment readiness. The source is an official validation subset, and annotations need broader visual auditing; do not tune the model against this set and then describe it as an untouched final test. Future claims should use a broader independently labeled real-world test set.

### Training losses and validation metrics

This training summary shows box, classification, and localization losses for train and validation data, alongside validation precision, recall, mAP50, and mAP50-95 by epoch. Loss and validation metric trends help diagnose training; the held-out test results are reported separately below.

![YOLO training losses and validation metrics across epochs](docs/assets/baseline/training-results.png)

### Validation precision-recall curve

The curve summarizes the precision/recall tradeoff as the confidence threshold varies. For a single class, the chart reports pallet-jack average precision; the displayed AP at IoU 0.50 is not the same as mAP50-95.

![Validation precision-recall curve](docs/assets/baseline/precision-recall.png)

### Validation confusion matrix

The raw matrix contains counts, while the normalized matrix expresses the counts as proportions. In an object detector, the background row/column helps show missed objects and detections without a matching labeled object. Read the axis labels as well as the cells: this plot is based on validation predictions and its normalization is performed by Ultralytics.

![Normalized validation confusion matrix](docs/assets/baseline/confusion-matrix-normalized.png)

<details>
<summary>Raw validation confusion matrix (counts)</summary>

![Raw validation confusion matrix](docs/assets/baseline/confusion-matrix-counts.png)

</details>

### Example validation images

The label montage shows the expected annotations; the prediction montage shows what the model predicted for those same validation examples. These images are a qualitative sample, not an independent accuracy measurement.

| Validation ground-truth boxes | Validation model predictions |
| --- | --- |
| ![Ground-truth boxes on validation images](docs/assets/baseline/validation-ground-truth.jpg) | ![Predicted boxes on validation images](docs/assets/baseline/validation-predictions.jpg) |

### Held-out synthetic test results

The test precision-recall curve and confusion matrices below were generated after training and checkpoint selection. The normalized matrix is useful for class-wise proportions; the raw-count matrix is included for inspection, but its object-detection background row/column should not be interpreted as a simple count of false-positive images. The montages are a small qualitative sample, not a substitute for the reported aggregate metrics.

![Held-out synthetic test precision-recall curve](docs/assets/test/precision-recall.png)

![Normalized held-out synthetic test confusion matrix](docs/assets/test/confusion-matrix-normalized.png)

<details>
<summary>Raw held-out synthetic test confusion matrix (counts)</summary>

![Raw held-out synthetic test confusion matrix](docs/assets/test/confusion-matrix-counts.png)

</details>

| Test ground-truth boxes | Test model predictions |
| --- | --- |
| ![Ground-truth boxes on held-out test images](docs/assets/test/test-ground-truth.jpg) | ![Predicted boxes on held-out test images](docs/assets/test/test-predictions.jpg) |

### Separate real-image evaluation: LOCO subset-1

The YOLO26n checkpoint trained on Isaac Sim data was evaluated against the LOCO subset-1 COCO pallet-truck boxes after class filtering and COCO-to-YOLO conversion. The precision-recall curve and normalized confusion matrix summarize this separate set; the image pair shows a sample of the ground truth and predictions.

![LOCO real-image precision-recall curve](docs/assets/real-loco/precision-recall.png)

![Normalized LOCO real-image confusion matrix](docs/assets/real-loco/confusion-matrix-normalized.png)

| LOCO ground-truth boxes | LOCO model predictions |
| --- | --- |
| ![LOCO ground-truth boxes](docs/assets/real-loco/evaluation-ground-truth.jpg) | ![LOCO model predictions](docs/assets/real-loco/evaluation-predictions.jpg) |

The low real-set scores indicate that this synthetic-trained baseline transfers poorly to these LOCO images. They do not identify the cause by themselves: appearance/domain shift, annotation differences, class mapping, and confidence/IoU behavior can all contribute. See [evaluation notes](docs/evaluation.md) for the exact settings and limits.

## Experiment 2 results

Experiment 2 added two locally supplied pallet-jack USDZ assets to the Isaac Sim scene alongside NVIDIA's original asset. Two YOLO26n models were trained for 100 epochs on the same 3,859-image training split. Validation ran every epoch and selected each run's `best.pt` checkpoint by mAP50-95. The held-out test split contains 482 images and 2,307 labeled boxes; it was evaluated after training on CUDA device 0, with 640 input size and batch 16.

| Initialization | Best validation epoch | Validation mAP50-95 | Test precision | Test recall | Test mAP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| `yolo26n.pt` | 84 | 0.7242 | 0.9132 | 0.7847 | 0.8845 | 0.7380 |
| Experiment 1 `best.pt` | 82 | 0.7264 | 0.9108 | 0.7850 | 0.8795 | 0.7369 |

The test mAP50-95 scores differ by approximately 0.0011, which is too small to support a meaningful winner from this single split. Initializing from Experiment 1's checkpoint did not improve the Experiment 2 test score in this run. These are synthetic test results for Experiment 2's asset mix; they do not establish performance on real photos. See the [full Experiment 2 record](docs/experiments/experiment-02.md) for complete validation metrics, plots, interpretation, and run details.

### Test-set examples

The following montage is a batch sampled by Ultralytics from the held-out test split:

| Ground truth | `yolo26n.pt` initialization | Experiment 1 `best.pt` initialization |
|---|---|---|
| ![Experiment 2 test ground truth](docs/assets/experiment-02/test-ground-truth.jpg) | ![Experiment 2 predictions, pretrained initialization](docs/assets/experiment-02/pretrained-test-predictions.jpg) | ![Experiment 2 predictions, Experiment 1 initialization](docs/assets/experiment-02/exp01-initialized-test-predictions.jpg) |

## Dataset and annotation format (Experiment 1)

Experiment 1's original raw export contains these three scenario groups:

| Scenario | Frames | Current labeled instances* | Empty annotations |
| --- | ---: | ---: | ---: |
| `distractors_additional` | 2,000 | 5,017 | 259 |
| `distractors_warehouse` | 2,000 | 5,826 | 190 |
| `no_distractors` | 1,000 | 3,242 | 65 |
| **Total** | **5,000** | **14,085** | **514** |

These are annotation counts; visual class correctness has not yet been fully audited.

Images are reported as 960x544. The raw folders contain RGB, object-detection labels, depth, instance segmentation, and semantic outputs. The current detector pipeline uses only RGB images and KITTI-style `object_detection` labels. It preserves original KITTI labels in the processed tree and writes YOLO labels alongside them; depth and segmentation outputs stay in the raw tree.

The converter reads the paired image dimensions and maps the 2D KITTI bounding box to a normalized YOLO row (`class_id center_x center_y width height`). `palletjack` is class `0`. The current KITTI maximum-edge interpretation is an inclusive-pixel working assumption; see [data preparation notes](docs/data.md) for the validation rules and this unresolved convention.

Experiment 1's split is randomized reproducibly within each scenario. Since these frames represent independent scenes, it uses an 80/10/10 allocation per scenario:

| Scenario | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `distractors_additional` | 1,600 | 200 | 200 |
| `distractors_warehouse` | 1,600 | 200 | 200 |
| `no_distractors` | 800 | 100 | 100 |
| **Total** | **4,000** | **500** | **500** |

## Get started

### Requirements

- Git and Python 3.10 or newer.
- The raw Isaac Sim export in `data/raw/experiment-01/` to run the Experiment 1 pipeline.
- A PyTorch build matching your hardware. For CUDA training, use the current command from the [official PyTorch install selector](https://pytorch.org/get-started/locally/); CUDA-enabled PyTorch bundles the CUDA runtime components it needs, while the installed NVIDIA driver must support it.
- Project packages listed in `requirements.txt` (Pillow, PyYAML, Ultralytics, Matplotlib, JupyterLab, and IPython kernel support).

From Git Bash on Windows, create and activate the virtual environment, install the hardware-matched PyTorch build, then install project requirements:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
# Install the correct PyTorch build for your CPU/GPU using pytorch.org/get-started/locally
python -m pip install -r requirements.txt
```

Check that the active interpreter is the virtual environment's Python with `which python` and `python --version`. See [training setup](docs/training.md) for GPU notes, dependency details, preflight, and run configuration.

### 1. Validate the raw export

The validator checks the raw RGB/annotation pairs, image readability, class names, annotation fields, and 2D box geometry. It is read-only; empty annotations are allowed as negative frames.

```bash
python scripts/validate_dataset.py --root data/raw/experiment-01 \
  --scenario distractors_additional \
  --scenario distractors_warehouse \
  --scenario no_distractors
```

The current export passed the structural and box-geometry checks with no issues. This does not verify that each labeled box visually contains a pallet jack; see the [data workflow](docs/data.md) for the validator's scope and the current annotation-quality caveat.

### 2. Create split manifests

This step assigns every raw frame to exactly one split and writes CSV manifests. It does not copy image files or convert labels.

```bash
python scripts/create_splits.py \
  --scenario distractors_additional \
  --scenario distractors_warehouse \
  --scenario no_distractors
```

Defaults are 80/10/10 and seed 42. The scenario options select the Isaac Sim folders and keep `real-pallet-jack-data` out of the synthetic split. Existing manifests are not silently replaced; pass `--overwrite` when intentionally regenerating them.

### 3. Process raw data into the YOLO dataset

Run this after creating the manifests. It copies only the selected RGB images and KITTI labels, converts those copied labels to YOLO format, and writes `data/processed/experiment-01/dataset.yaml`.

```bash
python scripts/process_raw_data.py
```

The processing step preserves raw inputs and uses the scenario names recorded in the split manifests, so the LOCO folder is not treated as an Isaac Sim scenario. Experiment 1 outputs are under `data/processed/experiment-01/`: original KITTI labels in `kitti_labels/`, images in `images/`, and YOLO annotations in `labels/`. For a deliberate rerun over generated outputs, use `--overwrite`. See [data workflow](docs/data.md) for the full layout and options.

### Use Experiment 2 data

Experiment 2 (`EXP-002`) adds two locally supplied pallet-jack USDZ assets to the Isaac Sim scene while retaining NVIDIA's original pallet-jack asset. Its raw exports are stored under `data/raw/experiment-02/`, leaving Experiment 1 untouched. The current exports were cleaned, validated, split, processed, used for two training runs, and evaluated on the held-out test split; see the [Experiment 2 record](docs/experiments/experiment-02.md) for exact counts and results. The commands below reproduce the data preparation after future data regeneration. Do not use `--overwrite` unless you intend to replace existing generated manifests or processed outputs.

```bash
python scripts/validate_dataset.py --root data/raw/experiment-02 \
  --scenario distractors_additional \
  --scenario distractors_warehouse \
  --scenario no_distractors
python scripts/create_splits.py \
  --root data/raw/experiment-02 \
  --output-dir data/processed/experiment-02/splits \
  --processed-root data/processed/experiment-02 \
  --scenario distractors_additional \
  --scenario distractors_warehouse \
  --scenario no_distractors
python scripts/process_raw_data.py \
  --root data/raw/experiment-02 \
  --manifest-dir data/processed/experiment-02/splits \
  --output-dir data/processed/experiment-02
```

The prepared dataset passed preflight for both training configs. `configs/train_yolo_e2_pretrained.yaml` initializes from `yolo26n.pt`; `configs/train_yolo_e2_exp01-best.yaml` initializes from Experiment 1's `best.pt`. Both runs and their results are recorded in the [Experiment 2 record](docs/experiments/experiment-02.md).

If you also want a separate Experiment 2 real-image evaluation, reuse the LOCO raw source from Experiment 1 and write its processed adapter into Experiment 2:

```bash
python scripts/process_loco_coco.py \
  --root data/raw/experiment-01/real-pallet-jack-data \
  --output-dir data/processed/experiment-02/real-pallet-jack-data
python scripts/convert_coco_to_yolo.py \
  --dataset-root data/processed/experiment-02/real-pallet-jack-data
python scripts/evaluate_test_set.py \
  --model runs/detect/experiment-02/palletjack-detection-yolo26n-seed42/weights/best.pt \
  --dataset data/processed/experiment-02/real-pallet-jack-data/dataset.yaml \
  --project runs/detect/experiment-02 \
  --name palletjack-detection-yolo26n-seed42-real-loco-subset1
```

### Explore data preparation interactively

Open [`notebooks/data_processing.ipynb`](notebooks/data_processing.ipynb) for a guided, visual walkthrough of the synthetic KITTI-to-YOLO pipeline and the real LOCO COCO filtering and COCO-to-YOLO evaluation adapter. The notebook’s write steps are off by default; turn them on only when you intend to generate or replace processed data.

### 4. Preflight and train

Edit `configs/train_yolo.yaml` for the initial checkpoint, dataset YAML, epochs, input size, batch size, device, seed, workers, and run name. The starter values are 100 epochs, 640 input size, batch 16, seed 42, and automatic device selection.

First validate the configuration without loading a model or starting training:

```bash
python scripts/train_yolo.py --config configs/train_yolo.yaml --validate-only
```

Then launch training:

```bash
python scripts/train_yolo.py --config configs/train_yolo.yaml
```

Ultralytics fine-tunes the configured pretrained detector on `images/train`, monitors `images/val`, and writes checkpoints, `results.csv`, plots, and run metadata under `runs/detect/<name>/`. It selects `best.pt` from validation performance and also saves `last.pt`. Use a new run name for each experiment; existing output directories are protected against accidental overwrite.

### 5. Explore training results

The interactive notebook reads a selected run from `runs/detect/`, plots training losses and validation metrics, and displays the PR curve, confusion matrices, and validation examples:

```bash
python -m ipykernel install --user --name palletjack-detection --display-name "Python (Pallet Jack Detection)"
jupyter lab notebooks/explore_training_results.ipynb
```

Choose the `Python (Pallet Jack Detection)` kernel in Jupyter if needed. Update `RUN_NAME` in the first code cell to inspect another experiment. Optional real-photo inference and training-launch cells are off by default and require explicit opt-in. More details are in [training setup](docs/training.md).

### 6. Evaluate the held-out synthetic test split

For Experiment 1, this was run once the checkpoint was selected. The script uses `split: test` from the dataset YAML, records the metrics as JSON, saves plots and sample predictions, and protects existing run folders from overwrite. Use a new `--name` for a distinct evaluation run.

```bash
python scripts/evaluate_test_set.py --device 0 --name palletjack-detection-yolo26n-seed42-test-reproduction
```

Omit `--device 0` for automatic device selection, or use `--device cpu`. Results are written under `runs/detect/<name>/`. See [evaluation notes](docs/evaluation.md) for the measured baseline and interpretation.

### 7. Convert LOCO COCO annotations and evaluate the real subset

The filtered COCO annotations remain the source of truth. This adapter writes normalized YOLO label files for each COCO-listed image, including empty files for negatives, and creates an Ultralytics dataset YAML. It does not alter the COCO JSON or images.

```bash
python scripts/convert_coco_to_yolo.py
python scripts/evaluate_test_set.py \
  --model runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt \
  --dataset data/processed/experiment-01/real-pallet-jack-data/dataset.yaml \
  --name palletjack-detection-yolo26n-seed42-real-loco-subset1 \
  --device 0
```

The converter refuses to replace existing generated outputs unless `--overwrite` is supplied. The evaluation run writes metrics and Ultralytics plots under `runs/detect/palletjack-detection-yolo26n-seed42-real-loco-subset1/`. See [evaluation notes](docs/evaluation.md) for the measured results and interpretation.

### 8. Visualize predictions on images

Use this command on a single image or a folder. The source images remain unchanged; annotated copies go to the output folder. This is visual inspection only and does not compute accuracy for unlabeled photos.

```bash
python scripts/predict_real_images.py \
  --source local/real-images \
  --model runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt \
  --output runs/predictions/real-photos
```

Add photos to the local `local/real-images/` folder. It is gitignored. For confidence, input-size, device, and output behavior, see [evaluation notes](docs/evaluation.md).

## Model evaluation: what is and is not measured

Training curves and the first set of figures come from validation, which is used during training and checkpoint selection. The second set of figures and test metrics come from the held-out synthetic test split. LOCO real-image metrics are now available using COCO ground truth converted to YOLO label files for the same Ultralytics evaluator. The two dataset results are listed separately because the LOCO subset has different provenance, domain, and annotation review status. Isaac Sim integration and simulator-side performance checks remain future work.

See [evaluation notes](docs/evaluation.md) and [Isaac Sim integration notes](docs/isaac-sim-integration.md) for the planned work and reporting guidance.

## Repository layout and what belongs in Git

```text
configs/                    Training and dataset configuration
data/raw/experiment-01/     Experiment 1 source data (ignored)
data/raw/experiment-02/     Experiment 2 source data (ignored)
data/processed/experiment-01/  Experiment 1 splits and prepared datasets (ignored)
data/processed/experiment-02/  Experiment 2 splits and prepared datasets (ignored)
docs/                       Workflow guides and small demonstration figures/video
local/                      Machine-local inputs and scratch data (ignored)
models/                     Model weights (ignored)
notebooks/                  Interactive data preparation and training-results exploration
runs/                       Training and prediction outputs (ignored)
scripts/                    Command-line entry points
simulation/                 Isaac Sim synthetic-data resources and scripts
src/palletjack_detection/   Reusable validation, processing, training, and inference code
tests/                      Automated regression tests
```

Commit source code, configuration templates, documentation, and selected small demonstration/evaluation media in `docs/assets/`. Do not commit the raw/processed datasets, training runs, prediction outputs, local images, virtual environment, or model checkpoints; these are excluded by `.gitignore`. The SDG video is about 42 MB; include it only if its size and the repository's media-storage policy are acceptable. The checked-in images are curated examples and evaluation snapshots, not the full generated dataset or run artifacts.

Run the repository test suite from the project root with:

```bash
python -m unittest discover -s tests -v
```

## Detailed documentation

- [Experiment 1 record: data, checkpoint, metrics, and figures](docs/experiments/experiment-01.md)
- [Experiment 2 record: changed assets, cleaned data, split, training, and test results](docs/experiments/experiment-02.md)

- [Data preparation and dataset details](docs/data.md)
- [Training configuration and Jupyter notebook](docs/training.md)
- [Prediction visualization and evaluation results](docs/evaluation.md)
- [Isaac Sim integration notes](docs/isaac-sim-integration.md)
