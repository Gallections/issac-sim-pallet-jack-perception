# Experiment 2 — Expanded Isaac Sim pallet-jack asset mix

## Purpose and status

**Experiment ID:** `EXP-002`  
**Status:** synthetic data generated and prepared; training and evaluation have not started.

Experiment 2 investigates whether adding larger and visually different pallet-jack assets to the synthetic scene improves the model's ability to detect the pallet-jack appearances that were weakly detected in the separate LOCO real-image diagnostic. It preserves Experiment 1 as the baseline and gives all new raw data, processed data, and future model runs an experiment-specific location.

The LOCO results motivated this change, so LOCO is useful as a diagnostic during development. It must not be described as an untouched independent test set for this design iteration. A broader independently labeled real-world holdout is still needed for stronger transfer claims.

## What changed from Experiment 1

| Area | Experiment 1 (`EXP-001`) | Experiment 2 (`EXP-002`) |
|---|---|---|
| Synthetic assets | NVIDIA Isaac Sim pallet-jack asset | NVIDIA asset plus two local USDZ variants: `electric_pallet_jack.usdz` and `pallet_jack_advanced.usdz` |
| Local model appearance | Not applicable | Original imported materials are visible on some frames; randomized-color instances are used on others |
| Local asset scale | Not applicable | Fixed `1.9x` multiplier in the generator |
| Render settling | Earlier setting | `/omni/replicator/RTSubframes` set to `10` after intermittent transparent-black images were observed |
| Data roots | `data/raw/experiment-01/`, `data/processed/experiment-01/` | `data/raw/experiment-02/`, `data/processed/experiment-02/` |
| Model results | Completed baseline | No Experiment 2 training or evaluation yet |

The local USDZ files are machine-local and gitignored. Their assets are expected at `simulation/assets/pallet_jacks/experiment-02/`; users reproducing generation must supply the files themselves. The simulation implementation and settings are described in [`simulation/README.md`](../../simulation/README.md) and the generator at [`simulation/scripts/palletjack_sdg/standalone_palletjack_sdg.py`](../../simulation/scripts/palletjack_sdg/standalone_palletjack_sdg.py).

## Raw data cleanup and validation

Isaac Sim generated three independently randomized scenario collections. Increasing RTSubframes reduced—but did not eliminate—the invalid transparent-black renders. A content scan removed each exact fully transparent black RGB frame and its paired KITTI object-detection annotation:

| Scenario | Transparent-black RGB/label pairs removed |
|---|---:|
| `distractors_additional` | 59 |
| `distractors_warehouse` | 64 |
| `no_distractors` | 53 |
| **Total** | **176** |

The cleaned collection was then checked with the KITTI object-detection validator. Three zero-width box rows were removed from their annotation files while preserving their images and all remaining valid boxes: one row in `distractors_additional/Camera/object_detection/956.txt`, and one row each in `distractors_warehouse/Camera/object_detection/221.txt` and `821.txt`. The subsequent validation passed with no issues.

| Scenario | Clean images and annotation files | Valid pallet-jack objects | Empty annotations |
|---|---:|---:|---:|
| `distractors_additional` | 1,941 | 8,740 | 129 |
| `distractors_warehouse` | 1,936 | 9,587 | 118 |
| `no_distractors` | 947 | 5,017 | 54 |
| **Total** | **4,824** | **23,344** | **301** |

Empty annotations remain valid negative frames. The raw collection is curated: it excludes the 176 unusable transparent-black frames and omits the three degenerate zero-width annotation rows. Original KITTI annotation files otherwise remain in the raw tree. As in Experiment 1, structural validation does not verify visually that every retained box encloses the intended object.

## Processed dataset and split

The same seed `42` and 80/10/10 per-scenario policy used in Experiment 1 was applied to Experiment 2. All 4,824 frames are represented once in the manifests:

| Scenario | Train | Validation | Test |
|---|---:|---:|---:|
| `distractors_additional` | 1,553 | 194 | 194 |
| `distractors_warehouse` | 1,549 | 194 | 193 |
| `no_distractors` | 757 | 95 | 95 |
| **Total** | **3,859** | **483** | **482** |

Processed images, preserved KITTI labels, YOLO labels, split manifests, and dataset YAML are under `data/processed/experiment-02/`. These generated data are local and gitignored. The preprocessing and converter behavior are documented in [`docs/data.md`](../data.md).

## Planned training comparison

Two models are planned on the same Experiment 2 split and training settings:

1. **Fresh pretrained initialization:** `yolo26n.pt`, using [`configs/train_yolo_experiment-02.yaml`](../../configs/train_yolo_experiment-02.yaml).
2. **Experiment 1 continuation:** initialize from `runs/detect/palletjack-detection-yolo26n-seed42/weights/best.pt`. This needs a separate training config and unique run name before launch.

Both planned runs use 100 epochs, 640 input size, batch size 16, seed 42, four workers, and automatic device selection unless intentionally changed and documented. The Experiment 2 config passed `--validate-only` against the prepared dataset. This confirms its dataset paths and settings; it does not load a model or start training.

After both runs, evaluate each selected checkpoint on the held-out Experiment 2 synthetic test split. Also evaluate against the Experiment 1 synthetic test split to measure regression on the earlier asset distribution. Keep all results separated by experiment and initialization. LOCO may be reported as a development diagnostic, with the caveat that its observed appearance/scale differences motivated the asset change. Do not tune on LOCO and then present it as an untouched final test.

## Results

No Experiment 2 training run, checkpoint, held-out test evaluation, or metrics have been produced. Add the run names, actual settings, test metrics, plots, qualitative findings, and any deviations here only after those steps complete. Until then, Experiment 1 remains the only completed model result; see the [Experiment 1 record](experiment-01.md).
