# Isaac Sim Synthetic Data Generation

This project generates pallet-jack detection data in NVIDIA Isaac Sim with Replicator. It starts from the Simple Warehouse environment, randomizes the scene for each render, and uses semantic labels to export KITTI-style object-detection annotations alongside rendered images.

## Generator

The main script is [`scripts/palletjack_sdg/standalone_palletjack_sdg.py`](scripts/palletjack_sdg/standalone_palletjack_sdg.py). It must run with Isaac Sim's Python runtime because it imports Kit, Omniverse, USD, and Replicator modules; the model-training virtual environment is not a substitute.

The script accepts these main options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--height` | `544` | Render height in pixels |
| `--width` | `960` | Render width in pixels |
| `--num_frames` | `1000` | Number of independently randomized frames to render |
| `--distractors` | `warehouse` | `warehouse`, `additional`, or `None` |
| `--distractors-only` | `False` | Disable pallet-jack targets and capture a target-free scene with the full distractor pool |
| `--data_dir` | `<current working directory>/_palletjack_data` | Export destination |
| `--headless` | `False` | Launch without a visible simulator UI |

The three Experiment 1 source collections were generated at 960x544 and are retained under `data/raw/experiment-01/`: 2,000 warehouse-distractor frames, 2,000 additional-distractor frames, and 1,000 frames without an added distractor group. “No distractors” still uses the warehouse environment; it omits the extra randomized distractor objects.

For each ordinary target-positive frame, Replicator samples a camera pose, pallet-jack placement and rotation, lighting values, and floor/wall material properties. The target assets are tagged with the semantic class `palletjack`, which the KITTI writer uses to produce annotations. The resulting frames are independent randomized scene states, not consecutive samples from a continuous camera trajectory.

Experiment 2 (`EXP-002`) also loaded two local USDZ files: `electric_pallet_jack.usdz` from `simulation/assets/pallet_jacks/` and `pallet_jack_advanced.usdz` from `simulation/assets/`. Their appearance alternated between imported and randomized materials at a fixed `1.9x` scale. A later review identified these as a powered electric pallet truck and a forklift stacker. Under the current target, manual and electric pallet trucks are positive `palletjack` instances; the forklift stacker is an unlabeled distractor. Experiment 2 labeled both local assets positive, so its stacker annotations do not match the current policy. Keep its results as historical results under that earlier label policy.

Four additional local distractors are stored directly in `simulation/assets/`: `forklift.usdz`, `ladder.usdz`, `shopping_cart.usdz`, and `warehouse_rack.usdz`. They, along with the stacker, are included in both the regular `warehouse` and `additional` distractor groups. They also enter the combined hard-negative scene pool. These assets have no `palletjack` semantic label, so they should never generate positive boxes. Local USDZ assets are machine-local and ignored by Git; provide them separately when reproducing simulation generation.

### Experiment 3 data collection

The revised target includes manual and electric pallet trucks; forklift stackers and the other local props are distractors. Run `simulation/scripts/generate_data.bat` on Windows or `generate_data.sh` from a shell configured with Isaac Sim to produce the complete collection in one run. The launcher generates 2,000 warehouse-distractor, 2,000 additional-distractor, and 1,000 no-added-distractor target-positive frames, followed by 1,000 target-free hard negatives under `data/raw/experiment-03/hard_negatives/`.

The hard-negative portion loads the combined warehouse and additional distractor pools once, including the four local USDZ props and the forklift stacker. As with the other conditions, it randomizes the distractors' positions, orientations, and scale for each captured frame, and randomizes the camera pose. It does not add pallet-truck targets. Distractor semantics are cleared, so the intended KITTI object labels are empty. Each captured frame is therefore a negative example for the detector.

The launcher creates its Experiment 3 output directories and leaves Experiments 1 and 2 untouched. Its `EXPERIMENT_ID` setting controls the folder under `data/raw/`; change this value for a separate future collection. The electric pallet-truck USDZ must be present under `simulation/assets/pallet_jacks/`; the stacker and four other local distractors must be present directly under `simulation/assets/`. The electric pallet truck belongs in positive training scenes and is not included in the hard-negative collection. Experiment 3 has been generated and curated: 238 invalid black RGB frames and their matching KITTI files were removed, leaving 5,762 pairs. The hard-negative group has 908 empty annotation files. Raw validation found one zero-width box in `distractors_warehouse/Camera/object_detection/1727.txt`; resolve that issue before processing. No split, YOLO conversion, training, or evaluation has been done for Experiment 3.

If the three positive Experiment 3 groups are already collected and only the hard negatives need another run, use `simulation/scripts/generate_hard_negatives.bat` on Windows or `generate_hard_negatives.sh` in a shell configured with Isaac Sim. This focused launcher writes 1,000 frames only to `data/raw/experiment-03/hard_negatives/`; it does not invoke the other three conditions.

The render setting `/omni/replicator/RTSubframes` is currently `32` to reduce intermittent transparent-black RGB frames observed during generation. More subframes increase render settling time, but do not guarantee every output is valid. Experiment 3 still required a scan and removal of 238 invalid black RGB frames with their paired KITTI labels; its per-group counts are in the [Experiment 3 record](../docs/experiments/experiment-03.md). Experiment 2's separate cleanup is recorded in the [Experiment 2 record](../docs/experiments/experiment-02.md).

The generated sample frames and walkthrough video are linked in the [main README](../README.md#how-the-synthetic-data-is-generated).

## Launch helpers and paths

`simulation/scripts/generate_data.bat` and `simulation/scripts/generate_data.sh` each launch all four Experiment 3 conditions. The focused `generate_hard_negatives` launchers rerun only the hard-negative condition. The Windows launchers resolve the repository and script paths relative to their own locations, so they can be started from the project root or another working directory. They write under `data/raw/experiment-03/`, preserving Experiments 1 and 2. Before using a launcher on another machine, set its Isaac Sim installation path and verify paths. Isaac Sim installations, asset server availability, and mounted Omniverse assets are machine-specific. The generator itself writes to the directory passed with `--data_dir`.

## Handoff to the perception pipeline

The original three raw collections are the synthetic source data for Experiment 1 (`EXP-001`); their processed train/validation/test split and model results are indexed in the [Experiment 1 record](../docs/experiments/experiment-01.md). Experiment 2 (`EXP-002`) generated a separate collection under `data/raw/experiment-02/`; its completed preparation, training, and evaluation are summarized in the [Experiment 2 record](../docs/experiments/experiment-02.md). Experiment 3 (`EXP-003`) uses the current target scope and combines three positive-scene groups with hard-negative-only scenes under `data/raw/experiment-03/`; its raw collection and outstanding validation issue are documented in the [Experiment 3 record](../docs/experiments/experiment-03.md). The real LOCO images were used only for separate external evaluation and were not part of synthetic model training.


The generator exports the source dataset; it does not perform splitting, YOLO conversion, model training, or model evaluation. Those steps are separate commands documented in the [main README](../README.md) and [data preparation guide](../docs/data.md). Raw exports should be retained so processed splits and labels can be regenerated.
