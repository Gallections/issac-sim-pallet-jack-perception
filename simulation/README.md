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
| `--data_dir` | `<current working directory>/_palletjack_data` | Export destination |
| `--headless` | `False` | Launch without a visible simulator UI |

The three Experiment 1 source collections were generated at 960x544 and are retained under `data/raw/experiment-01/`: 2,000 warehouse-distractor frames, 2,000 additional-distractor frames, and 1,000 frames without an added distractor group. “No distractors” still uses the warehouse environment; it omits the extra randomized distractor objects.

For each frame, Replicator samples a camera pose, pallet-jack placement and rotation, lighting values, and floor/wall material properties. The pallet-jack USD assets are tagged with the semantic class `palletjack`, which is used by the KITTI writer to produce annotations automatically. The resulting frames are independent randomized scene states, not consecutive samples from a continuous camera trajectory.

Experiment 2 (`EXP-002`) also loads the local USDZ variants from `assets/pallet_jacks/experiment-02/`: `electric_pallet_jack.usdz` and `pallet_jack_advanced.usdz`. Each local model has an original-material instance and a color-randomized instance; Replicator alternates which appearance is visible across frames so imported textures appear alongside randomized appearances. Both local variants use a fixed 1.9x scale multiplier. NVIDIA's pallet-jack assets retain their original 0.01 scale conversion and existing color randomization. The USDZ files are machine-local assets and are not committed.

The render setting `/omni/replicator/RTSubframes` is set to `10` to reduce intermittent transparent-black RGB frames observed during generation. This improves render settling at a runtime cost, but it does not guarantee every output is valid; the generated files were still scanned and invalid RGB frames were removed with their paired KITTI labels. See the [Experiment 2 record](../docs/experiments/experiment-02.md) for the exact cleanup and dataset counts.

The generated sample frames and walkthrough video are linked in the [main README](../README.md#how-the-synthetic-data-is-generated).

## Launch helpers and paths

`scripts/generate_data.bat` and `scripts/generate_data.sh` are convenience launchers for the three conditions. The Windows launcher resolves the repository and script paths relative to its own location, so it can be started from the project root or another working directory. It currently writes new data to `data/raw/experiment-02/` to preserve Experiment 1. Before using a launcher on another machine, set its Isaac Sim installation path and verify paths. Isaac Sim installations, asset server availability, and mounted Omniverse assets are machine-specific. The generator itself writes to the directory passed with `--data_dir`.

## Handoff to the perception pipeline

These three raw collections are the original synthetic source data for Experiment 1 (`EXP-001`); their processed train/validation/test split and model results are indexed in the [Experiment 1 record](../docs/experiments/experiment-01.md). Experiment 2 (`EXP-002`) generated a separate collection with the additional local assets, under `data/raw/experiment-02/`; its data preparation and next steps are summarized in the [Experiment 2 record](../docs/experiments/experiment-02.md). The real LOCO images were used only for the separate external evaluation and were not part of synthetic model training.


The generator exports the source dataset; it does not perform splitting, YOLO conversion, model training, or model evaluation. Those steps are separate commands documented in the [main README](../README.md) and [data preparation guide](../docs/data.md). Raw exports should be retained so processed splits and labels can be regenerated.
