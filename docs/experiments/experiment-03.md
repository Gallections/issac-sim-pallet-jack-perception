# Experiment 3 — Current target scope and hard negatives

**Status:** raw collection complete and invalid black frames removed. Raw KITTI validation found one zero-width box that must be resolved before data splitting and processing. No Experiment 3 split, training, or evaluation has been run.

## Objective and label policy

Experiment 3 updates the synthetic data to match the current detection target and adds target-free examples intended to teach the detector to reject confusing warehouse equipment.

- **Positive class (`palletjack`):** manual pallet jacks and the powered electric pallet truck represented by `electric_pallet_jack.usdz`.
- **Out of scope:** forklift stackers, including `pallet_jack_advanced.usdz`, and other warehouse equipment. These are distractors and do not receive pallet-jack boxes.
- **Hard negatives:** scenes with the combined warehouse and additional distractor pools, including the stacker and local props, but without a pallet-truck target. Their object-detection labels are empty.

The local distractor assets are `pallet_jack_advanced.usdz`, `forklift.usdz`, `ladder.usdz`, `shopping_cart.usdz`, and `warehouse_rack.usdz`. The electric pallet-truck asset is stored under `simulation/assets/pallet_jacks/`; the distractors are stored under `simulation/assets/`. The USDZ files are machine-local and ignored by Git, so they must be supplied separately to reproduce generation.

This policy differs from Experiment 2, where the powered electric pallet truck and forklift stacker were both labeled positive. Experiment 2 remains a historical result under its original labels; its annotations are not silently rewritten here.

## Scene generation

The generator uses the Isaac Sim Simple Warehouse scene and captures independently randomized 960×544 RGB frames with KITTI-style object-detection labels. It varies camera pose and scene-object pose per capture. The three target-positive conditions follow the same nominal frame counts as earlier collection:

| Raw group | Nominal frames | Scene contents |
|---|---:|---|
| `distractors_warehouse` | 2,000 | Pallet-truck targets with warehouse distractors |
| `distractors_additional` | 2,000 | Pallet-truck targets with additional distractor assets |
| `no_distractors` | 1,000 | Pallet-truck targets without an added distractor group |
| `hard_negatives` | 1,000 | Combined distractor pool, no pallet-truck targets, empty object labels |
| **Total generated** | **6,000** | **5,000 target-positive and 1,000 target-free frames** |

Run `simulation/scripts/generate_data.bat` on Windows or `generate_data.sh` in an Isaac Sim shell to generate all four groups under `data/raw/experiment-03/`. When only the target-free group needs to be regenerated, use `generate_hard_negatives.bat` or `generate_hard_negatives.sh`; it writes only to `data/raw/experiment-03/hard_negatives/`. The launchers and simulator settings are documented in the [simulation guide](../../simulation/README.md).

## Raw image cleanup and validation

After generation, the RGB folders were scanned for wholly transparent-black or all-black frames. Each invalid RGB image and the object-detection KITTI file with the same frame ID were removed together. The cleanup removed 238 pairs: 63 from `distractors_additional`, 52 from `distractors_warehouse`, 92 from `hard_negatives`, and 31 from `no_distractors`. Of these, 237 were fully transparent-black frames; one hard-negative frame was entirely black with an opaque alpha channel and was removed as an invalid black capture as well.

The current retained inventory is:

| Raw group | RGB images | KITTI files | Valid 2D objects | Empty labels |
|---|---:|---:|---:|---:|
| `distractors_additional` | 1,937 | 1,937 | 4,918 | 285 |
| `distractors_warehouse` | 1,948 | 1,948 | 5,829 | 230 |
| `hard_negatives` | 908 | 908 | 0 | 908 |
| `no_distractors` | 969 | 969 | 3,869 | 59 |
| **Total** | **5,762** | **5,762** | **14,616** | **1,482** |

The image and label stems match in each group. Validation was run with:

```powershell
python scripts/validate_dataset.py --root data/raw/experiment-03 `
  --scenario distractors_additional `
  --scenario distractors_warehouse `
  --scenario hard_negatives `
  --scenario no_distractors
```

The validator found one remaining malformed box in `distractors_warehouse/Camera/object_detection/1727.txt`: line 3 has `x_min == x_max == 0`, so it has zero width. The other labels passed the validator's structural and bounds checks. Resolve this row (by visually reviewing and correcting it, or removing the invalid annotation if it is not a valid target) and rerun validation before creating split manifests. The raw collection has not yet been visually audited end-to-end.

## Current preparation status and next steps

Experiment 3 is raw-only at this point. The data have not been split, copied into `data/processed/experiment-03/`, converted to YOLO labels, used for training, or evaluated. Experiment 1 and Experiment 2 processed datasets and checkpoints remain unchanged.

After the zero-width annotation is resolved and validation passes:

1. Review representative images and labels from each positive condition and the hard-negative group.
2. Create an experiment-specific seeded 80/10/10 split, keeping each generation condition represented in all splits.
3. Process the selected RGB/KITTI pairs into `data/processed/experiment-03/` and convert positive boxes to YOLO format. Empty hard-negative labels must remain empty YOLO label files paired with their images.
4. Train and evaluate a new model on the Experiment 3 split; compare against earlier models with the dataset and label-policy differences made explicit.

Do not treat the current 5,762-image raw inventory as training-ready until the outstanding annotation and visual-review steps are complete.
