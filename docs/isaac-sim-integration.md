# Isaac Sim integration

## Current status

The project has not yet connected a trained detector to a live Isaac Sim camera. Training and the current prediction/evaluation tools run outside Isaac Sim with Ultralytics. The Experiment 1–4 checkpoints are `.pt` files, and the real-image prediction script can draw detections on saved photos. No simulator-side inference runtime, exported deployment model, or latency measurement has been established.

The training source images are 960×544; Ultralytics receives them at a configured 640-pixel input size while preserving aspect ratio. The simulator integration must verify that its camera image color order, resizing/letterboxing, normalization, and box-coordinate mapping match the behavior used by the trained model.

## Decisions and validation still needed

- Record the Isaac Sim release and target runtime environment.
- Choose and verify the inference format/runtime for that environment. The current `.pt` checkpoints are Ultralytics/PyTorch artifacts; compatibility with a particular Isaac Sim runtime has not been established. Consider export formats only after the target runtime is known.
- Define how rendered camera frames are passed to inference and how detections (class, confidence, and box coordinates) are returned to the scene or UI.
- Measure end-to-end inference latency and throughput on the target hardware, including image transfer and preprocessing.
- Validate on a controlled Isaac Sim scene containing known pallet-jack positions, scales, orientations, occlusions, and negative scenes. Compare detections to the generated annotations and inspect both misses and false detections.
- Keep simulator validation results separate from the synthetic held-out test and LOCO subset-1 diagnostic. LOCO informed Experiment 2 and is not an untouched final benchmark.

Until these decisions and checks are complete, describe Isaac Sim inference as planned work rather than a demonstrated capability. See the [project README](../README.md) and the [experiment records](experiments/experiment-01.md) and [Experiment 2 record](experiments/experiment-02.md) for completed training and evaluation results.
