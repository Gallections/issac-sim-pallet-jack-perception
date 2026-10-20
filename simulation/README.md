# Isaac Sim Synthetic Data Generation

This directory is reserved for the Isaac Sim project resources and scripts used to generate synthetic pallet-jack perception data. The current repository has the folder layout documented below, but no simulation scripts or configs have been added yet.

## Scope

- Build or configure the simulation scenes and assets used for data generation.
- Run synthetic data generation (SDG) workflows and record their settings.
- Export images and annotations for the downstream detection pipeline.

## Contents

- `scripts/` — Isaac Sim scripts for scene setup, SDG, and export.
- `configs/` — simulation and SDG settings that are safe to version.
- `assets/` — small project-owned assets or references. Do not commit large or restricted assets without checking their license and size.

## Runtime

Before adding or running scripts, document the Isaac Sim version, extensions, launch method, and required environment here. Isaac Sim scripts may use its bundled Python runtime and may not run in the project's regular training environment.

## Data handoff

Export generated images and annotations to the local `data/raw/` directory (or another documented location). Generated datasets are excluded from Git; keep the export layout and annotation schema documented in `docs/data.md`.
