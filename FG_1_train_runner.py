#!/usr/bin/env python
# Gbg-INST-TreeSeg — tree-crown instance segmentation pipeline.
# Copyright (C) 2026 KitSimon
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

"""
FG_1 — fine-tune Cellpose-SAM on the tiles produced by FG_0.

Edit the constants below and run:

    python FG_1_train_runner.py
"""

import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import project_paths as paths
from utils.config import PhotometricAugConfig, TrainConfig
from utils.train_cellpose import train


# =============================================================================
# Edit these for your run
# =============================================================================

# Must match the FG_0 run that produced the training data
TRAIN_DATA_DIR = paths.FG_0_TRAINING_DATA_DIR

# Where checkpoints, flow_cache/, and aug_preview/ are written. Kept separate
# from TRAIN_DATA_DIR, mirroring Gbg-SEM-TreeSeg's AF_1_training_data /
# AF_2_training_runs split.
ARTIFACTS_DIR = paths.FG_1_TRAINING_RUNS_DIR

BAND_INDICES = [1, 2, 3]
# Same band selection used in FG_0; flexible (length 1, 2, or 3). See
# FG_0_preprocess_runner.py for the full menu of suggested combos.
#
# Source identities (ortho path, crowns gpkg, name, QC settings, val region)
# are loaded from <TRAIN_DATA_DIR>/sources_manifest.json (written by FG_0)
# and copied into the training info JSON for traceability — train() itself
# only reads the prebuilt train/ and val/ directories.

# --- Hyperparameters ---------------------------------------------------------

PRETRAINED_MODEL = "cpsam"  # Cellpose-SAM ViT-H starting point
N_EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
SAVE_EVERY = 50

MODEL_NAME = "cellpose_treecrown"

# --- Training monitoring ----------------------------------------------------

MONITOR_INTERVAL_EPOCHS = 10
"""train_seg is called in chunks of this many epochs. After each chunk the
loss CSV (<MODEL_NAME>_losses.csv) is appended to and the loss PNG
(<MODEL_NAME>_losses.png) is regenerated. Smaller = more frequent updates;
default 10 is a sensible compromise."""

EARLY_STOPPING_PATIENCE = None
"""Number of consecutive monitor intervals where val loss must fail to
improve on the best-so-far before training is stopped. None = disabled
(let the run reach N_EPOCHS). Typical value if you want it: 3."""

OVERFIT_WARN_WINDOW = 3
"""Window (in monitor intervals) for the overfitting warning heuristic:
if train loss has fallen monotonically and val loss has risen monotonically
across the last N intervals, a warning is logged (training continues
unless EARLY_STOPPING_PATIENCE also fires)."""

USE_BFLOAT16 = False
"""Load Cellpose-SAM weights in bfloat16. Torch <2.1 does not implement
F.interpolate(mode='linear') for bfloat16, which crashes inside SAM's
relative-positional-embedding code. Default False for torch 2.0.x
compatibility; flip to True on a newer env (torch >=2.1) for ~50% VRAM
savings. Keep consistent with USE_BFLOAT16 in FG_2_inference_runner.py."""

# --- Photometric augmentation -----------------------------------------------
#
# Per-chunk fresh re-augmentation: the same source tile gets a different
# perturbation for every monitor-interval chunk. Cellpose handles geometric
# augs internally (full rotation, h/v flips, scale), so this menu is
# photometric-only and skewed toward effects that survive Cellpose's
# per-tile percentile renormalisation: shadow polygons, gamma, CLAHE,
# HSV jitter, additive noise. Tune any knob below; set enabled=False to
# disable augmentation entirely. See PhotometricAugConfig in config.py
# for the full menu.
#
# Defaults are balanced. For a more shadow-focused recipe, bump:
#   random_shadow_p       -> 0.7
#   random_gamma_limit    -> (50, 140)
#   random_shadow_count   -> (2, 4)
#
# dump_examples_n writes that many augmented (image, mask) preview pairs
# to <ARTIFACTS_DIR>/aug_preview/ on the first chunk so you can
# sanity-check the recipe in QGIS before a long run.

PHOTOMETRIC_AUG = PhotometricAugConfig(
    enabled=True,
    dump_examples_n=4,
)


# =============================================================================
# Run
# =============================================================================

def _load_sources_manifest(train_data_dir):
    manifest_path = os.path.join(train_data_dir, "sources_manifest.json")
    if not os.path.isfile(manifest_path):
        print(
            f"[Gbg-INST-TreeSeg] WARNING: no sources_manifest.json in "
            f"{train_data_dir}; training info JSON will record an empty "
            f"sources list. (Re-run FG_0 to regenerate the manifest.)"
        )
        return []
    with open(manifest_path) as f:
        return json.load(f).get("sources", [])


def main():
    cfg = TrainConfig(
        sources=[],  # not needed at training time; manifest carries traceability
        band_indices=BAND_INDICES,
        output_dir=TRAIN_DATA_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        pretrained_model=PRETRAINED_MODEL,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_every=SAVE_EVERY,
        model_name=MODEL_NAME,
        monitor_interval_epochs=MONITOR_INTERVAL_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        overfit_warn_window=OVERFIT_WARN_WINDOW,
        use_bfloat16=USE_BFLOAT16,
        photometric_aug=PHOTOMETRIC_AUG,
    )
    cfg.sources_manifest = _load_sources_manifest(TRAIN_DATA_DIR)
    info = train(cfg)
    print(f"Final checkpoint: {info['checkpoint']}")
    print("Use this path as CELLPOSE_CHECKPOINT in FG_2_inference_runner.py.")


if __name__ == "__main__":
    main()
