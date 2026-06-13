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
Single source of truth for the project directory layout.

Every runner (and the tools/ scripts) imports its directories from here, so
switching to a new project is a one-line edit of PROJECT_NAME (e.g.
"Project_2"). Machine-specific external roots used by more than one script
live here too; run-specific values (timestamped stems, hyperparameters)
stay in the individual runners.

All paths are anchored to this file's location, so the scripts work no
matter which directory they are invoked from.

The Project_1/FG_* layout mirrors the Project_1/AF_* convention in the
companion Gbg-SEM-TreeSeg repo (https://github.com/KitSimon/Gbg-SEM-TreeSeg).
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Project folder -----------------------------------------------------------

PROJECT_NAME = "Project_1"
"""Name of the active project folder under the repo root. Change this (and
nothing else) to point the whole pipeline at a fresh project."""

PROJECT_DIR = os.path.join(REPO_ROOT, PROJECT_NAME)

# --- Per-stage subdirectories --------------------------------------------------

FG_0_TRAINING_DIR = os.path.join(PROJECT_DIR, "FG_0_training")
"""Stage-0 inputs you provide: training ortho tile(s), crowns gpkg, val AOI."""

FG_0_TRAINING_DATA_DIR = os.path.join(PROJECT_DIR, "FG_0_training_data")
"""FG_0 output / FG_1 input: train/ val/ review/ tiles + sources_manifest.json."""

FG_1_TRAINING_RUNS_DIR = os.path.join(PROJECT_DIR, "FG_1_training_runs")
"""FG_1 output: checkpoints/ flow_cache/ aug_preview/."""

FG_2_INFERENCE_RESULTS_DIR = os.path.join(PROJECT_DIR, "FG_2_inference_results")
"""FG_2 output: crowns / tile_grid / dedup_culled gpkgs, feature shards,
_test_subset/."""

FG_2_QA_DIR = os.path.join(FG_2_INFERENCE_RESULTS_DIR, "_qa")
"""tools/FG_2b_qa_preview.py output: preview PNGs of the FG_2 crowns."""

FG_3_LAS_FILTERED_DIR = os.path.join(PROJECT_DIR, "FG_3_las_filtered")
"""FG_3 output: height-filtered crowns gpkg + _las_streaming_work/ scratch."""

# --- Machine-specific external inputs (outside this repo) ----------------------
#
# These absolute paths differ per machine and live in local_paths.py, which is
# gitignored. Copy local_paths.example.py to local_paths.py and set them there;
# they are re-exported here so the rest of the pipeline keeps using paths.* .
#   ORTHO_PATH    source orthophoto mosaic (VRT/GeoTIFF). Used by FG_2/2a/2b.
#   GBG_SEM_ROOT  Gbg-SEM-TreeSeg working copy (AF_0 crowns, AF_3 raster).
#   LAS_DIR       LAS tiles, one per ortho tile. Used by FG_3.
#   TIF_DIR       ortho GeoTIFF tiles paired with the LAS tiles. Used by FG_3.

try:
    from local_paths import ORTHO_PATH, GBG_SEM_ROOT, LAS_DIR, TIF_DIR
except ImportError as e:
    raise SystemExit(
        "Missing local_paths.py — copy local_paths.example.py to local_paths.py "
        "and set your machine-specific paths."
    ) from e
