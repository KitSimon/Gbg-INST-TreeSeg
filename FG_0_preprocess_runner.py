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
FG_0 — build Cellpose fine-tuning data from one or more orthophoto + crowns
GeoPackage pairs.

Edit TRAINING_SOURCES (and the global tiling knobs) and run:

    python FG_0_preprocess_runner.py
"""

import os
import sys

# Add the parent of the gbg_inst_treeseg package to sys.path so `project_paths`
# and `utils` resolve no matter which directory the script is launched from.
_THIS = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import project_paths as paths
from utils.config import TrainConfig, TrainSource
from utils.preprocess_train import build_training_tiles


# =============================================================================
# Per-source inputs — one TrainSource per ortho/crowns pair
# =============================================================================
#
# Each TrainSource defines:
#   ortho_path        Source orthophoto (GeoTIFF or VRT). RGB-IR or RGB.
#   crowns_gpkg       GeoPackage with one polygon per tree crown.
#   name              (optional) Tile-filename prefix and per-source counts
#                     key. Falls back to the ortho's basename stem. MUST be
#                     unique across sources — duplicate names raise an error
#                     because output tiles would otherwise collide in train/.
#   crowns_id_field   (optional) Int attribute used as the instance ID.
#   qc_field          (optional) GeoPackage attribute encoding QC status.
#                     Tiles intersecting any 'failed' crown are routed to
#                     review/ (or skipped silently if QC_REVIEW_SUBDIR=None).
#   qc_pass_values    (optional) Values that count as 'passed'. None = any
#                     non-null value passes.
#   bbox_filter       (optional) (xmin, ymin, xmax, ymax) in raster CRS to
#                     restrict the tiled AOI for this source. None = full
#                     extent.
#   val_bbox_filter   (optional) Either:
#                       - (xmin, ymin, xmax, ymax) tuple, OR
#                       - path to a GeoPackage whose features (unioned)
#                         define the val AOI for this source.
#                     Tile-membership is centre-in-region for both forms.
#                     None = no val tiles for this source.
# =============================================================================

# The per-tree crown polygons digitised for AF_0 in the Gbg-SEM-TreeSeg repo
# (paths.GBG_SEM_ROOT) double as Cellpose training labels here.

TRAINING_SOURCES = [
    TrainSource(
        ortho_path=os.path.join(paths.FG_0_TRAINING_DIR, "6398_149.tif"),
        crowns_gpkg=os.path.join(paths.GBG_SEM_ROOT, "Project_1/AF_0_training/hull30.gpkg"),
        name="6398_149",
        crowns_id_field=None,
        qc_field="last_modified",
        qc_pass_values=None,
        bbox_filter=None,
        val_bbox_filter=os.path.join(paths.FG_0_TRAINING_DIR, "cellpose_val.gpkg"),
        # Examples for val_bbox_filter:
        #   (101000.0, 6395000.0, 102000.0, 6396000.0)        # raster-CRS tuple
        #   os.path.join(paths.FG_0_TRAINING_DIR, "val_aoi_6398_149.gpkg")
    ),
    # Add more sources by uncommenting and editing.
    # NOTE: each TrainSource is independent
    # — fields you omit fall back to the dataclass defaults
    # (qc_field=None, bbox_filter=None, etc.), they do NOT inherit from the
    # source above. Spell out every field you want set on this source.
    #
    # TrainSource(
    #     ortho_path=os.path.join(paths.FG_0_TRAINING_DIR, "6398_150.tif"),
    #     crowns_gpkg=os.path.join(paths.FG_0_TRAINING_DIR, "hull_6398_150.gpkg"),
    #     name="6398_150",
    #     crowns_id_field=None,
    #     qc_field="last_modified",     # set explicitly per source
    #     qc_pass_values=None,
    #     bbox_filter=None,
    #     val_bbox_filter=os.path.join(paths.FG_0_TRAINING_DIR, "val_aoi_6398_150.gpkg"),
    # ),
]

# =============================================================================
# Shared / global settings
# =============================================================================

OUTPUT_DIR = paths.FG_0_TRAINING_DATA_DIR
"""Directory where train/, val/, and (optionally) review/ subdirs will be
created. Tiles from every TRAINING_SOURCE land in the same flat directory."""

QC_REVIEW_SUBDIR = "review"
"""Sub-directory under OUTPUT_DIR where QC-flagged tiles are written.
None = drop them silently."""

# --- Bands -------------------------------------------------------------------

BAND_INDICES = [1, 2, 3]
"""1-based indices into the source bands. Length 1, 2, or 3 — flexible.
Cellpose-SAM consumes a 3-channel image, so any 3-index combo gives the
encoder full information; 1- or 2-index combos are replicated to fill the
unused channels.

Must match what you plan to pass to FG_2 at inference time.
Examples for an RGB-IR ortho (R=1, G=2, B=3, IR=4):
    [1, 2, 3]  — true colour RGB (default)
    [4, 1, 2]  — NRG false-colour: NIR, R, G — standard remote-sensing
                 vegetation composite (NIR-vs-R contrast)
    [4, 1, 3]  — NIR, R, B — similar, keeps blue
    [2, 1]     — paper-style G, R (third channel is a duplicate of R)
    [4]        — IR-only, replicated to all 3 channels"""

# --- Tiling ------------------------------------------------------------------

TILE_SIZE = 512
TILE_STRIDE = 384  # tile_size - tile_stride = overlap between adjacent training tiles

MIN_TILE_INSTANCES = 1


# =============================================================================
# Run
# =============================================================================

def main():
    cfg = TrainConfig(
        sources=TRAINING_SOURCES,
        qc_review_subdir=QC_REVIEW_SUBDIR,
        tile_size=TILE_SIZE,
        tile_stride=TILE_STRIDE,
        min_tile_instances=MIN_TILE_INSTANCES,
        band_indices=BAND_INDICES,
        output_dir=OUTPUT_DIR,
    )
    counts = build_training_tiles(cfg)
    print(counts)


if __name__ == "__main__":
    main()
