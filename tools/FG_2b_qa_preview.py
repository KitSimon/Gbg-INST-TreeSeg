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
FG_2b — quick QA visualization for a FG_2 instance-segmentation result.

Renders a PNG that overlays the GeoPackage polygons on a small subset of the
ortho. Useful for a fast sanity check before opening QGIS.

Edit the constants below and run (from the repo root):

    python tools/FG_2b_qa_preview.py
"""

import os
import sys

# Repo root = parent of tools/ — needed on sys.path for `from utils import …`
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window as RioWindow
from shapely.geometry import box

import project_paths as paths
from utils.bands import select_bands


# =============================================================================
# Edit these
# =============================================================================

ORTHO_PATH = paths.ORTHO_PATH
GPKG_PATH = os.path.join(
    paths.FG_2_INFERENCE_RESULTS_DIR, "_mosaik_2026_05_13__0829_crowns.gpkg"
)
OUTPUT_PNG = os.path.join(paths.FG_2_QA_DIR, "qa_preview.png")

BAND_INDICES = [1, 2, 3]  # which bands to render in the preview (RGB)

# Window in pixel coordinates of the ortho
WINDOW = (0, 0, 2048, 2048)  # (x, y, w, h)


# =============================================================================
# Run
# =============================================================================

def main():
    x, y, w, h = WINDOW
    with rasterio.open(ORTHO_PATH) as ortho:
        win = RioWindow(x, y, w, h)
        img = ortho.read(window=win)
        win_transform = rasterio.windows.transform(win, ortho.transform)
        bounds = rasterio.windows.bounds(win, ortho.transform)
        ortho_crs = ortho.crs

    rgb = select_bands(img, BAND_INDICES, percentile_clip=(2.0, 98.0))

    crowns = gpd.read_file(GPKG_PATH)
    if crowns.crs is not None and ortho_crs is not None and crowns.crs != ortho_crs:
        crowns = crowns.to_crs(ortho_crs)
    crowns = crowns[crowns.intersects(box(*bounds))]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, extent=(bounds[0], bounds[2], bounds[1], bounds[3]))
    crowns.boundary.plot(ax=ax, color="cyan", linewidth=0.7)
    ax.set_title(f"{len(crowns)} crowns in window")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_dir = os.path.dirname(OUTPUT_PNG)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
