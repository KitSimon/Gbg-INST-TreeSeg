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
FG_2a — resume FG_2's chunked postprocess stage from existing per-tile
GeoParquet shards when the main runner was OOM-killed.

HOW TO USE
----------
1. Edit the constants in the "Edit these" block below to match the interrupted
   run (the feature_dir path is the key one; everything else should match the
   FG_2_inference_runner.py that was killed).
2. Run (from the repo root):  python tools/FG_2a_postprocess_resume.py

WHAT THIS SCRIPT DOES
---------------------
Stage 1 — Chunk processing (checkpointed):
    For each of the 1435 super-tile chunks, reads the parquet shards that
    intersect the chunk, runs stitch → dedup → mosaic, and writes the result
    to a per-chunk parquet file on disk. Peak RAM = one chunk at a time.
    Already-done chunks are skipped on resume.

Stage 2 — Streaming GPKG write:
    Reads the per-chunk parquets one at a time, tags image-edge crowns,
    computes output columns, and appends to the final GPKG with pyogrio's
    append mode. Peak RAM = one chunk at a time. Never accumulates all
    crowns in memory.

WHY BOTH STAGES MATTER
----------------------
The original run_chunked accumulated every chunk result in a Python list
(kept_parts) and only pd.concat'd at the end — so with 1435 chunks and
~6M total crowns the process was killed by the Linux OOM killer at chunk
~1000. The first version of this script fixed Stage 1 but still tried to
concat all results before writing, hitting the same OOM in Stage 2.
Streaming the GPKG write eliminates the last large accumulation.
"""

import os
import sys

# Repo root = parent of tools/ — needed on sys.path for `from utils import …`
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gc
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

import project_paths as paths
from utils.chunked_postprocess import build_chunk_grid, _process_one
from utils.inference import _tag_image_edge
from utils.progress import EtaTracker, format_duration, format_int


# =============================================================================
# Edit these to match the interrupted run
# =============================================================================

# Directory of per-tile GeoParquet shards written by the original run.
# Must contain _shard_index.parquet and _tile_grid.parquet.
FEATURE_DIR = os.path.join(
    paths.FG_2_INFERENCE_RESULTS_DIR, "_mosaik_2026_05_10__1722_per_tile_features"
)

# Output directory and stem for the resumed crowns GPKG.
OUTPUT_DIR = paths.FG_2_INFERENCE_RESULTS_DIR
OUTPUT_STEM = "_mosaik_2026_05_10__1722_resumed"

# Ortho path — needed only to read pixel_area and edge_tolerance.
ORTHO_PATH = paths.ORTHO_PATH

# --- Postprocess parameters — must mirror FG_2_inference_runner.py ----------

CHUNK_SIZE_M = 1000.0
CHUNK_BUFFER_M = 40.96
STITCH_SHIFT_M = 1.0
DEDUP_IOU_THRESHOLD = 0.7
DEDUP_CONTAINMENT_THRESHOLD = 0.92
MOSAIC_DUST_AREA_M2 = 0.1
DROP_IMAGE_EDGE_CROWNS = False


# =============================================================================
# Helpers
# =============================================================================

_LAYER = "tree_crowns"


def _chunk_result_path(chunk_results_dir: Path, idx: int) -> Path:
    return chunk_results_dir / f"chunk_{idx:06d}.parquet"


def _chunk_is_done(path: Path) -> bool:
    """True if the parquet exists and has a readable row-group header."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import pyarrow.parquet as pq
        pq.read_metadata(path)  # cheap — only reads the footer
        return True
    except Exception:
        return False


def _prep_chunk_for_gpkg(
    gdf: gpd.GeoDataFrame,
    image_bounds: tuple,
    edge_tolerance: float,
    pixel_area: float,
    id_start: int,
) -> gpd.GeoDataFrame:
    """Tag image-edge crowns, compute output columns, assign instance_ids."""
    gdf = _tag_image_edge(gdf, image_bounds, edge_tolerance)

    if DROP_IMAGE_EDGE_CROWNS:
        gdf = gdf[~gdf["touches_image_edge"]].reset_index(drop=True)

    if len(gdf) == 0:
        return gdf

    gdf = gdf.reset_index(drop=True)
    gdf["instance_id"] = np.arange(id_start, id_start + len(gdf), dtype=np.uint32)
    gdf["area_m2"] = gdf.geometry.area
    gdf["area_px"] = (
        (gdf["area_m2"] / pixel_area).round().astype(np.int64)
        if pixel_area > 0
        else np.zeros(len(gdf), dtype=np.int64)
    )
    centroids = gdf.geometry.centroid
    gdf["centroid_x"] = centroids.x
    gdf["centroid_y"] = centroids.y

    cols = [
        "instance_id", "area_px", "area_m2",
        "centroid_x", "centroid_y", "touches_image_edge", "geometry",
    ]
    return gpd.GeoDataFrame(gdf[cols], crs=gdf.crs)


# =============================================================================
# Stage 1: per-chunk processing (checkpointed)
# =============================================================================

def run_chunks(
    feature_dir: Path,
    chunk_results_dir: Path,
    chunk_dicts: list,
    tile_grid_path: str,
) -> None:
    n_chunks = len(chunk_dicts)
    n_skipped = sum(
        1 for i in range(n_chunks)
        if _chunk_is_done(_chunk_result_path(chunk_results_dir, i))
    )
    n_todo = n_chunks - n_skipped
    print(f"[resume] stage 1: {format_int(n_todo)} chunks to process, "
          f"{format_int(n_skipped)} already done")

    if n_todo == 0:
        print("[resume] stage 1: all chunks already processed, skipping")
        return

    tracker = EtaTracker(n_todo)
    n_done_this_run = 0

    for i, chunk in enumerate(chunk_dicts):
        out_path = _chunk_result_path(chunk_results_dir, i)
        if _chunk_is_done(out_path):
            continue

        args = (
            chunk,
            str(feature_dir),
            tile_grid_path,
            STITCH_SHIFT_M,
            DEDUP_IOU_THRESHOLD,
            DEDUP_CONTAINMENT_THRESHOLD,
            MOSAIC_DUST_AREA_M2,
            False,  # return_culled
        )
        result = _process_one(args)
        result.to_parquet(out_path, index=False)
        del result
        gc.collect()

        n_done_this_run += 1
        tracker.tick()
        if n_done_this_run % 5 == 0 or n_done_this_run == n_todo:
            print(
                f"[resume] chunk {format_int(i + 1)}/{format_int(n_chunks)} "
                f"(+{format_int(n_done_this_run)} this run), "
                f"ETA {format_duration(tracker.eta_seconds())}"
            )

    print(f"[resume] stage 1: all {format_int(n_chunks)} chunks complete")


# =============================================================================
# Stage 2: streaming GPKG write (one chunk at a time, append mode)
# =============================================================================

def stream_write_gpkg(
    chunk_results_dir: Path,
    n_chunks: int,
    image_bounds: tuple,
    edge_tolerance: float,
    pixel_area: float,
    gpkg_path: str,
) -> int:
    """
    Read chunk parquets one at a time, tag edges, compute columns, and append
    to the output GPKG. Peak RAM = one chunk's data (~3 000–5 000 polygons).

    Returns the total number of crowns written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(gpkg_path)), exist_ok=True)

    # Remove a stale partial GPKG from a previous killed attempt so we always
    # write a consistent file. The chunk parquets are the durable checkpoint.
    if os.path.exists(gpkg_path):
        print(f"[resume] removing stale {gpkg_path} to start a fresh write")
        os.remove(gpkg_path)

    instance_id_counter = 1
    total_written = 0
    gpkg_created = False
    crs = None

    for i in range(n_chunks):
        path = _chunk_result_path(chunk_results_dir, i)
        gdf = gpd.read_parquet(path)

        if crs is None and getattr(gdf, "crs", None) is not None:
            crs = gdf.crs

        if len(gdf) == 0:
            continue

        out = _prep_chunk_for_gpkg(
            gdf, image_bounds, edge_tolerance, pixel_area, instance_id_counter
        )
        del gdf
        gc.collect()

        if len(out) == 0:
            continue

        instance_id_counter += len(out)

        if not gpkg_created:
            out.to_file(gpkg_path, layer=_LAYER, driver="GPKG", engine="pyogrio")
            gpkg_created = True
        else:
            out.to_file(gpkg_path, layer=_LAYER, driver="GPKG", engine="pyogrio",
                        mode="a")

        total_written += len(out)
        del out
        gc.collect()

        if (i + 1) % 50 == 0 or i == n_chunks - 1:
            print(f"[resume] stage 2: wrote chunk {format_int(i + 1)}/"
                  f"{format_int(n_chunks)}, {format_int(total_written)} crowns")

    # Edge case: every chunk was empty — write an empty layer with the schema.
    if not gpkg_created:
        empty = gpd.GeoDataFrame(
            {
                "instance_id": np.array([], dtype=np.uint32),
                "area_px": np.array([], dtype=np.int64),
                "area_m2": np.array([], dtype=np.float64),
                "centroid_x": np.array([], dtype=np.float64),
                "centroid_y": np.array([], dtype=np.float64),
                "touches_image_edge": np.array([], dtype=bool),
            },
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )
        empty.to_file(gpkg_path, layer=_LAYER, driver="GPKG", engine="pyogrio")

    return total_written


# =============================================================================
# Main
# =============================================================================

def main():
    feature_dir = Path(FEATURE_DIR)
    chunk_results_dir = feature_dir / "_chunk_results"
    chunk_results_dir.mkdir(exist_ok=True)

    tile_grid_parquet = feature_dir / "_tile_grid.parquet"
    if not tile_grid_parquet.exists():
        raise FileNotFoundError(
            f"Tile-grid parquet not found: {tile_grid_parquet}\n"
            "This file is written inside the feature_dir by run_chunked. "
            "If it is missing the original run must be restarted."
        )

    tile_grid = gpd.read_parquet(tile_grid_parquet)
    bounds = tile_grid.total_bounds  # [minx, miny, maxx, maxy]
    image_bounds = (float(bounds[0]), float(bounds[1]),
                    float(bounds[2]), float(bounds[3]))
    print(f"[resume] image_bounds = {image_bounds}")

    chunk_grid = build_chunk_grid(image_bounds, CHUNK_SIZE_M, CHUNK_BUFFER_M)
    chunk_dicts = chunk_grid.drop(columns=["geometry"]).to_dict("records")
    n_chunks = len(chunk_dicts)
    print(f"[resume] {format_int(n_chunks)} super-tile chunks "
          f"(chunk_size={CHUNK_SIZE_M} m, buffer={CHUNK_BUFFER_M} m)")

    with rasterio.open(ORTHO_PATH) as ortho:
        pixel_area = abs(ortho.transform.a * ortho.transform.e)
        edge_tolerance = max(abs(ortho.transform.a), abs(ortho.transform.e)) * 0.5

    # --- Stage 1 --------------------------------------------------------------
    run_chunks(feature_dir, chunk_results_dir, chunk_dicts,
               str(tile_grid_parquet))

    # --- Stage 2 --------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gpkg_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_STEM}_crowns.gpkg")
    print(f"[resume] stage 2: streaming {format_int(n_chunks)} chunk parquets "
          f"-> {gpkg_path}")

    n_written = stream_write_gpkg(
        chunk_results_dir, n_chunks, image_bounds, edge_tolerance,
        pixel_area, gpkg_path,
    )
    print(f"[resume] done — {format_int(n_written)} crowns written to {gpkg_path}")


if __name__ == "__main__":
    main()
