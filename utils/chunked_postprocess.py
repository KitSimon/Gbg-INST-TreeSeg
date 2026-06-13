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
Spatial chunking of the cross-tile vector reconciliation.

The single-mosaic path in `crown_postprocess.merge_crowns_across_tiles`
holds every per-tile crown in RAM and runs `resolve_mosaic` over the whole
set. That works comfortably up to ~10⁴ crowns; beyond that the working set
grows linearly and the algorithm hits memory and CPU limits.

This module slices the AOI into super-tile **chunks** that each run the
identical three-pass reconciliation, with two architectural properties:

1. **Bounded working set.** Each chunk reads only the per-tile features
   intersecting its (core + buffer) extent. The buffer is wider than the
   largest realistic crown so every cluster a chunk evaluates is fully
   visible; the chunk's reconciliation reproduces what the global
   single-mosaic algorithm would have produced inside the core.

2. **Centroid-in-core ownership.** Each chunk emits only polygons whose
   centroid lies inside its core (half-open intervals avoid double-claim
   at boundaries). Adjacent chunks' buffer bands overlap by `buffer_m` —
   crowns there are computed by both chunks but emitted by exactly one.
   This eliminates the need for a separate cross-chunk dedup pass while
   preserving the global "largest first" mosaic ordering for any crown
   centred inside its chunk.

Together the two make the postprocess:

  * Memory-bounded by `chunk_size_m` rather than by AOI size.
  * Embarrassingly parallel — N workers process N chunks, no shared state.
  * Identical in output to the single-mosaic path when the buffer is wide
    enough (the verification tests assert this on a synthetic AOI split
    into one or many chunks).

Buffer sizing rule of thumb: ≥ 2 × maximum realistic crown diameter.
Wider is safe and cheap; the buffer band is a few percent of total work
for a typical 1 km super-tile.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box

from .crown_postprocess import (
    merge_duplicates,
    resolve_mosaic,
    stitch_crowns_from_grid,
)
from .feature_store import read_shards_in_bbox
from .progress import EtaTracker, format_duration, format_int


@dataclass
class ChunkedOutput:
    """Disk-backed result from `run_chunked`.

    Chunk results are written to per-file parquets in `chunk_dir` (and
    `culled_dir` when `return_culled=True`) so that the main process never
    accumulates all results in RAM. Callers iterate over kept/culled chunks
    one at a time and call `cleanup()` when done.
    """

    chunk_dir: Path
    n_chunks: int
    crs: Any
    culled_dir: Optional[Path] = None

    def iter_kept(self) -> Iterator[gpd.GeoDataFrame]:
        """Yield per-chunk kept GeoDataFrames in chunk order."""
        for i in range(self.n_chunks):
            yield gpd.read_parquet(self.chunk_dir / f"chunk_{i:06d}.parquet")

    def iter_culled(self) -> Iterator[gpd.GeoDataFrame]:
        """Yield per-chunk culled GeoDataFrames (only valid if culled_dir set)."""
        if self.culled_dir is None:
            return
        for i in range(self.n_chunks):
            yield gpd.read_parquet(self.culled_dir / f"chunk_{i:06d}.parquet")

    def cleanup(self) -> None:
        """Remove the temporary chunk directories."""
        for d in (self.chunk_dir, self.culled_dir):
            if d is not None and Path(d).exists():
                shutil.rmtree(d, ignore_errors=True)


def build_chunk_grid(
    image_bounds: Tuple[float, float, float, float],
    chunk_size_m: float,
    buffer_m: float,
) -> gpd.GeoDataFrame:
    """
    Build a non-overlapping core grid covering `image_bounds`. For each
    chunk, the core bbox is the canonical "owns these centroids" region,
    and the extended bbox = core + `buffer_m` on every side that does
    not coincide with the AOI's outer boundary.

    Returns a GeoDataFrame whose `geometry` is the core box. Other columns
    expose the extended bbox for use by `process_chunk`:
        chunk_id, core_minx/miny/maxx/maxy, ext_minx/miny/maxx/maxy.
    """
    minx, miny, maxx, maxy = image_bounds
    rows: List[Dict[str, Any]] = []

    chunk_id = 0
    y = miny
    while y < maxy - 1e-9:
        y_next = min(y + chunk_size_m, maxy)
        x = minx
        while x < maxx - 1e-9:
            x_next = min(x + chunk_size_m, maxx)
            on_w = x <= minx + 1e-9
            on_e = x_next >= maxx - 1e-9
            on_s = y <= miny + 1e-9
            on_n = y_next >= maxy - 1e-9
            ext_minx = x if on_w else x - buffer_m
            ext_maxx = x_next if on_e else x_next + buffer_m
            ext_miny = y if on_s else y - buffer_m
            ext_maxy = y_next if on_n else y_next + buffer_m
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "core_minx": x,
                    "core_miny": y,
                    "core_maxx": x_next,
                    "core_maxy": y_next,
                    "ext_minx": ext_minx,
                    "ext_miny": ext_miny,
                    "ext_maxx": ext_maxx,
                    "ext_maxy": ext_maxy,
                    "geometry": shapely_box(x, y, x_next, y_next),
                }
            )
            chunk_id += 1
            x = x_next
        y = y_next
    return gpd.GeoDataFrame(rows, geometry="geometry")


def _crowns_centred_in_core(
    crowns: gpd.GeoDataFrame, core: Tuple[float, float, float, float]
) -> gpd.GeoDataFrame:
    """Filter to crowns whose centroid lies in [core_minx, core_maxx) x
    [core_miny, core_maxy). Half-open intervals prevent double-claim at
    chunk boundaries."""
    if len(crowns) == 0:
        return crowns
    centroids = crowns.geometry.centroid
    cx = centroids.x.to_numpy()
    cy = centroids.y.to_numpy()
    minx, miny, maxx, maxy = core
    mask = (cx >= minx) & (cx < maxx) & (cy >= miny) & (cy < maxy)
    return crowns[mask].reset_index(drop=True)


def process_chunk(
    chunk: Dict[str, Any],
    per_tile_crowns: gpd.GeoDataFrame,
    tile_grid: gpd.GeoDataFrame,
    *,
    stitch_shift_m: float,
    dedup_iou_threshold: float,
    dedup_containment_threshold: float,
    mosaic_dust_area_m2: float,
    return_culled: bool = False,
):
    """
    Run the three-pass reconciliation for a single chunk. Inputs are
    already restricted to the chunk's extended bbox (i.e. the caller has
    spatial-filtered both the per-tile crowns and the tile grid).

    Emits only polygons whose centroid is in the chunk's core.

    Returns the kept GeoDataFrame, or `(kept, culled)` if return_culled.
    Culled rows from `merge_duplicates` and `resolve_mosaic` carry their
    upstream metadata; the orchestrator concatenates them across chunks.
    """
    core = (
        chunk["core_minx"],
        chunk["core_miny"],
        chunk["core_maxx"],
        chunk["core_maxy"],
    )

    if len(per_tile_crowns) == 0:
        empty = per_tile_crowns.iloc[0:0].copy()
        if return_culled:
            return empty, empty
        return empty

    # Geometry repair (idiomatic before any sjoin / intersection)
    repaired = per_tile_crowns.copy()
    repaired["geometry"] = repaired.geometry.buffer(0)
    repaired = repaired[~repaired.is_empty & repaired.is_valid]

    stitched = stitch_crowns_from_grid(
        repaired, tile_grid, shift=stitch_shift_m, verbose=False
    )
    if len(stitched) == 0:
        if return_culled:
            return stitched, stitched.copy()
        return stitched

    if return_culled:
        merged, m_culled = merge_duplicates(
            stitched,
            iou_threshold=dedup_iou_threshold,
            containment_threshold=dedup_containment_threshold,
            verbose=False,
            return_culled=True,
        )
        kept, mo_culled = resolve_mosaic(
            merged,
            dust_area=mosaic_dust_area_m2,
            verbose=False,
            return_culled=True,
        )
    else:
        merged = merge_duplicates(
            stitched,
            iou_threshold=dedup_iou_threshold,
            containment_threshold=dedup_containment_threshold,
            verbose=False,
        )
        kept = resolve_mosaic(
            merged, dust_area=mosaic_dust_area_m2, verbose=False
        )

    kept = _crowns_centred_in_core(kept, core)

    if not return_culled:
        return kept

    # Filter culled rows to those whose ORIGINAL polygon (the rasterised
    # tile-local detection) had its centroid in this chunk's core. This
    # prevents the same culled row from being emitted by multiple chunks.
    # The geometry on culled rows is the removed/clipped piece, so we
    # use a heuristic: take the row's bbox centre. For merge_duplicate
    # phase the geometry is the original polygon; for mosaic_clipped it's
    # the clipped-away piece (which is geographically near the original
    # polygon's centroid, so this is still a reasonable proxy).
    culled_parts = []
    for df in (m_culled, mo_culled):
        if len(df) == 0:
            continue
        df = df.copy()
        bounds = df.bounds
        cx = (bounds["minx"] + bounds["maxx"]) / 2.0
        cy = (bounds["miny"] + bounds["maxy"]) / 2.0
        m = (
            (cx >= core[0])
            & (cx < core[2])
            & (cy >= core[1])
            & (cy < core[3])
        )
        culled_parts.append(df[m])

    if culled_parts:
        culled = gpd.GeoDataFrame(
            pd.concat(culled_parts, ignore_index=True), crs=kept.crs
        )
    else:
        culled = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries([], crs=kept.crs), crs=kept.crs
        )
    return kept, culled


def _process_one(args):
    """Worker-side entry point. Reads the chunk's slice from the parquet
    feature store and runs `process_chunk`. Used both in-process (when
    n_workers=1) and via ProcessPoolExecutor."""
    (
        chunk,
        feature_dir,
        tile_grid_path,
        stitch_shift_m,
        dedup_iou_threshold,
        dedup_containment_threshold,
        mosaic_dust_area_m2,
        return_culled,
    ) = args

    ext_bbox = (
        chunk["ext_minx"],
        chunk["ext_miny"],
        chunk["ext_maxx"],
        chunk["ext_maxy"],
    )
    per_tile = read_shards_in_bbox(feature_dir, bbox=ext_bbox)
    if len(per_tile) == 0:
        empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))
        if return_culled:
            return empty, empty
        return empty

    tile_grid = gpd.read_parquet(tile_grid_path)
    # Restrict tile grid to tiles intersecting the extended bbox.
    if len(tile_grid) > 0:
        tg_bounds = tile_grid.bounds
        m = (
            (tg_bounds["maxx"] >= ext_bbox[0])
            & (tg_bounds["minx"] <= ext_bbox[2])
            & (tg_bounds["maxy"] >= ext_bbox[1])
            & (tg_bounds["miny"] <= ext_bbox[3])
        )
        tile_grid = tile_grid[m].reset_index(drop=True)

    return process_chunk(
        chunk,
        per_tile,
        tile_grid,
        stitch_shift_m=stitch_shift_m,
        dedup_iou_threshold=dedup_iou_threshold,
        dedup_containment_threshold=dedup_containment_threshold,
        mosaic_dust_area_m2=mosaic_dust_area_m2,
        return_culled=return_culled,
    )


def _process_one_to_disk(packed):
    """Worker entry point that writes results to disk rather than returning them.

    Used by both the sequential and parallel paths in `run_chunked` so that
    chunk GeoDataFrames are never accumulated in the main-process heap.
    `packed` is a tuple of (process_one_args, kept_path, culled_path).
    culled_path may be None when return_culled is False.
    """
    args, kept_path, culled_path = packed
    result = _process_one(args)
    if culled_path is not None:
        kept, culled = result
        kept.to_parquet(kept_path, index=False)
        culled.to_parquet(culled_path, index=False)
    else:
        result.to_parquet(kept_path, index=False)
    return True  # cheap sentinel — workers must not return large objects


def run_chunked(
    feature_dir: str,
    tile_grid: gpd.GeoDataFrame,
    image_bounds: Tuple[float, float, float, float],
    *,
    chunk_size_m: float,
    buffer_m: float,
    stitch_shift_m: float,
    dedup_iou_threshold: float,
    dedup_containment_threshold: float,
    mosaic_dust_area_m2: float,
    n_workers: int = 1,
    return_culled: bool = False,
    verbose: bool = True,
) -> ChunkedOutput:
    """
    Build the chunk grid, run reconciliation per chunk (sequentially or in
    parallel), and return a `ChunkedOutput` backed by per-chunk parquet files
    on disk.

    Each chunk's result is written to disk immediately after processing and
    freed from RAM. This bounds peak memory to O(one chunk) regardless of
    AOI size, fixing the OOM that occurs when accumulating all chunk
    GeoDataFrames in a list before a final pd.concat.

    `tile_grid` is written to a parquet alongside the feature shards so
    worker processes can read it independently.

    The caller is responsible for calling `ChunkedOutput.cleanup()` once it
    has finished reading the chunk parquets.
    """
    chunk_grid = build_chunk_grid(image_bounds, chunk_size_m, buffer_m)
    if verbose:
        print(
            f"[chunked_postprocess] {format_int(len(chunk_grid))} super-tile(s), "
            f"chunk_size={chunk_size_m} m, buffer={buffer_m} m"
        )

    feature_dir_p = Path(feature_dir)
    tile_grid_path = feature_dir_p / "_tile_grid.parquet"
    tile_grid.to_parquet(tile_grid_path, index=False)

    chunk_dicts = chunk_grid.drop(columns=["geometry"]).to_dict("records")
    job_args = [
        (
            c,
            str(feature_dir_p),
            str(tile_grid_path),
            stitch_shift_m,
            dedup_iou_threshold,
            dedup_containment_threshold,
            mosaic_dust_area_m2,
            return_culled,
        )
        for c in chunk_dicts
    ]
    n_jobs = len(job_args)

    kept_dir = Path(tempfile.mkdtemp(prefix="_treeseg_kept_"))
    culled_dir = Path(tempfile.mkdtemp(prefix="_treeseg_culled_")) if return_culled else None

    def _kept_path(i: int) -> str:
        return str(kept_dir / f"chunk_{i:06d}.parquet")

    def _culled_path(i: int) -> Optional[str]:
        return str(culled_dir / f"chunk_{i:06d}.parquet") if culled_dir else None

    tracker = EtaTracker(n_jobs)

    if n_workers <= 1:
        for i, args in enumerate(job_args):
            _process_one_to_disk((args, _kept_path(i), _culled_path(i)))
            gc.collect()
            tracker.tick()
            if verbose and ((i + 1) % 5 == 0 or i == n_jobs - 1):
                print(
                    f"[chunked_postprocess] chunk "
                    f"{format_int(i + 1)}/{format_int(n_jobs)} done, "
                    f"ETA {format_duration(tracker.eta_seconds())}"
                )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        packed = [
            (args, _kept_path(i), _culled_path(i))
            for i, args in enumerate(job_args)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_process_one_to_disk, p): i for i, p in enumerate(packed)}
            done = 0
            for fut in as_completed(futs):
                fut.result()  # re-raises any worker exception
                done += 1
                tracker.tick()
                if verbose and (done % 5 == 0 or done == n_jobs):
                    print(
                        f"[chunked_postprocess] chunk "
                        f"{format_int(done)}/{format_int(n_jobs)} done, "
                        f"ETA {format_duration(tracker.eta_seconds())}"
                    )

    # Discover CRS cheaply from the first input shard — avoids reading any
    # (potentially empty) chunk result just to obtain the projection.
    from .feature_store import read_shard_index
    crs = None
    index = read_shard_index(feature_dir)
    if len(index) > 0:
        sample = gpd.read_parquet(feature_dir_p / index.iloc[0]["shard_path"])
        crs = sample.crs

    return ChunkedOutput(
        chunk_dir=kept_dir,
        n_chunks=n_jobs,
        crs=crs,
        culled_dir=culled_dir,
    )
