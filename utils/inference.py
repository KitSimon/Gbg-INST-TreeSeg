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
End-to-end orchestrator: ortho + AerialFormer raster → GeoPackage of crowns.

Pure vector pipeline with bounded memory:

    1. `instance_stitching.run_inference` runs Cellpose-SAM per tile and
       streams polygonised features to a directory of GeoParquet shards.
       Memory bound: one batch (~5000 features) at a time, regardless of
       AOI size.

    2. `chunked_postprocess.run_chunked` splits the AOI into super-tile
       chunks. For each chunk, it reads only the per-tile features
       intersecting the chunk's (core + buffer) bbox and runs the same
       three-pass reconciliation as before:

            stitch -> merge_duplicates (union) -> resolve_mosaic (clip)

       Each chunk emits only polygons whose centroid is in its core, so
       every output polygon is produced by exactly one chunk. Per-chunk
       working set is bounded by `chunk_size_m`; chunks can run in
       parallel via `n_postprocess_workers`.

    3. Surviving crowns are tagged `touches_image_edge` and (optionally)
       culled. Sequential `instance_id` is assigned at this layer so the
       diagnostic culled GPKG carries a `winner_instance_id` matching the
       final crowns layer.

    4. The final crowns are written as `<stem>_crowns.gpkg` (via pyogrio).
       A sidecar `<stem>_tile_grid.gpkg` is always written. Optionally a
       diagnostic `<stem>_dedup_culled.gpkg` is written too. The streaming
       feature directory is cleaned up unless retained for debugging.
"""

from __future__ import annotations

import gc
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box as shapely_box

from .chunked_postprocess import ChunkedOutput, run_chunked
from .config import InferenceConfig
from .instance_stitching import run_inference as _run_inference
from .progress import format_int
from .vector_export import write_crowns_gpkg, write_tile_grid_gpkg


def _tag_image_edge(crowns: gpd.GeoDataFrame, image_bounds, tolerance: float):
    """Add a `touches_image_edge` column based on proximity to the ortho's
    outer boundary."""
    if len(crowns) == 0:
        crowns = crowns.copy()
        crowns["touches_image_edge"] = []
        return crowns

    minx, miny, maxx, maxy = image_bounds
    image_box = shapely_box(minx, miny, maxx, maxy)
    inner_box = shapely_box(
        minx + tolerance,
        miny + tolerance,
        maxx - tolerance,
        maxy - tolerance,
    )
    out = crowns.copy()
    out["touches_image_edge"] = (
        ~out.geometry.within(inner_box) & out.geometry.intersects(image_box)
    )
    return out


def _attach_winner_instance_id(
    culled: gpd.GeoDataFrame, survivors: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """For each culled polygon, find the surviving polygon with the largest
    intersection area and copy its `instance_id` into `winner_instance_id`."""
    out = culled.copy()
    if len(out) == 0:
        out["winner_instance_id"] = np.array([], dtype=np.int64)
        return out
    if "instance_id" not in survivors.columns:
        raise ValueError("survivors must carry an 'instance_id' column")

    sindex = survivors.sindex
    winners = np.full(len(out), -1, dtype=np.int64)
    for i, geom in enumerate(out.geometry.values):
        if geom is None or geom.is_empty:
            continue
        cand_iloc = list(sindex.query(geom, predicate="intersects"))
        if not cand_iloc:
            continue
        best_inter = 0.0
        best_iid = -1
        for j in cand_iloc:
            inter = geom.intersection(survivors.geometry.iloc[j]).area
            if inter > best_inter:
                best_inter = inter
                best_iid = int(survivors.iloc[j]["instance_id"])
        winners[i] = best_iid

    out["winner_instance_id"] = winners
    return out


_GPKG_LAYER = "tree_crowns"
_TAG = "[Gbg-INST-TreeSeg]"


def _fmt_filesize(path: str) -> str:
    """Human-readable file size, e.g. '847 MB'."""
    try:
        b = os.path.getsize(path)
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.0f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def _print_run_summary(
    t_stage1: float,
    t_stage2: float,
    t_stage3: float,
    tile_stats: dict,
    n_final_crowns: int,
    n_edge_crowns: int,
    aoi_area_km2: float,
    gpkg_path: str,
) -> None:
    """Print a formatted run-summary block to stdout."""
    from .progress import format_duration, format_int

    SEP = "─" * 56

    def _pct(num: int, denom: int) -> str:
        return f"{100 * num / denom:.1f}%" if denom else "—"

    def _row(label: str, value: str, note: str = "", indent: int = 1) -> str:
        pad = "  " * indent
        return f"{_TAG}  {pad}{label:<24}{value:>13}  {note}"

    n_tiles = tile_stats["n_tiles"]
    n_skipped = tile_stats["n_skipped"]
    n_compute = tile_stats["n_compute"]
    n_compute_det = tile_stats["n_compute_detections"]
    n_compute_no_det = n_compute - n_compute_det
    n_raw = tile_stats["n_raw_crowns"]
    t_total = t_stage1 + t_stage2 + t_stage3
    reduction = f"{n_raw / n_final_crowns:.1f}×" if n_final_crowns > 0 else "—"

    lines: list[str] = [
        f"{_TAG} {SEP}",
        f"{_TAG}                  run summary",
        f"{_TAG} {SEP}",
        _TAG,
        f"{_TAG}  Timing",
        _row("tile inference", format_duration(t_stage1)),
        _row("postprocess", format_duration(t_stage2)),
        _row("GPKG write", format_duration(t_stage3)),
        f"{_TAG}    {'─' * 40}",
        _row("total", format_duration(t_total)),
        _TAG,
        f"{_TAG}  Tile processing   {format_int(n_tiles)} tiles",
        _row("skipped", format_int(n_skipped),
             f"({_pct(n_skipped, n_tiles)})  no tree mask"),
        _row("compute", format_int(n_compute),
             f"({_pct(n_compute, n_tiles)})  Cellpose ran"),
    ]
    if n_compute > 0:
        lines += [
            _row("with detections", format_int(n_compute_det),
                 f"({_pct(n_compute_det, n_compute)} of compute)", indent=2),
            _row("no detections", format_int(n_compute_no_det),
                 f"({_pct(n_compute_no_det, n_compute)} of compute)", indent=2),
        ]
    lines += [
        _TAG,
        f"{_TAG}  Crowns",
        _row("raw per-tile", format_int(n_raw)),
        _row("final (deduped)", format_int(n_final_crowns),
             f"{reduction} reduction"),
        _row("image-edge", format_int(n_edge_crowns),
             f"({_pct(n_edge_crowns, n_final_crowns)} of final)"),
        _TAG,
        f"{_TAG}  Coverage",
        _row("AOI area", f"{aoi_area_km2:.1f} km²"),
    ]
    if n_final_crowns > 0 and aoi_area_km2 > 0:
        density = int(n_final_crowns / aoi_area_km2)
        lines.append(_row("crown density", f"{format_int(density)} / km²"))
    lines += [
        _TAG,
        f"{_TAG}  Output",
        _row("crowns GPKG", _fmt_filesize(gpkg_path), os.path.basename(gpkg_path)),
        _TAG,
        f"{_TAG} {SEP}",
    ]

    print("\n".join(lines))


def _stream_write_gpkg(
    chunked: ChunkedOutput,
    gpkg_path: str,
    image_bounds: tuple,
    edge_tolerance: float,
    pixel_area: float,
    drop_image_edge: bool = False,
    simplify_tolerance: Optional[float] = None,
) -> int:
    """Write crowns to a GeoPackage by streaming one chunk at a time.

    Reads each chunk parquet, tags image-edge crowns, computes output
    columns, and appends to the GPKG with pyogrio's append mode. Peak RAM
    is bounded to one chunk's worth of polygons (~3 000–5 000 typically),
    regardless of total crown count.

    Returns (n_written, n_edge_crowns).
    """
    instance_id_counter = 1
    total_written = 0
    total_edge = 0
    gpkg_created = False

    for gdf in chunked.iter_kept():
        if len(gdf) == 0:
            continue

        tagged = _tag_image_edge(gdf, image_bounds, edge_tolerance)
        del gdf

        total_edge += int(tagged["touches_image_edge"].sum())

        if drop_image_edge:
            tagged = tagged[~tagged["touches_image_edge"]].reset_index(drop=True)

        if len(tagged) == 0:
            continue

        if simplify_tolerance is not None:
            tagged["geometry"] = tagged.geometry.simplify(
                simplify_tolerance, preserve_topology=True
            )

        tagged = tagged.reset_index(drop=True)
        tagged["instance_id"] = np.arange(
            instance_id_counter,
            instance_id_counter + len(tagged),
            dtype=np.uint32,
        )
        tagged["area_m2"] = tagged.geometry.area
        tagged["area_px"] = (
            (tagged["area_m2"] / pixel_area).round().astype(np.int64)
            if pixel_area > 0
            else np.zeros(len(tagged), dtype=np.int64)
        )
        centroids = tagged.geometry.centroid
        tagged["centroid_x"] = centroids.x
        tagged["centroid_y"] = centroids.y

        out = gpd.GeoDataFrame(
            tagged[["instance_id", "area_px", "area_m2",
                    "centroid_x", "centroid_y", "touches_image_edge", "geometry"]],
            crs=tagged.crs,
        )
        del tagged

        if not gpkg_created:
            out.to_file(gpkg_path, layer=_GPKG_LAYER, driver="GPKG", engine="pyogrio")
            gpkg_created = True
        else:
            out.to_file(gpkg_path, layer=_GPKG_LAYER, driver="GPKG", engine="pyogrio",
                        mode="a")

        instance_id_counter += len(out)
        total_written += len(out)
        del out
        gc.collect()

    if not gpkg_created:
        # Every chunk was empty — write an empty layer with the correct schema.
        write_crowns_gpkg(
            gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=chunked.crs),
                             crs=chunked.crs),
            gpkg_path,
            pixel_area=pixel_area,
        )

    return total_written, total_edge


def run_pipeline(cfg: InferenceConfig, cp_model=None) -> dict:
    """
    Run the full instance pipeline. Returns a dict with the produced paths.

    Args:
        cp_model: optional pre-instantiated Cellpose model (to avoid reloading
            the checkpoint when running on multiple AOIs in one process).
    """
    import time as _time
    t0 = _time.monotonic()

    os.makedirs(cfg.output_dir, exist_ok=True)
    stem = cfg.output_stem or os.path.splitext(os.path.basename(cfg.ortho_path))[0]
    if cfg.add_timestamp_suffix:
        # _YYYY_MM_DD__HHMM (double underscore separates date from time).
        # Computed once so every output file in this run shares the suffix.
        stem = f"{stem}_{datetime.now().strftime('%Y_%m_%d__%H%M')}"
    gpkg_path = os.path.join(cfg.output_dir, f"{stem}_crowns.gpkg")
    tile_grid_path = os.path.join(cfg.output_dir, f"{stem}_tile_grid.gpkg")
    feature_dir = os.path.join(cfg.output_dir, f"{stem}_per_tile_features")

    # Stage 1: streaming Cellpose inference -> GeoParquet shards
    _, tile_grid, tile_stats = _run_inference(
        cfg, cp_model=cp_model, feature_sink_dir=feature_dir
    )
    write_tile_grid_gpkg(tile_grid, tile_grid_path)
    t1 = _time.monotonic()

    with rasterio.open(cfg.ortho_path) as ortho:
        pixel_area = abs(ortho.transform.a * ortho.transform.e)
        edge_tolerance = max(abs(ortho.transform.a), abs(ortho.transform.e)) * 0.5

    # AOI = the geographic intersection that the stitcher actually inferred over,
    # not the ortho's full extent. Crowns at the AOI edge are real boundary
    # crowns from the pipeline's POV (inference never saw beyond it).
    from rasterio.coords import BoundingBox
    image_bounds = BoundingBox(*tile_grid.total_bounds)

    _bounds = (
        image_bounds.left, image_bounds.bottom,
        image_bounds.right, image_bounds.top,
    )

    # AOI bounding-box area (used for crown density in the summary).
    aoi_area_km2 = (
        (_bounds[2] - _bounds[0]) * (_bounds[3] - _bounds[1])
    ) / 1e6

    # Stage 2: chunked postprocess — results written to disk one chunk at a
    # time; the main process never accumulates a list of GeoDataFrames.
    chunked = run_chunked(
        feature_dir,
        tile_grid,
        _bounds,
        chunk_size_m=cfg.chunk_size_m,
        buffer_m=cfg.chunk_buffer_m,
        stitch_shift_m=cfg.stitch_shift_m,
        dedup_iou_threshold=cfg.dedup_iou_threshold,
        dedup_containment_threshold=cfg.dedup_containment_threshold,
        mosaic_dust_area_m2=cfg.mosaic_dust_area_m2,
        n_workers=cfg.n_postprocess_workers,
        return_culled=cfg.write_dedup_culled,
    )

    t2 = _time.monotonic()

    # Stage 3: streaming GPKG write — image-edge tagging, instance_id
    # assignment, and output column computation all happen per chunk.
    # Peak RAM = one chunk's polygons at a time.
    n_written, n_edge_crowns = _stream_write_gpkg(
        chunked,
        gpkg_path=gpkg_path,
        image_bounds=_bounds,
        edge_tolerance=edge_tolerance,
        pixel_area=pixel_area,
        drop_image_edge=cfg.drop_image_edge_crowns,
    )
    t3 = _time.monotonic()
    print(f"{_TAG} wrote {format_int(n_written)} crowns -> {gpkg_path}")

    out = {"gpkg": gpkg_path, "tile_grid": tile_grid_path}

    if cfg.write_dedup_culled:
        # Accumulate culled chunks — these are polygons removed by dedup, so
        # significantly fewer than the kept crowns and manageable in RAM.
        culled_parts = [c for c in chunked.iter_culled() if len(c) > 0]
        if culled_parts:
            dedup_culled = gpd.GeoDataFrame(
                pd.concat(culled_parts, ignore_index=True), crs=chunked.crs
            )
        else:
            dedup_culled = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=chunked.crs), crs=chunked.crs
            )
        del culled_parts

        # Read survivors back from the GPKG to match culled polygons to their
        # winning instance_id. This re-read is necessary because we never held
        # the full survivor set in RAM during the streaming write above.
        survivors = gpd.read_file(gpkg_path, layer=_GPKG_LAYER, engine="pyogrio")
        culled_path = os.path.join(cfg.output_dir, f"{stem}_dedup_culled.gpkg")
        diag = _attach_winner_instance_id(dedup_culled, survivors)
        del survivors, dedup_culled

        if len(diag) == 0:
            stub = gpd.GeoDataFrame(
                {
                    "phase": np.array([], dtype=object),
                    "winner_instance_id": np.array([], dtype=np.int64),
                    "original_area_m2": np.array([], dtype=np.float64),
                    "removed_area_m2": np.array([], dtype=np.float64),
                },
                geometry=gpd.GeoSeries([], crs=chunked.crs),
                crs=chunked.crs,
            )
            stub.to_file(culled_path, layer="dedup_culled", driver="GPKG",
                         engine="pyogrio")
        else:
            diag.to_file(culled_path, layer="dedup_culled", driver="GPKG",
                         engine="pyogrio")
        print(
            f"[Gbg-INST-TreeSeg] wrote {format_int(len(diag))} dedup-culled "
            f"polygons -> {culled_path}"
        )
        out["dedup_culled"] = culled_path

    _print_run_summary(
        t_stage1=t1 - t0,
        t_stage2=t2 - t1,
        t_stage3=t3 - t2,
        tile_stats=tile_stats,
        n_final_crowns=n_written,
        n_edge_crowns=n_edge_crowns,
        aoi_area_km2=aoi_area_km2,
        gpkg_path=gpkg_path,
    )

    chunked.cleanup()

    # Cleanup the streaming feature dir unless explicitly retained
    if not cfg.keep_feature_shards:
        try:
            shutil.rmtree(feature_dir)
        except OSError as e:
            print(f"[Gbg-INST-TreeSeg] could not remove feature shards: {e}")
    else:
        out["feature_shards"] = feature_dir

    return out
