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
Tiled Cellpose inference with per-tile vector emission.

Each tile that contains canopy according to the AerialFormer semantic prior is
run through Cellpose-SAM. The resulting tile-local instance mask is polygonised
immediately into the ortho's CRS via a window-aware affine transform; one row
is appended to a GeoDataFrame per Cellpose instance, tagged with the source
`tile_id` and `local_instance_id`.

Cross-tile reconciliation (collapsing duplicates from the overlap region,
culling crowns clipped at internal seams) lives in `crown_postprocess.py` and
runs after this stage. This module is intentionally agnostic about cross-tile
identity: the same tree may appear here as several rows, one per tile that
saw it. That's fine — the post-process handles it.

Outputs (returned to the caller, not written here):
    per_tile_crowns: GeoDataFrame with columns
        - tile_id (int), local_instance_id (int), area_m2 (float), geometry
    tile_grid: GeoDataFrame with columns
        - tile_id (int), is_image_edge_{n,s,e,w} (bool), geometry (rectangle)
"""

from __future__ import annotations

from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.windows import Window as RioWindow
from rasterio.windows import transform as window_transform
from scipy.ndimage import gaussian_filter
from shapely.geometry import box as shapely_box
from shapely.geometry import shape as shapely_shape

from .bands import select_bands
from .config import InferenceConfig
from .feature_store import ParquetShardSink
from .progress import EtaTracker, format_duration, format_int
from .tiling import create_grids


def _open_with_crs_fallback(path: str, fallback_epsg: int):
    """Open a raster, warning if CRS is missing."""
    src = rasterio.open(path)
    if src.crs is None:
        print(
            f"[Gbg-INST-TreeSeg] WARNING: {path} has no CRS; "
            f"falling back to EPSG:{fallback_epsg}"
        )
    return src


def _intersection_window(src, left, bottom, right, top, tol_px=1e-3):
    """
    Pixel `Window` in `src` covering the CRS bbox (left, bottom, right, top).
    Verifies the bbox edges fall on `src`'s pixel grid (sub-pixel error < tol_px).
    """
    inv = ~src.transform
    col0, row0 = inv * (left, top)        # top-left → (col, row) origin of window
    col1, row1 = inv * (right, bottom)    # bottom-right
    c0, r0 = int(round(col0)), int(round(row0))
    c1, r1 = int(round(col1)), int(round(row1))
    for v_orig, v_int, label in (
        (col0, c0, "left"), (row0, r0, "top"),
        (col1, c1, "right"), (row1, r1, "bottom"),
    ):
        if abs(v_orig - v_int) > tol_px:
            raise ValueError(
                f"Intersection {label} edge is not on the pixel grid of "
                f"{src.name} (sub-pixel offset {abs(v_orig - v_int):.4f}px)."
            )
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f"Empty intersection window in {src.name}.")
    return RioWindow(c0, r0, c1 - c0, r1 - r0)


def _compute_overlap_windows(ortho, sem):
    """
    Find the geographic intersection of `ortho` and `sem` and return the
    pixel window into each plus the intersection's affine transform.

    The two rasters must share the same pixel size and have origins aligned
    on the same pixel grid (sub-pixel offsets are not handled — re-export one
    raster onto the other's grid if that ever becomes a real case). CRS
    mismatch (e.g. PROJCS vs LOCAL_CS for the same physical CRS) is warned
    about, not raised, since `rasterio.crs.CRS` equality is strict.
    """
    if (
        abs(ortho.transform.a - sem.transform.a) > 1e-9
        or abs(ortho.transform.e - sem.transform.e) > 1e-9
    ):
        raise ValueError(
            f"Pixel sizes differ: ortho=({ortho.transform.a}, {ortho.transform.e}) "
            f"vs sem=({sem.transform.a}, {sem.transform.e}). The two rasters must "
            "share the same pixel grid."
        )

    px = abs(ortho.transform.a)
    dx = (sem.transform.c - ortho.transform.c) / px
    dy = (sem.transform.f - ortho.transform.f) / px
    if abs(dx - round(dx)) > 1e-3 or abs(dy - round(dy)) > 1e-3:
        raise ValueError(
            f"Origins are not aligned on the pixel grid (sub-pixel offset "
            f"dx={dx - round(dx):.4f}px, dy={dy - round(dy):.4f}px). "
            "Re-export one raster onto the other's grid before running."
        )

    ob, sb = ortho.bounds, sem.bounds
    left = max(ob.left, sb.left)
    right = min(ob.right, sb.right)
    bottom = max(ob.bottom, sb.bottom)
    top = min(ob.top, sb.top)
    if right <= left or top <= bottom:
        raise ValueError(
            f"Ortho bounds {tuple(ob)} and semantic bounds {tuple(sb)} do "
            "not overlap geographically — cannot run inference."
        )

    if ortho.crs and sem.crs and ortho.crs != sem.crs:
        print(
            f"[Gbg-INST-TreeSeg] WARNING: ortho CRS != semantic CRS "
            f"({ortho.crs.to_string()[:60]}... vs "
            f"{sem.crs.to_string()[:60]}...). Proceeding because pixel grids "
            "are aligned — verify both rasters really are in the same projection."
        )

    ortho_win = _intersection_window(ortho, left, bottom, right, top)
    sem_win = _intersection_window(sem, left, bottom, right, top)
    if (ortho_win.width, ortho_win.height) != (sem_win.width, sem_win.height):
        raise ValueError(
            f"Intersection produced different shapes per raster "
            f"({ortho_win.width}x{ortho_win.height} vs "
            f"{sem_win.width}x{sem_win.height}). This should be unreachable "
            "after the pixel-size and origin checks."
        )

    aoi_transform = window_transform(ortho_win, ortho.transform)
    return ortho_win, sem_win, aoi_transform


def run_cellpose_on_tile(
    cp_model,
    tile_array_bgr: np.ndarray,
    tree_mask: np.ndarray,
    cfg: InferenceConfig,
) -> np.ndarray:
    """
    Apply the masking + blur recipe and run Cellpose-SAM. Returns a uint32
    array of tile-local instance IDs (0 = background).
    """
    img = tile_array_bgr.copy()
    img[tree_mask == 0] = 0

    if cfg.gaussian_blur_size > 0:
        sigma = max(0.5, cfg.gaussian_blur_size / 6.0)  # ~paper's 3x3 blur
        img = gaussian_filter(img, sigma=(sigma, sigma, 0)).astype(img.dtype)

    masks, _flows, _styles = cp_model.eval(
        img,
        diameter=cfg.diameter,
        flow_threshold=cfg.flow_threshold,
        cellprob_threshold=cfg.cellprob_threshold,
        min_size=cfg.min_size,
        normalize=True,
    )
    masks = np.asarray(masks)
    # Clamp to canopy pixels in case Cellpose grew an instance outside the
    # masked region (matches the original recipe).
    masks = masks * (tree_mask > 0)
    return masks.astype(np.uint32)


def _polygonize_tile_mask(
    tile_labels: np.ndarray,
    affine,
    tile_id: int,
) -> list:
    """
    Convert a tile-local uint32 instance mask into a list of feature dicts
    using `rasterio.features.shapes`. Feature geometries land in the ortho CRS
    via the window-aware `affine` transform.

    `rasterio.features.shapes` does not accept uint32; cast to int32 (lossless
    for any tile-local instance count we care about).
    """
    if tile_labels.max() == 0:
        return []

    if tile_labels.dtype == np.uint32:
        if tile_labels.max() > np.iinfo(np.int32).max:
            raise OverflowError(
                f"tile_labels has IDs above int32 max; got max={tile_labels.max()}"
            )
        data_for_shapes = tile_labels.astype(np.int32)
    else:
        data_for_shapes = tile_labels

    mask = tile_labels > 0
    feats = []
    for geom_dict, value in rio_shapes(
        data_for_shapes, mask=mask, transform=affine, connectivity=8
    ):
        geom = shapely_shape(geom_dict)
        if geom.is_empty:
            continue
        feats.append(
            {
                "tile_id": int(tile_id),
                "local_instance_id": int(value),
                "geometry": geom,
            }
        )
    return feats


def _build_tile_grid_gdf(grids, ortho_transform, crs) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of tile rectangles in the ortho CRS."""
    rows = []
    for tid, g in enumerate(grids):
        win = RioWindow(g["read_x"], g["read_y"], g["win_w"], g["win_h"])
        affine = window_transform(win, ortho_transform)
        # Window pixel corners (0,0) and (win_w, win_h) → CRS coords.
        x0, y0 = affine * (0, 0)
        x1, y1 = affine * (g["win_w"], g["win_h"])
        minx, maxx = (x0, x1) if x0 <= x1 else (x1, x0)
        miny, maxy = (y0, y1) if y0 <= y1 else (y1, y0)
        rows.append(
            {
                "tile_id": tid,
                "is_image_edge_n": bool(g["is_image_edge_n"]),
                "is_image_edge_s": bool(g["is_image_edge_s"]),
                "is_image_edge_w": bool(g["is_image_edge_w"]),
                "is_image_edge_e": bool(g["is_image_edge_e"]),
                "geometry": shapely_box(minx, miny, maxx, maxy),
            }
        )
    return gpd.GeoDataFrame(rows, crs=crs)


def run_inference(
    cfg: InferenceConfig,
    cp_model=None,
    feature_sink_dir: Optional[str] = None,
) -> Tuple[Optional[gpd.GeoDataFrame], gpd.GeoDataFrame, dict]:
    """
    Run the tiled Cellpose pipeline.

    Returns:
        (per_tile_crowns, tile_grid, tile_stats) where per_tile_crowns is
        the in-memory GeoDataFrame of every tile's polygonised Cellpose
        detections and tile_stats is a dict with tile-processing counts:
            n_tiles, n_skipped, n_compute,
            n_compute_detections, n_raw_crowns.

        If `feature_sink_dir` is given, per-tile features are streamed there
        as GeoParquet shards instead of accumulated in memory; the returned
        per_tile_crowns is None and downstream callers should read the
        feature store via `feature_store.read_shards_in_bbox`. This is the
        memory-bounded mode used by the chunked postprocess for large AOIs.

    `cp_model` may be supplied by the caller (e.g. tests pass a stub); if None,
    a Cellpose-SAM model is instantiated.
    """
    if cp_model is None:
        from cellpose import models as _cp_models  # lazy import (heavy)

        cp_model = _cp_models.CellposeModel(
            gpu=cfg.use_gpu,
            pretrained_model=cfg.cellpose_checkpoint or "cpsam",
            use_bfloat16=cfg.use_bfloat16,
        )

    sink: Optional[ParquetShardSink] = None
    if feature_sink_dir is not None:
        sink = ParquetShardSink(feature_sink_dir)

    feature_rows = []
    n_features_streamed = 0

    with _open_with_crs_fallback(cfg.ortho_path, cfg.fallback_crs_epsg) as ortho:
        with _open_with_crs_fallback(
            cfg.semantic_raster_path, cfg.fallback_crs_epsg
        ) as sem:
            ortho_aoi, sem_aoi, aoi_transform = _compute_overlap_windows(
                ortho, sem
            )
            if (ortho_aoi.width, ortho_aoi.height) != (ortho.width, ortho.height) \
                    or (sem_aoi.width, sem_aoi.height) != (sem.width, sem.height):
                print(
                    f"[Gbg-INST-TreeSeg] ortho is {ortho.width}x{ortho.height}, "
                    f"semantic is {sem.width}x{sem.height}; running on their "
                    f"geographic intersection: {ortho_aoi.width}x{ortho_aoi.height} px"
                )

            crs = ortho.crs or rasterio.crs.CRS.from_epsg(cfg.fallback_crs_epsg)

            grids = create_grids(
                ortho_aoi.width, ortho_aoi.height, cfg.window_size, cfg.stride
            )
            print(f"[Gbg-INST-TreeSeg] {format_int(len(grids))} tiles to process")

            tile_grid_gdf = _build_tile_grid_gdf(grids, aoi_transform, crs)

            tracker = EtaTracker(len(grids))
            n_compute_detections = 0

            for tid, g in enumerate(grids):
                ortho_win = RioWindow(
                    ortho_aoi.col_off + g["read_x"],
                    ortho_aoi.row_off + g["read_y"],
                    g["win_w"], g["win_h"],
                )
                sem_win = RioWindow(
                    sem_aoi.col_off + g["read_x"],
                    sem_aoi.row_off + g["read_y"],
                    g["win_w"], g["win_h"],
                )

                sem_data = sem.read(1, window=sem_win)
                tree_mask = (sem_data == cfg.tree_class_id).astype(np.uint8)
                if tree_mask.sum() == 0:
                    tracker.tick("skip")
                else:
                    img_data = ortho.read(window=ortho_win)  # (C, H, W)
                    img_3 = select_bands(
                        img_data, cfg.band_indices, cfg.percentile_clip
                    )

                    tile_labels = run_cellpose_on_tile(
                        cp_model, img_3, tree_mask, cfg
                    )
                    if tile_labels.max() == 0:
                        tracker.tick("compute")
                    else:
                        affine = window_transform(ortho_win, ortho.transform)
                        tile_feats = _polygonize_tile_mask(
                            tile_labels, affine, tile_id=tid
                        )

                        n_compute_detections += 1

                        if sink is not None and tile_feats:
                            # Dissolve per local_instance_id within the tile so
                            # each Cellpose instance is one row downstream
                            # (matches the in-memory branch's later groupby).
                            tile_gdf = gpd.GeoDataFrame(tile_feats, crs=crs)
                            tile_gdf = tile_gdf.dissolve(
                                by=["tile_id", "local_instance_id"],
                                as_index=False,
                            )
                            tile_gdf["area_m2"] = tile_gdf.geometry.area
                            tile_gdf = tile_gdf[
                                ["tile_id", "local_instance_id", "area_m2",
                                 "geometry"]
                            ]
                            sink.append(tile_gdf)
                            n_features_streamed += len(tile_gdf)
                            _n_crowns_tile = len(tile_gdf)
                        else:
                            feature_rows.extend(tile_feats)
                            _n_crowns_tile = len(tile_feats)

                        tracker.tick("compute", weight=max(1, _n_crowns_tile))

                if (tid + 1) % 25 == 0 or tid == len(grids) - 1:
                    n = (
                        n_features_streamed
                        if sink is not None
                        else len(feature_rows)
                    )
                    mode = "streamed" if sink is not None else "in memory"
                    print(
                        f"[Gbg-INST-TreeSeg] processed "
                        f"{format_int(tid + 1)}/{format_int(len(grids))} tiles, "
                        f"{format_int(n)} per-tile crowns ({mode}), "
                        f"ETA {format_duration(tracker.eta_seconds())}"
                    )

    if sink is not None:
        sink.close()
        _tile_stats = {
            "n_tiles": len(grids),
            "n_skipped": tracker._counts.get("skip", 0),
            "n_compute": tracker._counts.get("compute", 0),
            "n_compute_detections": n_compute_detections,
            "n_raw_crowns": n_features_streamed,
        }
        return None, tile_grid_gdf, _tile_stats

    if feature_rows:
        per_tile = gpd.GeoDataFrame(feature_rows, crs=crs)
        # Dissolve per (tile_id, local_instance_id) — `rasterio.features.shapes`
        # emits one polygon per connected component, so a multi-part Cellpose
        # instance (rare, e.g. canopy with a hole) would be split otherwise.
        per_tile = per_tile.dissolve(
            by=["tile_id", "local_instance_id"], as_index=False
        )
        per_tile["area_m2"] = per_tile.geometry.area
        per_tile = per_tile[
            ["tile_id", "local_instance_id", "area_m2", "geometry"]
        ]
    else:
        per_tile = gpd.GeoDataFrame(
            {
                "tile_id": np.array([], dtype=np.int64),
                "local_instance_id": np.array([], dtype=np.int64),
                "area_m2": np.array([], dtype=np.float64),
            },
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )

    n_raw = len(per_tile) if feature_rows else 0
    _tile_stats = {
        "n_tiles": len(grids),
        "n_skipped": tracker._counts.get("skip", 0),
        "n_compute": tracker._counts.get("compute", 0),
        "n_compute_detections": n_compute_detections,
        "n_raw_crowns": n_raw,
    }
    return per_tile, tile_grid_gdf, _tile_stats
