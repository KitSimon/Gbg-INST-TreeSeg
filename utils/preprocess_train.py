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
Build (image, instance-mask) tile pairs for Cellpose fine-tuning.

Inputs (cfg.sources, list[TrainSource]):
    - One or more orthophoto (GeoTIFF or VRT) + crowns GeoPackage pairs.
      Each TrainSource carries its own optional QC attribute, bbox filter,
      and val region (tuple bbox OR a gpkg whose features define the val AOI).

Outputs (under cfg.output_dir):
    train/<src_name>_<x>_<y>_img.tif       (H, W, 3) uint8 image, Cellpose-ready
    train/<src_name>_<x>_<y>_masks.tif     (H, W)    uint16 instance IDs
    val/...                                 (same layout, only if a source has
                                             a val_bbox_filter set — written
                                             whenever any source produces val
                                             tiles)
    review/...                              (only if a source has qc_field set
                                             and a tile intersects any
                                             QC-failed crown — the pair is
                                             written here so a human can
                                             eyeball why the tile was rejected)
    sources_manifest.json                   (per-source identities + counts;
                                             FG_1_train_runner.py reads this
                                             to record traceability)

Tiles whose ortho window contains fewer than cfg.min_tile_instances passed
crowns are skipped. The filename embedding (start_x, start_y) is what
AF_1_preprocess uses, so the two preprocessors interoperate visually if the
user wants to compare.

Each polygon is assigned a unique uint16 ID per tile (1..N within the tile).
Cellpose's training pipeline does not care that IDs are tile-local rather
than globally unique.

QC filtering (when src.qc_field is set on a given TrainSource):
    - Crowns are split into 'passed' and 'failed' based on src.qc_field and
      src.qc_pass_values (DT_1_preprocess.py-style).
    - A tile is routed to review/ (or skipped, if cfg.qc_review_subdir is
      None) if ANY failed crown intersects it. Otherwise it is written to
      train/ or val/ with only the passed crowns rasterised. This mirrors
      DT_1's "ANY-fails-the-tile" rule and keeps training labels clean.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window as RioWindow
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .bands import select_bands
from .config import TrainConfig, TrainSource


def _split_crowns_by_qc(
    crowns: gpd.GeoDataFrame,
    qc_field: Optional[str],
    qc_pass_values: Optional[list],
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Split a crown GeoDataFrame into (passed, failed) by QC field.

    Mirrors DT_1_preprocess._load_and_filter_crowns. If qc_field is None or
    not present in the columns, all crowns are returned as 'passed' and the
    'failed' frame is empty (preserving column schema).
    """
    if qc_field is None:
        return crowns, crowns.iloc[0:0].copy()
    if qc_field not in crowns.columns:
        print(
            f"[Gbg-INST-TreeSeg] WARNING: QC field '{qc_field}' missing from "
            f"crowns; available columns: {list(crowns.columns)}. "
            f"Treating all crowns as passed."
        )
        return crowns, crowns.iloc[0:0].copy()
    if qc_pass_values is not None:
        mask = crowns[qc_field].isin(qc_pass_values)
    else:
        mask = crowns[qc_field].notna()
    return crowns[mask].copy(), crowns[~mask].copy()


def _tile_origins(width: int, height: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    """Top-left (x, y) corners of all tiles, edge-aligned."""
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if xs and xs[-1] != width - tile_size:
        xs.append(max(0, width - tile_size))
    if ys and ys[-1] != height - tile_size:
        ys.append(max(0, height - tile_size))
    return [(x, y) for y in ys for x in xs]


def _tile_centre_world(transform, x: int, y: int, tile_size: int) -> Tuple[float, float]:
    cx = (x + tile_size / 2) * transform.a + transform.c
    cy = (y + tile_size / 2) * transform.e + transform.f
    return cx, cy


def _centre_in_region(cx: float, cy: float, region: Any) -> bool:
    """region is None | (xmin, ymin, xmax, ymax) tuple | shapely (Multi)Polygon.
    None never contains anything (caller decides what that means)."""
    if region is None:
        return False
    if isinstance(region, tuple):
        xmin, ymin, xmax, ymax = region
        return xmin <= cx <= xmax and ymin <= cy <= ymax
    return region.contains(Point(cx, cy))


def _split_for_tile(
    transform, x: int, y: int, tile_size: int,
    bbox_region: Any, val_region: Any,
) -> Optional[str]:
    """Return 'val', 'train', or None (drop).

    Same semantics as the pre-multi-source code: tile centre in val_region →
    val, tile centre outside bbox_region (when set) → drop, otherwise train.
    Both regions accept the same union type (tuple bbox or Shapely geometry)."""
    cx, cy = _tile_centre_world(transform, x, y, tile_size)
    if val_region is not None and _centre_in_region(cx, cy, val_region):
        return "val"
    if bbox_region is not None and not _centre_in_region(cx, cy, bbox_region):
        return None
    return "train"


def _resolve_val_region(val_bbox_filter: Any, ortho_crs, fallback_crs_epsg: int) -> Any:
    """Normalize val_bbox_filter into a form suitable for _centre_in_region.

    None         → None (no val for this source)
    tuple        → tuple (passed through unchanged)
    str (path)   → shapely (Multi)Polygon in the ortho's CRS, built by
                   unioning all features in the GeoPackage.
    """
    if val_bbox_filter is None:
        return None
    if isinstance(val_bbox_filter, tuple):
        return val_bbox_filter
    if not isinstance(val_bbox_filter, str):
        raise TypeError(
            f"val_bbox_filter must be a 4-tuple, a gpkg path string, or None; "
            f"got {type(val_bbox_filter).__name__}"
        )
    val_gdf = gpd.read_file(val_bbox_filter)
    if val_gdf.empty:
        print(
            f"[Gbg-INST-TreeSeg] WARNING: val gpkg '{val_bbox_filter}' is empty; "
            f"no val tiles will be produced for this source."
        )
        return None
    if val_gdf.crs is None:
        print(
            f"[Gbg-INST-TreeSeg] WARNING: val gpkg '{val_bbox_filter}' has no CRS; "
            f"assuming EPSG:{fallback_crs_epsg}"
        )
        val_gdf = val_gdf.set_crs(epsg=fallback_crs_epsg)
    if val_gdf.crs != ortho_crs:
        try:
            val_gdf = val_gdf.to_crs(ortho_crs)
        except Exception:
            print(
                f"[Gbg-INST-TreeSeg] WARNING: could not reproject val gpkg "
                f"'{val_bbox_filter}' to ortho CRS; falling back to "
                f"EPSG:{fallback_crs_epsg}"
            )
            val_gdf = val_gdf.to_crs(epsg=fallback_crs_epsg)
    geom = unary_union(val_gdf.geometry.values)
    if geom.is_empty:
        print(
            f"[Gbg-INST-TreeSeg] WARNING: val gpkg '{val_bbox_filter}' produced "
            f"an empty union geometry; no val tiles will be produced."
        )
        return None
    return geom


def _write_tile_pair(
    output_dir: str, subdir: str, stem: str,
    img_3: np.ndarray, mask_array: np.ndarray,
    tile_size: int, ortho_crs, tile_transform,
) -> None:
    """Write a (image, mask) pair to <output_dir>/<subdir>/. Subdir is created
    on demand so we don't proactively create review/ unless QC is enabled."""
    out_dir = os.path.join(output_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"{stem}_img.tif")
    mask_path = os.path.join(out_dir, f"{stem}_masks.tif")
    with rasterio.open(
        img_path, "w",
        driver="GTiff",
        width=tile_size, height=tile_size, count=3, dtype="uint8",
        crs=ortho_crs, transform=tile_transform, compress="lzw",
    ) as dst:
        dst.write(np.transpose(img_3, (2, 0, 1)))
    with rasterio.open(
        mask_path, "w",
        driver="GTiff",
        width=tile_size, height=tile_size, count=1, dtype="uint16",
        crs=ortho_crs, transform=tile_transform, compress="lzw", nodata=0,
    ) as dst:
        dst.write(mask_array, 1)


_EMPTY_COUNTS = {
    "train": 0,
    "val": 0,
    "review": 0,
    "skipped_no_crowns": 0,
    "skipped_filtered": 0,
    "skipped_qc_failed": 0,
}


def _fallback_crs(fallback_crs_epsg: int):
    """rasterio.crs.CRS.from_epsg() consults GDAL's libproj proj.db, which is
    broken in some envs (PROJ db version mismatch). Round-tripping through
    pyproj's bundled libproj avoids that path entirely."""
    import pyproj
    wkt = pyproj.CRS.from_epsg(fallback_crs_epsg).to_wkt()
    return rasterio.crs.CRS.from_wkt(wkt)


def _process_source(
    src: TrainSource, cfg: TrainConfig, src_name: str,
) -> dict:
    """Tile a single ortho/crowns pair. Returns a counts dict for this source.

    Mirrors the pre-multi-source body, with per-source bbox/val/qc/id_field
    rules taken from `src` and global tile_size/stride/min_instances/output_dir
    taken from `cfg`. Tile filenames are prefixed with src_name so multiple
    sources can share the same flat output directory without collisions."""
    crowns_all = gpd.read_file(src.crowns_gpkg)
    if crowns_all.empty:
        print(
            f"[Gbg-INST-TreeSeg] WARNING: '{src.crowns_gpkg}' has no features; "
            f"skipping source '{src_name}'"
        )
        return dict(_EMPTY_COUNTS)

    with rasterio.open(src.ortho_path) as ortho:
        if crowns_all.crs is None:
            print(
                "[Gbg-INST-TreeSeg] WARNING: crowns GeoPackage has no CRS; "
                f"assuming EPSG:{cfg.fallback_crs_epsg}"
            )
            crowns_all = crowns_all.set_crs(epsg=cfg.fallback_crs_epsg)

        # Some envs have a PROJ db / library mismatch where GDAL silently
        # degrades a perfectly good projected CRS to LOCAL_CS at read time
        # (it can't identify the WKT against the on-disk proj.db). The
        # rasterio CRS then has no defined relationship to anything and any
        # to_crs(...) call will fail with "Error creating Transformer from
        # CRS." When we detect this, fall back to fallback_crs_epsg — pyproj
        # can still build a transformer to/from a real EPSG code.
        ortho_crs = ortho.crs or _fallback_crs(cfg.fallback_crs_epsg)
        if ortho_crs.to_epsg() is None and not ortho_crs.is_projected:
            print(
                f"[Gbg-INST-TreeSeg] WARNING: ortho CRS unparseable by GDAL "
                f"(likely a PROJ db version mismatch — see README env caveats). "
                f"Falling back to EPSG:{cfg.fallback_crs_epsg} for both the "
                f"crown reprojection and the output tile CRS."
            )
            ortho_crs = _fallback_crs(cfg.fallback_crs_epsg)
            crowns_all = crowns_all.to_crs(epsg=cfg.fallback_crs_epsg)
        elif crowns_all.crs != ortho_crs:
            crowns_all = crowns_all.to_crs(ortho_crs)

        # Resolve val region (tuple or gpkg path) to a single comparable form.
        val_region = _resolve_val_region(
            src.val_bbox_filter, ortho_crs, cfg.fallback_crs_epsg
        )
        bbox_region = src.bbox_filter

        if val_region is not None:
            os.makedirs(os.path.join(cfg.output_dir, "val"), exist_ok=True)

        crowns_pass, crowns_fail = _split_crowns_by_qc(
            crowns_all, src.qc_field, src.qc_pass_values
        )
        if src.qc_field is not None:
            print(
                f"[Gbg-INST-TreeSeg] [{src_name}] QC split: {len(crowns_pass)} "
                f"passed, {len(crowns_fail)} failed (field='{src.qc_field}', "
                f"pass_values={src.qc_pass_values})"
            )

        if src.crowns_id_field is not None:
            id_values_pass = crowns_pass[src.crowns_id_field].to_numpy()
            if not np.issubdtype(id_values_pass.dtype, np.integer):
                raise ValueError(
                    f"crowns_id_field='{src.crowns_id_field}' must be int-typed; "
                    f"got dtype {id_values_pass.dtype}"
                )
        else:
            id_values_pass = np.arange(1, len(crowns_pass) + 1, dtype=np.int64)

        sindex_pass = crowns_pass.sindex if len(crowns_pass) > 0 else None
        sindex_fail = crowns_fail.sindex if len(crowns_fail) > 0 else None

        tiles = _tile_origins(
            ortho.width, ortho.height, cfg.tile_size, cfg.tile_stride
        )

        counts = dict(_EMPTY_COUNTS)

        for x, y in tiles:
            split = _split_for_tile(
                ortho.transform, x, y, cfg.tile_size, bbox_region, val_region
            )
            if split is None:
                counts["skipped_filtered"] += 1
                continue

            tile_transform = rasterio.windows.transform(
                RioWindow(x, y, cfg.tile_size, cfg.tile_size), ortho.transform
            )
            tile_bounds = box(*rasterio.windows.bounds(
                RioWindow(x, y, cfg.tile_size, cfg.tile_size), ortho.transform
            ))

            # 1. QC tile-level reject: ANY failed crown intersecting → review/skip
            qc_failed_tile = False
            if sindex_fail is not None:
                fail_idx = list(sindex_fail.intersection(tile_bounds.bounds))
                if fail_idx:
                    fail_candidates = crowns_fail.iloc[fail_idx]
                    if fail_candidates.intersects(tile_bounds).any():
                        qc_failed_tile = True

            # 2. Find passed crowns that intersect this tile
            keep = None
            if sindex_pass is not None:
                pass_idx = list(sindex_pass.intersection(tile_bounds.bounds))
                if pass_idx:
                    candidates = crowns_pass.iloc[pass_idx]
                    keep = candidates[candidates.intersects(tile_bounds)]

            if qc_failed_tile:
                counts["skipped_qc_failed"] += 1
                if cfg.qc_review_subdir is None:
                    continue
                if keep is not None and len(keep) > 0:
                    tile_local_ids = np.arange(1, len(keep) + 1, dtype=np.uint16)
                    shape_iter = list(zip(keep.geometry.values, tile_local_ids))
                    mask_array = rasterize(
                        shape_iter,
                        out_shape=(cfg.tile_size, cfg.tile_size),
                        transform=tile_transform,
                        fill=0, all_touched=False, dtype=np.uint16,
                    )
                else:
                    mask_array = np.zeros((cfg.tile_size, cfg.tile_size), dtype=np.uint16)

                win = RioWindow(x, y, cfg.tile_size, cfg.tile_size)
                img = ortho.read(window=win)
                img_3 = select_bands(img, cfg.band_indices, cfg.percentile_clip)
                stem = f"{src_name}_{x}_{y}"
                _write_tile_pair(
                    cfg.output_dir, cfg.qc_review_subdir, stem,
                    img_3, mask_array, cfg.tile_size, ortho_crs, tile_transform,
                )
                counts["review"] += 1
                continue

            if keep is None or len(keep) < cfg.min_tile_instances:
                counts["skipped_no_crowns"] += 1
                continue

            tile_local_ids = np.arange(1, len(keep) + 1, dtype=np.uint16)
            shape_iter = list(zip(keep.geometry.values, tile_local_ids))
            mask_array = rasterize(
                shape_iter,
                out_shape=(cfg.tile_size, cfg.tile_size),
                transform=tile_transform,
                fill=0, all_touched=False, dtype=np.uint16,
            )

            win = RioWindow(x, y, cfg.tile_size, cfg.tile_size)
            img = ortho.read(window=win)
            img_3 = select_bands(img, cfg.band_indices, cfg.percentile_clip)

            stem = f"{src_name}_{x}_{y}"
            _write_tile_pair(
                cfg.output_dir, split, stem,
                img_3, mask_array, cfg.tile_size, ortho_crs, tile_transform,
            )
            counts[split] += 1

    return counts


def _resolve_source_name(src: TrainSource) -> str:
    return src.name or os.path.splitext(os.path.basename(src.ortho_path))[0]


def _write_sources_manifest(
    cfg: TrainConfig, totals: dict, per_source: dict,
) -> str:
    """Write sources_manifest.json into cfg.output_dir so FG_1 can pick up
    source identities for the training info JSON without the user re-typing
    them in the train runner."""
    def _serialize_val(v):
        if v is None or isinstance(v, tuple):
            return v
        return str(v)

    manifest = {
        "sources": [
            {
                "name": _resolve_source_name(s),
                "ortho_path": s.ortho_path,
                "crowns_gpkg": s.crowns_gpkg,
                "crowns_id_field": s.crowns_id_field,
                "qc_field": s.qc_field,
                "qc_pass_values": s.qc_pass_values,
                "bbox_filter": s.bbox_filter,
                "val_bbox_filter": _serialize_val(s.val_bbox_filter),
            }
            for s in cfg.sources
        ],
        "tile_size": cfg.tile_size,
        "tile_stride": cfg.tile_stride,
        "band_indices": list(cfg.band_indices),
        "counts": {"totals": totals, "per_source": per_source},
    }
    manifest_path = os.path.join(cfg.output_dir, "sources_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path


def build_training_tiles(cfg: TrainConfig) -> dict:
    """
    Generate train/[val]/[review] tile pairs from one or more TrainSources.

    Returns a dict of:
        {"totals": {...}, "per_source": {<src_name>: {...}, ...}}

    Each per-source counts dict has the same shape as the pre-multi-source
    return value (train, val, review, skipped_*); totals is the sum.
    A sources_manifest.json is also written into cfg.output_dir so the train
    runner can record source identities in its info JSON without having to
    re-type them.
    """
    if not cfg.sources:
        raise ValueError(
            "cfg.sources is empty — provide at least one TrainSource. "
            "(Empty is allowed at training time but not when building tiles.)"
        )

    os.makedirs(os.path.join(cfg.output_dir, "train"), exist_ok=True)

    # Uniqueness check on source names so output tiles don't collide silently.
    seen_names: set = set()
    resolved_names: List[str] = []
    for src in cfg.sources:
        name = _resolve_source_name(src)
        if name in seen_names:
            raise ValueError(
                f"Duplicate source name '{name}' — set TrainSource.name "
                f"explicitly to disambiguate, otherwise output tiles will "
                f"collide in the flat train/ directory."
            )
        seen_names.add(name)
        resolved_names.append(name)

    totals = dict(_EMPTY_COUNTS)
    per_source: dict = {}

    for src, src_name in zip(cfg.sources, resolved_names):
        print(
            f"[Gbg-INST-TreeSeg] processing source '{src_name}': "
            f"{src.ortho_path} + {src.crowns_gpkg}"
        )
        counts = _process_source(src, cfg, src_name)
        per_source[src_name] = counts
        for k in totals:
            totals[k] += counts[k]

    manifest_path = _write_sources_manifest(cfg, totals, per_source)
    print(
        f"[Gbg-INST-TreeSeg] training-tile build complete. "
        f"Totals: {totals}. Manifest: {manifest_path}"
    )
    return {"totals": totals, "per_source": per_source}
