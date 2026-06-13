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
Cross-tile vector merge for tree-crown polygons.

Three sequential passes turn the per-tile Cellpose detections into a clean,
non-overlapping crown mosaic with no gaps where canopy was detected:

Pass A — internal-seam stitching (`stitch_crowns_from_grid`)
    For each tile, shrink the tile's bounds inward by `stitch_shift_m` on
    sides that face another tile (i.e. NOT on sides that sit on the ortho's
    outer boundary). Keep only crowns whose geometry lies fully within the
    shrunk box. Crowns clipped at an internal seam fail this test and are
    dropped; the same crown observed whole in the neighbouring tile (where
    the seam is on the *opposite* side of the overlap region) survives.

Pass B — duplicate merge by union (`merge_duplicates`)
    Cluster polygons whose IoU exceeds `merge_iou_threshold` OR whose
    containment ratio (intersection / min area) exceeds
    `merge_containment_threshold`. Replace each cluster with the **union**
    of its members. This handles "same crown observed twice with slightly
    different shapes": the merged polygon is the maximal observed extent,
    so unique area from any cluster member is preserved. (The previous
    keep-largest-drop-rest approach silently lost any unique area.)

Pass C — mosaic resolution by clipping (`resolve_mosaic`)
    Sort survivors by area descending. For each polygon in order, subtract
    the running union of already-kept polygons; if the clipped remnant has
    area above `mosaic_dust_area_m2`, keep it. This guarantees:
      • single ownership: every pixel is in at most one output polygon
      • no spurious gaps: every pixel covered by any input polygon is
        covered by some output polygon (modulo the dust threshold)
    The same pixel-level invariant the raster pipeline gave for free, made
    explicit at the polygon level.

Image-edge crowns are deliberately preserved by stitching: their tile's
image-boundary sides are not shrunk, so a partial crown there can fall fully
within its tile's shrunk box and survive pass A. They are tagged downstream
in `inference.run_pipeline` and may be culled there if the user opts in.
"""

from __future__ import annotations

from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from rtree.index import Index as RTreeIndex
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union


def _shrunk_tile_box(tile_row, shift: float):
    """Build a tile box shrunk inward by `shift` on every side that is not
    flagged as an image-boundary side."""
    minx, miny, maxx, maxy = tile_row.geometry.bounds
    if not bool(tile_row.get("is_image_edge_w", False)):
        minx += shift
    if not bool(tile_row.get("is_image_edge_s", False)):
        miny += shift
    if not bool(tile_row.get("is_image_edge_e", False)):
        maxx -= shift
    if not bool(tile_row.get("is_image_edge_n", False)):
        maxy -= shift
    if maxx <= minx or maxy <= miny:
        return None
    return shapely_box(minx, miny, maxx, maxy)


def stitch_crowns_from_grid(
    crowns_gdf: gpd.GeoDataFrame,
    tile_grid_gdf: gpd.GeoDataFrame,
    shift: float = 1.0,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Cull seam-cropped crowns by shrinking each tile inward (internal sides only)
    and keeping crowns that lie fully within the shrunk tile.
    """
    if "tile_id" not in crowns_gdf.columns:
        raise ValueError("crowns_gdf must have a 'tile_id' column")
    if "tile_id" not in tile_grid_gdf.columns:
        raise ValueError("tile_grid_gdf must have a 'tile_id' column")

    if len(crowns_gdf) == 0:
        return crowns_gdf.copy()

    tile_bounds: Dict[int, object] = {}
    for _, row in tile_grid_gdf.iterrows():
        tile_bounds[int(row["tile_id"])] = _shrunk_tile_box(row, shift)

    parts = []
    for tid, tile_crowns in crowns_gdf.groupby("tile_id"):
        shrunk_box = tile_bounds.get(int(tid))
        if shrunk_box is None or shrunk_box.is_empty:
            continue

        filter_gdf = gpd.GeoDataFrame(
            {"geometry": [shrunk_box]}, crs=crowns_gdf.crs
        )
        within = gpd.sjoin(
            tile_crowns, filter_gdf, how="inner", predicate="within"
        )
        if len(within) > 0:
            within = within.drop(columns=["index_right"], errors="ignore")
            parts.append(within)

    if not parts:
        if verbose:
            print(
                "[crown_postprocess] stitch: no crowns survived "
                f"(shift={shift}); consider lowering it."
            )
        return crowns_gdf.iloc[0:0].copy()

    stitched = pd.concat(parts, ignore_index=True)
    stitched = stitched.drop_duplicates(subset=["geometry"], keep="first")
    stitched = stitched.reset_index(drop=True)
    out = gpd.GeoDataFrame(stitched, crs=crowns_gdf.crs)

    if verbose:
        print(
            f"[crown_postprocess] stitch: {len(crowns_gdf)} -> {len(out)} "
            f"(shift={shift})"
        )
    return out


def merge_duplicates(
    crowns_gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.7,
    containment_threshold: float = 0.92,
    verbose: bool = True,
    return_culled: bool = False,
):
    """
    Cluster overlapping crowns by HIGH IoU or containment, then replace each
    cluster with the **union** of its members. The union preserves any unique
    area from any cluster member — it is the maximal observed extent of the
    crown.

    Thresholds should be high enough that only true duplicates of the same
    physical crown are clustered. Two distinct neighbouring crowns must NOT
    end up clustered, or they would be wrongly fused into one polygon.

    Args:
        crowns_gdf:           Stitched per-tile crowns.
        iou_threshold:        Cluster edge if intersection / union > this.
        containment_threshold: Cluster edge if intersection / min(area_i, area_j) > this.
        verbose:              Print summary.
        return_culled:        If True, also return a GeoDataFrame of the
                              cluster members that were absorbed into a union
                              (i.e. all members except the cluster's anchor —
                              the row whose position is preserved).

    Returns:
        merged                       if return_culled is False
        (merged, culled)             if return_culled is True
    """
    if len(crowns_gdf) == 0:
        if return_culled:
            return crowns_gdf.copy(), crowns_gdf.copy()
        return crowns_gdf.copy()

    crowns = crowns_gdf.copy()
    crowns = crowns[~crowns.is_empty & crowns.is_valid].reset_index(drop=True)
    n = len(crowns)
    if n == 0:
        if return_culled:
            return crowns, crowns.copy()
        return crowns

    areas = crowns.geometry.area.to_numpy()

    join = gpd.sjoin(crowns, crowns, how="inner", predicate="intersects")
    join = join[join.index != join.index_right]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    geoms = crowns.geometry.values
    for i, j in zip(join.index.to_numpy(), join["index_right"].to_numpy()):
        if i >= j:
            continue
        if find(i) == find(j):
            continue
        gi, gj = geoms[i], geoms[j]
        inter_area = gi.intersection(gj).area
        if inter_area <= 0:
            continue
        ai, aj = areas[i], areas[j]
        iou = inter_area / (ai + aj - inter_area) if (ai + aj - inter_area) > 0 else 0
        if iou > iou_threshold:
            union(i, j)
            continue
        min_area = ai if ai < aj else aj
        if min_area > 0 and (inter_area / min_area) > containment_threshold:
            union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    anchor_for: Dict[int, int] = {}
    keep_idx: List[int] = []
    new_geoms: Dict[int, object] = {}
    for comp in groups.values():
        anchor = max(comp, key=lambda k: areas[k])
        keep_idx.append(anchor)
        if len(comp) == 1:
            new_geoms[anchor] = geoms[anchor]
        else:
            new_geoms[anchor] = unary_union([geoms[k] for k in comp])
        for k in comp:
            anchor_for[k] = anchor

    keep_idx_sorted = sorted(keep_idx)
    merged = crowns.iloc[keep_idx_sorted].copy()
    merged["geometry"] = [new_geoms[k] for k in keep_idx_sorted]
    merged = merged.reset_index(drop=True)
    merged = gpd.GeoDataFrame(merged, crs=crowns_gdf.crs)

    if verbose:
        print(
            f"[crown_postprocess] merge_duplicates: {n} -> {len(merged)} "
            f"(iou>{iou_threshold}, contain>{containment_threshold})"
        )

    if not return_culled:
        return merged

    culled_idx = [k for k in range(n) if k not in set(keep_idx)]
    culled = crowns.iloc[culled_idx].copy()
    culled["original_area_m2"] = [areas[k] for k in culled_idx]
    culled["phase"] = "merge_duplicate"
    culled = culled.reset_index(drop=True)
    return merged, gpd.GeoDataFrame(culled, crs=crowns_gdf.crs)


def resolve_mosaic(
    crowns_gdf: gpd.GeoDataFrame,
    dust_area: float = 0.1,
    verbose: bool = True,
    return_culled: bool = False,
):
    """
    Greedy mosaic resolution: sort by area descending, walk through, subtract
    the running union of already-kept polygons from each polygon. Keep the
    clipped remnant if its area exceeds `dust_area`.

    The output is a non-overlapping polygon set (single-ownership invariant)
    with the property that any pixel covered by an input polygon is covered
    by some output polygon, except for slivers below `dust_area`.

    Args:
        crowns_gdf:    Crowns to resolve. May contain overlapping geometries.
        dust_area:     Threshold (CRS units squared) below which a clipped
                       remnant is discarded as a sliver.
        verbose:       Print summary.
        return_culled: If True, also return culled diagnostic GeoDataFrame
                       containing two phases:
                         - phase="mosaic_clipped": the clipped-away portion
                           (i.e. original_geom - kept_geom) of polygons that
                           survived but lost area. Geometry is the lost piece.
                         - phase="mosaic_dropped": full original geometry of
                           polygons that were entirely consumed.

    Returns:
        kept                       if return_culled is False
        (kept, culled)             if return_culled is True
    """
    if len(crowns_gdf) == 0:
        if return_culled:
            return crowns_gdf.copy(), crowns_gdf.copy()
        return crowns_gdf.copy()

    crowns = crowns_gdf.copy()
    crowns = crowns[~crowns.is_empty & crowns.is_valid].copy()
    crowns["_area"] = crowns.geometry.area
    crowns = crowns.sort_values("_area", ascending=False).reset_index(drop=True)

    kept_rows: List[dict] = []
    kept_geoms: List[object] = []
    culled_rows: List[dict] = []
    # Incremental spatial index for the bbox prefilter. rtree supports
    # insert-as-you-go without rebuilds, so the candidate query is
    # O(log |kept|) instead of O(|kept|). Critical at >~1e4 polygons.
    sindex = RTreeIndex()

    n = len(crowns)
    for i in range(n):
        row = crowns.iloc[i]
        original_geom = row.geometry
        original_area = float(row["_area"])
        clipped_geom = original_geom

        if kept_geoms:
            cand_ids = list(sindex.intersection(original_geom.bounds))
            if cand_ids:
                intersecting = [
                    kept_geoms[c] for c in cand_ids
                    if kept_geoms[c].intersects(original_geom)
                ]
                if intersecting:
                    claimed = unary_union(intersecting)
                    clipped_geom = original_geom.difference(claimed)

        clipped_area = clipped_geom.area if not clipped_geom.is_empty else 0.0

        if clipped_geom.is_empty or clipped_area <= dust_area:
            if return_culled:
                culled_rows.append(
                    {
                        **{c: row[c] for c in crowns.columns if c != "_area"},
                        "phase": "mosaic_dropped",
                        "original_area_m2": original_area,
                        "removed_area_m2": original_area,
                    }
                )
            continue

        new_row = {c: row[c] for c in crowns.columns if c != "_area"}
        new_row["geometry"] = clipped_geom
        kept_idx = len(kept_geoms)
        kept_rows.append(new_row)
        kept_geoms.append(clipped_geom)
        sindex.insert(kept_idx, clipped_geom.bounds)

        if return_culled and (original_area - clipped_area) > dust_area:
            removed_geom = original_geom.difference(clipped_geom)
            if not removed_geom.is_empty:
                culled_rows.append(
                    {
                        **{c: row[c] for c in crowns.columns if c not in ("_area", "geometry")},
                        "geometry": removed_geom,
                        "phase": "mosaic_clipped",
                        "original_area_m2": original_area,
                        "removed_area_m2": float(original_area - clipped_area),
                    }
                )

    kept = gpd.GeoDataFrame(kept_rows, geometry="geometry", crs=crowns_gdf.crs)
    if verbose:
        n_dropped = sum(1 for r in culled_rows if r.get("phase") == "mosaic_dropped")
        n_clipped = sum(1 for r in culled_rows if r.get("phase") == "mosaic_clipped")
        print(
            f"[crown_postprocess] resolve_mosaic: {n} -> {len(kept)} "
            f"(dropped={n_dropped}, partially_clipped={n_clipped}, "
            f"dust>{dust_area})"
        )

    if not return_culled:
        return kept

    if culled_rows:
        culled = gpd.GeoDataFrame(
            culled_rows, geometry="geometry", crs=crowns_gdf.crs
        )
    else:
        culled = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crowns_gdf.crs), crs=crowns_gdf.crs)
    return kept, culled


def merge_crowns_across_tiles(
    per_tile_crowns: gpd.GeoDataFrame,
    tile_grid: gpd.GeoDataFrame,
    stitch_shift_m: float = 1.0,
    dedup_iou_threshold: float = 0.7,
    dedup_containment_threshold: float = 0.92,
    mosaic_dust_area_m2: float = 0.1,
    verbose: bool = True,
    return_dedup_culled: bool = False,
):
    """
    Three-pass cross-tile vector merge:

        stitch  -> merge_duplicates  -> resolve_mosaic

    Optionally returns a combined diagnostic GeoDataFrame of polygons that
    were absorbed during the merge phase (`phase="merge_duplicate"`) and
    polygons / portions removed during the mosaic phase
    (`phase="mosaic_clipped"` or `"mosaic_dropped"`).
    """
    if len(per_tile_crowns) == 0:
        if return_dedup_culled:
            return per_tile_crowns.copy(), per_tile_crowns.copy()
        return per_tile_crowns.copy()

    repaired = per_tile_crowns.copy()
    repaired["geometry"] = repaired.geometry.buffer(0)
    repaired = repaired[~repaired.is_empty & repaired.is_valid]

    stitched = stitch_crowns_from_grid(
        repaired, tile_grid, shift=stitch_shift_m, verbose=verbose
    )
    if len(stitched) == 0:
        if return_dedup_culled:
            return stitched, stitched.copy()
        return stitched

    if return_dedup_culled:
        merged, merge_culled = merge_duplicates(
            stitched,
            iou_threshold=dedup_iou_threshold,
            containment_threshold=dedup_containment_threshold,
            verbose=verbose,
            return_culled=True,
        )
        kept, mosaic_culled = resolve_mosaic(
            merged,
            dust_area=mosaic_dust_area_m2,
            verbose=verbose,
            return_culled=True,
        )
        # Combine culled GDFs (they may have differing schemas — concat is
        # tolerant). Don't pass a CRS to GeoDataFrame() if both are empty.
        culled_parts = [
            df for df in (merge_culled, mosaic_culled) if len(df) > 0
        ]
        if culled_parts:
            culled = gpd.GeoDataFrame(
                pd.concat(culled_parts, ignore_index=True), crs=kept.crs
            )
        else:
            culled = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=kept.crs), crs=kept.crs
            )
        return kept, culled

    merged = merge_duplicates(
        stitched,
        iou_threshold=dedup_iou_threshold,
        containment_threshold=dedup_containment_threshold,
        verbose=verbose,
    )
    kept = resolve_mosaic(
        merged, dust_area=mosaic_dust_area_m2, verbose=verbose,
    )
    return kept


# Backwards-compat shim: keep the old name pointing to the new pipeline so
# existing callers / tests that imported `clean_crowns` keep working. The
# semantics changed (now union-merge + mosaic-clip instead of keep-largest),
# so callers should migrate to the explicit names.
def clean_crowns(
    crowns_gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.7,
    containment_threshold: float = 0.92,
    area_threshold: float = 0.0,
    verbose: bool = True,
    return_culled: bool = False,
    mosaic_dust_area: float = 0.1,
):
    """Legacy entry point. Runs merge_duplicates followed by resolve_mosaic.
    Prefer calling those two directly in new code."""
    if area_threshold > 0:
        crowns_gdf = crowns_gdf[crowns_gdf.geometry.area > area_threshold]

    if return_culled:
        merged, m_culled = merge_duplicates(
            crowns_gdf,
            iou_threshold=iou_threshold,
            containment_threshold=containment_threshold,
            verbose=verbose,
            return_culled=True,
        )
        kept, mo_culled = resolve_mosaic(
            merged, dust_area=mosaic_dust_area, verbose=verbose, return_culled=True
        )
        culled_parts = [df for df in (m_culled, mo_culled) if len(df) > 0]
        if culled_parts:
            culled = gpd.GeoDataFrame(
                pd.concat(culled_parts, ignore_index=True), crs=kept.crs
            )
        else:
            culled = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries([], crs=kept.crs), crs=kept.crs
            )
        return kept, culled

    merged = merge_duplicates(
        crowns_gdf,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
        verbose=verbose,
    )
    return resolve_mosaic(merged, dust_area=mosaic_dust_area, verbose=verbose)
