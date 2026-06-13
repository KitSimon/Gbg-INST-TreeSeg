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
Write the final tree-crown GeoDataFrame to a GeoPackage.

The polygonisation now happens per-tile inside `instance_stitching.run_inference`
(which uses `rasterio.features.shapes` with a window-aware affine transform),
so this module is only responsible for assembling output columns and writing
the file. It also writes the sidecar tile-grid GeoPackage.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np


def write_crowns_gpkg(
    crowns: gpd.GeoDataFrame,
    output_gpkg: str,
    layer: str = "tree_crowns",
    pixel_area: float = 1.0,
    simplify_tolerance: Optional[float] = None,
) -> str:
    """
    Write final crowns to a GeoPackage with the standard schema.

    Args:
        crowns:             GeoDataFrame of merged crown polygons. May carry a
                            `touches_image_edge` boolean column; if present, it
                            is preserved on the output.
        output_gpkg:        Output path.
        layer:              Layer name inside the GeoPackage.
        pixel_area:         Pixel area in CRS units squared, used to convert
                            `area_m2` -> `area_px`.
        simplify_tolerance: If set, simplify polygons by this distance (CRS
                            units, usually metres). Useful for jagged
                            pixel-staircase edges. None = no simplification.

    Returns:
        The output path.
    """
    crs = crowns.crs

    if len(crowns) == 0:
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
        empty.to_file(output_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
        return output_gpkg

    out = crowns.copy().reset_index(drop=True)

    if simplify_tolerance is not None:
        out["geometry"] = out.geometry.simplify(
            simplify_tolerance, preserve_topology=True
        )

    if "instance_id" not in out.columns:
        # Caller didn't assign IDs — fall back to sequential numbering.
        out["instance_id"] = np.arange(1, len(out) + 1, dtype=np.uint32)
    out["area_m2"] = out.geometry.area
    if pixel_area > 0:
        out["area_px"] = (out["area_m2"] / pixel_area).round().astype(np.int64)
    else:
        out["area_px"] = np.zeros(len(out), dtype=np.int64)
    centroids = out.geometry.centroid
    out["centroid_x"] = centroids.x
    out["centroid_y"] = centroids.y

    if "touches_image_edge" not in out.columns:
        out["touches_image_edge"] = False

    cols = [
        "instance_id",
        "area_px",
        "area_m2",
        "centroid_x",
        "centroid_y",
        "touches_image_edge",
        "geometry",
    ]
    out = gpd.GeoDataFrame(out[cols], crs=crs)
    out.to_file(output_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
    return output_gpkg


def write_tile_grid_gpkg(
    tile_grid: gpd.GeoDataFrame,
    output_gpkg: str,
    layer: str = "tile_grid",
) -> str:
    """Write the tile-grid GeoDataFrame to a sidecar GeoPackage."""
    tile_grid.to_file(output_gpkg, layer=layer, driver="GPKG", engine="pyogrio")
    return output_gpkg
