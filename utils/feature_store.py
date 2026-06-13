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
GeoParquet feature store for per-tile crowns.

The inference loop streams each tile's polygonised Cellpose detections to a
directory of GeoParquet shards via `ParquetShardSink`. Bounded memory: one
batch (default ~5000 features, ~5 MB) is held in RAM at a time, the rest
lives on disk and is read back per super-tile during postprocess.

Reading: `read_shards_in_bbox` filters shards by their bbox metadata before
loading rows, so a chunked postprocess only touches the shards intersecting
its (chunk + buffer) extent. This is the single feature that makes the
pipeline scale beyond a single workstation's RAM.

Schema written to disk per shard:
    tile_id : int64
    local_instance_id : int64
    area_m2 : float64
    geometry : Polygon | MultiPolygon

Layout on disk:
    <stem>_per_tile_features/
        shard_000000.parquet
        shard_000001.parquet
        ...
        _shard_index.parquet     (one row per shard: shard_path, bbox)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import box as shapely_box


_SHARD_INDEX_NAME = "_shard_index.parquet"


class ParquetShardSink:
    """
    Buffered sink that flushes each accumulated batch to its own GeoParquet
    file. The per-shard size is chosen so a few hundred shards cover even
    very large mosaics — small enough that the per-shard read is cheap, big
    enough that we don't hit filesystem overhead.
    """

    def __init__(self, dir_path: str, batch_features: int = 5000):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.batch_features = batch_features

        self._batch: List[gpd.GeoDataFrame] = []
        self._batch_count = 0
        self._shard_idx = 0
        self._index_rows: List[dict] = []
        self._crs = None

    def append(self, gdf: gpd.GeoDataFrame) -> None:
        if len(gdf) == 0:
            return
        if self._crs is None:
            self._crs = gdf.crs
        self._batch.append(gdf)
        self._batch_count += len(gdf)
        if self._batch_count >= self.batch_features:
            self._flush()

    def _flush(self) -> None:
        if not self._batch:
            return
        combined = pd.concat(self._batch, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=self._crs)
        shard_path = self.dir_path / f"shard_{self._shard_idx:06d}.parquet"
        combined.to_parquet(shard_path, index=False)
        bounds = combined.total_bounds  # (minx, miny, maxx, maxy)
        self._index_rows.append(
            {
                "shard_path": shard_path.name,
                "n_features": len(combined),
                "minx": float(bounds[0]),
                "miny": float(bounds[1]),
                "maxx": float(bounds[2]),
                "maxy": float(bounds[3]),
            }
        )
        self._shard_idx += 1
        self._batch = []
        self._batch_count = 0

    def close(self) -> None:
        self._flush()
        if not self._index_rows:
            # Write an empty index so readers can detect "no features".
            pd.DataFrame(
                columns=["shard_path", "n_features", "minx", "miny", "maxx", "maxy"]
            ).to_parquet(self.dir_path / _SHARD_INDEX_NAME, index=False)
            return
        index_df = pd.DataFrame(self._index_rows)
        index_df.to_parquet(self.dir_path / _SHARD_INDEX_NAME, index=False)


def read_shard_index(dir_path: str) -> pd.DataFrame:
    """Read the shard-index table (one row per shard with its bbox)."""
    index_path = Path(dir_path) / _SHARD_INDEX_NAME
    if not index_path.exists():
        return pd.DataFrame(
            columns=["shard_path", "n_features", "minx", "miny", "maxx", "maxy"]
        )
    return pd.read_parquet(index_path)


def read_shards_in_bbox(
    dir_path: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> gpd.GeoDataFrame:
    """
    Read all per-tile features whose source shard's bbox intersects `bbox`.
    `bbox` = (minx, miny, maxx, maxy). When None, reads everything.
    """
    dir_path = Path(dir_path)
    index = read_shard_index(dir_path)

    if len(index) == 0:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))

    if bbox is None:
        candidate_paths = [dir_path / p for p in index["shard_path"]]
    else:
        minx, miny, maxx, maxy = bbox
        m = (
            (index["maxx"] >= minx)
            & (index["minx"] <= maxx)
            & (index["maxy"] >= miny)
            & (index["miny"] <= maxy)
        )
        candidate_paths = [dir_path / p for p in index.loc[m, "shard_path"]]

    if not candidate_paths:
        # No shards intersect — return an empty GDF whose CRS matches the
        # data, by reading the schema from any shard if available.
        if len(index) > 0:
            sample = gpd.read_parquet(dir_path / index.iloc[0]["shard_path"])
            return sample.iloc[0:0]
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([]))

    parts = [gpd.read_parquet(p) for p in candidate_paths]
    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs=parts[0].crs
    )
    if bbox is not None:
        # Final per-row filter: drop polygons whose bbox doesn't intersect.
        gb = combined.bounds
        m = (
            (gb["maxx"] >= bbox[0])
            & (gb["minx"] <= bbox[2])
            & (gb["maxy"] >= bbox[1])
            & (gb["miny"] <= bbox[3])
        )
        combined = combined[m].reset_index(drop=True)
    return combined
