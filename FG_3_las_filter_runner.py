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
FG_3 — LiDAR height filter: drop FG_2 crowns without real canopy under them.

For every crown polygon from FG_2, samples the LAS point cloud, computes the
p99 normalized height (z above the DTM-derived ground) of in-polygon points,
and writes a copy of the crowns GPKG with `p99_height`, `pct_above`, and
`passed_height_filter` columns. Crowns whose share of points above
HEIGHT_THRESHOLD is below MIN_PCT_ABOVE are flagged as failed (e.g. shadows,
bushes, segmentation noise on grass).

Edit the constants below and run:

    python FG_3_las_filter_runner.py

Memory-safe streaming rewrite of the earlier test_las_mt.py (kept as scratch).
Three landmines the original hits on the 8.5M-crown / 23 GB GPKG:
 1. gpd.read_file(gpkg_path) tries to materialize all polygons in RAM.
 2. ProcessPoolExecutor(initargs=(trees_gdf,)) pickles a copy per worker.
 3. tree_z_accum keeps every per-tree z value in RAM until the final loop.

This script avoids all three:
 - Workers do per-tile bbox queries against the GPKG (R-Tree backed),
   using the GPKG's persistent `fid` as the stable tree_id.
 - Each worker writes a small (tree_id, z_norm) Parquet shard per tile.
 - A repartition pass hash-buckets the shards by fid; each bucket is then
   small enough to load and groupby in pandas for an exact p99.
 - The output GPKG is written in fid-range batches; the source GPKG is
   never fully resident.
"""

import os
import glob
import shutil
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pads
import rasterio
from rasterio.features import rasterize
from rasterio.fill import fillnodata
import laspy
import pyogrio
import concurrent.futures

import project_paths as paths


SHARD_SCHEMA = pa.schema([("tree_id", pa.int32()), ("z_norm", pa.float32())])


# =============================================================================
# Edit these for your run
# =============================================================================

# --- External inputs (outside this repo) -------------------------------------
# Machine-specific; set in local_paths.py (see local_paths.example.py).

LAS_DIR = paths.LAS_DIR
"""Directory of LAS tiles (Lantmäteriet laser scan), one per ortho tile."""

TIF_DIR = paths.TIF_DIR
"""Directory of ortho GeoTIFF tiles. Each LAS tile is paired with the TIF of
the same stem; the TIF grid defines the per-tile DTM rasterization."""

# --- Pipeline paths (inside this repo, dirs from project_paths.py) -----------

GPKG_PATH = os.path.join(
    paths.FG_2_INFERENCE_RESULTS_DIR, "_mosaik_2026_05_13__0829_crowns.gpkg"
)
"""Crowns GeoPackage produced by FG_2 (or tools/FG_2a_postprocess_resume.py)."""

OUT_GPKG = os.path.join(
    paths.FG_3_LAS_FILTERED_DIR, "_mosaik_2026_05_13__0829_crowns_filtered.gpkg"
)

WORK_DIR = os.path.join(paths.FG_3_LAS_FILTERED_DIR, "_las_streaming_work")
"""Scratch dir for Parquet shards/buckets and the stats checkpoint. A finished
stats.parquet here lets a re-run skip straight to the GPKG write."""

# --- Filter parameters ---------------------------------------------------------

HEIGHT_THRESHOLD = 3.0    # m above ground a point must be to count as canopy
MIN_PCT_ABOVE = 0.30      # min fraction of in-crown points above the threshold
MAX_WORKERS = 12
N_BUCKETS = 64
KEEP_INTERMEDIATE = False


# ---------------------------------------------------------
# Stage 1 — per-tile worker (streamed LAS via chunk_iterator)
# ---------------------------------------------------------
def process_tile(las_path, tif_path, gpkg_path, shard_path,
                 chunk_size=2_000_000):
    stem = Path(las_path).stem
    if os.path.exists(shard_path):
        return stem, "skipped"

    with rasterio.open(tif_path) as src:
        transform = src.transform
        width, height = src.width, src.height
        b = src.bounds
    inv = ~transform

    trees = pyogrio.read_dataframe(
        gpkg_path,
        bbox=(b.left, b.bottom, b.right, b.top),
        columns=[],
        fid_as_index=True,
    )

    tmp_path = shard_path + ".tmp"

    def _write_empty_and_return(status):
        pq.write_table(
            pa.table(
                {"tree_id": np.empty(0, np.int32),
                 "z_norm": np.empty(0, np.float32)},
                schema=SHARD_SCHEMA,
            ),
            tmp_path,
            compression="zstd",
        )
        os.replace(tmp_path, shard_path)
        return stem, status

    if trees.empty:
        return _write_empty_and_return("empty")

    fids = trees.index.to_numpy().astype(np.int32, copy=False)
    tree_mask = rasterize(
        ((geom, int(fid)) for geom, fid in zip(trees.geometry, fids)),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
    )
    del trees, fids

    # --- Pass 1: stream LAS, build DTM (per-cell min z of ground points) ----
    dtm_flat = np.full(height * width, np.inf, dtype=np.float32)
    saw_ground = False
    with laspy.open(las_path) as reader:
        for pts in reader.chunk_iterator(chunk_size):
            cls = np.asarray(pts.classification)
            gmask = cls == 2
            if not gmask.any():
                continue
            gx = np.asarray(pts.x)[gmask]
            gy = np.asarray(pts.y)[gmask]
            gz = np.asarray(pts.z, dtype=np.float32)[gmask]
            gc, gr = inv * (gx, gy)
            gc = np.floor(gc).astype(np.int32)
            gr = np.floor(gr).astype(np.int32)
            ok = (gc >= 0) & (gc < width) & (gr >= 0) & (gr < height)
            gc, gr, gz = gc[ok], gr[ok], gz[ok]
            if gc.size == 0:
                continue
            flat = gr.astype(np.int64) * width + gc.astype(np.int64)
            df = pd.DataFrame({"f": flat, "z": gz})
            mins = df.groupby("f", sort=False)["z"].min()
            np.minimum.at(dtm_flat, mins.index.to_numpy(), mins.to_numpy())
            saw_ground = True

    if not saw_ground:
        return _write_empty_and_return("no_ground")

    dtm = dtm_flat.reshape(height, width)
    dtm[dtm == np.inf] = np.nan
    dtm = fillnodata(dtm, mask=~np.isnan(dtm), max_search_distance=100.0)
    del dtm_flat

    # --- Pass 2: stream LAS again, emit in-canopy (tree_id, z_norm) ----
    writer = pq.ParquetWriter(tmp_path, SHARD_SCHEMA, compression="zstd")
    wrote_any = False
    n_pts = 0
    try:
        with laspy.open(las_path) as reader:
            for pts in reader.chunk_iterator(chunk_size):
                cls = np.asarray(pts.classification)
                ng = cls != 2
                if not ng.any():
                    continue
                nx = np.asarray(pts.x)[ng]
                ny = np.asarray(pts.y)[ng]
                nz = np.asarray(pts.z, dtype=np.float32)[ng]
                c, r = inv * (nx, ny)
                c = np.floor(c).astype(np.int32)
                r = np.floor(r).astype(np.int32)
                ok = (c >= 0) & (c < width) & (r >= 0) & (r < height)
                c, r, nz = c[ok], r[ok], nz[ok]
                pid = tree_mask[r, c]
                gz = dtm[r, c]
                keep = (pid > 0) & ~np.isnan(gz)
                if not keep.any():
                    continue
                pid = pid[keep].astype(np.int32, copy=False)
                znorm = (nz[keep] - gz[keep]).astype(np.float32)
                writer.write_table(
                    pa.table({"tree_id": pid, "z_norm": znorm},
                             schema=SHARD_SCHEMA)
                )
                wrote_any = True
                n_pts += pid.size
        if not wrote_any:
            writer.write_table(
                pa.table(
                    {"tree_id": np.empty(0, np.int32),
                     "z_norm": np.empty(0, np.float32)},
                    schema=SHARD_SCHEMA,
                )
            )
    finally:
        writer.close()
    os.replace(tmp_path, shard_path)
    return stem, f"ok({n_pts:,})"


# ---------------------------------------------------------
# Stage 2 — repartition shards into fid-hash buckets
# ---------------------------------------------------------
def repartition_shards(shard_dir, bucket_dir, n_buckets, batch_size=2_000_000):
    os.makedirs(bucket_dir, exist_ok=True)
    bucket_paths = {b: os.path.join(bucket_dir, f"bucket_{b:04d}.parquet")
                    for b in range(n_buckets)}
    writers = {}
    n_rows = 0
    try:
        dataset = pads.dataset(shard_dir, format="parquet", schema=SHARD_SCHEMA)
        for batch in dataset.to_batches(batch_size=batch_size):
            if batch.num_rows == 0:
                continue
            tid = batch.column("tree_id").to_numpy(zero_copy_only=False)
            z = batch.column("z_norm").to_numpy(zero_copy_only=False)
            buckets = (tid % n_buckets).astype(np.int16)
            for b in np.unique(buckets):
                m = buckets == b
                w = writers.get(int(b))
                if w is None:
                    w = pq.ParquetWriter(bucket_paths[int(b)], SHARD_SCHEMA, compression="zstd")
                    writers[int(b)] = w
                w.write_table(
                    pa.table({"tree_id": tid[m], "z_norm": z[m]}, schema=SHARD_SCHEMA)
                )
            n_rows += batch.num_rows
    finally:
        for w in writers.values():
            w.close()
    return n_rows


# ---------------------------------------------------------
# Stage 3 — per-bucket exact aggregation
# ---------------------------------------------------------
def reduce_bucket(bucket_path, height_threshold):
    df = pq.read_table(bucket_path).to_pandas()
    if df.empty:
        return None
    df["above"] = df["z_norm"] > height_threshold
    g = df.groupby("tree_id", sort=False)
    p99 = g["z_norm"].quantile(0.99).rename("p99_height")
    cnt = g.size().rename("n_total")
    nab = g["above"].sum().rename("n_above")
    out = pd.concat([p99, cnt, nab], axis=1).reset_index()
    out["pct_above"] = out["n_above"] / out["n_total"]
    return out


# ---------------------------------------------------------
# Stage 4 — stream source GPKG -> output GPKG with stats
# ---------------------------------------------------------
def write_filtered_gpkg(src_gpkg, dst_gpkg, stats_df, min_pct_above, batch=200_000):
    if os.path.exists(dst_gpkg):
        os.remove(dst_gpkg)

    con = sqlite3.connect(src_gpkg)
    fid_min, fid_max = con.execute(
        "SELECT MIN(fid), MAX(fid) FROM tree_crowns"
    ).fetchone()
    con.close()

    stats_df = stats_df.set_index("tree_id")
    first = True
    written = 0
    passed = 0

    lo = fid_min
    t0 = time.time()
    while lo <= fid_max:
        hi = lo + batch - 1
        gdf = pyogrio.read_dataframe(
            src_gpkg,
            sql=f"SELECT * FROM tree_crowns WHERE fid >= {lo} AND fid <= {hi}",
            sql_dialect="SQLITE",
            fid_as_index=True,
        )
        if gdf.empty:
            lo += batch
            continue

        fids = gdf.index.to_numpy()
        s = stats_df.reindex(fids)
        gdf = gdf.reset_index(drop=True)
        gdf["p99_height"] = s["p99_height"].fillna(0.0).to_numpy(dtype=np.float32)
        gdf["pct_above"] = s["pct_above"].fillna(0.0).to_numpy(dtype=np.float32)
        gdf["n_total"] = s["n_total"].fillna(0).to_numpy(dtype=np.int64)
        gdf["n_above"] = s["n_above"].fillna(0).to_numpy(dtype=np.int64)
        gdf["passed_height_filter"] = gdf["pct_above"] >= min_pct_above

        pyogrio.write_dataframe(
            gdf, dst_gpkg, driver="GPKG", layer="tree_crowns", append=not first,
        )
        first = False
        written += len(gdf)
        passed += int(gdf["passed_height_filter"].sum())
        if written % (batch * 5) == 0 or hi >= fid_max:
            print(f"  wrote {written:,} (passed {passed:,})  fid<={hi}  ({time.time()-t0:.0f}s)")
        lo += batch

    return written, passed


# ---------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------
def run(las_dir, tif_dir, gpkg_path, out_gpkg, work_dir,
        height_threshold=3.0, min_pct_above=0.30,
        max_workers=8, n_buckets=64, keep_intermediate=False):

    shard_dir = os.path.join(work_dir, "shards")
    bucket_dir = os.path.join(work_dir, "buckets")
    stats_path = os.path.join(work_dir, "stats.parquet")
    os.makedirs(work_dir, exist_ok=True)

    if os.path.exists(stats_path):
        print(f"[1-3/4] reusing existing stats checkpoint: {stats_path}")
        stats_df = pd.read_parquet(stats_path)
        print(f"  {len(stats_df):,} trees with >=1 in-canopy point")
    else:
        os.makedirs(shard_dir, exist_ok=True)
        las_files = sorted(
            glob.glob(os.path.join(las_dir, "*.las"))
            + glob.glob(os.path.join(las_dir, "*.laz"))
        )
        tasks = []
        for las in las_files:
            stem = Path(las).stem
            tif = os.path.join(tif_dir, f"{stem}.tif")
            if not os.path.exists(tif):
                continue
            shard = os.path.join(shard_dir, f"{stem}.parquet")
            tasks.append((las, tif, gpkg_path, shard))

        print(f"[1/4] tile inference: {len(tasks)} tiles, {max_workers} workers")
        t0 = time.time()
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_tile, *a) for a in tasks]
            for f in concurrent.futures.as_completed(futures):
                stem, status = f.result()
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] {stem}: {status}  ({time.time()-t0:.0f}s)")

        print(f"[2/4] repartitioning shards into {n_buckets} buckets")
        n_rows = repartition_shards(shard_dir, bucket_dir, n_buckets)
        print(f"  repartitioned {n_rows:,} rows")

        print(f"[3/4] per-bucket exact p99 + counts")
        stats = []
        for b in range(n_buckets):
            bp = os.path.join(bucket_dir, f"bucket_{b:04d}.parquet")
            if not os.path.exists(bp):
                continue
            part = reduce_bucket(bp, height_threshold)
            if part is not None:
                stats.append(part)
            if (b + 1) % 8 == 0 or b == n_buckets - 1:
                print(f"  bucket {b+1}/{n_buckets} done")
        if stats:
            stats_df = pd.concat(stats, ignore_index=True)
        else:
            stats_df = pd.DataFrame(
                columns=["tree_id", "p99_height", "n_total", "n_above", "pct_above"]
            )
        print(f"  {len(stats_df):,} trees have >=1 in-canopy point")
        stats_df.to_parquet(stats_path)
        print(f"  saved stats checkpoint -> {stats_path}")

    print(f"[4/4] streaming output GPKG -> {out_gpkg}")
    written, passed = write_filtered_gpkg(
        gpkg_path, out_gpkg, stats_df, min_pct_above
    )
    print(f"  done: {written:,} written, {passed:,} passed filter "
          f"(threshold {height_threshold} m, min pct {min_pct_above:.0%})")

    if not keep_intermediate:
        shutil.rmtree(shard_dir, ignore_errors=True)
        shutil.rmtree(bucket_dir, ignore_errors=True)


if __name__ == "__main__":
    run(
        las_dir=LAS_DIR,
        tif_dir=TIF_DIR,
        gpkg_path=GPKG_PATH,
        out_gpkg=OUT_GPKG,
        work_dir=WORK_DIR,
        height_threshold=HEIGHT_THRESHOLD,
        min_pct_above=MIN_PCT_ABOVE,
        max_workers=MAX_WORKERS,
        n_buckets=N_BUCKETS,
        keep_intermediate=KEEP_INTERMEDIATE,
    )
