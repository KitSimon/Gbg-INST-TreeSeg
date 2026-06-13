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
FG_2 — full instance inference: ortho + AerialFormer raster → GeoPackage of crowns.

Run AF_3_inference_runner.py in the Gbg-SEM-TreeSeg repo first to produce the
AerialFormer semantic raster, then edit the constants below and run:

    python FG_2_inference_runner.py
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.abspath(os.path.join(_THIS, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import project_paths as paths
from utils.config import InferenceConfig
from utils.inference import run_pipeline


# =============================================================================
# Edit these for your run
# =============================================================================

# --- External inputs (outside this repo, see project_paths.py) ---------------

ORTHO_PATH = paths.ORTHO_PATH
"""Source orthophoto. Either GeoTIFF or VRT. RGB-IR or RGB."""

SEMANTIC_RASTER_PATH = os.path.join(
    paths.GBG_SEM_ROOT,
    "Project_1/AF_3_inference_results/segmentation1/_mosaik_segmentation.tif",
)
"""AerialFormer output produced by AF_3_inference_runner.py in
Gbg-SEM-TreeSeg. Must be aligned to ORTHO_PATH (same width, height,
transform, CRS) — it is, if AF_3 was run on the same input."""

# --- Pipeline paths (inside this repo) ----------------------------------------

CELLPOSE_CHECKPOINT = os.path.join(
    paths.FG_1_TRAINING_RUNS_DIR, "checkpoints", "models", "cellpose_treecrown"
)
"""Path to a fine-tuned Cellpose checkpoint (cellpose itself inserts the
models/ level under checkpoints/), or None to use the built-in 'cpsam'
(Cellpose-SAM ViT-H) checkpoint without fine-tuning."""

OUTPUT_DIR = paths.FG_2_INFERENCE_RESULTS_DIR

# --- Semantic prior interpretation ------------------------------------------

TREE_CLASS_ID = 2
"""Class index in SEMANTIC_RASTER_PATH that represents tree canopy.
Default 2 matches the Gbg-SEM-TreeSeg CLASSES order (bg=0, water=1, tree=2;
255 = nodata)."""

# --- Bands -------------------------------------------------------------------

BAND_INDICES = [1, 2, 3]
"""Flexible band selection — length 1, 2, or 3. Cellpose-SAM consumes a
3-channel image; any 3-index combo gives the encoder full information.
For an RGB-IR ortho (R=1, G=2, B=3, IR=4) good choices are:
    [1, 2, 3]  — true colour RGB (default)
    [4, 1, 2]  — NRG false-colour (vegetation composite; strong NIR-vs-R)
    [4, 1, 3]  — NIR, R, B
    [2, 1]     — paper-style G, R (third channel duplicated; suboptimal)
    [4]        — IR-only, replicated to all 3 channels
Must match the bands you trained Cellpose on, if you fine-tuned."""

# --- Tiling ------------------------------------------------------------------

WINDOW_SIZE = 1024
STRIDE = 768  # 256 px overlap (25%) — should be > expected max crown diameter

# --- Cellpose ----------------------------------------------------------------

DIAMETER = 70.0           # ~50–80 px per the user's setup
FLOW_THRESHOLD = 1.0
CELLPROB_THRESHOLD = -3.0
MIN_SIZE = 50

# --- Vector cross-tile merge -------------------------------------------------

STITCH_SHIFT_M = 1.0
"""Internal-seam shrink (CRS units, typically metres). For each tile, sides
that face another tile are shrunk inward by this much; crowns clipped at the
seam fall outside the shrunk box and are dropped. The neighbouring tile sees
the same crown whole inside its own shrunk box and survives. Sides on the
ortho's outer boundary are NOT shrunk, so image-edge crowns are preserved.
Set roughly equal to the maximum sliver width you want to cull (lower it if
real crowns are getting culled near seams)."""

DEDUP_IOU_THRESHOLD = 0.7
"""Duplicate-merge IoU threshold. Polygons whose IoU exceeds this are merged
into one polygon = the UNION of the cluster (preserving any unique area from
any member, eliminating gaps where the previous keep-largest-drop-rest rule
would have cut). Set high enough that distinct neighbouring crowns are not
fused — 0.7 is a safe baseline."""

DEDUP_CONTAINMENT_THRESHOLD = 0.92
"""Duplicate-merge containment threshold: merge polygons where
intersection / min(area_i, area_j) exceeds this. Catches the
nearly-fully-nested case (small inside big). Set high so a small distinct
neighbour partly poking into a larger crown is not absorbed."""

MOSAIC_DUST_AREA_M2 = 0.1
"""After the merge pass, residual overlaps between distinct neighbouring
crowns are resolved by greedy mosaic clipping: largest polygon keeps its
shape; smaller ones are clipped to the complement of larger neighbours.
Clipped remnants below this area (CRS units squared, m^2 in projected
CRS) are dropped as slivers. 0.1 m^2 = a 10x10 cm patch — well below any
real crown."""

DROP_IMAGE_EDGE_CROWNS = False
"""If True, drop all crowns whose polygon touches the ortho's outer boundary.
Default False — keeps them and tags them with touches_image_edge=True. Switch
on for a strict 'complete crowns only' output."""

# --- Postprocess chunking (memory + parallelism) -----------------------------

CHUNK_SIZE_M = 1000.0
"""Side length (CRS units, typically m) of super-tiles used to chunk the
postprocess. AOI is split into a grid of non-overlapping core boxes of this
size; reconciliation runs independently inside each. Larger = less per-chunk
overhead; smaller = lower peak memory and better parallel utilisation. 1024 m
at 0.16 m px = ~6400 px, matches the 6250x6250 single-mosaic scale we know
fits comfortably."""

CHUNK_BUFFER_M = 40.96
"""Buffer width (m) on each internal chunk side, ensuring polygons centred
near a chunk boundary see the same neighbours the global single-mosaic
algorithm would have seen. Must be >= largest realistic crown radius;
ideally 2x. With ~10 m max crown diameter, 15 m is conservative and cheap."""

N_POSTPROCESS_WORKERS = 6
"""Worker processes for the chunked reconciliation. 1 = sequential in-process.
>1 = ProcessPoolExecutor. Set to ~min(physical_cores, n_chunks) for best
wall-time on large mosaics."""

KEEP_FEATURE_SHARDS = False
"""If True, retain the per-tile GeoParquet feature store after the run.
Useful for debugging / re-running just the postprocess with different
reconciliation params. Default False removes it once the final GPKG is
written."""

WRITE_DEDUP_CULLED = False
"""If True, also write `<stem>_dedup_culled.gpkg` with every polygon removed
by the dedup pass. Each culled polygon carries `winner_idx` (row index into
the cleaned set; join via cleaned.iloc[winner_idx]), `winner_area_m2`, and
`culled_area_m2` for inspection. Use to diagnose over-aggressive dedup —
load alongside the main crowns GPKG in QGIS and look for places where a
real crown was collapsed into a neighbour."""

# --- Output ------------------------------------------------------------------

USE_GPU = True

USE_BFLOAT16 = False
"""Cellpose 4 defaults to bfloat16 model weights for inference. Torch <2.1
does not implement F.interpolate(mode='linear') for bfloat16, which is hit
inside SAM's relative-positional-embedding code path and crashes with
'upsample_linear1d_out_frame not implemented for BFloat16'. Default False
keeps the pipeline working on the local torch 2.0.1 env. On a newer env
(torch >=2.1), setting this to True saves ~50% VRAM and is faster."""

ADD_TIMESTAMP_SUFFIX = True
"""If True, append _YYYY_MM_DD__HHMM to all output filenames so each run
produces a fresh, sortable set instead of overwriting the previous run's
crowns/tile_grid/dedup_culled gpkgs. The same timestamp is shared across
every file produced in a single run. Set False to overwrite in place."""

# --- Test-subset picker ------------------------------------------------------

TEST_SUBSET_PICKER = False
"""If True, open an interactive AOI picker (Qt + Leaflet) before inference,
showing the ortho's footprint over an Esri/OSM basemap. Draw a rectangle to
select a small test subset; the ortho and semantic raster are clipped to
that bbox via gdal_translate -of VRT (no pixel copy) and inference runs on
the subset only. Use for fast iteration on a large VRT — flip back to False
for full production runs."""

TEST_SUBSET_BBOX_OVERRIDE = None
"""Optional (xmin, ymin, xmax, ymax) tuple in the ortho's raster CRS. If set,
takes precedence over the picker — the runner clips to this bbox without
opening any window. Useful for re-running the same test subset
deterministically (the previous run's bbox is saved in
<OUTPUT_DIR>/<TEST_SUBSET_OUT_SUBDIR>/bbox.json)."""

TEST_SUBSET_OUT_SUBDIR = "_test_subset"
"""Subdirectory under OUTPUT_DIR where the clipped VRTs and bbox.json are
written. Contents are overwritten on each test-subset run."""


# =============================================================================
# Run
# =============================================================================

def _maybe_pick_test_subset():
    """If the test-subset toggle (or bbox override) is set, pick/clip and
    return (ortho_path, semantic_raster_path) pointing at clipped VRTs.
    Otherwise return the original full-resolution paths unchanged."""
    if not TEST_SUBSET_PICKER and TEST_SUBSET_BBOX_OVERRIDE is None:
        return ORTHO_PATH, SEMANTIC_RASTER_PATH

    import json as _json
    from utils.aoi_picker import clip_to_bbox_vrt, pick_aoi

    if TEST_SUBSET_BBOX_OVERRIDE is not None:
        bbox = tuple(TEST_SUBSET_BBOX_OVERRIDE)
        print(f"[FG_2] using TEST_SUBSET_BBOX_OVERRIDE: {bbox}")
    else:
        print("[FG_2] launching AOI picker — draw a rectangle to select a test subset")
        bbox = pick_aoi(ORTHO_PATH)
        print(f"[FG_2] AOI picked (ortho CRS): {bbox}")

    subset_dir = os.path.join(OUTPUT_DIR, TEST_SUBSET_OUT_SUBDIR)
    os.makedirs(subset_dir, exist_ok=True)
    ortho_clipped = os.path.join(subset_dir, "ortho_subset.vrt")
    sem_clipped = os.path.join(subset_dir, "semantic_subset.vrt")
    clip_to_bbox_vrt(ORTHO_PATH, ortho_clipped, bbox)
    clip_to_bbox_vrt(SEMANTIC_RASTER_PATH, sem_clipped, bbox)
    with open(os.path.join(subset_dir, "bbox.json"), "w") as f:
        _json.dump({
            "bbox": list(bbox),
            "note": "axis-aligned bbox in the original ortho's raster CRS",
            "source_ortho": ORTHO_PATH,
            "source_semantic_raster": SEMANTIC_RASTER_PATH,
        }, f, indent=2)
    print(f"[FG_2] clipped inputs written to {subset_dir}")
    return ortho_clipped, sem_clipped


def main():
    ortho_path, semantic_raster_path = _maybe_pick_test_subset()

    cfg = InferenceConfig(
        ortho_path=ortho_path,
        semantic_raster_path=semantic_raster_path,
        cellpose_checkpoint=CELLPOSE_CHECKPOINT,
        output_dir=OUTPUT_DIR,
        tree_class_id=TREE_CLASS_ID,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        band_indices=BAND_INDICES,
        diameter=DIAMETER,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
        stitch_shift_m=STITCH_SHIFT_M,
        dedup_iou_threshold=DEDUP_IOU_THRESHOLD,
        dedup_containment_threshold=DEDUP_CONTAINMENT_THRESHOLD,
        mosaic_dust_area_m2=MOSAIC_DUST_AREA_M2,
        drop_image_edge_crowns=DROP_IMAGE_EDGE_CROWNS,
        chunk_size_m=CHUNK_SIZE_M,
        chunk_buffer_m=CHUNK_BUFFER_M,
        n_postprocess_workers=N_POSTPROCESS_WORKERS,
        keep_feature_shards=KEEP_FEATURE_SHARDS,
        write_dedup_culled=WRITE_DEDUP_CULLED,
        use_gpu=USE_GPU,
        use_bfloat16=USE_BFLOAT16,
        add_timestamp_suffix=ADD_TIMESTAMP_SUFFIX,
    )
    paths = run_pipeline(cfg)
    print("Outputs:")
    print(f"  crowns:    {paths['gpkg']}")
    print(f"  tile_grid: {paths['tile_grid']}")
    if "dedup_culled" in paths:
        print(f"  dedup_culled: {paths['dedup_culled']}")


if __name__ == "__main__":
    main()
