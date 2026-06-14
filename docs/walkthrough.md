# Outline & walkthrough

[← back to README](../README.md)

This document is the detailed companion to the README. It contains:

1. **[Detailed outline](#detailed-outline)** — a chart of every stage and
   the modules that implement it.
2. **[Repository layout](#repository-layout)** — an annotated tree of the script
   files.
3. **[Walkthrough](#walkthrough)** — how to prepare data, set up, and run each
   runner, with the decisions and gotchas that matter.

For the full per-knob reference see [configuration.md](configuration.md); for the
rationale behind the design see [design.md](design.md).

## Detailed outline

```
INPUTS
  ortho mosaic ............ VRT / GeoTIFF, RGB-IR, any size, EPSG:3007
  semantic raster ......... AF_3 output of Gbg-SEM-TreeSeg
                            uint8: 0=bg, 1=water, 2=tree, 255=nodata
  Cellpose checkpoint ..... 'cpsam' default, or a fine-tuned one from FG_1
        │
        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ FG_2   utils.inference.run_pipeline(InferenceConfig)                          │
│                                                                               │
│  1. tile grid ............ tiling.py            1024 px tiles, 256 px overlap │
│                                                                               │
│  2. per-tile inference ... instance_stitching.py                              │
│       • read ortho window + semantic window (the canopy mask)                 │
│       • skip tiles with no `tree` pixels (semantic prior gate)                │
│       • select 1–3 raw bands → uint8 percentile stretch ... bands.py          │
│       • Cellpose-SAM → instance mask → polygons (rasterio.features.shapes)    │
│       • stream features → GeoParquet shards ............... feature_store.py  │
│                                                                               │
│  3. chunked postprocess .. chunked_postprocess.py   (super-tiles, parallel)   │
│       for each chunk (core + buffer) read only intersecting shards, then      │
│       three vector passes ............................... crown_postprocess.py│
│         A  stitch ............. drop crowns clipped at an internal seam       │
│         B  merge_duplicates ... union IoU / containment-overlapping clusters  │
│         C  resolve_mosaic ..... clip overlaps, keep by area, drop dust        │
│       emit only crowns whose centroid lies in the chunk core (no double-emit) │
│                                                                               │
│  4. assign instance_id, tag image-edge crowns, write GPKG ... vector_export.py│
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUTS   Project_1/FG_2_inference_results/
  <stem>_crowns.gpkg ......... instance_id, area_px, area_m2,
                               centroid_x, centroid_y, touches_image_edge, geometry
  <stem>_tile_grid.gpkg ...... tile rectangles + per-side image-edge flags (sidecar)
  [<stem>_dedup_culled.gpkg] . removed / clipped geometry      (if WRITE_DEDUP_CULLED)
  [<stem>_per_tile_features/]  GeoParquet shards               (if KEEP_FEATURE_SHARDS)

FOLLOW-UPS
  FG_3 ............ las_filter: p99 LiDAR height per crown → flag/drop no-canopy detections
  tools/FG_2a ..... resume the chunked postprocess from the per-tile shards (after OOM)
  tools/FG_2b ..... render a QA PNG overlay for a fast sanity check
```

The optional fine-tuning sub-pipeline that produces the checkpoint:

```
FG_0  preprocess_train.py   ortho + per-tree crown GPKG ─► labelled 512 px tiles
                            (Project_1/FG_0_training_data/{train,val,review}/)
        │
        ▼
FG_1  train_cellpose.py     fine-tune 'cpsam' ─► checkpoint + live loss/metric monitors
                            (Project_1/FG_1_training_runs/checkpoints/…)
        │
        └─► set as CELLPOSE_CHECKPOINT in FG_2
```

## Repository layout

```
gbg_inst_treeseg/                   ← run every runner from here
├── FG_0_preprocess_runner.py         build Cellpose training tiles from a per-tree crown GPKG
├── FG_1_train_runner.py              fine-tune Cellpose-SAM on the FG_0 tiles
├── FG_2_inference_runner.py          full instance inference — the headline runner
├── FG_3_las_filter_runner.py         LiDAR p99 height filter on the FG_2 crowns
├── project_paths.py                  directory layout — single source of truth (set PROJECT_NAME)
├── local_paths.example.py            template for machine-specific paths → copy to local_paths.py
├── environment.yml                   conda environment spec
├── CITATION.cff                      citation metadata
├── LICENSE                           GNU AGPL-3.0-or-later
├── tools/
│   ├── FG_2a_postprocess_resume.py   resume FG_2's postprocess from feature shards (after OOM)
│   └── FG_2b_qa_preview.py           render a quick PNG overlay of FG_2 output
├── utils/                         pipeline implementation
│   ├── config.py                  InferenceConfig & TrainConfig dataclasses (every knob, documented)
│   ├── inference.py               FG_2 orchestrator — run_pipeline()
│   ├── tiling.py                  tile-grid generation
│   ├── instance_stitching.py      tiled Cellpose-SAM + per-tile polygon emission
│   ├── bands.py                   band selection + per-band 2–98 % uint8 stretch
│   ├── feature_store.py           GeoParquet streaming shard I/O (bounded RAM)
│   ├── chunked_postprocess.py     memory-bounded, parallel super-tile postprocess
│   ├── crown_postprocess.py       three-pass vector reconciliation (stitch / merge / mosaic)
│   ├── vector_export.py           GPKG I/O via pyogrio
│   ├── preprocess_train.py        FG_0: ortho tiling + crown labelling
│   ├── train_cellpose.py          FG_1: Cellpose-SAM fine-tuning wrapper
│   ├── training_monitor.py        live loss/metric CSV + PNG during training
│   ├── photometric_aug.py         albumentations photometric augmentation (shadow, CLAHE, …)
│   ├── aoi_picker.py              Leaflet AOI picker for selecting a test subset of a big VRT
│   └── progress.py                progress-bar / formatting helpers
├── docs/                          configuration.md · design.md · limitations.md · walkthrough.md
└── Project_1/                     project data, gitignored (per-stage dirs defined in project_paths.py)
```

## Walkthrough

### Before you start

1. **Create the environment** (once):

   ```bash
   conda env create -f environment.yml
   conda activate gbg_inst_treeseg
   ```

   The `environment.yml` defaults to a CUDA PyTorch build — edit `pytorch-cuda`
   to match your driver, or drop it and add `cpuonly` for CPU-only inference.

2. **Set your machine-specific paths.** `local_paths.py` is **gitignored** so
   absolute paths are never committed:

   ```bash
   cp local_paths.example.py local_paths.py
   # edit: ORTHO_PATH, GBG_SEM_ROOT, LAS_DIR, TIF_DIR
   ```

   - `ORTHO_PATH` — the RGB-IR ortho mosaic (a `.vrt` over many tiles is fine).
   - `GBG_SEM_ROOT` — the Gbg-SEM-TreeSeg working copy, so FG_2/FG_0 can reach
     its AF_3 segmentation raster and per-tree crown labels.
   - `LAS_DIR`, `TIF_DIR` — only needed for the FG_3 LiDAR filter.

   `project_paths.py` reads these and raises a clear error if `local_paths.py` is
   missing. To start a fresh project, change `PROJECT_NAME` there.

> **Two-environment workflow.** The semantic raster is produced by AF_3 in the
> **Gbg-SEM-TreeSeg / AerialFormer** environment (it pins old mmcv/mmseg
> binaries). This repo runs in its own modern env. Produce the raster once over
> there, then point this pipeline at it — there is no live coupling between the
> two environments.

### A. Inference (FG_2) — the main path

This is the path most users want: ortho + semantic raster → crown polygons.

1. **Produce the semantic raster** (Gbg-SEM-TreeSeg, AerialFormer env). Run its
   `AF_3_inference_runner.py` on the same ortho. The output is a single-band
   uint8 GeoTIFF of class indices (`0=bg, 1=water, 2=tree, 255=nodata`).

2. **Point FG_2 at its inputs.** Edit the constants at the top of
   `FG_2_inference_runner.py`:

   | Constant | What to set | Notes / decisions |
   |---|---|---|
   | `ORTHO_PATH` | the ortho mosaic | inherited from `local_paths.py` by default |
   | `SEMANTIC_RASTER_PATH` | the AF_3 output | **must be pixel-aligned** with the ortho (same grid, CRS, extent) — it is, if AF_3 ran on the same input |
   | `TREE_CLASS_ID` | `2` | the class index that means *canopy* in the semantic raster. `2` matches Gbg-SEM-TreeSeg's `bg/water/tree`. Change only if your raster uses a different scheme |
   | `BANDS` | `[1, 2, 3]` | which ortho bands feed Cellpose. `[1,2,3]` = true-colour RGB (closest to cpsam pretraining); `[4,1,2]` = NIR-R-G, a strong alternative for vegetation |
   | `CELLPOSE_CHECKPOINT` | `'cpsam'` or a path | leave as the default `'cpsam'`, or use a checkpoint from FG_1 |

3. **Run it:**

   ```bash
   python FG_2_inference_runner.py
   ```

**Knobs you usually leave alone** (full tables in
[configuration.md](configuration.md)):

- *Reconciliation* — `STITCH_SHIFT_M`, `DEDUP_IOU_THRESHOLD`,
  `DEDUP_CONTAINMENT_THRESHOLD`, `MOSAIC_DUST_AREA_M2`. The defaults suit
  Gothenburg crowns (~50–100 px). Touch them only if the `dedup_culled`
  diagnostic shows over- or under-merging.
- *Scaling* — for **city-scale** mosaics, raise `N_POSTPROCESS_WORKERS` toward
  your physical core count and check `CHUNK_BUFFER_M` is **≥ the largest
  realistic crown radius** (ideally 2×). `CHUNK_SIZE_M` trades per-chunk memory
  against overhead. A small reference ortho runs as a single chunk with no tuning.
- *Diagnostics* — set `WRITE_DEDUP_CULLED=True` to emit the removed/clipped
  geometry, and `KEEP_FEATURE_SHARDS=True` to retain the per-tile GeoParquet so
  you can re-run **only** the postprocess (via `tools/FG_2a_…`) with different
  reconciliation params without re-running Cellpose.
- *Image edges* — crowns at the ortho boundary are kept and tagged
  `touches_image_edge=True`. Set `DROP_IMAGE_EDGE_CROWNS=True` for a strict
  "complete crowns only" output.
- *Test subset* — on a huge VRT, set `TEST_SUBSET_PICKER=True` to open a Leaflet
  map (needs the Qt deps) and draw an AOI to run on instead of the whole mosaic.

**Outputs** land in `Project_1/FG_2_inference_results/` — see the README for the
schema. Quick checks afterwards:

- `tools/FG_2b_qa_preview.py` — render a PNG overlay before opening QGIS.
- `tools/FG_2a_postprocess_resume.py` — if a large run was OOM-killed *during
  postprocess*, resume from the shards (requires `KEEP_FEATURE_SHARDS=True`).

### B. Fine-tuning (FG_0 → FG_1) — optional

The default `cpsam` checkpoint already works; fine-tune only to adapt Cellpose to
your imagery.

1. **Build training tiles (FG_0).** Put the per-tree crown GeoPackage and ortho
   into `Project_1/FG_0_training/` (these come from Gbg-SEM-TreeSeg's
   `AF_0_training/`), then edit `FG_0_preprocess_runner.py`:

   - `TRAINING_SOURCES` — one entry per (ortho, crowns) pair.
   - `VAL_BBOX_FILTER` — optional bbox carving a spatial validation split.
   - `QC_FIELD` / `QC_PASS_VALUES` — if your GeoPackage has a quality flag, tiles
     overlapping a QC-failed crown are routed to `review/` instead of
     `train`/`val` (set `QC_REVIEW_SUBDIR=None` to drop them).

   ```bash
   python FG_0_preprocess_runner.py
   ```

2. **Fine-tune (FG_1).** Edit `FG_1_train_runner.py`:

   - `N_EPOCHS`, `LEARNING_RATE`, batch size — the usual training knobs.
   - `MONITOR_INTERVAL_EPOCHS` — how often the loss/metric CSV + PNGs refresh.
   - `EARLY_STOPPING_PATIENCE` — stop if val loss stops improving;
     `OVERFIT_WARN_WINDOW` logs a warning (but keeps training) if train/val
     curves diverge.

   ```bash
   python FG_1_train_runner.py
   ```

   Watch `Project_1/FG_1_training_runs/checkpoints/*_losses.png` live.

3. **Use the checkpoint.** Set `CELLPOSE_CHECKPOINT` in `FG_2_inference_runner.py`
   to the path FG_1 printed, then re-run inference (path A).

### C. Post-inference LiDAR filter (FG_3) — optional

`FG_3_las_filter_runner.py` cross-checks each crown against the LiDAR point cloud
and flags detections sitting on no real canopy height.

- Set `LAS_DIR` and `TIF_DIR` in `local_paths.py`.
- The runner computes a normalized p99 height per crown; tune the height
  threshold and choose whether to **flag** (add an attribute) or **drop** crowns
  below it.

```bash
python FG_3_las_filter_runner.py
```

Output: `Project_1/FG_3_las_filtered/<stem>_crowns.gpkg`.
