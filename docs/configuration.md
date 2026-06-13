# Configuration & diagnostics

[← back to README](../README.md)

The two dataclasses in `utils/config.py` document every available knob with
inline comments. The runner scripts surface a small subset as module-level
constants so a typical user only edits a handful of lines.

## Inference reconciliation parameters

These knobs control the vector cross-tile reconciliation. The defaults work
for the Gothenburg setup (≈ 0.16 m pixel size, crown diameters 50–100 px) and
are usually fine; tune only if the diagnostic output suggests a problem.

| Knob | Default | What it controls |
|------|---------|------------------|
| `STITCH_SHIFT_M` | `1.0` | Distance (CRS units) by which each Cellpose tile's bounds are shrunk inward on internal sides. Crowns clipped at an internal seam fall outside the shrunk box and are dropped; image-boundary sides are not shrunk, so image-edge crowns are preserved. |
| `DEDUP_IOU_THRESHOLD` | `0.7` | IoU threshold for the duplicate-merge pass. Polygons whose IoU exceeds this are clustered and replaced with the **union** of the cluster — i.e. treated as different views of the same physical crown. Set high so distinct neighbours are not fused. |
| `DEDUP_CONTAINMENT_THRESHOLD` | `0.92` | Containment threshold (intersection / smaller-area) for the merge pass. Catches the nearly-fully-nested case (small inside big). |
| `MOSAIC_DUST_AREA_M2` | `0.1` | After the merge pass, residual overlaps are resolved by clipping smaller polygons against larger ones. Clipped remnants below this area (m²) are dropped as slivers. |

## Postprocess chunking and parallelism

Used by the `chunked_postprocess` orchestrator that bounds RAM and runs
super-tiles independently. See [Scaling](design.md#scaling) for the rationale.

| Knob | Default | What it controls |
|------|---------|------------------|
| `CHUNK_SIZE_M` | `1024.0` | Side length (m) of super-tiles used to chunk the postprocess. AOI is split into a grid of non-overlapping core boxes; reconciliation runs independently per chunk. Larger = less per-chunk overhead; smaller = lower peak memory and more parallel headroom. |
| `CHUNK_BUFFER_M` | `15.0` | Buffer width (m) on each internal chunk side, ensuring polygons centred near a chunk boundary see the same neighbours the global single-mosaic algorithm would have seen. Must be ≥ largest realistic crown radius; ideally 2×. |
| `N_POSTPROCESS_WORKERS` | `1` | Worker processes for the chunked reconciliation. 1 = sequential in-process. >1 = `ProcessPoolExecutor`. Set to ~min(physical_cores, n_chunks) on large mosaics. |
| `KEEP_FEATURE_SHARDS` | `False` | If True, retain the streaming `<stem>_per_tile_features/` GeoParquet directory after the run. Useful for re-running just the postprocess with different reconciliation params (no re-Cellpose). |

## Image-edge handling and diagnostics

| Knob | Default | What it controls |
|------|---------|------------------|
| `DROP_IMAGE_EDGE_CROWNS` | `False` | If True, drop all crowns whose polygon touches the ortho's outer boundary. Default keeps them and tags them with `touches_image_edge=True`. |
| `WRITE_DEDUP_CULLED` | `False` | If True, also write `<stem>_dedup_culled.gpkg` containing every polygon (or piece) removed during reconciliation. |

## QC tile filtering (preprocessing)

If your per-tree GeoPackage carries a quality-control attribute, set
`QC_FIELD` in `FG_0_preprocess_runner.py` to its name. The preprocessor
splits crowns into 'passed' and 'failed' (`QC_PASS_VALUES` lets you specify
the allowed values; if `None`, any non-null value passes), and any tile that
overlaps a single QC-failed crown is routed to `<OUTPUT_DIR>/review/`
instead of `train/` or `val/`. Set `QC_REVIEW_SUBDIR = None` to drop
silently. The counts dict returned by `build_training_tiles` reports
`train`, `val`, `review`, `skipped_no_crowns`, `skipped_filtered`, and
`skipped_qc_failed` so you can audit the run.

## Training monitoring (fine-tuning)

`utils/train_cellpose.py` runs `cellpose.train.train_seg` in chunks of
`MONITOR_INTERVAL_EPOCHS`. After each chunk it appends to a CSV and
regenerates loss/metric PNGs next to the checkpoint, so you can `tail -f`
the CSV or open the PNGs mid-run.

```
Project_1/
├── FG_0_training_data/                 # FG_0 output / FG_1 input
│   ├── train/                          # training tiles
│   ├── val/                            # val tiles, if val_bbox_filter set
│   └── review/                         # QC-failed tiles
└── FG_1_training_runs/                 # FG_1 output (ARTIFACTS_DIR)
    ├── checkpoints/
    │   ├── models/cellpose_treecrown   # final Cellpose checkpoint
    │   │                               # (cellpose inserts the models/ level)
    │   ├── cellpose_treecrown_info.json   # losses, metrics, run config
    │   ├── cellpose_treecrown_losses.csv  # epoch,train_loss,val_loss,ap50,mean_iou,wall_time_s
    │   ├── cellpose_treecrown_losses.png  # train + val curves, refreshed live
    │   └── cellpose_treecrown_metrics.png # AP@0.5 + mean IoU vs epoch
    ├── flow_cache/{train,val}/         # Cellpose's per-mask flow cache (~4 MB/tile)
    └── aug_preview/                    # photometric-aug example pairs
```

Two heuristics also run on chunk boundaries:

- **Overfitting warning** — if train loss has fallen monotonically and val
  loss has risen monotonically across the last `OVERFIT_WARN_WINDOW`
  intervals, a warning is logged. Training continues.
- **Early stopping** — if `EARLY_STOPPING_PATIENCE` is set and val loss
  fails to improve on its best-so-far for that many consecutive intervals,
  the loop breaks out.

---

## Diagnostics — `<stem>_dedup_culled.gpkg`

When `WRITE_DEDUP_CULLED=True`, the runner emits a third GeoPackage
recording every polygon (or piece of polygon) the reconciliation passes
removed. Schema:

| Column | Meaning |
|--------|---------|
| `phase` | `merge_duplicate` (absorbed into a union in pass B), `mosaic_clipped` (the *removed piece* of a polygon that survived in clipped form in pass C), or `mosaic_dropped` (entirely consumed by a larger neighbour in pass C). |
| `winner_instance_id` | `instance_id` of the surviving crown that absorbed/displaced this row. Looked up via spatial-largest-intersection against the final crowns layer, so it is robust to the row-shuffling that happens through the chunked pipeline. Join via `crowns[crowns.instance_id == winner_instance_id]`. |
| `original_area_m2` | The polygon's area before any modification. |
| `removed_area_m2` | Area removed from this polygon (= original area for full drops; clipped portion for partial clips). |
| `tile_id`, `local_instance_id` | Source tile + Cellpose ID, traceable back to the raw per-tile detection. |
| `geometry` | The removed/clipped-away geometry. For `merge_duplicate` and `mosaic_dropped`, the entire original polygon. For `mosaic_clipped`, only the lost piece. |

Open it in QGIS alongside `<stem>_crowns.gpkg` and `<stem>_tile_grid.gpkg`
to see exactly which canopy went where.
