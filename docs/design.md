# Design decisions & scaling

[← back to README](../README.md)

## Design decisions

### 1. Precomputed semantic raster instead of an internal AerialFormer call

The instance pipeline takes the AerialFormer output as a **file path**
rather than running mmsegmentation internally. Reasons:

- Decoupling from mmcv / mmsegmentation lets `gbg_inst_treeseg` import cleanly
  in environments where mmseg isn't available (e.g. for unit tests).
- The semantic raster is the natural unit of caching. Iterating on Cellpose
  parameters, bands, or fine-tuned checkpoints is fast because the heavy
  semantic stage isn't repeated.
- AF_3 already supports VRT mosaics, blended overlap, and threaded I/O —
  duplicating that here would be both redundant and a maintenance burden.

### 2. Direct band selection (no NDVI/derived indices)

Cellpose-SAM's ViT-H encoder takes a 3-channel image natively — there's no
"cytoplasm + nucleus" pairing like in the older Cellpose 2/3 U-Nets, and
the legacy `channels=[…]` argument is silently ignored when the `cpsam`
checkpoint is loaded. So the right framing for an RGB-IR ortho is "pick
three bands and feed them to the encoder."

The user selects 1–3 raw 1-based band indices; `utils/bands.py` handles
uint8 conversion (per-band 2–98 % percentile stretch) and replicates when
fewer than three are given so the encoder always sees `(H, W, 3)`. Default
`[1, 2, 3]` (true colour RGB) is closest to cpsam's pretraining; `[4, 1, 2]`
(NRG: NIR-R-G) is a strong alternative for vegetation work.

Derived indices (NDVI, GLI, GRVI) are deliberately deferred — they'd add a
normalisation layer that has to be identical between training and inference,
and the percentile-stretched raw bands already retain the vegetation
contrast Cellpose cares about.

### 3. Pure vector cross-tile reconciliation

FG-TreeSeg's reference implementation processes a single in-memory image.
For city-scale orthophotos that's not viable. We tile at 1024 px with 256 px
overlap (larger than a typical Gothenburg crown ~50–100 px) so any crown
that crosses a seam is fully visible in at least one tile.

Each tile's Cellpose mask is polygonised **immediately** in the tile's
window-aware affine transform; cross-tile reconciliation runs in vector
space rather than via a global uint32 raster. This makes the merge
order-agnostic (no "first-writer-wins" asymmetry tied to row-major tile
order) and decouples the per-tile inference from the cross-tile bookkeeping.

The reconciliation is three sequential passes in `utils/crown_postprocess.py`:

**A — `stitch_crowns_from_grid`.** For each tile, shrink the tile's bounds
inward by `stitch_shift_m` on sides that face another tile. Keep only
crowns whose geometry lies fully within the shrunk box. Crowns clipped at
an internal seam fail this test and are dropped; the same crown observed
whole in the neighbouring tile (where the seam is on the opposite side of
the overlap region) survives. Sides that face the ortho's outer boundary
are *not* shrunk, so image-edge crowns are preserved.

**B — `merge_duplicates`.** Cluster polygons whose IoU exceeds
`dedup_iou_threshold` *or* whose containment ratio (intersection / smaller
area) exceeds `dedup_containment_threshold`, and replace each cluster with
the **union** of its members (not just the largest). The union preserves
any unique area from any cluster member — a previous keep-largest-drop-rest
rule silently lost that area, creating gaps.

**C — `resolve_mosaic`.** Sort survivors by area descending; for each
polygon, subtract the running union of already-kept polygons from its
geometry. If the clipped remnant has area above `mosaic_dust_area_m2`, keep
it. This guarantees:

- *Single ownership*: every pixel is in at most one output polygon.
- *No spurious gaps*: every pixel covered by an input polygon is covered
  by some output polygon, except for sub-dust slivers.

### 4. Memory-bounded, parallel postprocess

For very large mosaics (city-scale; up to 218 750 × 250 000 px is supported
via VRT), the simple "load all per-tile features, run reconciliation
globally" path runs into both memory and CPU walls. The pipeline now scales
via four cooperating mechanisms (recent revisions referred to internally as
options A, B, C and D):

**(C) GeoParquet streaming** — `utils/instance_stitching.run_inference`
streams each tile's polygonised features to a directory of GeoParquet
shards (`<stem>_per_tile_features/shard_NNNNNN.parquet`). Memory is bound
to one batch (~5000 features / ~5 MB) at a time, regardless of AOI size.
A `_shard_index.parquet` records each shard's bbox so downstream readers
can spatial-filter without touching every shard.
*Implementation: `utils/feature_store.py`.*

**(A) Spatial chunking of the postprocess** — `utils/chunked_postprocess.py`
splits the AOI into super-tiles (`CHUNK_SIZE_M`) with a buffer band
(`CHUNK_BUFFER_M`) on each internal side. For each super-tile, it reads
only the per-tile features intersecting its (core + buffer) bbox via the
GeoParquet spatial filter and runs the same three-pass reconciliation
locally. Each chunk emits only polygons whose *centroid* lies in its core
(half-open intervals avoid double-emit at seams), so every output polygon
is produced by exactly one chunk — no cross-chunk dedup pass needed,
provided the buffer is wider than the largest realistic crown radius.
Chunks are independent; with `N_POSTPROCESS_WORKERS > 1` a
`ProcessPoolExecutor` runs them in parallel.

**(B) Spatial index in `resolve_mosaic`** — the candidate query inside
the greedy mosaic resolution uses an incremental `rtree.index.Index`
instead of a Python-list bbox check. O(log n) candidate query per polygon
instead of O(n); critical for chunks with >10⁴ crowns.

**(D) `pyogrio` engine** — every `.to_file(... GPKG ...)` call passes
`engine="pyogrio"` for ~5–10× faster GeoPackage I/O.

For our 6250×6250 px reference ortho the pipeline runs as a single chunk
with no overhead beyond a parquet round-trip. For a 1400× larger AOI
(≈ 218 750 × 250 000 px, ~6 M crowns), the chunked + parallel path keeps
peak RAM at chunk-size and finishes the postprocess in roughly
`n_chunks × ~5 s / n_workers`.

### 5. GeoPackage as the canonical output, plus sidecar diagnostics

The user's chosen final product is a GeoPackage of polygons. Each instance
is either a `Polygon` or a `MultiPolygon` (the latter when mosaic clipping
leaves a polygon in disconnected pieces), with attributes `instance_id`,
`area_px`, `area_m2`, `centroid_x`, `centroid_y`, and `touches_image_edge`.
Sequential `instance_id` (1..N) is assigned at the orchestrator layer
immediately after reconciliation, so downstream layers (diagnostics) can
join on it.

A sidecar `<stem>_tile_grid.gpkg` is always written: GeoDataFrame of tile
rectangles in the ortho CRS, with `is_image_edge_{n,s,e,w}` boolean flags.
Useful for QGIS overlay and debugging.

There is no intermediate uint32 instance raster on disk anymore. If a
raster representation is needed downstream, rasterise the final GeoPackage
with `rasterio.features.rasterize`.

### 6. Image-edge crowns: keep + tag (configurable cull)

A crown that lies at the ortho's outer boundary cannot be made whole — no
neighbouring tile exists past the boundary. Rather than dropping it, we
preserve the partial polygon and tag `touches_image_edge=True`, letting
the user filter downstream. The `drop_image_edge_crowns` config flag
flips behaviour to a strict "complete crowns only" output.
Image-boundary sides are not shrunk during the internal-seam stitch
(pass A), so image-edge crowns survive the reconciliation gauntlet by
construction.

### 7. Cellpose 4.x API

The installed Cellpose is 4.1.2-dev, where the default checkpoint key is
`'cpsam'` (Cellpose-SAM ViT-H). The older `model_type='vit_h'` /
`channels=[2,3]` API from the FG-TreeSeg paper is replaced with implicit
3-channel input. The pipeline produces 3-channel uint8 arrays via
`utils/bands.select_bands` and lets Cellpose pick its own normalisation.

---

## Scaling

| Pass | At 6250² px (reference) | At 218 750 × 250 000 px (target) |
|------|------------------------|----------------------------------|
| Per-tile Cellpose | 64 tiles · 2 min on RTX 4090 | ~93 000 tiles · ~2 days single-GPU; trivially parallel across GPUs |
| Feature accumulation | Streamed to GeoParquet — RAM bound to one batch | Same |
| Per-chunk reconciliation | 1 chunk · ~4 s | ~55 000 chunks at 1024 m. ~5 s/chunk single-process; ~1–2 hours on 8 cores |
| Final GPKG write | ~1 s (pyogrio) | ~10 min |

Memory: peak per worker is ~chunk_size × crown_density; for 1024 m chunks
in dense canopy, ~few hundred MB.

Buffer sizing rule of thumb: ≥ 2 × max realistic crown diameter. Wider is
safe and cheap; the buffer band is a few percent of total work for a
typical 1 km super-tile.
