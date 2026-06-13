# Limitations and further work

[← back to README](../README.md)

1. **Derived indices.** `utils/bands.py` does only direct band selection.
   Adding NDVI / GLI / GRVI would be a one-function change and could
   improve crown separation in mixed-canopy areas.

2. **Crowns wider than the tile overlap.** A crown that genuinely
   straddles a Cellpose-tile seam beyond the overlap region (i.e. so wide
   that no single tile sees it whole) cannot be reconstructed in vector
   space without speculative unions. With current params (1024 px window,
   256 px overlap, ~50–100 px expected crown diameter) this is rare. If it
   shows up, the right fix is to increase tile overlap (lower `STRIDE`).

3. **4-channel Cellpose.** The Cellpose-SAM ViT encoder expects 3 channels,
   so true 4-band ingestion would require modifying the patch embedding.

4. **Per-instance attributes.** Currently we write area + centroid +
   image-edge flag. Easy wins: mean band values per crown, estimated crown
   height from a DSM if available, height-percentile thresholds.

5. **Standalone evaluator.** Training emits AP@0.5 and mean IoU on the val
   set per chunk, but there's no standalone evaluator that scores a trained
   checkpoint against a held-out per-tree polygon set outside the training
   loop, and no panoptic-quality metric.

6. **Diameter auto-estimation.** Cellpose can estimate per-image diameter
   from a sample (`diameter=None`). For mosaics with mixed crown sizes,
   running this estimation on a few sentinel tiles and using the median
   would be more robust than a hard-coded value.

7. **Per-crown confidence score from cellprob.** Cellpose's `eval()`
   returns a per-pixel `cellprob` map (`flows[2]`). Capturing it and
   aggregating per instance would give each crown a `mean_cellprob` /
   `max_cellprob` attribute, and would let `merge_duplicates` pick the
   highest-confidence cluster member's shape on close calls instead of
   falling back to area.
