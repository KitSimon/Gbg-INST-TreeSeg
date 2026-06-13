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
Runtime configuration dataclasses for the instance segmentation pipeline.

Two configs:
    InferenceConfig — full-mosaic inference (Stage A raster + ortho → GeoPackage of crowns)
    TrainConfig     — Cellpose fine-tuning (per-tree GeoPackage + ortho → checkpoint)

The runners (FG_*_runner.py) instantiate one of these and pass it through to
the corresponding entrypoint. Keeping the config in a dataclass means the
runner files stay short and the module APIs stay stable.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

BBox = Tuple[float, float, float, float]
ValRegion = Union[BBox, str]
"""val_bbox_filter accepts either a (xmin, ymin, xmax, ymax) tuple in the
source's raster CRS, or a path to a GeoPackage whose features (unioned)
define the validation AOI. Tile-membership is centre-in-region for both."""


@dataclass
class InferenceConfig:
    # --- Inputs ---------------------------------------------------------
    ortho_path: str
    """Path to the source orthophoto. May be a single GeoTIFF or a VRT mosaic."""

    semantic_raster_path: str
    """Path to the AerialFormer semantic raster produced by AF_3_inference-vrt_new.py.
    Must be aligned to the same CRS/extent as ortho_path (the AF_3 output is)."""

    cellpose_checkpoint: Optional[str] = None
    """Path to a Cellpose checkpoint. If None, uses the built-in 'cpsam'
    (Cellpose-SAM ViT-H) checkpoint without fine-tuning. After fine-tuning via
    FG_1, set this to the produced checkpoint."""

    # --- Output ---------------------------------------------------------
    output_dir: str = "."
    """Directory where outputs are written."""

    output_stem: Optional[str] = None
    """Stem for output filenames. If None, derived from the ortho filename."""

    add_timestamp_suffix: bool = True
    """If True, append `_YYYY_MM_DD__HHMM` (computed once at the start of
    run_pipeline) to the output stem so every run produces a fresh, sortable
    set of files instead of overwriting the previous run's outputs.

    Affects: <stem>_crowns.gpkg, <stem>_tile_grid.gpkg,
    <stem>_per_tile_features/, and (when write_dedup_culled is True)
    <stem>_dedup_culled.gpkg. Set False to restore overwriting behaviour."""

    keep_raster: bool = True
    """If True, keep the intermediate uint32 instance raster on disk after the
    GeoPackage is produced. Useful for debugging."""

    # --- Semantic-prior interpretation ---------------------------------
    tree_class_id: int = 3
    """Class index in the AerialFormer raster that represents tree canopy.
    Defaults to 3 (matches the Gothenburg config: bg=0, water=1, building=2, tree=3)."""

    # --- Tiling --------------------------------------------------------
    window_size: int = 1024
    """Inference tile size in pixels."""

    stride: int = 768
    """Stride between tiles. window_size - stride = overlap. Default 256 px overlap
    (25%) is larger than the expected max crown diameter so any seam-crossing
    crown is fully visible in at least one tile."""

    # --- Bands ---------------------------------------------------------
    band_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    """1-based indices into the source raster bands. Length 1, 2, or 3.
    Cellpose-SAM consumes a 3-channel image natively (its ViT-H encoder
    treats all three as independent inputs); bands.py replicates when fewer
    indices are given so the encoder still sees a (H, W, 3) array.

    Band selection is flexible — anything that gives the encoder useful
    contrast works. For an RGB-IR ortho (R=1, G=2, B=3, IR=4):
        [1, 2, 3]  — true colour RGB (default; closest to cpsam's pretraining)
        [4, 1, 2]  — NRG false-colour (NIR, R, G); standard for vegetation work
                     because IR-vs-R is where canopy contrast is strongest
        [4, 1, 3]  — NIR / R / B; keeps blue for shadow/shaded-leaf signal
        [2, 1]     — paper-style G, R (replicated to (G, R, R); third
                     channel is a copy, so this wastes encoder capacity
                     compared to a true 3-band input)
        [4]        — IR-only, replicated to all 3 channels"""

    percentile_clip: Tuple[float, float] = (2.0, 98.0)
    """Per-band percentile stretch applied when converting raw raster values
    to uint8 for Cellpose."""

    # --- Cellpose ------------------------------------------------------
    diameter: float = 70.0
    """Approximate average crown diameter in pixels at the ortho resolution."""

    flow_threshold: float = 1.0
    """Cellpose flow_threshold. Higher = more permissive. Paper uses 1.0."""

    cellprob_threshold: float = -3.0
    """Cellpose cellprob_threshold. Negative values force segmentation in the
    masked canopy region (paper setting)."""

    gaussian_blur_size: int = 3
    """Kernel size for the Gaussian blur applied before Cellpose, to suppress
    leaf-level texture (paper setting). Set to 0 to disable."""

    min_size: int = 50
    """Cellpose min_size: drop tile-local instances below this many pixels."""

    # --- Vector cross-tile merge --------------------------------------
    stitch_shift_m: float = 1.0
    """Shrink distance (CRS units, typically metres) applied to each tile's
    bounds on internal sides during vector stitching. A crown whose extent
    falls outside the shrunk box is treated as cropped at an internal seam
    and dropped — the overlap region should provide a complete copy in the
    neighbouring tile. Sides that face the ortho's outer boundary are NOT
    shrunk, so image-edge crowns are preserved (see drop_image_edge_crowns
    for an explicit cull)."""

    dedup_iou_threshold: float = 0.7
    """IoU threshold for the duplicate-merge pass. Polygons whose IoU exceeds
    this are clustered and replaced with the **union** of the cluster — i.e.
    treated as different observations of the same physical crown. Set HIGH
    (0.7+) so that distinct neighbouring crowns are not wrongly fused."""

    dedup_containment_threshold: float = 0.92
    """Containment threshold for the duplicate-merge pass: cluster polygons
    where intersection / min(area_i, area_j) exceeds this. Catches the
    cropped-vs-whole case where IoU is dragged down by extra area. Set HIGH
    (0.9+) so that nested neighbours are not fused with their containers."""

    mosaic_dust_area_m2: float = 0.1
    """After the merge pass, surviving polygons can still spatially overlap.
    The mosaic resolution step sorts polygons by area descending and clips
    each polygon by the union of larger ones already kept. This area
    threshold (CRS units squared) is the minimum area a clipped remnant
    must have to survive — anything smaller is treated as a sliver and
    dropped. Set near zero to keep all canopy pixels covered; raise to
    suppress sliver noise from imperfect Cellpose boundaries."""

    drop_image_edge_crowns: bool = False
    """If True, cull all crowns whose polygon touches the ortho's outer
    boundary in the final output. Default False keeps them and tags them
    with touches_image_edge=True so users can filter in QGIS."""

    # --- Postprocess chunking (memory + parallelism) -------------------
    chunk_size_m: float = 1024.0
    """Side length (CRS units, typically m) of super-tiles used to chunk
    the postprocess. The AOI is split into a grid of non-overlapping core
    boxes of this size; reconciliation runs independently inside each. Set
    larger to reduce per-chunk overhead, smaller to bound peak memory or
    spread work across more workers. 1024 m at 0.16 m pixel size = 6400 px,
    matching the 6250x6250 single-mosaic scale we know fits comfortably."""

    chunk_buffer_m: float = 15.0
    """Buffer width on each internal chunk side, used to ensure that any
    polygon centred near a chunk boundary sees the same neighbours that the
    global single-mosaic algorithm would have seen. Must be >= largest
    realistic crown radius; ideally 2x to allow for shape variability.
    With ~10 m max crown diameter, 15 m is conservative and cheap."""

    n_postprocess_workers: int = 1
    """Number of worker processes used for the chunked reconciliation.
    1 = run chunks sequentially in-process. >1 spawns a ProcessPoolExecutor;
    each worker reads its chunk's slice from the GeoParquet feature store
    independently. Set to ~min(physical_cores, n_chunks) for best wall-time."""

    keep_feature_shards: bool = False
    """If True, retain the streaming `<stem>_per_tile_features/` GeoParquet
    directory after the run completes. Useful for debugging the chunked
    postprocess (you can re-run reconciliation against the same per-tile
    detections without re-running Cellpose). Default False removes it once
    the final GeoPackage is written."""

    write_dedup_culled: bool = False
    """If True, also write `<stem>_dedup_culled.gpkg` containing every
    polygon that was removed by the dedup pass, tagged with the cluster's
    winning row index (`winner_idx`) and areas. Useful for diagnosing
    cases where real crowns are being collapsed into a single survivor —
    inspect winner/culled pairs in QGIS to decide whether
    dedup_iou_threshold or dedup_containment_threshold needs raising."""

    # --- Deprecated (no-op under the pure-vector pipeline) -------------
    merge_iou_threshold: float = 0.5
    """Deprecated. Was the IoU threshold for the in-raster cross-tile merge.
    The pipeline is now pure-vector; cross-tile reconciliation lives in
    crown_postprocess and is controlled by stitch_shift_m,
    dedup_iou_threshold, dedup_containment_threshold."""

    min_edge_area: int = 100
    """Deprecated. Was the area threshold below which unmerged tile-edge
    instances were dropped in raster space. The pure-vector pipeline does
    not need this; crowns clipped at internal seams are handled by
    stitch_shift_m and dedup."""

    # --- GPU -----------------------------------------------------------
    use_gpu: bool = True

    use_bfloat16: bool = False
    """Load Cellpose-SAM weights in bfloat16 (cellpose's default) instead of
    float32. Saves ~50% VRAM at inference time, but requires torch >=2.1
    because torch 2.0.x does not implement F.interpolate(mode='linear') for
    bfloat16, which is hit inside SAM's relative-positional-embedding code.
    Default False so the pipeline works on torch 2.0.x out of the box. On a
    newer env (torch 2.1+) you can flip this to True for the speed/VRAM win."""

    # --- CRS fallback --------------------------------------------------
    fallback_crs_epsg: int = 3007
    """EPSG code used if the source raster has no CRS (default SWEREF99 12 00,
    matching the rest of the AerialFormer pipeline)."""


@dataclass
class PhotometricAugConfig:
    """Knobs for the photometric augmentation pipeline applied per chunk
    inside train_cellpose.train(). Targets shaded / low-contrast tree crowns
    in urban orthophotos.

    Cellpose 4.x already does its own geometric augmentation (full 360°
    rotation, h/v flips, scale 0.75–1.25×) and per-tile 1–99-percentile
    normalisation at load time. The transforms here are deliberately
    photometric-only, and skewed toward effects that *survive* per-tile
    percentile renormalisation: local shadow polygons, non-linear gamma,
    CLAHE, HSV jitter, additive noise. RandomBrightnessContrast is included
    for completeness but its global effect is partially erased by the
    renormalisation — keep its probability moderate.

    Set enabled=False to skip aug entirely. Set every *_p to 0 to keep the
    pipeline plumbed but inert (useful for ablation runs)."""

    enabled: bool = True
    seed: Optional[int] = None
    """Seed passed to albumentations.Compose. None lets numpy's global
    RNG drive sampling, which is generally what you want for training."""

    dump_examples_n: int = 4
    """Number of augmented (image, mask) pairs to write into
    <artifacts_dir>/aug_preview/ (default <output_dir>/training_artifacts/
    aug_preview/) on the first chunk so the user can sanity-check the aug
    recipe in QGIS before the full run. Set 0 to disable."""

    # --- Local / non-linear (survive percentile renormalisation) -------
    random_shadow_p: float = 0.5
    random_shadow_count: Tuple[int, int] = (1, 3)
    """(min, max) shadow polygon count per tile."""

    random_gamma_p: float = 0.5
    random_gamma_limit: Tuple[int, int] = (60, 140)
    """albumentations.RandomGamma takes percent gamma; (60, 140) covers
    moderate darkening through moderate brightening."""

    clahe_p: float = 0.3
    clahe_clip_limit: float = 2.0

    hue_sat_val_p: float = 0.4
    hue_shift_limit: int = 10
    sat_shift_limit: int = 20
    val_shift_limit: int = 10

    gauss_noise_p: float = 0.3
    gauss_noise_var_limit: Tuple[float, float] = (5.0, 25.0)

    # --- Global (partially erased by Cellpose's percentile renorm) -----
    brightness_contrast_p: float = 0.4
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2


@dataclass
class TrainSource:
    """One ortho + crowns gpkg pair, with its own optional QC and val-AOI rules.

    A TrainConfig can carry many of these; tiles from every source are written
    to the same output_dir/{train,val}/ directories so Cellpose sees one
    combined dataset. Per-source bbox/val rules let the user pick a different
    val region for each ortho — including a per-source GeoPackage whose
    features (unioned) define the val AOI for that source only.
    """

    ortho_path: str
    """Source orthophoto (GeoTIFF or VRT) used to extract image tiles."""

    crowns_gpkg: str
    """GeoPackage containing per-tree polygons for this ortho."""

    name: Optional[str] = None
    """Stable identifier; used as the tile-filename prefix and as the per-source
    key in the counts dict. Falls back to the ortho's basename stem. Must be
    unique across sources — duplicate names would cause output tiles to
    collide in the flat train/ directory."""

    crowns_id_field: Optional[str] = None
    """Optional int attribute used as the instance ID. If None, sequential IDs
    are assigned in feature order. Tile-local IDs are always renumbered, so
    this is mainly useful for downstream traceability."""

    qc_field: Optional[str] = None
    """GeoPackage attribute that encodes QC status. None = no QC filtering
    (all crowns are treated as passed)."""

    qc_pass_values: Optional[List] = None
    """Values in qc_field that count as 'passed'. None = any non-null value
    passes."""

    bbox_filter: Optional[BBox] = None
    """(xmin, ymin, xmax, ymax) in this source's raster CRS, or None for the
    full extent."""

    val_bbox_filter: Optional[ValRegion] = None
    """Either:
        - (xmin, ymin, xmax, ymax) tuple in this source's raster CRS, OR
        - path to a GeoPackage whose features (unioned) define the val AOI.
      None = no val tiles produced for this source. Tile-membership is
      centre-in-region for both forms (consistent semantics)."""


@dataclass
class TrainConfig:
    # --- Inputs (one entry per ortho/crowns pair) ----------------------
    sources: List[TrainSource] = field(default_factory=list)
    """One TrainSource per ortho + crowns gpkg pair. Tiles from every source
    are written to the same output_dir/{train,val}/ directories — Cellpose's
    flat-directory training loop sees them as a single combined dataset.
    Per-source bbox_filter, val_bbox_filter, and QC rules live on each
    TrainSource so they can vary between sources.

    Empty list is allowed: train_cellpose.train() does not need sources at
    runtime (it just reads the prebuilt train/ and val/ directories), so
    FG_1_train_runner.py can construct a TrainConfig with sources=[] and
    rely on the sources_manifest.json that FG_0 wrote for traceability.
    build_training_tiles() does require at least one source."""

    # --- QC filtering (shared across sources) -------------------------
    qc_review_subdir: Optional[str] = "review"
    """Sub-directory under output_dir where tiles flagged by the QC check are
    written for human inspection. None = drop them silently."""

    # --- Tiling for training data (shared across sources) -------------
    tile_size: int = 512
    """Training tile size."""

    tile_stride: int = 384
    """Stride during training-tile extraction. tile_size - tile_stride = overlap.
    A modest overlap helps Cellpose see partial crowns at multiple offsets."""

    min_tile_instances: int = 1
    """Drop tiles that contain fewer than this many crowns. Cellpose's training
    code also enforces a minimum (min_train_masks=5 by default) but a per-tile
    pre-filter saves disk."""

    # --- Bands ---------------------------------------------------------
    band_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    """See InferenceConfig.band_indices for the full rationale and options.
    Whatever you pick here MUST match what you pass at inference time."""

    percentile_clip: Tuple[float, float] = (2.0, 98.0)

    # --- Output --------------------------------------------------------
    output_dir: str = "."
    """Directory where train/ and val/ subdirs are created with image+mask pairs."""

    artifacts_dir: Optional[str] = None
    """Directory for training artifacts (checkpoints/, flow_cache/,
    aug_preview/). None = <output_dir>/training_artifacts. Setting it lets the
    training-run outputs live apart from the training data, mirroring the
    Gbg-SEM-TreeSeg AF_1_training_data / AF_2_training_runs split."""

    # --- Cellpose training hyperparameters -----------------------------
    pretrained_model: str = "cpsam"
    """Starting checkpoint. 'cpsam' = Cellpose-SAM ViT-H pretrained."""

    n_epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    save_every: int = 50

    model_name: str = "cellpose_treecrown"

    # --- Training monitoring ------------------------------------------
    monitor_interval_epochs: int = 10
    """Run cellpose.train.train_seg in chunks of this many epochs. After each
    chunk, append losses to <model_name>_losses.csv, regenerate the loss PNG,
    and evaluate early-stop / overfitting heuristics. Smaller = more frequent
    updates but slightly higher per-chunk overhead."""

    early_stopping_patience: Optional[int] = None
    """Number of consecutive monitor intervals where val loss must fail to
    improve (relative to the best so far) before training is stopped. None =
    no early stopping; let the run reach n_epochs."""

    overfit_warn_window: int = 3
    """Window (in monitor intervals) used by the overfitting-detection
    heuristic: if train loss has monotonically decreased and val loss has
    monotonically increased over the last N updates, a warning is logged."""

    use_bfloat16: bool = False
    """See InferenceConfig.use_bfloat16. Default False for torch 2.0.x
    compatibility; flip to True if torch >=2.1 is available for speed."""

    # --- Photometric augmentation -------------------------------------
    photometric_aug: PhotometricAugConfig = field(default_factory=PhotometricAugConfig)
    """Per-chunk photometric augmentation pipeline. Cellpose handles
    geometric augs internally; this knob set covers shadow / illumination
    / colour variation. Set photometric_aug.enabled=False to disable."""

    # --- CRS fallback --------------------------------------------------
    fallback_crs_epsg: int = 3007
