# Gbg-INST-TreeSeg — Tree-Crown Instance Segmentation

**G**othen**b**ur**g** **INST**ance segmentation — **Tree Seg**mentation.

Turns an RGB aerial orthophoto into a GeoPackage of individual tree-crown
polygons. An **AerialFormer** semantic prior — from the companion
[Gbg-SEM-TreeSeg](https://github.com/KitSimon/Gbg-SEM-TreeSeg) repository — masks
the canopy, then **Cellpose-SAM** flow-guided instance separation runs
tile-by-tile across an arbitrarily large mosaic, followed by a memory-bounded,
parallel **vector** reconciliation pass. It is an independent implementation of
the method described in **FG-TreeSeg** (Chen et al., 2026).

Input data during development was the Lantmäteriet orthophoto 2022 (RGB, 0.16 m) over Gothenburg,
Sweden, in SWEREF99 12 00 (EPSG:3007). The imagery and training data are **not**
included in this repository.

## Installation

```bash
conda env create -f environment.yml
conda activate gbg_inst_treeseg
```

The `environment.yml` defaults to a CUDA PyTorch build — edit `pytorch-cuda` to
match your driver, or drop it and add `cpuonly` for CPU-only inference.

Alternatively, if you already run AerialFormer, install the extra deps into that
env instead:

```bash
pip install "cellpose>=4.0" geopandas shapely pyogrio rtree pyarrow
pip install "numpy<2"   # only if the env pins numpy 2.x against NumPy-1.x-built wheels
```

### Machine-specific paths

Absolute paths to your inputs (ortho mosaic, LAS tiles, the Gbg-SEM-TreeSeg
working copy) live in `local_paths.py`, which is **gitignored** so your paths are
never committed. Create it from the template:

```bash
cp local_paths.example.py local_paths.py
# then edit: ORTHO_PATH, GBG_SEM_ROOT, LAS_DIR, TIF_DIR
```

`project_paths.py` imports these and raises a clear error if `local_paths.py` is
missing.

## Pipeline

```
 Orthophoto ─► AF_3 (Gbg-SEM-TreeSeg) ─► semantic raster ─┐
 Orthophoto ──────────────────────────────────────────────┴─► FG_2 ─► tree_crowns.gpkg
                                                              (Cellpose-SAM, tiled,
                                                               vector reconciliation)
```

The workflow is driven by numbered stages. Each `FG_*_runner.py` script at the
repository root holds its configuration at the top — edit and run the runners
from the repository root.

| Stage | Script | Purpose |
|---|---|---|
| FG_0 | `FG_0_preprocess_runner.py` | Build Cellpose training tiles from a per-tree crown GeoPackage |
| FG_1 | `FG_1_train_runner.py` | Fine-tune Cellpose-SAM on those tiles (optional; the default `cpsam` checkpoint also works) |
| FG_2 | `FG_2_inference_runner.py` | **Full instance inference** — tiled Cellpose-SAM under the semantic prior + parallel vector reconciliation → crown polygons |
| FG_3 | `FG_3_las_filter_runner.py` | Optional LiDAR height filter: flag/drop crowns with no real canopy height under them |
| tools | `tools/FG_2a_postprocess_resume.py`, `tools/FG_2b_qa_preview.py` | Resume FG_2's postprocess from feature shards; render a QA PNG preview |

Runners live at the repository root; all implementation is under `utils/` (see
[Documentation](#documentation)). Project data lives under `Project_1/`
(gitignored), mirroring the Gbg-SEM-TreeSeg convention; the per-stage
subdirectories are defined once in `project_paths.py` — change `PROJECT_NAME`
there to start a fresh project.

## Quickstart

### Inference

```bash
# 1. (AerialFormer env, Gbg-SEM-TreeSeg repo) produce the semantic raster —
#    edit the constants in AF_3_inference_runner.py, then:
python AF_3_inference_runner.py

# 2. (this repo) point SEMANTIC_RASTER_PATH and ORTHO_PATH in
#    FG_2_inference_runner.py at the AF_3 output and the ortho, then:
python FG_2_inference_runner.py
```

Outputs land in `Project_1/FG_2_inference_results/`:

- `<stem>_crowns.gpkg` — final crown polygons. Schema: `instance_id`, `area_px`,
  `area_m2`, `centroid_x`, `centroid_y`, `touches_image_edge`, `geometry`.
- `<stem>_tile_grid.gpkg` — sidecar of tile rectangles with image-edge flags.
- `<stem>_dedup_culled.gpkg` — optional diagnostic (when `WRITE_DEDUP_CULLED=True`;
  see [docs/configuration.md](docs/configuration.md)).

Follow-ups: `FG_3_las_filter_runner.py` cross-checks each crown against the LiDAR
point cloud (p99 normalized height) and flags detections with no real canopy
height; `tools/FG_2b_qa_preview.py` renders a quick PNG overlay for a sanity check
before opening QGIS; `tools/FG_2a_postprocess_resume.py` resumes an interrupted
postprocess from the per-tile feature shards.

### Fine-tuning (optional)

```bash
# 1. Build training tiles from a per-tree GeoPackage
#    (inputs in Project_1/FG_0_training/ → tiles in Project_1/FG_0_training_data/)
python FG_0_preprocess_runner.py

# 2. Fine-tune Cellpose-SAM on those tiles
#    (checkpoint + monitors in Project_1/FG_1_training_runs/)
python FG_1_train_runner.py

# 3. Set CELLPOSE_CHECKPOINT in FG_2_inference_runner.py to the path printed by
#    step 2, then re-run inference.
```

## Documentation

- [docs/walkthrough.md](docs/walkthrough.md) — detailed pipeline diagram,
  annotated repository layout, and a stage-by-stage data-prep / setup / run
  walkthrough with the config decisions and gotchas that matter.
- [docs/configuration.md](docs/configuration.md) — every runner knob, QC tile
  filtering, training monitoring, and the `dedup_culled` diagnostic schema.
- [docs/design.md](docs/design.md) — design decisions (precomputed semantic
  raster, vector reconciliation, memory-bounded parallel postprocess) and scaling
  figures.
- [docs/limitations.md](docs/limitations.md) — known limitations and further work.

## Related repositories

This pipeline is the **instance-segmentation stage** of a two-stage workflow. Its
upstream sibling,
**[Gbg-SEM-TreeSeg](https://github.com/KitSimon/Gbg-SEM-TreeSeg)**, trains the
AerialFormer semantic model and produces the artifacts this repo consumes (via
the `GBG_SEM_ROOT` path in `local_paths.py`):

| Gbg-SEM-TreeSeg artifact | Consumed by | As |
|---|---|---|
| `Project_1/AF_0_training/<tree polygons>.gpkg` (per-tree crown polygons) | FG_0 | Cellpose training labels |
| `Project_1/AF_3_inference_results/<run>/<stem>_segmentation.tif` | FG_2 | canopy prior (tree class = 2) |

The semantic raster must be pixel-aligned with the orthophoto (same grid, CRS,
extent) — it is, when AF_3 was run on the same input.

## Attribution

This pipeline is an **independent implementation of the method** described in
FG-TreeSeg. It also builds on **Cellpose-SAM** (flow-guided instance separation)
 and the companion
**[Gbg-SEM-TreeSeg](https://github.com/KitSimon/Gbg-SEM-TreeSeg)** / AerialFormer
semantic prior.

If you use this software in academic work, please cite both this repository (see
[CITATION.cff](CITATION.cff)) and the FG-TreeSeg paper:

```bibtex
@ARTICLE{11520829,
  author={Chen, Pengyu and Lyu, Fangzheng and Wang, Sicheng and Wang, Cuizhen},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={FG-TreeSeg: Flow-Guided Tree Crown Segmentation without Instance Annotations},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LGRS.2026.3693969}}
```

> Note: FG-TreeSeg was published under the name *ZS-TreeSeg* in its earlier (v1)
> form; the authors renamed it FG-TreeSeg in the revised paper. The citation above
> is the correct, current reference.

## License

Released under the **GNU Affero General Public License v3.0 or later**
(AGPL-3.0-or-later). See [LICENSE](LICENSE). Third-party dependencies (Cellpose,
GDAL/rasterio/geopandas, PyTorch, …) are used as installed packages, not vendored,
and retain their own licenses.
