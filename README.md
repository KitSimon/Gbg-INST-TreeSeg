# Gbg-INST-TreeSeg — Tree-Crown Instance Segmentation

A two-stage pipeline that turns an RGB-IR orthophoto into a GeoPackage of
individual tree-crown polygons. It is an independent implementation of the
method described in **FG-TreeSeg** (Chen et al., 2026; see
[Attribution & citation](#attribution--citation)): the semantic prior is
provided by the **AerialFormer** model trained in the companion
[Gbg-SEM-TreeSeg](https://github.com/KitSimon/Gbg-SEM-TreeSeg) repository, and
Cellpose-SAM flow-guided instance separation is run tile-by-tile across an
arbitrarily large mosaic with a memory-bounded, parallel **vector**
reconciliation pass.

```
                ┌─────────────────────┐
 Orthophoto ──► │ AF_3 (AerialFormer) │ ──► Semantic raster (uint8: bg/water/tree, 255=nodata)
   (VRT/TIF)    │  (Gbg-SEM-TreeSeg)  │
                └─────────────────────┘
                                              │
 Orthophoto ─────────────────────────────────►├─► gbg_inst_treeseg.utils.inference.run_pipeline
                                              │      ├─ tile grid (1024 px, 256 px overlap)
                                              │      ├─ Cellpose-SAM per tile (canopy-masked)
                                              │      ├─ rasterio.features.shapes → per-tile polygons
                                              │      ├─ stream → GeoParquet shards (bounded RAM)
                                              │      └─ chunked postprocess (super-tiles, parallel):
                                              │           ├─ stitch  (cull seam-cropped)
                                              │           ├─ merge_duplicates (union same-crown clusters)
                                              │           └─ resolve_mosaic   (clip overlaps, no gaps)
                                              ▼
                                          tree_crowns.gpkg
                                          (instance_id, area_px, area_m2,
                                           centroid_x, centroid_y,
                                           touches_image_edge)
                                          + tile_grid.gpkg (sidecar)
                                          + dedup_culled.gpkg (optional diagnostic)
```

A parallel pipeline (`FG_0` → `FG_1`) fine-tunes Cellpose-SAM on per-tree
crown polygons, producing a checkpoint that the inference runner can swap in.

## Layout

Runners live at the repository root (one file per pipeline stage) and are run
from there. All of the implementation lives under `utils/`. Project data lives
under `Project_1/` (gitignored), mirroring the Gbg-SEM-TreeSeg convention.

```
gbg_inst_treeseg/                 ← run all runners from here
├── project_paths.py              ← directory layout (single source of truth)
├── local_paths.example.py        ← template for machine-specific paths (copy → local_paths.py)
├── FG_0_preprocess_runner.py     ← build Cellpose training data
├── FG_1_train_runner.py          ← fine-tune Cellpose-SAM
├── FG_2_inference_runner.py      ← full instance inference (the headline runner)
├── FG_3_las_filter_runner.py     ← LiDAR height filter on the FG_2 crowns
├── tools/
│   ├── FG_2a_postprocess_resume.py  ← resume FG_2's postprocess from feature shards
│   └── FG_2b_qa_preview.py          ← quick PNG sanity check of FG_2 output
├── utils/                        ← pipeline implementation (see docs/)
├── docs/                         ← configuration, design notes, limitations
└── Project_1/                    ← project data (gitignored)
```

The per-stage `Project_1/` subdirectories (`FG_0_training` …
`FG_3_las_filtered`) are created by the stage that produces them and are
defined once in `project_paths.py` — change `PROJECT_NAME` there to start a
fresh project.

Two artifacts cross the boundary from the Gbg-SEM-TreeSeg working copy
(via the `GBG_SEM_ROOT` path):

| Gbg-SEM-TreeSeg artifact | Consumed by |
|---|---|
| `Project_1/AF_0_training/hull30.gpkg` (per-tree crown polygons) | FG_0, as Cellpose training labels |
| `Project_1/AF_3_inference_results/<run>/<stem>_segmentation.tif` | FG_2, as the canopy prior |

## Install

```bash
conda env create -f environment.yml
conda activate zs-treeseg
```

The `environment.yml` defaults to a CUDA PyTorch build — edit `pytorch-cuda`
to match your driver, or drop it and add `cpuonly` for CPU-only inference.

Alternatively, if you already run AerialFormer, you can install the extra
deps into that env instead:

```bash
pip install "cellpose>=4.0" geopandas shapely pyogrio rtree pyarrow
pip install "numpy<2"   # only if the env pins numpy 2.x against NumPy-1.x-built wheels
```

### Configure machine-specific paths

The absolute paths to your inputs (ortho mosaic, LAS tiles, the Gbg-SEM-TreeSeg
working copy) live in `local_paths.py`, which is **gitignored** so your paths
never get committed. Create it from the template:

```bash
cp local_paths.example.py local_paths.py
# then edit local_paths.py: set ORTHO_PATH, GBG_SEM_ROOT, LAS_DIR, TIF_DIR
```

`project_paths.py` imports these and raises a clear error if `local_paths.py`
is missing.

## Inference quick-start

```bash
# 1. (AerialFormer env, Gbg-SEM-TreeSeg repo) Run AF_3 to produce the
#    semantic raster — edit the constants in AF_3_inference_runner.py, then:
python AF_3_inference_runner.py

# 2. (this repo) Point SEMANTIC_RASTER_PATH / ORTHO_PATH in
#    FG_2_inference_runner.py at the AF_3 output and the ortho, then:
python FG_2_inference_runner.py
```

Outputs (in `Project_1/FG_2_inference_results/`):

- `<stem>_crowns.gpkg` — final crown polygons. Schema: `instance_id`,
  `area_px`, `area_m2`, `centroid_x`, `centroid_y`, `touches_image_edge`,
  `geometry`.
- `<stem>_tile_grid.gpkg` — sidecar of tile rectangles with image-edge flags.
- `<stem>_dedup_culled.gpkg` — optional diagnostic (when
  `WRITE_DEDUP_CULLED=True`); see [docs/configuration.md](docs/configuration.md).

Follow-ups:

- `FG_3_las_filter_runner.py` cross-checks every crown against the LiDAR
  point cloud (p99 normalized height) and flags detections with no real
  canopy height under them.
- `tools/FG_2b_qa_preview.py` renders a quick PNG overlay for a fast sanity
  check before opening QGIS.
- `tools/FG_2a_postprocess_resume.py` resumes FG_2's chunked postprocess from
  the per-tile feature shards if a run was interrupted.

## Fine-tuning quick-start

```bash
# 1. Build training tiles from a per-tree GeoPackage
#    (inputs in Project_1/FG_0_training/ → tiles in Project_1/FG_0_training_data/)
python FG_0_preprocess_runner.py

# 2. Fine-tune Cellpose-SAM on those tiles
#    (checkpoint + monitors in Project_1/FG_1_training_runs/)
python FG_1_train_runner.py

# 3. Set CELLPOSE_CHECKPOINT in FG_2_inference_runner.py to the path printed
#    by step 2, then re-run inference.
```

## Documentation

- [docs/configuration.md](docs/configuration.md) — every runner knob, QC tile
  filtering, training monitoring, and the `dedup_culled` diagnostic schema.
- [docs/design.md](docs/design.md) — design decisions (precomputed semantic
  raster, vector reconciliation, memory-bounded parallel postprocess, …) and
  scaling figures.
- [docs/limitations.md](docs/limitations.md) — known limitations and further
  work.

## License

Released under the **GNU Affero General Public License v3.0 or later**
(AGPL-3.0-or-later). See [LICENSE](LICENSE). Third-party dependencies
(Cellpose, GDAL/rasterio/geopandas, PyTorch, …) are used as installed
packages, not vendored, and retain their own licenses.

## Attribution & citation

This pipeline is an **independent implementation of the method** described in
FG-TreeSeg — no code of substance from the original work is reused. It also
builds on **Cellpose-SAM** (flow-guided instance separation) and the companion
**[Gbg-SEM-TreeSeg](https://github.com/KitSimon/Gbg-SEM-TreeSeg)** /
AerialFormer semantic prior.

If you use this software in academic work, please cite both this repository
(see [CITATION.cff](CITATION.cff)) and the FG-TreeSeg paper:

```bibtex
@ARTICLE{11520829,
  author={Chen, Pengyu and Lyu, Fangzheng and Wang, Sicheng and Wang, Cuizhen},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={FG-TreeSeg: Flow-Guided Tree Crown Segmentation without Instance Annotations},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Modeling;Trees (botanical);Vegetation;Fluid flow;Training;Instance segmentation;Remote sensing;Pixel;Annotations;Visualization;Instance segmentation;tree crown delineation;foundation model;GeoAI},
  doi={10.1109/LGRS.2026.3693969}}
```

> Note: FG-TreeSeg was published under the name *ZS-TreeSeg* in its earlier
> (v1) form; the authors renamed it FG-TreeSeg in the revised paper. The
> citation above is the correct, current reference.
