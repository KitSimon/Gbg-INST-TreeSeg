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
gbg_inst_treeseg.utils
====================

Utility modules that implement the tree-crown instance pipeline. Imported by
the runner scripts at the package root (`FG_*_runner.py`).

Layout:
  config.py              — InferenceConfig / TrainConfig dataclasses
  bands.py               — band selection → 3-channel uint8 array
  tiling.py              — sliding-window grid + per-tile image-edge flags
  instance_stitching.py  — tiled Cellpose + per-tile polygonisation
                           (with optional GeoParquet streaming via feature_store)
  feature_store.py       — ParquetShardSink + spatial-filtered shard reader
  crown_postprocess.py   — stitch / merge_duplicates / resolve_mosaic
  chunked_postprocess.py — super-tile orchestrator for memory-bounded postprocess
  inference.py           — end-to-end orchestrator (run_pipeline)
  vector_export.py       — final GeoPackage writer (pyogrio)
  preprocess_train.py    — build (image, instance-mask) tile pairs for training
  train_cellpose.py      — Cellpose-SAM fine-tuning entry point
  training_monitor.py    — chunked train loop, CSV/PNG monitors, early stopping
"""
