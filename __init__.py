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
gbg_inst_treeseg
==============

Tree-crown instance segmentation pipeline that combines:
  1. AerialFormer semantic prior — produced separately by the
     Gbg-SEM-TreeSeg repo (https://github.com/KitSimon/Gbg-SEM-TreeSeg,
     AF_3_inference_runner.py)
  2. Cellpose-SAM (flow-guided instance separation, this package)

Adapts the FG-TreeSeg framework for tiled inference over large geospatial
orthophoto mosaics (VRT or GeoTIFF), with cross-tile instance ID merging
and a parallel pipeline for Cellpose fine-tuning on tree-crown polygons.

The two stages communicate only through files (the semantic GeoTIFF), so this
package never imports mmcv/mmsegmentation.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ["PACKAGE_DIR"]
