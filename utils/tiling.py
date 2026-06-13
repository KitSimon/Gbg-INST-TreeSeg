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
Tile grid generation for instance inference.

The math here mirrors `create_grids` in AF_3_inference-vrt_new.py — we keep a
small private copy rather than importing it so the tiling logic can be unit
tested without dragging in mmcv/mmsegmentation/torch via the AF_3 module.
"""

from typing import Dict, List


def create_grids(width: int, height: int, window_size: int, stride: int) -> List[Dict]:
    """
    Compute sliding-window positions with overlap-aware crop regions.

    Each entry has read coordinates (full window), crop coordinates (region
    we keep when the tile is interior), and write coordinates (output offset).
    The crop fields are kept compatible with AF_3 callers; the instance
    stitcher in this package uses a different merging strategy and only
    needs (read_x, read_y, win_w, win_h).

    For images smaller than `window_size` along an axis, that axis collapses
    to a single tile of size `min(width, window_size)` (or height) starting
    at 0. The downstream rasterio reads/writes use win_w/win_h, so the tile
    stays inside the raster.
    """
    half_overlap = (window_size - stride) // 2

    grids: List[Dict] = []

    if height <= window_size:
        ys = [(0, height)]
    else:
        ys = []
        for y in range(0, height, stride):
            y_end = (y + window_size >= height)
            read_y = (height - window_size) if y_end else y
            ys.append((read_y, window_size))
            if y_end:
                break

    if width <= window_size:
        xs = [(0, width)]
    else:
        xs = []
        for x in range(0, width, stride):
            x_end = (x + window_size >= width)
            read_x = (width - window_size) if x_end else x
            xs.append((read_x, window_size))
            if x_end:
                break

    for read_y, win_h in ys:
        y_end = (read_y + win_h >= height)
        crop_y = 0 if read_y == 0 else half_overlap
        crop_h = (win_h - crop_y) if y_end else (win_h - crop_y if read_y == 0 else stride)
        if read_y == 0 and y_end:
            crop_h = win_h

        for read_x, win_w in xs:
            x_end = (read_x + win_w >= width)
            crop_x = 0 if read_x == 0 else half_overlap
            crop_w = (win_w - crop_x) if x_end else (win_w - crop_x if read_x == 0 else stride)
            if read_x == 0 and x_end:
                crop_w = win_w

            grids.append({
                "read_x": read_x, "read_y": read_y,
                "win_w": win_w, "win_h": win_h,
                "crop_x": crop_x, "crop_y": crop_y,
                "crop_w": crop_w, "crop_h": crop_h,
                "write_x": read_x + crop_x,
                "write_y": read_y + crop_y,
                # Sides of the tile that coincide with the ortho's outer
                # boundary. Vector post-processing uses these to skip the
                # internal-seam shift on image-boundary sides — otherwise it
                # would cull legitimate crowns at the ortho edge.
                "is_image_edge_n": (read_y == 0),
                "is_image_edge_s": (read_y + win_h >= height),
                "is_image_edge_w": (read_x == 0),
                "is_image_edge_e": (read_x + win_w >= width),
            })

    return grids
