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
Band selection and conversion to a Cellpose-ready array.

Cellpose-SAM's eval() expects 3-channel input — its ViT-H encoder treats
each of the three channels as an independent input. (The legacy
`channels=[cyto, nucleus]` argument from older Cellpose models is silently
ignored when the cpsam checkpoint is loaded; we don't pass it.)

Band selection is flexible: the user picks 1, 2, or 3 1-based band indices
into the source raster (R/G/B/IR for the typical RGB-IR ortho). This module:

    1. Pulls just those bands out of the rasterio-read array.
    2. Stretches each band to uint8 by per-band percentile clipping.
    3. Replicates to produce a (H, W, 3) uint8 array.

Replication strategy (so Cellpose-SAM's pretrained 3-channel weights stay
useful for 1- or 2-band inputs):

    1 band  → (B,  B,  B )         — grayscale-equivalent
    2 bands → (B1, B2, B2)         — last band repeated to fill channel 3
    3 bands → (B1, B2, B3)         — direct pass-through; preferred when
                                     you have at least three useful bands,
                                     because the encoder gets distinct
                                     signal in every channel
"""

from typing import List, Sequence, Tuple

import numpy as np


def _stretch_to_uint8(band: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Per-band percentile stretch to uint8. Constant or empty bands → all zeros."""
    finite = band[np.isfinite(band)] if band.dtype.kind == "f" else band
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    lo = np.percentile(finite, low_pct)
    hi = np.percentile(finite, high_pct)
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    out = (band.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def select_bands(
    window_array: np.ndarray,
    band_indices: Sequence[int],
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Convert a multi-band rasterio window into a Cellpose-ready (H, W, 3) uint8 array.

    Args:
        window_array:   (C_in, H, W) array as returned by rasterio.read().
        band_indices:   list of 1-based indices into the source bands.
                        Length must be 1, 2, or 3.
        percentile_clip: (low, high) percentiles for per-band uint8 stretch.

    Returns:
        (H, W, 3) uint8 array.
    """
    if window_array.ndim != 3:
        raise ValueError(f"Expected (C, H, W) array, got shape {window_array.shape}")
    if len(band_indices) not in (1, 2, 3):
        raise ValueError(
            f"band_indices must have length 1, 2, or 3; got {len(band_indices)}"
        )

    c_in = window_array.shape[0]
    for idx in band_indices:
        if idx < 1 or idx > c_in:
            raise ValueError(
                f"Band index {idx} out of range for source with {c_in} bands"
            )

    low, high = percentile_clip

    selected: List[np.ndarray] = []
    for idx in band_indices:
        band = window_array[idx - 1]
        selected.append(_stretch_to_uint8(band, low, high))

    if len(selected) == 1:
        b = selected[0]
        out = np.stack([b, b, b], axis=-1)
    elif len(selected) == 2:
        out = np.stack([selected[0], selected[1], selected[1]], axis=-1)
    else:
        out = np.stack(selected, axis=-1)

    return out
