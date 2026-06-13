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
Photometric augmentation pipeline for Cellpose-SAM training tiles.

Wraps a small, opinionated subset of albumentations transforms. Used by
train_cellpose.train(): before each chunk's call to cellpose.train.train_seg,
the originals list is run through this pipeline to produce a fresh
augmented list for that chunk. Masks are not touched (photometric augs
only); Cellpose's flow-field cache is keyed on masks, so the cache stays
valid across chunks.

Why this menu of transforms (and not others):

  - Cellpose 4.x already does its own geometric augmentation per batch
    (full 360° rotation, h/v flips, scale 0.75–1.25×). We do NOT add more.

  - Cellpose normalises every tile by 1–99 percentile per channel at load
    time (cellpose/train.py:449–451). That step largely *erases* uniform
    global brightness/contrast shifts, so we lean on transforms whose
    effect survives renormalisation: local shadow polygons, non-linear
    gamma, CLAHE, HSV jitter, additive noise. RandomBrightnessContrast is
    included for completeness — keep its probability moderate.

  - The whole point is helping the model on shaded / low-light crowns in
    urban orthophotos, hence shadow- and gamma-heavy defaults.

Inputs / outputs of the returned callable:
    image: np.ndarray, shape (H, W, 3), dtype uint8
    return: np.ndarray, shape (H, W, 3), dtype uint8
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import PhotometricAugConfig


def build_aug_pipeline(
    cfg: PhotometricAugConfig,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that applies the configured photometric augmentations.

    Lazy-imports albumentations so the rest of the codebase doesn't need the
    dependency unless training actually runs with augmentation enabled.

    The callable is safe to invoke many times; albumentations re-samples
    transform parameters on each call, so the same source tile gets a
    different perturbation per call.
    """
    import albumentations as A

    transforms = []

    if cfg.random_shadow_p > 0:
        # albumentations >=1.4: num_shadows_limit replaces the older
        # num_shadows_lower / num_shadows_upper pair. The shadow polygons
        # are rendered as semi-transparent dark regions over the image.
        transforms.append(A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=cfg.random_shadow_count,
            shadow_dimension=5,
            p=cfg.random_shadow_p,
        ))

    if cfg.random_gamma_p > 0:
        transforms.append(A.RandomGamma(
            gamma_limit=cfg.random_gamma_limit,
            p=cfg.random_gamma_p,
        ))

    if cfg.brightness_contrast_p > 0:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=cfg.brightness_limit,
            contrast_limit=cfg.contrast_limit,
            p=cfg.brightness_contrast_p,
        ))

    if cfg.clahe_p > 0:
        transforms.append(A.CLAHE(
            clip_limit=cfg.clahe_clip_limit,
            p=cfg.clahe_p,
        ))

    if cfg.hue_sat_val_p > 0:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=cfg.hue_shift_limit,
            sat_shift_limit=cfg.sat_shift_limit,
            val_shift_limit=cfg.val_shift_limit,
            p=cfg.hue_sat_val_p,
        ))

    if cfg.gauss_noise_p > 0:
        transforms.append(A.GaussNoise(
            var_limit=cfg.gauss_noise_var_limit,
            p=cfg.gauss_noise_p,
        ))

    pipeline = (
        A.Compose(transforms, seed=cfg.seed)
        if cfg.seed is not None
        else A.Compose(transforms)
    )

    def _apply(image: np.ndarray) -> np.ndarray:
        return pipeline(image=image)["image"]

    return _apply
