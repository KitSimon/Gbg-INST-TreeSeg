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
Cellpose fine-tuning wrapper.

Reads the training-tile directory produced by preprocess_train.build_training_tiles,
loads images and masks into memory, and calls cellpose.train.train_seg in chunks
of `cfg.monitor_interval_epochs` so a TrainingMonitor can record live loss
curves and (optionally) trigger early stopping.

The Cellpose 4.x training API takes pre-loaded numpy arrays rather than a
training directory, so we build the in-memory list here. Cellpose 4.x has no
callback hook for per-epoch metrics, so the chunked re-entry is the only way
to surface intermediate losses.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import rasterio

from .config import TrainConfig
from .training_monitor import TrainingMonitor


def _load_pair(img_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(img_path) as src:
        img = src.read()  # (3, H, W)
    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)  # (H, W, 3) for Cellpose
    with rasterio.open(mask_path) as src:
        masks = src.read(1).astype(np.int32)
    return img, masks


def _evaluate_val(cp_model, val_images, val_masks) -> Optional[dict]:
    """
    Run inference on the val set and compute (AP@0.5, mean foreground IoU).
    Returns None if there's no val set or if anything goes wrong (training
    must not be derailed by an evaluation error). Mean IoU is foreground-vs-
    background — a coarse signal that catches "model predicts nothing" runs;
    AP@0.5 is the per-instance metric and the headline number.
    """
    if not val_images:
        return None
    try:
        from cellpose import metrics as _cp_metrics
        pred_masks, _flows, _styles = cp_model.eval(val_images)
    except Exception as exc:
        print(f"[Gbg-INST-TreeSeg] WARNING: val evaluation skipped ({type(exc).__name__}: {exc})")
        return None

    try:
        ap, _tp, _fp, _fn = _cp_metrics.average_precision(
            val_masks, pred_masks, threshold=[0.5]
        )
        ap50 = float(np.asarray(ap)[:, 0].mean())
    except Exception as exc:
        print(f"[Gbg-INST-TreeSeg] WARNING: AP computation failed ({type(exc).__name__}: {exc})")
        ap50 = None

    ious = []
    for true, pred in zip(val_masks, pred_masks):
        t = (true > 0)
        p = (pred > 0)
        union = (t | p).sum()
        if union == 0:
            continue
        ious.append(float((t & p).sum()) / float(union))
    mean_iou = float(np.mean(ious)) if ious else None

    out = {}
    if ap50 is not None:
        out["ap50"] = ap50
    if mean_iou is not None:
        out["mean_iou"] = mean_iou
    return out or None


def _dump_aug_examples(
    images: List[np.ndarray], masks: List[np.ndarray], names: List[str],
    out_dir: str, n: int,
) -> None:
    """Write the first n (image, mask) pairs of an augmented chunk to disk
    so the user can sanity-check the aug recipe in QGIS before committing
    to a long training run. Pure pixel preview; no CRS/transform attached."""
    if n <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(n, len(images))):
        stem = os.path.splitext(names[i].replace("_img.tif", ""))[0]
        img = images[i]  # (H, W, 3) uint8
        mask = masks[i]  # (H, W) int32
        with rasterio.open(
            os.path.join(out_dir, f"{stem}_aug_img.tif"), "w",
            driver="GTiff", width=img.shape[1], height=img.shape[0],
            count=3, dtype="uint8", compress="lzw",
        ) as dst:
            dst.write(np.transpose(img, (2, 0, 1)))
        with rasterio.open(
            os.path.join(out_dir, f"{stem}_aug_masks.tif"), "w",
            driver="GTiff", width=mask.shape[1], height=mask.shape[0],
            count=1, dtype="uint16", compress="lzw", nodata=0,
        ) as dst:
            dst.write(mask.astype(np.uint16), 1)
    print(f"[Gbg-INST-TreeSeg] wrote {min(n, len(images))} aug preview pairs to {out_dir}")


def _gather_pairs(split_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    images, masks, names = [], [], []
    img_files = sorted(glob.glob(os.path.join(split_dir, "*_img.tif")))
    for ip in img_files:
        mp = ip.replace("_img.tif", "_masks.tif")
        if not os.path.exists(mp):
            continue
        i, m = _load_pair(ip, mp)
        images.append(i)
        masks.append(m)
        names.append(os.path.basename(ip))
    return images, masks, names


def train(cfg: TrainConfig) -> dict:
    """
    Fine-tune Cellpose. Returns a dict with the produced checkpoint path
    and the paths of any sidecar files. Also writes the dict as JSON next
    to the checkpoint for FG_2 to pick up automatically.
    """
    from cellpose import models as _cp_models  # lazy
    from cellpose.train import train_seg

    train_dir = os.path.join(cfg.output_dir, "train")
    val_dir = os.path.join(cfg.output_dir, "val")

    print(f"[Gbg-INST-TreeSeg] loading training pairs from {train_dir}")
    train_images_orig, train_masks, train_names = _gather_pairs(train_dir)
    if not train_images_orig:
        raise FileNotFoundError(f"No *_img.tif / *_masks.tif pairs found in {train_dir}")

    test_images: Optional[List[np.ndarray]] = None
    test_masks: Optional[List[np.ndarray]] = None
    test_names: Optional[List[str]] = None
    if os.path.isdir(val_dir):
        ti, tm, tn = _gather_pairs(val_dir)
        if ti:
            test_images, test_masks, test_names = ti, tm, tn

    print(f"[Gbg-INST-TreeSeg] {len(train_images_orig)} train tiles, "
          f"{0 if test_images is None else len(test_images)} val tiles")

    cp_model = _cp_models.CellposeModel(
        gpu=True,
        pretrained_model=cfg.pretrained_model,
        use_bfloat16=cfg.use_bfloat16,
    )

    # All training outputs live under cfg.artifacts_dir (default:
    # <output_dir>/training_artifacts). Inside it:
    #   checkpoints/    - cellpose model weights + losses CSV/PNG + info JSON
    #   flow_cache/     - cellpose's per-mask flow-field cache (one *_flows.tif
    #                     per training/val tile, ~4 MB each). Cellpose computes
    #                     these once at the first epoch and reuses them across
    #                     epochs; before this re-org they leaked into cwd.
    artifacts_dir = cfg.artifacts_dir or os.path.join(cfg.output_dir, "training_artifacts")
    save_dir = os.path.join(artifacts_dir, "checkpoints")
    flow_cache_train = os.path.join(artifacts_dir, "flow_cache", "train")
    flow_cache_val = os.path.join(artifacts_dir, "flow_cache", "val")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(flow_cache_train, exist_ok=True)
    if test_images is not None:
        os.makedirs(flow_cache_val, exist_ok=True)

    # Cellpose names flow files as os.path.splitext(files[n])[0] + "_flows.tif",
    # so passing <flow_cache_dir>/<basename> makes the cache land in our
    # managed dir without affecting how cellpose loads inputs (it loads from
    # the in-memory train_data list, not from disk).
    train_files_for_flows = [os.path.join(flow_cache_train, n) for n in train_names]
    test_files_for_flows = (
        [os.path.join(flow_cache_val, n) for n in test_names]
        if test_names is not None else None
    )

    monitor = TrainingMonitor(
        save_dir=save_dir,
        model_name=cfg.model_name,
        early_stopping_patience=cfg.early_stopping_patience,
        overfit_window=cfg.overfit_warn_window,
        monitor_interval_epochs=cfg.monitor_interval_epochs,
        total_epochs=cfg.n_epochs,
    )

    # Build the photometric aug pipeline once. albumentations re-samples
    # transform parameters internally per call, so calling the pipeline
    # again on the same tile gives a different perturbation each time.
    # The val list is never augmented — keep the held-out signal honest.
    aug_fn = None
    if cfg.photometric_aug.enabled:
        from .photometric_aug import build_aug_pipeline
        aug_fn = build_aug_pipeline(cfg.photometric_aug)
        print(
            f"[Gbg-INST-TreeSeg] photometric augmentation enabled "
            f"(per-chunk fresh re-aug, {len(train_images_orig)} tiles)"
        )

    chunk = max(1, cfg.monitor_interval_epochs)
    total_epochs = cfg.n_epochs
    epochs_done = 0
    all_train: List[float] = []
    all_test: List[Optional[float]] = []
    final_ckpt = None
    early_stopped = False
    early_stop_reason = ""
    first_chunk = True

    while epochs_done < total_epochs:
        n_this = min(chunk, total_epochs - epochs_done)

        # Regenerate the augmented training list for this chunk. Masks,
        # filenames, and the flow-cache paths stay the same — Cellpose's
        # mask-keyed flow cache hits across chunks unchanged.
        if aug_fn is not None:
            train_images = [aug_fn(im) for im in train_images_orig]
            if first_chunk and cfg.photometric_aug.dump_examples_n > 0:
                _dump_aug_examples(
                    train_images, train_masks, train_names,
                    out_dir=os.path.join(artifacts_dir, "aug_preview"),
                    n=cfg.photometric_aug.dump_examples_n,
                )
        else:
            train_images = train_images_orig

        first_chunk = False

        ckpt_path, train_losses, test_losses = train_seg(
            cp_model.net,
            train_data=train_images,
            train_labels=train_masks,
            train_files=train_files_for_flows,
            test_data=test_images,
            test_labels=test_masks,
            test_files=test_files_for_flows,
            load_files=False,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            n_epochs=n_this,
            weight_decay=cfg.weight_decay,
            save_path=save_dir,
            save_every=cfg.save_every,
            model_name=cfg.model_name,
        )
        final_ckpt = ckpt_path

        chunk_train = np.asarray(train_losses).reshape(-1).tolist()
        chunk_test = (
            np.asarray(test_losses).reshape(-1).tolist()
            if test_losses is not None and len(np.asarray(test_losses)) > 0
            else [None] * n_this
        )
        # train_seg may not always return exactly n_this loss values (it can
        # average over iterations or skip recording on degenerate batches).
        # Pad/truncate to n_this so monitor epoch counts stay aligned.
        chunk_train = (chunk_train + [chunk_train[-1] if chunk_train else 0.0] * n_this)[:n_this]
        chunk_test = (chunk_test + [None] * n_this)[:n_this]

        all_train.extend(chunk_train)
        all_test.extend(chunk_test)

        # Compute eval metrics (AP@0.5, mean foreground IoU) on the val set
        # once per chunk. Skipped silently if there's no val set or if
        # cellpose's metrics module fails for any reason — training itself
        # is unaffected.
        chunk_metrics = _evaluate_val(cp_model, test_images, test_masks)

        for i, (tr, te) in enumerate(zip(chunk_train, chunk_test)):
            epochs_done += 1
            # Metrics attach only to the chunk-boundary epoch (the last one).
            metrics_arg = chunk_metrics if i == len(chunk_train) - 1 else None
            stop, reason = monitor.update(epochs_done, tr, te, metrics=metrics_arg)
            if stop:
                early_stopped = True
                early_stop_reason = reason
                break
        if early_stopped:
            break

    monitor.finalize()

    # Source-identity record for traceability. Prefer the manifest written by
    # FG_0 (which the train runner attaches as cfg.sources_manifest); fall
    # back to cfg.sources if the runner constructed sources directly.
    sources_record = getattr(cfg, "sources_manifest", None)
    if not sources_record:
        sources_record = [
            {
                "name": s.name,
                "ortho_path": s.ortho_path,
                "crowns_gpkg": s.crowns_gpkg,
                "qc_field": s.qc_field,
                "val_bbox_filter": (
                    s.val_bbox_filter
                    if (s.val_bbox_filter is None
                        or isinstance(s.val_bbox_filter, tuple))
                    else str(s.val_bbox_filter)
                ),
            }
            for s in cfg.sources
        ]

    info = {
        "checkpoint": str(final_ckpt) if final_ckpt is not None else None,
        "epochs_completed": epochs_done,
        "early_stopped": early_stopped,
        "early_stop_reason": early_stop_reason if early_stopped else "",
        "train_losses": [float(x) for x in all_train],
        "test_losses": [None if v is None else float(v) for v in all_test],
        "loss_csv": monitor.csv_path,
        "loss_png": monitor.png_path,
        "metrics_png": monitor.metrics_png_path,
        "config": {
            "n_epochs": cfg.n_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "pretrained_model": cfg.pretrained_model,
            "band_indices": list(cfg.band_indices),
            "monitor_interval_epochs": cfg.monitor_interval_epochs,
            "early_stopping_patience": cfg.early_stopping_patience,
            "sources": sources_record,
            "photometric_aug": dataclasses.asdict(cfg.photometric_aug),
        },
    }
    info_path = os.path.join(save_dir, f"{cfg.model_name}_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(
        f"[Gbg-INST-TreeSeg] training complete: {final_ckpt} "
        f"({epochs_done}/{total_epochs} epochs"
        f"{' — early stopped' if early_stopped else ''})"
    )
    return info
