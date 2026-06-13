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
Live monitoring for Cellpose fine-tuning runs.

`TrainingMonitor` accumulates per-epoch losses (and optional eval metrics),
persists them as a CSV, refreshes two PNG plots — one for losses, one for
metrics — and runs two heuristics:

    1. Overfitting detection — over the last `overfit_window` updates, train
       loss monotonically decreased while val loss monotonically increased.
       Logs a warning the first time this happens; only re-warns after the
       state recovers and recurs.

    2. Early stopping (opt-in) — if val loss has not improved over its best
       seen value for `early_stopping_patience` consecutive monitor intervals,
       the next call to `update()` returns `(True, reason)` so the training
       loop can break out.

The class lazy-imports matplotlib (only at first PNG render) so importing
this module costs nothing in environments without matplotlib.

Caveat: the train() refactor calls update() once per epoch, but `train_seg`
itself is invoked in chunks (see TrainConfig.monitor_interval_epochs). Cellpose
4.x's AdamW has no built-in LR schedule, so re-entering train_seg every chunk
does not currently disturb training; if Cellpose ever gains a cosine/step LR
schedule, this assumption needs revisiting.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional, Tuple


# Metric columns recorded in the CSV (in addition to epoch/loss/wall_time).
# Values come from train_cellpose.py via the `metrics` dict passed to
# update(); cells are blank when the metric wasn't supplied for a given
# epoch (e.g. non-chunk-boundary epochs, or runs without a val set).
METRIC_COLUMNS = ("ap50", "mean_iou")


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class TrainingMonitor:
    def __init__(
        self,
        save_dir: str,
        model_name: str,
        early_stopping_patience: Optional[int] = None,
        overfit_window: int = 3,
        monitor_interval_epochs: int = 10,
        total_epochs: Optional[int] = None,
    ):
        self.save_dir = save_dir
        self.model_name = model_name
        self.early_stopping_patience = early_stopping_patience
        self.overfit_window = overfit_window
        self.monitor_interval_epochs = max(1, monitor_interval_epochs)
        self.total_epochs = total_epochs

        self.csv_path = os.path.join(save_dir, f"{model_name}_losses.csv")
        self.png_path = os.path.join(save_dir, f"{model_name}_losses.png")
        self.metrics_png_path = os.path.join(save_dir, f"{model_name}_metrics.png")

        os.makedirs(save_dir, exist_ok=True)

        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[Optional[float]] = []
        self.wall_times: List[float] = []
        # Per-metric history. Same length as self.epochs; None where the
        # metric wasn't supplied at that epoch.
        self.metrics_history: Dict[str, List[Optional[float]]] = {
            k: [] for k in METRIC_COLUMNS
        }
        self._t0 = time.time()

        self._best_val: Optional[float] = None
        self._intervals_since_improve = 0
        self._last_overfit_warned_at: Optional[int] = None

        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", *METRIC_COLUMNS, "wall_time_s"]
            )

    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        """
        Record one epoch. `metrics` is an optional dict like
        {"ap50": 0.42, "mean_iou": 0.65}; only keys in METRIC_COLUMNS are
        recorded, others are silently ignored. Returns (should_stop, reason).
        """
        wall = time.time() - self._t0
        self.epochs.append(int(epoch))
        self.train_losses.append(float(train_loss))
        self.val_losses.append(None if val_loss is None else float(val_loss))
        self.wall_times.append(wall)
        for k in METRIC_COLUMNS:
            v = metrics.get(k) if metrics else None
            self.metrics_history[k].append(None if v is None else float(v))

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                "" if val_loss is None else f"{val_loss:.6f}",
                *[
                    "" if (metrics is None or metrics.get(k) is None)
                    else f"{float(metrics[k]):.6f}"
                    for k in METRIC_COLUMNS
                ],
                f"{wall:.1f}",
            ])

        is_chunk_boundary = (epoch % self.monitor_interval_epochs == 0)
        if is_chunk_boundary:
            self._render_loss_png()
            if any(any(v is not None for v in vs) for vs in self.metrics_history.values()):
                self._render_metrics_png()

        # ETA is the smoothed avg-time-per-epoch over the last 10 epochs
        # times the number of epochs left. Falls back to the overall avg
        # if we have fewer than 10 samples yet.
        eta_str = ""
        if self.total_epochs is not None and len(self.wall_times) >= 1:
            recent = min(10, len(self.wall_times))
            if recent >= 2:
                dt = self.wall_times[-1] - self.wall_times[-recent]
                per_epoch = dt / max(1, recent - 1)
            else:
                per_epoch = self.wall_times[-1] / max(1, len(self.wall_times))
            remaining = max(0, self.total_epochs - epoch)
            eta_str = f"  eta={_format_eta(per_epoch * remaining)}"

        epoch_label = (
            f"epoch {epoch:>4d}/{self.total_epochs}"
            if self.total_epochs is not None
            else f"epoch {epoch:>4d}"
        )
        pct = (
            f" ({100.0 * epoch / self.total_epochs:5.1f}%)"
            if self.total_epochs
            else ""
        )
        line = f"[Gbg-INST-TreeSeg] {epoch_label}{pct}  train={train_loss:.4f}"
        if val_loss is not None:
            line += f"  val={val_loss:.4f}"
        if metrics:
            for k in METRIC_COLUMNS:
                if metrics.get(k) is not None:
                    line += f"  {k}={metrics[k]:.4f}"
        line += f"{eta_str}  ({wall:.0f}s)"
        print(line)

        # Heuristic checks only run on chunk boundaries — otherwise we'd
        # spam warnings for every epoch in a divergent run.
        if not is_chunk_boundary:
            return False, ""

        overfitting, msg = self.detect_overfitting()
        if overfitting and self._last_overfit_warned_at != epoch:
            print(f"[Gbg-INST-TreeSeg] WARNING: {msg}")
            self._last_overfit_warned_at = epoch

        if val_loss is not None and self.early_stopping_patience is not None:
            if self._best_val is None or val_loss < self._best_val:
                self._best_val = val_loss
                self._intervals_since_improve = 0
            else:
                self._intervals_since_improve += 1
            if self._intervals_since_improve >= self.early_stopping_patience:
                return True, (
                    f"val loss has not improved on {self._best_val:.4f} for "
                    f"{self._intervals_since_improve} consecutive monitor intervals"
                )

        return False, ""

    def detect_overfitting(self) -> Tuple[bool, str]:
        """
        Return (overfitting, message). Looks at the last `overfit_window`
        chunk-boundary epochs only (we down-sample to monitor_interval_epochs
        granularity to avoid noise from per-epoch jitter).
        """
        # Pick out only the chunk-boundary samples
        boundary = [
            (e, t, v)
            for e, t, v in zip(self.epochs, self.train_losses, self.val_losses)
            if e % self.monitor_interval_epochs == 0
        ]
        boundary = [(e, t, v) for e, t, v in boundary if v is not None]
        if len(boundary) < self.overfit_window + 1:
            return False, ""

        recent = boundary[-(self.overfit_window + 1):]
        train_seq = [t for _, t, _ in recent]
        val_seq = [v for _, _, v in recent]

        train_mono_down = all(b < a for a, b in zip(train_seq[:-1], train_seq[1:]))
        val_mono_up = all(b > a for a, b in zip(val_seq[:-1], val_seq[1:]))

        if train_mono_down and val_mono_up:
            return True, (
                f"possible overfitting: across the last {self.overfit_window} "
                f"monitor intervals (epochs {recent[0][0]}→{recent[-1][0]}), "
                f"train loss fell {train_seq[0]:.4f}→{train_seq[-1]:.4f} "
                f"while val loss rose {val_seq[0]:.4f}→{val_seq[-1]:.4f}"
            )
        return False, ""

    def _matplotlib(self):
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            return None

    def _render_loss_png(self) -> None:
        plt = self._matplotlib()
        if plt is None:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.epochs, self.train_losses, label="train", color="C0", linewidth=1.5)
        val_pairs = [(e, v) for e, v in zip(self.epochs, self.val_losses) if v is not None]
        if val_pairs:
            ev, vv = zip(*val_pairs)
            ax.plot(ev, vv, label="val", color="C3", linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{self.model_name} — losses ({len(self.epochs)} epochs)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.png_path, dpi=120)
        plt.close(fig)

    def _render_metrics_png(self) -> None:
        plt = self._matplotlib()
        if plt is None:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        styles = {"ap50": ("AP@0.5", "C2"), "mean_iou": ("mean IoU", "C4")}
        plotted_any = False
        for k in METRIC_COLUMNS:
            pairs = [(e, v) for e, v in zip(self.epochs, self.metrics_history[k]) if v is not None]
            if not pairs:
                continue
            label, color = styles.get(k, (k, None))
            ex, vx = zip(*pairs)
            ax.plot(ex, vx, marker="o", linewidth=1.5, label=label, color=color)
            plotted_any = True
        if not plotted_any:
            plt.close(fig)
            return
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.set_title(f"{self.model_name} — eval metrics on val set")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.metrics_png_path, dpi=120)
        plt.close(fig)

    def finalize(self) -> None:
        """Force a final render of both PNGs — call this after the training
        loop ends to make sure the plots reflect the last epoch even if it
        wasn't on a chunk boundary."""
        self._render_loss_png()
        if any(any(v is not None for v in vs) for vs in self.metrics_history.values()):
            self._render_metrics_png()
