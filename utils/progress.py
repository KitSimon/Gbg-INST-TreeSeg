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
Progress + ETA helpers for long-running stages.

`EtaTracker` estimates remaining time in one of two modes:

* ``mode="window"`` (default) — unweighted mean of recent per-unit
  durations over a sliding window of size ``window``.  During warmup
  (fewer than ``window`` ticks) it falls back to the global cumulative
  mean (``elapsed / done``); at exactly ``window`` ticks the two
  formulas are arithmetically identical, so the transition is seamless.

* ``mode="ema"`` — global exponentially-weighted moving average of
  ``dt`` with smoothing factor ``alpha``.  Recent ticks count
  exponentially more than old ones; no hard window edge.  Storage is
  one float.  Effective memory ≈ ``1/alpha`` ticks.

Each tick carries an optional *work weight* (default 1).  It is stored
for downstream diagnostics (e.g. average crowns per compute tile) but
does not influence ``eta_seconds()``: under the i.i.d. assumption either
estimator already makes, the unweighted form is the unbiased estimator
of expected time per tile, and re-weighting ``dt`` by ``w``
double-counts the heaviness already encoded in ``dt`` itself.

Optional named buckets (e.g. ``"skip"`` vs ``"compute"``) are tracked for
accounting purposes but do not affect ``eta_seconds()``.

``format_int`` formats an integer with a thin-space thousands separator.
``format_duration`` renders seconds as ``1h23m``, ``12m04s``, or ``45s``.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Optional


def format_int(n: int) -> str:
    """Thin-space thousands separator, e.g. 217600 -> '217 600'."""
    return f"{int(n):,}".replace(",", " ")


def format_duration(seconds: Optional[float]) -> str:
    """Compact duration: '1h23m', '12m04s', '45s', or '—' when unknown."""
    if seconds is None:
        return "—"
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


class EtaTracker:
    """
    Sliding-window or EMA ETA estimator.

    Call ``tick(bucket, weight)`` once per completed unit;
    ``eta_seconds()`` returns the current estimate (None until any tick).

    Modes
    -----
    * ``mode="window"`` (default): rate is the plain mean of ``dt`` over
      the last ``window`` ticks::

          rate = Σ(dt_i) / N

      During warmup (fewer than ``window`` ticks) uses the global
      cumulative mean (``elapsed / done``).  At exactly ``window`` ticks
      this equals the unweighted window mean, so the transition into
      steady state is seamless.

    * ``mode="ema"``: rate is the global exponentially-weighted moving
      average of ``dt`` with smoothing factor ``alpha``::

          rate_n = (1 − α) · rate_{n−1} + α · dt_n,   rate_1 = dt_1

      No warmup branch and no hard window edge; old ticks fade
      exponentially.  Effective memory ≈ ``1/alpha`` ticks (α=0.2 ≈ 5,
      α=0.01 ≈ 100, α=0.001 ≈ 1000).  Early estimates are noisier than
      window mode because the cumulative-mean warmup is absent.

    Both forms are unbiased estimators of expected time per tile under
    the i.i.d. assumption.  Per-tick ``weight`` is stored for diagnostics
    but intentionally does not enter either formula — re-weighting
    ``dt`` by ``w`` would double-count the cost already encoded in
    ``dt``.

    Adaptivity
    ----------
    Sensitivity to workload changes is controlled by ``window`` (window
    mode) or ``alpha`` (ema mode).  Smaller windows / larger alpha
    respond faster to dense/sparse transitions at the cost of higher
    variance.

    Buckets
    -------
    The ``bucket`` argument is tracked for accounting (``_counts``,
    ``_ema``) but does not affect ``eta_seconds()``.
    """

    def __init__(
        self,
        total: int,
        alpha: float = 0.001,
        window: int = 1000,
        mode: str = "window",
    ):
        if total <= 0:
            raise ValueError(f"total must be positive, got {total}")
        if mode not in ("window", "ema"):
            raise ValueError(
                f"mode must be 'window' or 'ema', got {mode!r}"
            )
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.total = int(total)
        self.alpha = float(alpha)
        self.mode = mode
        self.done = 0
        self._start = time.monotonic()
        self._last_tick = self._start
        self._ema: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._ema_dt: Optional[float] = None
        # Each entry is (dt_seconds, work_weight ≥ 1).
        self._recent: deque = deque(maxlen=window)

    def tick(self, bucket: str = "default", weight: float = 1.0) -> None:
        """Record one completed unit.

        Args:
            bucket: Accounting label (e.g. ``"skip"`` or ``"compute"``).
                Has no effect on ``eta_seconds()``.
            weight: Work units this tick represents (stored for
                diagnostics only — does not affect ``eta_seconds()``).
                Values below 1 are clamped to 1.
        """
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        prev = self._ema.get(bucket)
        self._ema[bucket] = dt if prev is None else (1.0 - self.alpha) * prev + self.alpha * dt
        self._counts[bucket] = self._counts.get(bucket, 0) + 1
        self._ema_dt = (
            dt
            if self._ema_dt is None
            else (1.0 - self.alpha) * self._ema_dt + self.alpha * dt
        )
        self._recent.append((dt, max(1.0, float(weight))))
        self.done += 1

    def eta_seconds(self) -> Optional[float]:
        if self.done >= self.total:
            return 0.0
        if self._ema_dt is None:
            return None
        remaining = self.total - self.done
        if self.mode == "ema":
            rate = self._ema_dt
        elif len(self._recent) >= self._recent.maxlen:
            # Steady state: unweighted sliding-window mean.
            rate = sum(dt for dt, _ in self._recent) / len(self._recent)
        else:
            # Warmup: global cumulative mean. Equals the unweighted
            # window mean at the moment the window first fills, so the
            # transition into steady state is continuous.
            rate = self.elapsed_seconds() / self.done
        return remaining * rate

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start
