"""Session-relative calibration of one-class model scores.

Problem (measured 2026-07-07, `docs/IMPROVEMENT_PLANS_2026-07-07.md` §1.2):
a seed model trained on other sessions produces bimodally shifted scores on a
new session — ~29 % of all windows cross the alert threshold (precision 0.048)
because the *absolute* score scale does not transfer across sessions/parts.

Fix: express the model signal relative to the session's own recent score
distribution. `SessionScoreCalibrator` keeps a rolling window of raw scores and
returns

- a **robust z** — deviation from the rolling median in MAD units (median/MAD
  tolerate up to ~50 % contaminated windows, so sustained anomalies do not
  silently become the new normal within the window horizon), and
- a **rolling percentile** in [0, 1] — the fraction of recent scores below the
  current one, a session-relative severity that self-adjusts to shift.

During the warm-up (first `warmup` scores) the calibrator is blind and returns
neutral values — callers must expect a documented no-model-alert window at
session start (patterns still work there).
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass


_MAD_TO_SIGMA = 1.4826  # MAD → σ for a normal distribution


@dataclass
class CalibratedScore:
    """Session-relative view of one raw model score."""
    z: float            # robust z (0.0 during warm-up)
    percentile: float   # rolling percentile in [0, 1] (0.0 during warm-up)
    warmed_up: bool


class SessionScoreCalibrator:
    """Rolling robust normalization of one-class scores within a session.

    Parameters
    ----------
    warmup : int
        Scores to observe before emitting non-neutral output.
    window : int
        Rolling window length (scores). With 60 s windows the default keeps
        roughly the last 8 hours of session context.
    """

    def __init__(self, warmup: int = 30, window: int = 480):
        if warmup < 2:
            raise ValueError("warmup must be >= 2")
        self._warmup = int(warmup)
        self._buf: deque[float] = deque(maxlen=int(window))
        self._sorted: list[float] = []  # kept in step with _buf for O(log n) rank

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def warmed_up(self) -> bool:
        return len(self._buf) >= self._warmup

    def update(self, score: float) -> CalibratedScore:
        """Calibrate `score` against the window so far, then add it."""
        score = float(score)
        result = self._calibrate(score)
        if len(self._buf) == self._buf.maxlen:
            evicted = self._buf[0]
            del self._sorted[bisect_left(self._sorted, evicted)]
        self._buf.append(score)
        insort(self._sorted, score)
        return result

    def _calibrate(self, score: float) -> CalibratedScore:
        if not self.warmed_up:
            return CalibratedScore(z=0.0, percentile=0.0, warmed_up=False)
        n = len(self._sorted)
        median = self._sorted[n // 2] if n % 2 else 0.5 * (
            self._sorted[n // 2 - 1] + self._sorted[n // 2]
        )
        # MAD via the sorted deviations' middle element (exact enough here and
        # avoids re-sorting): compute directly.
        deviations = sorted(abs(v - median) for v in self._sorted)
        mad = deviations[n // 2] if n % 2 else 0.5 * (
            deviations[n // 2 - 1] + deviations[n // 2]
        )
        sigma = max(_MAD_TO_SIGMA * mad, 1e-6)
        z = (score - median) / sigma
        percentile = bisect_left(self._sorted, score) / n
        return CalibratedScore(z=z, percentile=percentile, warmed_up=True)
