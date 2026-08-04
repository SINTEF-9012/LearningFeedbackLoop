"""Largest-Triangle-Three-Buckets (LTTB) downsampling.

Agent Q (2026-04-24). Used by streaming plot endpoints to cap the
number of points sent to the browser without losing visual shape.

Reference: Sveinn Steinarsson, "Downsampling Time Series for Visual
Representation" (2013). The algorithm partitions the input into
``threshold - 2`` equal-size buckets (plus first/last points kept
verbatim), then picks the point in each bucket that forms the
largest triangle with the previous chosen point and the average of
the next bucket.

Numerically stable on float64; input is coerced. The function never
mutates its inputs.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def lttb(
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
    threshold: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample ``(xs, ys)`` to at most ``threshold`` points via LTTB.

    Args:
        xs: Monotonic x-axis (typically time). Length N.
        ys: Y values. Length N. Must match ``xs`` length.
        threshold: Desired point count. Must be >= 3; if <= 3 or
            >= N, the inputs are returned unchanged (as float64 arrays).

    Returns:
        Tuple ``(xs_out, ys_out)`` of 1-D float64 arrays with length
        min(N, threshold).

    Raises:
        ValueError: if ``xs`` and ``ys`` have different lengths.
    """
    xs_arr = np.asarray(xs, dtype=np.float64).ravel()
    ys_arr = np.asarray(ys, dtype=np.float64).ravel()
    if xs_arr.shape[0] != ys_arr.shape[0]:
        raise ValueError(
            f"lttb: xs and ys must have equal length, got {xs_arr.shape[0]} vs {ys_arr.shape[0]}"
        )
    n = xs_arr.shape[0]
    if threshold >= n or threshold <= 2:
        return xs_arr.copy(), ys_arr.copy()

    # Bucket size excludes the first and last sample which are always kept.
    bucket_size = (n - 2) / (threshold - 2)

    sampled_x = np.empty(threshold, dtype=np.float64)
    sampled_y = np.empty(threshold, dtype=np.float64)

    # First point always included.
    sampled_x[0] = xs_arr[0]
    sampled_y[0] = ys_arr[0]
    a = 0  # index of previously selected point

    for i in range(threshold - 2):
        # Range for the next bucket (used to compute the averaged "anchor").
        next_start = int(np.floor((i + 1) * bucket_size)) + 1
        next_end = min(int(np.floor((i + 2) * bucket_size)) + 1, n)
        if next_end <= next_start:
            next_end = next_start + 1
        avg_x = xs_arr[next_start:next_end].mean()
        avg_y = ys_arr[next_start:next_end].mean()

        # Range for the current bucket.
        cur_start = int(np.floor(i * bucket_size)) + 1
        cur_end = int(np.floor((i + 1) * bucket_size)) + 1
        if cur_end <= cur_start:
            cur_end = cur_start + 1

        # Vectorised triangle area computation: the magnitude of the
        # cross product (ax - px)(cy - ay) - (ay - py)(cx - ax) where
        # p = previous point (a), a = candidate, c = avg of next bucket.
        px = xs_arr[a]
        py = ys_arr[a]
        cand_x = xs_arr[cur_start:cur_end]
        cand_y = ys_arr[cur_start:cur_end]
        area = np.abs(
            (px - avg_x) * (cand_y - py) - (px - cand_x) * (avg_y - py)
        )
        local = int(np.argmax(area))
        chosen = cur_start + local
        sampled_x[i + 1] = xs_arr[chosen]
        sampled_y[i + 1] = ys_arr[chosen]
        a = chosen

    # Last point always included.
    sampled_x[-1] = xs_arr[-1]
    sampled_y[-1] = ys_arr[-1]
    return sampled_x, sampled_y


__all__ = ["lttb"]
