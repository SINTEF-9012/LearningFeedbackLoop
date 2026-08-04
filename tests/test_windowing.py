"""C2: parametric windower — leakage-safe labelling + window-size parametrisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.agents.processing.windowing import (
    make_windows,
    WindowingParams,
    PHASE_NORMAL,
    PHASE_PRE_EVENT,
    PHASE_EVENT,
    PHASE_IDLE,
    LABEL_POSITIVE,
    LABEL_NEGATIVE,
)


def _series(n=600, fs=1.0, op="OP1"):
    """A clean continuous series with one event at t=300 and a pre-event horizon."""
    t = np.arange(n, dtype=float) / fs
    rows = []
    event_t = 300.0
    for ti in t:
        tte = event_t - ti
        if ti == event_t:
            phase = PHASE_EVENT
        elif 0 < tte <= 60:
            phase = PHASE_PRE_EVENT
        else:
            phase = PHASE_NORMAL
        rows.append({
            "operation_id": op,
            "t": ti,
            "sig": float(np.sin(ti / 10.0)),
            "phase": phase,
            "time_to_event_s": tte if tte > 0 else np.nan,
            "event_id": "E1" if tte > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def test_nonoverlapping_default_and_feature_columns():
    df = _series()
    out = make_windows(df, WindowingParams(window_s=60, stride_s=60, horizon_s=60))
    assert not out.empty
    # non-overlapping windows do not share samples
    starts = sorted(out["window_start_t"])
    assert all(b - a >= 60 for a, b in zip(starts, starts[1:]))
    # aggregated feature columns exist
    for suf in ("mean", "std", "min", "max", "slope"):
        assert f"sig_{suf}" in out.columns


def test_window_straddling_event_is_dropped():
    df = _series()
    out = make_windows(df, WindowingParams(window_s=60, stride_s=1, horizon_s=60))
    # no surviving window may contain the event sample (t=300)
    for _, r in out.iterrows():
        assert not (r["window_start_t"] <= 300.0 <= r["window_end_t"] and r["label"] == LABEL_POSITIVE
                    and r["window_end_t"] == 300.0)
    # a window whose span includes t=300 (event) must be absent entirely
    spans_event = out[(out["window_start_t"] <= 300.0) & (out["window_end_t"] >= 300.0)]
    assert spans_event.empty


def test_positive_requires_end_within_horizon():
    df = _series()
    # horizon 30: only windows ending within 30s before the event are positive
    out = make_windows(df, WindowingParams(window_s=20, stride_s=1, horizon_s=30, negative_margin_s=120))
    pos = out[out["label"] == LABEL_POSITIVE]
    assert not pos.empty
    # every positive ends within [0,30]s before the event (t=300)
    assert ((300.0 - pos["window_end_t"]) <= 30.0 + 1e-6).all()
    assert ((300.0 - pos["window_end_t"]) >= 0.0).all()


def test_gap_shifts_positive_window_back():
    df = _series()
    # gap=10: a positive window must end >=10s before the event
    out = make_windows(df, WindowingParams(window_s=20, stride_s=1, gap_s=10, horizon_s=30))
    pos = out[out["label"] == LABEL_POSITIVE]
    assert not pos.empty
    assert ((300.0 - pos["window_end_t"]) >= 10.0 - 1e-6).all()


def test_negatives_are_far_from_event():
    df = _series()
    out = make_windows(df, WindowingParams(window_s=30, stride_s=30, horizon_s=60, negative_margin_s=120))
    neg = out[out["label"] == LABEL_NEGATIVE]
    assert not neg.empty
    # negatives are either comfortably BEFORE the event (>=margin lead) or AFTER
    # it (no upcoming event). None may sit in the pre-event/ambiguous band.
    before = neg["window_end_t"] <= (300.0 - 120.0) + 1e-6
    after = neg["window_end_t"] > 300.0
    assert (before | after).all()


def test_idle_samples_block_a_window():
    df = _series()
    # inject idle in [100,140)
    df.loc[(df["t"] >= 100) & (df["t"] < 140), "phase"] = PHASE_IDLE
    out = make_windows(df, WindowingParams(window_s=30, stride_s=1, horizon_s=60))
    # no window may span the idle region
    spans_idle = out[(out["window_start_t"] < 140.0) & (out["window_end_t"] >= 100.0)]
    assert spans_idle.empty


def test_window_size_changes_row_count_and_balance():
    df = _series()
    small = make_windows(df, WindowingParams(window_s=10, stride_s=10, horizon_s=60))
    big = make_windows(df, WindowingParams(window_s=60, stride_s=60, horizon_s=60))
    # smaller non-overlapping windows -> more rows (the core "variable size" win)
    assert len(small) > len(big)


def test_event_id_present_for_block_bootstrap():
    df = _series()
    out = make_windows(df, WindowingParams(window_s=20, stride_s=5, horizon_s=40))
    assert "event_id" in out.columns
    # positives carry the real event id; negatives carry a per-block id
    pos = out[out["label"] == LABEL_POSITIVE]
    assert (pos["event_id"] == "E1").all()
    neg = out[out["label"] == LABEL_NEGATIVE]
    assert neg["event_id"].astype(str).str.startswith("neg::").all()


def test_windows_never_cross_operations():
    a = _series(op="OP1")
    b = _series(op="OP2")
    b["t"] = b["t"]  # same local time axis; grouping must keep them separate
    out = make_windows(pd.concat([a, b], ignore_index=True),
                       WindowingParams(window_s=30, stride_s=30, horizon_s=60))
    assert set(out["operation_id"]) == {"OP1", "OP2"}


# --------------------------------------------------------------------------
# C4: overlap-aware (block) bootstrap
# --------------------------------------------------------------------------

def test_block_bootstrap_wider_than_naive_under_autocorrelation():
    """Block bootstrap by event must be WIDER (more honest) than naive
    per-window bootstrap when windows are clustered within events."""
    from backend.agents.processing.window_stats import (
        block_bootstrap_metric_ci, f1_score, n_effective,
    )
    rng = np.random.RandomState(0)
    # 6 events; each contributes 20 near-identical windows (autocorrelated).
    y_true, y_pred, groups = [], [], []
    for ev in range(6):
        correct = ev % 2 == 0          # whole events are right or wrong together
        for _ in range(20):
            label = 1
            pred = 1 if correct else 0
            y_true.append(label); y_pred.append(pred); groups.append(f"E{ev}")
    # add some true negatives so f1 is defined
    for j in range(40):
        y_true.append(0); y_pred.append(0); groups.append(f"neg{j}")

    # naive (each window its own group) vs block (by event)
    naive_groups = list(range(len(y_true)))
    _, nlo, nhi = block_bootstrap_metric_ci(y_true, y_pred, naive_groups, seed=1)
    _, blo, bhi = block_bootstrap_metric_ci(y_true, y_pred, groups, seed=1)
    assert (bhi - blo) > (nhi - nlo)          # block CI is wider
    assert n_effective(groups) < len(y_true)  # fewer effective samples


def test_assert_non_overlapping_guard():
    from backend.agents.processing.window_stats import assert_non_overlapping
    assert_non_overlapping([0.0, 60.0, 120.0], window_s=60.0)  # ok
    with pytest.raises(ValueError):
        assert_non_overlapping([0.0, 30.0, 60.0], window_s=60.0)  # overlap


def test_block_bootstrap_preserves_continuous_scores_for_auc():
    """Regression: block bootstrap must NOT int-cast y_pred — continuous AUC
    scores (0.9) would collapse to 0 and force the metric to 0.5."""
    from sklearn.metrics import roc_auc_score
    from backend.agents.processing.window_stats import block_bootstrap_metric_ci
    rng = np.random.RandomState(0)
    y, s, g = [], [], []
    for ev in range(10):
        for _ in range(5):
            y.append(1); s.append(rng.uniform(0.6, 1.0)); g.append(f"E{ev}")
    for j in range(200):
        y.append(0); s.append(rng.uniform(0.0, 0.5)); g.append(f"n{j}")
    pt, lo, hi = block_bootstrap_metric_ci(
        y, s, g,
        metric_fn=lambda yt, ys: roc_auc_score(yt, ys) if len(set(yt)) > 1 else 0.5,
        n_boot=200,
    )
    assert pt > 0.9 and lo > 0.5


# --------------------------------------------------------------------------
# D2: windower emits WindowMetrics that revive the anomaly-deviation rule
# --------------------------------------------------------------------------

def test_window_metrics_warm_baseline_detects_anomaly():
    """D2: a per-channel WindowMetrics built from a window, fed to the scorer's
    rolling baseline, must flag an anomalous window above the z threshold —
    the input the anomaly-deviation rule needs (it is inert when metrics=None)."""
    import tempfile, os
    from backend.agents.memory.scorer import SignificanceScorer, SignificanceConfig
    from backend.agents.processing.windowing import window_to_window_metrics

    rng = np.random.RandomState(0)
    s = SignificanceScorer(config=SignificanceConfig(),
                           priors_path=os.path.join(tempfile.mkdtemp(), "p.json"))
    sess = "op1"
    # warm the baseline with many NORMAL windows
    for _ in range(40):
        w = pd.DataFrame({"chan": rng.normal(0.0, 1.0, size=60)})
        s.update_baseline(sess, window_to_window_metrics(w, ["chan"]))

    baseline = s._session_baselines[sess]
    assert baseline.is_ready

    # an anomalous window sits far from the baseline mean
    anom = pd.DataFrame({"chan": rng.normal(20.0, 1.0, size=60)})
    z = baseline.max_deviation(window_to_window_metrics(anom, ["chan"]))
    assert z is not None and z > s.config.anomaly_z_threshold  # rule would trigger

    # a normal window does NOT
    norm = pd.DataFrame({"chan": rng.normal(0.0, 1.0, size=60)})
    z_norm = baseline.max_deviation(window_to_window_metrics(norm, ["chan"]))
    assert z_norm is not None and z_norm < s.config.anomaly_z_threshold
