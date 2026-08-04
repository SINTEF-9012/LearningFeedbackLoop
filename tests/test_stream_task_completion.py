from __future__ import annotations

import asyncio

import pytest

from backend.fft_streamer import fft_stream_task
from backend.inference_streamer import inference_stream_task


@pytest.mark.asyncio
async def test_inference_stream_task_exits_when_session_completed_before_window(monkeypatch):
    session = {
        "session_id": "short-inference",
        "data": {"A": [1.0, 2.0]},
        "metadata": {"sample_frequency": 1.0},
        "config": {"channels": ["A"], "speed": 1.0},
        "inference_config": {"window_samples": 10, "stride_samples": 5},
        "position": 2,
        "running": False,
        "running_inference": True,
        "inference_subscribers": [],
        "inference_task": None,
    }

    await asyncio.wait_for(inference_stream_task(session), timeout=1.0)

    assert session["running_inference"] is False
    assert session["inference_task"] is None


@pytest.mark.asyncio
async def test_fft_stream_task_exits_when_session_completed_before_window():
    session = {
        "session_id": "short-fft",
        "data": {"A": [1.0, 2.0]},
        "metadata": {"sample_frequency": 1.0},
        "config": {"channels": ["A"], "speed": 1.0},
        "fft_config": {"nfft": 8, "overlap": 0.5, "inherit_speed": True},
        "position": 2,
        "running": False,
        "running_fft": True,
        "fft_subscribers": [],
        "fft_task": None,
    }

    await asyncio.wait_for(fft_stream_task(session), timeout=1.0)

    assert session["running_fft"] is False
    assert session["fft_task"] is None