import asyncio

import pytest

from backend.events import bus, publish_learning, subscribe_learnings
from backend.ingestion.schema import LearningEnvelope


@pytest.mark.asyncio
async def test_publish_learning_fans_out_to_global_and_session_channels():
    session_id = "learning-session"
    session_queue = subscribe_learnings(session_id)
    global_queue = subscribe_learnings()
    envelope = LearningEnvelope(
        kind="alert",
        ts_unix=10.0,
        session_id=session_id,
        payload={"status": "ok"},
    )

    try:
        await publish_learning(envelope)

        assert await asyncio.wait_for(session_queue.get(), timeout=1.0) is envelope
        assert await asyncio.wait_for(global_queue.get(), timeout=1.0) is envelope
    finally:
        bus.unsubscribe(f"learnings.{session_id}", session_queue)
        bus.unsubscribe("learnings", global_queue)