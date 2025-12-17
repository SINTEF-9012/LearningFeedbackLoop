from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
import json
import uvicorn
import os 
import numpy as np
from pathlib import Path
import time

start_time = None

# replace with your real import
from turnmill_process import TurnmillProcess  # assumes this exists on PYTHONPATH

app = FastAPI()


@app.post("/ingest")
async def ingest(request: Request):
    payload = await request.json()
    q = getattr(request.app.state, "queue", None)
    if q is None:
        return JSONResponse({"error": "service not ready"}, status_code=503)
    await q.put(payload)
    return {"accepted": True}


async def worker(queue, tm):
    while True:
        event = await queue.get()
        try:
            # SEND EVENT TO ANALYSIS MODULE (turnmill_process)
            # e.g. result = await asyncio.to_thread(tm.process, event)
            # if result: await publish_flag(result)
            # elapsed time since server start (perf_counter is monotonic)
            app_start = getattr(app.state, "start_time", None)
            elapsed = None if app_start is None else time.perf_counter() - app_start
            print("PROCESS", event.get("source"), event.get("i"), "elapsed=", elapsed)

            # extract values (event['data'] may be awaitable depending on source)
            values = event.get("data")
            if asyncio.iscoroutine(values):
                values = await values
            for idx in range(len(values[0])):
                timestamp = elapsed
                datapoint = [values[ch][idx] for ch in range(tm.nChannels)]
                tm.add_datapoint(timestamp, datapoint)

        finally:
            queue.task_done()


async def publish_flag(flag):
    # placeholder to push a flag to downstream (HTTP/MQTT)
    print("PUBLISH FLAG", flag)


async def main(source_file: str):
    q = asyncio.Queue(maxsize=100)

    # instantiate turnmill_process from a json file if present
    init_path = os.path.join(Path.cwd(), "testdata", source_file)
    init = None
    try:
        with open(init_path, "r", encoding="utf-8") as fh:
            init = json.load(fh)
    except FileNotFoundError:
        print(f"Initialization file not found: {init_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {init_path}: {e}")

    # if init is None, TurnmillProcess should decide how to initialize from defaults
    tm = TurnmillProcess(init, nChannels=3)

    # attach to app for access from endpoint
    app.state.queue = q
    app.state.tm = tm
    # record server start time (monotonic)
    app.state.start_time = time.perf_counter()
    print("Server start_time set:", app.state.start_time)

    # start the FastAPI server in background
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, loop="asyncio", lifespan="on")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    workers = [asyncio.create_task(worker(q, tm)) for _ in range(2)]

    try:
        # sleep forever (any of these works)
        await asyncio.Future()          # never completes
        # or: while True: await asyncio.sleep(3600)
    except asyncio.CancelledError:
        # gets triggered if outer code cancels main()
        pass
    finally:
        # graceful shutdown
        for w in workers:
            w.cancel()
        # ask uvicorn to exit
        server.should_exit = True
        # optional: finish in-flight queue items before exiting
        # await q.join()
        await asyncio.gather(*workers, return_exceptions=True)

if __name__ == "__main__":
    try:
        testdata_root = Path.cwd() / "testdata"
        if not testdata_root.exists():
            print(f"testdata directory not found: {testdata_root}")
            raise SystemExit(1)

        while True:
            source = input("Enter Source File (e.g. Diameter16Z4endmill/0.json): ").strip()
            candidate = testdata_root / source
            if candidate.exists():
                break
            print(f"File not found: {candidate}. Try again.")

        asyncio.run(main(source))
    except KeyboardInterrupt:
        print("shutdown")