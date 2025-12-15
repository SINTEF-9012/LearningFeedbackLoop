import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
            print("PROCESS", event.get("source"), event.get("i"))
        finally:
            queue.task_done()


async def publish_flag(flag):
    # placeholder to push a flag to downstream (HTTP/MQTT)
    print("PUBLISH FLAG", flag)


async def main(run_seconds: float = 60.0, run_fake: bool = False):
    q = asyncio.Queue(maxsize=100)

    # instantiate turnmill_process from a json file if present
    init_path = Path("testdata") / "init.json"
    if init_path.exists():
        init = json.loads(init_path.read_text())
        tm = TurnmillProcess(init)
    else:
        tm = TurnmillProcess({})

    # attach to app for access from endpoint
    app.state.queue = q
    app.state.tm = tm

    # start the FastAPI server in background
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, loop="asyncio", lifespan="on")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # start worker(s)
    workers = [asyncio.create_task(worker(q, tm)) for _ in range(2)]

    # optionally start the test device (keeps running until cancelled)
    fake_task = None
    if run_fake:
        from .fake_device import fake_device  # relative import; file provided below
        fake_task = asyncio.create_task(fake_task := fake_device("http://127.0.0.1:8000/ingest", interval=0.1))

    try:
        await asyncio.sleep(run_seconds)
    finally:
        if fake_task:
            fake_task.cancel()
        for w in workers:
            w.cancel()
        server.should_exit = True
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main(run_seconds=60.0, run_fake=False))
    except KeyboardInterrupt:
        print("shutdown")