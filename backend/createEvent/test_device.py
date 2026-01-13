
# test_device.py
import json
import time
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path

def prompt():
    url = input("Server endpoint: ").strip()
    path = input("Data file path: ").strip()
    testdata_root = Path.cwd() / "testdata"
    path = str((testdata_root / path).resolve())
    interval = float(input("Interval (seconds): ").strip())
    batch = int(input("Batch size per POST: ").strip())
    source = input("Source name [fake-device]: ").strip() or "fake-device"
    return url, path, interval, batch, source

def post_json(url, payload, timeout=10):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()

def main():
    url, path, interval, batch, source = prompt()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data_dict = json.load(fh)

    except FileNotFoundError:
        print(f"Data file not found: {path}")

    fh = data_dict["File_Header"]
    ch = int(fh["NumberOfChannels"])

    values = np.array([data_dict[f"Channel_{i}"]["Signal"] for i in range(1, ch + 1)])

    total = len(values[0])
    i = 0
    try:
        while i < total:
            end = min(i + batch, total)
            chunk = values[:, i:end].tolist()
            if not chunk or len(chunk[0]) == 0:
                break
            payload = {
                "source": source,
                "i": i,                    # batch start index
                "count": len(chunk),
                "data": chunk,             # your orchestrator will receive this in 'payload'
            }
            try:
                status, _ = post_json(url, payload)
                print(f"POST {i}-{i+len(chunk)-1}: HTTP {status}")
            except urllib.error.URLError as e:
                print(f"POST failed at i={i}: {e}")
                # Optional: backoff / retry; for now just break
                break

            i += len(chunk[0])
            if i < total:
                time.sleep(interval)  # wait until next interval
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print(f"Done. Sent {i} batches.")

if __name__ == "__main__":
    main()