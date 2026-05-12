#!/bin/bash
# Launch both backend and frontend servers
set -e

echo "Starting ToolBreak Harmonic Pipeline..."

# Start backend
echo "→ Starting backend on http://localhost:8000"
uvicorn backend.app:app --port 8000 &
BACKEND_PID=$!

# Wait for the backend to actually accept connections before launching the
# frontend, otherwise Vite's proxy fires off the browser's initial requests
# (config, folders, test_files, ...) before uvicorn is bound and they all
# fail with ECONNREFUSED. ~30s cap is plenty for cold torch/sklearn imports.
echo -n "→ Waiting for backend to be ready"
for i in {1..60}; do
  if curl -sf http://localhost:8000/api/config >/dev/null 2>&1; then
    echo " ✓"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo ""
    echo "Backend exited before becoming ready. Aborting."
    exit 1
  fi
  echo -n "."
  sleep 0.5
done

# Start frontend
echo "→ Starting frontend on http://localhost:5173"
cd frontend
npx --yes vite@5 &
FRONTEND_PID=$!
cd ..

echo ""
echo "Both servers running. Open http://localhost:5173"
echo "Press Ctrl+C to stop both."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
