#!/bin/bash
# Launch both backend and frontend servers
set -e

echo "Starting ToolBreak Harmonic Pipeline..."

# Start backend
echo "→ Starting backend on http://localhost:8000"
uvicorn backend.app:app --port 8000 &
BACKEND_PID=$!

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
