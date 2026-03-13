#!/bin/bash
# Kills any process using port 8080 (Loom WASM Server)

PORT=8080
PID=$(lsof -t -i:$PORT)

if [ -n "$PID" ]; then
  echo "Killing process $PID using port $PORT..."
  kill -9 $PID
  echo "Process killed."
else
  echo "No process found on port $PORT."
fi
