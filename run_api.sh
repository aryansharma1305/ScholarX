#!/bin/bash
# Quick start script for ScholarX FastAPI server

echo "Starting ScholarX API server..."
echo "API docs will be available at http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"
uvicorn api.server:app --host 0.0.0.0 --port 8000
