#!/bin/bash
# Quick start script for Streamlit app

echo "🚀 Starting ScholarX Streamlit App..."
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
streamlit run app.py

