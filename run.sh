#!/bin/bash

echo "🚀 Starting Agentic-RAG System..."

# Start Ollama in the background if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "📡 Starting Ollama server..."
    ollama serve &
    sleep 3
    echo "✓ Ollama started"
else
    echo "✓ Ollama is already running"
fi

# Activate virtual environment and start Streamlit
echo "🎨 Starting Streamlit app..."
source venv/bin/activate
streamlit run streamlit_app.py
