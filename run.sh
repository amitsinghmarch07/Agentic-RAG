#!/bin/bash

echo "🚀 Starting Agentic-RAG System..."
echo ""

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

echo ""

# Start Ollama in the background if not running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "📡 Starting Ollama server..."
    ollama serve &
    sleep 3
    echo "✓ Ollama started"
else
    echo "✓ Ollama is already running"
fi

echo ""

# Start Streamlit app
echo "🎨 Starting Streamlit app..."
echo "📱 Open your browser at: http://localhost:8501"
echo ""
streamlit run streamlit_app.py
