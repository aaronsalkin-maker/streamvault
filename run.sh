#!/usr/bin/env bash
# StreamVault - Setup & Run Script
set -e

echo "🎬 StreamVault Setup"
echo "────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "❌ Python 3 is required. Install from https://python.org"
  exit 1
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
  echo "⚠️  FFmpeg not found. Transcoding will be unavailable."
  echo "   Install: https://ffmpeg.org/download.html"
  echo "   macOS:   brew install ffmpeg"
  echo "   Ubuntu:  sudo apt install ffmpeg"
fi

# Create virtual environment if missing
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

# Activate & install
source venv/bin/activate
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Create media folder
mkdir -p media thumbnails

echo ""
echo "✅ Ready!"
echo ""
echo "📁 Drop your media files into: ./media/"
echo "   (or add library paths via the Admin panel)"
echo ""
echo "🌐 Starting server at http://localhost:8000"
echo "🔑 Default login: admin / admin"
echo "   ⚠️  Change the password after first login!"
echo ""

# Run
python3 main.py
