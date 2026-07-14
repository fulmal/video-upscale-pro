#!/bin/bash
# ─── Real-ESRGAN Local Server Launcher ────────────────────────────────────────
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "🍌 Real-ESRGAN Local Server"
echo "──────────────────────────"

# Check venv
if [ ! -f "$DIR/.venv/bin/python3" ]; then
    echo "❌ Virtual environment belum dibuat."
    echo "   Jalankan: python3 -m uv venv .venv --python 3.9 --seed"
    exit 1
fi

# Check weights
if [ ! -f "$DIR/weights/RealESRGAN_x4plus.pth" ]; then
    echo "⬇  Weight belum ada — mengunduh sekarang..."
    "$DIR/.venv/bin/python3" "$DIR/download_weights.py"
fi

# Auto-patch basicsr torchvision compatibility
bash "$DIR/patch_basicsr.sh"

echo ""
echo "🚀 Server berjalan di http://127.0.0.1:5001"
echo "   Tekan Ctrl+C untuk berhenti"
echo ""

exec "$DIR/.venv/bin/python3" "$DIR/server.py"
