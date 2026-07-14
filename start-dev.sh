#!/bin/bash
# start-dev.sh — Menjalankan Video Upscale Pro dev server
# Gunakan ini jika 'npm run dev' gagal karena PATH tidak ada node

# Load NVM
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Pindah ke direktori project
cd "$(dirname "$0")"

echo "🚀 Starting Video Upscale Pro..."
node --watch server.js
