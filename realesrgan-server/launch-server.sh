#!/bin/bash
# Wrapper for LaunchAgent — starts the AI upscale server
# Called by: ~/Library/LaunchAgents/com.videoupscale.aiserver.plist
DIR="/Users/mac/Desktop/Video Upscale/realesrgan-server"
LOG="/Users/mac/Library/Logs/videoupscale-server.log"

echo "[$(date)] === launch-server.sh starting ===" >> "$LOG"

# Ensure basicsr patch is applied
if [ -f "$DIR/patch_basicsr.sh" ]; then
    bash "$DIR/patch_basicsr.sh" >> "$LOG" 2>&1
fi

echo "[$(date)] Starting server.py..." >> "$LOG"

# Start server — redirect to log
exec "$DIR/.venv/bin/python3" "$DIR/server.py" >> "$LOG" 2>&1
