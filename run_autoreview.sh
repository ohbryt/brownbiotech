#!/bin/bash
# BrownBioTech Autoreview Runner
# Loads .env and runs the autoreview loop

cd "$(dirname "$0")"

# Load .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Verify API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ OPENROUTER_API_KEY not found in .env"
    exit 1
fi

echo "✅ API key loaded: ${OPENROUTER_API_KEY:0:20}..."

# Run the autoreview
python3 autoreview_loop_v2.py
