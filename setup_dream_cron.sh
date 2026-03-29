#!/bin/bash
# Demis Dream Cycle Setup
# Run once to schedule Dream Cycle

echo "🌙 Demis Dream Cycle Setup"
echo "================================"
echo ""

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "❌ curl not found. Please install curl."
    exit 1
fi

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  OPENROUTER_API_KEY not set. Deep research will be skipped."
    echo "    Set it with: export OPENROUTER_API_KEY=your_key"
fi

# Create dream directory
DREAM_DIR="$HOME/demis_dreams"
mkdir -p "$DREAM_DIR/output"

echo "✅ Dream directory: $DREAM_DIR"
echo ""

# ============================================
# Schedule Cron Jobs
# ============================================

echo "📅 Scheduling Dream Cycle..."

# Remove existing dream cron jobs
crontab -l 2>/dev/null | grep -v "demis_dream\|demis_morning" | crontab - 2>/dev/null || true

# Dream Cycle: Every night at 23:00 KST (14:00 UTC)
DREAM_CRON="0 14 * * * $HOME/demis_dream.sh >> $HOME/demis_dreams/dream_cron.log 2>&1"

# Morning check: Every morning at 07:00 KST (22:00 UTC previous day)
MORNING_CRON="0 22 * * * echo 'Morning check: Dream results ready' && ls -la $HOME/demis_dreams/output/*.md 2>/dev/null | tail -3"

# Add to crontab
(crontab -l 2>/dev/null; echo "$DREAM_CRON"; echo "$MORNING_CRON") | crontab -

echo "✅ Dream Cycle scheduled: Every night at 23:00 KST"
echo ""

# ============================================
# Show Current Crontab
# ============================================
echo "📋 Current crontab:"
crontab -l 2>/dev/null || echo "(empty)"
echo ""

# ============================================
# Test Run Option
# ============================================
echo ""
read -p "Run Dream Cycle now for testing? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Running Dream Cycle test..."
    $HOME/demis_dream.sh
fi

echo ""
echo "================================"
echo "🌙 Dream Cycle Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "  1. Dream runs automatically at 23:00 KST"
echo "  2. Check results: ls ~/demis_dreams/output/"
echo "  3. View log: cat ~/demis_dreams/dream_log.txt"
echo ""
echo "To remove:"
echo "  crontab -e  # then delete dream lines"
