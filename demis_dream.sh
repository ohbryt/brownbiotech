#!/bin/bash
# Demis Dream Cycle 🌙
# Every night at 23:00 KST
# AI self-improvement while you sleep

set -e

DATE=$(date +%Y-%m-%d_%H-%M)
DREAM_DIR="$HOME/demis_dreams"
LOG_FILE="$DREAM_DIR/dream_log.txt"
OUTPUT_DIR="$DREAM_DIR/output"
mkdir -p "$DREAM_DIR" "$OUTPUT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🌙 Demis Dream Cycle started: $DATE"

# ============================================
# PHASE 1: GitHub/AI News Scan (Haiku - cheap)
# ============================================
log "📰 Phase 1: Scanning AI news..."

cd /Users/ocm/.openclaw/workspace

# Scan BrownBioTech repo updates
if [ -d "brownbiotech" ]; then
    cd brownbiotech
    git fetch origin 2>/dev/null
    LATEST=$(git log --oneline -5 origin/main 2>/dev/null || echo "No updates")
    log "BrownBioTech recent: $LATEST"
    cd ..
fi

# Scan for new AI/ML papers on GitHub
GITHUB_NEWS=$(curl -s "https://api.github.com/search/repositories?q=ai+OR+machine-learning+OR+drug-discovery&sort=updated&per_page=5" 2>/dev/null | \
    jq -r '.items[:3] | .[] | "\(.full_name): \(.description)"' 2>/dev/null || echo "GitHub scan skipped")

log "GitHub trending:\n$GITHUB_NEWS"

# Save Phase 1 results
echo "$GITHUB_NEWS" > "$OUTPUT_DIR/phase1_github_$DATE.txt"

# ============================================
# PHASE 2: Today Performance Review
# ============================================
log "📊 Phase 2: Today performance review..."

# Review today's commits
TODAY_COMMITS=$(git log --since="00:00" --oneline --all 2>/dev/null | head -20 || echo "No commits today")
log "Today's commits:\n$TODAY_COMMITS"

# Review session stats
if [ -f "$HOME/.openclaw/sessions/stats.json" ]; then
    STATS=$(cat "$HOME/.openclaw/sessions/stats.json" 2>/dev/null | jq -r '.today // "No stats"' || echo "Stats unavailable")
    log "Session stats: $STATS"
fi

# Save Phase 2 results
echo "$TODAY_COMMITS" > "$OUTPUT_DIR/phase2_review_$DATE.txt"

# ============================================
# PHASE 3: Deep Research (Opus - for judgment)
# ============================================
log "🧠 Phase 3: Deep research on relevant topics..."

# Research topics for BrownBioTech
RESEARCH_TOPICS="AI agents for drug discovery, DGAT1 inhibitors, cancer metabolism, multi-omics integration"

# Create research prompt
RESEARCH_PROMPT="You are Demis, an AI assistant for Dr. OCM at BrownBioTech (cancer metabolism drug discovery).

Today is $DATE.

Research summary needed on:
1. Latest advances in: $RESEARCH_TOPICS
2. Any relevant papers from arXiv (cs.AI, q-bio.QM)
3. Startups doing similar work
4. Technical insights applicable to BrownBioTech

Provide:
- Top 3 paper recommendations with abstracts
- 2 actionable insights for BrownBioTech
- 1 potential improvement to our workflow

Keep response concise (500 words max)."

# Run deep research (using OpenRouter with Opus if available, Haiku fallback)
if command -v curl &> /dev/null; then
    DEEP_RESEARCH=$(curl -s -X POST "https://openrouter.ai/api/v1/chat/completions" \
        -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"anthropic/claude-3.5-haiku\",\"messages\":[{\"role\":\"user\",\"content\":\"$RESEARCH_PROMPT\"}],\"max_tokens\":1000}" 2>/dev/null | \
        jq -r '.choices[0].message.content // "Research skipped"' 2>/dev/null || echo "Deep research skipped")
    
    log "Deep research result:\n$DEEP_RESEARCH"
    echo "$DEEP_RESEARCH" > "$OUTPUT_DIR/phase3_research_$DATE.txt"
fi

# ============================================
# PHASE 4: Self-Improvement Decision
# ============================================
log "🎯 Phase 4: Self-improvement decision..."

# Analyze if changes are needed
NEEDS_UPDATE=false
UPDATE_REASON=""

# Check if there are relevant improvements found
if grep -q "insight\|improvement\|update" "$OUTPUT_DIR/phase3_research_$DATE.txt" 2>/dev/null; then
    NEEDS_UPDATE=true
    UPDATE_REASON="Found relevant improvements in research"
fi

# Check if BrownBioTech needs attention
if [ -d "brownbiotech" ]; then
    BROWN_UPDATES=$(git log --since="1 day ago" --oneline 2>/dev/null | wc -l)
    if [ "$BROWN_UPDATES" -gt 5 ]; then
        NEEDS_UPDATE=true
        UPDATE_REASON="Significant BrownBioTech updates detected"
    fi
fi

log "Update needed: $NEEDS_UPDATE (Reason: $UPDATE_REASON)"

# ============================================
# PHASE 5: Apply Safe Changes
# ============================================
if [ "$NEEDS_UPDATE" = true ]; then
    log "🚀 Applying improvements..."
    
    # Create improvement branch
    IMPR_BRANCH="dream/improvement-$DATE"
    git checkout -b "$IMPR_BRANCH" 2>/dev/null || true
    
    # Log improvements
    cat > "$OUTPUT_DIR/improvements_$DATE.md" << IMPROV
# Dream Cycle Improvements - $DATE

## Research Findings
$(cat "$OUTPUT_DIR/phase3_research_$DATE.txt" 2>/dev/null || echo "No research available")

## Reason for Update
$UPDATE_REASON

## Next Steps
- Review findings in morning
- Apply relevant improvements
- Discuss with Dr. OCM

Generated by Demis Dream Cycle 🌙
IMPROV
    
    # Commit improvements
    git add "$OUTPUT_DIR/improvements_$DATE.md" 2>/dev/null || true
    git commit -m "dream: self-improvement from $DATE" 2>/dev/null || true
    
    # If on main branch, just save notes
    git checkout main 2>/dev/null || true
    
    log "✓ Improvements logged to $OUTPUT_DIR/improvements_$DATE.md"
else
    log "✓ No immediate updates needed"
fi

# ============================================
# PHASE 6: Summary & Telegram Alert
# ============================================
log "📱 Phase 6: Sending summary..."

# Create summary
SUMMARY="🌙 Demis Dream Complete

📅 Date: $DATE

📰 GitHub: Trending AI repos scanned
📊 Today: $TODAY_COMMITS commits reviewed  
🧠 Research: $(wc -w < "$OUTPUT_DIR/phase3_research_$DATE.txt" 2>/dev/null || echo 0) words generated
🎯 Update: $([ "$NEEDS_UPDATE" = true ] && echo "Yes - improvements found" || echo "Not needed")

💰 Cost estimate: ~\$0.40 (Haiku + GitHub API)
⏰ Next dream: Tomorrow 23:00 KST"

# Save summary
echo "$SUMMARY" > "$OUTPUT_DIR/dream_summary_$DATE.txt"

log "🌅 Dream Cycle complete!"

# Show summary
echo ""
echo "=========================================="
echo "$SUMMARY"
echo "=========================================="

# Exit cleanly
exit 0
