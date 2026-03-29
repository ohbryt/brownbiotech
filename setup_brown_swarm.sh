#!/bin/bash
# BROWN Swarm Setup Script
# Initializes ClawTeam for BrownBioTech drug discovery

set -e

echo "=========================================="
echo "BROWN Swarm: ClawTeam for BrownBioTech"
echo "=========================================="
echo ""

# Check requirements
echo "[1/5] Checking requirements..."

if ! command -v clawteam &> /dev/null; then
    echo "❌ ClawTeam not found. Install with: pipx install clawteam"
    exit 1
fi
echo "✓ ClawTeam installed"

if ! command -v tmux &> /dev/null; then
    echo "❌ tmux not found. Install with: brew install tmux"
    exit 1
fi
echo "✓ tmux installed"

echo ""
echo "[2/5] Creating BROWN Swarm team..."

# Create team
clawteam team create brown-swarm \
    --leader "You are the BROWN Swarm leader for BrownBioTech, an AI-first cancer therapeutics company. Your mission: discover DGAT1/YARS2 inhibitors for solid tumors. Coordinate targetscan, moleculeforge, atlas-vs, admet-pred, and ind-prep agents."

echo "✓ Team 'brown-swarm' created"

echo ""
echo "[3/5] Spawning BROWN Swarm agents..."

# Target Discovery Agents
clawteam spawn brown-swarm \
    --name targetscan-dgat1 \
    --agent-name "DGAT1 Target Researcher" \
    --task "Analyze DGAT1 expression in TCGA NSCLC datasets. Find survival correlations, CRISPR dependencies, and validate as therapeutic target."

clawteam spawn brown-swarm \
    --name targetscan-yars2 \
    --agent-name "YARS2 Target Researcher" \
    --task "Analyze YARS2 expression in TCGA liver cancer datasets. Find metabolic vulnerabilities and validate as therapeutic target."

# Molecule Generation
clawteam spawn brown-swarm \
    --name moleculeforge-diff \
    --agent-name "Diffusion Chemist" \
    --task "Generate 100 DGAT1 inhibitor candidates using diffusion models. Optimize for lipophilicity and synthesizability."

clawteam spawn brown-swarm \
    --name moleculeforge-gnn \
    --agent-name "GNN Researcher" \
    --task "Search ChEMBL for DGAT1 similar compounds. Apply scaffold hopping to generate novel lead series."

# Virtual Screening
clawteam spawn brown-swarm \
    --name atlas-dock \
    --agent-name "Docking Specialist" \
    --task "Dock generated compounds against DGAT1 binding pocket using QVina-W. Score binding affinity and filter hits <-8.0 kcal/mol."

clawteam spawn brown-swarm \
    --name atlas-repair \
    --agent-name "Repair Engineer" \
    --task "Apply ATLAS self-repair for failed docking hits. Use PR-CoT reasoning to suggest modifications improving binding."

# ADMET
clawteam spawn brown-swarm \
    --name admet-pred \
    --agent-name "ADMET Analyst" \
    --task "Predict solubility, permeability, CYP inhibition, and hERG liability for top 20 DGAT1 inhibitors."

# Documentation
clawteam spawn brown-swarm \
    --name ind-prep \
    --agent-name "Regulatory Writer" \
    --task "Draft BROWN-1 IND application sections: introduction, target validation, preclinical pharmacology, and toxicology summary."

echo "✓ 8 BROWN Swarm agents spawned"

echo ""
echo "[4/5] BROWN Swarm status..."

clawteam task list brown-swarm

echo ""
echo "[5/5] Monitoring BROWN Swarm..."

echo "To monitor the swarm, run:"
echo "  clawteam board attach brown-swarm    # tmux view"
echo "  clawteam board serve --port 8080    # Web UI"

echo ""
echo "=========================================="
echo "BROWN Swarm initialized successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Monitor: clawteam board attach brown-swarm"
echo "  2. Check results: clawteam task list brown-swarm"
echo "  3. Get leader summary: clawteam inbox list brown-swarm"
