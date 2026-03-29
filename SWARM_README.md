# BROWN-Bio Swarm: ClawTeam for Drug Discovery

**Inspired by HKUDS ClawTeam**  
**For: BrownBioTech Cancer Metabolism Research**

---

## Concept

```
One Command → Full Drug Discovery Automation

Human sets goal → BROWN Swarm executes
├── TargetScan™ agents (TCGA/DepMap)
├── MoleculeForge™ agents (Diffusion/GNN)
├── ATLAS-VS™ agents (Docking/Repair)
└── IND-Prep™ agents (Documents)
```

---

## BROWN Swarm Configuration

### Team Structure

```
brown-swarm (Leader)
├── targetscan-gdat1     → DGAT1 TCGA/DepMap
├── targetscan-yars2      → YARS2 TCGA/DepMap  
├── moleculeforge-v1      → Diffusion generation
├── moleculeforge-v2      → GNN search
├── atlas-docking        → QVina-W docking
├── atlas-repair         → Self-verified repair
├── admet-pred           → ADMET prediction
├── ind-prep             → IND documentation
└── literature           → PubMed/Patent search
```

### Hardware Allocation (Example)

| Agent | GPU | Task |
|-------|-----|------|
| targetscan-gdat1 | GPU 0 | TCGA expression |
| targetscan-yars2 | GPU 1 | TCGA expression |
| moleculeforge-v1 | GPU 2 | Diffusion |
| moleculeforge-v2 | GPU 3 | GNN search |
| atlas-docking | GPU 4 | Docking |
| atlas-repair | GPU 5 | Repair |
| admet-pred | GPU 6 | ADMET |
| ind-prep | GPU 7 | Documents |

---

## Usage

### 1. Initialize BROWN Swarm

```bash
cd ~/brownbiotech

clawteam team create brown-swarm \
  --leader "You are BROWN Swarm leader for BrownBioTech drug discovery"
```

### 2. Spawn Research Agents

```bash
# Target Discovery Agents
clawteam spawn brown-swarm \
  --name targetscan-gdat1 \
  --agent-name "DGAT1 Target Researcher" \
  --task "Analyze DGAT1 in TCGA NSCLC datasets. Find correlations with survival."

clawteam spawn brown-swarm \
  --name targetscan-yars2 \
  --agent-name "YARS2 Target Researcher" \
  --task "Analyze YARS2 in TCGA liver cancer datasets. Find metabolic dependencies."

# Molecule Generation Agents
clawteam spawn brown-swarm \
  --name moleculeforge-diff \
  --agent-name "Diffusion Chemist" \
  --task "Generate 100 DGAT1 inhibitor candidates using diffusion model."

clawteam spawn brown-swarm \
  --name moleculeforge-gnn \
  --agent-name "GNN Researcher" \
  --task "Search ChEMBL for DGAT1 similar compounds. Apply scaffold hopping."

# Virtual Screening Agents
clawteam spawn brown-swarm \
  --name atlas-dock \
  --agent-name "Docking Specialist" \
  --task "Dock generated compounds against DGAT1 binding pocket using QVina-W."

clawteam spawn brown-swarm \
  --name atlas-repair \
  --agent-name "Repair Engineer" \
  --task "Apply ATLAS self-repair for failed docking hits. Use PR-CoT reasoning."

# ADMET Agent
clawteam spawn brown-swarm \
  --name admet-pred \
  --agent-name "ADMET Analyst" \
  --task "Predict solubility, permeability, toxicity for top 10 compounds."

# Documentation Agent
clawteam spawn brown-swarm \
  --name ind-prep \
  --agent-name "Regulatory Writer" \
  --task "Draft BROWN-1 IND application sections based on results."
```

### 3. Monitor Swarm

```bash
# Watch all agents in tmux
clawteam board attach brown-swarm

# Or serve web dashboard
clawteam board serve --port 8080
```

### 4. Get Results

```bash
# List all agent results
clawteam task list brown-swarm

# Get leader summary
clawteam inbox list brown-swarm --recipient leader
```

---

## BROWN Swarm Tasks

### Task Library

| Task ID | Description | Priority |
|---------|-------------|----------|
| BROWN-T001 | DGAT1 TCGA Analysis | High |
| BROWN-T002 | YARS2 DepMap CRISPR | High |
| BROWN-T003 | Diffusion Lead Gen | High |
| BROWN-T004 | GNN Similarity Search | Medium |
| BROWN-T005 | Virtual Screening | High |
| BROWN-T006 | ATLAS Self-Repair | Medium |
| BROWN-T007 | ADMET Prediction | High |
| BROWN-T008 | IND Draft | Medium |

### Task Dependencies

```
BROWN-T001 ─┬─→ BROWN-T003 ─┬─→ BROWN-T005 ─┬─→ BROWN-T007
            │               │               │
BROWN-T002 ─┴─→ BROWN-T004 ─┴─→ BROWN-T006 ─┴─→ BROWN-T008
```

---

## BROWN Swarm Leader Prompt

```
You are the BROWN Swarm leader for BrownBioTech, an AI-first cancer 
therapeutics company. Your team is discovering drugs targeting cancer 
metabolism.

Team members:
- targetscan-gdat1: DGAT1 lipid metabolism expert
- targetscan-yars2: YARS2 mitochondrial expert  
- moleculeforge: Computational chemist (diffusion + GNN)
- atlas-vs: Virtual screening specialist (docking + repair)
- admet-pred: ADMET prediction expert
- ind-prep: Regulatory writer

Current mission: Identify DGAT1 inhibitors for NSCLC

Workflow:
1. Analyze TCGA/DepMap data (targetscan agents)
2. Generate lead compounds (moleculeforge agents)
3. Screen and repair (atlas-vs agents)
4. Predict ADMET (admet-pred agent)
5. Document for IND (ind-prep agent)

Report progress to leader inbox every 30 minutes.
```

---

## AutoResearch Mode (like karpathy's)

For intensive GPU exploration:

```bash
clawteam team create brown-autoresearch

# Leader prompts 8 GPUs to explore:
# - GPU 0-1: DGAT1 binding modes
# - GPU 2-3: Novel scaffolds
# - GPU 4-5: ADMET optimization
# - GPU 6-7: Patent freedom

clawteam spawn brown-autoresearch \
  --agent-name "AutoResearch Leader" \
  --task "Run 1000 virtual screens for DGAT1. Optimize for binding + ADMET."
```

---

## Integration with BROWN-AI™ 2.0

ClawTeam + BROWN-AI™ 2.0 = Complete Automation

```
BROWN-AI™ 2.0 (Task/Agent Core)
├── Task Manager (1 Task = 1 Agent)
├── 4-Layer Context Engine
├── SQLite Experiment DB
│
ClawTeam (Swarm Orchestration)
├── Leader Agent (GPT/Claude)
├── Worker Agents (spawned sub-agents)
├── Git Worktrees (isolation)
└── tmux (parallel execution)
```

---

## Commands Reference

```bash
# Team management
clawteam team create <name>
clawteam team list
clawteam team delete <name>

# Spawn agents
clawteam spawn <team> --name <worker> --task "<task>"

# Task management
clawteam task list <team>
clawteam task status <team> <task>
clawteam task done <team> <task>

# Communication
clawteam inbox send <team> <recipient> "<message>"
clawteam inbox list <team> --recipient <user>

# Monitoring
clawteam board attach <team>   # tmux view
clawteam board serve --port 8080  # Web UI
```

---

## Requirements

- Python 3.11+
- tmux
- Git
- CUDA GPUs (optional, for AutoResearch)
- OpenRouter API key (for leader agent)

---

## Reference

- ClawTeam: https://github.com/HKUDS/ClawTeam
- AutoResearch: https://github.com/karpathy/autoresearch
- BROWN-AI™ 2.0: ./BROWN_AI_PLATFORM.md
