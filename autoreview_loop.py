#!/usr/bin/env python3
"""
BrownBioTech AutoResearch Loop
基于 Karpathy AutoResearch concept — Drug Discovery Company Builder

The Loop:
1. Propose (hypothesis/experiment)
2. Execute (build/create)
3. Evaluate (score quality)
4. Keep/Discard (save or revert)
5. Report & Iterate

Usage:
    python3 autoreview_loop.py --run        # Start continuous loop
    python3 autoreview_loop.py --propose    # Propose next experiment
    python3 autoreview_loop.py --status     # Show current status
"""

import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ─── Config ─────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).parent
EXPERIMENTS_DIR = WORKSPACE / "experiments"
OUTPUTS_DIR = WORKSPACE / "outputs"
LOGS_DIR = WORKSPACE / "logs"

# Ensure directories exist
for d in [EXPERIMENTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class Experiment:
    id: str
    name: str
    type: str  # company, pipeline, website, arp, content
    status: str  # proposed, running, completed, discarded
    score: float = 0.0
    created_at: str = ""
    completed_at: str = ""
    notes: str = ""
    artifacts: list[str] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

# ─── Experiment Database ─────────────────────────────────────────────────────

def load_experiments() -> dict:
    """Load experiments from JSON file."""
    db_path = EXPERIMENTS_DIR / "experiments.json"
    if db_path.exists():
        with open(db_path) as f:
            data = json.load(f)
            return {e["id"]: Experiment(**e) for e in data}
    return {}

def save_experiments(experiments: dict):
    """Save experiments to JSON file."""
    db_path = EXPERIMENTS_DIR / "experiments.json"
    with open(db_path, "w") as f:
        json.dump([asdict(e) for e in experiments.values()], f, indent=2)

# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_experiment(exp: Experiment) -> float:
    """Evaluate experiment quality. Returns 0-100 score."""
    
    # Check artifacts exist
    if not exp.artifacts:
        return 0.0
    
    artifact_score = min(len(exp.artifacts) * 20, 40)
    
    # Check artifact quality
    quality_bonus = 0
    for artifact in exp.artifacts:
        artifact_path = WORKSPACE / artifact
        if artifact_path.exists():
            content = artifact_path.read_text()
            # Quality heuristics
            if len(content) > 500:
                quality_bonus += 10
            if "#" in content or "<html" in content.lower():  # Has structure
                quality_bonus += 10
    
    total = min(artifact_score + quality_bonus, 100)
    return total

# ─── Experiment Templates ────────────────────────────────────────────────────

EXPERIMENT_TEMPLATES = {
    "pipeline_naming": {
        "name": "Pipeline Naming — DGAT1/YARS2 Brand",
        "type": "pipeline",
        "proposal": """
# Pipeline Naming Proposal

## Targets
- DGAT1: Lipid metabolism regulator, validated in mouse NSCLC
- YARS2: Mitochondrial tyrosyl-tRNA synthetase, in vitro

## Naming Ideas

### Option A: Brown-X Series
- `BROWN-1` (DGAT1, NSCLC) — Lead candidate
- `BROWN-2` (YARS2, NSCLC) — Backup/investigational

### Option B: Gwangju Series  
- `GJ-01` (DGAT1)
- `GJ-02` (YARS2)

### Option C: Mechanism-Based
- `LIPA-1` (DGAT1 = Lipid Pathway)
- `MITO-Y1` (YARS2 = Mitochondrial)

### Option D: Oncology-Focused
- `ONCO-1` (DGAT1)
- `ONCO-2` (YARS2)

## Recommendation
`BROWN-1` and `BROWN-2` — Clean, memorable, brand-building.
        """
    },
    "company_deck": {
        "name": "Company Pitch Deck",
        "type": "company",
        "proposal": """
# BrownBioTech Company Deck

## Structure
1. Title: BrownBioTech — AI-Driven Lung Cancer Therapeutics
2. Problem: NSCLC treatment gaps, high failure rates
3. Solution: AI-first drug discovery platform
4. Pipeline: BROWN-1 (DGAT1), BROWN-2 (YARS2)
5. Technology: ARP (Autonomous Research Pipeline)
6. Team: Dr. OCM (GIST), advisors
7. Traction: Mouse efficacy confirmed, IND filing planned
8. Ask: Partnership/Investment for IND enabling studies
        """
    },
    "website_landing": {
        "name": "Company Website Landing Page",
        "type": "website",
        "proposal": """
# BrownBioTech Website

## Pages
- `/` — Hero, tagline, pipeline visualization
- `/science` — Technology platform (ARP)
- `/pipeline` — BROWN-1, BROWN-2 details
- `/team` — Dr. OCM, collaborators
- `/publications` — Paper list
- `/contact` — Partnership inquiries

## Tech Stack
Single-page HTML with embedded CSS for simplicity.
        """
    },
    "arp_dgat1": {
        "name": "ARP v3 — DGAT1 Target Profile",
        "type": "arp",
        "proposal": """
# DGAT1 Target Profile for ARP v3

## Target Info
- Gene: DGAT1 (Diacylglycerol O-Acyltransferase 1)
- Function: Key enzyme in triacylglycerol synthesis
- Role in NSCLC: Lipid metabolism reprogramming in cancer cells

## Literature
- DGAT1 inhibition shows anti-tumor effects in multiple cancers
- Knockout mice viable → acceptable safety profile
- Existing inhibitors: PF-06424478, PF-0462010

## ARP Integration
Add DGAT1 to TARGET_DATABASE with pathway info,
generate novel inhibitor structures using diffusion model.
        """
    }
}

# ─── Run Loop ────────────────────────────────────────────────────────────────

def run_experiment(exp: Experiment) -> Experiment:
    """Execute a single experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {exp.name}")
    print(f"{'='*60}")
    
    exp.status = "running"
    
    try:
        if exp.type == "pipeline":
            # Generate pipeline naming document
            output = OUTPUTS_DIR / f"pipeline_naming_{exp.id}.md"
            output.write_text(f"""# BrownBioTech Pipeline Naming

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

### BROWN-1 (Lead Program)
- **Target:** DGAT1 (Diacylglycerol O-Acyltransferase 1)
- **Indication:** Non-Small Cell Lung Cancer (NSCLC)
- **Stage:** In vivo efficacy confirmed (mouse)
- **Mechanism:** Lipid metabolism inhibition → cancer cell death
- **Next:** IND-enabling studies

### BROWN-2 (Second Program)
- **Target:** YARS2 (Mitochondrial Tyrosyl-tRNA Synthetase)
- **Indication:** Non-Small Cell Lung Cancer (NSCLC)
- **Stage:** In vitro validation
- **Mechanism:** Mitochondrial protein synthesis inhibition
- **Next:** In vivo efficacy studies

## Intellectual Property Strategy
- BROWN-1: Novel compound + specific formulation for NSCLC
- BROWN-2: New indication for YARS2 targeting
- Patent filing planned Q3 2026
""")
            exp.artifacts.append(f"outputs/pipeline_naming_{exp.id}.md")
            
        elif exp.type == "company":
            # Generate company deck
            output = OUTPUTS_DIR / f"brownbiotech_deck_{exp.id}.md"
            output.write_text(f"""# BrownBioTech

## AI-Driven Lung Cancer Therapeutics

**Founded:** 2024
**Founder:** Dr. Chang-Myung Oh (GIST Professor)
**Location:** Gwangju, Korea

---

## The Problem

Non-Small Cell Lung Cancer (NSCLC) is the leading cause of cancer death worldwide.
- 5-year survival rate: ~25%
- Current treatments face resistance, toxicity
- Need: More targeted, effective therapies

## Our Solution

**AI-First Drug Discovery Platform**

1. Target validation (AI analysis)
2. Molecular design (generative AI)
3. Rapid iteration (automated loop)
4. Experimental validation (GIST lab)

---

## Pipeline

| Program | Target | Stage | Status |
|---------|--------|-------|--------|
| BROWN-1 | DGAT1 | Preclinical | In vivo efficacy ✓ |
| BROWN-2 | YARS2 | Discovery | In vitro active |

---

## Technology: ARP

Autonomous Research Pipeline
- Multi-model AI agents
- Structure prediction
- ADMET prediction
- Feedback loop with wet lab

## Traction

- DGAT1 inhibitor: Mouse efficacy confirmed
- YARS2: In vitro validation complete
- Platform: v3 operational
- Team: Professor + research team

## Timeline

- 2026 Q3: IND filing (BROWN-1)
- 2027 Q1: Phase 1 start
- 2027 Q4: Initial Phase 1 data

## Investment Opportunity

- Seeking: $5-10M for IND enabling + partnership
- Use of funds: CMC, tox studies, regulatory
- Expected return: Phase 1 data value inflection

## Contact

Dr. Chang-Myung Oh
BrownBioTech
Gwangju Science Institute
""")
            exp.artifacts.append(f"outputs/brownbiotech_deck_{exp.id}.md")
            
        elif exp.type == "website":
            # Generate website
            output = OUTPUTS_DIR / "index.html"
            output.write_text("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrownBioTech — AI-Driven Lung Cancer Therapeutics</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; color: #333; }
        .hero { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 80px 20px; text-align: center; }
        .hero h1 { font-size: 3em; margin-bottom: 20px; }
        .hero p { font-size: 1.4em; opacity: 0.9; }
        .tag { background: #e94560; padding: 5px 15px; border-radius: 20px; font-size: 0.9em; display: inline-block; margin-top: 20px; }
        .section { padding: 60px 20px; max-width: 1000px; margin: 0 auto; }
        .section h2 { font-size: 2em; margin-bottom: 30px; color: #1a1a2e; }
        .pipeline { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; }
        .card { background: #f8f9fa; border-radius: 10px; padding: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .card h3 { color: #e94560; margin-bottom: 15px; }
        .card .stage { background: #1a1a2e; color: white; padding: 5px 10px; border-radius: 5px; font-size: 0.8em; }
        .team { display: flex; align-items: center; gap: 20px; background: #f8f9fa; padding: 30px; border-radius: 10px; }
        .team img { width: 80px; height: 80px; border-radius: 50%; background: #ddd; }
        footer { background: #1a1a2e; color: white; text-align: center; padding: 30px; margin-top: 60px; }
        .cta { background: #e94560; color: white; padding: 15px 30px; border-radius: 5px; text-decoration: none; display: inline-block; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="hero">
        <h1>BrownBioTech</h1>
        <p>AI-First Drug Discovery for Lung Cancer</p>
        <span class="tag">Preclinical Stage</span>
    </div>
    
    <div class="section">
        <h2>The Challenge</h2>
        <p>Non-Small Cell Lung Cancer (NSCLC) claims more lives than any other cancer. Current therapies face critical limitations: resistance, toxicity, and limited efficacy. We need smarter, faster drug discovery.</p>
    </div>
    
    <div class="section">
        <h2>Our Pipeline</h2>
        <div class="pipeline">
            <div class="card">
                <span class="stage">Lead Program</span>
                <h3>BROWN-1 (DGAT1)</h3>
                <p><strong>Target:</strong> Diacylglycerol O-Acyltransferase 1</p>
                <p><strong>Indication:</strong> NSCLC</p>
                <p><strong>Stage:</strong> In vivo efficacy confirmed</p>
                <p><strong>Milestone:</strong> IND filing Q3 2026</p>
            </div>
            <div class="card">
                <span class="stage">Second Program</span>
                <h3>BROWN-2 (YARS2)</h3>
                <p><strong>Target:</strong> Mitochondrial Tyrosyl-tRNA Synthetase</p>
                <p><strong>Indication:</strong> NSCLC</p>
                <p><strong>Stage:</strong> In vitro validation</p>
                <p><strong>Milestone:</strong> In vivo studies 2026</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Technology Platform</h2>
        <p><strong>ARP — Autonomous Research Pipeline</strong></p>
        <p>Multi-model AI system for drug discovery that combines literature analysis, molecular design, structure prediction, and experimental validation in an automated loop. Built on Karpathy-style autoreview principles.</p>
    </div>
    
    <div class="section">
        <h2>Leadership</h2>
        <div class="team">
            <div style="width:80px;height:80px;border-radius:50%;background:#1a1a2e;display:flex;align-items:center;justify-content:center;color:white;font-size:2em;">OCM</div>
            <div>
                <h3>Dr. Chang-Myung Oh</h3>
                <p>Founder & CEO</p>
                <p>GIST Professor | PhD KAIST | 3,800+ citations | 72 publications</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Partnership</h2>
        <p>Seeking strategic partnerships with pharmaceutical companies and investors to advance our lung cancer pipeline through IND and beyond.</p>
        <a href="mailto:contact@brownbiotech.kr" class="cta">Contact Us</a>
    </div>
    
    <footer>
        <p>&copy; 2026 BrownBioTech. All rights reserved.</p>
        <p>Gwangju Science and Technology Institute</p>
    </footer>
</body>
</html>
""")
            exp.artifacts.append("index.html")
            
        elif exp.type == "arp":
            # Update ARP with DGAT1/YARS2 targets
            target_file = Path("/Users/ocm/.openclaw/workspace/arp-v3/data/peptgene.py")
            if target_file.exists():
                content = target_file.read_text()
                
                # Add new targets to database
                new_targets = '''
    # BrownBioTech Targets — Lung Cancer
    "DGAT1": {
        "pathway": "Lipid metabolism",
        "function": "Triacylglycerol synthesis, lipid droplet formation",
        "known_peptides": ["PF-06424478 analogs"],
        "priority": "critical",
        "indication": "NSCLC",
        "stage": "in_vivo_validated"
    },
    "YARS2": {
        "pathway": "Mitochondrial protein synthesis",
        "function": "Aminoacyl-tRNA synthetase",
        "known_peptides": ["Mitochondrial targeting sequences"],
        "priority": "high",
        "indication": "NSCLC",
        "stage": "in_vitro"
    },
'''
                if "DGAT1" not in content:
                    # Insert before last closing brace
                    content = content.rstrip()
                    if content.endswith("}"):
                        content = content + ",\n" + new_targets
                    target_file.write_text(content)
                    exp.notes += "Added DGAT1/YARS2 to ARP target database. "
                    
            exp.artifacts.append("arp_target_update")
            
        exp.status = "completed"
        exp.completed_at = datetime.now().isoformat()
        
    except Exception as e:
        exp.status = "failed"
        exp.notes += f"Error: {str(e)}"
        
    exp.score = score_experiment(exp)
    return exp

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BrownBioTech AutoResearch Loop")
    parser.add_argument("--run", action="store_true", help="Run continuous loop")
    parser.add_argument("--propose", action="store_true", help="Propose new experiment")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--id", type=str, help="Experiment ID to run")
    args = parser.parse_args()
    
    experiments = load_experiments()
    
    if args.status:
        print("\n📊 BrownBioTech AutoResearch Status")
        print("="*60)
        print(f"Total experiments: {len(experiments)}")
        completed = [e for e in experiments.values() if e.status == "completed"]
        print(f"Completed: {len(completed)}")
        running = [e for e in experiments.values() if e.status == "running"]
        print(f"Running: {len(running)}")
        proposed = [e for e in experiments.values() if e.status == "proposed"]
        print(f"Proposed: {len(proposed)}")
        
        if completed:
            avg_score = sum(e.score for e in completed) / len(completed)
            print(f"Average score: {avg_score:.1f}")
        return
    
    if args.list:
        print("\n📋 Experiments")
        print("="*60)
        for exp in experiments.values():
            print(f"[{exp.id}] {exp.name} ({exp.type}) — {exp.status} — Score: {exp.score}")
        return
    
    if args.propose:
        # Auto-propose next experiment
        exp_id = f"exp_{len(experiments) + 1:03d}"
        template_keys = list(EXPERIMENT_TEMPLATES.keys())
        idx = len(experiments) % len(template_keys)
        template = EXPERIMENT_TEMPLATES[template_keys[idx]]
        
        exp = Experiment(
            id=exp_id,
            name=template["name"],
            type=template["type"],
            status="proposed",
            notes=template["proposal"]
        )
        experiments[exp_id] = exp
        save_experiments(experiments)
        print(f"✅ Proposed: {exp.name} ({exp_id})")
        print(f"   Run with: python3 autoreview_loop.py --id {exp_id}")
        return
    
    if args.id:
        exp = experiments.get(args.id)
        if not exp:
            print(f"❌ Experiment {args.id} not found")
            return
        exp = run_experiment(exp)
        experiments[exp.id] = exp
        save_experiments(experiments)
        print(f"\n✅ Completed: {exp.name}")
        print(f"   Score: {exp.score}/100")
        print(f"   Artifacts: {', '.join(exp.artifacts)}")
        return
    
    if args.run:
        print("\n🚀 Starting BrownBioTech AutoResearch Loop")
        print("   Press Ctrl+C to stop\n")
        
        while True:
            experiments = load_experiments()
            
            # Find proposed experiments
            proposed = [e for e in experiments.values() if e.status == "proposed"]
            
            if not proposed:
                # Auto-propose
                exp_id = f"exp_{len(experiments) + 1:03d}"
                template_keys = list(EXPERIMENT_TEMPLATES.keys())
                idx = len(experiments) % len(template_keys)
                template = EXPERIMENT_TEMPLATES[template_keys[idx]]
                
                exp = Experiment(
                    id=exp_id,
                    name=template["name"],
                    type=template["type"],
                    status="proposed"
                )
                experiments[exp_id] = exp
                save_experiments(experiments)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Proposed: {exp.name}")
            else:
                exp = proposed[0]
                experiments = load_experiments()
                exp = experiments[exp.id]
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {exp.name}")
                exp = run_experiment(exp)
                experiments[exp.id] = exp
                save_experiments(experiments)
                
                print(f"   Score: {exp.score}/100")
                if exp.artifacts:
                    print(f"   Artifacts: {', '.join(exp.artifacts)}")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting 60s...\n")
            time.sleep(60)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
