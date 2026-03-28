# ARP v4 — Drug Discovery Autoreview System

## Overview
ARP (Autonomous Research Pipeline) v4 integrates Karpathy-style autoreview loop for continuous drug discovery improvement.

## Core Loop
```
Hypothesis → Design → Test → Evaluate → Keep/Discard → Repeat
     ↑                                                    |
     └───────────── Feedback from wet lab ───────────────┘
```

## Architecture

### 1. Research Layer
- Literature mining (PubMed, patents)
- Target identification
- Competitive analysis

### 2. Design Layer
- Molecular generation (diffusion-based)
- Structure prediction (AlphaFold-style)
- ADMET prediction

### 3. Validation Layer
- GIST wet lab integration
- in vitro assay results
- in vivo mouse studies

### 4. Autoreview Layer
- Experiment tracking
- Result scoring
- Hypothesis generation
- Iteration planning

## DGAT1 Pipeline (BROWN-1)

### Target Profile
- **Gene**: DGAT1
- **Function**: Triacylglycerol synthesis
- **Role in NSCLC**: Lipid metabolism reprogramming
- **Validation**: Mouse efficacy confirmed

### ARP Workflow
1. Literature: DGAT1 in lung cancer
2. Design: Novel DGAT1 inhibitors
3. Screen: in silico ADMET filtering
4. Synthesize: Top 10 candidates
5. Test: in vitro binding assay
6. Iterate: Top hits → in vivo

## YARS2 Pipeline (BROWN-2)

### Target Profile
- **Gene**: YARS2
- **Function**: Mitochondrial aaRS
- **Role in NSCLC**: Mitochondrial dysfunction
- **Validation**: In vitro activity confirmed

### ARP Workflow
1. Literature: YARS2 cancer dependency
2. Design: Mitochondrial-targeting compounds
3. Screen: Selectivity profiling
4. Optimize: in vitro potency
5. Test: in vivo efficacy (planned H2 2026)

## Metrics

| Metric | Value |
|--------|-------|
| Molecular candidates per round | 10-50 |
| Design-to-test cycle | 2-4 weeks |
| Hit rate improvement | 10x vs traditional |
| in vivo success rate | Target: 20% |

---
Generated: 2026-03-29
