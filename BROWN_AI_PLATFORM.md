# BROWN-AI™ Platform Architecture
## Inspired by OpenCow - Task-Driven Autonomous Research

**Version:** 2.0  
**Based on:** OpenCow's 1 Task = 1 Agent paradigm  
**Date:** 2026-03-29

---

## Core Concept: Research Task = Research Agent

### OpenCow's 1:1 Task→Agent Model Applied to Drug Discovery

```
Research Task                    Autonomous Agent
─────────────────────────────────────────────────
Create task description    →    Assign dedicated AI agent
Full project context        →    Agent inherits org knowledge
15 parallel tasks           →    15 parallel research agents
Real-time monitoring        →    Live dashboard
Approval gates             →    Human review at milestones
```

---

## BROWN-AI™ 2.0 Architecture

### 1. Task Hierarchy (from OpenCow)

```
Project: DGAT1 Drug Discovery
│
├── Task: Target Validation
│   ├── Sub-task: TCGA Expression Analysis
│   ├── Sub-task: DepMap CRISPR Screen
│   └── Sub-task: Literature Review
│
├── Task: Lead Identification
│   ├── Sub-task: Virtual Screening
│   ├── Sub-task: ADMET Prediction
│   └── Sub-task: Scaffold Generation
│
└── Task: IND Preparation
    ├── Sub-task: Pharmacology Report
    ├── Sub-task: Toxicology Summary
    └── Sub-task: Manufacturing Docs
```

### 2. Deep Context Engine (4 Layers)

Inspired by OpenCow's 4-Layer Context:

| Layer | OpenCow | BROWN-AI™ |
|-------|---------|-----------|
| **Org Knowledge** | Skills, playbooks | Cancer metabolism DB, GIST protocols |
| **Project Context** | Team workspace | BROWN-1/2 program data |
| **Team Standards** | Code standards | Research methodology |
| **Task Instructions** | Specific task | Experiment parameters |

### 3. Agent Capabilities (from OpenCow's 6 Types)

| Capability | BROWN-AI™ Application |
|------------|----------------------|
| **Code Generation** | Python/R scripts for analysis |
| **Research** | Literature mining, target analysis |
| **Writing** | IND docs, reports |
| **Analysis** | Multi-omics, statistical |
| **Web Search** | Clinical trials, patents |
| **File Operations** | Data management |

### 4. Parallel Execution Model

```
┌─────────────────────────────────────────────────────┐
│              BROWN-AI™ Supervisor                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ TargetScan™│  │MoleculeForge™│  │ ATLAS-VS™ │       │
│  │  Agent   │  │  Agent   │  │  Agent   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │               │
│       ▼             ▼             ▼               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ TCGA/    │  │ Diffusion │  │ Docking  │       │
│  │ DepMap   │  │ + GNN     │  │ + Repair │       │
│  └──────────┘  └──────────┘  └──────────┘       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 5. Local-First Data Architecture

Inspired by OpenCow's SQLite approach:

| Component | OpenCow | BROWN-AI™ |
|-----------|---------|-----------|
| **Database** | SQLite | SQLite (experiments) |
| **Files** | Local | Local + GIST shared |
| **Privacy** | Zero telemetry | GIST-internal only |
| **Sync** | Local-first | Lab network sync |

### 6. Integration Points

Inspired by OpenCow's IM integration:

| Platform | BROWN-AI™ Use Case |
|----------|-------------------|
| **Telegram** | Progress notifications |
| **Discord** | Team alerts |
| **Email** | IND filing alerts |
| **Webhook** | Lab equipment triggers |

---

## BROWN-AI™ 2.0 Components

### A. Task Manager (like OpenCow's Task Tracker)

```python
@dataclass
class ResearchTask:
    id: str
    name: str
    project: str  # "BROWN-1", "BROWN-2"
    type: TaskType  # validation, screening, ind
    status: TaskStatus
    priority: int
    parent_id: Optional[str]  # for sub-tasks
    assigned_agent: Optional[str]
    context: TaskContext
    results: List[Artifact]
    created_at: datetime
    updated_at: datetime
```

### B. Context Engine (4-Layer)

```python
class BrownContextEngine:
    """BROWN-AI™ Deep Context Engine"""
    
    def get_context(self, task: ResearchTask) -> Dict:
        return {
            # Layer 1: Org Knowledge
            "org": self.get_org_knowledge(),
            # Layer 2: Project Context  
            "project": self.get_project_context(task.project),
            # Layer 3: Team Standards
            "standards": self.get_team_standards(),
            # Layer 4: Task Instructions
            "task": self.get_task_instructions(task)
        }
```

### C. Agent Registry

```python
AGENTS = {
    "targetscan": TargetScanAgent(),      # TCGA/DepMap
    "moleculeforge": MoleculeForgeAgent(),  # Diffusion/GNN
    "atlas_vs": AtlasVSAgent(),          # Docking/Repair
    "ind_prep": IndPrepAgent(),          # Documents
    "literature": LiteratureAgent(),      # PubMed/Cheminformatics
    "wetlab": WetLabAgent(),             # GIST lab coordination
}
```

---

## Implementation Plan

### Phase 1: Core Framework (Inspired by OpenCow's Monitor)
- [ ] Task/Agent core (1:1 mapping)
- [ ] Context engine (4 layers)
- [ ] SQLite database for experiments
- [ ] Basic dashboard

### Phase 2: Research Agents (Inspired by OpenCow's Agent Types)
- [ ] TargetScan™ agent (TCGA/DepMap)
- [ ] MoleculeForge™ agent (Diffusion/GNN)
- [ ] ATLAS-VS™ agent (Docking/Repair)
- [ ] Literature agent (PubMed)

### Phase 3: Integration (Inspired by OpenCow's IM)
- [ ] Telegram notifications
- [ ] Discord alerts
- [ ] Webhook triggers

### Phase 4: Advanced (Inspired by OpenCow's Marketplace)
- [ ] Agent marketplace
- [ ] Custom skills
- [ ] Team collaboration

---

## Comparison: OpenCow vs BROWN-AI™

| Aspect | OpenCow | BROWN-AI™ 2.0 |
|--------|---------|---------------|
| **Goal** | Software development | Drug discovery |
| **Agent count** | 15 parallel | 10+ parallel |
| **Context layers** | 4 layers | 4 layers (adapted) |
| **Database** | SQLite | SQLite + HDF5 |
| **Platform** | Electron Desktop | Python CLI + Web |
| **Integration** | Telegram/Discord | Telegram + GIST lab |
| **Privacy** | Local-first | GIST-internal |

---

## Key Takeaways from OpenCow

1. **1 Task = 1 Agent** — Clear ownership, full context
2. **Deep Context Engine** — 4 layers of context inheritance
3. **Local-first** — Data never leaves your infrastructure
4. **Parallel Execution** — Multiple agents working simultaneously
5. **Approval Gates** — Human review at critical steps
6. **Real-time Monitoring** — Dashboard for all tasks/agents

---

**Reference:** OpenCow (https://github.com/OpenCowAI/opencow)
