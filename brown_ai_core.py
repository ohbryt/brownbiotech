"""
BROWN-AI™ 2.0 Core Framework
=============================
Inspired by OpenCow's 1 Task = 1 Agent paradigm.

Task-driven autonomous research for cancer drug discovery.
"""

import json
import asyncio
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

# ─── Enums ──────────────────────────────────────────────────────────────────

class TaskType(Enum):
    TARGET_VALIDATION = "target_validation"
    LEAD_IDENTIFICATION = "lead_identification"
    VIRTUAL_SCREENING = "virtual_screening"
    ADMET_PREDICTION = "admet_prediction"
    IND_PREPARATION = "ind_preparation"
    LITERATURE_REVIEW = "literature_review"
    WET_LAB_COORDINATION = "wet_lab"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"

class AgentType(Enum):
    TARGETSCAN = "targetscan"          # TCGA/DepMap analysis
    MOLECULEFORGE = "moleculeforge"  # Diffusion/GNN generation
    ATLAS_VS = "atlas_vs"            # Docking + repair
    IND_PREP = "ind_prep"            # Document generation
    LITERATURE = "literature"        # PubMed/patent search
    WETLAB = "wetlab"               # Lab coordination

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class TaskContext:
    """4-layer context for research tasks."""
    # Layer 1: Org Knowledge
    org_knowledge: Dict[str, Any] = field(default_factory=dict)
    # Layer 2: Project Context
    project_context: Dict[str, Any] = field(default_factory=dict)
    # Layer 3: Team Standards
    team_standards: Dict[str, Any] = field(default_factory=dict)
    # Layer 4: Task Instructions
    task_instructions: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt(self) -> str:
        """Convert context to prompt string."""
        sections = []
        
        if self.org_knowledge:
            sections.append("=== ORGANIZATION KNOWLEDGE ===")
            sections.append(json.dumps(self.org_knowledge, indent=2))
        
        if self.project_context:
            sections.append("=== PROJECT CONTEXT ===")
            sections.append(json.dumps(self.project_context, indent=2))
        
        if self.team_standards:
            sections.append("=== TEAM STANDARDS ===")
            sections.append(json.dumps(self.team_standards, indent=2))
        
        if self.task_instructions:
            sections.append("=== TASK INSTRUCTIONS ===")
            sections.append(json.dumps(self.task_instructions, indent=2))
        
        return "\n\n".join(sections)

@dataclass
class TaskResult:
    """Result from a research task."""
    task_id: str
    status: TaskStatus
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    agent_type: Optional[AgentType] = None
    completed_at: Optional[datetime] = None

@dataclass
class ResearchTask:
    """A research task that maps to an autonomous agent."""
    id: str
    name: str
    description: str
    project: str  # "BROWN-1", "BROWN-2", "Sarcopenic"
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    parent_id: Optional[str] = None
    assigned_agent: Optional[AgentType] = None
    context: Optional[TaskContext] = None
    results: List[TaskResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "project": self.project,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "priority": self.priority,
            "parent_id": self.parent_id,
            "assigned_agent": self.assigned_agent.value if self.assigned_agent else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

# ─── Agent Base Class ─────────────────────────────────────────────────────────

class ResearchAgent(ABC):
    """Base class for BROWN-AI™ research agents."""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.name = agent_type.value
    
    @abstractmethod
    async def execute(self, task: ResearchTask, context: TaskContext) -> TaskResult:
        """Execute the task with given context."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        pass

# ─── BROWN-AI™ Agents ────────────────────────────────────────────────────────

class TargetScanAgent(ResearchAgent):
    """Agent for target discovery and validation (TCGA/DepMap)."""
    
    def __init__(self):
        super().__init__(AgentType.TARGETSCAN)
    
    async def execute(self, task: ResearchTask, context: TaskContext) -> TaskResult:
        start = datetime.now()
        
        # Simulated execution
        await asyncio.sleep(0.1)  # Placeholder
        
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            artifacts=[
                {"type": "expression_data", "source": "TCGA"},
                {"type": "crispr_scores", "source": "DepMap"},
            ],
            summary=f"TargetScan™ completed for {task.project}",
            execution_time_seconds=(datetime.now() - start).total_seconds(),
            agent_type=self.agent_type,
            completed_at=datetime.now(),
        )
    
    def get_capabilities(self) -> List[str]:
        return [
            "TCGA expression analysis",
            "DepMap CRISPR screening",
            "Survival correlation",
            "Drug target ranking",
        ]

class MoleculeForgeAgent(ResearchAgent):
    """Agent for molecular design (Diffusion + GNN)."""
    
    def __init__(self):
        super().__init__(AgentType.MOLECULEFORGE)
    
    async def execute(self, task: ResearchTask, context: TaskContext) -> TaskResult:
        start = datetime.now()
        
        # Simulated execution
        await asyncio.sleep(0.1)
        
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            artifacts=[
                {"type": "generated_molecules", "count": 100},
                {"type": "similarity_scores", "source": "GNN"},
            ],
            summary=f"MoleculeForge™ generated leads for {task.project}",
            execution_time_seconds=(datetime.now() - start).total_seconds(),
            agent_type=self.agent_type,
            completed_at=datetime.now(),
        )
    
    def get_capabilities(self) -> List[str]:
        return [
            "Diffusion model generation",
            "GNN similarity search",
            "Scaffold hopping",
            "Property optimization",
        ]

class AtlasVSAgent(ResearchAgent):
    """Agent for virtual screening (Docking + Self-Repair)."""
    
    def __init__(self):
        super().__init__(AgentType.ATLAS_VS)
    
    async def execute(self, task: ResearchTask, context: TaskContext) -> TaskResult:
        start = datetime.now()
        
        # Simulated execution with ATLAS-style repair
        await asyncio.sleep(0.1)
        
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            artifacts=[
                {"type": "docking_scores", "hits": 15},
                {"type": "repaired_molecules", "count": 5},
                {"type": "admet_predictions"},
            ],
            summary=f"ATLAS-VS™ screened and repaired molecules",
            execution_time_seconds=(datetime.now() - start).total_seconds(),
            agent_type=self.agent_type,
            completed_at=datetime.now(),
        )
    
    def get_capabilities(self) -> List[str]:
        return [
            "High-throughput docking",
            "Self-verified repair (ATLAS)",
            "PR-CoT reasoning",
            "ADMET prediction",
        ]

class IndPrepAgent(ResearchAgent):
    """Agent for IND document preparation."""
    
    def __init__(self):
        super().__init__(AgentType.IND_PREP)
    
    async def execute(self, task: ResearchTask, context: TaskContext) -> TaskResult:
        start = datetime.now()
        
        await asyncio.sleep(0.1)
        
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            artifacts=[
                {"type": "ind_section", "name": "introduction"},
                {"type": "ind_section", "name": "pharmacology"},
            ],
            summary=f"IND-Prep™ generated documents for {task.project}",
            execution_time_seconds=(datetime.now() - start).total_seconds(),
            agent_type=self.agent_type,
            completed_at=datetime.now(),
        )
    
    def get_capabilities(self) -> List[str]:
        return [
            "IND section generation",
            "Pharmacology reports",
            "Toxicology summaries",
            "Manufacturing specs",
        ]

# ─── Agent Registry ──────────────────────────────────────────────────────────

class AgentRegistry:
    """Registry of all BROWN-AI™ agents."""
    
    def __init__(self):
        self.agents: Dict[AgentType, ResearchAgent] = {
            AgentType.TARGETSCAN: TargetScanAgent(),
            AgentType.MOLECULEFORGE: MoleculeForgeAgent(),
            AgentType.ATLAS_VS: AtlasVSAgent(),
            AgentType.IND_PREP: IndPrepAgent(),
        }
    
    def get_agent(self, agent_type: AgentType) -> ResearchAgent:
        return self.agents.get(agent_type)
    
    def get_all_agents(self) -> List[ResearchAgent]:
        return list(self.agents.values())
    
    def recommend_agent(self, task_type: TaskType) -> AgentType:
        """Recommend agent for task type."""
        mapping = {
            TaskType.TARGET_VALIDATION: AgentType.TARGETSCAN,
            TaskType.LEAD_IDENTIFICATION: AgentType.MOLECULEFORGE,
            TaskType.VIRTUAL_SCREENING: AgentType.ATLAS_VS,
            TaskType.ADMET_PREDICTION: AgentType.ATLAS_VS,
            TaskType.IND_PREPARATION: AgentType.IND_PREP,
            TaskType.LITERATURE_REVIEW: AgentType.LITERATURE,
            TaskType.WET_LAB_COORDINATION: AgentType.WETLAB,
        }
        return mapping.get(task_type, AgentType.TARGETSCAN)

# ─── BROWN-AI™ Supervisor ────────────────────────────────────────────────────

class BrownAISupervisor:
    """
    BROWN-AI™ Supervisor - orchestrates research tasks and agents.
    
    Inspired by OpenCow's Task → Agent paradigm.
    1 Task = 1 Agent with full context inheritance.
    """
    
    def __init__(self, db_path: str = "brown_ai.db"):
        self.db_path = db_path
        self.registry = AgentRegistry()
        self.tasks: Dict[str, ResearchTask] = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                project TEXT,
                task_type TEXT,
                status TEXT,
                priority INTEGER DEFAULT 0,
                parent_id TEXT,
                assigned_agent TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                status TEXT,
                summary TEXT,
                error TEXT,
                execution_time REAL,
                agent_type TEXT,
                completed_at TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_task(
        self,
        name: str,
        description: str,
        project: str,
        task_type: TaskType,
        priority: int = 0,
        parent_id: Optional[str] = None,
    ) -> ResearchTask:
        """Create a new research task."""
        import uuid
        task = ResearchTask(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            project=project,
            task_type=task_type,
            priority=priority,
            parent_id=parent_id,
            assigned_agent=self.registry.recommend_agent(task_type),
        )
        
        self.tasks[task.id] = task
        self._save_task(task)
        
        return task
    
    def _save_task(self, task: ResearchTask):
        """Save task to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                project TEXT,
                task_type TEXT,
                status TEXT,
                priority INTEGER DEFAULT 0,
                parent_id TEXT,
                assigned_agent TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        cursor.execute("""
            INSERT OR REPLACE INTO tasks 
            (id, name, description, project, task_type, status, priority, parent_id, assigned_agent, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.id,
            task.name,
            task.description,
            task.project,
            task.task_type.value,
            task.status.value,
            task.priority,
            task.parent_id,
            task.assigned_agent.value if task.assigned_agent else None,
            task.created_at.isoformat(),
            task.updated_at.isoformat(),
        ))
        
        conn.commit()
        conn.close()
    
    async def execute_task(self, task_id: str, context: TaskContext) -> TaskResult:
        """Execute a task with the assigned agent."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        agent = self.registry.get_agent(task.assigned_agent)
        if not agent:
            raise ValueError(f"Agent {task.assigned_agent} not found")
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.updated_at = datetime.now()
        self._save_task(task)
        
        try:
            # Execute with context
            result = await agent.execute(task, context)
            
            # Update task with result
            task.status = result.status
            task.results.append(result)
            task.updated_at = datetime.now()
            self._save_task(task)
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now()
            self._save_task(task)
            
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )
    
    async def execute_parallel(self, task_ids: List[str], context: TaskContext) -> List[TaskResult]:
        """Execute multiple tasks in parallel."""
        coroutines = [self.execute_task(tid, context) for tid in task_ids]
        return await asyncio.gather(*coroutines)
    
    def get_task_tree(self, project: str) -> Dict:
        """Get task hierarchy for a project."""
        project_tasks = [t for t in self.tasks.values() if t.project == project]
        
        # Build tree
        root_tasks = [t for t in project_tasks if not t.parent_id]
        
        def build_subtree(task: ResearchTask) -> Dict:
            children = [t for t in project_tasks if t.parent_id == task.id]
            return {
                **task.to_dict(),
                "children": [build_subtree(c) for c in children],
            }
        
        return {
            "project": project,
            "tasks": [build_subtree(t) for t in root_tasks],
            "total": len(project_tasks),
        }

# ─── Demo ───────────────────────────────────────────────────────────────────

async def demo():
    """Demonstrate BROWN-AI™ 2.0 framework."""
    print("="*60)
    print("BROWN-AI™ 2.0 Framework Demo")
    print("Inspired by OpenCow's 1 Task = 1 Agent")
    print("="*60)
    
    # Create supervisor
    supervisor = BrownAISupervisor(":memory:")  # In-memory for demo
    
    # Create context (4 layers)
    context = TaskContext(
        org_knowledge={
            "cancer_types": ["NSCLC", "LIHC", "BRCA"],
            "targets": {"DGAT1": "Lipid metabolism", "YARS2": "Mitochondrial"},
        },
        project_context={
            "program": "BROWN-1",
            "target": "DGAT1",
            "stage": "IND-enabling",
        },
        team_standards={
            "min_docking_score": -8.0,
            "min_admet_pass_rate": 0.7,
        },
        task_instructions={
            "priority_targets": ["MTOR", "AKT1", "INS"],
            "min_similarity": 0.4,
        }
    )
    
    # Create tasks
    print("\n[1] Creating research tasks...")
    
    task1 = supervisor.create_task(
        name="TCGA Expression Analysis",
        description="Analyze DGAT1 expression in TCGA datasets",
        project="BROWN-1",
        task_type=TaskType.TARGET_VALIDATION,
        priority=1,
    )
    print(f"  Created: {task1.name} → {task1.assigned_agent.value}")
    
    task2 = supervisor.create_task(
        name="Virtual Screening",
        description="Screen compounds against DGAT1",
        project="BROWN-1",
        task_type=TaskType.VIRTUAL_SCREENING,
        priority=2,
    )
    print(f"  Created: {task2.name} → {task2.assigned_agent.value}")
    
    task3 = supervisor.create_task(
        name="IND Section Draft",
        description="Generate IND application sections",
        project="BROWN-1",
        task_type=TaskType.IND_PREPARATION,
        priority=3,
    )
    print(f"  Created: {task3.name} → {task3.assigned_agent.value}")
    
    # Execute parallel
    print("\n[2] Executing tasks in parallel...")
    results = await supervisor.execute_parallel(
        [task1.id, task2.id, task3.id],
        context
    )
    
    for result in results:
        print(f"  ✓ {result.task_id}: {result.status.value} ({result.execution_time_seconds:.3f}s)")
        print(f"    Summary: {result.summary}")
    
    # Get task tree
    print("\n[3] Project task hierarchy:")
    tree = supervisor.get_task_tree("BROWN-1")
    print(f"  Project: {tree['project']}")
    print(f"  Total tasks: {tree['total']}")
    
    print("\n" + "="*60)
    print("BROWN-AI™ 2.0 Framework Demo Complete!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(demo())
