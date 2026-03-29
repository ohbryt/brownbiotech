"""
BrownBioTech Pipeline Tracker
Tracks DGAT1/YARS2 inhibitor development milestones for BROWN-1 and BROWN-2 programs.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Optional


class MilestoneStatus(Enum):
    """Status of a pipeline milestone."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELAYED = "delayed"
    CANCELLED = "cancelled"


class ExperimentType(Enum):
    """Type of experiment."""
    IN_VITRO = "in_vitro"
    IN_VIVO = "in_vivo"
    EX_VIVO = "ex_vivo"
    CLINICAL = "clinical"


class ProgramStage(Enum):
    """Current stage of a pipeline program."""
    DISCOVERY = "discovery"
    LEAD_OPTIMIZATION = "lead_optimization"
    PRECLINICAL = "preclinical"
    IND_ENABLEMENT = "ind_enablement"
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    APPROVED = "approved"


@dataclass
class Milestone:
    """Represents a single milestone in the pipeline."""
    id: str
    name: str
    description: str
    target_date: date
    status: MilestoneStatus = MilestoneStatus.PENDING
    completed_date: Optional[date] = None
    notes: str = ""
    deliverable: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        d["target_date"] = self.target_date.isoformat()
        d["completed_date"] = self.completed_date.isoformat() if self.completed_date else None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Milestone:
        """Create Milestone from dictionary."""
        data = data.copy()
        data["status"] = MilestoneStatus(data["status"])
        data["target_date"] = date.fromisoformat(data["target_date"])
        if data.get("completed_date"):
            data["completed_date"] = date.fromisoformat(data["completed_date"])
        return cls(**data)


@dataclass
class Experiment:
    """Represents an experiment or study in the pipeline."""
    id: str
    program_id: str
    name: str
    experiment_type: ExperimentType
    description: str
    start_date: date
    end_date: Optional[date] = None
    status: MilestoneStatus = MilestoneStatus.PENDING
    results_summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["experiment_type"] = self.experiment_type.value
        d["status"] = self.status.value
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat() if self.end_date else None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Experiment:
        """Create Experiment from dictionary."""
        data = data.copy()
        data["experiment_type"] = ExperimentType(data["experiment_type"])
        data["status"] = MilestoneStatus(data["status"])
        data["start_date"] = date.fromisoformat(data["start_date"])
        if data.get("end_date"):
            data["end_date"] = date.fromisoformat(data["end_date"])
        return cls(**data)


@dataclass
class PipelineProgram:
    """
    Represents a drug development program (BROWN-1 or BROWN-2).
    """
    id: str
    code: str
    name: str
    target: str
    mechanism: str
    indication: str
    stage: ProgramStage = ProgramStage.DISCOVERY
    current_stage_description: str = ""
    founded_date: Optional[date] = None
    milestones: list[Milestone] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["stage"] = self.stage.value
        d["founded_date"] = self.founded_date.isoformat() if self.founded_date else None
        d["milestones"] = [m.to_dict() for m in self.milestones]
        d["experiments"] = [e.to_dict() for e in self.experiments]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PipelineProgram:
        """Create PipelineProgram from dictionary."""
        data = data.copy()
        data["stage"] = ProgramStage(data["stage"])
        if data.get("founded_date"):
            data["founded_date"] = date.fromisoformat(data["founded_date"])
        data["milestones"] = [Milestone.from_dict(m) for m in data.get("milestones", [])]
        data["experiments"] = [Experiment.from_dict(e) for e in data.get("experiments", [])]
        return cls(**data)

    def get_milestone(self, milestone_id: str) -> Optional[Milestone]:
        """Get a milestone by ID."""
        for m in self.milestones:
            if m.id == milestone_id:
                return m
        return None

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        for e in self.experiments:
            if e.id == experiment_id:
                return e
        return None

    def add_milestone(self, milestone: Milestone) -> None:
        """Add a milestone to the program."""
        self.milestones.append(milestone)

    def add_experiment(self, experiment: Experiment) -> None:
        """Add an experiment to the program."""
        self.experiments.append(experiment)


class PipelineTracker:
    """
    Main tracker for BrownBioTech pipeline programs.
    Manages BROWN-1 (DGAT1) and BROWN-2 (YARS2) inhibitor programs.
    """

    DEFAULT_MILESTONES = [
        ("Founded", "Company founded", "2025-07-01"),
        ("Target Validation", "Target validation and hit identification", "2025-10-01"),
        ("Lead Optimization", "Lead compound optimization", "2025-12-01"),
        ("In Vivo Studies", "Initiate in vivo efficacy studies", "2026-01-01"),
        ("IND Filing", "Submit IND application", "2026-07-01"),
        ("Phase 1 Start", "Initiate Phase 1 clinical trial", "2027-01-01"),
    ]

    def __init__(self):
        self.programs: dict[str, PipelineProgram] = {}
        self._init_brown_programs()

    def _init_brown_programs(self) -> None:
        """Initialize BROWN-1 and BROWN-2 programs with default milestones."""

        # BROWN-1: DGAT1 Inhibitor
        brown1 = PipelineProgram(
            id="brown-1",
            code="BROWN-1",
            name="DGAT1 Inhibitor Program",
            target="DGAT1 (Diacylglycerol O-Acyltransferase 1)",
            mechanism="Lipid Metabolism Inhibition",
            indication="Non-Small Cell Lung Cancer (NSCLC)",
            stage=ProgramStage.PRECLINICAL,
            current_stage_description="In vivo efficacy studies ongoing",
            founded_date=date(2025, 7, 1),
            metadata={
                "development_stage": "in_vivo",
                "therapeutic_area": "Oncology",
                "route_of_administration": "Oral",
            },
        )

        for name, desc, target in self.DEFAULT_MILESTONES:
            ms = Milestone(
                id=str(uuid.uuid4())[:8],
                name=name,
                description=desc,
                target_date=date.fromisoformat(target),
            )
            if name == "Target Validation":
                ms.status = MilestoneStatus.COMPLETED
                ms.completed_date = date(2025, 10, 15)
            elif name == "Founded":
                ms.status = MilestoneStatus.COMPLETED
                ms.completed_date = date(2025, 7, 1)
            brown1.add_milestone(ms)

        self.programs["brown-1"] = brown1

        # BROWN-2: YARS2 Inhibitor
        brown2 = PipelineProgram(
            id="brown-2",
            code="BROWN-2",
            name="YARS2 Inhibitor Program",
            target="YARS2 (Tyrosyl-tRNA Synthetase 2)",
            mechanism="Mitochondrial Protein Synthesis Inhibition",
            indication="Solid Tumors",
            stage=ProgramStage.DISCOVERY,
            current_stage_description="In vitro validation stage",
            founded_date=date(2025, 7, 1),
            metadata={
                "development_stage": "in_vitro",
                "therapeutic_area": "Oncology",
                "route_of_administration": "Oral",
            },
        )

        for name, desc, target in self.DEFAULT_MILESTONES:
            ms = Milestone(
                id=str(uuid.uuid4())[:8],
                name=name,
                description=desc,
                target_date=date.fromisoformat(target),
            )
            if name == "Founded":
                ms.status = MilestoneStatus.COMPLETED
                ms.completed_date = date(2025, 7, 1)
            brown2.add_milestone(ms)

        self.programs["brown-2"] = brown2

    def get_program(self, program_id: str) -> Optional[PipelineProgram]:
        """Get a program by ID."""
        return self.programs.get(program_id)

    def update_milestone(
        self,
        program_id: str,
        milestone_id: str,
        status: MilestoneStatus,
        completed_date: Optional[date] = None,
        notes: str = "",
    ) -> bool:
        """
        Update a milestone's status.

        Args:
            program_id: ID of the program (e.g., 'brown-1')
            milestone_id: ID of the milestone to update
            status: New status for the milestone
            completed_date: Date milestone was completed (if applicable)
            notes: Additional notes about the update

        Returns:
            True if milestone was found and updated, False otherwise
        """
        program = self.get_program(program_id)
        if not program:
            return False

        milestone = program.get_milestone(milestone_id)
        if not milestone:
            return False

        milestone.status = status
        if status == MilestoneStatus.COMPLETED and completed_date:
            milestone.completed_date = completed_date
        elif status == MilestoneStatus.COMPLETED and not milestone.completed_date:
            milestone.completed_date = date.today()

        if notes:
            milestone.notes = notes

        # Update program stage based on milestone
        self._update_program_stage(program)

        return True

    def _update_program_stage(self, program: PipelineProgram) -> None:
        """Update program stage based on milestone completion."""
        completed = [m for m in program.milestones if m.status == MilestoneStatus.COMPLETED]
        milestone_names = {m.name for m in completed}

        if "Phase 1 Start" in milestone_names:
            program.stage = ProgramStage.PHASE_1
        elif "IND Filing" in milestone_names:
            program.stage = ProgramStage.IND_ENABLEMENT
        elif "In Vivo Studies" in milestone_names:
            program.stage = ProgramStage.PRECLINICAL
        elif "Lead Optimization" in milestone_names:
            program.stage = ProgramStage.LEAD_OPTIMIZATION
        elif "Target Validation" in milestone_names:
            program.stage = ProgramStage.DISCOVERY

    def get_progress(self, program_id: str) -> dict:
        """
        Get progress report for a program.

        Args:
            program_id: ID of the program

        Returns:
            Dictionary with progress information
        """
        program = self.get_program(program_id)
        if not program:
            return {"error": f"Program {program_id} not found"}

        total_milestones = len(program.milestones)
        completed_milestones = sum(
            1 for m in program.milestones if m.status == MilestoneStatus.COMPLETED
        )
        in_progress_milestones = sum(
            1 for m in program.milestones if m.status == MilestoneStatus.IN_PROGRESS
        )
        pending_milestones = sum(
            1 for m in program.milestones if m.status == MilestoneStatus.PENDING
        )

        total_experiments = len(program.experiments)
        completed_experiments = sum(
            1 for e in program.experiments if e.status == MilestoneStatus.COMPLETED
        )

        # Calculate overall progress percentage
        progress_pct = (completed_milestones / total_milestones * 100) if total_milestones > 0 else 0

        # Find next upcoming milestone
        today = date.today()
        upcoming = [
            m for m in program.milestones
            if m.status in (MilestoneStatus.PENDING, MilestoneStatus.IN_PROGRESS)
            and m.target_date >= today
        ]
        next_milestone = min(upcoming, key=lambda m: m.target_date) if upcoming else None

        return {
            "program_id": program.id,
            "program_code": program.code,
            "program_name": program.name,
            "stage": program.stage.value,
            "progress_percentage": round(progress_pct, 1),
            "milestones": {
                "total": total_milestones,
                "completed": completed_milestones,
                "in_progress": in_progress_milestones,
                "pending": pending_milestones,
            },
            "experiments": {
                "total": total_experiments,
                "completed": completed_experiments,
            },
            "next_milestone": {
                "name": next_milestone.name,
                "target_date": next_milestone.target_date.isoformat(),
                "days_until": (next_milestone.target_date - today).days,
            } if next_milestone else None,
        }

    def generate_report(self, program_id: Optional[str] = None) -> str:
        """
        Generate a human-readable progress report.

        Args:
            program_id: Optional specific program ID. If None, generates report for all programs.

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BrownBioTech Pipeline Progress Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 60)

        programs_to_report = (
            [self.get_program(program_id)] if program_id else list(self.programs.values())
        )

        for program in programs_to_report:
            if not program:
                continue

            progress = self.get_progress(program.id)

            lines.append(f"\n{program.code}: {program.name}")
            lines.append("-" * 40)
            lines.append(f"Target: {program.target}")
            lines.append(f"Mechanism: {program.mechanism}")
            lines.append(f"Indication: {program.indication}")
            lines.append(f"Current Stage: {program.stage.value.replace('_', ' ').title()}")
            lines.append(f"Progress: {progress['progress_percentage']}%")
            lines.append(f"Milestones: {progress['milestones']['completed']}/{progress['milestones']['total']} completed")

            if progress.get("next_milestone"):
                nm = progress["next_milestone"]
                lines.append(f"Next Milestone: {nm['name']} ({nm['target_date']}, {nm['days_until']} days)")

            lines.append("\nMilestone Details:")
            for ms in program.milestones:
                status_icon = {
                    MilestoneStatus.COMPLETED: "[✓]",
                    MilestoneStatus.IN_PROGRESS: "[→]",
                    MilestoneStatus.PENDING: "[ ]",
                    MilestoneStatus.DELAYED: "[!]",
                    MilestoneStatus.CANCELLED: "[×]",
                }.get(ms.status, "[?]")
                completed_str = f" (completed: {ms.completed_date.isoformat()})" if ms.completed_date else ""
                lines.append(f"  {status_icon} {ms.name}: {ms.target_date.isoformat()}{completed_str}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def export_to_json(self, filepath: Optional[str] = None, program_id: Optional[str] = None) -> str:
        """
        Export pipeline data to JSON.

        Args:
            filepath: Optional path to save JSON file. If None, returns JSON string.
            program_id: Optional specific program ID. If None, exports all programs.

        Returns:
            JSON string, or empty string if saved to file
        """
        if program_id:
            program = self.get_program(program_id)
            data = program.to_dict() if program else {}
        else:
            data = {pid: p.to_dict() for pid, p in self.programs.items()}

        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            return ""

        return json_str

    def add_experiment(
        self,
        program_id: str,
        name: str,
        experiment_type: ExperimentType,
        description: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> Optional[Experiment]:
        """Add an experiment to a program."""
        program = self.get_program(program_id)
        if not program:
            return None

        experiment = Experiment(
            id=str(uuid.uuid4())[:8],
            program_id=program_id,
            name=name,
            experiment_type=experiment_type,
            description=description,
            start_date=start_date,
            end_date=end_date,
        )
        program.add_experiment(experiment)
        return experiment

    @classmethod
    def from_json(cls, json_str: str) -> PipelineTracker:
        """Load PipelineTracker from JSON string."""
        data = json.loads(json_str)
        tracker = cls()
        tracker.programs = {
            pid: PipelineProgram.from_dict(p) for pid, p in data.items()
        }
        return tracker

    @classmethod
    def load_from_file(cls, filepath: str) -> PipelineTracker:
        """Load PipelineTracker from JSON file."""
        with open(filepath, "r") as f:
            return cls.from_json(f.read())


def main():
    """Demo usage of PipelineTracker."""
    tracker = PipelineTracker()

    # Print initial report
    print(tracker.generate_report())

    # Update a milestone
    tracker.update_milestone(
        "brown-1",
        tracker.programs["brown-1"].milestones[3].id,  # In Vivo Studies
        MilestoneStatus.IN_PROGRESS,
        notes="Started xenograft model studies",
    )

    # Add an experiment
    tracker.add_experiment(
        "brown-1",
        name="NSCLC Xenograft Efficacy Study",
        experiment_type=ExperimentType.IN_VIVO,
        description="Evaluate BROWN-1 efficacy in NSCLC xenograft model",
        start_date=date(2026, 1, 15),
    )

    # Print updated report
    print("\n" + tracker.generate_report())

    # Export to JSON
    json_data = tracker.export_to_json(program_id="brown-1")
    print("\nBROWN-1 JSON Export:")
    print(json_data[:500] + "..." if len(json_data) > 500 else json_data)


if __name__ == "__main__":
    main()
