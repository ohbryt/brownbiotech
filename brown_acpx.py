"""
BrownBioTech acpx Integration
=============================
ACP (Agent Client Protocol) integration for BrownBioTech AI pipeline.

acpx enables structured agent-to-agent communication via CLI instead of PTY scraping.
Supports: Codex, Claude Code, Pi, OpenClaw ACP

Usage:
    python brown_acpx.py --agent codex --prompt "analyze DGAT1 DepMap data"
    python brown_acpx.py --agent codex --session brown1 --prompt "generate IND sections"
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ACPXConfig:
    """acpx configuration for BrownBioTech."""
    # Default agent
    default_agent: str = "codex"
    
    # Session management
    session_scope: str = "brownbiotech"
    session_name: Optional[str] = None
    
    # Agent paths
    codex_path: Optional[str] = None
    claude_path: Optional[str] = None
    
    # Output
    output_dir: Path = Path("outputs/acpx")
    structured_output: bool = True
    
    # Model settings
    reasoning_effort: str = "high"  # For Claude Code

# ─── Session Manager ──────────────────────────────────────────────────────────

class BrownACPXSession:
    """Manages acpx sessions for BrownBioTech projects."""
    
    def __init__(self, config: ACPXConfig = None):
        self.config = config or ACPXConfig()
        self.session_id: Optional[str] = None
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        prompt: str,
        agent: str = None,
        session_name: str = None,
        wait: bool = True,
        json_output: bool = True
    ) -> Dict[str, Any]:
        """
        Run acpx command with the given prompt.
        
        Args:
            prompt: The task to delegate to the agent
            agent: Agent to use (codex, claude, pi, openclaw)
            session_name: Named session for this workstream
            wait: Wait for completion (False = fire-and-forget)
            json_output: Capture structured output
        
        Returns:
            Dict with output, session_id, status
        """
        agent = agent or self.config.default_agent
        session_name = session_name or self.config.session_name
        
        # Build acpx command
        cmd = ["acpx", agent]
        
        if session_name:
            cmd.extend(["-s", session_name])
        
        if not wait:
            cmd.append("--no-wait")
        
        # Add prompt
        cmd.extend(["prompt", prompt])
        
        # Environment
        env = os.environ.copy()
        if self.config.reasoning_effort:
            env["REASONING_EFFORT"] = self.config.reasoning_effort
        
        print(f"→ Running: {' '.join(cmd)}")
        print(f"  Prompt: {prompt[:80]}...")
        
        start_time = datetime.now()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout
                env=env
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            output = {
                "status": "success" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": elapsed,
                "agent": agent,
                "session": session_name,
                "timestamp": start_time.isoformat(),
            }
            
            if json_output:
                # Save structured output
                output_file = self.output_dir / f"{agent}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(output, f, indent=2)
                print(f"  ✓ Output saved: {output_file}")
            
            if result.returncode != 0:
                print(f"  ✗ Error: {result.stderr[:200]}")
            else:
                print(f"  ✓ Completed in {elapsed:.1f}s")
            
            return output
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "elapsed_seconds": 300,
                "agent": agent,
                "session": session_name,
                "timestamp": start_time.isoformat(),
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "error": f"acpx not found. Install: npm install -g acpx@latest",
            }
    
    def exec_one_shot(self, prompt: str, agent: str = None) -> Dict[str, Any]:
        """Run one-shot task without session state."""
        agent = agent or self.config.default_agent
        
        cmd = ["acpx", agent, "exec", "--", prompt]
        
        print(f"→ One-shot: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=os.environ.copy()
        )
        
        return {
            "status": "success" if result.returncode == 0 else "failed",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "agent": agent,
        }
    
    def list_sessions(self, agent: str = None) -> List[Dict]:
        """List active sessions for an agent."""
        agent = agent or self.config.default_agent
        
        result = subprocess.run(
            ["acpx", agent, "sessions", "list"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Parse session list
            sessions = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    sessions.append({"raw": line})
            return sessions
        
        return []
    
    def cancel_session(self, agent: str = None) -> bool:
        """Cancel current running prompt."""
        agent = agent or self.config.default_agent
        
        result = subprocess.run(
            ["acpx", agent, "cancel"],
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0

# ─── BrownBioTech Workflows ───────────────────────────────────────────────────

class BrownACPXWorkflow:
    """Pre-defined workflows for BrownBioTech using acpx."""
    
    def __init__(self, config: ACPXConfig = None):
        self.session = BrownACPXSession(config)
    
    def target_analysis_workflow(self, target: str = "DGAT1") -> Dict[str, Any]:
        """
        Complete target analysis workflow using multiple agents.
        
        1. Literature search (Nemotron)
        2. Multi-omics analysis (Gemini Flash Lite)
        3. DepMap validation (Claude Code)
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"TARGET ANALYSIS WORKFLOW: {target}")
        print(f"{'='*60}")
        
        # Step 1: Literature
        print("\n[1/3] Literature Analysis...")
        results["literature"] = self.session.run(
            prompt=f"Search and summarize recent literature on {target} in cancer. "
                   f"Focus on: 1) role in tumor metabolism, 2) therapeutic targeting potential, "
                   f"3) clinical trial status. Format as markdown.",
            agent="codex",
            session_name=f"{target.lower()}_literature"
        )
        
        # Step 2: Multi-omics
        print("\n[2/3] Multi-Omics Analysis...")
        results["multiomics"] = self.session.run(
            prompt=f"Analyze TCGA and DepMap data for {target}. "
                   f"Include: expression in tumors vs normal, survival correlation, "
                   f"CRISPR dependency scores. Generate summary statistics.",
            agent="claude",
            session_name=f"{target.lower()}_multiomics"
        )
        
        # Step 3: Validation plan
        print("\n[3/3] Validation Plan...")
        results["validation"] = self.session.run(
            prompt=f"Based on the {target} analysis results, propose: "
                   f"1) in vitro assays to validate target importance, "
                   f"2) animal models for in vivo studies, "
                   f"3) biomarker strategy for clinical development.",
            agent="codex",
            session_name=f"{target.lower()}_validation"
        )
        
        # Save combined results
        output_file = self.session.output_dir / f"{target.lower()}_workflow_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Workflow complete. Results: {output_file}")
        return results
    
    def molecular_design_workflow(self, target: str = "DGAT1") -> Dict[str, Any]:
        """
        Molecular design workflow for compound generation.
        
        1. Structure analysis (Claude Code)
        2. Lead optimization (Pi)
        3. ADMET prediction (Claude Code)
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"MOLECULAR DESIGN WORKFLOW: {target}")
        print(f"{'='*60}")
        
        # Step 1: Structure analysis
        print("\n[1/3] Target Structure Analysis...")
        results["structure"] = self.session.run(
            prompt=f"Research the binding pocket of {target}. "
                   f"Find known co-crystal structures. "
                   f"Identify key residues for inhibitor design. "
                   f"Generate a detailed structural analysis.",
            agent="claude",
            session_name=f"{target.lower()}_structure"
        )
        
        # Step 2: Lead optimization
        print("\n[2/3] Lead Optimization...")
        results["optimization"] = self.session.run(
            prompt=f"Using the {target} structural insights, propose "
                   f"lead optimization strategies. Focus on: "
                   f"1) improving binding affinity, "
                   f"2) addressing ADMET properties, "
                   f"3) patentability considerations.",
            agent="pi",
            session_name=f"{target.lower()}_optimization"
        )
        
        # Step 3: ADMET
        print("\n[3/3] ADMET Assessment...")
        results["admet"] = self.session.run(
            prompt=f"Create a comprehensive ADMET prediction pipeline for {target} inhibitors. "
                   f"Include: solubility, permeability, CYP inhibition, hERG liability, "
                   f"and in vivo PK prediction.",
            agent="claude",
            session_name=f"{target.lower()}_admet"
        )
        
        return results
    
    def ind_prep_workflow(self, program: str = "BROWN-1") -> Dict[str, Any]:
        """
        IND preparation workflow.
        
        1. Draft sections (Claude Code)
        2. Review and polish (Pi)
        3. Final assembly (Claude Code)
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"IND PREPARATION WORKFLOW: {program}")
        print(f"{'='*60}")
        
        # Step 1: Generate IND sections
        print("\n[1/3] Generate IND Draft...")
        results["draft"] = self.session.run(
            prompt=f"Generate a detailed IND application draft for {program}. "
                   f"Include all required sections: 1) Introduction, 2) Target Validation, "
                   f"3) Preclinical Pharmacology, 4) Toxicology, 5) Manufacturing. "
                   f"Use placeholder data where actual data is not available.",
            agent="claude",
            session_name=f"{program.lower()}_draft"
        )
        
        # Step 2: Review
        print("\n[2/3] Scientific Review...")
        results["review"] = self.session.run(
            prompt=f"Review the IND draft for {program}. "
                   f"Identify gaps, inconsistencies, and areas needing "
                   f"additional data. Provide specific recommendations.",
            agent="pi",
            session_name=f"{program.lower()}_review"
        )
        
        # Step 3: Final assembly
        print("\n[3/3] Final Assembly...")
        results["final"] = self.session.run(
            prompt=f"Based on the review comments, finalize the {program} IND. "
                   f"Ensure all sections are consistent and complete. "
                   f"Format according to FDA eCTD guidelines.",
            agent="claude",
            session_name=f"{program.lower()}_final"
        )
        
        return results

# ─── CLI Interface ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BrownBioTech acpx Integration - AI Agent Pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run single task
    run_parser = subparsers.add_parser("run", help="Run a single task")
    run_parser.add_argument("--prompt", required=True, help="Task prompt")
    run_parser.add_argument("--agent", default="codex", choices=["codex", "claude", "pi", "openclaw"])
    run_parser.add_argument("--session", help="Named session")
    run_parser.add_argument("--no-wait", action="store_true", help="Fire and forget")
    
    # Workflows
    workflow_parser = subparsers.add_parser("workflow", help="Run predefined workflow")
    workflow_parser.add_argument("--type", required=True, 
                                choices=["target", "design", "ind"],
                                help="Workflow type")
    workflow_parser.add_argument("--target", default="DGAT1", help="Target gene/protein")
    workflow_parser.add_argument("--program", default="BROWN-1", help="Program name (for IND)")
    
    # Session management
    session_parser = subparsers.add_parser("sessions", help="Manage sessions")
    session_parser.add_argument("--list", action="store_true", help="List sessions")
    session_parser.add_argument("--cancel", action="store_true", help="Cancel current")
    
    # One-shot
    exec_parser = subparsers.add_parser("exec", help="One-shot execution")
    exec_parser.add_argument("--prompt", required=True, help="Task prompt")
    exec_parser.add_argument("--agent", default="codex")
    
    args = parser.parse_args()
    
    config = ACPXConfig()
    workflow = BrownACPXWorkflow(config)
    
    if args.command == "run":
        result = workflow.session.run(
            prompt=args.prompt,
            agent=args.agent,
            session_name=args.session,
            wait=not args.no_wait
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == "workflow":
        if args.type == "target":
            result = workflow.target_analysis_workflow(args.target)
        elif args.type == "design":
            result = workflow.molecular_design_workflow(args.target)
        elif args.type == "ind":
            result = workflow.ind_prep_workflow(args.program)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.command == "sessions":
        if args.list:
            sessions = workflow.session.list_sessions()
            for s in sessions:
                print(s)
        elif args.cancel:
            success = workflow.session.cancel_session()
            print(f"Cancel: {'success' if success else 'failed'}")
    
    elif args.command == "exec":
        result = workflow.session.exec_one_shot(args.prompt, args.agent)
        print(result.get("stdout", result.get("stderr", "")))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
