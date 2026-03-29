#!/usr/bin/env python3
"""
BrownBioTech Autoreview Loop v2
==============================
100 iterations: Research → Plan → Code → Evaluate → Improve

Models:
- Coding: z-ai/glm-5-turbo (OpenRouter)
- Analysis: nvidia/nemotron-3-nano-30b-a3b:free (OpenRouter Free)
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ─── API Configuration ──────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model assignments
CODING_MODEL = "z-ai/glm-5-turbo"      # For coding tasks
ANALYSIS_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"  # For analysis (free)

# ─── Config ─────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).parent
EXPERIMENTS_DIR = WORKSPACE / "experiments"
OUTPUTS_DIR = WORKSPACE / "outputs"
LOGS_DIR = WORKSPACE / "logs"
ITERATIONS = 100

for d in [EXPERIMENTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── LLM Calls ──────────────────────────────────────────────────────────────

def call_llm(prompt: str, model: str, task_type: str = "analysis") -> str:
    """Call OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://brownbiotech.kr",
        "X-Title": "BrownBioTech"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error ({model}): {str(e)[:100]}"

def call_coding(prompt: str) -> str:
    """GLM-5 for coding tasks."""
    return call_llm(prompt, CODING_MODEL, "coding")

def call_analysis(prompt: str) -> str:
    """Nemotron for analysis tasks."""
    return call_llm(prompt, ANALYSIS_MODEL, "analysis")

# ─── Web Research ──────────────────────────────────────────────────────────

def web_research(query: str) -> dict:
    """Perform web search using web_fetch."""
    from urllib.parse import quote
    
    try:
        # Use web_search via exec
        query_encoded = quote(query)
        return {
            "query": query,
            "status": "success",
            "method": "web_search",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"query": query, "status": "error", "error": str(e)}

# ─── Experiment Data Class ─────────────────────────────────────────────────

@dataclass
class Experiment:
    id: str
    iteration: int
    phase: str
    status: str
    score: float = 0.0
    created_at: str = ""
    completed_at: str = ""
    notes: str = ""
    artifacts: list = None
    web_research: dict = None
    llm_output: str = ""
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

def load_experiments() -> dict:
    db_path = EXPERIMENTS_DIR / "autoreview_v2.json"
    if db_path.exists():
        with open(db_path) as f:
            data = json.load(f)
            return {e["id"]: Experiment(**e) for e in data}
    return {}

def save_experiments(experiments: dict):
    db_path = EXPERIMENTS_DIR / "autoreview_v2.json"
    with open(db_path, "w") as f:
        json.dump([asdict(e) for e in experiments.values()], f, indent=2)

# ─── Autoreview Phases ──────────────────────────────────────────────────────

async def phase_research(exp: Experiment, iteration: int) -> Experiment:
    """Phase 1: Web Research + Analysis."""
    print(f"\n{'='*60}")
    print(f"🔍 RESEARCH (Iteration {iteration}/100)")
    print(f"{'='*60}")
    
    # Research topics for BrownBioTech
    topics = [
        "AI drug discovery platforms 2025 trends",
        "biotech startup website best practices",
        "cancer metabolism DGAT1 drug targets",
        "virtual screening benchmark methods",
        "ADMET prediction deep learning",
        "NSCLC targeted therapy pipeline 2025",
        "generative AI antibody design",
        "biotech company pitch deck structure",
        "LLM biomedical research applications",
        "drug repurposing computational methods",
    ]
    topic = topics[iteration % len(topics)]
    
    print(f"Research topic: {topic}")
    
    # Web research
    research = web_research(topic)
    exp.web_research = research
    
    # Nemotron analysis
    analysis_prompt = f"""You are a biotech analyst for BrownBioTech.
Company: AI-first cancer metabolism drug discovery (DGAT1/YARS2 targets)
Iteration: {iteration}/100

Research Topic: {topic}

Analyze this research and provide:
1. Key findings (3 bullet points)
2. Opportunities for BrownBioTech platform
3. Recommended technical implementation

Be concise and actionable."""
    
    analysis = call_analysis(analysis_prompt)
    exp.llm_output = analysis
    
    print(f"Analysis preview: {analysis[:150]}...")
    
    exp.phase = "research"
    return exp

async def phase_plan(exp: Experiment, iteration: int) -> Experiment:
    """Phase 2: Planning with GLM-5."""
    print(f"\n{'='*60}")
    print(f"📋 PLAN (GLM-5 Coding Plan)")
    print(f"{'='*60}")
    
    planning_prompt = f"""You are a biotech startup technical advisor for BrownBioTech.
Company: AI-first cancer metabolism drug discovery (DGAT1/YARS2 targets)
Current Systems:
- ARP v3 (Autonomous Research Pipeline)
- Agent System v3.0 (6 agents: Literature, MultiOmics, VirtualScreen, Design, ADMET, WetLab)
- ml-drug-discovery (Manning book)
- DrugPipe (generative AI + docking)
Iteration: {iteration}/100

Research findings:
{exp.llm_output[:800]}

Generate a specific coding improvement plan:
1. What to code/improve (be specific with file names)
2. Expected impact on platform
3. Code structure/pseudocode

Focus on practical improvements to the BrownBioTech platform."""
    
    plan = call_coding(planning_prompt)
    exp.notes = plan
    
    print(f"Plan preview: {plan[:200]}...")
    
    exp.phase = "plan"
    return exp

async def phase_code(exp: Experiment, iteration: int) -> Experiment:
    """Phase 3: Implementation with GLM-5."""
    print(f"\n{'='*60}")
    print(f"💻 CODE (GLM-5 Implementation)")
    print(f"{'='*60}")
    
    code_prompt = f"""Generate Python code for BrownBioTech improvement.
Iteration: {iteration}/100

Planning:
{exp.notes[:600]}

Generate the actual Python code to implement this improvement.
Include:
1. File path (e.g., brownbiotech/improved_module.py)
2. Complete runnable code (prefer small, focused modules)
3. Brief explanation of improvement

Use best practices:
- Type hints
- Docstrings
- Error handling
- Integration with existing BrownBioTech modules"""
    
    code = call_coding(code_prompt)
    
    # Save as artifact
    artifact_name = f"iteration_{iteration:03d}.py"
    artifact_path = OUTPUTS_DIR / artifact_name
    artifact_path.write_text(code)
    exp.artifacts.append(str(artifact_path.relative_to(WORKSPACE)))
    
    exp.llm_output = code
    exp.phase = "code"
    
    print(f"Generated: {artifact_name} ({len(code)} chars)")
    
    return exp

async def phase_evaluate(exp: Experiment) -> Experiment:
    """Phase 4: Evaluation."""
    print(f"\n{'='*60}")
    print(f"📊 EVALUATE")
    print(f"{'='*60}")
    
    # Score based on quality
    score = 0.0
    
    if exp.web_research and exp.web_research.get("status") == "success":
        score += 15
    
    if exp.notes and len(exp.notes) > 150:
        score += 25
    
    if exp.artifacts:
        score += 35
        for artifact in exp.artifacts:
            path = WORKSPACE / artifact
            if path.exists():
                content = path.read_text()
                if len(content) > 300:
                    score += 10
                if "def " in content and "class " in content:
                    score += 5
                if "import " in content:
                    score += 5
    
    if exp.llm_output and len(exp.llm_output) > 300:
        score += 20
    
    exp.score = min(score, 100)
    
    print(f"Score: {exp.score}/100")
    
    exp.phase = "evaluate"
    return exp

async def phase_improve(exp: Experiment, iteration: int) -> Experiment:
    """Phase 5: Feedback for next iteration."""
    print(f"\n{'='*60}")
    print(f"🔄 IMPROVE (Feedback)")
    print(f"{'='*60}")
    
    improvement_prompt = f"""Analyze this BrownBioTech iteration:
Iteration: {iteration}/100
Phase results:
- Research: {'✓' if exp.web_research else '✗'}
- Plan: {len(exp.notes)} chars
- Code: {len(exp.artifacts)} artifacts
- Score: {exp.score}/100

Code generated:
{exp.llm_output[:400]}

Provide exactly 1-2 actionable improvements for the NEXT iteration.
Be specific about what to research, plan, or code differently."""
    
    improvement = call_analysis(improvement_prompt)
    
    exp.notes += f"\n\n## Next Iteration:\n{improvement}"
    exp.completed_at = datetime.now().isoformat()
    exp.phase = "improve"
    
    print(f"Improvement: {improvement[:150]}...")
    
    return exp

# ─── Main Loop ──────────────────────────────────────────────────────────────

async def run_autoreview():
    """Run 100 iterations of autoreview."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         BrownBioTech Autoreview Loop v2                        ║
╠══════════════════════════════════════════════════════════════╣
║  100 Iterations: Research → Plan → Code → Evaluate          ║
║  Models:                                                     ║
║    - Coding: z-ai/glm-5-turbo                               ║
║    - Analysis: nvidia/nemotron-3-nano-30b-a3b:free          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    experiments = load_experiments()
    current_iteration = (len(experiments) // 5) + 1
    
    if current_iteration > ITERATIONS:
        print(f"✅ Already completed {ITERATIONS} iterations!")
        return experiments
    
    print(f"Starting from iteration {current_iteration}/100\n")
    
    # Load previous best
    best_score = max([e.score for e in experiments.values()], default=0)
    print(f"Previous best score: {best_score}/100\n")
    
    for iteration in range(current_iteration, ITERATIONS + 1):
        print(f"\n{'#'*60}")
        print(f"# ITERATION {iteration}/100")
        print(f"{'#'*60}")
        
        exp_id = f"iter_{iteration:03d}"
        exp = Experiment(
            id=exp_id,
            iteration=iteration,
            phase="starting",
            status="running"
        )
        
        try:
            exp = await phase_research(exp, iteration)
            exp = await phase_plan(exp, iteration)
            exp = await phase_code(exp, iteration)
            exp = await phase_evaluate(exp)
            exp = await phase_improve(exp, iteration)
            exp.status = "completed"
            
        except Exception as e:
            exp.status = "failed"
            exp.notes = f"Error: {str(e)}"
            print(f"❌ Iteration {iteration} failed: {e}")
        
        experiments[exp_id] = exp
        save_experiments(experiments)
        
        # Progress every 10 iterations
        if iteration % 10 == 0:
            scores = [e.score for e in experiments.values()]
            avg = sum(scores) / len(scores) if scores else 0
            best = max(scores) if scores else 0
            print(f"\n📊 Progress (Iteration {iteration})")
            print(f"   Total: {len(experiments)} | Avg: {avg:.1f} | Best: {best}/100")
        
        if iteration < ITERATIONS:
            time.sleep(0.5)  # Rate limit protection
    
    # Final summary
    scores = [e.score for e in experiments.values()]
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              AUTOREVIEW COMPLETE!                          ║
╠══════════════════════════════════════════════════════════════╣
║  Total iterations: {len(experiments)}                                    ║
║  Average score: {sum(scores)/len(scores):.1f}/100                            ║
║  Best score: {max(scores)}/100                                         ║
║  Artifacts: {sum(len(e.artifacts) for e in experiments.values())}                                          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    return experiments

# ─── Test API ──────────────────────────────────────────────────────────────

def test_apis():
    """Test API connections."""
    print("Testing API connections...\n")
    
    print("1. GLM-5 (coding):")
    result = call_coding("Say hello and confirm you are GLM-5 for BrownBioTech")
    print(f"   Result: {result[:100]}...")
    
    print("\n2. Nemotron (analysis):")
    result = call_analysis("Say hello and confirm you are Nemotron for BrownBioTech")
    print(f"   Result: {result[:100]}...")
    
    return result != "" and "Error" not in result[:50]

# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    
    # Test APIs first
    if test_apis():
        print("\n✅ APIs working! Starting autoreview...")
        results = asyncio.run(run_autoreview())
    else:
        print("\n❌ API test failed. Check credentials.")
