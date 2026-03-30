"""
Brown Biotech Research Agent — FastAPI Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.config.settings import Settings
from chatbot.agents.router import RouterAgent
from chatbot.agents.literature import LiteratureAgent
from chatbot.agents.pipeline import PipelineAgent
from chatbot.agents.market import MarketAgent
from chatbot.agents.dataset import DatasetAgent
from chatbot.agents.synthesizer import SynthesizerAgent

app = FastAPI(title="Brown Biotech Research Agent API")

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize settings and agents
settings = Settings()

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = "minimax/minimax-m2.7"

class QueryResponse(BaseModel):
    content: str
    sources: str
    references: List[str]
    cost: float
    intent_type: str
    sources_list: List[str]

@app.get("/")
async def root():
    return {"message": "Brown Biotech Research Agent API", "version": "1.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a research query and return analysis results."""
    try:
        # Route the query
        router = RouterAgent(settings)
        intent = router.classify(request.query)
        
        # Get appropriate agent
        agents = {
            "literature": LiteratureAgent,
            "pipeline": PipelineAgent,
            "market": MarketAgent,
            "dataset": DatasetAgent,
        }
        agent_class = agents.get(intent.agent_type, LiteratureAgent)
        agent = agent_class(settings)
        
        # Execute analysis
        result = agent.analyze(request.query, intent)
        
        # Synthesize response
        synthesizer = SynthesizerAgent(settings)
        response = synthesizer.synthesize(result, intent)
        
        # Extract sources list
        sources_list = []
        if result.get("papers"):
            for p in result["papers"]:
                sources_list.append(f"{p.get('journal', 'Unknown')}: {p.get('pmid', '')}")
        
        return QueryResponse(
            content=response.content,
            sources=response.sources,
            references=response.references,
            cost=response.cost,
            intent_type=response.intent_type,
            sources_list=sources_list
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples")
async def examples():
    """Return example queries."""
    return {
        "examples": [
            {"query": "Analyze the GLP-1 agonist pipeline", "type": "pipeline", "description": "Drug pipeline analysis for GLP-1 target"},
            {"query": "What are the latest findings on TGF-β in skin aging?", "type": "literature", "description": "Literature review on TGF-beta and skin aging"},
            {"query": "Market size of anti-aging cosmetics in Korea", "type": "market", "description": "Market research for Korean anti-aging market"},
            {"query": "Compare Novo Nordisk vs Eli Lilly obesity drugs", "type": "competitor", "description": "Competitive analysis of obesity treatments"},
            {"query": "Analyze skin aging genes from MERFISH data", "type": "dataset", "description": "Single-cell dataset analysis for skin aging targets"},
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
