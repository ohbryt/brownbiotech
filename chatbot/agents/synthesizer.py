"""
Synthesizer Agent — compiles agent results into final responses
"""
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from config.settings import Settings


@dataclass
class SynthesizedResponse:
    """Final synthesized response."""
    content: str
    sources: str
    references: list
    cost: float
    intent_type: str


class SynthesizerAgent:
    """
    Takes results from specialized agents and synthesizes
    a coherent, well-formatted response.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY
        ) if settings.OPENROUTER_API_KEY else None
    
    def synthesize(self, agent_result: dict, intent) -> SynthesizedResponse:
        """
        Synthesize a final response from agent results.
        
        Args:
            agent_result: Output from the appropriate agent
            intent: Original query intent
            
        Returns:
            SynthesizedResponse with formatted content
        """
        result_type = agent_result.get("type", "unknown")
        
        if result_type == "literature":
            return self._synthesize_literature(agent_result, intent)
        elif result_type == "pipeline":
            return self._synthesize_pipeline(agent_result, intent)
        elif result_type == "market":
            return self._synthesize_market(agent_result, intent)
        elif result_type == "dataset":
            return self._synthesize_dataset(agent_result, intent)
        else:
            return self._synthesize_generic(agent_result, intent)
    
    def _synthesize_literature(self, result: dict, intent) -> SynthesizedResponse:
        """Synthesize literature search results."""
        papers = result.get("papers", [])
        summary = result.get("summary", "")
        key_findings = result.get("key_findings", [])
        sources = result.get("sources", "")
        
        # Build references list
        references = []
        for paper in papers:
            ref = f"[PMID:{paper['pmid']}] {paper['authors']} ({paper['year']}). {paper['title']}. {paper['journal']}."
            references.append(ref)
        
        # Format content
        content = summary + "\n\n"
        
        if key_findings:
            content += "### 🎯 Actionable Insights\n\n"
            for finding in key_findings:
                content += f"- {finding['finding']}\n"
                content += f"  *Source: {finding['source']}*\n"
        
        return SynthesizedResponse(
            content=content,
            sources=sources,
            references=references,
            cost=result.get("cost", 0),
            intent_type="literature"
        )
    
    def _synthesize_pipeline(self, result: dict, intent) -> SynthesizedResponse:
        """Synthesize drug pipeline results."""
        summary = result.get("summary", "")
        sources = result.get("sources", "")
        mechanism = result.get("mechanism", result.get("company", "General"))
        
        # Add competitive insights
        content = summary + "\n\n"
        content += "### 💼 Business Implications\n\n"
        
        if result.get("stages", {}).get("approved"):
            content += f"✅ **{len(result['stages']['approved'])} approved drugs** indicate validated market\n"
        
        if result.get("stages", {}).get("phase3"):
            content += f"🔬 **{len(result['stages']['phase3'])} Phase 3 candidates** — expected approvals 2025-2027\n"
        
        content += "\n### 📋 Strategic Recommendations\n"
        
        if mechanism in ["GLP-1", "FXR", "PPAR"]:
            content += f"- **{mechanism}** space is competitive; consider:\n"
            content += "  - Combination therapy approaches\n"
            content += "  - Novel delivery mechanisms (oral, topical)\n"
            content += "  - Biomarker-driven patient selection\n"
        
        return SynthesizedResponse(
            content=content,
            sources=sources,
            references=[],
            cost=result.get("cost", 0),
            intent_type="pipeline"
        )
    
    def _synthesize_market(self, result: dict, intent) -> SynthesizedResponse:
        """Synthesize market analysis results."""
        summary = result.get("summary", "")
        sources = result.get("sources", "")
        segment = result.get("segment", "general market")
        
        content = summary + "\n\n"
        content += "### 🎯 Entry Opportunities\n\n"
        
        trends = result.get("trends", [])
        if trends:
            content += "**Emerging trends to leverage:**\n"
            for trend in trends[:3]:
                content += f"- 🚀 {trend}\n"
        
        content += "\n### ⚠️ Key Risks\n"
        content += "- Regulatory hurdles (FDA, MFDS approval)\n"
        content += "- Established player dominance\n"
        content += "- Reimbursement challenges\n"
        
        return SynthesizedResponse(
            content=content,
            sources=sources,
            references=[],
            cost=result.get("cost", 0),
            intent_type="market"
        )
    
    def _synthesize_dataset(self, result: dict, intent) -> SynthesizedResponse:
        """Synthesize dataset analysis results."""
        summary = result.get("summary", "")
        sources = result.get("sources", "")
        
        content = summary
        
        # Add visualization placeholder
        content += "\n\n### 📊 Visualizations Available\n"
        content += "- UMAP of cell clusters (age comparison)\n"
        content += "- Spatial map of gene expression\n"
        content += "- Differential expression volcano plot\n"
        content += "- Pathway enrichment analysis\n"
        
        return SynthesizedResponse(
            content=content,
            sources=sources,
            references=[],
            cost=result.get("cost", 0),
            intent_type="dataset"
        )
    
    def _synthesize_generic(self, result: dict, intent) -> SynthesizedResponse:
        """Generic synthesis for unknown types."""
        content = result.get("summary", str(result))
        
        return SynthesizedResponse(
            content=content,
            sources=result.get("sources", ""),
            references=[],
            cost=result.get("cost", 0),
            intent_type="generic"
        )
