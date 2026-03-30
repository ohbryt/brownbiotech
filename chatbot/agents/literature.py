"""
Literature Agent — searches and summarizes academic literature
"""
import time
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from config.settings import Settings


@dataclass
class LiteratureResult:
    """Structured literature search result."""
    papers: list
    summary: str
    key_findings: list
    sources: str
    cost: float = 0.0


class LiteratureAgent:
    """
    Searches academic literature using TinyFish + PubMed-style queries.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY
        ) if settings.OPENROUTER_API_KEY else None
    
    def analyze(self, query: str, intent) -> dict:
        """
        Search and summarize literature on the given topic.
        
        Args:
            query: User's natural language query
            intent: Classified intent from router
            
        Returns:
            dict with papers, summary, key_findings
        """
        start_time = time.time()
        cost = 0.0
        
        # Extract search terms from query
        keywords = intent.keywords or query.split()
        
        # Simulate literature search (replace with actual TinyFish/PubMed)
        papers = self._search_papers(query, keywords)
        
        # Generate summary
        summary = self._summarize_papers(papers, query)
        
        # Extract key findings
        key_findings = self._extract_findings(papers)
        
        # Format sources
        sources = f"PubMed ({len(papers)} papers), Web search"
        
        elapsed = time.time() - start_time
        cost = 0.02 * len(papers) + 0.01  # Rough estimate
        
        return {
            "type": "literature",
            "query": query,
            "papers": papers,
            "summary": summary,
            "key_findings": key_findings,
            "sources": sources,
            "cost": cost,
            "time": elapsed
        }
    
    def _search_papers(self, query: str, keywords: list) -> list:
        """
        Search for relevant papers.
        In production, this would use TinyFish API for web search + PubMed.
        """
        # Simulated paper database (in production, use PubMed API + TinyFish)
        mock_papers = [
            {
                "title": f"Role of {' '.join(keywords[:2])} in cellular aging and repair mechanisms",
                "authors": "Kim et al.",
                "journal": "Nature Aging",
                "year": 2024,
                "pmid": "12345678",
                "abstract": f"Recent studies have highlighted the importance of {' '.join(keywords[:2])} in regulating cellular senescence...",
                "key_result": "Expression levels correlate with age-related tissue decline"
            },
            {
                "title": f"Therapeutic targeting of {' '.join(keywords[:2])} in age-related diseases",
                "authors": "Park et al.",
                "journal": "Cell",
                "year": 2024,
                "pmid": "23456789",
                "abstract": f"Targeting {' '.join(keywords[:2])} shows promise in preclinical models of aging...",
                "key_result": "40% improvement in tissue function in aged mice"
            },
            {
                "title": f"Clinical implications of {' '.join(keywords[:2])} modulation in human tissues",
                "authors": "Lee et al.",
                "journal": "NEJM",
                "year": 2023,
                "pmid": "34567890",
                "abstract": f"Phase 2 trials demonstrate safety and efficacy of {' '.join(keywords[:2])} targeting...",
                "key_result": "Phase 2 success rate: 73%"
            },
        ]
        
        return mock_papers[:3]  # Return top 3
    
    def _summarize_papers(self, papers: list, query: str) -> str:
        """Generate a coherent summary of the literature."""
        if not papers:
            return f"No recent literature found for '{query}'."
        
        summary = f"## Literature Summary: {query}\n\n"
        summary += f"Found {len(papers)} relevant papers. "
        summary += "Key themes emerge across recent studies:\n\n"
        
        for i, paper in enumerate(papers, 1):
            summary += f"**{i}. {paper['title']}** ({paper['journal']}, {paper['year']})\n"
            summary += f"- {paper['key_result']}\n"
        
        summary += f"\n### Overall Assessment\n"
        summary += f"The literature suggests that {', '.join(papers[0]['title'].split()[:3])} "
        summary += f"are promising targets for therapeutic intervention in age-related conditions. "
        summary += f"Most studies emphasize the importance of early intervention and sustained modulation."
        
        return summary
    
    def _extract_findings(self, papers: list) -> list:
        """Extract actionable key findings."""
        findings = []
        for paper in papers:
            findings.append({
                "finding": paper['key_result'],
                "source": f"{paper['authors']} ({paper['year']}) - PMID: {paper['pmid']}"
            })
        return findings
