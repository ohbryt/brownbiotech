"""
Router Agent — classifies user queries and routes to appropriate agent
"""
import re
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from config.settings import Settings


@dataclass
class QueryIntent:
    """Structured query intent."""
    agent_type: str  # literature, pipeline, market, dataset, competitor
    confidence: float
    keywords: list
    target_disease: Optional[str] = None
    target_mechanism: Optional[str] = None
    target_company: Optional[str] = None
    target_region: Optional[str] = None


class RouterAgent:
    """
    Classifies user queries and determines which agent should handle them.
    Uses keyword matching + LLM for complex queries.
    """
    
    # Keyword patterns for different query types
    KEYWORD_PATTERNS = {
        "literature": [
            "paper", "study", "research", "article", "pubmed", 
            "recent", "latest", "findings", "evidence", "review",
            "journal", "publication", "meta-analysis", "clinical trial"
        ],
        "pipeline": [
            "pipeline", "clinical trial", "phase", "fda", "approval",
            "drug", "mechanism", "target", "candidate", "development",
            "IND", "NDA", "bLA", "efficacy", "safety"
        ],
        "market": [
            "market", "size", "revenue", "growth", "CAGR", "forecast",
            "trend", "industry", "competitive", "share", "segmentation",
            "opportunity", "landscape"
        ],
        "dataset": [
            "analyze", "data", "dataset", "expression", "cell", "single cell",
            "RNA", "sequencing", "MERFISH", "atlas", "transcriptomics",
            "gene", "cluster", "UMAP"
        ],
        "competitor": [
            "competitor", "company", "portfolio", "strategy", "acquisition",
            "merger", "partner", "deal", "license", "vs", "comparison"
        ]
    }
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY
        ) if settings.OPENROUTER_API_KEY else None
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify a user query and return routing intent.
        
        Strategy:
        1. First try keyword matching for obvious cases
        2. Use LLM for complex/ambiguous queries
        """
        query_lower = query.lower()
        
        # Check for dataset-related keywords first (most specific)
        if any(kw in query_lower for kw in ["upload", "my data", "h5ad", "single cell", "merfish"]):
            return QueryIntent(
                agent_type="dataset",
                confidence=0.9,
                keywords=self._extract_keywords(query)
            )
        
        # Count keyword matches for each category
        scores = {}
        for agent_type, keywords in self.KEYWORD_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[agent_type] = score
        
        # Get top scoring type
        if max(scores.values()) > 0:
            top_type = max(scores, key=scores.get)
            confidence = min(scores[top_type] / 3, 1.0)  # Normalize
            
            return QueryIntent(
                agent_type=top_type,
                confidence=confidence,
                keywords=self._extract_keywords(query)
            )
        
        # Fallback to LLM classification
        if self.client:
            return self._llm_classify(query)
        
        # Default to literature
        return QueryIntent(
            agent_type="literature",
            confidence=0.5,
            keywords=self._extract_keywords(query)
        )
    
    def _llm_classify(self, query: str) -> QueryIntent:
        """Use LLM to classify complex queries."""
        try:
            response = self.client.chat.completions.create(
                model="minimax/minimax-m2.7",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a query classifier for a biotech research agent.
Classify the user query into one of these categories:
- literature: academic papers, research findings, reviews
- pipeline: drug pipeline, clinical trials, FDA approvals
- market: market size, industry analysis, competitive landscape
- dataset: data analysis, gene expression, single-cell analysis
- competitor: company analysis, partnerships, deals

Return JSON with fields:
- agent_type: the category
- confidence: 0.0-1.0
- target_disease: if mentioned (e.g., "NASH", "cancer")
- target_mechanism: if mentioned (e.g., "GLP-1", "FXR")
- target_company: if mentioned (e.g., "Novo Nordisk")
- target_region: if mentioned (e.g., "Korea", "US")

Only set fields that are explicitly mentioned in the query."""
                    },
                    {
                        "role": "user",
                        "content": f"Classify this query: {query}"
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return QueryIntent(
                agent_type=result.get("agent_type", "literature"),
                confidence=result.get("confidence", 0.5),
                keywords=self._extract_keywords(query),
                target_disease=result.get("target_disease"),
                target_mechanism=result.get("target_mechanism"),
                target_company=result.get("target_company"),
                target_region=result.get("target_region")
            )
        except Exception:
            return QueryIntent(
                agent_type="literature",
                confidence=0.3,
                keywords=self._extract_keywords(query)
            )
    
    def _extract_keywords(self, query: str) -> list:
        """Extract key terms from query."""
        # Simple extraction - remove common words and punctuation
        stopwords = ["what", "is", "are", "the", "a", "an", "in", "on", "at", "to", "for", 
                    "of", "and", "or", "with", "how", "can", "please", "analyze", "find"]
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]
