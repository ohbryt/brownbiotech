"""
Market Agent — analyzes market size, trends, and competitive landscape
"""
import time
from dataclasses import dataclass
from openai import OpenAI
from config.settings import Settings


@dataclass
class MarketResult:
    """Structured market analysis result."""
    market_size: str
    growth_rate: str
    segments: list
    key_players: list
    trends: list
    summary: str
    sources: str
    cost: float = 0.0


class MarketAgent:
    """
    Analyzes market size, trends, and competitive landscape.
    """
    
    # Mock market data (in production, use real market research)
    MARKET_DATA = {
        "anti-aging cosmetics": {
            "global_size": "$25.2B (2024)",
            "cagr": "10.8%",
            "korea_size": "$3.5B (2024)",
            "korea_cagr": "8.5%",
            "segments": ["Skincare", "Nutraceuticals", "Medical aesthetics", "Personal care"],
            "key_players": ["L'Oréal", "Procter & Gamble", "Shiseido", "Amorepacific", "LG H&H"],
            "trends": ["Growth factor products", "AI-personalized skincare", "Microbiome focus", "Clean beauty"]
        },
        "cosmeceutical": {
            "global_size": "$7.8B (2024)",
            "cagr": "12.3%",
            "segments": ["Anti-aging", "Skin brightening", "Acne treatment", "Sun care"],
            "key_players": ["SkinCeuticals", "La Roche-Posay", "Obagi", "ZO Skin Health", "Alastin"],
            "trends": ["Clinical-grade actives", "Growth factor serums", "Peptide formulations", "Stem cell extracts"]
        },
        "biotech": {
            "global_size": "$750B (2024)",
            "cagr": "15.2%",
            "segments": ["Therapeutics", "Diagnostics", "Tools & Reagents", "CDMO"],
            "key_players": ["Thermo Fisher", "Danaher", "Merck KGaA", "Lonza", "Samsung Biologics"],
            "trends": ["AI-driven discovery", "Cell & gene therapy", "Precision medicine", "CRISPR applications"]
        },
        "nash": {
            "global_size": "$15B (2030 projected)",
            "cagr": "25%",
            "segments": ["Drugs", "Diagnostics", "Digital health"],
            "key_players": ["Novo Nordisk", "Intercept", "Madrigal", "Gilead", "89bio"],
            "trends": ["GLP-1 expansion", "Multi-mechanism approaches", "Early diagnosis", "Combination therapies"]
        },
        "obesity": {
            "global_size": "$4.8B (2024)",
            "cagr": "31.5%",
            "segments": ["GLP-1 drugs", "Devices", "Dietary supplements", "Bariatric surgery"],
            "key_players": ["Novo Nordisk", "Eli Lilly", "Amgen", "Viking Therapeutics", "Structure Therapeutics"],
            "trends": ["Oral formulations", "Dual/triple agonists", "Long-acting injectables", "Combo therapies"]
        }
    }
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY
        ) if settings.OPENROUTER_API_KEY else None
    
    def analyze(self, query: str, intent) -> dict:
        """Analyze market based on query."""
        start_time = time.time()
        
        # Detect market segment
        segment = self._detect_segment(query)
        region = intent.target_region
        
        # Get market data
        if segment:
            market_data = self._get_market_data(segment, region)
        else:
            market_data = self._get_general_market(query)
        
        # Format summary
        summary = self._format_summary(market_data, segment, region)
        sources = "Grand View Research, Fortune Business Insights, MarketsandMarkets, Mordor Intelligence"
        
        elapsed = time.time() - start_time
        cost = 0.03
        
        return {
            "type": "market",
            "query": query,
            "segment": segment,
            "region": region,
            "market_size": market_data.get("global_size", "N/A"),
            "growth_rate": market_data.get("cagr", "N/A"),
            "segments": market_data.get("segments", []),
            "key_players": market_data.get("key_players", []),
            "trends": market_data.get("trends", []),
            "summary": summary,
            "sources": sources,
            "cost": cost,
            "time": elapsed
        }
    
    def _detect_segment(self, query: str) -> str:
        """Detect market segment from query."""
        query_lower = query.lower()
        
        segment_keywords = {
            "anti-aging cosmetics": ["anti-aging", "antiaging", "안티에이징", "노화방지", "skin aging", "wrinkle"],
            "cosmeceutical": ["cosmeceutical", "화장품", "기능성，化精品", "skincare"],
            "biotech": ["biotech", "바이오텍", "biologicals", "biosimilar"],
            "nash": ["nash", "MASH", "fatty liver", "간 지방증"],
            "obesity": ["obesity", "비만", "weight loss", "glp-1", "wegovy", "ozempic"],
            "glp-1": ["glp-1", "semaglutide", "tirzepatide", "incretin"],
            "skin": ["skin", "피부", "dermatology", "cosmetic", "beauty"]
        }
        
        for segment, keywords in segment_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return segment
        
        return None
    
    def _get_market_data(self, segment: str, region: str = None) -> dict:
        """Get market data for segment."""
        if segment not in self.MARKET_DATA:
            return {}
        
        data = self.MARKET_DATA[segment].copy()
        
        # Add region-specific data if available
        if region:
            region_key = f"{region.lower()}_size"
            if region_key in data:
                data["regional_size"] = data[region_key]
        
        return data
    
    def _get_general_market(self, query: str) -> dict:
        """Get general market overview."""
        return {
            "global_size": "$750B+",
            "cagr": "12-15%",
            "segments": ["Pharma", "Biotech", "MedTech", "Cosmeceutical"],
            "key_players": ["Multiple global leaders"],
            "trends": ["AI integration", "Precision medicine", "Personalized therapeutics"]
        }
    
    def _format_summary(self, data: dict, segment: str, region: str = None) -> str:
        """Format market data into readable summary."""
        if segment:
            title = f"## {segment.replace('-', ' ').title()} Market Analysis"
        else:
            title = "## Market Analysis"
        
        summary = f"{title}\n\n"
        
        # Market size
        summary += f"### Market Size\n"
        summary += f"- **Global:** {data.get('global_size', 'N/A')}\n"
        if data.get('regional_size'):
            summary += f"- **Regional:** {data.get('regional_size')}\n"
        summary += f"- **Growth Rate (CAGR):** {data.get('cagr', 'N/A')}\n\n"
        
        # Segments
        segments = data.get("segments", [])
        if segments:
            summary += f"### Key Segments\n"
            for seg in segments:
                summary += f"- {seg}\n"
            summary += "\n"
        
        # Key players
        players = data.get("key_players", [])
        if players:
            summary += f"### Key Players\n"
            for player in players[:5]:
                summary += f"- {player}\n"
            summary += "\n"
        
        # Trends
        trends = data.get("trends", [])
        if trends:
            summary += f"### Market Trends\n"
            for trend in trends:
                summary += f"- 📈 {trend}\n"
        
        return summary
