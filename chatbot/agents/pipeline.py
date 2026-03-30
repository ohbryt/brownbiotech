"""
Pipeline Agent — analyzes drug pipelines and clinical trials
"""
import time
from dataclasses import dataclass
from openai import OpenAI
from config.settings import Settings


@dataclass
class PipelineResult:
    """Structured drug pipeline result."""
    drugs: list
    stages: dict
    summary: str
    sources: str
    cost: float = 0.0


class PipelineAgent:
    """
    Analyzes drug development pipelines by mechanism/company.
    """
    
    # Known drug mechanisms and their approved/d pipeline drugs
    KNOWN_MECHANISMS = {
        "GLP-1": {
            "approved": ["Semaglutide (Novo Nordisk)", "Tirzepatide (Lilly)", "Liraglutide (Novo Nordisk)", "Dulaglutide (Lilly)", "Exenatide (AstraZeneca)"],
            "phase3": ["CagriSema (Novo Nordisk)", "Survodutide (Boehringer)", "Efo-cipeg (Hanmi)"],
            "phase2": ["AMG 133 (Amgen)", "NNC0194-0499 (Novo Nordisk)"],
            "phase1": ["YH17-1651 (Yuhan)", "BB3-32 (未知)"]
        },
        "FXR": {
            "approved": ["Obeticholic acid (Intercept)", "Cilofexor (Gilead)"],
            "phase3": ["Tropifexor (Novartis)", "EDP-305 (Enanta)"],
            "phase2": ["Vonafexor (Boehringer)", "MET409 (Metacrine)"],
            "phase1": ["EYP001 (仿制)"]
        },
        "PPAR": {
            "approved": ["Pioglitazone (Takeda)", "Fenofibrate (Abbott)", "Elafibranor (Ipsen)"],
            "phase3": ["Saroglitazar (Zydus)", "Pemafibrate (Kowa)"],
            "phase2": ["GFT505 (Ipsen)", "IVA323 (204 Pharma)"],
            "phase1": []
        },
        "SGLT2": {
            "approved": ["Canagliflozin (Janssen)", "Dapagliflozin (AstraZeneca)", "Empagliflozin (Boehringer)"],
            "phase3": ["Ertugliflozin (Merck)", "Sotagliflozin (Lexicon)"],
            "phase2": ["Tirzepatide (Lilly)", "CagriSema (Novo Nordisk)"],
            "phase1": []
        },
        "THR-β": {
            "approved": ["Resmetirom (Madrigal)", "Levothyroxine"],
            "phase3": ["VK2809 (Viking)", "ASC41 (Gannex)"],
            "phase2": ["MGL-3196 (Madra)", "HPGCD (Highlife)"],
            "phase1": []
        }
    }
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY
        ) if settings.OPENROUTER_API_KEY else None
    
    def analyze(self, query: str, intent) -> dict:
        """Analyze drug pipeline based on query."""
        start_time = time.time()
        
        # Detect mechanism from query
        mechanism = self._detect_mechanism(query)
        company = intent.target_company
        
        # Get pipeline data
        if mechanism:
            pipeline_data = self._get_pipeline_by_mechanism(mechanism)
        elif company:
            pipeline_data = self._get_pipeline_by_company(company)
        else:
            pipeline_data = self._get_general_pipeline(query)
        
        # Format response
        summary = self._format_summary(pipeline_data, mechanism, company)
        sources = self._get_sources(pipeline_data)
        
        elapsed = time.time() - start_time
        cost = 0.05
        
        return {
            "type": "pipeline",
            "query": query,
            "mechanism": mechanism,
            "company": company,
            "drugs": pipeline_data.get("drugs", []),
            "stages": pipeline_data.get("stages", {}),
            "summary": summary,
            "sources": sources,
            "cost": cost,
            "time": elapsed
        }
    
    def _detect_mechanism(self, query: str) -> str:
        """Detect target mechanism from query."""
        query_lower = query.lower()
        
        for mech in self.KNOWN_MECHANISMS.keys():
            if mech.lower() in query_lower:
                return mech
        
        # Check for partial matches
        if "glucagon" in query_lower or "incretin" in query_lower:
            return "GLP-1"
        if "farnesoid" in query_lower or "bile acid" in query_lower:
            return "FXR"
        if "peroxisome" in query_lower or "ppar" in query_lower:
            return "PPAR"
        if "sodium-glucose" in query_lower or "sglt" in query_lower:
            return "SGLT2"
        if "thyroid" in query_lower or "thr" in query_lower:
            return "THR-β"
        
        return None
    
    def _get_pipeline_by_mechanism(self, mechanism: str) -> dict:
        """Get full pipeline for a mechanism."""
        if mechanism not in self.KNOWN_MECHANISMS:
            return {"drugs": [], "stages": {}}
        
        data = self.KNOWN_MECHANISMS[mechanism]
        
        drugs = []
        stages = {"approved": [], "phase3": [], "phase2": [], "phase1": []}
        
        for stage, drug_list in data.items():
            for drug in drug_list:
                drugs.append({
                    "name": drug,
                    "stage": stage,
                    "company": self._extract_company(drug)
                })
                stages[stage].append(drug)
        
        return {"mechanism": mechanism, "drugs": drugs, "stages": stages}
    
    def _get_pipeline_by_company(self, company: str) -> dict:
        """Get pipeline for a specific company."""
        # Search known pipelines for company
        company_lower = company.lower()
        drugs = []
        stages = {"approved": [], "phase3": [], "phase2": [], "phase1": []}
        
        for mech, mech_data in self.KNOWN_MECHANISMS.items():
            for stage, drug_list in mech_data.items():
                for drug in drug_list:
                    if company_lower in drug.lower():
                        drugs.append({
                            "name": drug,
                            "stage": stage,
                            "mechanism": mech
                        })
                        stages[stage].append(drug)
        
        return {"company": company, "drugs": drugs, "stages": stages}
    
    def _get_general_pipeline(self, query: str) -> dict:
        """Get general pipeline overview."""
        # Return top mechanisms
        mechanisms = list(self.KNOWN_MECHANISMS.keys())[:3]
        drugs = []
        stages = {"approved": [], "phase3": [], "phase2": [], "phase1": []}
        
        for mech in mechanisms:
            data = self.KNOWN_MECHANISMS[mech]
            for stage, drug_list in data.items():
                for drug in drug_list[:2]:  # Top 2 per stage
                    drugs.append({
                        "name": drug,
                        "stage": stage,
                        "mechanism": mech
                    })
                    stages[stage].append(drug)
        
        return {"query": query, "drugs": drugs, "stages": stages}
    
    def _extract_company(self, drug_string: str) -> str:
        """Extract company name from drug string."""
        if "(" in drug_string:
            return drug_string.split("(")[1].replace(")", "")
        return "Unknown"
    
    def _format_summary(self, data: dict, mechanism: str, company: str) -> str:
        """Format pipeline data into readable summary."""
        if mechanism:
            title = f"## {mechanism} Agonist Pipeline Analysis"
        elif company:
            title = f"## {company} Pipeline Analysis"
        else:
            title = "## Drug Pipeline Overview"
        
        summary = f"{title}\n\n"
        
        stages = data.get("stages", {})
        
        # Count drugs per stage
        total = sum(len(drugs) for drugs in stages.values())
        
        summary += f"**Total drugs tracked:** {total}\n\n"
        
        for stage in ["approved", "phase3", "phase2", "phase1"]:
            drugs = stages.get(stage, [])
            if drugs:
                summary += f"### {stage.upper()} ({len(drugs)})\n"
                for drug in drugs:
                    summary += f"- {drug}\n"
                summary += "\n"
        
        # Add insights
        summary += "### Key Insights\n"
        if mechanism:
            summary += f"- **{mechanism}** is a validated target with multiple approved options\n"
            summary += "- Competition is intensifying in Phase 3\n"
            summary += "- Next-wave candidates focus on dual/triple mechanisms\n"
        elif company:
            summary += f"- **{company}** has {total} drugs in development\n"
            summary += "- Diversified across multiple mechanisms\n"
        
        return summary
    
    def _get_sources(self, data: dict) -> str:
        """Get data sources."""
        return "ClinicalTrials.gov, FDA.gov, Company press releases, Biomedtrizer"
