"""
Citation formatter for research results
"""
import re
from typing import List, Dict, Optional


class CitationFormatter:
    """
    Formats citations in various styles (APA, Vancouver, etc.)
    """
    
    @staticmethod
    def format_paper(citation: Dict, style: str = "apa") -> str:
        """Format a single paper citation."""
        authors = citation.get("authors", "Unknown")
        year = citation.get("year", "n.d.")
        title = citation.get("title", "Untitled")
        journal = citation.get("journal", "")
        pmid = citation.get("pmid", "")
        doi = citation.get("doi", "")
        
        if style == "apa":
            result = f"{authors} ({year}). {title}."
            if journal:
                result += f" *{journal}*"
            if pmid:
                result += f" PMID: {pmid}"
            return result
        
        elif style == "vancouver":
            result = f"{authors}. {title}. {journal}. {year}"
            if pmid:
                result += f". PMID: {pmid}"
            return result
        
        elif style == "bibtex":
            key = re.sub(r'[^a-zA-Z0-9]', '', f"{authors.split()[0]}{year}")
            result = f"@article{{{key},\n"
            result += f"  author = {{{authors}}},\n"
            result += f"  title = {{{title}}},\n"
            if journal:
                result += f"  journal = {{{journal}}},\n"
            result += f"  year = {{{year}}},\n"
            if pmid:
                result += f"  pmid = {{{pmid}}},\n"
            if doi:
                result += f"  doi = {{{doi}}},\n"
            result += "}"
            return result
        
        return f"{authors} ({year}). {title}. {journal}."
    
    @staticmethod
    def format_reference_list(citations: List[Dict], style: str = "numbered") -> str:
        """Format a list of citations as references."""
        if not citations:
            return "No references available."
        
        if style == "numbered":
            lines = []
            for i, cite in enumerate(citations, 1):
                formatted = CitationFormatter.format_paper(cite, style="apa")
                lines.append(f"[{i}] {formatted}")
            return "\n".join(lines)
        
        elif style == "bulleted":
            lines = []
            for cite in citations:
                formatted = CitationFormatter.format_paper(cite, style="apa")
                lines.append(f"• {formatted}")
            return "\n".join(lines)
        
        return "\n".join([CitationFormatter.format_paper(c) for c in citations])
    
    @staticmethod
    def extract_pmids(text: str) -> List[str]:
        """Extract PMID numbers from text."""
        pattern = r'PMID:\s*(\d+)'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_dois(text: str) -> List[str]:
        """Extract DOI numbers from text."""
        pattern = r'doi:\s*(10\.\d+/[^\s]+)'
        return re.findall(pattern, text, re.IGNORECASE)
    
    @staticmethod
    def create_pubmed_link(pmid: str) -> str:
        """Create a PubMed URL for a PMID."""
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    
    @staticmethod
    def create_clinical_trial_link(nct_id: str) -> str:
        """Create a ClinicalTrials.gov URL for an NCT ID."""
        return f"https://clinicaltrials.gov/ct2/show/{nct_id}/"
