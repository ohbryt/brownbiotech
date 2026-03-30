"""
Configuration settings for Brown Biotech Research Agent
"""
import os
import json
from pathlib import Path


class Settings:
    """Application settings with persistence."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".brown-biotech-chatbot"
        self.config_file = self.config_dir / "config.json"
        
        # Default API keys (from environment or defaults)
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        self.TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY", "")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        
        # Model settings
        self.model = "minimax/minimax-m2.7"
        self.temperature = 0.7
        self.max_tokens = 4096
        
        # Data paths
        self.merfish_path = "/Users/ocm/.openclaw/workspace/skin_atlas_analysis/output/merfish.h5ad"
        self.arp_path = "/Users/ocm/.openclaw/workspace/arp-v3/"
        self.reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
        
        # Load saved settings
        self.load()
    
    def load(self):
        """Load settings from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                self.model = data.get("model", self.model)
                self.temperature = data.get("temperature", self.temperature)
                self.max_tokens = data.get("max_tokens", self.max_tokens)
            except Exception:
                pass
    
    def save(self):
        """Save settings to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump({
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }, f, indent=2)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "OPENROUTER_API_KEY": self.OPENROUTER_API_KEY[:10] + "..." if self.OPENROUTER_API_KEY else "",
            "TINYFISH_API_KEY": self.TINYFISH_API_KEY[:10] + "..." if self.TINYFISH_API_KEY else "",
            "GOOGLE_API_KEY": self.GOOGLE_API_KEY[:10] + "..." if self.GOOGLE_API_KEY else "",
            "model": self.model,
            "merfish_path": self.merfish_path,
            "arp_path": self.arp_path,
        }
