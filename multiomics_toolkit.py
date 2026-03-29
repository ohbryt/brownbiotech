"""
BrownBioTech Multi-Omics Toolkit
================================
Comprehensive toolkit for single-cell, spatial, MOFA, and metagenomics analysis.

Tools Integrated:
- Scanpy: Single-cell RNA-seq
- SpatialCell: Spatial transcriptomics
- Stereopy: Stereo-seq analysis
- MOSAIK: Multi-origin spatial transcriptomics
- mofapy2: Multi-omics factor analysis
- metaphlan: Metagenomics profiling
- humann: Metabolic pathway analysis

Based on 2025 best practices for cancer metabolism research.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

# ─── Tool Registry ─────────────────────────────────────────────────────────────

MULTIOMICS_TOOLS = {
    # Single-Cell Tools
    "scanpy": {
        "name": "Scanpy",
        "type": "single_cell",
        "description": "Scalable toolkit for single-cell gene expression analysis",
        "capabilities": [
            "Preprocessing (QC, normalization, feature selection)",
            "Clustering and cell type annotation",
            "Trajectory inference",
            "Differential expression testing",
            "Visualization (UMAP, t-SNE)",
        ],
        "data_format": ".h5ad",
        "scale": ">1M cells",
        "citation": "Wolf et al., 2018",
    },
    "pisces": {
        "name": "PISCES",
        "type": "single_cell",
        "description": "Single-cell lineage tracing toolkit",
        "capabilities": [
            "Automated barcode QC",
            "Clone-size metrics",
            "Dynamic Sankey visualization",
        ],
        "citation": "Tsinghua University, 2025",
    },
    "allors": {
        "name": "Allos",
        "type": "single_cell",
        "description": "Isoform-level single-cell and spatial transcriptomics",
        "capabilities": [
            "Alternative splicing analysis",
            "scRNA-seq + spatial integration",
            "Long-read sequencing support",
        ],
        "citation": "BioRxiv, 2026",
    },
    
    # Spatial Transcriptomics
    "spatialcell": {
        "name": "SpatialCell",
        "type": "spatial",
        "description": "Integrated spatial transcriptomics pipeline",
        "capabilities": [
            "Cell segmentation",
            "Automated cell type annotation",
            "Histological image analysis",
            "Integration with Stardist, Bin2cell, TopAct",
        ],
        "data_format": "SpatialData format",
        "citation": "PyPI, 2025",
    },
    "stereopy": {
        "name": "Stereopy",
        "type": "spatial",
        "description": "Stereo-seq spatial transcriptomics analysis",
        "capabilities": [
            "Stereo-seq data analysis",
            "Spatial visualization",
            "Cell-type specific expression",
        ],
        "citation": "Stereo-seq consortium",
    },
    "mosaik": {
        "name": "MOSAIK",
        "type": "spatial",
        "description": "Multi-Origin Spatial Transcriptomics Analysis Integration Kit",
        "capabilities": [
            "Unified SpatialData format",
            "Multi-modal integration",
            "Quality control",
            "Downstream analysis",
        ],
        "citation": "ArXiv, 2025",
    },
    "smint": {
        "name": "SMINT",
        "type": "spatial",
        "description": "Spatial Multi-Omics Integration Toolkit",
        "capabilities": [
            "Transcriptomics integration",
            "Metabolomics integration",
            "Cell-level analysis",
        ],
        "citation": "GitHub, 2025",
    },
    "bella_vista": {
        "name": "Bella Vista",
        "type": "spatial",
        "description": "Visualization for imaging-based spatial transcriptomics",
        "capabilities": [
            "Single-cell resolution visualization",
            "Open-source Python package",
        ],
        "citation": "Biophysical Journal, 2024",
    },
    
    # Multi-Omics Integration (MOFA)
    "mofapy2": {
        "name": "MOFA2",
        "type": "multiomics",
        "description": "Multi-Omics Factor Analysis",
        "capabilities": [
            "Unsupervised factor analysis",
            "PCA generalization for multi-omics",
            "Latent factor identification",
            "Variation decomposition",
        ],
        "data_formats": ["RNA-seq", "Proteomics", " Methylation", "Metabolomics"],
        "citation": "BioFAM, 2026",
    },
    "mofaflex": {
        "name": "MOFA-FLEX",
        "type": "multiomics",
        "description": "Flexible factor analysis framework",
        "capabilities": [
            "Complex matrix factorization",
            "Modular architecture",
            "Probabilistic programming",
        ],
        "citation": "ReadTheDocs, 2025",
    },
    "omicverse": {
        "name": "OmicVerse",
        "type": "multiomics",
        "description": "Deep learning for multi-omics integration",
        "capabilities": [
            "MOFA implementation",
            "pyMOFA for training",
            "pyMOFAART for visualization",
        ],
        "citation": "DeepWiki, 2025",
    },
    
    # Metagenomics
    "metaphlan": {
        "name": "MetaPhlAn",
        "type": "metagenomics",
        "description": "Metagenomic profiling tool",
        "capabilities": [
            "Taxonomic profiling",
            "Microbiome composition",
            "Pathogen detection",
        ],
        "citation": "Segata et al., 2011",
    },
    "humann": {
        "name": "HUMAnN",
        "type": "metagenomics",
        "description": "HMP Unified Metabolic Analysis Network",
        "capabilities": [
            "Metabolic pathway quantification",
            "Gene family profiling",
            " microbiome-metabolome integration",
        ],
        "citation": "Franzosa et al., 2018",
    },
    "qiime2": {
        "name": "QIIME2",
        "type": "metagenomics",
        "description": "Quantitative Insights Into Microbial Ecology",
        "capabilities": [
            "Amplicon analysis",
            " shotgun metagenomics",
            "Phylogenetics",
        ],
        "citation": "Bolyen et al., 2019",
    },
}

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class MultiOmicsAnalysis:
    """Multi-omics analysis configuration."""
    analysis_type: str  # single_cell, spatial, mofa, metagenomics
    tool: str
    input_data: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Optional[Dict] = None

@dataclass
class ToolCapability:
    """Capability of a multi-omics tool."""
    name: str
    input_types: List[str]
    output_types: List[str]
    cancer_applications: List[str]

# ─── BrownBioTech Integration ─────────────────────────────────────────────────

BROWN_MULTIOMICS_PIPELINE = {
    "description": "Multi-omics analysis for cancer metabolism drug discovery",
    "tools": list(MULTIOMICS_TOOLS.keys()),
    "cancer_applications": [
        "Tumor microenvironment analysis",
        "Metastatic niche identification",
        "Metabolism reprogramming",
        "Drug response prediction",
        "Resistance mechanism discovery",
    ],
    "integration_workflow": [
        "1. Single-cell: Scanpy → Cell type identification",
        "2. Spatial: SpatialCell → Tumor architecture",
        "3. MOFA: mofapy2 → Multi-omics integration",
        "4. Metagenomics: MetaPhlAn/HUMAnN → Microbiome-drug interactions",
    ]
}

def get_tools_by_type(tool_type: str) -> List[Dict]:
    """Get all tools of a specific type."""
    return [
        {k: v} for k, v in MULTIOMICS_TOOLS.items()
        if v["type"] == tool_type
    ]

def get_tool_info(tool_name: str) -> Optional[Dict]:
    """Get detailed information about a tool."""
    return MULTIOMICS_TOOLS.get(tool_name)

def create_analysis_config(
    analysis_type: str,
    tool: str,
    cancer_focus: str = "NSCLC"
) -> MultiOmicsAnalysis:
    """Create a multi-omics analysis configuration."""
    
    configs = {
        "single_cell": {
            "parameters": {
                "min_genes": 200,
                "min_cells": 3,
                "n_neighbors": 15,
                "n_pcs": 50,
            }
        },
        "spatial": {
            "parameters": {
                "segmentation": "bin2cell",
                "annotation": "automated",
            }
        },
        "mofa": {
            "parameters": {
                "n_factors": 10,
                "iterations": 1000,
            }
        },
        "metagenomics": {
            "parameters": {
                "rarefaction_depth": 10000,
                "pathways": "metabolic",
            }
        }
    }
    
    return MultiOmicsAnalysis(
        analysis_type=analysis_type,
        tool=tool,
        input_data=f"{cancer_focus}_multiomics_data",
        parameters=configs.get(analysis_type, {}).get("parameters", {})
    )

# ─── Installation Guide ────────────────────────────────────────────────────────

INSTALL_COMMANDS = {
    "scanpy": "pip install scanpy anndata",
    "spatialcell": "pip install spatialcell",
    "stereopy": "pip install stereopy",
    "mofapy2": "pip install mofapy2",
    "mofaflex": "pip install mofaflex",
    "metaphlan": "pip install metaphlan",
    "humann": "pip install humann",
    "qiime2": "conda install qiime2 -c qiime2",
}

def get_install_command(tool: str) -> str:
    """Get installation command for a tool."""
    return INSTALL_COMMANDS.get(tool, f"pip install {tool}")

# ─── Example Usage ─────────────────────────────────────────────────────────────

EXAMPLE_WORKFLOWS = {
    "tumor_microenvironment": {
        "title": "Tumor Microenvironment Analysis",
        "steps": [
            ("Single-cell RNA-seq", "scanpy", "Identify cell types in tumor"),
            ("Spatial transcriptomics", "spatialcell", "Map cell type locations"),
            ("Multi-omics integration", "mofapy2", "Find metabolic signatures"),
            ("Metagenomics", "metaphlan", "Assess microbiome composition"),
        ],
        "cancer_types": ["NSCLC", "Breast", "Colorectal"],
    },
    "drug_response": {
        "title": "Drug Response Prediction",
        "steps": [
            ("Single-cell", "scanpy", "Profile sensitive vs resistant cells"),
            ("MOFA", "mofapy2", "Identify response biomarkers"),
            ("Spatial", "spatialcell", "Validate spatial patterns"),
        ],
        "output": "Biomarker panel for DGAT1/YARS2 response",
    },
}

if __name__ == "__main__":
    print("BrownBioTech Multi-Omics Toolkit")
    print("="*50)
    print(f"\nTools available: {len(MULTIOMICS_TOOLS)}")
    
    for tool_type in ["single_cell", "spatial", "multiomics", "metagenomics"]:
        tools = get_tools_by_type(tool_type)
        print(f"\n{tool_type.upper()}: {len(tools)} tools")
        for tool_dict in tools:
            for name, info in tool_dict.items():
                print(f"  - {info['name']}: {info['description'][:50]}...")
