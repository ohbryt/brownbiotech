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

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ─── Tool Registry ─────────────────────────────────────────────────────────────

LONG_READ_TOOLS = {
    # Long-Read Sequencing Tools (from long-read-tools.org)
    # Ritchie Lab, Walter and Eliza Hall Institute of Medical Research
    
    # Alignment
    "minimap2": {
        "name": "Minimap2",
        "category": "alignment",
        "description": "Fast pairwise aligner for mapping long reads",
        "platform": "C/Python",
        "technologies": ["Oxford Nanopore", "PacBio", "Bionano"],
        "citation": "Li, 2018",
    },
    "blasr": {
        "name": "BLASR",
        "category": "alignment",
        "description": "Basic Local Alignment with Successive Refinement",
        "platform": "C++",
        "technologies": ["PacBio"],
        "citation": "Chaisson & Tesler, 2012",
    },
    "ngmlr": {
        "name": "NGMLR",
        "category": "alignment",
        "description": "Sensitive aligner for structural variation detection",
        "platform": "C++",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Sedlazeck et al., 2018",
    },
    
    # Assembly
    "canu": {
        "name": "Canu",
        "category": "denovo_assembly",
        "description": "Long-read assembler with error correction",
        "platform": "C++/Perl",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Koren et al., 2017",
    },
    "flye": {
        "name": "Flye",
        "category": "denovo_assembly",
        "description": "Fast and accurate long-read assembler",
        "platform": "Python/C++",
        "technologies": ["Oxford Nanopore", "PacBio", "Bionano"],
        "citation": "Kolmogorov et al., 2019",
    },
    "necat": {
        "name": "NECAT",
        "category": "denovo_assembly",
        "description": "Nanopore-based genome assembler",
        "platform": "C++",
        "technologies": ["Oxford Nanopore"],
        "citation": "Chen et al., 2021",
    },
    "wtdbg2": {
        "name": "WTDBG2",
        "category": "denovo_assembly",
        "description": "Fuzzy Bruijn graph assembler for long reads",
        "platform": "C",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Ruan & Li, 2020",
    },
    
    # Error Correction & Polishing
    "racon": {
        "name": "Racon",
        "category": "error_correction",
        "description": "Ultrafast consensus module for long reads",
        "platform": "C++",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Vaser et al., 2017",
    },
    "medaka": {
        "name": "Medaka",
        "category": "error_correction",
        "description": "Oxford Nanopore-specific sequence polisher",
        "platform": "Python/C++",
        "technologies": ["Oxford Nanopore"],
        "citation": "Oxford Nanopore Technologies",
    },
    "arrow": {
        "name": "Arrow",
        "category": "error_correction",
        "description": "PacBio consensus polisher",
        "platform": "C++",
        "technologies": ["PacBio"],
        "citation": "Pacific Biosciences",
    },
    "pilon": {
        "name": "Pilon",
        "category": "error_correction",
        "description": "Hybrid assembly polisher (long + short reads)",
        "platform": "Java",
        "technologies": ["Oxford Nanopore", "PacBio", "Bionano"],
        "citation": "Walker et al., 2014",
    },
    
    # Basecalling
    "guppy": {
        "name": "Guppy",
        "category": "basecalling",
        "description": "Oxford Nanopore basecaller",
        "platform": "C++/CUDA",
        "technologies": ["Oxford Nanopore"],
        "citation": "Oxford Nanopore Technologies",
    },
    "bonito": {
        "name": "Bonito",
        "category": "basecalling",
        "description": "Oxford Nanopore neural network basecaller",
        "platform": "Python/PyTorch",
        "technologies": ["Oxford Nanopore"],
        "citation": "Oxford Nanopore Technologies",
    },
    "dorado": {
        "name": "Dorado",
        "category": "basecalling",
        "description": "Oxford Nanopore high-accuracy basecaller",
        "platform": "C++/CUDA",
        "technologies": ["Oxford Nanopore"],
        "citation": "Oxford Nanopore Technologies",
    },
    
    # Base Modification Detection
    "nanopolish": {
        "name": "Nanopolish",
        "category": "base_modification",
        "description": "Methylation detection from nanopore signals",
        "platform": "C++",
        "technologies": ["Oxford Nanopore"],
        "citation": "Simpson et al., 2017",
    },
    "megatron": {
        "name": "Megatron",
        "category": "base_modification",
        "description": "5mC detection at CpG resolution",
        "platform": "Python/PyTorch",
        "technologies": ["Oxford Nanopore"],
        "citation": "Liu et al., 2022",
    },
    
    # Isoform Detection & RNA
    "flair": {
        "name": "FLAIR",
        "category": "isoform_detection",
        "description": "Full-length alternative isoform analysis",
        "platform": "Python",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Tang et al., 2020",
    },
    "stringtie": {
        "name": "StringTie",
        "category": "isoform_detection",
        "description": "Transcript assembly and quantification",
        "platform": "C++",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Pertea et al., 2015",
    },
    "isoseq": {
        "name": "IsoSeq",
        "category": "isoform_detection",
        "description": "PacBio isoform sequencing analysis",
        "platform": "Python/C++",
        "technologies": ["PacBio"],
        "citation": "PacBio/SMRT Analysis",
    },
    
    # Variant Detection
    "deepvariant": {
        "name": "DeepVariant",
        "category": "variant_analysis",
        "description": "Deep learning variant caller",
        "platform": "Python/TensorFlow",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Poplin et al., 2018",
    },
    "clairvoyante": {
        "name": "Clairvoyante",
        "category": "variant_analysis",
        "description": "Multi-technology variant caller",
        "platform": "Python/PyTorch",
        "technologies": ["Oxford Nanopore", "PacBio", "Bionano"],
        "citation": "Luo et al., 2020",
    },
    "longshot": {
        "name": "LongShot",
        "category": "variant_analysis",
        "description": "SNV and indel caller for long reads",
        "platform": "Rust",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Edge & Coop, 2019",
    },
    
    # Structural Variation
    "sniffles": {
        "name": "Sniffles",
        "category": "structural_variation",
        "description": "Structural variation caller for long reads",
        "platform": "C++",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Sedlazeck et al., 2018",
    },
    "cutesv": {
        "name": "CuteSV",
        "category": "structural_variation",
        "description": "High-performance structural variant caller",
        "platform": "Python/C++",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Jiang et al., 2020",
    },
    "pbsv": {
        "name": "pbsv",
        "category": "structural_variation",
        "description": "PacBio structural variation caller",
        "platform": "C++",
        "technologies": ["PacBio"],
        "citation": "Pacific Biosciences",
    },
    
    # Metagenomics
    "kraken2": {
        "name": "Kraken2",
        "category": "metagenomics",
        "description": "Taxonomic classification using k-mers",
        "platform": "C++",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Wood et al., 2019",
    },
    "metamaps": {
        "name": "MetaMaps",
        "category": "metagenomics",
        "description": "Long-read metagenomics classifier",
        "platform": "C++",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Berlin et al., 2015",
    },
    
    # Quality Control & QC
    "nanostat": {
        "name": "NanoStat",
        "category": "quality_checking",
        "description": "Statistics from Oxford Nanopore runs",
        "platform": "Python",
        "technologies": ["Oxford Nanopore"],
        "citation": "Oxford Nanopore Technologies",
    },
    "poretools": {
        "name": "PoreTools",
        "category": "quality_checking",
        "description": "Toolkit for Oxford Nanopore reads",
        "platform": "Python",
        "technologies": ["Oxford Nanopore"],
        "citation": "Loman & Quick, 2015",
    },
    "longreadqc": {
        "name": "LongReadQC",
        "category": "quality_checking",
        "description": "Quality control for long reads",
        "platform": "Python/R",
        "technologies": ["Oxford Nanopore", "PacBio"],
        "citation": "Long Read QC",
    },
}

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
            "Microbiome-metabolome integration",
        ],
        "citation": "Franzosa et al., 2018",
    },
    "qiime2": {
        "name": "QIIME2",
        "type": "metagenomics",
        "description": "Quantitative Insights Into Microbial Ecology",
        "capabilities": [
            "Amplicon analysis",
            "Shotgun metagenomics",
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
    "long_read_tools": list(LONG_READ_TOOLS.keys()),
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
        "5. Long-Read: guppy/nanopolish → Isoform, methylation, SV detection",
    ]
}

# ─── Long-Read Cancer Applications ─────────────────────────────────────────────

LONG_READ_CANCER_APPLICATIONS = {
    "isoform_detection": {
        "description": "Detect cancer-specific splice variants",
        "tools": ["flair", "isoseq", "stringtie"],
        "cancer_benefit": "Novel therapeutic targets through isoform switching",
    },
    "methylation_analysis": {
        "description": "Epigenetic profiling from long-read sequencing",
        "tools": ["nanopolish", "megatron"],
        "cancer_benefit": "Epigenetic biomarkers, imprinting analysis",
    },
    "structural_variation": {
        "description": "Detect large genomic alterations in cancer",
        "tools": ["sniffles", "cutesv", "pbsv", "ngmlr"],
        "cancer_benefit": "Fusion genes, copy number variations, translocations",
    },
    "metagenomics": {
        "description": "Microbiome profiling with long reads",
        "tools": ["kraken2", "metamaps"],
        "cancer_benefit": "Microbiome-drug interactions, immunotherapy response",
    },
    "variant_calling": {
        "description": "Somatic mutation detection",
        "tools": ["deepvariant", "clairvoyante", "longshot"],
        "cancer_benefit": "Driver mutations, resistance mechanisms",
    },
}

def get_tools_by_type(tool_type: str) -> List[Dict]:
    """Get all tools of a specific type (multiomics only)."""
    return [
        {k: v} for k, v in MULTIOMICS_TOOLS.items()
        if v["type"] == tool_type
    ]

def get_long_read_tools_by_category(category: str) -> List[Dict]:
    """Get all long-read tools in a specific category."""
    return [
        {k: v} for k, v in LONG_READ_TOOLS.items()
        if v.get("category") == category
    ]

def get_long_read_application(application: str) -> Optional[Dict]:
    """Get long-read cancer application details."""
    return LONG_READ_CANCER_APPLICATIONS.get(application)

def get_tool_info(tool_name: str) -> Optional[Dict]:
    """Get detailed information about a tool (searches both multiomics and long-read)."""
    return MULTIOMICS_TOOLS.get(tool_name) or LONG_READ_TOOLS.get(tool_name)

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
    # Multi-omics tools
    "scanpy": "pip install scanpy anndata",
    "spatialcell": "pip install spatialcell",
    "stereopy": "pip install stereopy",
    "mofapy2": "pip install mofapy2",
    "mofaflex": "pip install mofaflex",
    "metaphlan": "pip install metaphlan",
    "humann": "pip install humann",
    "qiime2": "conda install qiime2 -c qiime2",
    # Long-read tools
    "minimap2": "conda install -c bioconda minimap2",
    "canu": "conda install -c bioconda canu",
    "flye": "conda install -c bioconda flye",
    "racon": "conda install -c bioconda racon",
    "medaka": "pip install medaka",
    "nanopolish": "conda install -c bioconda nanopolish",
    "flair": "pip install flair",
    "sniffles": "pip install sniffles",
    "kraken2": "conda install -c bioconda kraken2",
    "guppy": "Download from Oxford Nanopore (requires account)",
    "dorado": "Download from Oxford Nanopore (requires account)",
    "deepvariant": "pip install deepvariant",
    "clairvoyante": "pip install clairvoyante",
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
